use super::{LanguageModel, Message, ResponseStream};
use async_stream::try_stream;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;

// Request Structures
#[derive(Serialize)]
struct RequestBody {
    contents: Vec<RequestContent>,
}
#[derive(Serialize)]
struct RequestContent {
    role: String,
    parts: Vec<RequestPart>,
}
#[derive(Serialize)]
struct RequestPart {
    text: String,
}

// Response Structures
#[derive(Deserialize)]
struct ResponseBody {
    candidates: Vec<ResponseCandidate>,
}
#[derive(Deserialize)]
struct ResponseCandidate {
    content: ResponseContent,
}
#[derive(Deserialize)]
struct ResponseContent {
    parts: Vec<ResponsePart>,
}
#[derive(Deserialize)]
struct ResponsePart {
    text: String,
}

pub struct Gemini {
    api_key: String,
    client: reqwest::Client,
}

impl Gemini {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl LanguageModel for Gemini {
    async fn ask(
        &self,
        messages: &[Message],
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut stream = self.ask_stream(messages).await?;
        let mut full_response = String::new();
        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => return Err(e),
            };
            full_response.push_str(&chunk);
        }
        Ok(full_response)
    }

    async fn ask_stream(
        &self,
        messages: &[Message],
    ) -> Result<ResponseStream, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?key={}",
            &self.api_key
        );

        let request_contents: Vec<RequestContent> = messages
            .iter()
            .map(|msg| RequestContent {
                role: msg.role.clone(),
                parts: vec![RequestPart {
                    text: msg.content.clone(),
                }],
            })
            .collect();

        let request_body = RequestBody {
            contents: request_contents,
        };

        let res = self.client.post(&url).json(&request_body).send().await?;

        if !res.status().is_success() {
            let status = res.status();
            let error_text = res.text().await?;
            return Err(format!("API Error: {} - {}", status, error_text).into());
        }

        let mut byte_stream = res.bytes_stream();

        let stream = try_stream! {
            let mut buffer = String::new();
            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = chunk_result?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                loop {
                    if let Some(start_idx) = buffer.find('{') {
                        let mut open_braces = 0;
                        let mut end_idx_opt = None;

                        for (i, c) in buffer[start_idx..].char_indices() {
                            if c == '{' { open_braces += 1; }
                            if c == '}' { open_braces -= 1; }
                            if open_braces == 0 {
                                end_idx_opt = Some(start_idx + i + 1);
                                break;
                            }
                        }

                        if let Some(end_idx) = end_idx_opt {
                            let object_str = &buffer[start_idx..end_idx];
                            if let Ok(rb) = serde_json::from_str::<ResponseBody>(object_str) {
                                if let Some(text) = rb.candidates.first().and_then(|c| c.content.parts.first()).map(|p| p.text.clone()) {
                                    if !text.is_empty() { yield text; }
                                }
                            }
                            buffer.drain(..end_idx);
                        } else { break; }
                    } else { buffer.clear(); break; }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}
