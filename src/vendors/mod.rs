use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use tokio_stream::Stream;

pub mod gemini;

pub type StreamChunk = Result<String, Box<dyn std::error::Error + Send + Sync>>;
pub type ResponseStream = Pin<Box<dyn Stream<Item = StreamChunk> + Send>>;

#[derive(Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[async_trait]
pub trait LanguageModel: Send + Sync {
    async fn ask(
        &self,
        messages: &[Message],
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;

    async fn ask_stream(
        &self,
        messages: &[Message],
    ) -> Result<ResponseStream, Box<dyn std::error::Error + Send + Sync>>;
}
