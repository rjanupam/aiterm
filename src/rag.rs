// its all into todo
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use walkdir::WalkDir;

// some Structures
#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    content: Content,
}
#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}
#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    embedding: EmbeddingObject,
}
#[derive(Deserialize)]
struct EmbeddingObject {
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct BatchEmbeddingResponse {
    embeddings: Vec<EmbeddingObject>,
}

// Represents a piece of text from a file.
#[derive(Debug, Clone)]
struct TextChunk {
    source: String,
    text: String,
}

// main store
pub struct RagStore {
    api_key: String,
    client: reqwest::Client,
    chunks: Vec<TextChunk>,
    embeddings: Vec<Vec<f32>>,
}

impl RagStore {
    pub async fn new(api_key: String, paths: &[String]) -> Result<Self> {
        println!("Initializing...");
        let client = reqwest::Client::new();
        let chunks = Self::load_and_chunk_files(paths)?;

        if chunks.is_empty() {
            println!("Warning: No text files found in context paths.");
            return Ok(Self {
                api_key,
                client,
                chunks,
                embeddings: vec![],
            });
        }

        println!("Embedding {} text chunks via API...", chunks.len());
        let documents: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = embed_batch(&client, &api_key, documents).await?;
        println!("Embedding complete.");

        Ok(Self {
            api_key,
            client,
            chunks,
            embeddings,
        })
    }

    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<String>> {
        if self.chunks.is_empty() {
            return Ok(vec![]);
        }
        let query_embedding = embed_batch(&self.client, &self.api_key, vec![query.to_string()])
            .await?
            .remove(0);

        let mut scored_chunks: Vec<_> = self
            .embeddings
            .iter()
            .zip(&self.chunks)
            .map(|(embedding, chunk)| {
                let similarity = cos_sim(&query_embedding, embedding);
                (similarity, chunk)
            })
            .collect();

        scored_chunks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let context: Vec<String> = scored_chunks
            .iter()
            .take(top_k)
            .map(|(_, chunk)| format!("---\nSource: {}\n```\n{}\n```\n", chunk.source, chunk.text))
            .collect();

        Ok(context)
    }

    fn load_and_chunk_files(paths: &[String]) -> Result<Vec<TextChunk>> {
        const MAX_CHUNK_SIZE: usize = 2000;
        const CHUNK_OVERLAP: usize = 200;
        let mut chunks = Vec::new();
        for path_str in paths {
            let path = Path::new(path_str);
            if path.is_dir() {
                for entry in WalkDir::new(path)
                    .into_iter()
                    .filter_map(Result::ok)
                    .filter(|e| e.path().is_file() && is_text_file(e.path()))
                {
                    if let Ok(content) = std::fs::read_to_string(entry.path()) {
                        let source = entry.path().to_str().unwrap_or("").to_string();
                        chunks.extend(chunk_text(&source, &content, MAX_CHUNK_SIZE, CHUNK_OVERLAP));
                    }
                }
            } else if path.is_file() && is_text_file(path) {
                if let Ok(content) = std::fs::read_to_string(path) {
                    let source = path.to_str().unwrap_or("").to_string();
                    chunks.extend(chunk_text(&source, &content, MAX_CHUNK_SIZE, CHUNK_OVERLAP));
                }
            }
        }
        Ok(chunks)
    }
}

fn chunk_text(source: &str, text: &str, max_size: usize, overlap: usize) -> Vec<TextChunk> {
    if text.len() <= max_size {
        return vec![TextChunk {
            source: source.to_string(),
            text: text.to_string(),
        }];
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let end = std::cmp::min(start + max_size, text.len());
        chunks.push(TextChunk {
            source: source.to_string(),
            text: text[start..end].to_string(),
        });
        if end == text.len() {
            break;
        }
        start += max_size - overlap;
    }
    chunks
}

fn is_text_file(path: &Path) -> bool {
    const TEXT_EXTENSIONS: &[&str] = &[
        "rs", "toml", "md", "txt", "json", "yaml", "yml", "html", "css", "js", "ts", "py", "go",
        "c", "cpp", "h", "hpp", "php", "sh", "sql",
    ];
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext_str| TEXT_EXTENSIONS.contains(&ext_str.to_lowercase().as_str()))
        .unwrap_or(false)
}

async fn embed_batch(
    client: &reqwest::Client,
    api_key: &str,
    texts: Vec<String>,
) -> Result<Vec<Vec<f32>>> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={}",
        api_key
    );

    let requests: Vec<EmbeddingRequest> = texts
        .into_iter()
        .map(|text| EmbeddingRequest {
            model: "models/text-embedding-004".to_string(),
            content: Content {
                parts: vec![Part { text }],
            },
        })
        .collect();

    let res = client
        .post(&url)
        .json(&serde_json::json!({ "requests": requests }))
        .send()
        .await
        .context("Failed to send embedding request to API")?;

    if !res.status().is_success() {
        let error_text = res
            .text()
            .await
            .unwrap_or_else(|_| "Unknown API error".to_string());
        return Err(anyhow::anyhow!("API embedding failed: {}", error_text));
    }

    let response_body: BatchEmbeddingResponse = res
        .json()
        .await
        .context("Failed to parse embedding response")?;
    Ok(response_body
        .embeddings
        .into_iter()
        .map(|e| e.values)
        .collect())
}

// Calculates cosine similarity between two vectors.
fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot_product = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot_product / (norm_a * norm_b)
}
