use anyhow::{Result, anyhow};
use clap::{Args, Parser, Subcommand};
use std::env;
use std::io::{self, Write};
use tokio_stream::StreamExt;

mod config;
mod rag;
mod vendors;

use crate::config::Persona;
use crate::rag::RagStore;
use vendors::gemini::Gemini;
use vendors::{LanguageModel, Message};

// CLI
#[derive(Parser, Debug)]
#[command(author, version, about = "Playful...🥙🥙🥙🥙🥙🥙🥙🥙🥙🥙🥙🥙🥙🥙🥙🥙", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Ask(AskArgs),
    Converse(ConverseArgs),
}

#[derive(Args, Debug)]
struct AskArgs {
    #[arg(short, long)]
    persona: String,

    #[arg(required = true, num_args = 1..)]
    prompt: Vec<String>,

    // stream response
    #[arg(long)]
    stream: bool,

    // num of context chunks to retrieve for RAG
    #[arg(long, default_value = "3")]
    rag_chunks: usize,
}

#[derive(Args, Debug)]
struct ConverseArgs {
    // personas who participate in converse
    #[arg(short, long, required = true, num_args = 2..)]
    persona: Vec<String>,

    // initial prompt, by user
    #[arg(required = true, num_args = 1.., last = true)]
    prompt: Vec<String>,

    // num of turns the conversation should last. Each agent speaking once is a turn.
    #[arg(long, default_value = "4")]
    turns: usize,

    /// Num of context chunks to retrieve for RAG for each turn.
    #[arg(long, default_value = "2")]
    rag_chunks: usize,
}

// Agent-}
struct Agent {
    persona: Persona,
    model: Box<dyn LanguageModel>,
    rag_store: Option<RagStore>,
}

// main--------
#[tokio::main]
async fn main() -> Result<()> {
    config::ensure_config_dir_exists()?;
    let cli = Cli::parse();

    match cli.command {
        Commands::Ask(args) => run_ask(args).await,
        Commands::Converse(args) => run_converse(args).await,
    }
}

async fn run_ask(args: AskArgs) -> Result<()> {
    let persona = config::load_persona(&args.persona)?;
    println!(
        "Using persona: '{}' (Model: {})",
        persona.name, persona.model
    );

    let api_key = env::var("GEMINI_API_KEY")
        .map_err(|_| anyhow!("GEMINI_API_KEY environment variable not set."))?;

    let rag_store = if !persona.context_paths.is_empty() {
        Some(RagStore::new(api_key.clone(), &persona.context_paths).await?)
    } else {
        None
    };

    let model: Box<dyn LanguageModel> = match persona.model.as_str() {
        "gemini" => Box::new(Gemini::new(api_key)),
        _ => return Err(anyhow!("Unknown model '{}'", persona.model)),
    };

    let prompt_str = args.prompt.join(" ");
    println!("\nAsking: {}...", prompt_str);

    let context_str = if let Some(store) = &rag_store {
        println!("Searching for relevant context via API...");
        let context_chunks = store.search(&prompt_str, args.rag_chunks).await?;
        if !context_chunks.is_empty() {
            println!("Found {} relevant context snippets.", context_chunks.len());
            format!(
                "Here is some relevant context from the local files:\n\n{}\n",
                context_chunks.join("\n")
            )
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    let final_content = format!(
        "{}\n\n{}\n\nUser question: {}",
        persona.system_prompt, context_str, prompt_str
    );

    let messages = vec![Message {
        role: "user".to_string(),
        content: final_content,
    }];

    if args.stream {
        println!("\n--- Response Stream ---");
        let mut response_stream = model.ask_stream(&messages).await.map_err(|e| anyhow!(e))?;
        while let Some(chunk_result) = response_stream.next().await {
            let chunk = chunk_result.map_err(|e| anyhow!(e))?;
            print!("{}", chunk);
            io::stdout().flush()?;
        }
        println!();
    } else {
        let response = model.ask(&messages).await.map_err(|e| anyhow!(e))?;
        println!("\n--- Response ---\n{}", response);
    }

    Ok(())
}

async fn run_converse(args: ConverseArgs) -> Result<()> {
    println!("Starting a conversation with: {}", args.persona.join(", "));
    let api_key = env::var("GEMINI_API_KEY")
        .map_err(|_| anyhow!("GEMINI_API_KEY environment variable not set."))?;

    // load agents
    let mut agents = Vec::new();
    for p_name in &args.persona {
        let persona = config::load_persona(p_name)?;
        let model: Box<dyn LanguageModel> = match persona.model.as_str() {
            "gemini" => Box::new(Gemini::new(api_key.clone())),
            _ => {
                return Err(anyhow!(
                    "Unknown model '{}' in persona '{}'",
                    persona.model,
                    p_name
                ));
            }
        };
        let rag_store = if !persona.context_paths.is_empty() {
            Some(RagStore::new(api_key.clone(), &persona.context_paths).await?)
        } else {
            None
        };
        agents.push(Agent {
            persona,
            model,
            rag_store,
        });
    }

    // initialize converse
    let initial_prompt = args.prompt.join(" ");
    let mut conversation_history = format!(
        "The user started the conversation with this prompt: \"{}\"",
        initial_prompt
    );

    // go
    for i in 0..args.turns {
        let current_agent_index = i % agents.len();
        let agent = &agents[current_agent_index];

        println!(
            "\n--- Turn {}/{} | Speaking: {} ---",
            i + 1,
            args.turns,
            agent.persona.name
        );

        // RAG search for the current turn based on the latest history
        let context_str = if let Some(store) = &agent.rag_store {
            let context_chunks = store.search(&conversation_history, args.rag_chunks).await?;
            if !context_chunks.is_empty() {
                format!("CONTEXT:\n{}\n", context_chunks.join("\n"))
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // abother prompt for this turn
        let turn_prompt = format!(
            "YOUR ROLE:\n{system_prompt}\n\n{context}\n\nCONVERSATION HISTORY:\n---\n{history}\n---\n\nINSTRUCTIONS: Your name is {name}. Based on your role and the history, provide your response. Do NOT include your name or role in the response itself. Just give your conversational reply.",
            system_prompt = agent.persona.system_prompt,
            context = context_str,
            history = conversation_history,
            name = agent.persona.name
        );

        let messages = vec![Message {
            role: "user".to_string(),
            content: turn_prompt,
        }];

        // agent's response
        let mut response_stream = agent
            .model
            .ask_stream(&messages)
            .await
            .map_err(|e| anyhow!(e))?;
        let mut full_response = String::new();
        while let Some(chunk_result) = response_stream.next().await {
            let chunk = chunk_result.map_err(|e| anyhow!(e))?;
            print!("{}", chunk);
            io::stdout().flush()?;
            full_response.push_str(&chunk);
        }

        // update history
        conversation_history.push_str(&format!(
            "\n\n{}: {}",
            agent.persona.name,
            full_response.trim()
        ));
    }

    println!("\n\n--- Conversation Finished ---");
    Ok(())
}
