use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

#[derive(Deserialize, Debug)]
pub struct Persona {
    pub name: String,
    pub model: String,
    pub system_prompt: String,

    #[serde(default)]
    pub context_paths: Vec<String>,
}

fn get_personas_dir() -> Result<PathBuf> {
    let config_dir =
        dirs::config_dir().ok_or_else(|| anyhow!("Could not find a valid config directory."))?;
    Ok(config_dir.join("aiterm").join("personas"))
}

pub fn load_persona(name: &str) -> Result<Persona> {
    let personas_dir = get_personas_dir()?;
    let persona_file = personas_dir.join(format!("{}.toml", name));

    if !persona_file.exists() {
        return Err(anyhow!(
            "Persona file not found: {:?}\nWill put a cool default later on",
            persona_file
        ));
    }

    let file_content = fs::read_to_string(&persona_file)
        .with_context(|| format!("Failed to read persona file: {:?}", persona_file))?;

    let persona: Persona = toml::from_str(&file_content)
        .with_context(|| format!("Failed to parse TOML: {:?}", persona_file))?;

    Ok(persona)
}

pub fn ensure_config_dir_exists() -> Result<()> {
    let personas_dir = get_personas_dir()?;
    fs::create_dir_all(&personas_dir)
        .with_context(|| format!("Failed to create config dir: {:?}", personas_dir))?;
    Ok(())
}
