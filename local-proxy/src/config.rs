//! Configuration load/save and capability policy structs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppConfig {
    pub device_id: String,
    pub bastion_url: String,
    pub token: String,
    #[serde(default = "default_auto_connect")]
    pub auto_connect: bool,
    #[serde(default)]
    pub capabilities: HashMap<String, CapabilityConfig>,
}

fn default_auto_connect() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub allowed_paths: Vec<String>,
    #[serde(default)]
    pub denied_paths: Vec<String>,
    #[serde(default)]
    pub allowed_commands: Vec<String>,
    #[serde(default)]
    pub denied_patterns: Vec<String>,
    #[serde(default)]
    pub max_output_bytes: Option<usize>,
    #[serde(default)]
    pub max_file_bytes: Option<usize>,
}

fn default_true() -> bool {
    true
}

impl Default for CapabilityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            allowed_paths: Vec::new(),
            denied_paths: Vec::new(),
            allowed_commands: Vec::new(),
            denied_patterns: Vec::new(),
            max_output_bytes: None,
            max_file_bytes: None,
        }
    }
}

/// Platform-specific config directory.
pub fn config_dir() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("bastion-proxy"))
}

/// Path to config file (config_dir/config.yml).
pub fn config_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("config.yml"))
}

/// Load config from path or default platform path.
pub fn load_config(path_override: Option<&std::path::Path>) -> Result<AppConfig, ConfigError> {
    let path = path_override
        .map(PathBuf::from)
        .or_else(config_path)
        .ok_or(ConfigError::NoConfigDir)?;

    if !path.exists() {
        return Ok(AppConfig::default());
    }

    let s = std::fs::read_to_string(&path).map_err(ConfigError::Io)?;
    serde_yaml::from_str(&s).map_err(ConfigError::Yaml)
}

/// Save config to path or default platform path.
pub fn save_config(config: &AppConfig, path_override: Option<&std::path::Path>) -> Result<(), ConfigError> {
    let path = path_override
        .map(PathBuf::from)
        .or_else(config_path)
        .ok_or(ConfigError::NoConfigDir)?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(ConfigError::Io)?;
    }

    let s = serde_yaml::to_string(config).map_err(ConfigError::Yaml)?;
    std::fs::write(&path, s).map_err(ConfigError::Io)
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("no config directory")]
    NoConfigDir,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml::Error),
}
