//! Capability trait and registry.

mod clipboard;
mod desktop_notify;
mod file_tree;
mod filesystem;
mod git_info;
mod open_url;
mod processes;
mod screenshot;
mod search_files;
mod shell;
mod system_info;

use crate::config::AppConfig;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

/// All capability IDs and display names for the settings UI.
pub const CAPABILITIES_UI: &[(&str, &str)] = &[
    ("screenshot", "Screenshot"),
    ("clipboard_read", "Clipboard (read)"),
    ("clipboard_write", "Clipboard (write)"),
    ("system_info", "System info"),
    ("desktop_notify", "Desktop notifications"),
    ("shell_execute", "Shell execute"),
    ("read_file", "Read file"),
    ("list_directory", "List directory"),
    ("write_file", "Write file"),
    ("create_directory", "Create directory"),
    ("file_tree", "File tree"),
    ("search_files", "Search files"),
    ("git_info", "Git info"),
    ("list_processes", "List processes"),
    ("open_url", "Open URL"),
];

pub use clipboard::{ClipboardReadCapability, ClipboardWriteCapability};
pub use desktop_notify::DesktopNotifyCapability;
pub use file_tree::FileTreeCapability;
pub use filesystem::{
    CreateDirectoryCapability, ListDirectoryCapability, ReadFileCapability, WriteFileCapability,
};
pub use git_info::GitInfoCapability;
pub use open_url::OpenUrlCapability;
pub use processes::ListProcessesCapability;
pub use screenshot::ScreenshotCapability;
pub use search_files::SearchFilesCapability;
pub use shell::ShellExecuteCapability;
pub use system_info::SystemInfoCapability;

#[allow(dead_code)]
#[async_trait]
pub trait Capability: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;

    async fn execute(&self, args: Value, config: &AppConfig) -> Result<CapabilityResult, String>;
}

pub struct CapabilityResult {
    pub result: Value,
    pub formatted: String,
}

pub struct CapabilityRegistry {
    capabilities: HashMap<String, Box<dyn Capability>>,
}

impl CapabilityRegistry {
    pub fn new() -> Self {
        let mut capabilities = HashMap::new();
        capabilities.insert(
            "screenshot".to_string(),
            Box::new(ScreenshotCapability) as Box<dyn Capability>,
        );
        capabilities.insert(
            "clipboard_read".to_string(),
            Box::new(ClipboardReadCapability) as Box<dyn Capability>,
        );
        capabilities.insert(
            "clipboard_write".to_string(),
            Box::new(ClipboardWriteCapability) as Box<dyn Capability>,
        );
        capabilities.insert(
            "system_info".to_string(),
            Box::new(SystemInfoCapability) as Box<dyn Capability>,
        );
        capabilities.insert(
            "desktop_notify".to_string(),
            Box::new(DesktopNotifyCapability) as Box<dyn Capability>,
        );
        capabilities.insert(
            "shell_execute".to_string(),
            Box::new(ShellExecuteCapability) as Box<dyn Capability>,
        );
        capabilities.insert(
            "read_file".to_string(),
            Box::new(ReadFileCapability) as Box<dyn Capability>,
        );
        capabilities.insert(
            "list_directory".to_string(),
            Box::new(ListDirectoryCapability) as Box<dyn Capability>,
        );
        capabilities.insert(
            "write_file".to_string(),
            Box::new(WriteFileCapability) as Box<dyn Capability>,
        );
        capabilities.insert(
            "create_directory".to_string(),
            Box::new(CreateDirectoryCapability) as Box<dyn Capability>,
        );
        capabilities.insert(
            "file_tree".to_string(),
            Box::new(FileTreeCapability) as Box<dyn Capability>,
        );
        capabilities.insert(
            "search_files".to_string(),
            Box::new(SearchFilesCapability) as Box<dyn Capability>,
        );
        capabilities.insert(
            "git_info".to_string(),
            Box::new(GitInfoCapability) as Box<dyn Capability>,
        );
        capabilities.insert(
            "list_processes".to_string(),
            Box::new(ListProcessesCapability) as Box<dyn Capability>,
        );
        capabilities.insert(
            "open_url".to_string(),
            Box::new(OpenUrlCapability) as Box<dyn Capability>,
        );
        Self { capabilities }
    }

    pub fn get(&self, name: &str) -> Option<&dyn Capability> {
        self.capabilities.get(name).map(|b| b.as_ref())
    }

    pub fn names(&self) -> Vec<String> {
        self.capabilities.keys().cloned().collect()
    }
}

impl Default for CapabilityRegistry {
    fn default() -> Self {
        Self::new()
    }
}
