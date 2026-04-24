//! Clipboard capabilities: read and write text via arboard (cross-platform).

use super::{Capability, CapabilityResult};
use crate::config::AppConfig;
use async_trait::async_trait;
use serde_json::json;
use serde_json::Value;

pub struct ClipboardReadCapability;
pub struct ClipboardWriteCapability;

#[async_trait]
impl Capability for ClipboardReadCapability {
    fn name(&self) -> &str {
        "clipboard_read"
    }

    fn description(&self) -> &str {
        "Read text from the system clipboard"
    }

    async fn execute(&self, _args: Value, _config: &AppConfig) -> Result<CapabilityResult, String> {
        let mut clipboard = arboard::Clipboard::new().map_err(|e| {
            format!("Clipboard read unavailable (no display or session clipboard): {}", e)
        })?;
        let content = clipboard.get_text().map_err(|e| {
            format!("Clipboard read failed (headless or no clipboard server): {}", e)
        })?;
        let length = content.len();

        let result = json!({
            "content": content,
            "length": length
        });

        Ok(CapabilityResult {
            formatted: format!("Clipboard read ({} bytes)", length),
            result,
        })
    }
}

#[async_trait]
impl Capability for ClipboardWriteCapability {
    fn name(&self) -> &str {
        "clipboard_write"
    }

    fn description(&self) -> &str {
        "Write text to the system clipboard"
    }

    async fn execute(&self, args: Value, _config: &AppConfig) -> Result<CapabilityResult, String> {
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let mut clipboard = arboard::Clipboard::new().map_err(|e| {
            format!("Clipboard write unavailable (no display or session clipboard): {}", e)
        })?;
        clipboard.set_text(content).map_err(|e| {
            format!("Clipboard write failed (headless or no clipboard server): {}", e)
        })?;

        let result = json!({
            "success": true
        });

        Ok(CapabilityResult {
            formatted: format!("Clipboard written ({} bytes)", content.len()),
            result,
        })
    }
}
