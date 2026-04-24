//! Open URL in default browser via the `open` crate (cross-platform).

use super::{Capability, CapabilityResult};
use crate::config::AppConfig;
use async_trait::async_trait;
use serde_json::json;
use serde_json::Value;

pub struct OpenUrlCapability;

#[async_trait]
impl Capability for OpenUrlCapability {
    fn name(&self) -> &str {
        "open_url"
    }

    fn description(&self) -> &str {
        "Open a URL in the default browser (args: url)"
    }

    async fn execute(&self, args: Value, _config: &AppConfig) -> Result<CapabilityResult, String> {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'url' argument")?;

        open::that(url).map_err(|e| {
            format!(
                "Cannot open URL in a browser (headless or no graphical session): {}",
                e
            )
        })?;

        let result = json!({
            "success": true,
            "url": url
        });

        Ok(CapabilityResult {
            formatted: format!("Opened URL in browser: {}", url),
            result,
        })
    }
}
