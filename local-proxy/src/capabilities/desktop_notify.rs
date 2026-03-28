//! Desktop notification capability. Platform-specific: notify-rust (Linux/macOS), winrt-notification (Windows).

use super::{Capability, CapabilityResult};
use crate::config::AppConfig;
use async_trait::async_trait;
use serde_json::json;
use serde_json::Value;

pub struct DesktopNotifyCapability;

#[async_trait]
impl Capability for DesktopNotifyCapability {
    fn name(&self) -> &str {
        "desktop_notify"
    }

    fn description(&self) -> &str {
        "Show a desktop notification (args: title, body, timeout_ms optional)"
    }

    async fn execute(&self, args: Value, _config: &AppConfig) -> Result<CapabilityResult, String> {
        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Bastion");
        let body = args.get("body").and_then(|v| v.as_str()).unwrap_or("");

        let success = show_notification(title, body);

        let result = json!({
            "success": success
        });

        Ok(CapabilityResult {
            formatted: if success {
                "Notification sent".to_string()
            } else {
                "Failed to show notification".to_string()
            },
            result,
        })
    }
}

#[cfg(not(windows))]
fn show_notification(title: &str, body: &str) -> bool {
    notify_rust::Notification::new()
        .summary(title)
        .body(body)
        .show()
        .is_ok()
}

#[cfg(windows)]
fn show_notification(title: &str, body: &str) -> bool {
    use winrt_notification::Toast;
    Toast::new(Toast::POWERSHELL_APP_ID)
        .title(title)
        .text1(body)
        .show()
        .is_ok()
}
