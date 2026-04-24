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

        let (success, err_detail) = show_notification(title, body);

        if !success {
            return Err(format!(
                "Desktop notifications unavailable (no D-Bus session, display, or notification daemon){}",
                err_detail
                    .map(|d| format!(": {}", d))
                    .unwrap_or_default()
            ));
        }

        let result = json!({
            "success": true
        });

        Ok(CapabilityResult {
            formatted: "Notification sent".to_string(),
            result,
        })
    }
}

#[cfg(not(windows))]
fn show_notification(title: &str, body: &str) -> (bool, Option<String>) {
    match notify_rust::Notification::new().summary(title).body(body).show() {
        Ok(_) => (true, None),
        Err(e) => (false, Some(e.to_string())),
    }
}

#[cfg(windows)]
fn show_notification(title: &str, body: &str) -> (bool, Option<String>) {
    use winrt_notification::Toast;
    let t = Toast::new(Toast::POWERSHELL_APP_ID).title(title).text1(body);
    match t.show() {
        Ok(()) => (true, None),
        Err(e) => (false, Some(e.to_string())),
    }
}
