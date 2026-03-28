//! Shell execute capability: run shell commands with policy validation and output limits.

use super::{Capability, CapabilityResult};
use crate::config::AppConfig;
use crate::policy;
use async_trait::async_trait;
use serde_json::json;
use serde_json::Value;
use std::time::Duration;
use tokio::process::Command;

pub struct ShellExecuteCapability;

#[async_trait]
impl Capability for ShellExecuteCapability {
    fn name(&self) -> &str {
        "shell_execute"
    }

    fn description(&self) -> &str {
        "Run a shell command (args: command, timeout_seconds?, cwd?). Policy: allowed_commands, denied_patterns, max_output_bytes."
    }

    async fn execute(&self, args: Value, config: &AppConfig) -> Result<CapabilityResult, String> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'command' argument")?
            .to_string();
        let timeout_secs = args
            .get("timeout_seconds")
            .and_then(|v| v.as_u64())
            .unwrap_or(60) as u64;
        let cwd = args.get("cwd").and_then(|v| v.as_str()).map(std::path::Path::new);

        policy::validate_command(config, "shell_execute", &command)?;

        let (shell, shell_arg) = if cfg!(windows) {
            ("cmd", "/C")
        } else {
            ("sh", "-c")
        };

        let mut cmd = Command::new(shell);
        cmd.arg(shell_arg).arg(&command);
        if let Some(cwd_path) = cwd {
            cmd.current_dir(cwd_path);
        }
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let max_bytes = config
            .capabilities
            .get("shell_execute")
            .and_then(|c| c.max_output_bytes)
            .unwrap_or(1024 * 1024);

        let output = tokio::time::timeout(
            Duration::from_secs(timeout_secs),
            cmd.output(),
        )
        .await
        .map_err(|_| "Command timed out")?
        .map_err(|e| e.to_string())?;

        let mut stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let mut stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let mut truncated = false;
        if stdout.len() > max_bytes {
            stdout = format!("{}... (truncated)", &stdout[..max_bytes]);
            truncated = true;
        }
        if stderr.len() > max_bytes {
            stderr = format!("{}... (truncated)", &stderr[..max_bytes]);
            truncated = true;
        }

        policy::check_output_limit(config, "shell_execute", stdout.len() + stderr.len())?;

        let result = json!({
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": output.status.code().unwrap_or(-1),
            "truncated": truncated,
            "timed_out": false
        });

        Ok(CapabilityResult {
            formatted: format!(
                "Exit code {} | stdout {} bytes | stderr {} bytes",
                output.status.code().unwrap_or(-1),
                stdout.len(),
                stderr.len()
            ),
            result,
        })
    }
}
