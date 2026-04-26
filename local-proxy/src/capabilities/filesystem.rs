//! Filesystem capabilities: read_file, list_directory, write_file with path policy.

use super::{Capability, CapabilityResult};
use crate::config::AppConfig;
use crate::policy;
use async_trait::async_trait;
use serde_json::json;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use tokio::fs;
use tokio::io::AsyncWriteExt;

pub struct ReadFileCapability;
pub struct ListDirectoryCapability;
pub struct WriteFileCapability;
pub struct PatchFileCapability;
pub struct CreateDirectoryCapability;

fn mkdir_policy_capability(config: &AppConfig) -> &'static str {
    if config.capabilities.contains_key("create_directory") {
        "create_directory"
    } else {
        "write_file"
    }
}

fn list_dir_recursive<'a>(
    config: &'a AppConfig,
    base_path: &'a std::path::Path,
    rel_prefix: String,
    depth: u32,
    max_depth: u32,
    entries: &'a mut Vec<Value>,
) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + 'a>> {
    Box::pin(async move {
        if depth > max_depth {
            return Ok(());
        }
        let mut rd = fs::read_dir(base_path).await.map_err(|e| e.to_string())?;
        while let Some(entry) = rd.next_entry().await.map_err(|e| e.to_string())? {
            let meta = entry.metadata().await.map_err(|e| e.to_string())?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let is_dir = meta.is_dir();
            let size_bytes = if meta.is_file() { meta.len() as usize } else { 0 };
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs());

            let full_name = if rel_prefix.is_empty() {
                name.clone()
            } else {
                format!("{}/{}", rel_prefix, name)
            };

            entries.push(json!({
                "name": full_name.clone(),
                "is_dir": is_dir,
                "size_bytes": size_bytes,
                "modified": modified
            }));

            if is_dir && depth < max_depth {
                let sub_path = entry.path();
                let sub_path_str = sub_path.to_string_lossy();
                if policy::validate_path(config, "list_directory", &sub_path_str).is_ok() {
                    list_dir_recursive(
                        config,
                        &sub_path,
                        full_name,
                        depth + 1,
                        max_depth,
                        entries,
                    )
                    .await?;
                }
            }
        }
        Ok(())
    })
}

#[async_trait]
impl Capability for ReadFileCapability {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a file's contents (args: path). Policy: allowed_paths, denied_paths, max_file_bytes."
    }

    async fn execute(&self, args: Value, config: &AppConfig) -> Result<CapabilityResult, String> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;

        policy::validate_path(config, "read_file", path)?;

        let max_bytes = config
            .capabilities
            .get("read_file")
            .and_then(|c| c.max_file_bytes)
            .unwrap_or(10 * 1024 * 1024);

        let content = fs::read_to_string(path).await.map_err(|e| e.to_string())?;
        let size_bytes = content.len();
        if size_bytes > max_bytes {
            return Err(format!(
                "File size {} exceeds max_file_bytes {}",
                size_bytes, max_bytes
            ));
        }

        let result = json!({
            "content": content,
            "size_bytes": size_bytes,
            "path": path
        });

        Ok(CapabilityResult {
            formatted: format!("Read {} bytes from {}", size_bytes, path),
            result,
        })
    }
}

#[async_trait]
impl Capability for ListDirectoryCapability {
    fn name(&self) -> &str {
        "list_directory"
    }

    fn description(&self) -> &str {
        "List directory contents (args: path, recursive?). Policy: allowed_paths, denied_paths."
    }

    async fn execute(&self, args: Value, config: &AppConfig) -> Result<CapabilityResult, String> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;
        let recursive = args.get("recursive").and_then(|v| v.as_bool()).unwrap_or(false);
        let max_depth = args
            .get("max_depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(if recursive { 3 } else { 1 }) as u32;

        policy::validate_path(config, "list_directory", path)?;

        let mut entries = Vec::new();
        let base = std::path::Path::new(path);
        list_dir_recursive(config, base, String::new(), 1, max_depth, &mut entries).await?;

        let result = json!({
            "entries": entries,
            "count": entries.len()
        });

        Ok(CapabilityResult {
            formatted: format!("Listed {} entries in {}", entries.len(), path),
            result,
        })
    }
}

#[async_trait]
impl Capability for WriteFileCapability {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file (args: path, content, append?). Policy: allowed_paths, denied_paths."
    }

    async fn execute(&self, args: Value, config: &AppConfig) -> Result<CapabilityResult, String> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let append = args.get("append").and_then(|v| v.as_bool()).unwrap_or(false);

        policy::validate_path_for_write(config, "write_file", path)?;

        let bytes_written = if append {
            let mut f = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .await
                .map_err(|e| e.to_string())?;
            let n = f.write(content.as_bytes()).await.map_err(|e| e.to_string())?;
            f.flush().await.map_err(|e| e.to_string())?;
            n as u64
        } else {
            fs::write(path, content).await.map_err(|e| e.to_string())?;
            content.len() as u64
        };

        let result = json!({
            "success": true,
            "path": path,
            "bytes_written": bytes_written
        });

        Ok(CapabilityResult {
            formatted: format!("Wrote {} bytes to {}", bytes_written, path),
            result,
        })
    }
}

#[async_trait]
impl Capability for PatchFileCapability {
    fn name(&self) -> &str {
        "patch_file"
    }

    fn description(&self) -> &str {
        "Replace a unique substring in a file (args: path, old_string, new_string, replace_all optional). Uses patch_file / write_file / read_file policy."
    }

    async fn execute(&self, args: Value, config: &AppConfig) -> Result<CapabilityResult, String> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;
        let old_string = args
            .get("old_string")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'old_string' argument")?;
        let new_string = args
            .get("new_string")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let replace_all = args.get("replace_all").and_then(|v| v.as_bool()).unwrap_or(false);

        if old_string.is_empty() {
            return Err("old_string must be non-empty".to_string());
        }

        policy::validate_path_for_write(config, "patch_file", path)?;

        let max_bytes = config
            .capabilities
            .get("read_file")
            .and_then(|c| c.max_file_bytes)
            .or_else(|| config.capabilities.get("patch_file").and_then(|c| c.max_file_bytes))
            .or_else(|| config.capabilities.get("write_file").and_then(|c| c.max_file_bytes))
            .unwrap_or(10 * 1024 * 1024);

        let content = fs::read_to_string(path).await.map_err(|e| e.to_string())?;
        if content.len() > max_bytes {
            return Err(format!(
                "File size {} exceeds max_file_bytes {}",
                content.len(),
                max_bytes
            ));
        }

        let count = content.matches(old_string).count();
        if count == 0 {
            return Err(format!(
                "old_string not found in file ({} bytes). Check exact whitespace and line endings.",
                content.len()
            ));
        }
        if count > 1 && !replace_all {
            return Err(format!(
                "old_string matched {} times; set replace_all=true to change all, or use a longer unique snippet",
                count
            ));
        }

        let updated = if replace_all {
            content.replace(old_string, new_string)
        } else {
            content.replacen(old_string, new_string, 1)
        };

        fs::write(path, &updated).await.map_err(|e| e.to_string())?;

        let result = json!({
            "success": true,
            "path": path,
            "replacements": if replace_all { count } else { 1 },
            "bytes_written": updated.len() as u64
        });

        Ok(CapabilityResult {
            formatted: format!(
                "Patched {} ({} replacement(s), {} bytes)",
                path,
                if replace_all { count } else { 1 },
                updated.len()
            ),
            result,
        })
    }
}

#[async_trait]
impl Capability for CreateDirectoryCapability {
    fn name(&self) -> &str {
        "create_directory"
    }

    fn description(&self) -> &str {
        "Create a directory and parents (args: path). Uses create_directory or write_file policy."
    }

    async fn execute(&self, args: Value, config: &AppConfig) -> Result<CapabilityResult, String> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;

        let cap = mkdir_policy_capability(config);
        policy::validate_path_for_write(config, cap, path)?;

        fs::create_dir_all(path).await.map_err(|e| e.to_string())?;

        let result = json!({
            "success": true,
            "path": path
        });

        Ok(CapabilityResult {
            formatted: format!("Created directory {}", path),
            result,
        })
    }
}
