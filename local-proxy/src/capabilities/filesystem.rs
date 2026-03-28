//! Filesystem capabilities: read_file, list_directory, write_file with path policy.

use super::{Capability, CapabilityResult};
use crate::config::AppConfig;
use crate::policy;
use async_trait::async_trait;
use serde_json::json;
use serde_json::Value;
use tokio::fs;
use tokio::io::AsyncWriteExt;

pub struct ReadFileCapability;
pub struct ListDirectoryCapability;
pub struct WriteFileCapability;

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

        policy::validate_path(config, "list_directory", path)?;

        let mut entries = Vec::new();
        let mut read_dir = fs::read_dir(path).await.map_err(|e| e.to_string())?;

        while let Some(entry) = read_dir.next_entry().await.map_err(|e| e.to_string())? {
            let meta = entry.metadata().await.map_err(|e| e.to_string())?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let is_dir = meta.is_dir();
            let size_bytes = if meta.is_file() { meta.len() as usize } else { 0 };
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs());

            entries.push(json!({
                "name": name,
                "is_dir": is_dir,
                "size_bytes": size_bytes,
                "modified": modified
            }));

            if recursive && is_dir {
                let sub_path = entry.path();
                let sub_path_str = sub_path.to_string_lossy();
                if let Ok(()) = policy::validate_path(config, "list_directory", &sub_path_str) {
                    if let Ok(mut sub) = fs::read_dir(&sub_path).await {
                        while let Ok(Some(e)) = sub.next_entry().await {
                            let m = e.metadata().await.ok();
                            let n = e.file_name().to_string_lossy().into_owned();
                            let is_d = m.as_ref().map(|x| x.is_dir()).unwrap_or(false);
                            let sz = m.as_ref().filter(|_| !is_d).map(|x| x.len() as usize).unwrap_or(0);
                            let mod_ = m.and_then(|x| x.modified().ok())
                                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                                .map(|d| d.as_secs());
                            entries.push(json!({
                                "name": format!("{}/{}", name, n),
                                "is_dir": is_d,
                                "size_bytes": sz,
                                "modified": mod_
                            }));
                        }
                    }
                }
            }
        }

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
