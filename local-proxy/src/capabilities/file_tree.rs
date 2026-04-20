//! File tree capability: recursive directory tree listing with ignore patterns and depth limits.

use super::{Capability, CapabilityResult};
use crate::config::AppConfig;
use crate::policy;
use async_trait::async_trait;
use serde_json::json;
use serde_json::Value;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use tokio::fs;

pub struct FileTreeCapability;

fn should_ignore(name: &str, ignore_patterns: &[String]) -> bool {
    ignore_patterns.iter().any(|p| p == name)
}

fn is_hidden(name: &str) -> bool {
    name.starts_with('.')
}

fn build_tree<'a>(
    dir: &'a Path,
    root: &'a Path,
    depth: u32,
    max_depth: u32,
    include_hidden: bool,
    ignore_patterns: &'a [String],
    node_count: &'a mut usize,
    max_nodes: usize,
    truncated: &'a mut bool,
) -> Pin<Box<dyn Future<Output = Result<Vec<Value>, String>> + Send + 'a>> {
    Box::pin(async move {
        if depth > max_depth {
            return Ok(Vec::new());
        }

        if *node_count >= max_nodes {
            *truncated = true;
            return Ok(Vec::new());
        }

        let mut out: Vec<Value> = Vec::new();
        let mut rd = fs::read_dir(dir).await.map_err(|e| e.to_string())?;

        while let Some(entry) = rd.next_entry().await.map_err(|e| e.to_string())? {
            if *node_count >= max_nodes {
                *truncated = true;
                break;
            }

            let file_name = entry.file_name();
            let name = file_name.to_string_lossy().into_owned();

            if !include_hidden && is_hidden(&name) {
                continue;
            }
            if should_ignore(&name, ignore_patterns) {
                continue;
            }

            let meta = entry.metadata().await.map_err(|e| e.to_string())?;
            let is_dir = meta.is_dir();
            let size_bytes = if meta.is_file() { meta.len() as u64 } else { 0 };
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs());

            let abs_path: PathBuf = entry.path();
            let rel_path = abs_path
                .strip_prefix(root)
                .unwrap_or(&abs_path)
                .to_string_lossy()
                .replace('\\', "/");

            *node_count += 1;

            let children = if is_dir && depth < max_depth && !*truncated {
                build_tree(
                    &abs_path,
                    root,
                    depth + 1,
                    max_depth,
                    include_hidden,
                    ignore_patterns,
                    node_count,
                    max_nodes,
                    truncated,
                )
                .await?
            } else {
                Vec::new()
            };

            out.push(json!({
                "path": rel_path,
                "name": name,
                "is_dir": is_dir,
                "size_bytes": size_bytes,
                "modified": modified,
                "children": children
            }));
        }

        // Sort: dirs first, then by name for stability
        out.sort_by(|a, b| {
            let ad = a.get("is_dir").and_then(|v| v.as_bool()).unwrap_or(false);
            let bd = b.get("is_dir").and_then(|v| v.as_bool()).unwrap_or(false);
            if ad != bd {
                return bd.cmp(&ad);
            }
            let an = a.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let bn = b.get("name").and_then(|v| v.as_str()).unwrap_or("");
            an.cmp(bn)
        });

        Ok(out)
    })
}

fn format_tree(nodes: &[Value], indent: usize, out: &mut String, max_chars: usize, truncated: bool) {
    if out.len() >= max_chars {
        return;
    }
    for n in nodes {
        if out.len() >= max_chars {
            break;
        }
        let name = n.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let is_dir = n.get("is_dir").and_then(|v| v.as_bool()).unwrap_or(false);
        let prefix = "  ".repeat(indent);
        let line = if is_dir {
            format!("{}{}/\n", prefix, name)
        } else {
            format!("{}{}\n", prefix, name)
        };
        if out.len() + line.len() > max_chars {
            break;
        }
        out.push_str(&line);

        if is_dir {
            if let Some(children) = n.get("children").and_then(|v| v.as_array()) {
                format_tree(children, indent + 1, out, max_chars, truncated);
            }
        }
    }

    if truncated && out.len() < max_chars {
        out.push_str("\n... (truncated)\n");
    }
}

#[async_trait]
impl Capability for FileTreeCapability {
    fn name(&self) -> &str {
        "file_tree"
    }

    fn description(&self) -> &str {
        "Build a recursive file tree (args: path, max_depth?, ignore_patterns?, include_hidden?). Policy: allowed_paths, denied_paths, max_output_bytes."
    }

    async fn execute(&self, args: Value, config: &AppConfig) -> Result<CapabilityResult, String> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;

        let max_depth = args
            .get("max_depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as u32;

        let include_hidden = args
            .get("include_hidden")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let ignore_patterns: Vec<String> = args
            .get("ignore_patterns")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_else(|| {
                vec![
                    ".git".to_string(),
                    "node_modules".to_string(),
                    "__pycache__".to_string(),
                    ".venv".to_string(),
                    "target".to_string(),
                    "dist".to_string(),
                    "build".to_string(),
                ]
            });

        policy::validate_path(config, "file_tree", path)?;

        let root = Path::new(path)
            .canonicalize()
            .map_err(|e| e.to_string())?;

        let max_nodes = 20_000usize;
        let mut node_count = 0usize;
        let mut truncated = false;

        let tree = build_tree(
            &root,
            &root,
            0,
            max_depth,
            include_hidden,
            &ignore_patterns,
            &mut node_count,
            max_nodes,
            &mut truncated,
        )
        .await?;

        let mut total_files = 0u64;
        let mut total_dirs = 0u64;
        let mut stack: Vec<&Value> = tree.iter().collect();
        while let Some(n) = stack.pop() {
            let is_dir = n.get("is_dir").and_then(|v| v.as_bool()).unwrap_or(false);
            if is_dir {
                total_dirs += 1;
                if let Some(children) = n.get("children").and_then(|v| v.as_array()) {
                    for c in children {
                        stack.push(c);
                    }
                }
            } else {
                total_files += 1;
            }
        }

        let result = json!({
            "tree": tree,
            "total_files": total_files,
            "total_dirs": total_dirs,
            "truncated": truncated
        });

        let max_formatted_chars = 12_000usize;
        let mut formatted_tree = String::new();
        let empty: Vec<Value> = Vec::new();
        let nodes = result.get("tree").and_then(|v| v.as_array()).unwrap_or(&empty);
        format_tree(nodes, 0, &mut formatted_tree, max_formatted_chars, truncated);

        // Enforce output size policy (JSON + formatted).
        let json_size = serde_json::to_string(&result)
            .map(|s| s.len())
            .unwrap_or(0);
        let combined_size = json_size + formatted_tree.len();
        policy::check_output_limit(config, "file_tree", combined_size)?;

        Ok(CapabilityResult {
            formatted: format!(
                "File tree for {} (dirs: {}, files: {}, depth: {}, nodes: {}, truncated: {})\n\n{}",
                root.to_string_lossy(),
                total_dirs,
                total_files,
                max_depth,
                node_count,
                truncated,
                formatted_tree
            ),
            result,
        })
    }
}

