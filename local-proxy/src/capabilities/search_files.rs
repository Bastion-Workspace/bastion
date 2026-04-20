//! Search files capability: ripgrep-like content search across a directory tree.

use super::{Capability, CapabilityResult};
use crate::config::AppConfig;
use crate::policy;
use async_trait::async_trait;
use serde_json::json;
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokio::fs;

pub struct SearchFilesCapability;

fn matches_glob(path: &Path, glob: &Option<String>) -> bool {
    let Some(g) = glob else {
        return true;
    };
    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");

    // Minimal glob support for Phase 1:
    // - "*.ext" matches file suffix
    // - exact filename match
    if let Some(suffix) = g.strip_prefix("*.") {
        return file_name.ends_with(&format!(".{}", suffix));
    }
    file_name == g
}

fn split_lines(s: &str) -> Vec<&str> {
    s.lines().collect()
}

fn is_ignored_dir(name: &str) -> bool {
    matches!(
        name,
        ".git" | "node_modules" | "__pycache__" | ".venv" | "target" | "dist" | "build"
    )
}

async fn walk_and_search(
    root: &Path,
    dir: &Path,
    pattern: &regex::Regex,
    glob: &Option<String>,
    max_results: usize,
    context_lines: usize,
    matches_out: &mut Vec<Value>,
    files_searched: &mut u64,
    truncated: &mut bool,
    max_file_bytes: usize,
    include_hidden: bool,
) -> Result<(), String> {
    let mut rd = fs::read_dir(dir).await.map_err(|e| e.to_string())?;
    while let Some(entry) = rd.next_entry().await.map_err(|e| e.to_string())? {
        if matches_out.len() >= max_results {
            *truncated = true;
            return Ok(());
        }

        let file_name = entry.file_name();
        let name = file_name.to_string_lossy().into_owned();
        if !include_hidden && name.starts_with('.') {
            continue;
        }

        let meta = entry.metadata().await.map_err(|e| e.to_string())?;
        let path: PathBuf = entry.path();
        if meta.is_dir() {
            if is_ignored_dir(&name) {
                continue;
            }
            walk_and_search(
                root,
                &path,
                pattern,
                glob,
                max_results,
                context_lines,
                matches_out,
                files_searched,
                truncated,
                max_file_bytes,
                include_hidden,
            )
            .await?;
            if *truncated {
                return Ok(());
            }
            continue;
        }

        if !meta.is_file() {
            continue;
        }
        if !matches_glob(&path, glob) {
            continue;
        }

        *files_searched += 1;
        let size = meta.len() as usize;
        if size > max_file_bytes {
            continue;
        }

        let content = match fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(_) => continue, // skip non-text / unreadable files
        };

        let lines = split_lines(&content);
        for (idx, line) in lines.iter().enumerate() {
            if matches_out.len() >= max_results {
                *truncated = true;
                return Ok(());
            }
            if !pattern.is_match(line) {
                continue;
            }

            let start = idx.saturating_sub(context_lines);
            let end = (idx + 1 + context_lines).min(lines.len());

            let before: Vec<String> = lines[start..idx].iter().map(|s| s.to_string()).collect();
            let after: Vec<String> = lines[(idx + 1)..end].iter().map(|s| s.to_string()).collect();

            let rel_path = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");

            matches_out.push(json!({
                "file": rel_path,
                "line_number": (idx + 1) as u64,
                "line_content": (*line).to_string(),
                "context_before": before,
                "context_after": after
            }));
        }
    }
    Ok(())
}

#[async_trait]
impl Capability for SearchFilesCapability {
    fn name(&self) -> &str {
        "search_files"
    }

    fn description(&self) -> &str {
        "Search for a regex pattern in files under a directory (args: pattern, path, glob?, max_results?, context_lines?, case_insensitive?, include_hidden?). Policy: allowed_paths, denied_paths, max_output_bytes, max_file_bytes."
    }

    async fn execute(&self, args: Value, config: &AppConfig) -> Result<CapabilityResult, String> {
        let pattern_str = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'pattern' argument")?;
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;

        let glob = args.get("glob").and_then(|v| v.as_str()).map(|s| s.to_string());
        let max_results = args
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(100) as usize;
        let context_lines = args
            .get("context_lines")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let case_insensitive = args
            .get("case_insensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let include_hidden = args
            .get("include_hidden")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        policy::validate_path(config, "search_files", path)?;

        let root = Path::new(path).canonicalize().map_err(|e| e.to_string())?;

        let cap_cfg = config.capabilities.get("search_files");
        let max_file_bytes = cap_cfg
            .and_then(|c| c.max_file_bytes)
            .unwrap_or(2 * 1024 * 1024);

        let regex_pat = if case_insensitive {
            format!("(?i){}", pattern_str)
        } else {
            pattern_str.to_string()
        };
        let pattern = regex::Regex::new(&regex_pat).map_err(|e| e.to_string())?;

        let mut matches_out: Vec<Value> = Vec::new();
        let mut files_searched = 0u64;
        let mut truncated = false;

        walk_and_search(
            &root,
            &root,
            &pattern,
            &glob,
            max_results,
            context_lines,
            &mut matches_out,
            &mut files_searched,
            &mut truncated,
            max_file_bytes,
            include_hidden,
        )
        .await?;

        let result = json!({
            "matches": matches_out,
            "total_matches": matches_out.len(),
            "files_searched": files_searched,
            "truncated": truncated
        });

        let mut formatted = String::new();
        formatted.push_str(&format!(
            "Search results for /{}/ under {} (matches: {}, files searched: {}, truncated: {})\n",
            pattern_str,
            root.to_string_lossy(),
            result
                .get("total_matches")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            files_searched,
            truncated
        ));

        // Include a short preview of the first N matches.
        let preview_max = 20usize;
        if let Some(arr) = result.get("matches").and_then(|v| v.as_array()) {
            for (i, m) in arr.iter().take(preview_max).enumerate() {
                let file = m.get("file").and_then(|v| v.as_str()).unwrap_or("");
                let line_number = m.get("line_number").and_then(|v| v.as_u64()).unwrap_or(0);
                let line = m.get("line_content").and_then(|v| v.as_str()).unwrap_or("");
                formatted.push_str(&format!("\n{}. {}:{}\n{}", i + 1, file, line_number, line));
            }
            if arr.len() > preview_max {
                formatted.push_str(&format!("\n\n... (showing first {}, {} total)", preview_max, arr.len()));
            }
        }

        let json_size = serde_json::to_string(&result)
            .map(|s| s.len())
            .unwrap_or(0);
        let combined_size = json_size + formatted.len();
        policy::check_output_limit(config, "search_files", combined_size)?;

        Ok(CapabilityResult { result, formatted })
    }
}

