//! Index workspace: walk source files and emit text chunks for server-side embedding.

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

pub struct IndexWorkspaceCapability;

const WINDOW_LINES: usize = 55;
const STEP_LINES: usize = 35;

fn is_ignored_dir(name: &str) -> bool {
    matches!(
        name,
        ".git" | "node_modules" | "__pycache__" | ".venv" | "target" | "dist" | "build"
    )
}

fn lang_for_ext(ext: &str) -> &'static str {
    match ext {
        "py" => "python",
        "rs" => "rust",
        "go" => "go",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" => "javascript",
        "java" => "java",
        "kt" | "kts" => "kotlin",
        "c" | "h" => "c",
        "cpp" | "hpp" | "cc" | "cxx" => "cpp",
        "cs" => "csharp",
        "rb" => "ruby",
        "php" => "php",
        "swift" => "swift",
        "scala" => "scala",
        "sql" => "sql",
        "md" => "markdown",
        "yml" | "yaml" => "yaml",
        "json" => "json",
        "toml" => "toml",
        "sh" | "bash" => "shell",
        "proto" => "protobuf",
        "dockerfile" => "dockerfile",
        _ => "text",
    }
}

/// Returns a lowercase extension (or `"dockerfile"`) when the path should be indexed.
fn indexable_extension(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?.to_lowercase();
    if name == "dockerfile" {
        return Some("dockerfile".to_string());
    }
    let ext = path.extension()?.to_str()?.to_lowercase();
    if matches!(
        ext.as_str(),
        "py" | "rs" | "go" | "ts" | "tsx" | "js" | "jsx" | "java" | "kt" | "kts" | "c" | "h"
            | "cpp" | "hpp" | "cc" | "cxx" | "cs" | "rb" | "php" | "swift" | "scala" | "sql" | "md"
            | "yml" | "yaml" | "json" | "toml" | "sh" | "bash" | "proto" | "vue" | "svelte" | "css"
            | "scss" | "html" | "htm" | "xml" | "gradle" | "properties" | "ini" | "cfg" | "tf"
            | "nix" | "lua" | "r" | "m" | "ex" | "exs" | "erl" | "hrl" | "clj" | "cljs" | "hs"
    ) {
        return Some(ext);
    }
    None
}

fn split_lines(s: &str) -> Vec<&str> {
    s.lines().collect()
}

/// Returns `Some(next_1based_line)` if stopped early inside the file.
fn push_chunks_from_file(
    rel_path: &str,
    lines: &[&str],
    lang: &str,
    git_sha: &str,
    start_line_1based: usize,
    chunks_out: &mut Vec<Value>,
    max_chunks: usize,
) -> Option<usize> {
    if lines.is_empty() {
        return None;
    }
    let start0 = start_line_1based.saturating_sub(1).min(lines.len().saturating_sub(1));
    let mut i = start0;
    let mut chunk_index: i32 = 0;
    while i < lines.len() {
        if chunks_out.len() >= max_chunks {
            return Some(i + 1);
        }
        let end = (i + WINDOW_LINES).min(lines.len());
        let body = lines[i..end].join("\n");
        let start_ln = i + 1;
        let end_ln = end;
        chunks_out.push(json!({
            "file_path": rel_path,
            "chunk_index": chunk_index,
            "start_line": start_ln as u64,
            "end_line": end_ln as u64,
            "content": body,
            "language": lang,
            "git_sha": git_sha,
        }));
        chunk_index += 1;
        if end >= lines.len() {
            break;
        }
        i += STEP_LINES;
    }
    None
}

struct IndexState {
    resume_path: Option<String>,
    resume_line: usize,
    resume_active: bool,
}

impl IndexState {
    fn start_line_for_file(&mut self, rel_path: &str) -> usize {
        if self.resume_active {
            if let Some(ref rp) = self.resume_path {
                if rel_path < rp.as_str() {
                    return 0; // skip file
                }
                if rel_path == rp.as_str() {
                    let ln = self.resume_line.max(1);
                    self.resume_active = false;
                    self.resume_path = None;
                    return ln;
                }
                if rel_path > rp.as_str() {
                    // Resume target missing or already passed — continue normally.
                    self.resume_active = false;
                    self.resume_path = None;
                }
            }
        }
        1
    }

    fn should_skip_file(&self, rel_path: &str) -> bool {
        if let Some(ref rp) = self.resume_path {
            if self.resume_active && rel_path < rp.as_str() {
                return true;
            }
        }
        false
    }
}

fn walk_index<'a>(
    root: &'a Path,
    dir: &'a Path,
    max_chunks: usize,
    max_files: usize,
    max_file_bytes: usize,
    include_hidden: bool,
    chunks_out: &'a mut Vec<Value>,
    files_done: &'a mut usize,
    truncated: &'a mut bool,
    next_resume: &'a mut Option<(String, usize)>,
    state: &'a mut IndexState,
) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + 'a>> {
    Box::pin(async move {
        let mut rd = fs::read_dir(dir).await.map_err(|e| e.to_string())?;
        let mut entries: Vec<PathBuf> = Vec::new();
        while let Some(entry) = rd.next_entry().await.map_err(|e| e.to_string())? {
            entries.push(entry.path());
        }
        entries.sort();
        for path in entries {
            if chunks_out.len() >= max_chunks {
                *truncated = true;
                return Ok(());
            }
            let file_name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            if !include_hidden && file_name.starts_with('.') {
                continue;
            }
            let meta = fs::metadata(&path).await.map_err(|e| e.to_string())?;
            if meta.is_dir() {
                if is_ignored_dir(&file_name) {
                    continue;
                }
                walk_index(
                    root,
                    &path,
                    max_chunks,
                    max_files,
                    max_file_bytes,
                    include_hidden,
                    chunks_out,
                    files_done,
                    truncated,
                    next_resume,
                    state,
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
            let rel_path = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");

            if state.should_skip_file(&rel_path) {
                continue;
            }

            let ext_lang = match indexable_extension(&path) {
                Some(ext) => lang_for_ext(ext.as_str()),
                None => continue,
            };

            let start_ln = state.start_line_for_file(&rel_path);

            if *files_done >= max_files {
                *truncated = true;
                *next_resume = Some((rel_path.clone(), start_ln));
                return Ok(());
            }

            let size = meta.len() as usize;
            if size > max_file_bytes {
                continue;
            }

            let content = match fs::read_to_string(&path).await {
                Ok(c) => c,
                Err(_) => continue,
            };
            *files_done += 1;

            let lines = split_lines(&content);
            if let Some(next_line) =
                push_chunks_from_file(&rel_path, &lines, ext_lang, "", start_ln, chunks_out, max_chunks)
            {
                *truncated = true;
                *next_resume = Some((rel_path, next_line));
                return Ok(());
            }
        }
        Ok(())
    })
}

#[async_trait]
impl Capability for IndexWorkspaceCapability {
    fn name(&self) -> &str {
        "index_workspace"
    }

    fn description(&self) -> &str {
        "Index source files under a directory into chunk JSON for Bastion semantic code search (args: path, max_chunks?, max_files?, resume_from_path?, resume_start_line?, include_hidden?)."
    }

    async fn execute(&self, args: Value, config: &AppConfig) -> Result<CapabilityResult, String> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;
        let max_chunks = args.get("max_chunks").and_then(|v| v.as_u64()).unwrap_or(200) as usize;
        let max_files = args.get("max_files").and_then(|v| v.as_u64()).unwrap_or(120) as usize;
        let include_hidden = args
            .get("include_hidden")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let resume_from_path = args
            .get("resume_from_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let resume_start_line = args
            .get("resume_start_line")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;

        policy::validate_path(config, "index_workspace", path)?;
        let root = Path::new(path).canonicalize().map_err(|e| e.to_string())?;

        let cap_cfg = config.capabilities.get("index_workspace");
        let max_file_bytes = cap_cfg
            .and_then(|c| c.max_file_bytes)
            .unwrap_or(512 * 1024);

        let mut chunks: Vec<Value> = Vec::new();
        let mut files_done = 0usize;
        let mut truncated = false;
        let mut next_resume: Option<(String, usize)> = None;
        let mut state = IndexState {
            resume_path: resume_from_path,
            resume_line: resume_start_line,
            resume_active: true,
        };

        walk_index(
            &root,
            &root,
            max_chunks,
            max_files,
            max_file_bytes,
            include_hidden,
            &mut chunks,
            &mut files_done,
            &mut truncated,
            &mut next_resume,
            &mut state,
        )
        .await?;

        let next_path = next_resume.as_ref().map(|(p, _)| p.clone());
        let next_line = next_resume.map(|(_, l)| l as u64);

        let result = json!({
            "chunks": chunks,
            "files_indexed": files_done,
            "truncated": truncated,
            "next_resume_path": next_path,
            "next_resume_line": next_line,
        });

        let mut formatted = String::new();
        formatted.push_str(&format!(
            "index_workspace: {} chunk(s) from {} file(s) under {} (truncated={})\n",
            chunks.len(),
            files_done,
            root.to_string_lossy(),
            truncated
        ));

        let json_size = serde_json::to_string(&result).map(|s| s.len()).unwrap_or(0);
        let combined_size = json_size + formatted.len();
        policy::check_output_limit(config, "index_workspace", combined_size)?;

        Ok(CapabilityResult { result, formatted })
    }
}
