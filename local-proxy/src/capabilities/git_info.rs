//! Git info capability: read-only git operations for project awareness.

use super::{Capability, CapabilityResult};
use crate::config::AppConfig;
use crate::policy;
use async_trait::async_trait;
use serde_json::json;
use serde_json::Value;
use std::path::Path;

pub struct GitInfoCapability;

fn oid_to_short(oid: git2::Oid) -> String {
    let s = oid.to_string();
    s.chars().take(8).collect()
}

fn signature_to_string(sig: &git2::Signature) -> String {
    let name = sig.name().unwrap_or("");
    let email = sig.email().unwrap_or("");
    if name.is_empty() && email.is_empty() {
        return "".to_string();
    }
    if email.is_empty() {
        return name.to_string();
    }
    if name.is_empty() {
        return email.to_string();
    }
    format!("{} <{}>", name, email)
}

#[async_trait]
impl Capability for GitInfoCapability {
    fn name(&self) -> &str {
        "git_info"
    }

    fn description(&self) -> &str {
        "Read-only git operations (args: path, operation, file?, limit?, commit?). Operations: status|diff|log|branch|show. Policy: allowed_paths, denied_paths, max_output_bytes."
    }

    async fn execute(&self, args: Value, config: &AppConfig) -> Result<CapabilityResult, String> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;
        let operation = args
            .get("operation")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'operation' argument")?;

        policy::validate_path(config, "git_info", path)?;

        let repo = git2::Repository::discover(Path::new(path)).map_err(|e| e.to_string())?;

        let result = match operation {
            "status" => {
                let mut opts = git2::StatusOptions::new();
                opts.include_untracked(true)
                    .recurse_untracked_dirs(true)
                    .include_ignored(false);
                let statuses = repo.statuses(Some(&mut opts)).map_err(|e| e.to_string())?;

                let mut staged: Vec<String> = Vec::new();
                let mut modified: Vec<String> = Vec::new();
                let mut untracked: Vec<String> = Vec::new();
                let mut deleted: Vec<String> = Vec::new();
                let mut renamed: Vec<String> = Vec::new();

                for entry in statuses.iter() {
                    let s = entry.status();
                    let p = entry.path().unwrap_or("").to_string();
                    if p.is_empty() {
                        continue;
                    }
                    if s.contains(git2::Status::WT_NEW) {
                        untracked.push(p);
                        continue;
                    }
                    if s.intersects(git2::Status::INDEX_NEW | git2::Status::INDEX_MODIFIED | git2::Status::INDEX_DELETED | git2::Status::INDEX_RENAMED) {
                        staged.push(p.clone());
                    }
                    if s.contains(git2::Status::WT_MODIFIED) {
                        modified.push(p.clone());
                    }
                    if s.contains(git2::Status::WT_DELETED) || s.contains(git2::Status::INDEX_DELETED) {
                        deleted.push(p.clone());
                    }
                    if s.contains(git2::Status::INDEX_RENAMED) {
                        renamed.push(p.clone());
                    }
                }

                json!({
                    "staged": staged,
                    "modified": modified,
                    "untracked": untracked,
                    "deleted": deleted,
                    "renamed": renamed
                })
            }
            "branch" => {
                let head = repo.head().ok();
                let current = head
                    .as_ref()
                    .and_then(|h| h.shorthand())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "".to_string());

                let mut local: Vec<String> = Vec::new();
                let mut remote: Vec<String> = Vec::new();

                let branches = repo.branches(None).map_err(|e| e.to_string())?;
                for b in branches {
                    let (branch, kind) = b.map_err(|e| e.to_string())?;
                    let name = branch.name().ok().flatten().unwrap_or("").to_string();
                    if name.is_empty() {
                        continue;
                    }
                    match kind {
                        git2::BranchType::Local => local.push(name),
                        git2::BranchType::Remote => remote.push(name),
                    }
                }
                local.sort();
                remote.sort();

                json!({
                    "current": current,
                    "local": local,
                    "remote": remote
                })
            }
            "log" => {
                let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
                let mut revwalk = repo.revwalk().map_err(|e| e.to_string())?;
                revwalk.push_head().map_err(|e| e.to_string())?;
                revwalk.set_sorting(git2::Sort::TIME).ok();

                let mut commits: Vec<Value> = Vec::new();
                for oid in revwalk.take(limit) {
                    let oid = oid.map_err(|e| e.to_string())?;
                    let commit = repo.find_commit(oid).map_err(|e| e.to_string())?;
                    let summary = commit.summary().unwrap_or("").to_string();
                    let author = signature_to_string(&commit.author());
                    let time = commit.time().seconds();
                    commits.push(json!({
                        "oid": oid.to_string(),
                        "short": oid_to_short(oid),
                        "summary": summary,
                        "author": author,
                        "time": time
                    }));
                }
                json!({ "commits": commits, "count": commits.len() })
            }
            "show" => {
                let commit_str = args
                    .get("commit")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing 'commit' argument for operation 'show'")?;
                let oid = git2::Oid::from_str(commit_str).map_err(|e| e.to_string())?;
                let commit = repo.find_commit(oid).map_err(|e| e.to_string())?;
                let author = signature_to_string(&commit.author());
                let committer = signature_to_string(&commit.committer());
                let time = commit.time().seconds();
                let message = commit.message().unwrap_or("").to_string();
                json!({
                    "oid": oid.to_string(),
                    "short": oid_to_short(oid),
                    "author": author,
                    "committer": committer,
                    "time": time,
                    "message": message
                })
            }
            "diff" => {
                let file = args.get("file").and_then(|v| v.as_str()).map(|s| s.to_string());

                let mut opts = git2::DiffOptions::new();
                if let Some(ref f) = file {
                    opts.pathspec(f);
                }

                let diff = repo
                    .diff_index_to_workdir(None, Some(&mut opts))
                    .map_err(|e| e.to_string())?;

                let mut patch = String::new();
                diff.print(git2::DiffFormat::Patch, |_, _, line| {
                    if let Ok(s) = std::str::from_utf8(line.content()) {
                        patch.push_str(s);
                    }
                    true
                })
                .map_err(|e| e.to_string())?;

                json!({
                    "file": file,
                    "patch": patch
                })
            }
            _ => return Err("Unsupported operation. Use: status|diff|log|branch|show".to_string()),
        };

        let formatted = match operation {
            "status" => {
                let staged = result.get("staged").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
                let modified = result.get("modified").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
                let untracked = result.get("untracked").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
                format!("Git status (staged: {}, modified: {}, untracked: {})", staged, modified, untracked)
            }
            "diff" => {
                let file = result.get("file").and_then(|v| v.as_str()).unwrap_or("");
                if file.is_empty() {
                    "Git diff (working tree)".to_string()
                } else {
                    format!("Git diff for {}", file)
                }
            }
            "log" => {
                let count = result.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
                format!("Git log ({} commit(s))", count)
            }
            "branch" => {
                let current = result.get("current").and_then(|v| v.as_str()).unwrap_or("");
                format!("Git branches (current: {})", current)
            }
            "show" => {
                let short = result.get("short").and_then(|v| v.as_str()).unwrap_or("");
                format!("Git commit {}", short)
            }
            _ => "Git info".to_string(),
        };

        let json_size = serde_json::to_string(&result)
            .map(|s| s.len())
            .unwrap_or(0);
        policy::check_output_limit(config, "git_info", json_size + formatted.len())?;

        Ok(CapabilityResult { result, formatted })
    }
}

