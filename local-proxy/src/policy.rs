//! Allowlist/denylist enforcement per capability.

use crate::config::AppConfig;
use std::path::Path;

pub fn capability_enabled(config: &AppConfig, name: &str) -> bool {
    config
        .capabilities
        .get(name)
        .map(|c| c.enabled)
        .unwrap_or(false)
}

/// Whether the daemon should advertise this capability to the backend on register.
pub fn capability_offered(config: &AppConfig, name: &str) -> bool {
    if name == "create_directory" {
        return capability_enabled(config, "create_directory")
            || capability_enabled(config, "write_file");
    }
    capability_enabled(config, name)
}

#[allow(dead_code)]
pub fn enabled_capabilities(config: &AppConfig) -> Vec<String> {
    config
        .capabilities
        .iter()
        .filter(|(_, c)| c.enabled)
        .map(|(k, _)| k.clone())
        .collect()
}

/// Validate path for read/list (path must exist). Canonicalizes to prevent traversal.
pub fn validate_path(
    config: &AppConfig,
    capability_name: &str,
    path: &str,
) -> Result<(), String> {
    let path_buf = Path::new(path)
        .canonicalize()
        .map_err(|e| format!("Invalid path or path does not exist: {}", e))?;
    let path_str = path_buf
        .to_str()
        .ok_or("Path contains invalid UTF-8")?;
    validate_path_inner(config, capability_name, path_str)
}

/// Validate path for write (file may not exist). Parent directory is canonicalized.
pub fn validate_path_for_write(
    config: &AppConfig,
    capability_name: &str,
    path: &str,
) -> Result<(), String> {
    let p = Path::new(path);
    let path_str = if p.exists() {
        p.canonicalize()
            .map_err(|e| format!("Invalid path: {}", e))?
            .to_str()
            .ok_or("Path contains invalid UTF-8")?
            .to_string()
    } else {
        let parent = p.parent().ok_or("Path has no parent")?;
        let canon_parent = parent
            .canonicalize()
            .map_err(|e| format!("Parent path does not exist: {}", e))?;
        let resolved = canon_parent.join(p.file_name().ok_or("Path has no file name")?);
        resolved
            .to_str()
            .ok_or("Path contains invalid UTF-8")?
            .to_string()
    };
    validate_path_inner(config, capability_name, &path_str)
}

fn validate_path_inner(config: &AppConfig, capability_name: &str, path_str: &str) -> Result<(), String> {
    let cap = config
        .capabilities
        .get(capability_name)
        .ok_or_else(|| format!("Capability {} not configured", capability_name))?;

    for denied in &cap.denied_paths {
        if path_contains(path_str, denied) {
            return Err(format!("Path denied by policy: {}", denied));
        }
    }

    if cap.allowed_paths.is_empty() {
        return Ok(());
    }

    let allowed = cap
        .allowed_paths
        .iter()
        .any(|p| path_starts_with(path_str, p));
    if !allowed {
        return Err(format!(
            "Path not in allowed list. Allowed prefixes: {:?}",
            cap.allowed_paths
        ));
    }

    Ok(())
}

fn path_starts_with(path: &str, prefix: &str) -> bool {
    let path_normalized = path.replace('\\', "/");
    let prefix_normalized = prefix.replace('\\', "/");
    path_normalized.starts_with(&prefix_normalized)
}

fn path_contains(path: &str, segment: &str) -> bool {
    let path_normalized = path.replace('\\', "/");
    let segment_normalized = segment.replace('\\', "/");
    path_normalized.contains(&segment_normalized)
}

/// Validate that command is allowed and not denied.
pub fn validate_command(
    config: &AppConfig,
    capability_name: &str,
    command: &str,
) -> Result<(), String> {
    let cap = config
        .capabilities
        .get(capability_name)
        .ok_or_else(|| format!("Capability {} not configured", capability_name))?;

    for pattern in &cap.denied_patterns {
        if command.contains(pattern) {
            return Err(format!("Command denied by policy (matches: {})", pattern));
        }
    }

    if cap.allowed_commands.is_empty() {
        return Ok(());
    }

    let first_token = command.trim().split_whitespace().next().unwrap_or("");
    let allowed = cap
        .allowed_commands
        .iter()
        .any(|c| first_token == c || first_token.ends_with(c));
    if !allowed {
        return Err(format!(
            "Command not in allowed list. Allowed: {:?}",
            cap.allowed_commands
        ));
    }

    Ok(())
}

/// Enforce max output size for a capability.
pub fn check_output_limit(
    config: &AppConfig,
    capability_name: &str,
    size: usize,
) -> Result<(), String> {
    let cap = config.capabilities.get(capability_name);
    let Some(max) = cap.and_then(|c| c.max_output_bytes) else {
        return Ok(());
    };
    if size > max {
        return Err(format!(
            "Output exceeds limit ({} > {} bytes)",
            size, max
        ));
    }
    Ok(())
}
