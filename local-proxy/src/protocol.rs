//! Protocol message types for WebSocket communication with Bastion backend.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonToBackend {
    Auth {
        token: String,
    },
    Register {
        device_id: String,
        capabilities: Vec<String>,
    },
    Result {
        request_id: String,
        result: serde_json::Value,
        formatted: String,
    },
    Error {
        request_id: String,
        error: String,
    },
    WorkspaceSet {
        request_id: String,
        workspace_root: String,
        file_count: u64,
        git_detected: bool,
    },
    Heartbeat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BackendToDaemon {
    Invoke {
        request_id: String,
        tool: String,
        args: serde_json::Value,
    },
    SetWorkspace {
        request_id: String,
        workspace_root: String,
    },
    Ping,
}
