//! Shared state between tray, daemon, and settings UI.

use crate::config::AppConfig;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::mpsc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionStatus {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Error,
}

#[derive(Debug, Clone)]
pub struct InvocationRecord {
    pub tool: String,
    pub request_id: String,
    pub success: bool,
    pub at: SystemTime,
}

pub type DaemonCommand = DaemonCommandKind;

#[derive(Debug, Clone)]
pub enum DaemonCommandKind {
    Connect,
    Disconnect,
    ReloadConfig,
    Quit,
}

pub struct AppState {
    pub config: AppConfig,
    pub connection_status: ConnectionStatus,
    pub connected_since: Option<std::time::Instant>,
    /// When true, daemon will auto-reconnect when the connection is lost.
    pub want_connected: bool,
    pub recent_invocations: VecDeque<InvocationRecord>,
    pub command_tx: Option<mpsc::Sender<DaemonCommand>>,
}

impl AppState {
    pub fn new(config: AppConfig) -> Self {
        Self {
            config,
            connection_status: ConnectionStatus::Disconnected,
            connected_since: None,
            want_connected: false,
            recent_invocations: VecDeque::new(),
            command_tx: None,
        }
    }

    pub fn push_invocation(&mut self, record: InvocationRecord) {
        self.recent_invocations.push_back(record);
        while self.recent_invocations.len() > 100 {
            self.recent_invocations.pop_front();
        }
    }
}

pub type SharedState = Arc<std::sync::Mutex<AppState>>;
