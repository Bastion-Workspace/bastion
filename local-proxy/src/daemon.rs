//! WebSocket client loop: connect, register, receive invokes, dispatch to capabilities, respond.

use crate::capabilities::CapabilityRegistry;
use crate::config::AppConfig;
use crate::policy;
use crate::protocol::{BackendToDaemon, DaemonToBackend};
use crate::shared_state::{ConnectionStatus, DaemonCommand, InvocationRecord, SharedState};
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{error, info, warn};

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);
const RECONNECT_BASE: Duration = Duration::from_secs(1);
const RECONNECT_MAX: Duration = Duration::from_secs(60);

fn ws_url(config: &AppConfig) -> String {
    let base = config.bastion_url.trim_end_matches('/');
    let scheme = if base.starts_with("https") { "wss" } else { "ws" };
    let host = base
        .trim_start_matches("https://")
        .trim_start_matches("http://");
    format!("{}://{}/api/ws/device", scheme, host)
}

fn resolve_paths_with_workspace_root(mut args: Value, workspace_root: Option<&str>) -> Value {
    let Some(root) = workspace_root else {
        return args;
    };
    let root_path = std::path::Path::new(root);
    let Value::Object(ref mut map) = args else {
        return args;
    };

    for key in ["path", "cwd"] {
        let Some(v) = map.get(key).and_then(|v| v.as_str()).map(|s| s.to_string()) else {
            continue;
        };
        let p = std::path::Path::new(&v);
        if p.is_absolute() {
            continue;
        }
        let joined = root_path.join(p);
        if let Some(s) = joined.to_str() {
            map.insert(key.to_string(), Value::String(s.to_string()));
        }
    }
    args
}

async fn quick_file_count(root: &std::path::Path) -> u64 {
    let mut count: u64 = 0;
    let mut stack: Vec<std::path::PathBuf> = vec![root.to_path_buf()];
    let max = 50_000u64;
    while let Some(p) = stack.pop() {
        if count >= max {
            break;
        }
        let mut rd = match tokio::fs::read_dir(&p).await {
            Ok(r) => r,
            Err(_) => continue,
        };
        while let Ok(Some(e)) = rd.next_entry().await {
            if count >= max {
                break;
            }
            let name = e.file_name().to_string_lossy().into_owned();
            if matches!(
                name.as_str(),
                ".git" | "node_modules" | "__pycache__" | ".venv" | "target" | "dist" | "build"
            ) {
                continue;
            }
            let meta = match e.metadata().await {
                Ok(m) => m,
                Err(_) => continue,
            };
            if meta.is_dir() {
                stack.push(e.path());
            } else if meta.is_file() {
                count += 1;
            }
        }
    }
    count
}

pub async fn run_daemon_loop(
    state: SharedState,
    registry: CapabilityRegistry,
    mut cmd_rx: mpsc::Receiver<DaemonCommand>,
) {
    let mut backoff = RECONNECT_BASE;
    loop {
        {
            let st = state.lock().unwrap();
            if st.connection_status == ConnectionStatus::Disconnected {
                let want = st.want_connected;
                let has_config = !st.config.bastion_url.is_empty() && !st.config.token.is_empty();
                drop(st);
                if !want || !has_config {
                    match cmd_rx.recv().await {
                        Some(DaemonCommand::Quit) => break,
                        Some(DaemonCommand::Connect) => {
                            let mut st = state.lock().unwrap();
                            st.want_connected = true;
                        }
                        Some(DaemonCommand::Disconnect) => {}
                        _ => continue,
                    }
                }
            }
        }

        let config = {
            let st = state.lock().unwrap();
            st.config.clone()
        };

        if config.bastion_url.is_empty() || config.token.is_empty() {
            warn!("Missing bastion_url or token; not connecting");
            tokio::time::sleep(Duration::from_secs(5)).await;
            continue;
        }

        let url = ws_url(&config);
        {
            let mut st = state.lock().unwrap();
            st.connection_status = ConnectionStatus::Connecting;
        }

        match connect_async(&url).await {
            Ok((ws_stream, _)) => {
                backoff = RECONNECT_BASE;
                let (mut write, mut read) = ws_stream.split();

                let auth_msg = DaemonToBackend::Auth {
                    token: config.token.clone(),
                };
                let auth_json = serde_json::to_string(&auth_msg).unwrap();
                if let Err(e) = write.send(Message::Text(auth_json)).await {
                    error!("Failed to send auth: {}", e);
                    let mut st = state.lock().unwrap();
                    st.connection_status = ConnectionStatus::Error;
                    tokio::time::sleep(backoff).await;
                    continue;
                }

                let capabilities: Vec<String> = registry
                    .names()
                    .into_iter()
                    .filter(|n| policy::capability_offered(&config, n))
                    .collect();

                let register_msg = DaemonToBackend::Register {
                    device_id: config.device_id.clone(),
                    capabilities: capabilities.clone(),
                };
                let register_json = serde_json::to_string(&register_msg).unwrap();
                if let Err(e) = write.send(Message::Text(register_json)).await {
                    error!("Failed to send register: {}", e);
                    let mut st = state.lock().unwrap();
                    st.connection_status = ConnectionStatus::Error;
                    tokio::time::sleep(backoff).await;
                    continue;
                }

                {
                    let mut st = state.lock().unwrap();
                    st.connection_status = ConnectionStatus::Connected;
                    st.connected_since = Some(std::time::Instant::now());
                }
                info!("Registered with backend: device_id={}", config.device_id);

                let mut heartbeat = tokio::time::interval(HEARTBEAT_INTERVAL);
                heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

                let mut quit_requested = false;
                loop {
                    tokio::select! {
                        cmd = cmd_rx.recv() => {
                            match cmd {
                                Some(DaemonCommand::Quit) => {
                                    quit_requested = true;
                                    let _ = write.send(Message::Close(None)).await;
                                    break;
                                }
                                Some(DaemonCommand::Disconnect) => {
                                    let mut st = state.lock().unwrap();
                                    st.want_connected = false;
                                    let _ = write.send(Message::Close(None)).await;
                                    break;
                                }
                                Some(DaemonCommand::ReloadConfig) => {
                                    let (new_config, new_caps) = {
                                        let st = state.lock().unwrap();
                                        let caps: Vec<String> = registry
                                            .names()
                                            .into_iter()
                                            .filter(|n| policy::capability_offered(&st.config, n))
                                            .collect();
                                        (st.config.clone(), caps)
                                    };
                                    let reg = DaemonToBackend::Register {
                                        device_id: new_config.device_id,
                                        capabilities: new_caps,
                                    };
                                    let _ = write.send(Message::Text(serde_json::to_string(&reg).unwrap())).await;
                                }
                                Some(DaemonCommand::Connect) => {}
                                None => break,
                            }
                        }
                        _ = heartbeat.tick() => {
                            let msg = DaemonToBackend::Heartbeat;
                            let _ = write.send(Message::Text(serde_json::to_string(&msg).unwrap())).await;
                        }
                        msg = read.next() => {
                            let msg = match msg {
                                Some(Ok(Message::Text(t))) => t,
                                Some(Ok(Message::Close(_))) => break,
                                Some(Err(e)) => {
                                    error!("WebSocket error: {}", e);
                                    break;
                                }
                                Some(Ok(_)) => continue,
                                None => break,
                            };

                            let backend_msg: BackendToDaemon = match serde_json::from_str(&msg) {
                                Ok(m) => m,
                                Err(_) => continue,
                            };

                            match backend_msg {
                                BackendToDaemon::Invoke { request_id, tool, args } => {
                                    let cap = registry.get(&tool);
                                    let result_msg = match cap {
                                        Some(c) => {
                                            let state_clone = state.clone();
                                            let workspace_root = {
                                                let st = state_clone.lock().unwrap();
                                                st.active_workspace_root.clone()
                                            };
                                            let resolved_args = resolve_paths_with_workspace_root(
                                                args,
                                                workspace_root.as_deref(),
                                            );
                                            let exec = c.execute(resolved_args, &config);
                                            let out = exec.await;
                                            match out {
                                                Ok(res) => {
                                                    let rec = InvocationRecord {
                                                        tool: tool.clone(),
                                                        request_id: request_id.clone(),
                                                        success: true,
                                                        at: std::time::SystemTime::now(),
                                                    };
                                                    {
                                                        let mut st = state_clone.lock().unwrap();
                                                        st.push_invocation(rec);
                                                    }
                                                    DaemonToBackend::Result {
                                                        request_id,
                                                        result: res.result,
                                                        formatted: res.formatted,
                                                    }
                                                }
                                                Err(e) => {
                                                    let rec = InvocationRecord {
                                                        tool: tool.clone(),
                                                        request_id: request_id.clone(),
                                                        success: false,
                                                        at: std::time::SystemTime::now(),
                                                    };
                                                    {
                                                        let mut st = state_clone.lock().unwrap();
                                                        st.push_invocation(rec);
                                                    }
                                                    DaemonToBackend::Error { request_id, error: e }
                                                }
                                            }
                                        }
                                        None => DaemonToBackend::Error {
                                            request_id: request_id.clone(),
                                            error: format!("Unknown capability: {}", tool),
                                        },
                                    };
                                    let json = serde_json::to_string(&result_msg).unwrap();
                                    if let Err(e) = write.send(Message::Text(json)).await {
                                        error!("Failed to send result: {}", e);
                                        break;
                                    }
                                }
                                BackendToDaemon::SetWorkspace { request_id, workspace_root } => {
                                    // Validate and set the active workspace root.
                                    let root_path = std::path::Path::new(&workspace_root);
                                    let canon = match root_path.canonicalize() {
                                        Ok(p) => p,
                                        Err(e) => {
                                            let err_msg = DaemonToBackend::Error {
                                                request_id,
                                                error: format!("Invalid workspace_root: {}", e),
                                            };
                                            let _ = write
                                                .send(Message::Text(
                                                    serde_json::to_string(&err_msg).unwrap(),
                                                ))
                                                .await;
                                            continue;
                                        }
                                    };

                                    let git_detected = canon.join(".git").exists();
                                    let file_count = quick_file_count(&canon).await;

                                    {
                                        let mut st = state.lock().unwrap();
                                        st.active_workspace_root =
                                            canon.to_str().map(|s| s.to_string());
                                    }

                                    let msg = DaemonToBackend::WorkspaceSet {
                                        request_id,
                                        workspace_root: canon.to_string_lossy().into_owned(),
                                        file_count,
                                        git_detected,
                                    };
                                    let _ = write
                                        .send(Message::Text(serde_json::to_string(&msg).unwrap()))
                                        .await;
                                }
                                BackendToDaemon::Ping => {}
                            }
                        }
                    }
                }

                {
                    let mut st = state.lock().unwrap();
                    st.connection_status = ConnectionStatus::Disconnected;
                    st.connected_since = None;
                }
                if quit_requested {
                    break;
                }
                let want_connected = {
                    let st = state.lock().unwrap();
                    st.want_connected
                };
                if want_connected {
                    let mut st = state.lock().unwrap();
                    st.connection_status = ConnectionStatus::Reconnecting;
                }
                if want_connected {
                    tokio::time::sleep(backoff).await;
                    let still_want = {
                        let st = state.lock().unwrap();
                        st.want_connected
                    };
                    if !still_want {
                        let mut st = state.lock().unwrap();
                        st.connection_status = ConnectionStatus::Disconnected;
                    } else {
                        backoff = (backoff * 2).min(RECONNECT_MAX);
                    }
                }
            }
            Err(e) => {
                error!("WebSocket connect failed: {}", e);
                let want_connected = {
                    let mut st = state.lock().unwrap();
                    st.connection_status = if st.want_connected {
                        ConnectionStatus::Reconnecting
                    } else {
                        ConnectionStatus::Disconnected
                    };
                    st.want_connected
                };
                if want_connected {
                    tokio::time::sleep(backoff).await;
                    let still_want = {
                        let st = state.lock().unwrap();
                        st.want_connected
                    };
                    if still_want {
                        backoff = (backoff * 2).min(RECONNECT_MAX);
                    }
                }
            }
        }
    }
}
