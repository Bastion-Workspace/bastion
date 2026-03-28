//! Bastion Local Proxy daemon: tray icon, settings window, WebSocket daemon.

#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

mod capabilities;
mod config;
mod daemon;
mod policy;
mod protocol;
mod shared_state;
mod settings_window;
mod tray;

use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tracing_subscriber::EnvFilter;

fn init_tracing() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let filter = EnvFilter::from_default_env().add_directive("bastion_local_proxy=info".parse()?);
    #[cfg(all(windows, not(debug_assertions)))]
    {
        let log_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("Bastion");
        let _ = std::fs::create_dir_all(&log_dir);
        let log_path = log_dir.join("bastion-proxy.log");
        if let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            tracing_subscriber::fmt()
                .with_writer(std::sync::Mutex::new(file))
                .with_ansi(false)
                .with_env_filter(filter)
                .init();
        } else {
            tracing_subscriber::fmt().with_env_filter(filter).init();
        }
    }
    #[cfg(not(all(windows, not(debug_assertions))))]
    {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }
    Ok(())
}

#[cfg(windows)]
fn pump_win32_messages() {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE,
    };
    unsafe {
        let mut msg: MSG = std::mem::zeroed();
        while PeekMessageW(&mut msg, 0, 0, 0, PM_REMOVE) != 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "bastion-proxy")]
#[command(about = "Bastion local proxy daemon")]
struct Args {
    #[arg(long)]
    config: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    init_tracing()?;

    let args = Args::parse();
    let config = config::load_config(args.config.as_deref()).unwrap_or_else(|_| config::AppConfig::default());
    let state = Arc::new(std::sync::Mutex::new(shared_state::AppState::new(config)));

    let (cmd_tx, cmd_rx) = tokio::sync::mpsc::channel(16);
    {
        let mut st = state.lock().unwrap();
        st.command_tx = Some(cmd_tx.clone());
        if st.config.auto_connect && !st.config.bastion_url.is_empty() && !st.config.token.is_empty() {
            st.want_connected = true;
            let _ = cmd_tx.try_send(shared_state::DaemonCommand::Connect);
        }
    }

    let registry = capabilities::CapabilityRegistry::new();
    let state_clone = state.clone();
    let daemon_handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        rt.block_on(daemon::run_daemon_loop(state_clone, registry, cmd_rx));
    });

    let mut tray_state = tray::create_tray_icon(state.clone())?;

    let mut open_settings = false;
    loop {
        tray_state.update_menu_labels();
        if let Some(action) = tray_state.try_recv() {
            match action {
                tray::TrayMenuAction::Disconnect => {
                    let connected = {
                        let st = state.lock().unwrap();
                        st.connection_status == shared_state::ConnectionStatus::Connected
                    };
                    if connected {
                        let mut st = state.lock().unwrap();
                        st.want_connected = false;
                        let _ = cmd_tx.try_send(shared_state::DaemonCommand::Disconnect);
                    } else {
                        let mut st = state.lock().unwrap();
                        st.want_connected = true;
                        let _ = cmd_tx.try_send(shared_state::DaemonCommand::Connect);
                    }
                }
                tray::TrayMenuAction::ToggleScreenshot => {
                    let mut st = state.lock().unwrap();
                    let enabled = st
                        .config
                        .capabilities
                        .get("screenshot")
                        .map(|c| c.enabled)
                        .unwrap_or(false);
                    st.config
                        .capabilities
                        .entry("screenshot".to_string())
                        .or_default()
                        .enabled = !enabled;
                    let _ = config::save_config(&st.config, None);
                    let _ = cmd_tx.try_send(shared_state::DaemonCommand::ReloadConfig);
                }
                tray::TrayMenuAction::OpenSettings => {
                    open_settings = true;
                }
                tray::TrayMenuAction::Quit => {
                    let _ = cmd_tx.try_send(shared_state::DaemonCommand::Quit);
                    break;
                }
            }
        }
        if open_settings {
            open_settings = false;
            settings_window::run_settings_window(state.clone());
        }
        #[cfg(windows)]
        pump_win32_messages();
        std::thread::sleep(Duration::from_millis(50));
    }

    let _ = daemon_handle.join();
    Ok(())
}
