//! egui/eframe settings window: Connection, Capabilities, Status tabs.

use crate::config::save_config;
use crate::policy;
use crate::shared_state::{ConnectionStatus, InvocationRecord, SharedState};
use eframe::egui;
use std::time::SystemTime;

fn format_relative_time(at: SystemTime) -> String {
    let now = SystemTime::now();
    let Ok(dur) = now.duration_since(at) else {
        return "just now".to_string();
    };
    let secs = dur.as_secs();
    if secs < 60 {
        format!("{}s ago", secs)
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86400)
    }
}

pub fn run_settings_window(state: SharedState) {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([420.0, 380.0]),
        ..Default::default()
    };
    let _ = eframe::run_native(
        "Bastion Local Proxy - Settings",
        options,
        Box::new(move |_cc| Ok(Box::new(SettingsApp::new(state)))),
    );
}

struct SettingsApp {
    state: SharedState,
    current_tab: u8,
    url_buf: String,
    token_buf: String,
    device_id_buf: String,
}

impl SettingsApp {
    fn new(state: SharedState) -> Self {
        let st = state.lock().unwrap();
        let url_buf = st.config.bastion_url.clone();
        let token_buf = st.config.token.clone();
        let device_id_buf = st.config.device_id.clone();
        drop(st);
        Self {
            state,
            current_tab: 0,
            url_buf,
            token_buf,
            device_id_buf,
        }
    }
}

impl eframe::App for SettingsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.heading("Bastion Local Proxy");
                ui.add_space(8.0);

                ui.horizontal(|ui| {
                    if ui.selectable_label(self.current_tab == 0, "Connection").clicked() {
                        self.current_tab = 0;
                    }
                    if ui.selectable_label(self.current_tab == 1, "Capabilities").clicked() {
                        self.current_tab = 1;
                    }
                    if ui.selectable_label(self.current_tab == 2, "Status").clicked() {
                        self.current_tab = 2;
                    }
                });
                ui.add_space(8.0);

                match self.current_tab {
                    0 => self.connection_tab(ui),
                    1 => self.capabilities_tab(ui),
                    2 => self.status_tab(ui),
                    _ => {}
                }
            });
    }
}

impl SettingsApp {
    fn connection_tab(&mut self, ui: &mut egui::Ui) {
        ui.label("Bastion URL:");
        ui.add_space(2.0);
        ui.text_edit_singleline(&mut self.url_buf);
        ui.add_space(4.0);
        ui.label("Device token:");
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new(
                "Generate a token in the Bastion web UI under\nSettings → Device Tokens, then paste it here.",
            )
            .small()
            .weak(),
        );
        ui.add_space(2.0);
        ui.text_edit_singleline(&mut self.token_buf);
        ui.add_space(4.0);
        ui.label("Device ID:");
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new("A name for this machine (e.g. \"my-laptop\").")
                .small()
                .weak(),
        );
        ui.add_space(2.0);
        ui.text_edit_singleline(&mut self.device_id_buf);
        ui.add_space(8.0);
        if ui.button("Save").clicked() {
            let mut st = self.state.lock().unwrap();
            st.config.bastion_url = self.url_buf.clone();
            st.config.token = self.token_buf.clone();
            st.config.device_id = self.device_id_buf.clone();
            if let Err(e) = save_config(&st.config, None) {
                tracing::error!("Failed to save config: {}", e);
            }
        }
        ui.add_space(8.0);
        let (status_text, want_connected) = {
            let st = self.state.lock().unwrap();
            let status_text = match st.connection_status {
                ConnectionStatus::Disconnected => "Disconnected",
                ConnectionStatus::Connecting => "Connecting…",
                ConnectionStatus::Connected => "Connected",
                ConnectionStatus::Reconnecting => "Reconnecting…",
                ConnectionStatus::Error => "Error",
            };
            (status_text.to_string(), st.want_connected)
        };
        ui.label(format!("Status: {}", status_text));
        ui.add_space(2.0);
        let mut want = want_connected;
        if ui.checkbox(&mut want, "Connect to server (auto-reconnect when connection is lost)").changed() {
            let mut st = self.state.lock().unwrap();
            st.want_connected = want;
            if let Some(ref tx) = st.command_tx {
                let _ = tx.try_send(if want {
                    crate::shared_state::DaemonCommand::Connect
                } else {
                    crate::shared_state::DaemonCommand::Disconnect
                });
            }
        }
    }

    fn capabilities_tab(&mut self, ui: &mut egui::Ui) {
        ui.label("Enable or disable capabilities offered to the server. Disabled capabilities are not advertised.");
        ui.add_space(4.0);
        egui::ScrollArea::vertical().show(ui, |ui| {
            for (id, label) in crate::capabilities::capabilities_ui_iter() {
                // Reflect implicit rules (e.g. write_file offered when read_file is on and write_file isn't explicitly disabled).
                let mut enabled = {
                    let st = self.state.lock().unwrap();
                    policy::capability_offered(&st.config, id)
                };
                if ui.checkbox(&mut enabled, label).changed() {
                    let mut st = self.state.lock().unwrap();
                    st.config
                        .capabilities
                        .entry(id.to_string())
                        .or_default()
                        .enabled = enabled;
                    if let Err(e) = save_config(&st.config, None) {
                        tracing::error!("Failed to save config: {}", e);
                    }
                    if let Some(ref tx) = st.command_tx {
                        let _ = tx.try_send(crate::shared_state::DaemonCommand::ReloadConfig);
                    }
                }
            }
        });
    }

    fn status_tab(&mut self, ui: &mut egui::Ui) {
        let (status_text, invocations): (String, Vec<InvocationRecord>) = {
            let st = self.state.lock().unwrap();
            let status_text = match st.connection_status {
                ConnectionStatus::Disconnected => "Disconnected".to_string(),
                ConnectionStatus::Connecting => "Connecting...".to_string(),
                ConnectionStatus::Connected => "Connected".to_string(),
                ConnectionStatus::Reconnecting => "Reconnecting...".to_string(),
                ConnectionStatus::Error => "Error".to_string(),
            };
            let invocations: Vec<_> = st.recent_invocations.iter().cloned().collect();
            (status_text, invocations)
        };
        ui.label(format!("Connection: {}", status_text));
        ui.add_space(8.0);
        ui.label("Recent invocations:");
        egui::ScrollArea::vertical().show(ui, |ui| {
            for inv in invocations.iter().rev().take(20) {
                let ago = format_relative_time(inv.at);
                ui.label(format!(
                    "{}  {}  {}",
                    inv.tool,
                    if inv.success { "OK" } else { "ERR" },
                    ago
                ));
            }
        });
    }
}
