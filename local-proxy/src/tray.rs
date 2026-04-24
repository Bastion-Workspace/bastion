//! System tray icon and right-click context menu.

use crate::shared_state::{ConnectionStatus, SharedState};
use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuItem};
use tray_icon::{Icon, TrayIconBuilder};
use muda::accelerator::Accelerator;
use muda::PredefinedMenuItem;

pub fn create_tray_icon(
    state: SharedState,
) -> Result<TrayMenuState, Box<dyn std::error::Error + Send + Sync>> {
    let menu = Menu::new();

    let status_item = MenuItem::with_id("status", "Disconnected", false, None::<Accelerator>);
    menu.append(&status_item)?;

    menu.append(&PredefinedMenuItem::separator())?;

    let connect_item = CheckMenuItem::with_id("connect", "Connect", true, false, None::<Accelerator>);
    menu.append(&connect_item)?;

    menu.append(&PredefinedMenuItem::separator())?;

    let screenshot_item = {
        #[cfg(feature = "native-screenshot")]
        {
            let screenshot_item =
                CheckMenuItem::with_id("screenshot", "Screenshot", true, false, None::<Accelerator>);
            menu.append(&screenshot_item)?;
            menu.append(&PredefinedMenuItem::separator())?;
            Some(screenshot_item)
        }
        #[cfg(not(feature = "native-screenshot"))]
        {
            None
        }
    };

    let settings_item = MenuItem::with_id("settings", "Settings...", true, None::<Accelerator>);
    menu.append(&settings_item)?;

    let quit_item = MenuItem::with_id("quit", "Quit", true, None::<Accelerator>);
    menu.append(&quit_item)?;

    let icon = make_icon(ConnectionStatus::Disconnected);
    let tray_icon = TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip("Bastion Local Proxy")
        .with_icon(icon)
        .build()?;

    Ok(TrayMenuState {
        state,
        tray_icon,
        last_status: ConnectionStatus::Disconnected,
        status_item,
        connect_item,
        screenshot_item,
    })
}

fn make_icon(status: ConnectionStatus) -> Icon {
    let (r, g, b) = match status {
        ConnectionStatus::Connected => (34u8, 197, 94),
        ConnectionStatus::Disconnected | ConnectionStatus::Error => (239, 68, 68),
        ConnectionStatus::Connecting | ConnectionStatus::Reconnecting => (234, 179, 8),
    };
    let (w, h) = (32, 32);
    let cx = 15.5f32;
    let cy = 15.5f32;
    let radius_sq = 12.0 * 12.0;
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist_sq = dx * dx + dy * dy;
            let a = if dist_sq <= radius_sq { 255u8 } else { 0u8 };
            rgba.extend_from_slice(&[r, g, b, a]);
        }
    }
    Icon::from_rgba(rgba, w, h).expect("tray icon from rgba")
}

pub struct TrayMenuState {
    pub state: SharedState,
    tray_icon: tray_icon::TrayIcon,
    last_status: ConnectionStatus,
    pub status_item: MenuItem,
    pub connect_item: CheckMenuItem,
    /// Present when built with `native-screenshot` (xcap / PipeWire).
    pub screenshot_item: Option<CheckMenuItem>,
}

impl TrayMenuState {
    pub fn update_menu_labels(&mut self) {
        let current_status = {
            let st = self.state.lock().unwrap();
            st.connection_status
        };
        if current_status != self.last_status {
            self.last_status = current_status;
            let _ = self.tray_icon.set_icon(Some(make_icon(current_status)));
        }
        let st = self.state.lock().unwrap();
        let status_text = match st.connection_status {
            ConnectionStatus::Disconnected => "Disconnected",
            ConnectionStatus::Connecting => "Connecting...",
            ConnectionStatus::Connected => {
                if st.config.bastion_url.is_empty() {
                    "Connected"
                } else {
                    st.config.bastion_url.as_str()
                }
            }
            ConnectionStatus::Reconnecting => "Reconnecting...",
            ConnectionStatus::Error => "Error",
        };
        self.status_item.set_text(status_text);
        self.connect_item.set_text(if st.connection_status == ConnectionStatus::Connected {
            "Disconnect"
        } else {
            "Connect (auto-reconnect)"
        });
        self.connect_item.set_checked(st.want_connected);
        if let Some(ref screenshot_item) = self.screenshot_item {
            let screenshot_enabled = st
                .config
                .capabilities
                .get("screenshot")
                .map(|c| c.enabled)
                .unwrap_or(false);
            screenshot_item.set_checked(screenshot_enabled);
        }
    }

    pub fn try_recv(&self) -> Option<TrayMenuAction> {
        let event = MenuEvent::receiver().try_recv().ok()?;
        let id = event.id.as_ref();
        Some(match id {
            "connect" => TrayMenuAction::Disconnect,
            #[cfg(feature = "native-screenshot")]
            "screenshot" => TrayMenuAction::ToggleScreenshot,
            "settings" => TrayMenuAction::OpenSettings,
            "quit" => TrayMenuAction::Quit,
            _ => return None,
        })
    }
}

pub enum TrayMenuAction {
    Disconnect,
    #[cfg(feature = "native-screenshot")]
    ToggleScreenshot,
    OpenSettings,
    Quit,
}
