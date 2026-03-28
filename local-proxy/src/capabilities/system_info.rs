//! System info capability: OS, CPU, memory, disks via sysinfo.

use super::{Capability, CapabilityResult};
use crate::config::AppConfig;
use async_trait::async_trait;
use serde_json::json;
use serde_json::Value;
use sysinfo::{Disk, Disks, System};

pub struct SystemInfoCapability;

#[async_trait]
impl Capability for SystemInfoCapability {
    fn name(&self) -> &str {
        "system_info"
    }

    fn description(&self) -> &str {
        "Get OS, hostname, CPU count, memory, and disk usage"
    }

    async fn execute(&self, _args: Value, _config: &AppConfig) -> Result<CapabilityResult, String> {
        let mut sys = System::new_all();
        sys.refresh_all();

        let os = System::name().unwrap_or_else(|| "Unknown".to_string());
        let os_version = System::os_version().unwrap_or_else(|| "Unknown".to_string());
        let hostname = System::host_name().unwrap_or_else(|| "Unknown".to_string());
        let cpu_count = sys.cpus().len() as u32;
        let cpu_model = sys
            .cpus()
            .first()
            .map(|c| c.brand().to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        let total_memory_mb = sys.total_memory() / (1024 * 1024);
        let used_memory_mb = sys.used_memory() / (1024 * 1024);
        let uptime_seconds = System::uptime();

        let disks_list = Disks::new_with_refreshed_list();
        let disks: Vec<serde_json::Value> = disks_list
            .list()
            .iter()
            .map(|d: &Disk| {
                let total_gb = d.total_space() / (1024 * 1024 * 1024);
                let free_gb = d.available_space() / (1024 * 1024 * 1024);
                let mount = d.mount_point().to_string_lossy().into_owned();
                json!({
                    "mount": mount,
                    "total_gb": total_gb,
                    "free_gb": free_gb
                })
            })
            .collect();

        let result = json!({
            "os": os,
            "os_version": os_version,
            "hostname": hostname,
            "cpu_count": cpu_count,
            "cpu_model": cpu_model,
            "total_memory_mb": total_memory_mb,
            "used_memory_mb": used_memory_mb,
            "uptime_seconds": uptime_seconds,
            "disks": disks
        });

        Ok(CapabilityResult {
            formatted: format!(
                "{} {} | {} CPUs | {} MB / {} MB RAM | {} disk(s)",
                os,
                os_version,
                cpu_count,
                used_memory_mb,
                total_memory_mb,
                disks.len()
            ),
            result,
        })
    }
}
