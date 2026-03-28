//! List running processes via sysinfo, sorted by CPU or memory.

use super::{Capability, CapabilityResult};
use crate::config::AppConfig;
use async_trait::async_trait;
use serde_json::json;
use serde_json::Value;
use sysinfo::{ProcessesToUpdate, System};

pub struct ListProcessesCapability;

#[async_trait]
impl Capability for ListProcessesCapability {
    fn name(&self) -> &str {
        "list_processes"
    }

    fn description(&self) -> &str {
        "List running processes (args: sort_by? = 'cpu'|'memory', limit? = 50)"
    }

    async fn execute(&self, args: Value, _config: &AppConfig) -> Result<CapabilityResult, String> {
        let sort_by = args
            .get("sort_by")
            .and_then(|v| v.as_str())
            .unwrap_or("cpu");
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(50) as usize;

        let mut sys = System::new_all();
        sys.refresh_processes(ProcessesToUpdate::All, true);
        sys.refresh_cpu_all();

        let mut processes: Vec<serde_json::Value> = sys
            .processes()
            .iter()
            .map(|(pid, proc_)| {
                let cpu = proc_.cpu_usage();
                let memory_mb = proc_.memory() / (1024 * 1024);
                let name = proc_.name().to_string_lossy().into_owned();
                let status = format!("{:?}", proc_.status());
                json!({
                    "pid": pid.as_u32(),
                    "name": name,
                    "cpu_percent": cpu,
                    "memory_mb": memory_mb,
                    "status": status
                })
            })
            .collect();

        if sort_by.eq_ignore_ascii_case("memory") {
            processes.sort_by(|a, b| {
                let ma = a["memory_mb"].as_u64().unwrap_or(0);
                let mb = b["memory_mb"].as_u64().unwrap_or(0);
                mb.cmp(&ma)
            });
        } else {
            processes.sort_by(|a, b| {
                let ca: f32 = a["cpu_percent"].as_f64().unwrap_or(0.0) as f32;
                let cb: f32 = b["cpu_percent"].as_f64().unwrap_or(0.0) as f32;
                cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        processes.truncate(limit);

        let result = json!({
            "processes": processes,
            "count": processes.len()
        });

        Ok(CapabilityResult {
            formatted: format!("Listed {} processes (top by {})", processes.len(), sort_by),
            result,
        })
    }
}
