//! Screenshot capability: capture screen via xcap, PNG encode, base64.
//! Supports single monitor (by index or "primary") or "all" (stitched horizontally).

use super::{Capability, CapabilityResult};
use crate::config::AppConfig;
use async_trait::async_trait;
use image::codecs::png::PngEncoder;
use image::{GenericImage, ImageEncoder, RgbaImage};
use serde_json::json;
use serde_json::Value;

pub struct ScreenshotCapability;

#[async_trait]
impl Capability for ScreenshotCapability {
    fn name(&self) -> &str {
        "screenshot"
    }

    fn description(&self) -> &str {
        "Capture a screenshot of one or all monitors (args: monitor = 0|1|2|'primary'|'all')"
    }

    async fn execute(&self, args: Value, _config: &AppConfig) -> Result<CapabilityResult, String> {
        let monitors = xcap::Monitor::all().map_err(|e| e.to_string())?;
        if monitors.is_empty() {
            return Err("No monitor found".to_string());
        }

        let monitor_arg = args
            .get("monitor")
            .and_then(|v| v.as_str())
            .unwrap_or("primary");

        let (image, monitor_count) = if monitor_arg.eq_ignore_ascii_case("all") {
            let mut images: Vec<(u32, u32, Vec<u8>)> = Vec::new();
            for m in &monitors {
                let img = m.capture_image().map_err(|e| e.to_string())?;
                let (w, h) = (img.width(), img.height());
                let raw = img.as_raw().to_vec();
                images.push((w, h, raw));
            }
            let total_width: u32 = images.iter().map(|(w, _, _)| w).sum();
            let max_height: u32 = images.iter().map(|(_, h, _)| *h).max().unwrap_or(0);
            let mut stitched = RgbaImage::new(total_width, max_height);
            let mut x_offset: u32 = 0;
            for (w, h, raw) in &images {
                let part = image::RgbaImage::from_raw(*w, *h, raw.clone())
                    .ok_or("Invalid image dimensions")?;
                stitched.copy_from(&part, x_offset, 0).map_err(|e| e.to_string())?;
                x_offset += w;
            }
            (stitched, images.len() as u32)
        } else {
            let index = if monitor_arg.eq_ignore_ascii_case("primary") {
                0
            } else {
                monitor_arg.parse::<usize>().map_err(|_| {
                    "monitor must be 'primary', 'all', or a numeric index (0, 1, 2, ...)"
                })?
            };
            let monitor = monitors
                .get(index)
                .ok_or_else(|| format!("Monitor index {} out of range (have {})", index, monitors.len()))?;
            let img = monitor.capture_image().map_err(|e| e.to_string())?;
            let (w, h) = (img.width(), img.height());
            let raw = img.as_raw().to_vec();
            let rgba = image::RgbaImage::from_raw(w, h, raw).ok_or("Invalid image dimensions")?;
            (rgba, 1)
        };

        let (width, height) = (image.width(), image.height());
        let raw = image.as_raw();
        let mut png_bytes: Vec<u8> = Vec::new();
        let encoder = PngEncoder::new(&mut png_bytes);
        encoder
            .write_image(raw, width, height, image::ExtendedColorType::Rgba8)
            .map_err(|e| e.to_string())?;

        let b64 = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            &png_bytes,
        );

        let result = json!({
            "image_base64": b64,
            "width": width,
            "height": height,
            "format": "png",
            "monitor_count": monitor_count
        });

        Ok(CapabilityResult {
            formatted: format!(
                "Screenshot captured ({}x{}, {} monitor(s))",
                width, height, monitor_count
            ),
            result,
        })
    }
}
