/**
 * Stable per-tab surface id for notification routing (X-Surface-Id, WebSocket surface_meta).
 */
const STORAGE_KEY = 'bastion_desktop_surface_id';

export function getOrCreateDesktopSurfaceId() {
  try {
    let s = sessionStorage.getItem(STORAGE_KEY);
    if (!s && typeof crypto !== 'undefined' && crypto.randomUUID) {
      s = crypto.randomUUID();
      sessionStorage.setItem(STORAGE_KEY, s);
    }
    return s || null;
  } catch {
    return null;
  }
}
