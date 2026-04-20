/**
 * Built-in UI wallpapers (served from public/wallpapers/).
 * Keys must stay in sync with backend.models.ui_wallpaper_models.UI_WALLPAPER_BUILTIN_KEYS.
 */

export const UI_WALLPAPER_QUERY_KEY = 'userUiWallpaper';

/**
 * Absolute URL for files from /public (Vite copies to build root).
 * Honors import.meta.env.BASE_URL when the app is not served from "/".
 */
export function wallpaperPublicUrl(path) {
  if (!path || /^https?:\/\//i.test(path)) {
    return path;
  }
  const rel = path.startsWith('/') ? path.slice(1) : path;
  const base = import.meta.env.BASE_URL || '/';
  if (base === '/' || base === '') {
    return `/${rel}`;
  }
  const baseTrim = base.endsWith('/') ? base.slice(0, -1) : base;
  return `${baseTrim}/${rel}`;
}

export const UI_WALLPAPER_BUILTINS = [
  {
    key: 'honeycomb',
    label: 'Honeycomb',
    path: '/wallpapers/honeycomb.jpg',
    defaultRepeat: 'repeat',
    defaultSize: 'auto',
  },
  {
    key: 'green',
    label: 'Green',
    path: '/wallpapers/green.jpg',
    defaultRepeat: 'repeat',
    defaultSize: 'auto',
  },
  {
    key: 'lineoleum',
    label: 'Linoleum',
    path: '/wallpapers/lineoleum.jpg',
    defaultRepeat: 'repeat',
    defaultSize: 'auto',
  },
  {
    key: 'mono',
    label: 'Mono',
    path: '/wallpapers/mono.jpg',
    defaultRepeat: 'repeat',
    defaultSize: 'auto',
  },
  {
    key: 'wheat',
    label: 'Wheat',
    path: '/wallpapers/wheat.jpg',
    defaultRepeat: 'repeat',
    defaultSize: 'auto',
  },
];

export function getUiWallpaperBuiltinByKey(key) {
  if (!key) return null;
  return UI_WALLPAPER_BUILTINS.find((b) => b.key === key) || null;
}
