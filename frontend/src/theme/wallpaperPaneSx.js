import { alpha } from '@mui/material/styles';

/** Matches legacy Documents TabbedContentManager dimming over wallpaper (one shared layer). */
export const MAIN_WORKSPACE_WALLPAPER_TINT_ALPHA = 0.65;

export function isUiWallpaperConfigActive(cfg) {
  return Boolean(cfg?.enabled && cfg?.source !== 'none');
}

/** Apply behind all main routes when wallpaper is on so brightness matches Documents. */
export function mainWorkspaceWallpaperTintBg(theme, wallpaperConfigActive) {
  if (!wallpaperConfigActive) return undefined;
  return alpha(theme.palette.background.default, MAIN_WORKSPACE_WALLPAPER_TINT_ALPHA);
}

/**
 * Opaque surfaces for chrome and cards so UI wallpaper does not show through panes.
 * Theme palette.surface is defined in themeConfig (opaque hex).
 */
export function solidSurfaceBg(theme) {
  return theme.palette.surface?.main ?? theme.palette.background.default;
}

/**
 * Main editor/content area over wallpaper: frosted panel so gutters show wallpaper,
 * while form content stays readable. Pair with outer padding on the parent.
 */
export function frostedEditorPaneSx(theme) {
  const base = alpha(
    theme.palette.background.default,
    theme.palette.mode === 'dark' ? 0.86 : 0.9
  );
  return {
    bgcolor: base,
    backdropFilter: 'saturate(140%) blur(14px)',
    WebkitBackdropFilter: 'saturate(140%) blur(14px)',
    '@media (prefers-reduced-transparency: reduce)': {
      bgcolor: alpha(theme.palette.background.default, theme.palette.mode === 'dark' ? 0.96 : 0.97),
      backdropFilter: 'none',
      WebkitBackdropFilter: 'none',
    },
  };
}
