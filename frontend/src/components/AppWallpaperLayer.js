import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Box, useTheme } from '@mui/material';
import useMediaQuery from '@mui/material/useMediaQuery';
import { useQuery } from 'react-query';
import { useAuth } from '../contexts/AuthContext';
import apiService from '../services/apiService';
import {
  getUiWallpaperBuiltinByKey,
  UI_WALLPAPER_QUERY_KEY,
  wallpaperPublicUrl,
} from '../config/uiWallpaperBuiltins';

function authHeader() {
  const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export default function AppWallpaperLayer() {
  const { isAuthenticated, loading: authLoading } = useAuth();
  const theme = useTheme();
  const prefersReducedTransparency = useMediaQuery('(prefers-reduced-transparency: reduce)', {
    noSsr: true,
  });

  const { data } = useQuery(
    [UI_WALLPAPER_QUERY_KEY],
    () => apiService.settings.getUserUiWallpaper(),
    {
      enabled: isAuthenticated && !authLoading,
      staleTime: 60_000,
      refetchOnWindowFocus: true,
    }
  );

  const cfg = data?.config;
  const [blobUrl, setBlobUrl] = useState(null);
  const blobUrlRef = useRef(null);

  useEffect(() => {
    if (!cfg?.enabled || cfg.source !== 'document' || !cfg.document_id) {
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
      setBlobUrl(null);
      return undefined;
    }

    const url = `/api/documents/${encodeURIComponent(cfg.document_id)}/file`;
    let cancelled = false;

    fetch(url, { headers: authHeader() })
      .then((res) => {
        if (!res.ok) throw new Error(String(res.status));
        return res.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        const objectUrl = URL.createObjectURL(blob);
        if (blobUrlRef.current) {
          URL.revokeObjectURL(blobUrlRef.current);
        }
        blobUrlRef.current = objectUrl;
        setBlobUrl(objectUrl);
      })
      .catch(() => {
        if (!cancelled) {
          if (blobUrlRef.current) {
            URL.revokeObjectURL(blobUrlRef.current);
            blobUrlRef.current = null;
          }
          setBlobUrl(null);
        }
      });

    return () => {
      cancelled = true;
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, [cfg?.enabled, cfg?.source, cfg?.document_id]);

  useEffect(
    () => () => {
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    },
    []
  );

  const showWallpaper = Boolean(cfg?.enabled && cfg.source !== 'none');

  const builtinReady =
    showWallpaper &&
    cfg?.source === 'builtin' &&
    Boolean(getUiWallpaperBuiltinByKey(cfg?.builtin_key || ''));
  const documentReady = showWallpaper && cfg?.source === 'document' && Boolean(blobUrl);

  useLayoutEffect(() => {
    if (typeof document === 'undefined' || !document.body) return undefined;
    if (builtinReady || documentReady) {
      document.body.setAttribute('data-ui-wallpaper', 'active');
      return () => document.body.removeAttribute('data-ui-wallpaper');
    }
    document.body.removeAttribute('data-ui-wallpaper');
    return undefined;
  }, [builtinReady, documentReady]);

  if (!cfg?.enabled || cfg.source === 'none') {
    return null;
  }

  const opacity = typeof cfg.opacity === 'number' ? cfg.opacity : 0.62;
  const scrimOpacity = typeof cfg.scrim_opacity === 'number' ? cfg.scrim_opacity : 0.22;
  const blurPx = typeof cfg.blur_px === 'number' ? cfg.blur_px : 0;
  const size = cfg.size === 'contain' || cfg.size === 'auto' ? cfg.size : 'cover';
  const repeat = cfg.repeat === 'repeat' ? 'repeat' : 'no-repeat';

  let imageUrl = null;
  if (cfg.source === 'builtin') {
    const b = getUiWallpaperBuiltinByKey(cfg.builtin_key);
    if (!b) return null;
    imageUrl = wallpaperPublicUrl(b.path);
  } else if (cfg.source === 'document') {
    if (!blobUrl) return null;
    imageUrl = blobUrl;
  } else {
    return null;
  }

  const isDark = theme.palette.mode === 'dark';
  const scrimScale = prefersReducedTransparency ? 0.55 : 1;
  const userScrim = Math.max(0, scrimOpacity) * scrimScale;
  // Map the same saved scrim slider to stronger dark overlays (readability on dark UI) and
  // lighter frosted scrims in light mode (tiles stay visible without washing out).
  const scrimMax = isDark ? 0.9 : 0.68;
  const scrimCurve = isDark ? 1.22 : 0.38;
  const tintStrength = Math.min(scrimMax, userScrim * scrimCurve);
  const tint = isDark ? `rgba(0, 0, 0, ${tintStrength})` : `rgba(255, 255, 255, ${tintStrength})`;
  const backgroundImage = `linear-gradient(${tint}, ${tint}), url("${imageUrl}")`;
  // Slightly reduce wallpaper layer opacity in dark mode so bright tiles do not read as "light theme"
  // behind dark surfaces; user master opacity still applies on top.
  const imageOpacityBase = Math.min(1, Math.max(0, opacity));
  const imageOpacity = isDark ? imageOpacityBase * 0.82 : imageOpacityBase;

  const layer = (
    <Box
      aria-hidden
      sx={{
        position: 'fixed',
        inset: 0,
        // Below Routes / MainContent (z-index 1+) but inside div.App stacking context
        zIndex: -1,
        pointerEvents: 'none',
        overflow: 'hidden',
        backgroundColor: 'transparent',
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          inset: 0,
          backgroundImage,
          backgroundPosition: 'center, center',
          backgroundSize: `${size}, ${size}`,
          backgroundRepeat: `${repeat}, ${repeat}`,
          opacity: imageOpacity,
          filter:
            !prefersReducedTransparency && blurPx > 0 ? `blur(${blurPx}px)` : 'none',
        }}
      />
    </Box>
  );

  return layer;
}
