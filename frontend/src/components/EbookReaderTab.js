import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Box,
  Typography,
  Button,
  IconButton,
  Slider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tooltip,
  Paper,
} from '@mui/material';
import {
  ContentCopy,
  Fullscreen,
  FullscreenExit,
  MenuBook,
  ChevronLeft,
  ChevronRight,
} from '@mui/icons-material';
import ePub from 'epubjs';
import ebooksService from '../services/ebooksService';
import {
  koreaderPartialMd5,
  KOREADER_PARTIAL_MD5_REF_HEX,
  buildKoreaderPartialMd5GoldenArrayBuffer,
} from '../utils/koreaderPartialMd5';
import { useChatSidebar } from '../contexts/ChatSidebarContext';
import { devLog } from '../utils/devConsole';
import { ebookCachePut, ebookCacheGet, ebookCacheEnforceQuota } from '../services/ebookIndexedDb';

const DEVICE_KEY = 'bastion_kosync_device_id';

/** IndexedDB may return ArrayBuffer or a typed array view; epubjs / SparkMD5 need a plain ArrayBuffer. */
function toStandaloneArrayBuffer(data) {
  if (!data) return null;
  if (data instanceof ArrayBuffer) {
    return data.byteLength === 0 ? data : data.slice(0);
  }
  if (ArrayBuffer.isView(data)) {
    const u = new Uint8Array(data.byteLength);
    u.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    return u.buffer;
  }
  return null;
}

const DEFAULT_PREFS = {
  fontSize: 18,
  fontFamily: 'Georgia, serif',
  lineHeight: 1.45,
  theme: 'light',
  margin: 0.08,
};

function getOrCreateDeviceId() {
  try {
    let id = localStorage.getItem(DEVICE_KEY);
    if (!id) {
      id = `web-${typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`}`;
      localStorage.setItem(DEVICE_KEY, id);
    }
    return id;
  } catch (_) {
    return 'web-unknown';
  }
}

function applyReaderTheme(rendition, readerPrefs) {
  if (!rendition) return;
  const themeKey = readerPrefs.theme || 'light';
  const themes = {
    light: { body: { background: '#fff', color: '#111' } },
    sepia: { body: { background: '#f4ecd8', color: '#3e3020' } },
    dark: { body: { background: '#1e1e1e', color: '#e6e6e6' } },
  };
  rendition.themes.default(
    Object.assign({}, themes[themeKey] || themes.light, {
      p: {
        'font-family': readerPrefs.fontFamily,
        'font-size': `${readerPrefs.fontSize}px !important`,
        'line-height': `${readerPrefs.lineHeight} !important`,
      },
    })
  );
}

const READER_MAX_WIDTH_PX = 1180;

export default function EbookReaderTab({
  catalogId,
  acquisitionUrl,
  title,
  digest: initialDigest,
  documentsFileTreeCollapsed = false,
  documentsIsMobile = false,
}) {
  const hostRef = useRef(null);
  const renditionRef = useRef(null);
  const bookRef = useRef(null);
  const blobUrlRef = useRef(null);
  const [status, setStatus] = useState('Loading…');
  const [error, setError] = useState('');
  const [digest, setDigest] = useState(initialDigest || '');
  const [prefs, setPrefs] = useState(DEFAULT_PREFS);
  const [chromeOpen, setChromeOpen] = useState(true);
  const [fs, setFs] = useState(false);
  const debounceRef = useRef(null);
  const prefsSaveTimerRef = useRef(null);
  const kosyncConfiguredRef = useRef(false);
  const keyboardNavRef = useRef(null);
  const [pageLocationLabel, setPageLocationLabel] = useState('—');
  const [readerReady, setReaderReady] = useState(false);
  const { isCollapsed: chatSidebarCollapsed } = useChatSidebar();

  const overlayInsetsPx = useMemo(() => {
    const left = documentsFileTreeCollapsed ? (documentsIsMobile ? 52 : 44) : 10;
    const right = chatSidebarCollapsed ? 72 : 12;
    return { left, right };
  }, [documentsFileTreeCollapsed, documentsIsMobile, chatSidebarCollapsed]);

  useEffect(() => {
    try {
      const h = koreaderPartialMd5(buildKoreaderPartialMd5GoldenArrayBuffer());
      if (h !== KOREADER_PARTIAL_MD5_REF_HEX) {
        devLog('koreaderPartialMd5 self-check failed', h, 'expected', KOREADER_PARTIAL_MD5_REF_HEX);
      }
    } catch (e) {
      devLog('koreaderPartialMd5 self-check error', e);
    }
  }, []);

  const copyDigest = useCallback(async () => {
    if (!digest) return;
    try {
      await navigator.clipboard.writeText(digest);
    } catch (_) {}
  }, [digest]);

  const pushProgress = useCallback(async (percentage, cfi, docDigest) => {
    const d = docDigest || digest;
    if (!d || !kosyncConfiguredRef.current) return;
    try {
      await ebooksService.putProgress({
        document: d,
        progress: cfi || '',
        percentage,
        device: 'BastionWeb',
        device_id: getOrCreateDeviceId(),
      });
    } catch (_) {}
  }, [digest]);

  const schedulePush = useCallback(
    (percentage, cfi, docDigest) => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        void pushProgress(percentage, cfi, docDigest);
      }, 2500);
    },
    [pushProgress]
  );

  useEffect(() => {
    let active = true;
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }
    try {
      renditionRef.current?.destroy();
    } catch (_) {}
    renditionRef.current = null;
    try {
      bookRef.current?.destroy();
    } catch (_) {}
    bookRef.current = null;

    (async () => {
      try {
        setReaderReady(false);
        setPageLocationLabel('—');
        setError('');
        const settings = await ebooksService.getSettings();
        if (!active) return;
        const mergedPrefs = { ...DEFAULT_PREFS, ...(settings?.reader_prefs || {}) };
        setPrefs(mergedPrefs);
        kosyncConfiguredRef.current = Boolean(settings?.kosync?.configured);

        let rawBytes = null;
        if (initialDigest) {
          const cached = await ebookCacheGet(initialDigest);
          if (cached?.data) {
            rawBytes = toStandaloneArrayBuffer(cached.data);
            if (rawBytes) setDigest(initialDigest);
          }
        }
        if (!rawBytes) {
          setStatus('Downloading…');
          const downloaded = await ebooksService.fetchOpds({
            catalog_id: catalogId,
            url: acquisitionUrl,
            want: 'binary',
          });
          rawBytes = toStandaloneArrayBuffer(downloaded);
          if (!rawBytes || rawBytes.byteLength === 0) {
            throw new Error('Empty or invalid download');
          }
        }
        if (!active) return;

        const d = koreaderPartialMd5(rawBytes);
        setDigest(d);
        setStatus('Saving to cache…');
        await ebookCachePut(d, rawBytes, {
          title,
          catalog_id: catalogId,
          acquisition_url: acquisitionUrl,
        });
        await ebookCacheEnforceQuota(8);

        const recent = settings?.recently_opened || [];
        const item = {
          digest: d,
          title: title || 'Book',
          catalog_id: catalogId,
          acquisition_url: acquisitionUrl,
          opened_at: new Date().toISOString(),
        };
        const mergedRecent = [
          item,
          ...recent.filter(
            (r) =>
              r.digest !== d &&
              !(String(r.catalog_id) === String(catalogId) && r.acquisition_url === acquisitionUrl)
          ),
        ].slice(0, 40);
        await ebooksService.putSettings({ recently_opened: mergedRecent });

        if (!active) return;
        setStatus('Opening book…');
        const book = ePub();
        bookRef.current = book;
        await book.open(rawBytes);
        await book.ready;
        if (!active) return;

        await new Promise((r) => {
          requestAnimationFrame(() => requestAnimationFrame(r));
        });
        if (!active) return;
        if (!hostRef.current) {
          throw new Error('Reader layout is not ready. Try again or resize the panel.');
        }

        const rendition = book.renderTo(hostRef.current, {
          width: '100%',
          height: '100%',
          flow: 'paginated',
          manager: 'default',
          spread: 'auto',
          minSpreadWidth: 720,
        });
        renditionRef.current = rendition;
        applyReaderTheme(rendition, mergedPrefs);

        setStatus('Indexing pages…');
        await book.locations.generate(1600);
        if (!active) return;

        if (kosyncConfiguredRef.current) {
          try {
            const remote = await ebooksService.getProgress(d);
            if (remote?.percentage != null && remote?.progress) {
              await rendition.display(remote.progress);
            } else {
              await rendition.display();
            }
          } catch (_) {
            await rendition.display();
          }
        } else {
          await rendition.display();
        }

        const updatePageLabel = (loc) => {
          try {
            if (!loc || !loc.start) {
              setPageLocationLabel('—');
              return;
            }
            const pg = loc?.start?.displayed?.page;
            const tot = loc?.start?.displayed?.total;
            if (typeof pg === 'number' && typeof tot === 'number' && tot > 0) {
              setPageLocationLabel(`Page ${pg} of ${tot}`);
              return;
            }
            const cfi = loc?.start?.cfi;
            const pct = book.locations.percentageFromCfi(cfi);
            if (typeof pct === 'number') {
              setPageLocationLabel(`${Math.round(Math.max(0, Math.min(1, pct)) * 100)}%`);
            } else {
              setPageLocationLabel('—');
            }
          } catch (_) {
            setPageLocationLabel('—');
          }
        };

        rendition.on('relocated', (loc) => {
          try {
            updatePageLabel(loc);
            const cfi = loc?.start?.cfi;
            const pct = book.locations.percentageFromCfi(cfi);
            const p = typeof pct === 'number' ? pct : 0;
            schedulePush(p, cfi, d);
            if (p >= 0.99) {
              void ebookCacheEnforceQuota(8);
            }
          } catch (_) {}
        });

        try {
          let loc = rendition.currentLocation();
          if (loc && typeof loc.then === 'function') {
            loc = await loc;
          }
          updatePageLabel(loc);
        } catch (_) {
          setPageLocationLabel('—');
        }

        keyboardNavRef.current = (e) => {
          if (!renditionRef.current) return;
          const t = e.target;
          if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
          if (e.key === 'ArrowLeft') {
            e.preventDefault();
            void renditionRef.current.prev();
          } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            void renditionRef.current.next();
          }
        };
        window.addEventListener('keydown', keyboardNavRef.current);

        setReaderReady(true);
        setStatus('');
      } catch (e) {
        if (active) {
          try {
            bookRef.current?.destroy();
          } catch (_) {}
          bookRef.current = null;
          try {
            renditionRef.current?.destroy();
          } catch (_) {}
          renditionRef.current = null;
          setReaderReady(false);
          setError(e?.message || String(e));
          setStatus('');
        }
      }
    })();
    return () => {
      active = false;
      setReaderReady(false);
      if (keyboardNavRef.current) {
        window.removeEventListener('keydown', keyboardNavRef.current);
        keyboardNavRef.current = null;
      }
      if (debounceRef.current) clearTimeout(debounceRef.current);
      try {
        renditionRef.current?.destroy();
      } catch (_) {}
      renditionRef.current = null;
      try {
        bookRef.current?.destroy();
      } catch (_) {}
      bookRef.current = null;
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, [catalogId, acquisitionUrl, title, initialDigest, schedulePush]);

  useEffect(() => {
    const r = renditionRef.current;
    if (r) applyReaderTheme(r, prefs);
  }, [prefs.fontSize, prefs.fontFamily, prefs.lineHeight, prefs.theme]);

  const scheduleReaderPrefsPersist = useCallback((readerPrefsSnapshot) => {
    if (prefsSaveTimerRef.current) clearTimeout(prefsSaveTimerRef.current);
    prefsSaveTimerRef.current = setTimeout(async () => {
      prefsSaveTimerRef.current = null;
      try {
        const s = await ebooksService.getSettings();
        await ebooksService.putSettings({
          reader_prefs: { ...(s?.reader_prefs || {}), ...readerPrefsSnapshot },
        });
      } catch (_) {}
    }, 450);
  }, []);

  useEffect(
    () => () => {
      if (prefsSaveTimerRef.current) clearTimeout(prefsSaveTimerRef.current);
    },
    []
  );

  const toggleFs = useCallback(() => {
    const el = hostRef.current?.parentElement;
    if (!el) return;
    if (!document.fullscreenElement) {
      el.requestFullscreen?.().then(() => setFs(true)).catch(() => {});
    } else {
      document.exitFullscreen?.().then(() => setFs(false)).catch(() => {});
    }
  }, []);

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        bgcolor: 'background.paper',
        position: 'relative',
      }}
    >
      {chromeOpen && (
        <Paper
          square
          elevation={1}
          sx={{
            position: 'relative',
            zIndex: 2,
            flexShrink: 0,
            px: 1.5,
            py: 1.25,
            pt: 2,
            display: 'flex',
            alignItems: 'flex-end',
            gap: 1.25,
            flexWrap: 'wrap',
            rowGap: 1.5,
          }}
        >
          <MenuBook fontSize="small" color="primary" sx={{ mb: 0.5 }} />
          <Box sx={{ flex: 1, minWidth: 140, mb: 0.75 }}>
            <Typography variant="subtitle2" noWrap>
              {title}
            </Typography>
            {digest ? (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.25 }}>
                <Tooltip
                  title={
                    'KoSync document id (partial MD5 of this EPUB’s bytes). Must match KOReader for the same file. ' +
                    `Regression check: 10KiB test pattern → ${KOREADER_PARTIAL_MD5_REF_HEX}.`
                  }
                >
                  <Typography variant="caption" color="text.secondary" sx={{ fontFamily: 'monospace', fontSize: 10 }}>
                    {digest}
                  </Typography>
                </Tooltip>
                <Tooltip title="Copy sync id">
                  <IconButton size="small" aria-label="Copy sync id" onClick={() => void copyDigest()} sx={{ p: 0.25 }}>
                    <ContentCopy sx={{ fontSize: 14 }} />
                  </IconButton>
                </Tooltip>
              </Box>
            ) : null}
          </Box>
          <FormControl size="small" sx={{ minWidth: 110 }}>
            <InputLabel id="ebook-reader-theme-label">Theme</InputLabel>
            <Select
              labelId="ebook-reader-theme-label"
              label="Theme"
              value={prefs.theme}
              onChange={(e) => {
                const theme = e.target.value;
                setPrefs((p) => {
                  const next = { ...p, theme };
                  scheduleReaderPrefsPersist(next);
                  return next;
                });
              }}
            >
              <MenuItem value="light">Light</MenuItem>
              <MenuItem value="sepia">Sepia</MenuItem>
              <MenuItem value="dark">Dark</MenuItem>
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel id="ebook-reader-font-label">Font</InputLabel>
            <Select
              labelId="ebook-reader-font-label"
              label="Font"
              value={prefs.fontFamily}
              onChange={(e) => {
                const fontFamily = e.target.value;
                setPrefs((p) => {
                  const next = { ...p, fontFamily };
                  scheduleReaderPrefsPersist(next);
                  return next;
                });
              }}
            >
              <MenuItem value="Georgia, serif">Georgia</MenuItem>
              <MenuItem value="'Literata', serif">Literata</MenuItem>
              <MenuItem value="system-ui, sans-serif">System UI</MenuItem>
              <MenuItem value="'Iowan Old Style', serif">Iowan</MenuItem>
            </Select>
          </FormControl>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0, mb: 0.25 }}>
            <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0 }}>
              Size
            </Typography>
            <Slider
              sx={{ width: 140, py: 0.5 }}
              size="small"
              min={12}
              max={28}
              value={prefs.fontSize}
              onChange={(_, v) => {
                setPrefs((p) => {
                  const next = { ...p, fontSize: v };
                  scheduleReaderPrefsPersist(next);
                  return next;
                });
              }}
            />
          </Box>
          <Tooltip title={fs ? 'Exit full screen' : 'Full screen'}>
            <IconButton size="small" onClick={toggleFs}>
              {fs ? <FullscreenExit /> : <Fullscreen />}
            </IconButton>
          </Tooltip>
          <Button size="small" onClick={() => setChromeOpen(false)}>
            Hide
          </Button>
        </Paper>
      )}
      {!chromeOpen && (
        <Box sx={{ position: 'absolute', top: 4, right: 4, zIndex: 3 }}>
          <Button size="small" variant="contained" onClick={() => setChromeOpen(true)}>
            Show controls
          </Button>
        </Box>
      )}
      {error && (
        <Box p={2}>
          <Typography color="error">{error}</Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
            Close this tab to leave the reader.
          </Typography>
        </Box>
      )}
      {status && !error && (
        <Box p={2} sx={{ flexShrink: 0, position: 'relative', zIndex: 2 }}>
          <Typography>{status}</Typography>
        </Box>
      )}
      <Box
        sx={{
          flex: 1,
          minHeight: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'stretch',
          width: '100%',
        }}
      >
        <Box
          sx={{
            flex: 1,
            minHeight: 0,
            width: '100%',
            maxWidth: { xs: '100%', md: `${READER_MAX_WIDTH_PX}px` },
            mx: 'auto',
            position: 'relative',
            zIndex: 0,
            isolation: 'isolate',
          }}
        >
          <Box
            ref={hostRef}
            sx={{
              position: 'absolute',
              inset: 0,
              overflow: 'hidden',
            }}
          />
        {!error && (
          <>
            <Tooltip title="Previous page (←)">
              <span>
                <IconButton
                  size="large"
                  aria-label="Previous page"
                  onClick={() => void renditionRef.current?.prev()}
                  disabled={!readerReady}
                  sx={{
                    position: 'absolute',
                    left: overlayInsetsPx.left,
                    top: '50%',
                    transform: 'translateY(-50%)',
                    zIndex: 5,
                    color: '#fff',
                    bgcolor: 'rgba(0,0,0,0.45)',
                    backdropFilter: 'blur(6px)',
                    '&:hover': { bgcolor: 'rgba(0,0,0,0.58)' },
                    '&.Mui-disabled': { color: 'rgba(255,255,255,0.45)', bgcolor: 'rgba(0,0,0,0.25)' },
                  }}
                >
                  <ChevronLeft fontSize="inherit" />
                </IconButton>
              </span>
            </Tooltip>
            <Tooltip title="Next page (→)">
              <span>
                <IconButton
                  size="large"
                  aria-label="Next page"
                  onClick={() => void renditionRef.current?.next()}
                  disabled={!readerReady}
                  sx={{
                    position: 'absolute',
                    right: overlayInsetsPx.right,
                    top: '50%',
                    transform: 'translateY(-50%)',
                    zIndex: 5,
                    color: '#fff',
                    bgcolor: 'rgba(0,0,0,0.45)',
                    backdropFilter: 'blur(6px)',
                    '&:hover': { bgcolor: 'rgba(0,0,0,0.58)' },
                    '&.Mui-disabled': { color: 'rgba(255,255,255,0.45)', bgcolor: 'rgba(0,0,0,0.25)' },
                  }}
                >
                  <ChevronRight fontSize="inherit" />
                </IconButton>
              </span>
            </Tooltip>
            <Typography
              variant="caption"
              sx={{
                position: 'absolute',
                left: '50%',
                bottom: 10,
                transform: 'translateX(-50%)',
                zIndex: 5,
                px: 1.25,
                py: 0.5,
                borderRadius: 1,
                color: '#fff',
                bgcolor: 'rgba(0,0,0,0.45)',
                backdropFilter: 'blur(6px)',
                pointerEvents: 'none',
                maxWidth: `calc(100% - ${overlayInsetsPx.left + overlayInsetsPx.right + 24}px)`,
              }}
            >
              {pageLocationLabel}
            </Typography>
          </>
        )}
        </Box>
      </Box>
    </Box>
  );
}
