import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Box,
  Typography,
  Button,
  IconButton,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tooltip,
  Paper,
  Alert,
} from '@mui/material';
import { Fullscreen, FullscreenExit } from '@mui/icons-material';
import ePub from 'epubjs';
import ebooksService from '../services/ebooksService';
import {
  koreaderPartialMd5,
  KOREADER_PARTIAL_MD5_REF_HEX,
  buildKoreaderPartialMd5GoldenArrayBuffer,
} from '../utils/koreaderPartialMd5';
import { devLog } from '../utils/devConsole';
import { ebookCachePut, ebookCacheGet, ebookCacheEnforceQuota } from '../services/ebookIndexedDb';
import { loadLocalEbookPosition, saveLocalEbookPosition } from '../utils/ebookPositionStore';
import PdfBytesViewer from './PdfBytesViewer';

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

/** Preset body sizes for EPUB reader (px); persisted as `fontSize` in reader_prefs. */
const FONT_SIZE_PRESETS = [
  { label: 'X-Small', px: 12 },
  { label: 'Small', px: 14 },
  { label: 'Medium', px: 18 },
  { label: 'Large', px: 22 },
  { label: 'X-Large', px: 28 },
];

function snapFontSizeToPresetPx(px) {
  const n = Number(px);
  const base = Number.isFinite(n) ? n : DEFAULT_PREFS.fontSize;
  let bestPx = FONT_SIZE_PRESETS[2].px;
  let bestDist = Infinity;
  for (const { px: presetPx } of FONT_SIZE_PRESETS) {
    const d = Math.abs(presetPx - base);
    if (d < bestDist) {
      bestDist = d;
      bestPx = presetPx;
    }
  }
  return bestPx;
}

function normalizeReaderPrefs(raw) {
  const merged = { ...DEFAULT_PREFS, ...raw };
  merged.fontSize = snapFontSizeToPresetPx(merged.fontSize);
  return merged;
}

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

/** When cloud position is this far behind local, show non-blocking conflict UI (0..1 scale). */
const SYNC_POSITION_THRESHOLD = 0.05;

function normalizeRemoteProgressPct(remote) {
  if (!remote || typeof remote !== 'object') return 0;
  const p = remote.percentage;
  if (typeof p !== 'number' || !Number.isFinite(p)) return 0;
  let x = p;
  if (x > 1) x /= 100;
  return Math.max(0, Math.min(1, x));
}

function remoteHasMeaningfulPosition(remote, remotePctNorm) {
  if (!remote || typeof remote !== 'object') return false;
  const prog = remote.progress;
  if (typeof prog === 'string' && prog.startsWith('epubcfi(')) return true;
  return remotePctNorm > 0.0001;
}

/** No default focus ring on pointer use; keyboard users still get :focus-visible. */
const readerTapZoneFocusSx = {
  outline: 'none',
  boxShadow: 'none',
  border: 0,
  '&:focus': { outline: 'none', boxShadow: 'none' },
  '&:focus:not(:focus-visible)': { outline: 'none', boxShadow: 'none' },
  '&:active': { outline: 'none', boxShadow: 'none' },
  '&:focus-visible': {
    outline: '2px solid',
    outlineColor: 'primary.main',
    outlineOffset: 2,
  },
  '&::-moz-focus-inner': { border: 0 },
};

/** Best-effort OPF/DC creator for recently-opened list (epubjs package shape varies). */
function extractEpubAuthor(book) {
  try {
    const meta = book?.package?.metadata || book?.packaging?.metadata;
    if (!meta) return null;
    const c = meta.creator || meta['dc:creator'] || meta.author;
    if (Array.isArray(c)) {
      const parts = [];
      for (const x of c) {
        if (typeof x === 'string' && x.trim()) parts.push(x.trim());
        else if (x && typeof x === 'object') {
          const t = x['#text'] || x.name || x.value;
          if (t && String(t).trim()) parts.push(String(t).trim());
        }
      }
      return parts.length ? parts.join(', ') : null;
    }
    if (typeof c === 'string' && c.trim()) return c.trim();
    if (c && typeof c === 'object') {
      const t = c['#text'] || c.name || c.value;
      return t && String(t).trim() ? String(t).trim() : null;
    }
    return null;
  } catch (_) {
    return null;
  }
}

function inferEbookFormat(acquisitionUrl, explicit) {
  if (explicit === 'pdf' || explicit === 'epub') return explicit;
  try {
    const u = new URL(String(acquisitionUrl || ''));
    if (u.pathname.toLowerCase().endsWith('.pdf')) return 'pdf';
  } catch (_) {
    /* ignore */
  }
  const s = String(acquisitionUrl || '').toLowerCase();
  if (s.includes('.pdf')) return 'pdf';
  return 'epub';
}

export default function EbookReaderTab({
  catalogId,
  acquisitionUrl,
  title,
  digest: initialDigest,
  ebookFormat,
  ebookAuthor: initialAuthor,
}) {
  const format = useMemo(() => inferEbookFormat(acquisitionUrl, ebookFormat), [acquisitionUrl, ebookFormat]);
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
  const [pdfBytes, setPdfBytes] = useState(null);
  const [syncConflict, setSyncConflict] = useState(null);
  const localPersistReadyRef = useRef(false);

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

  const pushProgress = useCallback(async (percentage, cfi, docDigest) => {
    if (format === 'pdf') return;
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
  }, [digest, format]);

  const schedulePush = useCallback(
    (percentage, cfi, docDigest) => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        void (async () => {
          await pushProgress(percentage, cfi, docDigest);
          const digestUse = docDigest || digest;
          if (localPersistReadyRef.current && digestUse && format !== 'pdf') {
            const cfiStr = cfi && String(cfi).startsWith('epubcfi(') ? String(cfi) : null;
            saveLocalEbookPosition(digestUse, percentage, cfiStr);
          }
        })();
      }, 2500);
    },
    [pushProgress, digest, format]
  );

  const handleDismissSyncConflict = useCallback(() => {
    setSyncConflict(null);
  }, []);

  const handleJumpToRemoteSync = useCallback(async () => {
    const c = syncConflict;
    if (!c) return;
    const book = bookRef.current;
    const rendition = renditionRef.current;
    if (!book || !rendition) {
      setSyncConflict(null);
      return;
    }
    try {
      if (c.remoteCfi) {
        await rendition.display(c.remoteCfi);
      } else if (typeof c.remotePct === 'number' && c.remotePct > 0) {
        const cfi = book.locations?.cfiFromPercentage?.(c.remotePct);
        await rendition.display(cfi || undefined);
      }
    } catch (_) {
      /* ignore */
    }
    setSyncConflict(null);
  }, [syncConflict]);

  useEffect(() => {
    if (!syncConflict) return undefined;
    const t = window.setTimeout(() => setSyncConflict(null), 10000);
    return () => window.clearTimeout(t);
  }, [syncConflict]);

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
    setPdfBytes(null);

    (async () => {
      try {
        setReaderReady(false);
        setPageLocationLabel('—');
        setError('');
        setSyncConflict(null);
        localPersistReadyRef.current = false;
        const settings = await ebooksService.getSettings();
        if (!active) return;
        const mergedPrefs = normalizeReaderPrefs(settings?.reader_prefs || {});
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
        const pushRecent = async (authorText) => {
          const author = authorText && String(authorText).trim() ? String(authorText).trim() : undefined;
          const item = {
            digest: d,
            title: title || 'Book',
            catalog_id: catalogId,
            acquisition_url: acquisitionUrl,
            opened_at: new Date().toISOString(),
            acquisition_format: format === 'pdf' ? 'pdf' : 'epub',
            ...(author ? { author } : {}),
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
        };

        if (!active) return;

        if (format === 'pdf') {
          await pushRecent(initialAuthor);
          if (!active) return;
          setPdfBytes(rawBytes);
          setStatus('');
          return;
        }

        setStatus('Opening book…');
        const book = ePub();
        bookRef.current = book;
        await book.open(rawBytes);
        await book.ready;
        if (!active) return;

        const resolvedAuthor = (initialAuthor && String(initialAuthor).trim()) || extractEpubAuthor(book) || undefined;
        await pushRecent(resolvedAuthor);
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
        await book.locations.generate(1024);
        if (!active) return;

        if (kosyncConfiguredRef.current) {
          try {
            const remote = await ebooksService.getProgress(d);
            const remotePct = normalizeRemoteProgressPct(remote);
            const remoteProg = remote?.progress;
            const remoteCfi =
              typeof remoteProg === 'string' && remoteProg.startsWith('epubcfi(') ? remoteProg : null;
            const remoteDevice =
              typeof remote?.device === 'string' && remote.device.trim() ? remote.device.trim() : 'sync';
            const hasRemotePos = remoteHasMeaningfulPosition(remote, remotePct);
            const localRow = loadLocalEbookPosition(d);
            const localPct =
              localRow && typeof localRow.percentage === 'number' ? localRow.percentage : 0;
            const localCfi =
              localRow?.cfi && String(localRow.cfi).startsWith('epubcfi(') ? localRow.cfi : null;

            if (!hasRemotePos) {
              if (localPct > 0) {
                if (localCfi) {
                  await rendition.display(localCfi);
                } else {
                  const lc = book.locations.cfiFromPercentage(localPct);
                  await rendition.display(lc || undefined);
                }
              } else {
                await rendition.display();
              }
            } else if (localPct <= 0 || remotePct >= localPct - SYNC_POSITION_THRESHOLD) {
              if (remoteCfi) {
                await rendition.display(remoteCfi);
              } else if (remotePct > 0) {
                const cfi = book.locations.cfiFromPercentage(remotePct);
                await rendition.display(cfi || undefined);
              } else {
                await rendition.display();
              }
            } else {
              if (localCfi) {
                await rendition.display(localCfi);
              } else {
                const lc = book.locations.cfiFromPercentage(localPct);
                await rendition.display(lc || undefined);
              }
              if (active) {
                setSyncConflict({
                  remotePct,
                  remoteCfi,
                  remoteDevice,
                  localPct,
                });
              }
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
        window.setTimeout(() => {
          localPersistReadyRef.current = true;
        }, 600);
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
      localPersistReadyRef.current = false;
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
  }, [catalogId, acquisitionUrl, title, initialDigest, initialAuthor, schedulePush, format]);

  useEffect(() => {
    const r = renditionRef.current;
    if (r) applyReaderTheme(r, prefs);
  }, [prefs.fontSize, prefs.fontFamily, prefs.lineHeight, prefs.theme]);

  /** epubjs measures the host once; reflow when the reader pane size changes (e.g. chrome hidden). */
  useEffect(() => {
    if (format !== 'epub' || !readerReady) return;
    const el = hostRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const resizeRendition = () => {
      const r = renditionRef.current;
      if (!r) return;
      try {
        const w = el.clientWidth;
        const h = el.clientHeight;
        if (w > 0 && h > 0) r.resize(w, h);
      } catch (_) {
        /* ignore */
      }
    };
    resizeRendition();
    const ro = new ResizeObserver(() => {
      requestAnimationFrame(resizeRendition);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [format, readerReady]);

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
            overflow: 'visible',
            display: 'flex',
            flexDirection: 'column',
            gap: 2.5,
          }}
        >
          <Typography
            variant="subtitle1"
            component="h2"
            sx={{
              fontWeight: 600,
              wordBreak: 'break-word',
              overflowWrap: 'anywhere',
              lineHeight: 1.35,
              pr: 0.5,
              pb: 0.25,
            }}
          >
            {title}
          </Typography>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'center',
              gap: 1.25,
              flexWrap: 'nowrap',
              overflowX: 'auto',
              overflowY: 'visible',
              pt: 0.75,
              pb: 0.25,
              width: '100%',
              minHeight: 52,
            }}
          >
            {format === 'epub' ? (
              <>
                <FormControl size="small" sx={{ minWidth: 100, flexShrink: 0 }}>
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
                <FormControl size="small" sx={{ minWidth: 130, flexShrink: 0 }}>
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
                <FormControl size="small" sx={{ minWidth: 118, flexShrink: 0 }}>
                  <InputLabel id="ebook-reader-size-label">Size</InputLabel>
                  <Select
                    labelId="ebook-reader-size-label"
                    label="Size"
                    value={snapFontSizeToPresetPx(prefs.fontSize)}
                    onChange={(e) => {
                      const fontSize = Number(e.target.value);
                      setPrefs((p) => {
                        const next = { ...p, fontSize };
                        scheduleReaderPrefsPersist(next);
                        return next;
                      });
                    }}
                  >
                    {FONT_SIZE_PRESETS.map(({ label, px }) => (
                      <MenuItem key={px} value={px}>
                        {label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </>
            ) : (
              <Typography variant="body2" color="text.secondary" sx={{ flexShrink: 0 }}>
                PDF
              </Typography>
            )}
            <Box sx={{ flex: '1 1 8px', minWidth: 8 }} />
            <Tooltip title={fs ? 'Exit full screen' : 'Full screen'}>
              <IconButton
                size="small"
                onClick={toggleFs}
                sx={{ flexShrink: 0 }}
                aria-label={fs ? 'Exit full screen' : 'Full screen'}
              >
                {fs ? <FullscreenExit /> : <Fullscreen />}
              </IconButton>
            </Tooltip>
            <Button size="small" variant="outlined" onClick={() => setChromeOpen(false)} sx={{ flexShrink: 0 }}>
              Hide
            </Button>
          </Box>
        </Paper>
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
      {syncConflict && format === 'epub' && !error && (
        <Alert
          severity="info"
          onClose={handleDismissSyncConflict}
          sx={{ flexShrink: 0, mx: 1.5, mb: 0.5, position: 'relative', zIndex: 2 }}
          action={
            <Button color="inherit" size="small" onClick={() => void handleJumpToRemoteSync()}>
              Jump to {Math.round(Math.max(0, Math.min(1, syncConflict.remotePct)) * 100)}%
            </Button>
          }
        >
          Your sync position ({Math.round(Math.max(0, Math.min(1, syncConflict.remotePct)) * 100)}%
          {syncConflict.remoteDevice ? ` on ${syncConflict.remoteDevice}` : ''}) is earlier than where you were
          here ({Math.round(Math.max(0, Math.min(1, syncConflict.localPct)) * 100)}%).
        </Alert>
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
          {format === 'pdf' && pdfBytes ? (
            <PdfBytesViewer
              rawBytes={pdfBytes}
              onPageLabelChange={setPageLocationLabel}
              onReadyChange={setReaderReady}
            />
          ) : (
            <Box
              ref={hostRef}
              sx={{
                position: 'absolute',
                inset: 0,
                overflow: 'hidden',
              }}
            />
          )}
        {!chromeOpen && (
          <Tooltip title="Tap or click top center — show reader controls" placement="bottom">
            <Box
              component="button"
              type="button"
              aria-label="Show reader controls"
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => setChromeOpen(true)}
              sx={{
                ...readerTapZoneFocusSx,
                position: 'absolute',
                top: 0,
                left: '50%',
                transform: 'translateX(-50%)',
                width: { xs: 'min(72%, 280px)', sm: 'min(50%, 320px)' },
                height: { xs: 52, sm: 48 },
                zIndex: 6,
                p: 0,
                m: 0,
                cursor: 'pointer',
                background: 'transparent',
                WebkitTapHighlightColor: 'transparent',
                touchAction: 'manipulation',
                borderBottomLeftRadius: 8,
                borderBottomRightRadius: 8,
                '&:hover': {
                  backgroundImage: (theme) =>
                    theme.palette.mode === 'dark'
                      ? 'linear-gradient(180deg, rgba(255,255,255,0.08) 0%, transparent 100%)'
                      : 'linear-gradient(180deg, rgba(0,0,0,0.06) 0%, transparent 100%)',
                },
              }}
            />
          </Tooltip>
        )}
        {!error && format === 'epub' && (
          <>
            <Tooltip title="Tap or click left edge — previous page (←)" placement="right">
              <Box
                component="button"
                type="button"
                aria-label="Previous page"
                disabled={!readerReady}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => void renditionRef.current?.prev()}
                sx={{
                  ...readerTapZoneFocusSx,
                  position: 'absolute',
                  left: 0,
                  top: 0,
                  bottom: 0,
                  width: { xs: 'min(28%, 112px)', sm: 'min(22%, 140px)' },
                  minWidth: 44,
                  zIndex: 5,
                  p: 0,
                  m: 0,
                  cursor: readerReady ? 'pointer' : 'default',
                  background: 'transparent',
                  WebkitTapHighlightColor: 'transparent',
                  touchAction: 'manipulation',
                  '&:hover:not(:disabled)': {
                    backgroundImage: (theme) =>
                      theme.palette.mode === 'dark'
                        ? 'linear-gradient(90deg, rgba(255,255,255,0.07) 0%, transparent 85%)'
                        : 'linear-gradient(90deg, rgba(0,0,0,0.05) 0%, transparent 85%)',
                  },
                  '&:disabled': { opacity: 0, pointerEvents: 'none' },
                }}
              />
            </Tooltip>
            <Tooltip title="Tap or click right edge — next page (→)" placement="left">
              <Box
                component="button"
                type="button"
                aria-label="Next page"
                disabled={!readerReady}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => void renditionRef.current?.next()}
                sx={{
                  ...readerTapZoneFocusSx,
                  position: 'absolute',
                  right: 0,
                  top: 0,
                  bottom: 0,
                  width: { xs: 'min(28%, 112px)', sm: 'min(22%, 140px)' },
                  minWidth: 44,
                  zIndex: 5,
                  p: 0,
                  m: 0,
                  cursor: readerReady ? 'pointer' : 'default',
                  background: 'transparent',
                  WebkitTapHighlightColor: 'transparent',
                  touchAction: 'manipulation',
                  '&:hover:not(:disabled)': {
                    backgroundImage: (theme) =>
                      theme.palette.mode === 'dark'
                        ? 'linear-gradient(270deg, rgba(255,255,255,0.07) 0%, transparent 85%)'
                        : 'linear-gradient(270deg, rgba(0,0,0,0.05) 0%, transparent 85%)',
                  },
                  '&:disabled': { opacity: 0, pointerEvents: 'none' },
                }}
              />
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
                maxWidth: 'min(92%, 520px)',
                textAlign: 'center',
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
