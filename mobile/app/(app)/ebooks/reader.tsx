import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  PanResponder,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import {
  fetchOpdsBinary,
  getEbooksSettings,
  getKosyncProgress,
  putEbooksSettings,
  putKosyncProgress,
} from '../../../src/api/ebooks';
import {
  EpubReaderWebView,
  type EpubReaderWebViewHandle,
  type ReaderTheme,
  type WebToNativeMessage,
} from '../../../src/components/ebooks/EpubReaderWebView';
import { buildReaderHtml } from '../../../src/components/ebooks/readerHtml';
import { buildPdfReaderHtml } from '../../../src/components/ebooks/readerHtmlPdf';
import { getOrCreateKosyncDeviceId } from '../../../src/session/kosyncDeviceStore';
import {
  arrayBufferToBase64,
  ebookCacheEnforceQuota,
  ebookCacheExists,
  ebookCacheGet,
  ebookCachePut,
} from '../../../src/utils/ebookFileStore';
import { koreaderPartialMd5 } from '../../../src/utils/koreaderPartialMd5';
import {
  clearLastEbookParams,
  saveLastEbookParams,
} from '../../../src/session/lastEbookParamsStore';

const DEFAULT_PREFS = {
  fontSize: 18,
  theme: 'light' as ReaderTheme,
  /** Stack matches web reader where possible; missing faces fall back per platform. */
  fontFamily: 'Georgia, serif',
};

const READER_FONT_OPTIONS: { label: string; value: string }[] = [
  { label: 'Georgia', value: 'Georgia, serif' },
  { label: 'Literata', value: "'Literata', serif" },
  { label: 'Iowan', value: "'Iowan Old Style', serif" },
  { label: 'System', value: 'system-ui, -apple-system, sans-serif' },
  { label: 'Serif', value: 'ui-serif, Georgia, serif' },
  { label: 'Sans', value: 'ui-sans-serif, system-ui, sans-serif' },
];

function toStandaloneArrayBuffer(data: ArrayBuffer | null): ArrayBuffer | null {
  if (!data) return null;
  if (data.byteLength === 0) return data;
  return data.slice(0);
}

function normParam(v: string | string[] | undefined): string {
  if (Array.isArray(v)) return v[0] || '';
  return v || '';
}

function inferReaderFormat(formatParam: string, acquisitionUrl: string): 'epub' | 'pdf' {
  if (formatParam === 'pdf') return 'pdf';
  if (formatParam === 'epub') return 'epub';
  try {
    const u = new URL(acquisitionUrl);
    if (u.pathname.toLowerCase().endsWith('.pdf')) return 'pdf';
  } catch {
    /* ignore */
  }
  if (acquisitionUrl.toLowerCase().includes('.pdf')) return 'pdf';
  return 'epub';
}

export default function EbookReaderScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const params = useLocalSearchParams<{
    catalogId?: string;
    acquisitionUrl?: string;
    title?: string;
    digest?: string;
    format?: string;
  }>();
  const catalogId = normParam(params.catalogId);
  const acquisitionUrl = normParam(params.acquisitionUrl);
  const title = normParam(params.title) || 'Book';
  const initialDigest = normParam(params.digest);
  const formatParam = normParam(params.format);
  const resolvedFormat = useMemo(
    () => inferReaderFormat(formatParam, acquisitionUrl),
    [formatParam, acquisitionUrl]
  );
  const readerFormatRef = useRef<'epub' | 'pdf'>('epub');

  const webRef = useRef<EpubReaderWebViewHandle>(null);
  const [status, setStatus] = useState('Loading…');
  const [error, setError] = useState('');
  const [readerHtml, setReaderHtml] = useState<string | null>(null);
  /** Hide WebView until KoSync position applied (avoids flashing first page). */
  const [contentPositioned, setContentPositioned] = useState(false);
  const [chromeOpen, setChromeOpen] = useState(false);
  const [theme, setTheme] = useState<ReaderTheme>(DEFAULT_PREFS.theme);
  const [fontSize, setFontSize] = useState(DEFAULT_PREFS.fontSize);
  const [fontFamily, setFontFamily] = useState(DEFAULT_PREFS.fontFamily);
  const [sliderPct, setSliderPct] = useState(0);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prefsSaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const readyRef = useRef(false);
  const digestRef = useRef('');
  const kosyncConfiguredRef = useRef(false);

  const gestureThresholds = useMemo(
    () => ({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: (_: unknown, gs: { dx: number; dy: number }) =>
        Math.abs(gs.dx) > 6 || Math.abs(gs.dy) > 6,
      onPanResponderTerminationRequest: () => false,
    }),
    []
  );

  const leftZonePan = useMemo(
    () =>
      PanResponder.create({
        ...gestureThresholds,
        onPanResponderRelease: (_, gs) => {
          if (Math.abs(gs.dx) < 12 && Math.abs(gs.dy) < 12) {
            webRef.current?.sendCommand({ type: 'PREV' });
            return;
          }
          if (gs.dy > 40 && gs.dy > Math.abs(gs.dx) * 1.8) {
            setChromeOpen(true);
            return;
          }
          if (gs.dx < -45 && Math.abs(gs.dx) > Math.abs(gs.dy) * 1.5) {
            webRef.current?.sendCommand({ type: 'NEXT' });
          }
        },
      }),
    [gestureThresholds]
  );

  const middleZonePan = useMemo(
    () =>
      PanResponder.create({
        onMoveShouldSetPanResponder: (_e, gs) => gs.dy > 10 && gs.dy > Math.abs(gs.dx) * 0.55,
        onStartShouldSetPanResponder: () => false,
        onPanResponderTerminationRequest: () => false,
        onPanResponderRelease: (_e, gs) => {
          if (gs.dy > 40 && gs.dy > Math.abs(gs.dx) * 1.8) {
            setChromeOpen(true);
          }
        },
      }),
    []
  );

  const rightZonePan = useMemo(
    () =>
      PanResponder.create({
        ...gestureThresholds,
        onPanResponderRelease: (_, gs) => {
          if (Math.abs(gs.dx) < 12 && Math.abs(gs.dy) < 12) {
            webRef.current?.sendCommand({ type: 'NEXT' });
            return;
          }
          if (gs.dy > 40 && gs.dy > Math.abs(gs.dx) * 1.8) {
            setChromeOpen(true);
            return;
          }
          if (gs.dx > 45 && Math.abs(gs.dx) > Math.abs(gs.dy) * 1.5) {
            webRef.current?.sendCommand({ type: 'PREV' });
          }
        },
      }),
    [gestureThresholds]
  );

  const chromeBackdropPan = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onPanResponderTerminationRequest: () => false,
        onPanResponderRelease: (_, gs) => {
          if (gs.dy < -40) {
            setChromeOpen(false);
            return;
          }
          if (Math.abs(gs.dx) < 12 && Math.abs(gs.dy) < 12) {
            setChromeOpen(false);
          }
        },
      }),
    []
  );

  const pushProgress = useCallback(
    async (percentage: number, cfi: string, docDigest: string) => {
      if (readerFormatRef.current === 'pdf') return;
      if (!docDigest || !kosyncConfiguredRef.current) return;
      try {
        const device_id = await getOrCreateKosyncDeviceId();
        await putKosyncProgress({
          document: docDigest,
          progress: cfi || '',
          percentage,
          device: 'BastionMobile',
          device_id,
        });
      } catch {
        // ignore
      }
    },
    []
  );

  const schedulePush = useCallback(
    (percentage: number, cfi: string, docDigest: string) => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        void pushProgress(percentage, cfi, docDigest);
      }, 2500);
    },
    [pushProgress]
  );

  const schedulePersistReaderPrefs = useCallback(
    (next: { fontSize: number; theme: ReaderTheme; fontFamily: string }) => {
      if (prefsSaveTimerRef.current) clearTimeout(prefsSaveTimerRef.current);
      prefsSaveTimerRef.current = setTimeout(() => {
        prefsSaveTimerRef.current = null;
        void (async () => {
          try {
            const s = await getEbooksSettings();
            const raw = s.reader_prefs;
            const prev =
              raw && typeof raw === 'object' && !Array.isArray(raw)
                ? { ...(raw as Record<string, unknown>) }
                : {};
            await putEbooksSettings({
              reader_prefs: {
                ...prev,
                fontSize: next.fontSize,
                theme: next.theme,
                fontFamily: next.fontFamily,
              },
            });
          } catch {
            // ignore network errors
          }
        })();
      }, 450);
    },
    []
  );

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!catalogId || !acquisitionUrl) {
        setError('Missing book parameters.');
        setStatus('');
        return;
      }
      setError('');
      setStatus('Loading…');
      setReaderHtml(null);
      setContentPositioned(false);
      readyRef.current = false;
      readerFormatRef.current = resolvedFormat;
      try {
        const settings = await getEbooksSettings();
        if (cancelled) return;
        const mergedPrefs = { ...DEFAULT_PREFS, ...(settings.reader_prefs as Record<string, unknown>) };
        if (typeof mergedPrefs.fontSize === 'number') setFontSize(mergedPrefs.fontSize);
        if (mergedPrefs.theme === 'light' || mergedPrefs.theme === 'sepia' || mergedPrefs.theme === 'dark') {
          setTheme(mergedPrefs.theme);
        }
        if (typeof mergedPrefs.fontFamily === 'string' && mergedPrefs.fontFamily.trim()) {
          setFontFamily(mergedPrefs.fontFamily.trim());
        }
        const ksOk = Boolean(settings.kosync?.configured);
        kosyncConfiguredRef.current = ksOk;

        let rawBytes: ArrayBuffer | null = null;
        let d: string;

        if (initialDigest && (await ebookCacheExists(initialDigest))) {
          d = initialDigest;
          digestRef.current = d;
          const cached = await ebookCacheGet(initialDigest);
          if (!cached?.data?.byteLength) {
            throw new Error('Cached book is missing or empty');
          }
          rawBytes = toStandaloneArrayBuffer(cached.data);
        } else {
          setStatus('Downloading…');
          const downloaded = await fetchOpdsBinary({
            catalog_id: catalogId,
            url: acquisitionUrl,
            want: 'binary',
          });
          rawBytes = toStandaloneArrayBuffer(downloaded);
          if (!rawBytes || rawBytes.byteLength === 0) {
            throw new Error('Empty or invalid download');
          }
          d = koreaderPartialMd5(rawBytes);
          digestRef.current = d;
          setStatus('Saving…');
          await ebookCachePut(d, rawBytes, {
            title,
            catalog_id: catalogId,
            acquisition_url: acquisitionUrl,
          });
        }
        if (cancelled) return;

        await ebookCacheEnforceQuota(8);

        const recent = settings.recently_opened || [];
        const item = {
          digest: d,
          title: title || 'Book',
          catalog_id: catalogId,
          acquisition_url: acquisitionUrl,
          opened_at: new Date().toISOString(),
          acquisition_format: (resolvedFormat === 'pdf' ? 'pdf' : 'epub') as 'epub' | 'pdf',
        };
        const mergedRecent = [
          item,
          ...recent.filter(
            (r) =>
              r.digest !== d &&
              !(String(r.catalog_id) === String(catalogId) && r.acquisition_url === acquisitionUrl)
          ),
        ].slice(0, 40);
        await putEbooksSettings({ recently_opened: mergedRecent });

        if (cancelled) return;
        setStatus('Opening…');
        if (!rawBytes || !rawBytes.byteLength) {
          throw new Error('No book data');
        }
        const html =
          resolvedFormat === 'pdf'
            ? buildPdfReaderHtml(arrayBufferToBase64(rawBytes))
            : buildReaderHtml(arrayBufferToBase64(rawBytes));
        if (cancelled) return;
        setReaderHtml(html);
        setStatus('');
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Failed to open book');
          setStatus('');
        }
      }
    })();
    return () => {
      cancelled = true;
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [catalogId, acquisitionUrl, initialDigest, title, resolvedFormat]);

  useEffect(() => {
    if (!catalogId || !acquisitionUrl) return;
    void saveLastEbookParams({
      catalogId,
      acquisitionUrl,
      title,
      digest: initialDigest || undefined,
      format: formatParam || undefined,
    });
  }, [catalogId, acquisitionUrl, title, initialDigest, formatParam]);

  useEffect(
    () => () => {
      void clearLastEbookParams();
    },
    []
  );

  useEffect(
    () => () => {
      if (prefsSaveTimerRef.current) clearTimeout(prefsSaveTimerRef.current);
    },
    []
  );

  const onWebMessage = useCallback(
    (msg: WebToNativeMessage) => {
      if (msg.type === 'OPEN_CHROME') {
        setChromeOpen(true);
        return;
      }
      if (msg.type === 'ERROR') {
        setError(msg.message);
        return;
      }
      if (msg.type === 'RELOCATED') {
        setSliderPct(typeof msg.percentage === 'number' ? msg.percentage : 0);
        const doc = digestRef.current;
        if (doc && readerFormatRef.current === 'epub') {
          schedulePush(msg.percentage, msg.cfi, doc);
        }
        return;
      }
      if (msg.type === 'READY') {
        readyRef.current = true;
        if (readerFormatRef.current === 'pdf') {
          setContentPositioned(true);
          return;
        }
        if (readerFormatRef.current === 'epub') {
          webRef.current?.sendCommand({ type: 'SET_THEME', theme, fontSize, fontFamily });
        }
        const doc = digestRef.current;
        if (!doc || !kosyncConfiguredRef.current) {
          setContentPositioned(true);
          return;
        }
        void (async () => {
          try {
            const remote = await getKosyncProgress(doc);
            const prog = remote?.progress;
            let didSeek = false;
            if (typeof prog === 'string' && prog.startsWith('epubcfi(')) {
              webRef.current?.sendCommand({ type: 'GOTO_CFI', cfi: prog });
              didSeek = true;
            } else if (typeof remote?.percentage === 'number' && remote.percentage > 0) {
              webRef.current?.sendCommand({ type: 'SEEK_PERCENT', pct: remote.percentage });
              didSeek = true;
            }
            if (didSeek) {
              setTimeout(() => setContentPositioned(true), 450);
            } else {
              setContentPositioned(true);
            }
          } catch {
            setContentPositioned(true);
          }
        })();
      }
    },
    [schedulePush, theme, fontSize, fontFamily]
  );

  return (
    <View style={[styles.root, { backgroundColor: '#000' }]}>
      <StatusBar style="light" />
      {readerHtml ? (
        <View
          style={[
            styles.readerWebWrap,
            { paddingTop: insets.top, opacity: contentPositioned ? 1 : 0 },
          ]}
        >
          <EpubReaderWebView ref={webRef} html={readerHtml} onMessage={onWebMessage} />
        </View>
      ) : null}

      {readerHtml && !error && !chromeOpen ? (
        <View style={styles.zoneRow} pointerEvents="box-none" accessibilityLabel="Reader tap zones">
          <View style={styles.zone} {...leftZonePan.panHandlers} accessibilityLabel="Previous page zone" />
          <View style={styles.zone} {...middleZonePan.panHandlers} accessibilityLabel="Open reader menu zone" />
          <View style={styles.zone} {...rightZonePan.panHandlers} accessibilityLabel="Next page zone" />
        </View>
      ) : null}

      {chromeOpen && readerHtml && !error ? (
        <View
          style={[styles.chromeBackdrop, styles.chromeBackdropDim]}
          {...chromeBackdropPan.panHandlers}
          accessibilityLabel="Dismiss reader menu"
        />
      ) : null}

      {status ? (
        <View style={[styles.overlayCenter, { paddingTop: insets.top }]}>
          <ActivityIndicator color="#fff" />
          <Text style={styles.overlayText}>{status}</Text>
        </View>
      ) : null}
      {readerHtml && !contentPositioned && !error && !status ? (
        <View style={[styles.overlayCenter, { paddingTop: insets.top }]}>
          <ActivityIndicator color="#fff" />
          <Text style={styles.overlayText}>Opening…</Text>
        </View>
      ) : null}
      {error ? (
        <View style={[styles.overlayCenter, { paddingTop: insets.top }]}>
          <Text style={styles.errText}>{error}</Text>
          <Pressable style={styles.backBtn} onPress={() => router.back()}>
            <Text style={styles.backBtnText}>Go back</Text>
          </Pressable>
        </View>
      ) : null}

      {chromeOpen && readerHtml && !error ? (
        <View style={[styles.chrome, { paddingTop: insets.top + 8, backgroundColor: 'rgba(0,0,0,0.75)' }]}>
          <View style={styles.chromeRow}>
            <Pressable onPress={() => router.back()} accessibilityLabel="Close reader">
              <Ionicons name="chevron-back" size={28} color="#fff" />
            </Pressable>
            <Text style={styles.chromeTitle} numberOfLines={1}>
              {title}
            </Text>
            <View style={{ width: 28 }} />
          </View>
          <View style={styles.chromeRow}>
            <Pressable onPress={() => webRef.current?.sendCommand({ type: 'PREV' })}>
              <Ionicons name="chevron-back-circle-outline" size={36} color="#fff" />
            </Pressable>
            <View style={styles.seekRow}>
              <Pressable
                onPress={() => {
                  const v = Math.max(0, sliderPct - 0.05);
                  setSliderPct(v);
                  webRef.current?.sendCommand({ type: 'SEEK_PERCENT', pct: v });
                }}
              >
                <Text style={styles.seekBtn}>-5%</Text>
              </Pressable>
              <Text style={styles.seekPct}>{Math.round(sliderPct * 100)}%</Text>
              <Pressable
                onPress={() => {
                  const v = Math.min(1, sliderPct + 0.05);
                  setSliderPct(v);
                  webRef.current?.sendCommand({ type: 'SEEK_PERCENT', pct: v });
                }}
              >
                <Text style={styles.seekBtn}>+5%</Text>
              </Pressable>
            </View>
            <Pressable onPress={() => webRef.current?.sendCommand({ type: 'NEXT' })}>
              <Ionicons name="chevron-forward-circle-outline" size={36} color="#fff" />
            </Pressable>
          </View>
          <View style={styles.chromeRow}>
            <Pressable
              onPress={() => {
                setFontSize((s) => {
                  const n = Math.max(12, s - 1);
                  if (readyRef.current) {
                    webRef.current?.sendCommand({ type: 'SET_FONT_SIZE', fontSize: n, theme, fontFamily });
                  }
                  schedulePersistReaderPrefs({ fontSize: n, theme, fontFamily });
                  return n;
                });
              }}
            >
              <Text style={styles.fsBtn}>A−</Text>
            </Pressable>
            <Text style={styles.fsLabel}>{fontSize}px</Text>
            <Pressable
              onPress={() => {
                setFontSize((s) => {
                  const n = Math.min(32, s + 1);
                  if (readyRef.current) {
                    webRef.current?.sendCommand({ type: 'SET_FONT_SIZE', fontSize: n, theme, fontFamily });
                  }
                  schedulePersistReaderPrefs({ fontSize: n, theme, fontFamily });
                  return n;
                });
              }}
            >
              <Text style={styles.fsBtn}>A+</Text>
            </Pressable>
            <View style={{ flex: 1 }} />
            {(['light', 'sepia', 'dark'] as ReaderTheme[]).map((t) => (
              <Pressable
                key={t}
                onPress={() => {
                  setTheme(t);
                  if (readyRef.current) {
                    webRef.current?.sendCommand({ type: 'SET_THEME', theme: t, fontSize, fontFamily });
                  }
                  schedulePersistReaderPrefs({ fontSize, theme: t, fontFamily });
                }}
                style={styles.themeChip}
              >
                <Text style={[styles.themeChipText, theme === t && styles.themeChipOn]}>{t[0].toUpperCase()}</Text>
              </Pressable>
            ))}
          </View>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.fontScrollContent}
            style={styles.fontScroll}
          >
            {READER_FONT_OPTIONS.map((opt) => (
              <Pressable
                key={opt.value}
                onPress={() => {
                  setFontFamily(opt.value);
                  if (readyRef.current) {
                    webRef.current?.sendCommand({ type: 'SET_THEME', theme, fontSize, fontFamily: opt.value });
                  }
                  schedulePersistReaderPrefs({ fontSize, theme, fontFamily: opt.value });
                }}
                style={[styles.fontChip, fontFamily === opt.value && styles.fontChipOn]}
              >
                <Text style={[styles.fontChipText, fontFamily === opt.value && styles.fontChipTextOn]}>
                  {opt.label}
                </Text>
              </Pressable>
            ))}
          </ScrollView>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  readerWebWrap: { flex: 1 },
  zoneRow: {
    ...StyleSheet.absoluteFillObject,
    flexDirection: 'row',
  },
  chromeBackdrop: {
    ...StyleSheet.absoluteFillObject,
  },
  zone: {
    flex: 1,
    backgroundColor: 'transparent',
  },
  chromeBackdropDim: {
    backgroundColor: 'rgba(0,0,0,0.35)',
  },
  overlayCenter: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
  },
  overlayText: { color: '#eee', marginTop: 12, fontSize: 15 },
  errText: { color: '#f88', textAlign: 'center', marginBottom: 16 },
  backBtn: { padding: 12 },
  backBtnText: { color: '#9cf', fontSize: 16, fontWeight: '600' },
  chrome: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    paddingHorizontal: 12,
    paddingBottom: 12,
  },
  chromeRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 8 },
  chromeTitle: { flex: 1, color: '#fff', fontSize: 16, fontWeight: '600', textAlign: 'center' },
  seekRow: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 12 },
  seekBtn: { color: '#fff', fontWeight: '700', fontSize: 14 },
  seekPct: { color: '#ccc', minWidth: 48, textAlign: 'center' },
  fsBtn: { color: '#fff', fontSize: 18, fontWeight: '700', paddingHorizontal: 8 },
  fsLabel: { color: '#ccc', minWidth: 48, textAlign: 'center' },
  themeChip: { paddingHorizontal: 8, paddingVertical: 4 },
  themeChipText: { color: '#888', fontWeight: '700' },
  themeChipOn: { color: '#fff' },
  fontScroll: { maxHeight: 40, marginBottom: 4 },
  fontScrollContent: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingRight: 8 },
  fontChip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    backgroundColor: 'rgba(255,255,255,0.12)',
  },
  fontChipOn: { backgroundColor: 'rgba(255,255,255,0.28)' },
  fontChipText: { color: '#aaa', fontSize: 13, fontWeight: '600' },
  fontChipTextOn: { color: '#fff' },
});
