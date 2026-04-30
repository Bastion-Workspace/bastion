import { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';
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
import { prepareReaderSession } from '../../../src/components/ebooks/readerSession';
import { getOrCreateKosyncDeviceId } from '../../../src/session/kosyncDeviceStore';
import { ebookCacheEnforceQuota, ebookCacheGet, ebookCachePut } from '../../../src/utils/ebookFileStore';
import { koreaderPartialMd5 } from '../../../src/utils/koreaderPartialMd5';

const DEFAULT_PREFS = {
  fontSize: 18,
  theme: 'light' as ReaderTheme,
};

function toStandaloneArrayBuffer(data: ArrayBuffer | null): ArrayBuffer | null {
  if (!data) return null;
  if (data.byteLength === 0) return data;
  return data.slice(0);
}

function normParam(v: string | string[] | undefined): string {
  if (Array.isArray(v)) return v[0] || '';
  return v || '';
}

export default function EbookReaderScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const params = useLocalSearchParams<{
    catalogId?: string;
    acquisitionUrl?: string;
    title?: string;
    digest?: string;
  }>();
  const catalogId = normParam(params.catalogId);
  const acquisitionUrl = normParam(params.acquisitionUrl);
  const title = normParam(params.title) || 'Book';
  const initialDigest = normParam(params.digest);

  const webRef = useRef<EpubReaderWebViewHandle>(null);
  const [status, setStatus] = useState('Loading…');
  const [error, setError] = useState('');
  const [sourceHtml, setSourceHtml] = useState<string | null>(null);
  const [chromeOpen, setChromeOpen] = useState(false);
  const [theme, setTheme] = useState<ReaderTheme>(DEFAULT_PREFS.theme);
  const [fontSize, setFontSize] = useState(DEFAULT_PREFS.fontSize);
  const [sliderPct, setSliderPct] = useState(0);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const readyRef = useRef(false);
  const digestRef = useRef('');
  const kosyncConfiguredRef = useRef(false);

  const pushProgress = useCallback(
    async (percentage: number, cfi: string, docDigest: string) => {
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
      setSourceHtml(null);
      readyRef.current = false;
      try {
        const settings = await getEbooksSettings();
        if (cancelled) return;
        const mergedPrefs = { ...DEFAULT_PREFS, ...(settings.reader_prefs as Record<string, unknown>) };
        if (typeof mergedPrefs.fontSize === 'number') setFontSize(mergedPrefs.fontSize);
        if (mergedPrefs.theme === 'light' || mergedPrefs.theme === 'sepia' || mergedPrefs.theme === 'dark') {
          setTheme(mergedPrefs.theme);
        }
        const ksOk = Boolean(settings.kosync?.configured);
        kosyncConfiguredRef.current = ksOk;

        let rawBytes: ArrayBuffer | null = null;
        if (initialDigest) {
          const cached = await ebookCacheGet(initialDigest);
          if (cached?.data) {
            rawBytes = toStandaloneArrayBuffer(cached.data);
          }
        }
        if (!rawBytes) {
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
        }
        if (cancelled) return;

        const d = koreaderPartialMd5(rawBytes);
        digestRef.current = d;
        setStatus('Saving…');
        await ebookCachePut(d, rawBytes, {
          title,
          catalog_id: catalogId,
          acquisition_url: acquisitionUrl,
        });
        await ebookCacheEnforceQuota(8);

        const recent = settings.recently_opened || [];
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
        await putEbooksSettings({ recently_opened: mergedRecent });

        if (cancelled) return;
        setStatus('Opening…');
        const { sourceHtml: html } = await prepareReaderSession(rawBytes);
        if (cancelled) return;
        setSourceHtml(html);
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
  }, [catalogId, acquisitionUrl, initialDigest, title]);

  const onWebMessage = useCallback(
    (msg: WebToNativeMessage) => {
      if (msg.type === 'ERROR') {
        setError(msg.message);
        return;
      }
      if (msg.type === 'RELOCATED') {
        setSliderPct(typeof msg.percentage === 'number' ? msg.percentage : 0);
        const doc = digestRef.current;
        if (doc) schedulePush(msg.percentage, msg.cfi, doc);
        return;
      }
      if (msg.type === 'READY') {
        readyRef.current = true;
        webRef.current?.sendCommand({ type: 'SET_THEME', theme, fontSize });
        void (async () => {
          const doc = digestRef.current;
          if (!doc || !kosyncConfiguredRef.current) return;
          try {
            const remote = await getKosyncProgress(doc);
            if (remote?.percentage != null && remote?.progress) {
              webRef.current?.sendCommand({ type: 'GOTO_CFI', cfi: String(remote.progress) });
            }
          } catch {
            // ignore
          }
        })();
      }
    },
    [schedulePush, theme, fontSize]
  );

  return (
    <View style={[styles.root, { backgroundColor: '#000' }]}>
      <StatusBar style="light" />
      {sourceHtml ? <EpubReaderWebView ref={webRef} sourceHtml={sourceHtml} onMessage={onWebMessage} /> : null}

      {sourceHtml && !error ? (
        <Pressable
          style={[styles.fab, { bottom: insets.bottom + 72, right: 16 }]}
          onPress={() => setChromeOpen((v) => !v)}
          accessibilityLabel="Reader controls"
        >
          <Ionicons name="menu" size={26} color="#fff" />
        </Pressable>
      ) : null}

      {status ? (
        <View style={[styles.overlayCenter, { paddingTop: insets.top }]}>
          <ActivityIndicator color="#fff" />
          <Text style={styles.overlayText}>{status}</Text>
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

      {chromeOpen && sourceHtml && !error ? (
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
                  if (readyRef.current) webRef.current?.sendCommand({ type: 'SET_FONT_SIZE', fontSize: n, theme });
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
                  if (readyRef.current) webRef.current?.sendCommand({ type: 'SET_FONT_SIZE', fontSize: n, theme });
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
                  if (readyRef.current) webRef.current?.sendCommand({ type: 'SET_THEME', theme: t, fontSize });
                }}
                style={styles.themeChip}
              >
                <Text style={[styles.themeChipText, theme === t && styles.themeChipOn]}>{t[0].toUpperCase()}</Text>
              </Pressable>
            ))}
          </View>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  fab: {
    position: 'absolute',
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(0,0,0,0.55)',
    alignItems: 'center',
    justifyContent: 'center',
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
});
