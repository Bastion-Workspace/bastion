import { Ionicons } from '@expo/vector-icons';
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Linking,
  Modal,
  PanResponder,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  useColorScheme,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import WebView from 'react-native-webview';
import { getRssArticle, type RssArticle } from '../api/rss';
import { absolutizeMessageMediaRefs, getApiBaseUrl } from '../api/config';
import { getColors, type AppColors } from '../theme/colors';
import { useRssPrefs, type RssArticleFontFamily, type RssReaderTheme } from '../hooks/useRssPrefs';

type Props = {
  visible: boolean;
  article: RssArticle | null;
  onClose: () => void;
  hasPrev?: boolean;
  hasNext?: boolean;
  onPrevArticle?: () => void;
  onNextArticle?: () => void;
};

const RSS_FONT_STACKS: Record<RssArticleFontFamily, string> = {
  sans: "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif",
  serif: "Georgia,'Times New Roman',serif",
  mono: "'Courier New',Courier,monospace",
};

function rssArticlePalette(theme: RssReaderTheme, systemIsDark: boolean): AppColors {
  if (theme === 'auto') {
    return getColors(systemIsDark ? 'dark' : 'light');
  }
  if (theme === 'light') {
    return getColors('light');
  }
  if (theme === 'dark') {
    return getColors('dark');
  }
  return {
    background: '#f4ecd8',
    surface: '#efe6d4',
    surfaceMuted: '#ebe0c8',
    text: '#3e3020',
    textSecondary: '#6d5c4d',
    border: '#d4c4a8',
    link: '#6d4c1a',
    danger: '#a52',
    chipBg: '#e8dcc4',
    chipBgActive: '#6d4c1a',
    chipText: '#5c4a38',
    chipTextActive: '#fff',
  };
}

function looksLikeHtml(s: string): boolean {
  const t = s.trim().toLowerCase();
  return (
    t.startsWith('<') &&
    (t.includes('<p') ||
      t.includes('<div') ||
      t.includes('<br') ||
      t.includes('<img') ||
      t.includes('<article') ||
      t.includes('<span') ||
      t.includes('<html'))
  );
}

function escapeHtmlText(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function wrapHtmlDocument(
  innerBody: string,
  palette: AppColors,
  fontSizePx: number,
  fontFamilyCss: string
): string {
  const base = (getApiBaseUrl() || '').replace(/\/$/, '');
  const baseTag = base ? `<base href="${escapeHtmlText(base)}/"/>` : '';
  const { background, text, textSecondary, link, border, surfaceMuted } = palette;
  return `<!DOCTYPE html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>${baseTag}<style>
html,body{background-color:${background};margin:0;}
body{font-family:${fontFamilyCss};font-size:${fontSizePx}px;line-height:1.45;color:${text};padding:12px;}
img,video{max-width:100%;height:auto;}
pre{white-space:pre-wrap;font-family:inherit;overflow-x:auto;background-color:${surfaceMuted};border:1px solid ${border};padding:8px;border-radius:6px;}
a{color:${link};}
blockquote{margin:12px 0;padding:8px 12px;border-left:3px solid ${border};color:${textSecondary};}
code{font-size:0.95em;background-color:${surfaceMuted};padding:2px 5px;border-radius:6px;}
</style></head><body>${innerBody}</body></html>`;
}

function buildArticleHtml(
  a: RssArticle,
  palette: AppColors,
  fontSizePx: number,
  fontFamilyCss: string
): string {
  const wrap = (inner: string) => wrapHtmlDocument(inner, palette, fontSizePx, fontFamilyCss);
  const htmlRaw = (a.full_content_html || '').trim();
  if (htmlRaw) {
    return absolutizeMessageMediaRefs(wrap(htmlRaw));
  }
  const plain = (a.full_content || '').trim();
  if (plain) {
    return wrap(`<pre>${escapeHtmlText(plain)}</pre>`);
  }
  const desc = (a.description || '').trim();
  if (desc && looksLikeHtml(desc)) {
    return absolutizeMessageMediaRefs(wrap(desc));
  }
  if (desc) {
    return wrap(`<p>${escapeHtmlText(desc)}</p>`);
  }
  return '';
}

export function RssArticleReaderModal({
  visible,
  article,
  onClose,
  hasPrev = false,
  hasNext = false,
  onPrevArticle,
  onNextArticle,
}: Props) {
  const insets = useSafeAreaInsets();
  const systemIsDark = useColorScheme() === 'dark';
  const {
    articleFontSize: fontSize,
    setArticleFontSize,
    articleFontFamily: fontFamily,
    setArticleFontFamily,
    articleTheme: rssTheme,
    setArticleTheme,
  } = useRssPrefs();
  const [detail, setDetail] = useState<RssArticle | null>(null);
  const [loading, setLoading] = useState(false);
  const [chromeOpen, setChromeOpen] = useState(false);

  const colors = useMemo(() => rssArticlePalette(rssTheme, systemIsDark), [rssTheme, systemIsDark]);

  const chromeBackdropPan = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onPanResponderTerminationRequest: () => false,
        onPanResponderRelease: (_, gs) => {
          if (gs.dy < -40 || (Math.abs(gs.dx) < 12 && Math.abs(gs.dy) < 12)) {
            setChromeOpen(false);
          }
        },
      }),
    []
  );

  const chromePanelPan = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => false,
        onMoveShouldSetPanResponder: (_e, gs) =>
          gs.dy > 10 && gs.dy > Math.abs(gs.dx) * 0.55,
        onPanResponderTerminationRequest: () => false,
        onPanResponderRelease: (_e, gs) => {
          if (gs.dy > 44 && gs.dy > Math.abs(gs.dx) * 1.2) {
            setChromeOpen(false);
          }
        },
      }),
    []
  );

  useEffect(() => {
    if (!visible || !article) {
      setDetail(null);
      setLoading(false);
      setChromeOpen(false);
      return;
    }
    setChromeOpen(false);
    let cancelled = false;
    setLoading(true);
    setDetail(null);
    void (async () => {
      try {
        const full = await getRssArticle(article.article_id);
        if (!cancelled) {
          setDetail(full);
        }
      } catch {
        if (!cancelled) {
          setDetail(article);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [visible, article?.article_id]);

  const display = detail ?? article;
  const fontStack = RSS_FONT_STACKS[fontFamily];
  const htmlDoc = useMemo(
    () => (display ? buildArticleHtml(display, colors, fontSize, fontStack) : ''),
    [display, colors, fontSize, fontStack]
  );

  const openOriginal = useCallback(async () => {
    const url = (display?.link || '').trim();
    if (!url) return;
    const ok = await Linking.canOpenURL(url);
    if (ok) await Linking.openURL(url);
  }, [display?.link]);

  const themeOptions: { key: RssReaderTheme; label: string }[] = [
    { key: 'auto', label: 'Auto' },
    { key: 'light', label: 'Light' },
    { key: 'sepia', label: 'Sepia' },
    { key: 'dark', label: 'Dark' },
  ];

  const fontOptions: { key: RssArticleFontFamily; label: string }[] = [
    { key: 'sans', label: 'Sans' },
    { key: 'serif', label: 'Serif' },
    { key: 'mono', label: 'Mono' },
  ];

  if (!article) {
    return null;
  }

  const showArticleNav = Boolean(onPrevArticle || onNextArticle);

  const persistentHeader = (
    <View
      style={[
        styles.persistentHeader,
        {
          paddingTop: Math.max(insets.top, 8),
          backgroundColor: colors.surface,
          borderBottomColor: colors.border,
        },
      ]}
    >
      <Pressable onPress={onClose} style={styles.headerIconBtn} accessibilityRole="button" accessibilityLabel="Back to article list">
        <Ionicons name="chevron-back" size={26} color={colors.link} />
      </Pressable>
      <Text style={[styles.persistentTitle, { color: colors.text }]} numberOfLines={1} ellipsizeMode="tail">
        {display?.title || 'Article'}
      </Text>
      <View style={styles.headerActions}>
        <Pressable
          onPress={() => void openOriginal()}
          style={styles.headerIconBtn}
          accessibilityRole="link"
          accessibilityLabel="Open in browser"
        >
          <Ionicons name="globe-outline" size={24} color={colors.link} />
        </Pressable>
        <Pressable
          onPress={() => setChromeOpen(true)}
          style={styles.headerIconBtn}
          accessibilityRole="button"
          accessibilityLabel="Article display settings"
        >
          <Ionicons name="options-outline" size={24} color={colors.link} />
        </Pressable>
      </View>
    </View>
  );

  const chromePanel = (
    <View
      {...chromePanelPan.panHandlers}
      style={[
        styles.chromePanel,
        {
          paddingBottom: Math.max(insets.bottom, 12),
          backgroundColor: colors.surface,
          borderTopColor: colors.border,
        },
      ]}
    >
      {showArticleNav ? (
        <View style={[styles.navRow, { borderBottomColor: colors.border }]}>
          <Pressable
            onPress={() => hasPrev && onPrevArticle?.()}
            disabled={!hasPrev}
            style={[styles.navBtn, !hasPrev && styles.navBtnDisabled]}
            accessibilityRole="button"
            accessibilityLabel="Previous article"
          >
            <Ionicons name="chevron-back" size={28} color={hasPrev ? colors.link : colors.textSecondary} />
          </Pressable>
          <Text style={[styles.navHint, { color: colors.textSecondary }]} numberOfLines={1}>
            Article
          </Text>
          <Pressable
            onPress={() => hasNext && onNextArticle?.()}
            disabled={!hasNext}
            style={[styles.navBtn, !hasNext && styles.navBtnDisabled]}
            accessibilityRole="button"
            accessibilityLabel="Next article"
          >
            <Ionicons name="chevron-forward" size={28} color={hasNext ? colors.link : colors.textSecondary} />
          </Pressable>
        </View>
      ) : null}
      {display?.feed_name ? (
        <Text style={[styles.feedName, { color: colors.link }]} numberOfLines={1}>
          {display.feed_name}
        </Text>
      ) : null}
      <View
        style={[styles.settingsStrip, { borderBottomColor: colors.border, borderBottomWidth: StyleSheet.hairlineWidth }]}
      >
        <View style={styles.settingsRow}>
          <Pressable
            onPress={() => void setArticleFontSize(fontSize - 1)}
            style={styles.fsBtn}
            accessibilityRole="button"
            accessibilityLabel="Decrease article font size"
          >
            <Text style={[styles.fsBtnText, { color: colors.link }]}>A−</Text>
          </Pressable>
          <Text style={[styles.fsLabel, { color: colors.textSecondary }]}>{fontSize}px</Text>
          <Pressable
            onPress={() => void setArticleFontSize(fontSize + 1)}
            style={styles.fsBtn}
            accessibilityRole="button"
            accessibilityLabel="Increase article font size"
          >
            <Text style={[styles.fsBtnText, { color: colors.link }]}>A+</Text>
          </Pressable>
        </View>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.chipScroll}>
          {fontOptions.map(({ key, label }) => {
            const on = fontFamily === key;
            return (
              <Pressable
                key={key}
                onPress={() => void setArticleFontFamily(key)}
                style={[styles.chip, { backgroundColor: on ? colors.chipBgActive : colors.chipBg }]}
                accessibilityRole="button"
                accessibilityState={{ selected: on }}
              >
                <Text style={{ color: on ? colors.chipTextActive : colors.chipText, fontWeight: '600', fontSize: 13 }}>
                  {label}
                </Text>
              </Pressable>
            );
          })}
        </ScrollView>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.chipScroll}>
          {themeOptions.map(({ key, label }) => {
            const on = rssTheme === key;
            return (
              <Pressable
                key={key}
                onPress={() => void setArticleTheme(key)}
                style={[styles.chip, { backgroundColor: on ? colors.chipBgActive : colors.chipBg }]}
                accessibilityRole="button"
                accessibilityState={{ selected: on }}
              >
                <Text style={{ color: on ? colors.chipTextActive : colors.chipText, fontWeight: '600', fontSize: 13 }}>
                  {label}
                </Text>
              </Pressable>
            );
          })}
        </ScrollView>
      </View>
    </View>
  );

  return (
    <Modal visible={visible} animationType="slide" onRequestClose={onClose}>
      <View style={[styles.sheet, { backgroundColor: colors.background }]}>
        {loading ? (
          <>
            {persistentHeader}
            <View style={styles.center}>
              <ActivityIndicator size="large" color={colors.text} />
            </View>
          </>
        ) : htmlDoc ? (
          <>
            {persistentHeader}
            <View style={styles.readerBody}>
              <WebView
                style={[styles.web, { backgroundColor: colors.background }]}
                originWhitelist={['*']}
                source={{ html: htmlDoc }}
                javaScriptEnabled={false}
                domStorageEnabled={false}
                setSupportMultipleWindows={false}
              />
            </View>
            {chromeOpen ? (
              <>
                <View
                  style={[styles.chromeBackdrop, { backgroundColor: 'rgba(0,0,0,0.35)' }]}
                  {...chromeBackdropPan.panHandlers}
                  accessibilityLabel="Dismiss article settings"
                />
                <View style={styles.chromeWrap} pointerEvents="box-none">
                  {chromePanel}
                </View>
              </>
            ) : null}
          </>
        ) : (
          <>
            {persistentHeader}
            <View style={styles.emptyBlock}>
              <Text style={[styles.emptyText, { color: colors.textSecondary }]}>
                No article body is stored on the server yet.
              </Text>
              <Pressable onPress={() => void openOriginal()} style={styles.openLink}>
                <Text style={[styles.openLinkText, { color: colors.link }]}>Open original link</Text>
              </Pressable>
            </View>
          </>
        )}
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  sheet: { flex: 1 },
  persistentHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 4,
    paddingBottom: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    gap: 4,
  },
  headerIconBtn: { paddingVertical: 8, paddingHorizontal: 8 },
  headerActions: { flexDirection: 'row', alignItems: 'center' },
  persistentTitle: { flex: 1, fontSize: 16, fontWeight: '700', minWidth: 0, marginHorizontal: 4 },
  readerBody: { flex: 1 },
  web: { flex: 1 },
  chromeBackdrop: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 8,
  },
  chromeWrap: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 9,
    justifyContent: 'flex-end',
  },
  chromePanel: {
    borderTopLeftRadius: 12,
    borderTopRightRadius: 12,
    paddingHorizontal: 4,
    paddingTop: 12,
    borderTopWidth: StyleSheet.hairlineWidth,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.15,
    shadowRadius: 6,
    elevation: 12,
    maxHeight: '55%',
  },
  navRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 8,
    paddingBottom: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  navBtn: { padding: 8 },
  navBtnDisabled: { opacity: 0.35 },
  navHint: { fontSize: 13, fontWeight: '600', flex: 1, textAlign: 'center' },
  feedName: { fontSize: 13, fontWeight: '600', paddingHorizontal: 16, paddingBottom: 6, paddingTop: 8 },
  settingsStrip: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    gap: 8,
  },
  settingsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
  },
  fsBtn: { paddingVertical: 6, paddingHorizontal: 14 },
  fsBtnText: { fontWeight: '700', fontSize: 18 },
  fsLabel: { fontSize: 13, fontWeight: '600', minWidth: 44, textAlign: 'center' },
  chipScroll: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 2,
  },
  chip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    marginRight: 8,
  },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
  emptyBlock: { flex: 1, justifyContent: 'center', paddingHorizontal: 24, paddingBottom: 24 },
  emptyText: { fontSize: 15, textAlign: 'center', marginBottom: 16 },
  openLink: { paddingVertical: 12, paddingHorizontal: 20, alignSelf: 'center' },
  openLinkText: { fontWeight: '700', fontSize: 16 },
});
