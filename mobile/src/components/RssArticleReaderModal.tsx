import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Linking,
  Modal,
  Pressable,
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

type Props = {
  visible: boolean;
  article: RssArticle | null;
  onClose: () => void;
};

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

function wrapHtmlDocument(innerBody: string, palette: AppColors): string {
  const base = (getApiBaseUrl() || '').replace(/\/$/, '');
  const baseTag = base ? `<base href="${escapeHtmlText(base)}/"/>` : '';
  const { background, text, textSecondary, link, border, surfaceMuted } = palette;
  return `<!DOCTYPE html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>${baseTag}<style>
html,body{background-color:${background};margin:0;}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:16px;line-height:1.45;color:${text};padding:12px;}
img,video{max-width:100%;height:auto;}
pre{white-space:pre-wrap;font-family:inherit;overflow-x:auto;background-color:${surfaceMuted};border:1px solid ${border};padding:8px;border-radius:6px;}
a{color:${link};}
blockquote{margin:12px 0;padding:8px 12px;border-left:3px solid ${border};color:${textSecondary};}
code{font-size:0.95em;background-color:${surfaceMuted};padding:2px 5px;border-radius:4px;}
</style></head><body>${innerBody}</body></html>`;
}

function buildArticleHtml(a: RssArticle, palette: AppColors): string {
  const htmlRaw = (a.full_content_html || '').trim();
  if (htmlRaw) {
    return absolutizeMessageMediaRefs(wrapHtmlDocument(htmlRaw, palette));
  }
  const plain = (a.full_content || '').trim();
  if (plain) {
    return wrapHtmlDocument(`<pre>${escapeHtmlText(plain)}</pre>`, palette);
  }
  const desc = (a.description || '').trim();
  if (desc && looksLikeHtml(desc)) {
    return absolutizeMessageMediaRefs(wrapHtmlDocument(desc, palette));
  }
  if (desc) {
    return wrapHtmlDocument(`<p>${escapeHtmlText(desc)}</p>`, palette);
  }
  return '';
}

export function RssArticleReaderModal({ visible, article, onClose }: Props) {
  const insets = useSafeAreaInsets();
  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const colors = useMemo(() => getColors(scheme), [scheme]);
  const [detail, setDetail] = useState<RssArticle | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!visible || !article) {
      setDetail(null);
      setLoading(false);
      return;
    }
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
  const htmlDoc = useMemo(
    () => (display ? buildArticleHtml(display, colors) : ''),
    [display, colors]
  );

  const openOriginal = useCallback(async () => {
    const url = (display?.link || '').trim();
    if (!url) return;
    const ok = await Linking.canOpenURL(url);
    if (ok) await Linking.openURL(url);
  }, [display?.link]);

  if (!article) {
    return null;
  }

  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet" onRequestClose={onClose}>
      <View style={[styles.sheet, { backgroundColor: colors.background, paddingTop: Math.max(insets.top, 8) }]}>
        <View style={[styles.header, { borderBottomColor: colors.border }]}>
          <Pressable onPress={onClose} style={styles.headerBtn} accessibilityRole="button">
            <Text style={[styles.headerBtnText, { color: colors.link }]}>Close</Text>
          </Pressable>
          <Text style={[styles.headerTitle, { color: colors.text }]} numberOfLines={2}>
            {display?.title || 'Article'}
          </Text>
          <Pressable onPress={() => void openOriginal()} style={styles.headerBtn} accessibilityRole="link">
            <Text style={[styles.headerBtnText, { color: colors.link }]}>Browser</Text>
          </Pressable>
        </View>
        {display?.feed_name ? (
          <Text style={[styles.feedName, { color: colors.link }]} numberOfLines={1}>
            {display.feed_name}
          </Text>
        ) : null}
        {loading ? (
          <View style={styles.center}>
            <ActivityIndicator size="large" color={colors.text} />
          </View>
        ) : htmlDoc ? (
          <WebView
            style={[styles.web, { backgroundColor: colors.background }]}
            originWhitelist={['*']}
            source={{ html: htmlDoc }}
            javaScriptEnabled={false}
            domStorageEnabled={false}
            setSupportMultipleWindows={false}
          />
        ) : (
          <View style={styles.center}>
            <Text style={[styles.emptyText, { color: colors.textSecondary }]}>
              No article body is stored on the server yet.
            </Text>
            <Pressable onPress={() => void openOriginal()} style={styles.openLink}>
              <Text style={[styles.openLinkText, { color: colors.link }]}>Open original link</Text>
            </Pressable>
          </View>
        )}
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  sheet: { flex: 1 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 8,
    paddingBottom: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  headerBtn: { paddingVertical: 8, paddingHorizontal: 10 },
  headerBtnText: { fontWeight: '600', fontSize: 15 },
  headerTitle: { flex: 1, fontSize: 16, fontWeight: '700' },
  feedName: { fontSize: 13, fontWeight: '600', paddingHorizontal: 16, paddingBottom: 6 },
  web: { flex: 1 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
  emptyText: { fontSize: 15, textAlign: 'center', marginBottom: 16 },
  openLink: { paddingVertical: 12, paddingHorizontal: 20 },
  openLinkText: { fontWeight: '700', fontSize: 16 },
});
