import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import { useCallback, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  useColorScheme,
  View,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { getDocumentContent } from '../../../src/api/documents';
import { isApiError } from '../../../src/api/client';
import { recordRecentDocument } from '../../../src/session/recentDocumentsStore';
import { getColors } from '../../../src/theme/colors';
import { parseFrontmatter } from '../../../src/utils/parseFrontmatter';

const DOC_SNIPPET_MAX = 12_000;

export default function DocumentDetailScreen() {
  const scheme = useColorScheme();
  const colors = useMemo(() => getColors(scheme === 'dark' ? 'dark' : 'light'), [scheme]);

  const router = useRouter();
  const { id: idParam, documentTitle: titleParam } = useLocalSearchParams<{
    id: string;
    documentTitle?: string;
  }>();
  const id =
    typeof idParam === 'string' ? idParam : Array.isArray(idParam) ? idParam[0] ?? '' : '';
  const [text, setText] = useState<string | null>(null);
  const [meta, setMeta] = useState<string>('');
  const [docFrontmatter, setDocFrontmatter] = useState<Record<string, unknown>>({});
  const [loading, setLoading] = useState(true);
  const documentTitle =
    typeof titleParam === 'string' && titleParam.trim() ? titleParam.trim() : 'Document';

  const fetchDoc = useCallback(async () => {
    if (!id || id === '[id]' || id.includes('..')) {
      setText(null);
      setDocFrontmatter({});
      setMeta('Invalid document link.');
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      const res = await getDocumentContent(id);
      if (res.requires_password || res.is_encrypted) {
        setText(null);
        setDocFrontmatter({});
        setMeta('This document is encrypted. Open it in the web app to unlock.');
        void recordRecentDocument(id, documentTitle);
      } else {
        const body = res.content ?? '';
        setText(body);
        setDocFrontmatter(parseFrontmatter(body));
        setMeta('');
        void recordRecentDocument(id, documentTitle);
      }
    } catch (e) {
      setText(null);
      setDocFrontmatter({});
      if (isApiError(e) && e.status === 404) {
        setMeta(
          'Document not found. It may have been deleted, or your session may need a refresh. Use the back arrow and open it again from the list.'
        );
      } else {
        setMeta(e instanceof Error ? e.message : 'Failed to load');
      }
    } finally {
      setLoading(false);
    }
  }, [id]);

  useFocusEffect(
    useCallback(() => {
      void fetchDoc();
    }, [fetchDoc])
  );

  function openBastionChatWithDocument() {
    if (!id || text == null) return;
    const snippet = text.slice(0, DOC_SNIPPET_MAX);
    router.push({
      pathname: '/chat',
      params: {
        docId: id,
        docTitle: documentTitle,
        docSnippet: snippet,
        docSession: String(Date.now()),
        docFrontmatter: JSON.stringify(docFrontmatter),
      },
    });
  }

  if (loading) {
    return (
      <View style={[styles.center, { backgroundColor: colors.background }]}>
        <ActivityIndicator size="large" color={colors.text} />
      </View>
    );
  }

  const showChatFab = text != null && !meta;

  return (
    <View style={[styles.flex, { backgroundColor: colors.background }]}>
      <ScrollView contentContainerStyle={styles.scroll}>
        {meta ? <Text style={[styles.warn, { color: colors.danger }]}>{meta}</Text> : null}
        {text != null ? (
          <Text style={[styles.body, { color: colors.text }]}>{text}</Text>
        ) : null}
      </ScrollView>
      {showChatFab ? (
        <Pressable
          style={[styles.fab, { backgroundColor: colors.chipBgActive }]}
          onPress={openBastionChatWithDocument}
          accessibilityRole="button"
          accessibilityLabel="Open Bastion Chat with this document"
        >
          <Ionicons name="chatbubbles-outline" size={26} color="#fff" />
        </Pressable>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  scroll: { padding: 16, paddingBottom: 96 },
  warn: { marginBottom: 12 },
  body: { fontSize: 14, lineHeight: 22, fontFamily: 'monospace' },
  fab: {
    position: 'absolute',
    right: 20,
    bottom: 28,
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 6,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
});
