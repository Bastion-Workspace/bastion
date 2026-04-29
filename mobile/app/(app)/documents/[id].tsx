import { Ionicons } from '@expo/vector-icons';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { getDocumentContent } from '../../../src/api/documents';

const DOC_SNIPPET_MAX = 12_000;

export default function DocumentDetailScreen() {
  const router = useRouter();
  const { id, documentTitle: titleParam } = useLocalSearchParams<{
    id: string;
    documentTitle?: string;
  }>();
  const [text, setText] = useState<string | null>(null);
  const [meta, setMeta] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const documentTitle =
    typeof titleParam === 'string' && titleParam.trim() ? titleParam.trim() : 'Document';

  useEffect(() => {
    if (!id) return;
    void (async () => {
      try {
        const res = await getDocumentContent(id);
        if (res.requires_password || res.is_encrypted) {
          setText(null);
          setMeta('This document is encrypted. Open it in the web app to unlock.');
        } else {
          setText(res.content ?? '');
          setMeta('');
        }
      } catch (e) {
        setText(null);
        setMeta(e instanceof Error ? e.message : 'Failed to load');
      } finally {
        setLoading(false);
      }
    })();
  }, [id]);

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
      },
    });
  }

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  const showChatFab = text != null && !meta;

  return (
    <View style={styles.flex}>
      <ScrollView contentContainerStyle={styles.scroll}>
        {meta ? <Text style={styles.warn}>{meta}</Text> : null}
        {text != null ? <Text style={styles.body}>{text}</Text> : null}
      </ScrollView>
      {showChatFab ? (
        <Pressable
          style={styles.fab}
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
  warn: { color: '#a60', marginBottom: 12 },
  body: { fontSize: 14, lineHeight: 22, fontFamily: 'monospace' },
  fab: {
    position: 'absolute',
    right: 20,
    bottom: 28,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#1a1a2e',
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 6,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
});
