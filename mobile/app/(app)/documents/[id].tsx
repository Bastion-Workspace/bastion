import { useEffect, useState } from 'react';
import { ActivityIndicator, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useLocalSearchParams } from 'expo-router';
import { getDocumentContent } from '../../../src/api/documents';

export default function DocumentDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const [text, setText] = useState<string | null>(null);
  const [meta, setMeta] = useState<string>('');
  const [loading, setLoading] = useState(true);

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

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  return (
    <ScrollView contentContainerStyle={styles.scroll}>
      {meta ? <Text style={styles.warn}>{meta}</Text> : null}
      {text != null ? <Text style={styles.body}>{text}</Text> : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  scroll: { padding: 16, paddingBottom: 32 },
  warn: { color: '#a60', marginBottom: 12 },
  body: { fontSize: 14, lineHeight: 22, fontFamily: 'monospace' },
});
