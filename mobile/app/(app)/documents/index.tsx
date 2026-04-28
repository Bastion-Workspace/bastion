import { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Pressable,
  RefreshControl,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { useRouter } from 'expo-router';
import { listUserDocuments, type DocumentInfo } from '../../../src/api/documents';

export default function DocumentsListScreen() {
  const router = useRouter();
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const res = await listUserDocuments(0, 100);
      setDocs(res.documents ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load documents');
      setDocs([]);
    }
  }, []);

  useEffect(() => {
    void (async () => {
      await load();
      setLoading(false);
    })();
  }, [load]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await load();
    } finally {
      setRefreshing(false);
    }
  }, [load]);

  if (loading && docs.length === 0) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  return (
    <FlatList
      data={docs}
      keyExtractor={(item) => item.document_id}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
      contentContainerStyle={styles.list}
      ListHeaderComponent={
        error ? (
          <Text style={styles.errorBanner} accessibilityRole="alert">
            {error}
          </Text>
        ) : null
      }
      ListEmptyComponent={<Text style={styles.empty}>No documents.</Text>}
      renderItem={({ item }) => (
        <Pressable
          style={styles.row}
          onPress={() => router.push(`/documents/${item.document_id}`)}
        >
          <Text style={styles.title}>{item.title || item.filename}</Text>
          <Text style={styles.sub}>{item.filename}</Text>
        </Pressable>
      )}
    />
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  list: { padding: 16 },
  errorBanner: {
    backgroundColor: '#fee',
    color: '#a00',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
    fontSize: 14,
  },
  empty: { textAlign: 'center', marginTop: 48, color: '#666' },
  row: {
    backgroundColor: '#fff',
    padding: 14,
    borderRadius: 8,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  title: { fontSize: 16, fontWeight: '600' },
  sub: { fontSize: 13, color: '#666', marginTop: 4 },
});
