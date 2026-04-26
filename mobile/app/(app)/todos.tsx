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
import { listTodos, toggleTodo, type TodoListResult } from '../../src/api/todos';

type TodoRow = {
  file_path?: string;
  line_number?: number;
  line?: string;
  state?: string;
  heading_text?: string;
};

export default function TodosScreen() {
  const [data, setData] = useState<TodoListResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    const res = await listTodos({ scope: 'all', limit: 200 });
    setData(res);
  }, []);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await load();
    } finally {
      setRefreshing(false);
    }
  }, [load]);

  useEffect(() => {
    void (async () => {
      try {
        await load();
      } finally {
        setLoading(false);
      }
    })();
  }, [load]);

  if (loading && !data) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  const results = (data?.results ?? []) as TodoRow[];

  return (
    <FlatList
      data={results}
      keyExtractor={(item, index) => `${item.file_path}-${item.line_number}-${index}`}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
      contentContainerStyle={styles.list}
      ListEmptyComponent={<Text style={styles.empty}>No todos found.</Text>}
      renderItem={({ item }) => (
        <View style={styles.card}>
          <Text style={styles.line} numberOfLines={3}>
            {item.line ?? JSON.stringify(item)}
          </Text>
          <Text style={styles.meta}>
            {item.state ?? ''} · {item.file_path ?? ''}
          </Text>
          {item.file_path != null && item.line_number != null ? (
            <Pressable
              style={styles.toggle}
              onPress={async () => {
                try {
                  await toggleTodo({
                    file_path: item.file_path!,
                    line_number: item.line_number!,
                    heading_text: item.heading_text ?? null,
                  });
                  await load();
                } catch {
                  /* toast optional */
                }
              }}
            >
              <Text style={styles.toggleText}>Toggle TODO/DONE</Text>
            </Pressable>
          ) : null}
        </View>
      )}
    />
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  list: { padding: 16, paddingBottom: 32 },
  empty: { textAlign: 'center', marginTop: 48, color: '#666' },
  card: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  line: { fontSize: 15, marginBottom: 6 },
  meta: { fontSize: 12, color: '#666', marginBottom: 8 },
  toggle: { alignSelf: 'flex-start', backgroundColor: '#eef', paddingVertical: 6, paddingHorizontal: 12, borderRadius: 6 },
  toggleText: { color: '#1a1a2e', fontWeight: '600' },
});
