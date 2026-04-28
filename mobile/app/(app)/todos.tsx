import { Ionicons } from '@expo/vector-icons';
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Pressable,
  RefreshControl,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import {
  listTodos,
  toggleTodo,
  type OrgTodoListItem,
  type TodoListResult,
} from '../../src/api/todos';

const DONE_STATES = new Set(['DONE', 'CANCELED', 'CANCELLED']);

function isDoneState(state: string | null | undefined): boolean {
  return DONE_STATES.has((state ?? '').toUpperCase());
}

function stateChipStyle(state: string | null | undefined): { backgroundColor: string; color: string } {
  const s = (state ?? '').toUpperCase();
  if (DONE_STATES.has(s)) return { backgroundColor: '#e8f5e9', color: '#1b5e20' };
  if (s === 'NEXT' || s === 'STARTED') return { backgroundColor: '#e3f2fd', color: '#0d47a1' };
  if (s === 'WAITING' || s === 'HOLD') return { backgroundColor: '#fff3e0', color: '#e65100' };
  if (s === 'TODO') return { backgroundColor: '#f3e5f5', color: '#4a148c' };
  return { backgroundColor: '#eceff1', color: '#37474f' };
}

function formatPlanningLine(scheduled?: string | null, deadline?: string | null): string | null {
  const parts: string[] = [];
  if (scheduled?.trim()) parts.push(`Scheduled: ${scheduled.trim()}`);
  if (deadline?.trim()) parts.push(`Deadline: ${deadline.trim()}`);
  return parts.length ? parts.join(' · ') : null;
}

function toggleLabel(state: string | null | undefined): string {
  return isDoneState(state) ? 'Reopen' : 'Mark done';
}

function TodoCard({
  item,
  onToggled,
}: {
  item: OrgTodoListItem;
  onToggled: () => Promise<void>;
}) {
  const preview = (item.preview ?? item.body_preview ?? '').trim();
  const planning = formatPlanningLine(item.scheduled, item.deadline);
  const tags = (item.tags ?? []).filter(Boolean);
  const chip = stateChipStyle(item.todo_state);
  const canToggle = Boolean(item.file_path) && item.line_number != null;

  return (
    <View style={styles.card}>
      <View style={styles.cardTop}>
        <View style={[styles.stateChip, { backgroundColor: chip.backgroundColor }]}>
          <Text style={[styles.stateChipText, { color: chip.color }]}>
            {(item.todo_state ?? 'TODO').toUpperCase()}
          </Text>
        </View>
        {item.priority ? (
          <View style={styles.priorityChip}>
            <Text style={styles.priorityChipText}>#{String(item.priority).toUpperCase()}</Text>
          </View>
        ) : null}
        <Text style={styles.filename} numberOfLines={1}>
          {item.filename || '—'}
        </Text>
      </View>

      <Text style={styles.heading} accessibilityRole="header">
        {item.heading}
      </Text>

      {preview ? (
        <Text style={styles.preview} numberOfLines={3}>
          {preview}
        </Text>
      ) : null}

      {planning ? (
        <View style={styles.planningRow}>
          <Ionicons name="calendar-outline" size={14} color="#5c6bc0" style={styles.planningIcon} />
          <Text style={styles.planningText} numberOfLines={2}>
            {planning}
          </Text>
        </View>
      ) : null}

      {tags.length > 0 ? (
        <View style={styles.tagsRow}>
          {tags.slice(0, 6).map((tag) => (
            <View key={tag} style={styles.tagChip}>
              <Text style={styles.tagChipText}>{tag}</Text>
            </View>
          ))}
          {tags.length > 6 ? (
            <Text style={styles.tagOverflow}>+{tags.length - 6}</Text>
          ) : null}
        </View>
      ) : null}

      <Text style={styles.pathHint} numberOfLines={1} accessibilityLabel={item.file_path}>
        {item.file_path}
      </Text>

      {canToggle ? (
        <Pressable
          style={({ pressed }) => [styles.toggle, pressed && styles.togglePressed]}
          onPress={async () => {
            try {
              await toggleTodo({
                file_path: item.file_path,
                line_number: item.line_number,
                heading_text: item.heading ?? null,
              });
              await onToggled();
            } catch (e) {
              Alert.alert(
                'Could not update todo',
                e instanceof Error ? e.message : 'Request failed'
              );
            }
          }}
        >
          <Text style={styles.toggleText}>{toggleLabel(item.todo_state)}</Text>
        </Pressable>
      ) : null}
    </View>
  );
}

export default function TodosScreen() {
  const [data, setData] = useState<TodoListResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scope, setScope] = useState<'all' | 'inbox'>('all');
  const [queryInput, setQueryInput] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [queryPending, setQueryPending] = useState(false);

  useEffect(() => {
    setQueryPending(true);
    const t = setTimeout(() => {
      setDebouncedQuery(queryInput.trim());
      setQueryPending(false);
    }, 400);
    return () => clearTimeout(t);
  }, [queryInput]);

  const load = useCallback(async () => {
    setError(null);
    try {
      const res = await listTodos({
        scope,
        limit: 200,
        query: debouncedQuery || undefined,
      });
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load todos');
      setData(null);
    }
  }, [scope, debouncedQuery]);

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

  const results = data?.results ?? [];

  const summaryLine = useMemo(() => {
    if (!data?.success && data != null) return null;
    const n = data?.count ?? results.length;
    const files = data?.files_searched;
    if (files != null && files >= 0) {
      return `${n} todo${n === 1 ? '' : 's'} · ${files} file${files === 1 ? '' : 's'}`;
    }
    return `${n} todo${n === 1 ? '' : 's'}`;
  }, [data, results.length]);

  if (loading && !data) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  return (
    <FlatList
      data={results}
      keyExtractor={(item, index) => `${item.file_path}-${item.line_number}-${index}`}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
      contentContainerStyle={styles.list}
      keyboardShouldPersistTaps="handled"
      ListHeaderComponent={
        <View style={styles.headerBlock}>
          {error ? (
            <Text style={styles.errorBanner} accessibilityRole="alert">
              {error}
            </Text>
          ) : null}

          <View style={styles.scopeRow}>
            {(['all', 'inbox'] as const).map((key) => (
              <Pressable
                key={key}
                style={[styles.scopeChip, scope === key && styles.scopeChipActive]}
                onPress={() => setScope(key)}
              >
                <Text style={[styles.scopeChipText, scope === key && styles.scopeChipTextActive]}>
                  {key === 'all' ? 'All org files' : 'Inbox'}
                </Text>
              </Pressable>
            ))}
          </View>

          <View style={styles.searchRow}>
            <Ionicons name="search" size={18} color="#757575" style={styles.searchIcon} />
            <TextInput
              style={styles.searchInput}
              placeholder="Search headings and notes"
              placeholderTextColor="#9e9e9e"
              value={queryInput}
              onChangeText={setQueryInput}
              returnKeyType="search"
              autoCorrect={false}
              autoCapitalize="none"
            />
            {queryPending ? <ActivityIndicator size="small" color="#5c6bc0" /> : null}
          </View>

          {summaryLine ? <Text style={styles.summary}>{summaryLine}</Text> : null}
        </View>
      }
      ListEmptyComponent={
        <Text style={styles.empty}>
          {debouncedQuery ? 'No todos match your search.' : 'No todos found.'}
        </Text>
      }
      renderItem={({ item }) => <TodoCard item={item} onToggled={load} />}
    />
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  list: { padding: 16, paddingBottom: 32 },
  headerBlock: { marginBottom: 8 },
  errorBanner: {
    backgroundColor: '#fee',
    color: '#a00',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
    fontSize: 14,
  },
  scopeRow: { flexDirection: 'row', gap: 8, marginBottom: 12 },
  scopeChip: {
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 20,
    backgroundColor: '#e0e0e0',
  },
  scopeChipActive: { backgroundColor: '#1a1a2e' },
  scopeChipText: { fontSize: 14, fontWeight: '600', color: '#424242' },
  scopeChipTextActive: { color: '#fff' },
  searchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    paddingHorizontal: 10,
    marginBottom: 8,
  },
  searchIcon: { marginRight: 6 },
  searchInput: { flex: 1, fontSize: 16, paddingVertical: 10, color: '#212121' },
  summary: { fontSize: 13, color: '#616161', marginBottom: 4 },
  empty: { textAlign: 'center', marginTop: 48, color: '#666', paddingHorizontal: 24 },
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e8e8e8',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 2,
    elevation: 2,
  },
  cardTop: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    gap: 8,
  },
  stateChip: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  stateChipText: { fontSize: 11, fontWeight: '700', letterSpacing: 0.3 },
  priorityChip: {
    backgroundColor: '#fce4ec',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  priorityChipText: { fontSize: 11, fontWeight: '700', color: '#880e4f' },
  filename: { flex: 1, fontSize: 12, color: '#757575', textAlign: 'right' },
  heading: { fontSize: 17, fontWeight: '600', color: '#1a1a2e', marginBottom: 6, lineHeight: 22 },
  preview: { fontSize: 14, color: '#555', lineHeight: 20, marginBottom: 8 },
  planningRow: { flexDirection: 'row', alignItems: 'flex-start', marginBottom: 8 },
  planningIcon: { marginTop: 2, marginRight: 6 },
  planningText: { flex: 1, fontSize: 13, color: '#5c6bc0', lineHeight: 18 },
  tagsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginBottom: 8 },
  tagChip: {
    backgroundColor: '#f5f5f5',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  tagChipText: { fontSize: 12, color: '#424242' },
  tagOverflow: { fontSize: 12, color: '#9e9e9e', alignSelf: 'center' },
  pathHint: { fontSize: 11, color: '#9e9e9e', marginBottom: 10 },
  toggle: {
    alignSelf: 'flex-start',
    backgroundColor: '#e8eaf6',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 8,
  },
  togglePressed: { opacity: 0.85 },
  toggleText: { color: '#1a1a2e', fontWeight: '700', fontSize: 14 },
});
