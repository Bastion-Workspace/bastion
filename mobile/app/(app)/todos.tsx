import { Ionicons } from '@expo/vector-icons';
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Modal,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { OrgTodoCard, type TodoStates } from '../../src/components/org/OrgTodoCard';
import { getTodoStates } from '../../src/api/orgSettings';
import { listTodos, type OrgTodoListItem, type TodoListResult } from '../../src/api/todos';
import {
  buildHierarchicalTree,
  collectPathKeys,
  flattenTodosForList,
  groupTodosByFileOrdered,
  pathKey,
  type FlatTodoRow,
} from '../../src/utils/todoTree';

const FALLBACK_DONE = ['CANCELED', 'CANCELLED', 'DONE'];

function basenamePath(p: string): string {
  const s = p.replace(/\\/g, '/');
  const i = s.lastIndexOf('/');
  return i >= 0 ? s.slice(i + 1) : s;
}

function lastTwoPathSegments(filePath: string): string {
  const parts = filePath.replace(/\\/g, '/').split('/').filter(Boolean);
  if (parts.length === 0) return '';
  if (parts.length === 1) return parts[0];
  return `${parts[parts.length - 2]}/${parts[parts.length - 1]}`;
}

type ListMode = 'all' | 'inbox' | 'file';

type OrgFileRow = { file_path: string; filename: string };

function initExpandedState(results: OrgTodoListItem[]): {
  expandedFiles: Set<string>;
  expandedPaths: Set<string>;
} {
  const byFile = groupTodosByFileOrdered(results);
  const expandedFiles = new Set(byFile.keys());
  const expandedPaths = new Set<string>();
  for (const items of byFile.values()) {
    collectPathKeys(buildHierarchicalTree(items), expandedPaths);
  }
  return { expandedFiles, expandedPaths };
}

export default function TodosScreen() {
  const [data, setData] = useState<TodoListResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [listMode, setListMode] = useState<ListMode>('all');
  const [singleFilePath, setSingleFilePath] = useState<string | null>(null);
  const [filePickerOpen, setFilePickerOpen] = useState(false);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [queryInput, setQueryInput] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [queryPending, setQueryPending] = useState(false);
  const [todoStates, setTodoStates] = useState<TodoStates>({ active: [], done: [] });
  const [expandedFiles, setExpandedFiles] = useState<Set<string>>(() => new Set());
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(() => new Set());

  const doneSet = useMemo(() => {
    const d = todoStates.done.length ? todoStates.done : FALLBACK_DONE;
    return new Set(d.map((x) => x.toUpperCase()));
  }, [todoStates.done]);

  const loadTodoStates = useCallback(async () => {
    try {
      const s = await getTodoStates();
      setTodoStates({ active: s.active, done: s.done });
    } catch {
      setTodoStates({ active: [], done: [] });
    }
  }, []);

  useEffect(() => {
    void loadTodoStates();
  }, [loadTodoStates]);

  useEffect(() => {
    setQueryPending(true);
    const t = setTimeout(() => {
      setDebouncedQuery(queryInput.trim());
      setQueryPending(false);
    }, 400);
    return () => clearTimeout(t);
  }, [queryInput]);

  const scopeParam = useMemo(() => {
    if (listMode === 'file' && singleFilePath) return singleFilePath;
    if (listMode === 'inbox') return 'inbox';
    return 'all';
  }, [listMode, singleFilePath]);

  const load = useCallback(async () => {
    setError(null);
    try {
      const res = await listTodos({
        scope: scopeParam,
        limit: 200,
        query: debouncedQuery || undefined,
        tags: selectedTags.length > 0 ? selectedTags : undefined,
      });
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load todos');
      setData(null);
    }
  }, [scopeParam, debouncedQuery, selectedTags]);

  useEffect(() => {
    void (async () => {
      await load();
      setLoading(false);
    })();
  }, [load]);

  const results = data?.results ?? [];

  useEffect(() => {
    const { expandedFiles: ef, expandedPaths: ep } = initExpandedState(results);
    setExpandedFiles(ef);
    setExpandedPaths(ep);
  }, [results]);

  const flatRows = useMemo(
    () => flattenTodosForList(results, expandedFiles, expandedPaths),
    [results, expandedFiles, expandedPaths]
  );

  const toggleFile = useCallback((filePath: string) => {
    setExpandedFiles((prev) => {
      const n = new Set(prev);
      if (n.has(filePath)) n.delete(filePath);
      else n.add(filePath);
      return n;
    });
  }, []);

  const togglePath = useCallback((path: string[]) => {
    const k = pathKey(path);
    setExpandedPaths((prev) => {
      const n = new Set(prev);
      if (n.has(k)) n.delete(k);
      else n.add(k);
      return n;
    });
  }, []);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await loadTodoStates();
      await load();
    } finally {
      setRefreshing(false);
    }
  }, [load, loadTodoStates]);

  const orgFilesWithTodos = useMemo((): OrgFileRow[] => {
    const map = new Map<string, OrgFileRow>();
    for (const r of results) {
      const fp = r.file_path?.trim();
      if (!fp) continue;
      if (!map.has(fp)) {
        map.set(fp, { file_path: fp, filename: r.filename || basenamePath(fp) });
      }
    }
    return [...map.values()].sort((a, b) => a.filename.localeCompare(b.filename));
  }, [results]);

  const tagOptions = useMemo(() => {
    const s = new Set<string>();
    for (const r of results) {
      for (const t of r.tags ?? []) {
        if (t && String(t).trim()) s.add(String(t).trim());
      }
    }
    return [...s].sort((a, b) => a.localeCompare(b));
  }, [results]);

  function toggleFilterTag(tag: string) {
    setSelectedTags((prev) =>
      prev.includes(tag) ? prev.filter((x) => x !== tag) : [...prev, tag]
    );
  }

  const summaryLine = useMemo(() => {
    if (!data?.success && data != null) return null;
    const n = data?.count ?? results.length;
    const files = data?.files_searched;
    if (files != null && files >= 0) {
      return `${n} todo${n === 1 ? '' : 's'} · ${files} file${files === 1 ? '' : 's'}`;
    }
    return `${n} todo${n === 1 ? '' : 's'}`;
  }, [data, results.length]);

  const filesChipLabel =
    listMode === 'file' && singleFilePath
      ? basenamePath(singleFilePath)
      : 'All org files';

  const inboxActive = listMode === 'inbox';
  const filesChipActive = listMode === 'all' || listMode === 'file';

  const renderRow = useCallback(
    ({ item: row }: { item: FlatTodoRow }) => {
      if (row.kind === 'file') {
        return (
          <Pressable
            onPress={() => toggleFile(row.filePath)}
            style={({ pressed }) => [styles.fileHeader, pressed && styles.fileHeaderPressed]}
            accessibilityRole="button"
            accessibilityState={{ expanded: row.expanded }}
            accessibilityLabel={`${row.filename}, ${row.todoCount} todos`}
          >
            <Ionicons
              name={row.expanded ? 'chevron-down' : 'chevron-forward'}
              size={20}
              color="#1a1a2e"
              style={styles.sectionChevron}
            />
            <Ionicons name="document-text-outline" size={18} color="#3949ab" style={styles.fileIcon} />
            <Text style={styles.fileHeaderTitle} numberOfLines={1}>
              {row.filename}
            </Text>
            <View style={styles.countBadge}>
              <Text style={styles.countBadgeText}>{row.todoCount}</Text>
            </View>
          </Pressable>
        );
      }
      if (row.kind === 'section') {
        return (
          <Pressable
            onPress={() => togglePath(row.path)}
            style={({ pressed }) => [
              styles.sectionHeader,
              { marginLeft: 8 + row.depth * 14 },
              pressed && styles.sectionHeaderPressed,
            ]}
            accessibilityRole="button"
            accessibilityState={{ expanded: row.expanded }}
            accessibilityLabel={`Section ${row.heading}, ${row.todoCount} todos`}
          >
            <Ionicons
              name={row.expanded ? 'chevron-down' : 'chevron-forward'}
              size={18}
              color="#3949ab"
              style={styles.sectionChevron}
            />
            <Text style={styles.sectionTitle} numberOfLines={2}>
              {row.heading}
            </Text>
            <View style={[styles.countBadge, styles.countBadgeMuted]}>
              <Text style={styles.countBadgeText}>{row.todoCount}</Text>
            </View>
          </Pressable>
        );
      }
      return (
        <OrgTodoCard
          item={row.item}
          depth={row.depth}
          onToggled={load}
          todoStates={todoStates}
          doneSet={doneSet}
        />
      );
    },
    [doneSet, load, todoStates, toggleFile, togglePath]
  );

  if (loading && !data) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  return (
    <>
      <FlatList
        data={flatRows}
        keyExtractor={(item) => item.key}
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
              <Pressable
                style={[styles.scopeChip, inboxActive && styles.scopeChipActive]}
                onPress={() => {
                  setListMode('inbox');
                  setSingleFilePath(null);
                }}
              >
                <Text style={[styles.scopeChipText, inboxActive && styles.scopeChipTextActive]}>Inbox</Text>
              </Pressable>
              <Pressable
                style={[styles.scopeChip, filesChipActive && styles.scopeChipActive]}
                onPress={() => {
                  if (listMode === 'file') {
                    setListMode('all');
                    setSingleFilePath(null);
                  }
                }}
                onLongPress={() => setFilePickerOpen(true)}
                delayLongPress={400}
              >
                <Text
                  style={[styles.scopeChipText, filesChipActive && styles.scopeChipTextActive]}
                  numberOfLines={1}
                >
                  {filesChipLabel}
                </Text>
              </Pressable>
            </View>
            <Text style={styles.scopeHint}>Long-press the file chip to filter by one org file</Text>

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

            {tagOptions.length > 0 || selectedTags.length > 0 ? (
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.tagFilterScroll}>
                {selectedTags.length > 0 ? (
                  <Pressable style={styles.clearTagsChip} onPress={() => setSelectedTags([])}>
                    <Text style={styles.clearTagsText}>Clear tags</Text>
                  </Pressable>
                ) : null}
                {tagOptions.map((tag) => {
                  const on = selectedTags.includes(tag);
                  return (
                    <Pressable
                      key={tag}
                      style={[styles.filterTagChip, on && styles.filterTagChipOn]}
                      onPress={() => toggleFilterTag(tag)}
                    >
                      <Text style={[styles.filterTagChipText, on && styles.filterTagChipTextOn]}>{tag}</Text>
                    </Pressable>
                  );
                })}
              </ScrollView>
            ) : null}

            {summaryLine ? <Text style={styles.summary}>{summaryLine}</Text> : null}
            {groupTodosByFileOrdered(results).size > 1 ? (
              <Text style={styles.hierarchyHint}>Use file and section rows to expand or collapse nested todos.</Text>
            ) : null}
          </View>
        }
        ListEmptyComponent={
          <Text style={styles.empty}>
            {debouncedQuery ? 'No todos match your search.' : 'No todos found.'}
          </Text>
        }
        renderItem={renderRow}
        ListFooterComponent={<View style={{ height: 8 }} />}
      />
      <Modal visible={filePickerOpen} animationType="slide" transparent onRequestClose={() => setFilePickerOpen(false)}>
        <Pressable style={styles.modalBackdrop} onPress={() => setFilePickerOpen(false)}>
          <Pressable style={styles.sheet} onPress={(e) => e.stopPropagation()}>
            <Text style={styles.sheetTitle}>Org files with todos</Text>
            <FlatList
              data={[{ file_path: '__all__', filename: 'All org files' }, ...orgFilesWithTodos]}
              keyExtractor={(item) => item.file_path}
              style={styles.sheetList}
              renderItem={({ item }) => (
                <Pressable
                  style={styles.sheetRow}
                  onPress={() => {
                    if (item.file_path === '__all__') {
                      setListMode('all');
                      setSingleFilePath(null);
                    } else {
                      setListMode('file');
                      setSingleFilePath(item.file_path);
                    }
                    setFilePickerOpen(false);
                  }}
                >
                  <Text style={styles.sheetRowTitle} numberOfLines={1}>
                    {item.filename}
                  </Text>
                  {item.file_path !== '__all__' ? (
                    <Text style={styles.sheetRowSub} numberOfLines={1}>
                      {lastTwoPathSegments(item.file_path)}
                    </Text>
                  ) : null}
                </Pressable>
              )}
            />
            <Pressable style={styles.sheetClose} onPress={() => setFilePickerOpen(false)}>
              <Text style={styles.sheetCloseText}>Close</Text>
            </Pressable>
          </Pressable>
        </Pressable>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  list: { padding: 16, paddingBottom: 32 },
  headerBlock: { marginBottom: 8 },
  hierarchyHint: { fontSize: 12, color: '#78909c', marginTop: 4, marginBottom: 2 },
  errorBanner: {
    backgroundColor: '#fee',
    color: '#a00',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
    fontSize: 14,
  },
  scopeRow: { flexDirection: 'row', gap: 8, marginBottom: 4 },
  scopeHint: { fontSize: 11, color: '#888', marginBottom: 10 },
  scopeChip: {
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 20,
    backgroundColor: '#e0e0e0',
    flexShrink: 1,
    maxWidth: '52%',
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
  tagFilterScroll: { marginBottom: 8, maxHeight: 40 },
  clearTagsChip: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 16,
    backgroundColor: '#ffebee',
    marginRight: 8,
    alignSelf: 'center',
  },
  clearTagsText: { fontSize: 12, fontWeight: '700', color: '#b71c1c' },
  filterTagChip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    backgroundColor: '#f0f0f0',
    marginRight: 8,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  filterTagChipOn: { backgroundColor: '#e8eaf6', borderColor: '#5c6bc0' },
  filterTagChipText: { fontSize: 13, color: '#424242' },
  filterTagChipTextOn: { color: '#1a237e', fontWeight: '700' },
  summary: { fontSize: 13, color: '#616161', marginBottom: 4 },
  empty: { textAlign: 'center', marginTop: 48, color: '#666', paddingHorizontal: 24 },
  modalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'flex-end',
  },
  sheet: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    maxHeight: '60%',
    paddingBottom: 16,
  },
  sheetTitle: { fontSize: 18, fontWeight: '700', padding: 16, borderBottomWidth: 1, borderColor: '#eee' },
  sheetList: { maxHeight: 360 },
  sheetRow: { paddingVertical: 14, paddingHorizontal: 16, borderBottomWidth: 1, borderColor: '#f0f0f0' },
  sheetRowTitle: { fontSize: 16, fontWeight: '600', color: '#111' },
  sheetRowSub: { fontSize: 12, color: '#888', marginTop: 4 },
  sheetClose: { marginTop: 8, alignSelf: 'center', paddingVertical: 10, paddingHorizontal: 24 },
  sheetCloseText: { fontSize: 16, color: '#1a5090', fontWeight: '600' },
  fileHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 10,
    marginBottom: 10,
    backgroundColor: '#f5f5ff',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#c5cae9',
    gap: 6,
  },
  fileHeaderPressed: { opacity: 0.88 },
  fileIcon: { flexShrink: 0 },
  fileHeaderTitle: { flex: 1, fontSize: 16, fontWeight: '700', color: '#1a237e', minWidth: 0 },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 8,
    marginBottom: 6,
    marginTop: 2,
    backgroundColor: '#fafafa',
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#5c6bc0',
    gap: 4,
  },
  sectionHeaderPressed: { opacity: 0.9 },
  sectionChevron: { flexShrink: 0 },
  sectionTitle: { flex: 1, fontSize: 14, fontWeight: '700', color: '#3949ab', minWidth: 0 },
  countBadge: {
    backgroundColor: '#3949ab',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 10,
    minWidth: 28,
    alignItems: 'center',
  },
  countBadgeMuted: { backgroundColor: '#7986cb' },
  countBadgeText: { fontSize: 12, fontWeight: '700', color: '#fff' },
});
