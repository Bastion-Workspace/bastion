import { Ionicons } from '@expo/vector-icons';
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
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
import { getTodoStates } from '../../src/api/orgSettings';
import {
  listTodos,
  toggleTodo,
  updateTodo,
  type OrgTodoListItem,
  type TodoListResult,
} from '../../src/api/todos';
import { formatOrgPlanningTimestamp } from '../../src/utils/orgTimestampFormat';

const FALLBACK_DONE = ['CANCELED', 'CANCELLED', 'DONE'];
const FALLBACK_ACTIVE = ['HOLD', 'NEXT', 'STARTED', 'TODO', 'WAITING'];

function basenamePath(p: string): string {
  const s = p.replace(/\\/g, '/');
  const i = s.lastIndexOf('/');
  return i >= 0 ? s.slice(i + 1) : s;
}

/** Last two segments of a path for compact display (e.g. Projects/inbox.org). */
function lastTwoPathSegments(filePath: string): string {
  const parts = filePath.replace(/\\/g, '/').split('/').filter(Boolean);
  if (parts.length === 0) return '';
  if (parts.length === 1) return parts[0];
  return `${parts[parts.length - 2]}/${parts[parts.length - 1]}`;
}

function isDoneState(state: string | null | undefined, doneSet: Set<string>): boolean {
  return doneSet.has((state ?? '').toUpperCase());
}

function stateChipStyle(
  state: string | null | undefined,
  doneSet: Set<string>
): { backgroundColor: string; color: string } {
  const s = (state ?? '').toUpperCase();
  if (doneSet.has(s)) return { backgroundColor: '#e8f5e9', color: '#1b5e20' };
  if (s === 'NEXT' || s === 'STARTED') return { backgroundColor: '#e3f2fd', color: '#0d47a1' };
  if (s === 'WAITING' || s === 'HOLD') return { backgroundColor: '#fff3e0', color: '#e65100' };
  if (s === 'TODO') return { backgroundColor: '#f3e5f5', color: '#4a148c' };
  return { backgroundColor: '#eceff1', color: '#37474f' };
}

function isMarkdownCheckboxHeading(heading: string | null | undefined): boolean {
  return /^\s*-\s*\[[ xX]\]\s/.test((heading ?? '').trim());
}

function isCheckboxChecked(heading: string | null | undefined): boolean {
  return /-\s*\[[xX]\]/.test(heading ?? '');
}

type TodoStates = { active: string[]; done: string[] };

type ListMode = 'all' | 'inbox' | 'file';

function TodoCard({
  item,
  onToggled,
  todoStates,
  doneSet,
}: {
  item: OrgTodoListItem;
  onToggled: () => Promise<void>;
  todoStates: TodoStates;
  doneSet: Set<string>;
}) {
  const [stateModalOpen, setStateModalOpen] = useState(false);
  const preview = (item.preview ?? item.body_preview ?? '').trim();
  const schedFmt = formatOrgPlanningTimestamp(item.scheduled);
  const deadFmt = formatOrgPlanningTimestamp(item.deadline);
  const tags = (item.tags ?? []).filter(Boolean);
  const canToggle = Boolean(item.file_path) && item.line_number != null;
  const checkbox = isMarkdownCheckboxHeading(item.heading);
  const displayState = checkbox
    ? isCheckboxChecked(item.heading)
      ? 'DONE'
      : 'TODO'
    : item.todo_state ?? 'TODO';
  const chip = stateChipStyle(displayState, doneSet);
  const pathContext = item.file_path ? lastTwoPathSegments(item.file_path) : '';

  const doneList = todoStates.done.length ? todoStates.done : [...FALLBACK_DONE].sort();
  const activeList = todoStates.active.length ? todoStates.active : [...FALLBACK_ACTIVE].sort();
  const defaultDone = doneList[0] ?? 'DONE';
  const defaultActive = activeList[0] ?? 'TODO';
  const currentDone = checkbox
    ? isCheckboxChecked(item.heading)
    : isDoneState(item.todo_state, doneSet);
  const headingSnippet = (item.heading ?? '').trim().slice(0, 120);

  async function runToggleCheckbox() {
    await toggleTodo({
      file_path: item.file_path,
      line_number: item.line_number,
      heading_text: item.heading ?? null,
    });
    await onToggled();
  }

  async function runUpdateState(newState: string) {
    await updateTodo({
      file_path: item.file_path,
      line_number: item.line_number,
      heading_text: item.heading ?? null,
      new_state: newState,
    });
    await onToggled();
  }

  function confirmStateChange(title: string, message: string, onConfirm: () => Promise<void>) {
    Alert.alert(title, message, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Confirm',
        onPress: () => {
          void (async () => {
            try {
              await onConfirm();
            } catch (e) {
              Alert.alert(
                'Could not update todo',
                e instanceof Error ? e.message : 'Request failed'
              );
            }
          })();
        },
      },
    ]);
  }

  function onPressQuickState() {
    if (!canToggle) return;
    if (checkbox) {
      const checked = isCheckboxChecked(item.heading);
      if (checked) {
        confirmStateChange(
          'Mark incomplete',
          `Uncheck this item?\n\n${headingSnippet}`,
          runToggleCheckbox
        );
      } else {
        confirmStateChange(
          'Mark complete',
          `Check this item off?\n\n${headingSnippet}`,
          runToggleCheckbox
        );
      }
      return;
    }
    if (currentDone) {
      confirmStateChange(
        `Reopen as ${defaultActive}?`,
        `Heading:\n${headingSnippet}`,
        () => runUpdateState(defaultActive)
      );
    } else {
      confirmStateChange(
        `Mark as ${defaultDone}?`,
        `Heading:\n${headingSnippet}`,
        () => runUpdateState(defaultDone)
      );
    }
  }

  function onLongPressState() {
    if (!canToggle) return;
    setStateModalOpen(true);
  }

  function pickFromModal(targetState: string) {
    setStateModalOpen(false);
    const verb = doneSet.has(targetState.toUpperCase()) ? 'Mark' : 'Reopen';
    confirmStateChange(
      `${verb} as ${targetState}?`,
      `Heading:\n${headingSnippet}`,
      () => runUpdateState(targetState)
    );
  }

  function pickCheckboxFromModal(markComplete: boolean) {
    setStateModalOpen(false);
    const checked = isCheckboxChecked(item.heading);
    if (markComplete === checked) return;
    confirmStateChange(
      markComplete ? 'Mark complete' : 'Mark incomplete',
      headingSnippet,
      runToggleCheckbox
    );
  }

  const checkedNow = isCheckboxChecked(item.heading);
  const quickActionLabel = currentDone ? 'Open' : 'Done';

  const modalTargets = checkbox
    ? [
        !checkedNow
          ? {
              key: 'check',
              label: 'Mark complete (checked)',
              onPick: () => pickCheckboxFromModal(true),
            }
          : null,
        checkedNow
          ? {
              key: 'uncheck',
              label: 'Mark incomplete (unchecked)',
              onPick: () => pickCheckboxFromModal(false),
            }
          : null,
      ].filter(Boolean) as { key: string; label: string; onPick: () => void }[]
    : currentDone
      ? activeList.map((s) => ({
          key: s,
          label: `Reopen as ${s}`,
          onPick: () => pickFromModal(s),
        }))
      : doneList.map((s) => ({
          key: s,
          label: `Mark as ${s}`,
          onPick: () => pickFromModal(s),
        }));

  return (
    <View style={styles.card}>
      <View style={styles.cardTop}>
        <View style={[styles.stateChip, { backgroundColor: chip.backgroundColor }]}>
          <Text style={[styles.stateChipText, { color: chip.color }]}>{displayState.toUpperCase()}</Text>
        </View>
        {item.priority ? (
          <View style={styles.priorityChip}>
            <Text style={styles.priorityChipText}>#{String(item.priority).toUpperCase()}</Text>
          </View>
        ) : null}
        <Text style={styles.filename} numberOfLines={1}>
          {item.filename || basenamePath(item.file_path || '') || '—'}
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

      {schedFmt ? (
        <View style={styles.planningRow}>
          <Ionicons name="time-outline" size={14} color="#5c6bc0" style={styles.planningIcon} />
          <Text style={styles.planningLabel}>Scheduled</Text>
          <Text style={styles.planningValue} numberOfLines={2}>
            {schedFmt}
          </Text>
        </View>
      ) : null}
      {deadFmt ? (
        <View style={styles.planningRow}>
          <Ionicons name="flag-outline" size={14} color="#c62828" style={styles.planningIcon} />
          <Text style={styles.planningLabel}>Deadline</Text>
          <Text style={[styles.planningValue, styles.planningDeadline]} numberOfLines={2}>
            {deadFmt}
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

      {pathContext ? (
        <Text style={styles.pathShort} numberOfLines={1}>
          {pathContext}
        </Text>
      ) : null}

      {canToggle ? (
        <Pressable
          style={({ pressed }) => [styles.toggle, pressed && styles.togglePressed]}
          onPress={onPressQuickState}
          onLongPress={onLongPressState}
          delayLongPress={400}
          accessibilityRole="button"
        >
          <Text style={styles.toggleText}>{quickActionLabel}</Text>
        </Pressable>
      ) : null}

      <Modal
        visible={stateModalOpen}
        animationType="slide"
        transparent
        onRequestClose={() => setStateModalOpen(false)}
      >
        <Pressable style={styles.modalBackdrop} onPress={() => setStateModalOpen(false)}>
          <Pressable style={styles.sheet} onPress={(e) => e.stopPropagation()}>
            <Text style={styles.sheetTitle}>Choose state</Text>
            <ScrollView style={styles.sheetList} keyboardShouldPersistTaps="handled">
              {modalTargets.map((row) => (
                <Pressable
                  key={row.key}
                  style={styles.sheetRow}
                  onPress={() => {
                    row.onPick();
                  }}
                >
                  <Text style={styles.sheetRowTitle}>{row.label}</Text>
                </Pressable>
              ))}
            </ScrollView>
            <Pressable style={styles.sheetClose} onPress={() => setStateModalOpen(false)}>
              <Text style={styles.sheetCloseText}>Cancel</Text>
            </Pressable>
          </Pressable>
        </Pressable>
      </Modal>
    </View>
  );
}

type OrgFileRow = { file_path: string; filename: string };

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

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await loadTodoStates();
      await load();
    } finally {
      setRefreshing(false);
    }
  }, [load, loadTodoStates]);

  const results = data?.results ?? [];

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
          </View>
        }
        ListEmptyComponent={
          <Text style={styles.empty}>
            {debouncedQuery ? 'No todos match your search.' : 'No todos found.'}
          </Text>
        }
        renderItem={({ item }) => (
          <TodoCard item={item} onToggled={load} todoStates={todoStates} doneSet={doneSet} />
        )}
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
  planningRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    marginBottom: 6,
    gap: 6,
  },
  planningIcon: { marginRight: 0 },
  planningLabel: { fontSize: 12, fontWeight: '700', color: '#616161', width: 72 },
  planningValue: { flex: 1, fontSize: 13, color: '#3949ab', lineHeight: 18, minWidth: 120 },
  planningDeadline: { color: '#c62828' },
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
  pathShort: { fontSize: 11, color: '#9e9e9e', marginBottom: 8 },
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
