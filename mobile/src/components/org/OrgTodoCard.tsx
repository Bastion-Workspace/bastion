import { Ionicons } from '@expo/vector-icons';
import { useState } from 'react';
import {
  Alert,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { toggleTodo, updateTodo, type OrgTodoListItem } from '../../api/todos';
import { formatOrgPlanningTimestamp } from '../../utils/orgTimestampFormat';

const FALLBACK_DONE = ['CANCELED', 'CANCELLED', 'DONE'];
const FALLBACK_ACTIVE = ['HOLD', 'NEXT', 'STARTED', 'TODO', 'WAITING'];

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

export type TodoStates = { active: string[]; done: string[] };

export function OrgTodoCard({
  item,
  onToggled,
  todoStates,
  doneSet,
  depth = 0,
}: {
  item: OrgTodoListItem;
  onToggled: () => Promise<void>;
  todoStates: TodoStates;
  doneSet: Set<string>;
  /** Nesting depth from hierarchy (0 = top-level under file). */
  depth?: number;
}) {
  const [stateModalOpen, setStateModalOpen] = useState(false);
  const preview = (item.preview ?? item.body_preview ?? '').trim();
  const schedFmt = formatOrgPlanningTimestamp(item.scheduled);
  const deadFmt = formatOrgPlanningTimestamp(item.deadline);
  const createdFmt = formatOrgPlanningTimestamp(item.creation_timestamp);
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
    ? ([
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
      ].filter(Boolean) as { key: string; label: string; onPick: () => void }[])
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

  const showParentCrumb =
    depth === 0 &&
    item.level != null &&
    item.level > 1 &&
    item.parent_path &&
    item.parent_path.length > 0;

  return (
    <View style={[depth > 0 ? styles.nestedWrap : null, { marginLeft: depth * 12 }]}>
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

        {showParentCrumb ? (
          <Text style={styles.parentCrumb} numberOfLines={1}>
            ↳ {item.parent_path![item.parent_path!.length - 1]}
          </Text>
        ) : null}

        <Text style={styles.heading} accessibilityRole="header">
          {item.heading}
        </Text>

        {(item.notes ?? []).length > 0 ? (
          (item.notes ?? []).map((n, i) => (
            <View key={`${n.timestamp}-${i}`} style={styles.noteRow}>
              <Text style={styles.noteTimestamp}>
                Note · {formatOrgPlanningTimestamp(n.timestamp) ?? n.timestamp}
              </Text>
              <Text style={styles.noteText} numberOfLines={4}>
                {n.text}
              </Text>
            </View>
          ))
        ) : preview ? (
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
        {createdFmt ? (
          <View style={styles.planningRow}>
            <Ionicons name="calendar-outline" size={14} color="#6d4c41" style={styles.planningIcon} />
            <Text style={styles.planningLabel}>Created</Text>
            <Text style={[styles.planningValue, styles.planningCreated]} numberOfLines={2}>
              {createdFmt}
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
    </View>
  );
}

const styles = StyleSheet.create({
  nestedWrap: {
    borderLeftWidth: 3,
    borderLeftColor: '#c5cae9',
    paddingLeft: 4,
  },
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
  parentCrumb: {
    fontSize: 12,
    color: '#757575',
    marginBottom: 4,
    fontStyle: 'italic',
  },
  heading: { fontSize: 17, fontWeight: '600', color: '#1a1a2e', marginBottom: 6, lineHeight: 22 },
  noteRow: { marginBottom: 8 },
  noteTimestamp: { fontSize: 12, fontWeight: '600', color: '#5c6bc0', marginBottom: 2 },
  noteText: { fontSize: 14, color: '#424242', lineHeight: 20, fontStyle: 'italic' },
  preview: { fontSize: 14, color: '#555', lineHeight: 20, marginBottom: 8 },
  planningRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    marginBottom: 6,
    gap: 6,
  },
  planningIcon: { marginRight: 0 },
  planningLabel: { fontSize: 12, fontWeight: '700', color: '#616161', width: 76 },
  planningValue: { flex: 1, fontSize: 13, color: '#3949ab', lineHeight: 18, minWidth: 120 },
  planningDeadline: { color: '#c62828' },
  planningCreated: { color: '#5d4037' },
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
  sheetClose: { marginTop: 8, alignSelf: 'center', paddingVertical: 10, paddingHorizontal: 24 },
  sheetCloseText: { fontSize: 16, color: '#1a5090', fontWeight: '600' },
});
