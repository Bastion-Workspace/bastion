import { Ionicons } from '@expo/vector-icons';
import { useMemo, useState } from 'react';
import {
  Alert,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  useColorScheme,
  View,
} from 'react-native';
import { toggleTodo, updateTodo, type OrgTodoListItem } from '../../api/todos';
import { getColors } from '../../theme/colors';
import { formatOrgPlanningTimestamp } from '../../utils/orgTimestampFormat';

const FALLBACK_DONE = ['CANCELED', 'CANCELLED', 'DONE'];
const FALLBACK_ACTIVE = ['HOLD', 'NEXT', 'STARTED', 'TODO', 'WAITING'];

function basenamePath(p: string): string {
  const s = p.replace(/\\/g, '/');
  const i = s.lastIndexOf('/');
  return i >= 0 ? s.slice(i + 1) : s;
}

function isDoneState(state: string | null | undefined, doneSet: Set<string>): boolean {
  return doneSet.has((state ?? '').toUpperCase());
}

function stateChipStyle(
  state: string | null | undefined,
  doneSet: Set<string>,
  dark: boolean
): { backgroundColor: string; color: string } {
  const s = (state ?? '').toUpperCase();
  if (doneSet.has(s)) {
    return dark
      ? { backgroundColor: '#1b3d24', color: '#a5d6a7' }
      : { backgroundColor: '#e8f5e9', color: '#1b5e20' };
  }
  if (s === 'NEXT' || s === 'STARTED') {
    return dark
      ? { backgroundColor: '#1a237e', color: '#90caf9' }
      : { backgroundColor: '#e3f2fd', color: '#0d47a1' };
  }
  if (s === 'WAITING' || s === 'HOLD') {
    return dark
      ? { backgroundColor: '#4e342e', color: '#ffcc80' }
      : { backgroundColor: '#fff3e0', color: '#e65100' };
  }
  if (s === 'TODO') {
    return dark
      ? { backgroundColor: '#311b92', color: '#ce93d8' }
      : { backgroundColor: '#f3e5f5', color: '#4a148c' };
  }
  return dark
    ? { backgroundColor: '#37474f', color: '#cfd8dc' }
    : { backgroundColor: '#eceff1', color: '#37474f' };
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
  const scheme = useColorScheme();
  const dark = scheme === 'dark';
  const colors = useMemo(() => getColors(dark ? 'dark' : 'light'), [dark]);

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
  const chip = stateChipStyle(displayState, doneSet, dark);

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

  /** Org log lines like `[2026-04-28 Tue 21:51]` duplicate the formatted Created row — hide on the card. */
  const previewLooksLikeOrgLogTimestamp = (() => {
    const p = preview.trim();
    if (!p || !createdFmt) return false;
    return /^\[[^\]]+\]$/.test(p);
  })();

  const notes = item.notes ?? [];
  const showPreview = Boolean(preview) && !previewLooksLikeOrgLogTimestamp && notes.length === 0;

  const firstNote = notes[0];
  const firstNoteTsFmt = firstNote ? formatOrgPlanningTimestamp(firstNote.timestamp) : null;
  const noteTimestampSameAsCreated = Boolean(
    createdFmt && firstNoteTsFmt && firstNoteTsFmt === createdFmt
  );
  const noteHeaderLabel = firstNote
    ? noteTimestampSameAsCreated
      ? 'Note'
      : `Note · ${firstNoteTsFmt ?? firstNote.timestamp ?? ''}`
    : '';
  const extraNoteCount = notes.length > 1 ? notes.length - 1 : 0;

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

  const priorityBg = dark ? '#4a1942' : '#fce4ec';
  const priorityFg = dark ? '#f48fb1' : '#880e4f';

  return (
    <View
      style={[
        depth > 0 ? [styles.nestedWrap, { borderLeftColor: colors.border }] : null,
        { marginLeft: depth * 12 },
      ]}
    >
      <View
        style={[
          styles.card,
          {
            backgroundColor: colors.surface,
            borderColor: colors.border,
            shadowColor: dark ? '#000' : '#000',
            shadowOpacity: dark ? 0.35 : 0.06,
          },
        ]}
      >
        <View style={styles.cardTop}>
          {canToggle ? (
            <Pressable
              style={({ pressed }) => [
                styles.stateChip,
                { backgroundColor: chip.backgroundColor },
                pressed && styles.stateChipPressed,
              ]}
              onPress={onPressQuickState}
              onLongPress={onLongPressState}
              delayLongPress={400}
              accessibilityRole="button"
              accessibilityLabel={`Status ${displayState.toUpperCase()}`}
              accessibilityHint="Tap to toggle done or reopen. Long press to choose a different state."
            >
              <Text style={[styles.stateChipText, { color: chip.color }]}>{displayState.toUpperCase()}</Text>
            </Pressable>
          ) : (
            <View style={[styles.stateChip, { backgroundColor: chip.backgroundColor }]}>
              <Text style={[styles.stateChipText, { color: chip.color }]}>{displayState.toUpperCase()}</Text>
            </View>
          )}
          {item.priority ? (
            <View style={[styles.priorityChip, { backgroundColor: priorityBg }]}>
              <Text style={[styles.priorityChipText, { color: priorityFg }]}>
                #{String(item.priority).toUpperCase()}
              </Text>
            </View>
          ) : null}
          <Text style={[styles.filename, { color: colors.textSecondary }]} numberOfLines={1}>
            {item.filename || basenamePath(item.file_path || '') || '—'}
          </Text>
        </View>

        {showParentCrumb ? (
          <Text style={[styles.parentCrumb, { color: colors.textSecondary }]} numberOfLines={1}>
            ↳ {item.parent_path![item.parent_path!.length - 1]}
          </Text>
        ) : null}

        <Text style={[styles.heading, { color: colors.text }]} accessibilityRole="header">
          {item.heading}
        </Text>

        {notes.length > 0 && firstNote ? (
          <View style={styles.noteRow}>
            <Text style={[styles.noteTimestamp, { color: colors.link }]}>{noteHeaderLabel}</Text>
            <Text style={[styles.noteText, { color: colors.textSecondary }]} numberOfLines={2}>
              {firstNote.text?.trim() || ' '}
            </Text>
            {extraNoteCount > 0 ? (
              <Text style={[styles.noteMore, { color: colors.textSecondary }]}>
                +{extraNoteCount} more in file
              </Text>
            ) : null}
          </View>
        ) : showPreview ? (
          <Text style={[styles.preview, { color: colors.textSecondary }]} numberOfLines={3}>
            {preview}
          </Text>
        ) : null}

        {schedFmt ? (
          <View style={styles.planningRow}>
            <Ionicons name="time-outline" size={14} color={colors.link} style={styles.planningIcon} />
            <Text style={[styles.planningLabel, { color: colors.textSecondary }]}>Scheduled</Text>
            <Text style={[styles.planningValue, { color: colors.link }]} numberOfLines={2}>
              {schedFmt}
            </Text>
          </View>
        ) : null}
        {deadFmt ? (
          <View style={styles.planningRow}>
            <Ionicons name="flag-outline" size={14} color={colors.danger} style={styles.planningIcon} />
            <Text style={[styles.planningLabel, { color: colors.textSecondary }]}>Deadline</Text>
            <Text
              style={[styles.planningValue, styles.planningDeadline, { color: colors.danger }]}
              numberOfLines={2}
            >
              {deadFmt}
            </Text>
          </View>
        ) : null}
        {createdFmt ? (
          <Text style={[styles.createdOnly, { color: colors.textSecondary }]} numberOfLines={2}>
            {createdFmt}
          </Text>
        ) : null}

        {tags.length > 0 ? (
          <View style={styles.tagsRow}>
            {tags.slice(0, 6).map((tag) => (
              <View
                key={tag}
                style={[
                  styles.tagChip,
                  { backgroundColor: colors.surfaceMuted, borderColor: colors.border },
                ]}
              >
                <Text style={[styles.tagChipText, { color: colors.textSecondary }]}>{tag}</Text>
              </View>
            ))}
            {tags.length > 6 ? (
              <Text style={[styles.tagOverflow, { color: colors.textSecondary }]}>
                +{tags.length - 6}
              </Text>
            ) : null}
          </View>
        ) : null}

        <Modal
          visible={stateModalOpen}
          animationType="slide"
          transparent
          onRequestClose={() => setStateModalOpen(false)}
        >
          <Pressable style={styles.modalBackdrop} onPress={() => setStateModalOpen(false)}>
            <Pressable
              style={[
                styles.sheet,
                {
                  backgroundColor: colors.surface,
                  borderTopLeftRadius: 16,
                  borderTopRightRadius: 16,
                },
              ]}
              onPress={(e) => e.stopPropagation()}
            >
              <Text
                style={[
                  styles.sheetTitle,
                  { color: colors.text, borderBottomColor: colors.border },
                ]}
              >
                Choose state
              </Text>
              <ScrollView style={styles.sheetList} keyboardShouldPersistTaps="handled">
                {modalTargets.map((row) => (
                  <Pressable
                    key={row.key}
                    style={[styles.sheetRow, { borderBottomColor: colors.border }]}
                    onPress={() => {
                      row.onPick();
                    }}
                  >
                    <Text style={[styles.sheetRowTitle, { color: colors.text }]}>{row.label}</Text>
                  </Pressable>
                ))}
              </ScrollView>
              <Pressable style={styles.sheetClose} onPress={() => setStateModalOpen(false)}>
                <Text style={[styles.sheetCloseText, { color: colors.link }]}>Cancel</Text>
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
    paddingLeft: 4,
  },
  card: {
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1,
    shadowOffset: { width: 0, height: 1 },
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
  stateChipPressed: { opacity: 0.82 },
  stateChipText: { fontSize: 11, fontWeight: '700', letterSpacing: 0.3 },
  priorityChip: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  priorityChipText: { fontSize: 11, fontWeight: '700' },
  filename: { flex: 1, fontSize: 12, textAlign: 'right' },
  parentCrumb: {
    fontSize: 12,
    marginBottom: 4,
    fontStyle: 'italic',
  },
  heading: { fontSize: 17, fontWeight: '600', marginBottom: 6, lineHeight: 22 },
  noteRow: { marginBottom: 8 },
  noteTimestamp: { fontSize: 12, fontWeight: '600', marginBottom: 2 },
  noteText: { fontSize: 14, lineHeight: 20, fontStyle: 'italic' },
  noteMore: { fontSize: 11, marginTop: 4 },
  preview: { fontSize: 14, lineHeight: 20, marginBottom: 8 },
  createdOnly: {
    fontSize: 13,
    marginBottom: 6,
    lineHeight: 18,
  },
  planningRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    marginBottom: 6,
    gap: 6,
  },
  planningIcon: { marginRight: 0 },
  planningLabel: { fontSize: 12, fontWeight: '700', width: 76 },
  planningValue: { flex: 1, fontSize: 13, lineHeight: 18, minWidth: 120 },
  planningDeadline: {},
  tagsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginBottom: 8 },
  tagChip: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 4,
    borderWidth: 1,
  },
  tagChipText: { fontSize: 12 },
  tagOverflow: { fontSize: 12, alignSelf: 'center' },
  modalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'flex-end',
  },
  sheet: {
    maxHeight: '60%',
    paddingBottom: 16,
  },
  sheetTitle: { fontSize: 18, fontWeight: '700', padding: 16, borderBottomWidth: 1 },
  sheetList: { maxHeight: 360 },
  sheetRow: { paddingVertical: 14, paddingHorizontal: 16, borderBottomWidth: 1 },
  sheetRowTitle: { fontSize: 16, fontWeight: '600' },
  sheetClose: { marginTop: 8, alignSelf: 'center', paddingVertical: 10, paddingHorizontal: 24 },
  sheetCloseText: { fontSize: 16, fontWeight: '600' },
});
