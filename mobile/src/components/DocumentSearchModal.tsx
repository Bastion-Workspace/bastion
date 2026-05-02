import { Ionicons } from '@expo/vector-icons';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Modal,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { searchDocuments, type DocumentSearchResultRow } from '../api/documents';
import type { AppColors } from '../theme/colors';
import { useModalSheetBottomPadding } from './ScreenShell';

type Props = {
  visible: boolean;
  onClose: () => void;
  onPickDocument: (documentId: string, title: string) => void;
  colors: AppColors;
};

function normalizeSearchRow(r: DocumentSearchResultRow): {
  document_id: string;
  title: string;
  subtitle: string;
} | null {
  const documentId = (r.document_id || r.document?.document_id || '').trim();
  if (!documentId) return null;
  const fn = (r.document?.filename || '').trim();
  const tit = (r.document?.title != null ? String(r.document.title).trim() : '') || fn || 'Document';
  const ctx = r.context && typeof r.context === 'object' && typeof r.context.text === 'string' ? r.context.text : '';
  const raw = ctx || (typeof r.text === 'string' ? r.text : '');
  const subtitle = raw.replace(/\s+/g, ' ').trim().slice(0, 180);
  return { document_id: documentId, title: tit, subtitle };
}

export function DocumentSearchModal({ visible, onClose, onPickDocument, colors }: Props) {
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const modalSheetBottomPad = useModalSheetBottomPadding(16);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<
    { document_id: string; title: string; subtitle: string; key: string }[]
  >([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const runSearch = useCallback(async (q: string) => {
    const trimmed = q.trim();
    if (!trimmed) {
      setResults([]);
      setError(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await searchDocuments(trimmed, 30);
      const seen = new Set<string>();
      const rows: { document_id: string; title: string; subtitle: string; key: string }[] = [];
      for (const r of res.results ?? []) {
        const n = normalizeSearchRow(r);
        if (!n || seen.has(n.document_id)) continue;
        seen.add(n.document_id);
        rows.push({ ...n, key: n.document_id });
      }
      setResults(rows);
    } catch (e) {
      setResults([]);
      setError(e instanceof Error ? e.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!visible) {
      setQuery('');
      setResults([]);
      setError(null);
      setLoading(false);
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
        debounceRef.current = null;
      }
    }
  }, [visible]);

  useEffect(() => {
    if (!visible) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    const q = query.trim();
    if (!q) {
      setResults([]);
      setError(null);
      setLoading(false);
      return;
    }
    debounceRef.current = setTimeout(() => {
      debounceRef.current = null;
      void runSearch(q);
    }, 400);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query, visible, runSearch]);

  return (
    <Modal
      visible={visible}
      animationType="slide"
      presentationStyle={Platform.OS === 'ios' ? 'pageSheet' : undefined}
      onRequestClose={onClose}
    >
      <SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]} edges={['top', 'left', 'right']}>
        <View style={[styles.header, { borderBottomColor: colors.border }]}>
          <Text style={[styles.headerTitle, { color: colors.text }]}>Search documents</Text>
          <Pressable onPress={onClose} hitSlop={12} accessibilityRole="button" accessibilityLabel="Close search">
            <Ionicons name="close" size={28} color={colors.text} />
          </Pressable>
        </View>
        <View style={[styles.searchRow, { borderBottomColor: colors.border }]}>
          <Ionicons name="search-outline" size={22} color={colors.textSecondary} style={styles.searchIcon} />
          <TextInput
            style={[styles.input, { color: colors.text, borderColor: colors.border, backgroundColor: colors.surface }]}
            placeholder="Search across your documents…"
            placeholderTextColor={colors.textSecondary}
            value={query}
            onChangeText={setQuery}
            returnKeyType="search"
            onSubmitEditing={() => void runSearch(query)}
            autoCapitalize="none"
            autoCorrect={false}
            clearButtonMode="while-editing"
          />
        </View>
        {error ? (
          <Text style={[styles.error, { color: colors.danger }]} accessibilityRole="alert">
            {error}
          </Text>
        ) : null}
        {loading && results.length === 0 && query.trim() ? (
          <View style={styles.center}>
            <ActivityIndicator size="large" color={colors.text} />
          </View>
        ) : (
          <FlatList
            data={results}
            keyExtractor={(item) => item.key}
            contentContainerStyle={[styles.list, { paddingBottom: modalSheetBottomPad }]}
            keyboardShouldPersistTaps="handled"
            ListEmptyComponent={
              query.trim() ? (
                loading ? null : (
                  <Text style={[styles.empty, { color: colors.textSecondary }]}>No results.</Text>
                )
              ) : (
                <Text style={[styles.empty, { color: colors.textSecondary }]}>
                  Type a few words to search titles and content.
                </Text>
              )
            }
            renderItem={({ item }) => (
              <Pressable
                style={[styles.row, { borderColor: colors.border, backgroundColor: colors.surface }]}
                onPress={() => {
                  onPickDocument(item.document_id, item.title);
                  onClose();
                }}
              >
                <Text style={[styles.rowTitle, { color: colors.text }]} numberOfLines={2}>
                  {item.title}
                </Text>
                {item.subtitle ? (
                  <Text style={[styles.rowSub, { color: colors.textSecondary }]} numberOfLines={3}>
                    {item.subtitle}
                  </Text>
                ) : null}
              </Pressable>
            )}
          />
        )}
      </SafeAreaView>
    </Modal>
  );
}

function makeStyles(colors: AppColors) {
  return StyleSheet.create({
    safe: { flex: 1 },
    header: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      paddingHorizontal: 12,
      paddingVertical: 10,
      borderBottomWidth: StyleSheet.hairlineWidth,
    },
    headerTitle: { fontSize: 18, fontWeight: '700' },
    searchRow: {
      flexDirection: 'row',
      alignItems: 'center',
      paddingHorizontal: 12,
      paddingVertical: 10,
      gap: 8,
      borderBottomWidth: StyleSheet.hairlineWidth,
    },
    searchIcon: { marginTop: 2 },
    input: {
      flex: 1,
      minHeight: 44,
      borderWidth: 1,
      borderRadius: 8,
      paddingHorizontal: 12,
      paddingVertical: Platform.OS === 'ios' ? 10 : 8,
      fontSize: 16,
    },
    error: { padding: 12, fontSize: 14 },
    center: { paddingTop: 32, alignItems: 'center' },
    list: { padding: 12, flexGrow: 1 },
    empty: { textAlign: 'center', marginTop: 32, paddingHorizontal: 24, fontSize: 15 },
    row: {
      padding: 14,
      borderRadius: 8,
      marginBottom: 10,
      borderWidth: 1,
    },
    rowTitle: { fontSize: 16, fontWeight: '600' },
    rowSub: { fontSize: 13, marginTop: 6, lineHeight: 18 },
  });
}
