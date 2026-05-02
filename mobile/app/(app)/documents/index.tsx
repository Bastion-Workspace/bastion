import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Modal,
  Platform,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  useColorScheme,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { DocumentSearchModal } from '../../../src/components/DocumentSearchModal';
import { ScreenShell } from '../../../src/components/ScreenShell';
import { listUserDocuments, type DocumentInfo } from '../../../src/api/documents';
import {
  getFolderContents,
  getFolderTree,
  type DocumentFolderNode,
  type FolderContentsApiResponse,
  type FolderDocumentRow,
  type FolderTreeApiResponse,
} from '../../../src/api/folders';
import { loadRecentDocuments, type RecentDocumentEntry } from '../../../src/session/recentDocumentsStore';
import { getColors, type AppColors } from '../../../src/theme/colors';

type ScopeFilter = 'all' | 'user' | 'team' | 'global';

type StackEntry =
  | { kind: 'roots' }
  | { kind: 'folder'; folderId: string; title: string };

function scopeLabel(s: ScopeFilter): string {
  switch (s) {
    case 'user':
      return 'My';
    case 'team':
      return 'Team';
    case 'global':
      return 'Global';
    default:
      return 'All';
  }
}

function filterRoots(roots: DocumentFolderNode[], scope: ScopeFilter): DocumentFolderNode[] {
  if (scope === 'all') return roots;
  return roots.filter((f) => (f.collection_type || 'user') === scope);
}

function makeListStyles(colors: AppColors) {
  return StyleSheet.create({
    center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
    list: { padding: 16, paddingTop: 8 },
    listHeader: { marginBottom: 12 },
    titleRow: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      marginBottom: 8,
    },
    screenTitle: { fontSize: 22, fontWeight: '700', color: colors.text },
    browseIconBtn: { padding: 6 },
    errorBanner: {
      backgroundColor: colors.surfaceMuted,
      color: colors.danger,
      padding: 12,
      borderRadius: 8,
      fontSize: 14,
    },
    empty: { textAlign: 'center', marginTop: 48, color: colors.textSecondary },
    recentSection: { marginBottom: 16 },
    recentLabel: { fontSize: 13, fontWeight: '700', color: colors.textSecondary, marginBottom: 8 },
    recentScroll: { flexGrow: 0 },
    recentScrollInner: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingRight: 8 },
    recentChip: {
      maxWidth: 200,
      paddingVertical: 8,
      paddingHorizontal: 12,
      borderRadius: 20,
      borderWidth: 1,
      borderColor: colors.border,
      backgroundColor: colors.surfaceMuted,
    },
    recentChipText: { fontSize: 14, fontWeight: '600', color: colors.text },
    headerIcons: { flexDirection: 'row', alignItems: 'center', gap: 4 },
    row: {
      backgroundColor: colors.surface,
      padding: 14,
      borderRadius: 8,
      marginBottom: 10,
      borderWidth: 1,
      borderColor: colors.border,
    },
    title: { fontSize: 16, fontWeight: '600', color: colors.text },
    sub: { fontSize: 13, color: colors.textSecondary, marginTop: 4 },
  });
}

function makeModalStyles(colors: AppColors) {
  return StyleSheet.create({
    modalSafe: { flex: 1, backgroundColor: colors.background },
    modalHeader: {
      flexDirection: 'row',
      alignItems: 'center',
      paddingHorizontal: 8,
      paddingVertical: 10,
      borderBottomWidth: StyleSheet.hairlineWidth,
      borderBottomColor: colors.border,
    },
    modalBackBtn: { padding: 4, width: 44 },
    modalTitle: { flex: 1, fontSize: 17, fontWeight: '700', color: colors.text, textAlign: 'center' },
    scopeScroll: { maxHeight: 48, marginBottom: 8 },
    scopeScrollInner: { paddingHorizontal: 12, paddingVertical: 8, flexDirection: 'row', alignItems: 'center' },
    scopeChip: {
      paddingHorizontal: 14,
      paddingVertical: 8,
      borderRadius: 20,
      backgroundColor: colors.chipBg,
      marginRight: 8,
    },
    scopeChipOn: { backgroundColor: colors.chipBgActive },
    scopeChipText: { fontSize: 14, fontWeight: '600', color: colors.chipText },
    scopeChipTextOn: { color: colors.chipTextActive },
    modalError: {
      marginHorizontal: 16,
      marginBottom: 8,
      padding: 12,
      backgroundColor: colors.surfaceMuted,
      color: colors.danger,
      borderRadius: 8,
      fontSize: 14,
    },
    modalCenter: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
    modalList: { paddingHorizontal: 12, paddingBottom: 24 },
    modalEmpty: { textAlign: 'center', marginTop: 32, color: colors.textSecondary, paddingHorizontal: 24 },
    folderRow: {
      flexDirection: 'row',
      alignItems: 'center',
      paddingVertical: 14,
      paddingHorizontal: 12,
      borderBottomWidth: StyleSheet.hairlineWidth,
      borderBottomColor: colors.border,
      backgroundColor: colors.surface,
      marginBottom: 4,
      borderRadius: 8,
    },
    docRow: {
      flexDirection: 'row',
      alignItems: 'center',
      paddingVertical: 12,
      paddingHorizontal: 12,
      borderBottomWidth: StyleSheet.hairlineWidth,
      borderBottomColor: colors.border,
    },
    folderRowIcon: { marginRight: 12 },
    folderRowText: { flex: 1 },
    folderRowTitle: { fontSize: 16, fontWeight: '600', color: colors.text },
    folderRowMeta: { fontSize: 12, color: colors.textSecondary, marginTop: 4 },
    docRowTitle: { fontSize: 15, fontWeight: '600', color: colors.text },
  });
}

function FolderBrowseModal({
  visible,
  onClose,
  onOpenDocument,
}: {
  visible: boolean;
  onClose: () => void;
  onOpenDocument: (documentId: string, title: string) => void;
}) {
  const scheme = useColorScheme();
  const modalColors = useMemo(() => getColors(scheme === 'dark' ? 'dark' : 'light'), [scheme]);
  const styles = useMemo(() => makeModalStyles(modalColors), [modalColors]);

  const [scope, setScope] = useState<ScopeFilter>('all');
  const [stack, setStack] = useState<StackEntry[]>([{ kind: 'roots' }]);
  const [tree, setTree] = useState<FolderTreeApiResponse | null>(null);
  const [treeError, setTreeError] = useState<string | null>(null);
  const [treeLoading, setTreeLoading] = useState(false);
  const [contents, setContents] = useState<FolderContentsApiResponse | null>(null);
  const [contentsLoading, setContentsLoading] = useState(false);
  const [contentsError, setContentsError] = useState<string | null>(null);

  const top = stack[stack.length - 1];
  const atRoots = top.kind === 'roots';

  const loadTree = useCallback(async () => {
    setTreeError(null);
    setTreeLoading(true);
    try {
      const res = await getFolderTree();
      setTree(res);
    } catch (e) {
      setTreeError(e instanceof Error ? e.message : 'Failed to load folders');
      setTree(null);
    } finally {
      setTreeLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!visible) return;
    void loadTree();
  }, [visible, loadTree]);

  const openFolderId = visible && !atRoots && top.kind === 'folder' ? top.folderId : null;

  useEffect(() => {
    if (!visible) {
      setStack([{ kind: 'roots' }]);
      setScope('all');
      setTree(null);
      setTreeError(null);
      setContents(null);
      setContentsError(null);
      setContentsLoading(false);
      return;
    }
    if (openFolderId == null) {
      setContents(null);
      setContentsError(null);
      setContentsLoading(false);
      return;
    }
    let cancelled = false;
    setContentsLoading(true);
    setContentsError(null);
    void (async () => {
      try {
        const c = await getFolderContents(openFolderId);
        if (!cancelled) setContents(c);
      } catch (e) {
        if (!cancelled) {
          setContentsError(e instanceof Error ? e.message : 'Failed to load folder');
          setContents(null);
        }
      } finally {
        if (!cancelled) setContentsLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [visible, openFolderId]);

  const filteredRoots = useMemo(
    () => filterRoots(tree?.folders ?? [], scope),
    [tree, scope]
  );

  type FolderListRow = { type: 'subfolder'; sf: DocumentFolderNode } | { type: 'doc'; d: FolderDocumentRow };

  const folderContentsRows = useMemo((): FolderListRow[] => {
    if (!contents) return [];
    return [
      ...(contents.subfolders ?? []).map((sf) => ({ type: 'subfolder' as const, sf })),
      ...(contents.documents ?? []).map((d) => ({ type: 'doc' as const, d })),
    ];
  }, [contents]);

  function closeAndReset() {
    setStack([{ kind: 'roots' }]);
    setScope('all');
    setContents(null);
    setTreeError(null);
    setContentsError(null);
    onClose();
  }

  function goBack() {
    if (stack.length <= 1) {
      closeAndReset();
      return;
    }
    setStack((s) => s.slice(0, -1));
  }

  const scopes: ScopeFilter[] = ['all', 'user', 'team', 'global'];

  return (
    <Modal
      visible={visible}
      animationType="slide"
      presentationStyle={Platform.OS === 'ios' ? 'pageSheet' : undefined}
      onRequestClose={goBack}
    >
      <SafeAreaView style={styles.modalSafe} edges={['top', 'left', 'right']}>
        <View style={styles.modalHeader}>
          <Pressable onPress={goBack} hitSlop={12} style={styles.modalBackBtn} accessibilityRole="button">
            <Ionicons name="chevron-back" size={28} color={modalColors.text} />
          </Pressable>
          <Text style={styles.modalTitle} numberOfLines={1}>
            {atRoots ? 'Browse folders' : top.kind === 'folder' ? top.title : 'Browse'}
          </Text>
          <Pressable onPress={closeAndReset} hitSlop={12} accessibilityRole="button" accessibilityLabel="Close">
            <Ionicons name="close" size={26} color={modalColors.text} />
          </Pressable>
        </View>

        {atRoots ? (
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            style={styles.scopeScroll}
            contentContainerStyle={styles.scopeScrollInner}
            keyboardShouldPersistTaps="handled"
          >
            {scopes.map((s) => (
              <Pressable
                key={s}
                style={[styles.scopeChip, scope === s && styles.scopeChipOn]}
                onPress={() => setScope(s)}
              >
                <Text style={[styles.scopeChipText, scope === s && styles.scopeChipTextOn]}>{scopeLabel(s)}</Text>
              </Pressable>
            ))}
          </ScrollView>
        ) : null}

        {treeError ? (
          <Text style={styles.modalError} accessibilityRole="alert">
            {treeError}
          </Text>
        ) : null}

        {atRoots ? (
            treeLoading ? (
            <View style={styles.modalCenter}>
              <ActivityIndicator size="large" color={modalColors.text} />
            </View>
          ) : (
            <FlatList
              data={filteredRoots}
              keyExtractor={(item) => item.folder_id}
              contentContainerStyle={styles.modalList}
              ListEmptyComponent={
                <Text style={styles.modalEmpty}>No folders in this scope.</Text>
              }
              renderItem={({ item }) => (
                <Pressable
                  style={styles.folderRow}
                  onPress={() =>
                    setStack((prev) => [...prev, { kind: 'folder', folderId: item.folder_id, title: item.name }])
                  }
                >
                  <Ionicons name="folder-outline" size={22} color={modalColors.link} style={styles.folderRowIcon} />
                  <View style={styles.folderRowText}>
                    <Text style={styles.folderRowTitle} numberOfLines={2}>
                      {item.name}
                    </Text>
                    {item.document_count != null && item.document_count > 0 ? (
                      <Text style={styles.folderRowMeta}>{item.document_count} document(s)</Text>
                    ) : null}
                  </View>
                  <Ionicons name="chevron-forward" size={20} color={modalColors.textSecondary} />
                </Pressable>
              )}
            />
          )
        ) : contentsLoading ? (
          <View style={styles.modalCenter}>
            <ActivityIndicator size="large" color={modalColors.text} />
          </View>
        ) : contentsError ? (
          <Text style={styles.modalError}>{contentsError}</Text>
        ) : (
          <FlatList
            data={folderContentsRows}
            keyExtractor={(item) =>
              item.type === 'subfolder' ? `sf-${item.sf.folder_id}` : `doc-${item.d.document_id}`
            }
            contentContainerStyle={styles.modalList}
            ListEmptyComponent={<Text style={styles.modalEmpty}>This folder is empty.</Text>}
            renderItem={({ item }) =>
              item.type === 'subfolder' ? (
                <Pressable
                  style={styles.folderRow}
                  onPress={() =>
                    setStack((prev) => [
                      ...prev,
                      { kind: 'folder', folderId: item.sf.folder_id, title: item.sf.name },
                    ])
                  }
                >
                  <Ionicons name="folder-outline" size={22} color={modalColors.link} style={styles.folderRowIcon} />
                  <Text style={styles.folderRowTitle}>{item.sf.name}</Text>
                  <Ionicons name="chevron-forward" size={20} color={modalColors.textSecondary} />
                </Pressable>
              ) : (
                <Pressable
                  style={styles.docRow}
                  onPress={() => {
                    const title = item.d.title || item.d.filename || 'Document';
                    onOpenDocument(item.d.document_id, title);
                    closeAndReset();
                  }}
                >
                  <Ionicons name="document-text-outline" size={22} color={modalColors.text} style={styles.folderRowIcon} />
                  <View style={styles.folderRowText}>
                    <Text style={styles.docRowTitle} numberOfLines={2}>
                      {item.d.title || item.d.filename}
                    </Text>
                    <Text style={styles.folderRowMeta} numberOfLines={1}>
                      {item.d.filename}
                    </Text>
                  </View>
                </Pressable>
              )
            }
          />
        )}
      </SafeAreaView>
    </Modal>
  );
}

export default function DocumentsListScreen() {
  const router = useRouter();
  const scheme = useColorScheme();
  const colors = useMemo(() => getColors(scheme === 'dark' ? 'dark' : 'light'), [scheme]);
  const styles = useMemo(() => makeListStyles(colors), [colors]);

  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [browseOpen, setBrowseOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [recentDocs, setRecentDocs] = useState<RecentDocumentEntry[]>([]);

  const refreshRecents = useCallback(async () => {
    const list = await loadRecentDocuments();
    setRecentDocs(list.slice(0, 6));
  }, []);

  useFocusEffect(
    useCallback(() => {
      void refreshRecents();
    }, [refreshRecents])
  );

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
      <ScreenShell>
        <View style={styles.center}>
          <ActivityIndicator size="large" color={colors.text} />
        </View>
      </ScreenShell>
    );
  }

  return (
    <ScreenShell>
      <FlatList
        data={docs}
        keyExtractor={(item) => item.document_id}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        contentContainerStyle={styles.list}
        ListHeaderComponent={
          <View style={styles.listHeader}>
            {recentDocs.length > 0 ? (
              <View style={styles.recentSection}>
                <Text style={styles.recentLabel}>Recently opened</Text>
                <ScrollView
                  horizontal
                  showsHorizontalScrollIndicator={false}
                  style={styles.recentScroll}
                  contentContainerStyle={styles.recentScrollInner}
                  keyboardShouldPersistTaps="handled"
                >
                  {recentDocs.map((r) => (
                    <Pressable
                      key={r.document_id}
                      style={styles.recentChip}
                      onPress={() =>
                        router.push({
                          pathname: `/documents/${r.document_id}`,
                          params: { documentTitle: r.title },
                        })
                      }
                      accessibilityRole="button"
                      accessibilityLabel={`Open ${r.title}`}
                    >
                      <Text style={styles.recentChipText} numberOfLines={1}>
                        {r.title}
                      </Text>
                    </Pressable>
                  ))}
                </ScrollView>
              </View>
            ) : null}
            <View style={styles.titleRow}>
              <Text style={styles.screenTitle}>Documents</Text>
              <View style={styles.headerIcons}>
                <Pressable
                  onPress={() => setSearchOpen(true)}
                  hitSlop={10}
                  style={styles.browseIconBtn}
                  accessibilityRole="button"
                  accessibilityLabel="Search documents"
                >
                  <Ionicons name="search-outline" size={26} color={colors.text} />
                </Pressable>
                <Pressable
                  onPress={() => setBrowseOpen(true)}
                  hitSlop={10}
                  style={styles.browseIconBtn}
                  accessibilityRole="button"
                  accessibilityLabel="Browse folder tree"
                >
                  <Ionicons name="folder-open-outline" size={26} color={colors.text} />
                </Pressable>
              </View>
            </View>
            {error ? (
              <Text style={styles.errorBanner} accessibilityRole="alert">
                {error}
              </Text>
            ) : null}
          </View>
        }
        ListEmptyComponent={<Text style={styles.empty}>No documents.</Text>}
        renderItem={({ item }) => (
          <Pressable
            style={styles.row}
            onPress={() =>
              router.push({
                pathname: `/documents/${item.document_id}`,
                params: { documentTitle: item.title || item.filename || 'Document' },
              })
            }
          >
            <Text style={styles.title}>{item.title || item.filename}</Text>
            <Text style={styles.sub}>{item.filename}</Text>
          </Pressable>
        )}
      />
      <FolderBrowseModal
        visible={browseOpen}
        onClose={() => setBrowseOpen(false)}
        onOpenDocument={(documentId, documentTitle) => {
          router.push({
            pathname: `/documents/${documentId}`,
            params: { documentTitle },
          });
        }}
      />
      <DocumentSearchModal
        visible={searchOpen}
        onClose={() => setSearchOpen(false)}
        colors={colors}
        onPickDocument={(documentId, documentTitle) => {
          router.push({
            pathname: `/documents/${documentId}`,
            params: { documentTitle },
          });
        }}
      />
    </ScreenShell>
  );
}
