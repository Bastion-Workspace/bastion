import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Modal,
  Pressable,
  RefreshControl,
  StyleSheet,
  Text,
  View,
  useColorScheme,
  type ViewToken,
} from 'react-native';
import type { RssArticle, RssFeed } from '../../../src/api/rss';
import {
  getRssUnreadByFeed,
  listAllArticles,
  listFeedArticles,
  listRssFeeds,
  markAllFeedRead,
  markAllUserRead,
  markArticleRead,
  toggleArticleStar,
} from '../../../src/api/rss';
import { RSS_SOURCE_ALL, useRssPrefs } from '../../../src/hooks/useRssPrefs';
import { RssArticleReaderModal } from '../../../src/components/RssArticleReaderModal';
import { useModalSheetBottomPadding } from '../../../src/components/ScreenShell';
import {
  dequeuePendingMarkRead,
  enqueuePendingMarkRead,
  takeAllPendingMarkReads,
} from '../../../src/session/rssPendingReadsStore';
import {
  clearLastOpenRssArticleId,
  loadLastOpenRssArticleId,
  saveLastOpenRssArticleId,
} from '../../../src/session/lastRssArticleStore';
import { getColors } from '../../../src/theme/colors';

type FeedRow = RssFeed & { unread: number };

type ListFilter = 'unread' | 'all' | 'starred';

export default function RssUnifiedReaderScreen() {
  const colorScheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const colors = useMemo(() => getColors(colorScheme), [colorScheme]);
  const { source, setSource, autoMarkRead, hydrated, refreshFromStore } = useRssPrefs();
  const [feeds, setFeeds] = useState<FeedRow[]>([]);
  const [articles, setArticles] = useState<RssArticle[]>([]);
  const [listFilter, setListFilter] = useState<ListFilter>('unread');
  const [pickerOpen, setPickerOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [readerArticle, setReaderArticle] = useState<RssArticle | null>(null);
  const modalSheetBottomPad = useModalSheetBottomPadding(16);

  const autoMarkReadRef = useRef(autoMarkRead);
  autoMarkReadRef.current = autoMarkRead;
  const markingIds = useRef<Set<string>>(new Set());

  useEffect(() => {
    setListFilter(source === RSS_SOURCE_ALL ? 'unread' : 'all');
  }, [source]);

  const load = useCallback(async () => {
    if (!hydrated) return;
    await refreshFromStore();
    setError(null);
    try {
      const pendingIds = await takeAllPendingMarkReads();
      for (const aid of pendingIds) {
        try {
          await markArticleRead(aid);
        } catch {
          await enqueuePendingMarkRead(aid);
        }
      }

      const [feedList, unreadMap] = await Promise.all([listRssFeeds(), getRssUnreadByFeed()]);
      const rows = feedList.map((f) => ({
        ...f,
        unread: unreadMap[f.feed_id] ?? 0,
      }));
      setFeeds(rows);

      if (source !== RSS_SOURCE_ALL && !feedList.some((f) => f.feed_id === source)) {
        await setSource(RSS_SOURCE_ALL);
        return;
      }

      const apiReadFilter: 'all' | 'unread' | 'read' =
        listFilter === 'starred' ? 'all' : listFilter === 'unread' ? 'unread' : 'all';

      let raw: RssArticle[];
      if (source === RSS_SOURCE_ALL) {
        raw = await listAllArticles({ limit: 200, readFilter: apiReadFilter });
      } else {
        raw = await listFeedArticles(source, { limit: 200, readFilter: apiReadFilter });
      }
      const visible = listFilter === 'starred' ? raw.filter((a) => a.is_starred) : raw;
      setArticles(visible);

      const lastOpenId = await loadLastOpenRssArticleId();
      if (lastOpenId) {
        const found = visible.find((a) => a.article_id === lastOpenId);
        if (found) {
          setReaderArticle(found);
        } else {
          await clearLastOpenRssArticleId();
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load');
      setArticles([]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [hydrated, source, listFilter, setSource, refreshFromStore]);

  useFocusEffect(
    useCallback(() => {
      if (!hydrated) return;
      void load();
    }, [hydrated, load])
  );

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    void load();
  }, [load]);

  const sourceLabel = useMemo(() => {
    if (source === RSS_SOURCE_ALL) return 'All feeds';
    const f = feeds.find((x) => x.feed_id === source);
    return f?.feed_name ?? 'Feed';
  }, [source, feeds]);

  const openPicker = useCallback(() => setPickerOpen(true), []);
  const closePicker = useCallback(() => setPickerOpen(false), []);

  const selectSource = useCallback(
    async (next: string) => {
      closePicker();
      await setSource(next);
    },
    [setSource, closePicker]
  );

  const openArticle = useCallback(async (a: RssArticle) => {
    try {
      if (!a.is_read) {
        await enqueuePendingMarkRead(a.article_id);
        try {
          await markArticleRead(a.article_id);
          await dequeuePendingMarkRead(a.article_id);
          setArticles((prev) =>
            prev.map((x) => (x.article_id === a.article_id ? { ...x, is_read: true } : x))
          );
        } catch {
          /* pending queue will retry */
        }
      }
    } catch {
      /* still open reader */
    }
    await saveLastOpenRssArticleId(a.article_id);
    setReaderArticle(a);
  }, []);

  const showArticleActions = useCallback((a: RssArticle) => {
    const actions: {
      text: string;
      onPress?: () => void | Promise<void>;
      style?: 'default' | 'cancel' | 'destructive';
    }[] = [];
    if (!a.is_read) {
      actions.push({
        text: 'Mark read',
        onPress: async () => {
          try {
            await markArticleRead(a.article_id);
            setArticles((prev) =>
              prev.map((x) => (x.article_id === a.article_id ? { ...x, is_read: true } : x))
            );
          } catch {
            /* ignore */
          }
        },
      });
    }
    actions.push({
      text: a.is_starred ? 'Unstar' : 'Star',
      onPress: async () => {
        try {
          const starred = await toggleArticleStar(a.article_id);
          setArticles((prev) =>
            prev.map((x) => (x.article_id === a.article_id ? { ...x, is_starred: starred } : x))
          );
        } catch {
          /* ignore */
        }
      },
    });
    actions.push({ text: 'Cancel', style: 'cancel' });
    Alert.alert(a.title || 'Article', undefined, actions);
  }, []);

  const onMarkAllRead = useCallback(() => {
    const title = source === RSS_SOURCE_ALL ? 'Mark all read?' : 'Mark all read in this feed?';
    const message =
      source === RSS_SOURCE_ALL
        ? 'All unread articles across your feeds will be marked as read.'
        : 'All unread articles in this feed will be marked as read.';
    Alert.alert(title, message, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Mark all',
        onPress: async () => {
          try {
            if (source === RSS_SOURCE_ALL) {
              await markAllUserRead();
            } else {
              await markAllFeedRead(source);
            }
            setArticles((prev) => prev.map((x) => ({ ...x, is_read: true })));
          } catch {
            /* ignore */
          }
        },
      },
    ]);
  }, [source]);

  const onViewableItemsChanged = useRef(
    ({ changed }: { viewableItems: ViewToken[]; changed: ViewToken[] }) => {
      if (!autoMarkReadRef.current) return;
      for (const vt of changed ?? []) {
        if (vt.isViewable) continue;
        const a = vt.item as RssArticle | undefined;
        if (!a?.article_id || a.is_read) continue;
        if (markingIds.current.has(a.article_id)) continue;
        markingIds.current.add(a.article_id);
        void (async () => {
          await enqueuePendingMarkRead(a.article_id);
          try {
            await markArticleRead(a.article_id);
            await dequeuePendingMarkRead(a.article_id);
            setArticles((prev) =>
              prev.map((x) => (x.article_id === a.article_id ? { ...x, is_read: true } : x))
            );
          } catch {
            /* pending queue retries on next load */
          } finally {
            markingIds.current.delete(a.article_id);
          }
        })();
      }
    }
  ).current;

  const viewabilityConfig = useRef({
    itemVisiblePercentThreshold: 75,
    minimumViewTime: 600,
  }).current;

  const renderArticle = useCallback(
    ({ item: a }: { item: RssArticle }) => (
      <Pressable
        style={[
          styles.row,
          { borderBottomColor: colors.border },
          !a.is_read && { backgroundColor: colorScheme === 'dark' ? colors.surfaceMuted : '#f8f9ff' },
        ]}
        onPress={() => void openArticle(a)}
        onLongPress={() => showArticleActions(a)}
        accessibilityRole="button"
        accessibilityHint="Long press for star and mark read"
      >
        <View style={styles.rowText}>
          {source === RSS_SOURCE_ALL && (a.feed_name || a.feed_id) ? (
            <Text style={[styles.feedLine, { color: colors.link }]} numberOfLines={1}>
              {a.feed_name ?? a.feed_id}
            </Text>
          ) : null}
          <Text
            style={[styles.titleBase, { color: colors.textSecondary }, !a.is_read && { color: colors.text, fontWeight: '700' as const }]}
            numberOfLines={3}
          >
            {a.title || 'Untitled'}
          </Text>
          {a.published_date ? (
            <Text style={[styles.date, { color: colors.textSecondary }]} numberOfLines={1}>
              {String(a.published_date).slice(0, 19).replace('T', ' ')}
            </Text>
          ) : null}
        </View>
        {a.is_starred ? <Ionicons name="star" size={18} color="#f9a825" style={styles.star} /> : null}
      </Pressable>
    ),
    [openArticle, showArticleActions, source, colors, colorScheme]
  );

  const totalUnread = useMemo(() => feeds.reduce((n, f) => n + (f.unread > 0 ? f.unread : 0), 0), [feeds]);

  if (!hydrated) {
    return (
      <View style={[styles.centered, { backgroundColor: colors.background }]}>
        <ActivityIndicator size="large" color={colors.text} />
      </View>
    );
  }

  const listLoading = loading && !refreshing;

  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={styles.topBar}>
        <Pressable
          style={[
            styles.sourceChip,
            { backgroundColor: colors.surfaceMuted, borderColor: colors.border },
          ]}
          onPress={openPicker}
          onLongPress={openPicker}
          accessibilityRole="button"
          accessibilityLabel={`Source: ${sourceLabel}. Tap to change.`}
        >
          <Text style={[styles.sourceChipText, { color: colors.text }]} numberOfLines={1}>
            {sourceLabel}
          </Text>
          <Ionicons name="chevron-down" size={18} color={colors.text} />
        </Pressable>
      </View>

      <View style={styles.filterRow}>
        {(['unread', 'all', 'starred'] as const).map((key) => (
          <Pressable
            key={key}
            style={[
              styles.filterChip,
              { backgroundColor: colors.chipBg },
              listFilter === key && { backgroundColor: colors.chipBgActive },
            ]}
            onPress={() => setListFilter(key)}
          >
            <Text
              style={[
                styles.filterChipText,
                { color: colors.chipText },
                listFilter === key && { color: colors.chipTextActive },
              ]}
            >
              {key === 'unread' ? 'Unread' : key === 'all' ? 'All' : 'Starred'}
            </Text>
          </Pressable>
        ))}
      </View>

      <View style={[styles.toolbar, { borderBottomColor: colors.border }]}>
        <Pressable onPress={onMarkAllRead} style={styles.toolbarBtn} accessibilityRole="button">
          <Text style={[styles.toolbarBtnText, { color: colors.link }]}>Mark all read</Text>
        </Pressable>
      </View>

      {error ? (
        <View style={[styles.banner, { backgroundColor: colorScheme === 'dark' ? '#4a2c2c' : '#ffebee' }]}>
          <Text style={[styles.bannerText, { color: colorScheme === 'dark' ? '#ffcdd2' : '#b71c1c' }]}>{error}</Text>
          <Pressable onPress={() => void load()}>
            <Text style={[styles.retry, { color: colors.link }]}>Retry</Text>
          </Pressable>
        </View>
      ) : null}

      <FlatList
        data={articles}
        keyExtractor={(a) => a.article_id}
        renderItem={renderArticle}
        contentContainerStyle={[styles.list, listLoading && articles.length === 0 ? styles.listLoadingPad : null]}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        onViewableItemsChanged={onViewableItemsChanged}
        viewabilityConfig={viewabilityConfig}
        ListEmptyComponent={
          listLoading && articles.length === 0 ? (
            <View style={styles.listLoadingBlock}>
              <ActivityIndicator size="large" color={colors.text} />
              <Text style={[styles.listLoadingHint, { color: colors.textSecondary }]}>Loading articles…</Text>
            </View>
          ) : (
            <View style={styles.empty}>
              <Text style={[styles.emptyText, { color: colors.textSecondary }]}>
                {feeds.length === 0
                  ? 'No RSS feeds. Add feeds in the Bastion web app.'
                  : 'No articles match this filter.'}
              </Text>
            </View>
          )
        }
      />

      <RssArticleReaderModal
        visible={readerArticle !== null}
        article={readerArticle}
        hasPrev={
          readerArticle != null &&
          articles.findIndex((x) => x.article_id === readerArticle.article_id) > 0
        }
        hasNext={
          readerArticle != null &&
          articles.findIndex((x) => x.article_id === readerArticle.article_id) >= 0 &&
          articles.findIndex((x) => x.article_id === readerArticle.article_id) < articles.length - 1
        }
        onPrevArticle={() => {
          if (!readerArticle) return;
          const i = articles.findIndex((x) => x.article_id === readerArticle.article_id);
          if (i <= 0) return;
          const p = articles[i - 1];
          setReaderArticle(p);
          void saveLastOpenRssArticleId(p.article_id);
        }}
        onNextArticle={() => {
          if (!readerArticle) return;
          const i = articles.findIndex((x) => x.article_id === readerArticle.article_id);
          if (i < 0 || i >= articles.length - 1) return;
          const n = articles[i + 1];
          setReaderArticle(n);
          void saveLastOpenRssArticleId(n.article_id);
        }}
        onClose={() => {
          void clearLastOpenRssArticleId();
          setReaderArticle(null);
        }}
      />

      <Modal visible={pickerOpen} animationType="slide" transparent onRequestClose={closePicker}>
        <Pressable style={styles.modalBackdrop} onPress={closePicker}>
          <Pressable
            style={[styles.sheet, { backgroundColor: colors.background, paddingBottom: modalSheetBottomPad }]}
            onPress={(e) => e.stopPropagation()}
          >
            <Text style={[styles.sheetTitle, { color: colors.text, borderBottomColor: colors.border }]}>
              Choose source
            </Text>
            <FlatList
              data={[
                { feed_id: RSS_SOURCE_ALL, feed_name: 'All feeds', unread: totalUnread } as FeedRow,
                ...feeds,
              ]}
              keyExtractor={(item) => item.feed_id}
              style={styles.sheetList}
              renderItem={({ item }) => (
                <Pressable
                  style={[styles.sheetRow, { borderBottomColor: colors.border }]}
                  onPress={() => void selectSource(item.feed_id)}
                >
                  <Text style={[styles.sheetRowTitle, { color: colors.text }]} numberOfLines={2}>
                    {item.feed_id === RSS_SOURCE_ALL ? 'All feeds' : item.feed_name}
                  </Text>
                  {item.unread > 0 ? (
                    <View style={[styles.sheetBadge, { backgroundColor: colors.chipBgActive }]}>
                      <Text style={styles.sheetBadgeText}>{item.unread > 99 ? '99+' : item.unread}</Text>
                    </View>
                  ) : null}
                </Pressable>
              )}
            />
            <Pressable style={styles.sheetClose} onPress={closePicker}>
              <Text style={[styles.sheetCloseText, { color: colors.link }]}>Close</Text>
            </Pressable>
          </Pressable>
        </Pressable>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingTop: 8,
    gap: 10,
  },
  sourceChip: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 10,
    borderWidth: StyleSheet.hairlineWidth,
  },
  sourceChipText: { flex: 1, fontSize: 16, fontWeight: '700', marginRight: 8 },
  listLoadingPad: { flexGrow: 1 },
  listLoadingBlock: { paddingVertical: 48, alignItems: 'center', gap: 12 },
  listLoadingHint: { fontSize: 14 },
  filterRow: {
    flexDirection: 'row',
    gap: 8,
    paddingHorizontal: 12,
    paddingTop: 10,
    paddingBottom: 8,
  },
  filterChip: {
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 20,
  },
  filterChipText: { fontSize: 13, fontWeight: '600' },
  toolbar: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    paddingHorizontal: 12,
    paddingBottom: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  toolbarBtn: { paddingVertical: 8, paddingHorizontal: 12 },
  toolbarBtnText: { fontWeight: '700', fontSize: 14 },
  list: { paddingBottom: 32 },
  row: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
    gap: 8,
  },
  rowText: { flex: 1 },
  feedLine: { fontSize: 12, fontWeight: '600', marginBottom: 4 },
  titleBase: { fontSize: 15, lineHeight: 21 },
  date: { marginTop: 4, fontSize: 12 },
  star: { marginTop: 2 },
  banner: {
    margin: 12,
    padding: 10,
    borderRadius: 8,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 8,
  },
  bannerText: { flex: 1, fontSize: 13 },
  retry: { fontWeight: '700' },
  empty: { padding: 32, alignItems: 'center' },
  emptyText: { fontSize: 15, textAlign: 'center' },
  modalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'flex-end',
  },
  sheet: {
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    maxHeight: '55%',
  },
  sheetTitle: { fontSize: 18, fontWeight: '700', padding: 16, borderBottomWidth: 1 },
  sheetList: { maxHeight: 320 },
  sheetRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
    gap: 12,
  },
  sheetRowTitle: { flex: 1, fontSize: 16, fontWeight: '600' },
  sheetBadge: {
    minWidth: 26,
    height: 26,
    paddingHorizontal: 8,
    borderRadius: 13,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sheetBadgeText: { color: '#fff', fontSize: 12, fontWeight: '700' },
  sheetClose: { marginTop: 8, alignSelf: 'center', paddingVertical: 10, paddingHorizontal: 24 },
  sheetCloseText: { fontSize: 16, fontWeight: '600' },
});
