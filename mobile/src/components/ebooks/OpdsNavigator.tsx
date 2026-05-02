import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
  useColorScheme,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import {
  fetchOpdsAtom,
  type OpdsCatalogEntryResponse,
  type OpdsFeed,
  type OpdsFeedEntry,
} from '../../api/ebooks';
import { getColors } from '../../theme/colors';
import { BookCard } from './BookCard';
import { OpdsCatalogPicker } from './OpdsCatalogPicker';
import { pickAcquisitionFormatFromEntry, pickAcquisitionHrefFromEntry, pickNavigationHrefFromEntry } from './opdsUtils';

type Props = {
  catalogs: OpdsCatalogEntryResponse[];
  onOpenEbook: (args: { catalogId: string; acquisitionUrl: string; title: string; format: 'epub' | 'pdf' }) => void;
};

export function OpdsNavigator({ catalogs, onOpenEbook }: Props) {
  const scheme = useColorScheme();
  const c = useMemo(() => getColors(scheme === 'dark' ? 'dark' : 'light'), [scheme]);
  const [catalogId, setCatalogId] = useState('');
  const [tab, setTab] = useState<'browse' | 'search'>('browse');
  const [navStack, setNavStack] = useState<string[]>([]);
  const [feed, setFeed] = useState<OpdsFeed | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    if (!catalogId && catalogs.length > 0) {
      setCatalogId(catalogs[0].id);
    }
  }, [catalogId, catalogs]);

  const activeCatalog = useMemo(
    () => catalogs.find((x) => x.id === catalogId) || null,
    [catalogs, catalogId]
  );

  const fetchFeed = useCallback(
    async (url: string) => {
      if (!catalogId || !url) return;
      setLoading(true);
      setError('');
      try {
        const res = await fetchOpdsAtom({ catalog_id: catalogId, url, want: 'atom' });
        setFeed(res.feed || null);
      } catch (e) {
        setFeed(null);
        setError(e instanceof Error ? e.message : 'Fetch failed');
      } finally {
        setLoading(false);
      }
    },
    [catalogId]
  );

  useEffect(() => {
    if (tab !== 'browse' || !activeCatalog?.root_url) return;
    const root = activeCatalog.root_url;
    setNavStack([root]);
    void fetchFeed(root);
  }, [tab, catalogId, activeCatalog?.root_url, fetchFeed]);

  const currentBase = navStack.length > 0 ? navStack[navStack.length - 1] : activeCatalog?.root_url || '';

  const onSearch = useCallback(() => {
    const tmpl = feed?.search_template;
    if (!tmpl || !searchQuery.trim()) return;
    let u = tmpl;
    if (u.includes('{searchTerms}')) {
      u = u.split('{searchTerms}').join(encodeURIComponent(searchQuery.trim()));
    } else {
      u = u.replace(/\{[qQ]\}/g, encodeURIComponent(searchQuery.trim()));
    }
    setNavStack([u]);
    setTab('browse');
    void fetchFeed(u);
  }, [feed?.search_template, searchQuery, fetchFeed]);

  const onBack = useCallback(() => {
    setNavStack((s) => {
      if (s.length <= 1) return s;
      const next = s.slice(0, -1);
      const url = next[next.length - 1];
      void fetchFeed(url);
      return next;
    });
  }, [fetchFeed]);

  const entries = feed?.entries || [];

  const renderEntry = useCallback(
    ({ item }: { item: OpdsFeedEntry }) => {
      const acq = pickAcquisitionHrefFromEntry(item);
      const acqFormat = pickAcquisitionFormatFromEntry(item);
      const navHref = pickNavigationHrefFromEntry(item);
      return (
        <BookCard
          entry={item}
          baseUrl={currentBase}
          scheme={scheme}
          acquisitionFormat={acq ? acqFormat : undefined}
          onPressAcquisition={
            acq
              ? () => {
                  const abs = new URL(acq, currentBase).href;
                  onOpenEbook({
                    catalogId,
                    acquisitionUrl: abs,
                    title: item.title || 'Book',
                    format: acqFormat,
                  });
                }
              : undefined
          }
          onPressNavigation={
            navHref
              ? () => {
                  const next = new URL(navHref, currentBase).href;
                  setNavStack((s) => [...s, next]);
                  void fetchFeed(next);
                }
              : undefined
          }
        />
      );
    },
    [catalogId, currentBase, fetchFeed, onOpenEbook, scheme]
  );

  return (
    <View style={styles.wrap}>
      <OpdsCatalogPicker catalogs={catalogs} selectedId={catalogId} onSelect={setCatalogId} scheme={scheme} />

      <View style={styles.tabs}>
        <Pressable
          style={[styles.tab, tab === 'browse' && { borderBottomColor: c.link }]}
          onPress={() => setTab('browse')}
        >
          <Text style={[styles.tabText, { color: tab === 'browse' ? c.link : c.textSecondary }]}>Browse</Text>
        </Pressable>
        <Pressable
          style={[styles.tab, tab === 'search' && { borderBottomColor: c.link }]}
          onPress={() => setTab('search')}
        >
          <Text style={[styles.tabText, { color: tab === 'search' ? c.link : c.textSecondary }]}>Search</Text>
        </Pressable>
      </View>

      {tab === 'search' ? (
        <View style={styles.searchBox}>
          <TextInput
            style={[styles.input, { borderColor: c.border, color: c.text, backgroundColor: c.surface }]}
            placeholder="Search query"
            placeholderTextColor={c.textSecondary}
            value={searchQuery}
            onChangeText={setSearchQuery}
            onSubmitEditing={onSearch}
          />
          <Pressable
            style={[styles.searchBtn, { backgroundColor: c.chipBgActive }]}
            onPress={onSearch}
            disabled={!feed?.search_template}
          >
            <Text style={[styles.searchBtnText, { color: c.chipTextActive }]}>Search</Text>
          </Pressable>
          {!feed?.search_template ? (
            <Text style={[styles.help, { color: c.textSecondary }]}>Open Browse on a feed that exposes search.</Text>
          ) : null}
        </View>
      ) : null}

      {tab === 'browse' ? (
        <View style={styles.browseHeader}>
          {navStack.length > 1 ? (
            <Pressable style={styles.backRow} onPress={onBack} accessibilityRole="button">
              <Ionicons name="arrow-back" size={22} color={c.text} />
              <Text style={[styles.backText, { color: c.link }]}>Back</Text>
            </Pressable>
          ) : (
            <Text style={[styles.feedTitle, { color: c.textSecondary }]} numberOfLines={1}>
              {feed?.feed_title || 'Catalog'}
            </Text>
          )}
        </View>
      ) : null}

      {loading ? <ActivityIndicator color={c.text} style={styles.spinner} /> : null}
      {error ? <Text style={[styles.err, { color: c.danger }]}>{error}</Text> : null}

      {tab === 'browse' && !loading ? (
        <FlatList
          style={styles.entryList}
          data={entries}
          keyExtractor={(en, i) => String(en.id || en.title || i)}
          renderItem={renderEntry}
          contentContainerStyle={styles.listPad}
          nestedScrollEnabled
        />
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, minHeight: 200 },
  entryList: { flex: 1 },
  tabs: { flexDirection: 'row', marginTop: 8, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#8882' },
  tab: { flex: 1, paddingVertical: 10, alignItems: 'center', borderBottomWidth: 2, borderBottomColor: 'transparent' },
  tabText: { fontSize: 15, fontWeight: '600' },
  searchBox: { paddingVertical: 12, gap: 8 },
  input: { borderWidth: StyleSheet.hairlineWidth, borderRadius: 8, paddingHorizontal: 12, paddingVertical: 10 },
  searchBtn: { alignSelf: 'flex-start', paddingHorizontal: 16, paddingVertical: 10, borderRadius: 8 },
  searchBtnText: { fontWeight: '700' },
  help: { fontSize: 12 },
  browseHeader: { paddingVertical: 8 },
  backRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  backText: { fontSize: 16, fontWeight: '600' },
  feedTitle: { fontSize: 14 },
  spinner: { marginVertical: 16 },
  err: { marginBottom: 8 },
  listPad: { paddingBottom: 24 },
});
