import { useCallback, useEffect, useLayoutEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Dimensions,
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  View,
  useColorScheme,
} from 'react-native';
import { Image } from 'expo-image';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useNavigation } from '@react-navigation/native';
import { buildEmbyImageUrl, getEmbyItems, type EmbyItem } from '../../../src/api/emby';
import { assertApiBaseUrl } from '../../../src/api/config';
import { navigateFromEmbyItem } from '../../../src/media/embyItemNavigation';
import { getStoredToken } from '../../../src/session/tokenStore';
import { getColors } from '../../../src/theme/colors';

function itemTypesForLibrary(collectionType: string): string | undefined {
  const c = (collectionType || '').toLowerCase();
  if (c === 'movies' || c === 'movie') return 'Movie';
  if (c === 'tvshows' || c === 'tv') return 'Series';
  if (c === 'music') return 'MusicAlbum,MusicArtist,Playlist';
  if (c === 'musicvideos') return 'MusicVideo';
  if (c === 'boxsets' || c === 'boxset') return 'BoxSet';
  return undefined;
}

function filterBrowseItems(items: EmbyItem[], collectionType: string): EmbyItem[] {
  if (itemTypesForLibrary(collectionType)) return items;
  const allowed = new Set(['Movie', 'Series', 'MusicAlbum', 'MusicArtist', 'BoxSet']);
  return items.filter((it) => allowed.has(it.Type ?? ''));
}

export default function EmbyLibraryScreen() {
  const router = useRouter();
  const navigation = useNavigation();
  const params = useLocalSearchParams<{
    libraryId?: string;
    libraryName?: string;
    collectionType?: string;
  }>();
  const libraryId = typeof params.libraryId === 'string' ? params.libraryId : params.libraryId?.[0] ?? '';
  const libraryName =
    typeof params.libraryName === 'string' ? params.libraryName : params.libraryName?.[0] ?? 'Library';
  const collectionType =
    typeof params.collectionType === 'string' ? params.collectionType : params.collectionType?.[0] ?? '';

  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const c = useMemo(() => getColors(scheme), [scheme]);
  const [items, setItems] = useState<EmbyItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [apiAuth, setApiAuth] = useState<{ base: string; token: string } | null>(null);

  const colWidth = useMemo(() => {
    const w = Dimensions.get('window').width;
    const pad = 12;
    const gap = 10;
    return Math.max(140, Math.floor((w - pad * 2 - gap) / 2));
  }, []);

  useLayoutEffect(() => {
    navigation.setOptions({ title: libraryName });
  }, [navigation, libraryName]);

  const load = useCallback(async () => {
    if (!libraryId) {
      setError('Missing library');
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const base = assertApiBaseUrl();
      const token = (await getStoredToken()) ?? '';
      setApiAuth({ base, token });
      const it = itemTypesForLibrary(collectionType);
      const data = await getEmbyItems({
        parentId: libraryId,
        itemTypes: it,
        recursive: true,
        limit: 200,
        sortBy: 'SortName',
        sortOrder: 'Ascending',
      });
      setItems(filterBrowseItems(data.Items ?? [], collectionType));
    } catch {
      setError('Could not load library');
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, [libraryId, collectionType]);

  useEffect(() => {
    void load();
  }, [load]);

  const renderItem = useCallback(
    ({ item }: { item: EmbyItem }) => {
      const uri =
        apiAuth?.token && item.Id
          ? buildEmbyImageUrl(item.Id, apiAuth.base, apiAuth.token, { maxWidth: 280 })
          : null;
      const typeLabel = item.Type ?? '';
      return (
        <Pressable
          style={[styles.cell, { width: colWidth, backgroundColor: c.surface, borderColor: c.border }]}
          onPress={() => navigateFromEmbyItem(item, router)}
        >
          <View style={[styles.posterWrap, { backgroundColor: '#222' }]}>
            {uri ? (
              <Image source={{ uri }} style={styles.poster} contentFit="cover" cachePolicy="disk" />
            ) : null}
          </View>
          <Text style={[styles.cellTitle, { color: c.text }]} numberOfLines={2}>
            {item.Name ?? item.Id}
          </Text>
          {!!typeLabel && (
            <Text style={[styles.cellType, { color: c.textSecondary }]} numberOfLines={1}>
              {typeLabel}
            </Text>
          )}
        </Pressable>
      );
    },
    [apiAuth, c.surface, c.border, c.text, c.textSecondary, colWidth, router]
  );

  if (loading) {
    return (
      <View style={[styles.center, { backgroundColor: c.background }]}>
        <ActivityIndicator color={c.textSecondary} />
      </View>
    );
  }

  if (error) {
    return (
      <View style={[styles.center, { backgroundColor: c.background }]}>
        <Text style={{ color: c.danger }}>{error}</Text>
      </View>
    );
  }

  return (
    <View style={[styles.root, { backgroundColor: c.background }]}>
      <FlatList
        data={items}
        keyExtractor={(it) => it.Id}
        numColumns={2}
        columnWrapperStyle={styles.rowWrap}
        contentContainerStyle={styles.listContent}
        ListEmptyComponent={
          <Text style={[styles.empty, { color: c.textSecondary }]}>No items in this library.</Text>
        }
        renderItem={renderItem}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  listContent: { paddingHorizontal: 12, paddingBottom: 24 },
  rowWrap: { justifyContent: 'space-between', marginBottom: 10 },
  cell: {
    borderRadius: 10,
    borderWidth: StyleSheet.hairlineWidth,
    overflow: 'hidden',
    paddingBottom: 8,
  },
  posterWrap: { width: '100%', aspectRatio: 2 / 3 },
  poster: { width: '100%', height: '100%' },
  cellTitle: { fontSize: 13, fontWeight: '600', marginTop: 8, paddingHorizontal: 8, minHeight: 36 },
  cellType: { fontSize: 11, paddingHorizontal: 8, marginTop: 2 },
  empty: { textAlign: 'center', marginTop: 32, fontSize: 15 },
});
