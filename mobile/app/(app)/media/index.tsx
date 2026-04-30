import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
  useColorScheme,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import {
  getLibrary,
  getMediaSources,
  refreshMusicCache,
  searchAlbums,
  searchArtists,
  searchTracks,
  type MusicAlbum,
  type MusicArtist,
  type MusicLibraryResponse,
  type MusicPlaylist,
  type MusicTracksResponse,
} from '../../../src/api/media';
import { isApiError } from '../../../src/api/client';
import { getColors } from '../../../src/theme/colors';
import { setMediaSearchTracksSeed } from '../../../src/session/mediaSearchSeed';

type TabKey = 'artists' | 'albums' | 'playlists';

export default function MediaLibraryScreen() {
  const router = useRouter();
  const navigation = useNavigation();
  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const c = useMemo(() => getColors(scheme), [scheme]);
  const [tab, setTab] = useState<TabKey>('albums');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sources, setSources] = useState<{ service_type: string; label: string }[]>([]);
  const [serviceType, setServiceType] = useState<string | null>(null);
  const [library, setLibrary] = useState<MusicLibraryResponse | null>(null);
  const [query, setQuery] = useState('');
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchAlbumsRes, setSearchAlbumsRes] = useState<MusicAlbum[] | null>(null);
  const [searchArtistsRes, setSearchArtistsRes] = useState<MusicArtist[] | null>(null);
  const [searchTracksRes, setSearchTracksRes] = useState<MusicTracksResponse | null>(null);

  const load = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      const src = await getMediaSources();
      const list = (src.sources || [])
        .filter((s) => s.has_config !== false && s.is_active !== false)
        .map((s) => ({
          service_type: s.service_type,
          label: s.service_name?.trim() || s.service_type,
        }));
      setSources(list);
      if (list.length === 0) {
        setLibrary({ albums: [], artists: [], playlists: [] });
        return;
      }
      const st = serviceType ?? list[0]?.service_type ?? 'subsonic';
      const lib = await getLibrary(st);
      setLibrary(lib);
    } catch (e) {
      const msg = isApiError(e) ? `Could not load library (${e.status})` : 'Could not load library';
      setError(msg);
      setLibrary(null);
    } finally {
      setLoading(false);
    }
  }, [serviceType]);

  useEffect(() => {
    if (serviceType == null && sources.length > 0) {
      setServiceType(sources[0].service_type);
    }
  }, [sources, serviceType]);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load])
  );

  const onRefreshCache = useCallback(async () => {
    const st = serviceType ?? sources[0]?.service_type;
    if (!st) return;
    setRefreshing(true);
    setError(null);
    try {
      await refreshMusicCache(st);
      const lib = await getLibrary(st);
      setLibrary(lib);
    } catch (e) {
      setError(isApiError(e) ? `Refresh failed (${e.status})` : 'Refresh failed');
    } finally {
      setRefreshing(false);
    }
  }, [serviceType, sources]);

  const runSearch = useCallback(async () => {
    const st = serviceType ?? sources[0]?.service_type;
    if (!st || !query.trim()) {
      setSearchAlbumsRes(null);
      setSearchArtistsRes(null);
      setSearchTracksRes(null);
      return;
    }
    setSearchLoading(true);
    setError(null);
    try {
      const [a, ar, t] = await Promise.all([
        searchAlbums({ query: query.trim(), service_type: st, limit: 20 }),
        searchArtists({ query: query.trim(), service_type: st, limit: 20 }),
        searchTracks({ query: query.trim(), service_type: st, limit: 20 }),
      ]);
      setSearchAlbumsRes(a.albums);
      setSearchArtistsRes(ar.artists);
      setSearchTracksRes(t);
    } catch (e) {
      setError(isApiError(e) ? `Search failed (${e.status})` : 'Search failed');
    } finally {
      setSearchLoading(false);
    }
  }, [query, serviceType, sources]);

  const openAlbum = useCallback(
    (id: string, title: string) => {
      router.push({
        pathname: '/(app)/media/[parentId]',
        params: { parentId: id, type: 'album', title, serviceType: serviceType ?? sources[0]?.service_type ?? '' },
      });
    },
    [router, serviceType, sources]
  );

  const openPlaylist = useCallback(
    (id: string, title: string) => {
      router.push({
        pathname: '/(app)/media/[parentId]',
        params: { parentId: id, type: 'playlist', title, serviceType: serviceType ?? sources[0]?.service_type ?? '' },
      });
    },
    [router, serviceType, sources]
  );

  const openArtist = useCallback(
    (id: string, title: string) => {
      router.push({
        pathname: '/(app)/media/[parentId]',
        params: { parentId: id, type: 'artist', title, serviceType: serviceType ?? sources[0]?.service_type ?? '' },
      });
    },
    [router, serviceType, sources]
  );

  const openTrackResults = useCallback(() => {
    if (!searchTracksRes?.tracks?.length) return;
    setMediaSearchTracksSeed(searchTracksRes.tracks);
    router.push({
      pathname: '/(app)/media/[parentId]',
      params: {
        parentId: 'search',
        type: 'search',
        title: 'Search results',
        serviceType: serviceType ?? sources[0]?.service_type ?? '',
      },
    });
  }, [router, searchTracksRes, serviceType, sources]);

  useEffect(() => {
    navigation.setOptions({
      headerRight: () => (
        <View style={styles.headerActions}>
          <Pressable onPress={() => router.push('/(app)/media/downloads')} accessibilityLabel="Downloads">
            <Ionicons name="download-outline" size={22} color={c.text} style={{ marginRight: 14 }} />
          </Pressable>
          <Pressable onPress={() => void onRefreshCache()} disabled={refreshing} accessibilityLabel="Refresh library">
            <Ionicons name="refresh" size={22} color={c.text} />
          </Pressable>
        </View>
      ),
    });
  }, [navigation, router, onRefreshCache, refreshing, c.text]);

  const albums = library?.albums ?? [];
  const artists = library?.artists ?? [];
  const playlists = library?.playlists ?? [];

  return (
    <View style={[styles.root, { backgroundColor: c.background }]}>
      {sources.length > 1 && (
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.sourceScroll}>
          {sources.map((s) => (
            <Pressable
              key={s.service_type}
              onPress={() => setServiceType(s.service_type)}
              style={[
                styles.sourceChip,
                { backgroundColor: (serviceType ?? sources[0]?.service_type) === s.service_type ? c.chipBgActive : c.chipBg },
              ]}
            >
              <Text
                style={{
                  color: (serviceType ?? sources[0]?.service_type) === s.service_type ? c.chipTextActive : c.chipText,
                  fontWeight: '600',
                }}
              >
                {s.label}
              </Text>
            </Pressable>
          ))}
        </ScrollView>
      )}

      <View style={[styles.searchRow, { backgroundColor: c.surface, borderColor: c.border }]}>
        <TextInput
          value={query}
          onChangeText={setQuery}
          placeholder="Search library…"
          placeholderTextColor={c.textSecondary}
          style={[styles.searchInput, { color: c.text }]}
          onSubmitEditing={() => void runSearch()}
          returnKeyType="search"
        />
        <Pressable onPress={() => void runSearch()} accessibilityLabel="Search">
          {searchLoading ? (
            <ActivityIndicator color={c.text} />
          ) : (
            <Ionicons name="search" size={22} color={c.text} />
          )}
        </Pressable>
      </View>

      {query.trim() && (searchAlbumsRes || searchArtistsRes || searchTracksRes) && (
        <View style={[styles.searchResults, { borderBottomColor: c.border }]}>
          <Text style={[styles.sectionLabel, { color: c.textSecondary }]}>
            Results · {searchTracksRes?.tracks?.length ?? 0} tracks
          </Text>
          {!!searchTracksRes?.tracks?.length && (
            <Pressable onPress={openTrackResults} style={[styles.linkRow, { backgroundColor: c.surface }]}>
              <Text style={[styles.linkTxt, { color: c.link }]}>Play search results as queue</Text>
              <Ionicons name="chevron-forward" size={18} color={c.link} />
            </Pressable>
          )}
          {(searchAlbumsRes || []).slice(0, 5).map((a) => (
            <Pressable key={a.id} onPress={() => openAlbum(a.id, a.title)} style={styles.searchHit}>
              <Text style={{ color: c.text }} numberOfLines={1}>
                {a.title}
              </Text>
            </Pressable>
          ))}
          {(searchArtistsRes || []).slice(0, 5).map((a) => (
            <Pressable key={a.id} onPress={() => openArtist(a.id, a.name)} style={styles.searchHit}>
              <Text style={{ color: c.text }} numberOfLines={1}>
                {a.name}
              </Text>
            </Pressable>
          ))}
        </View>
      )}

      <View style={styles.tabs}>
        {(['artists', 'albums', 'playlists'] as const).map((k) => (
          <Pressable
            key={k}
            onPress={() => setTab(k)}
            style={[styles.tabBtn, tab === k && { borderBottomColor: c.link, borderBottomWidth: 2 }]}
          >
            <Text style={[styles.tabLabel, { color: tab === k ? c.text : c.textSecondary }]}>
              {k === 'artists' ? 'Artists' : k === 'albums' ? 'Albums' : 'Playlists'}
            </Text>
          </Pressable>
        ))}
      </View>

      {error && (
        <Text style={[styles.err, { color: c.danger }]} accessibilityLiveRegion="polite">
          {error}
        </Text>
      )}

      {!loading && sources.length === 0 && (
        <Text style={[styles.hint, { color: c.textSecondary }]}>
          No media sources configured. Add Subsonic, Audiobookshelf, or Emby in the Bastion web app, then tap refresh above.
        </Text>
      )}

      {loading ? (
        <ActivityIndicator style={{ marginTop: 24 }} color={c.textSecondary} />
      ) : (
        <ScrollView contentContainerStyle={styles.listPad}>
          {tab === 'artists' &&
            artists.map((a) => (
              <Pressable
                key={a.id}
                style={[styles.row, { borderBottomColor: c.border, backgroundColor: c.surface }]}
                onPress={() => openArtist(a.id, a.name)}
              >
                <Ionicons name="person-outline" size={22} color={c.textSecondary} />
                <View style={styles.rowText}>
                  <Text style={[styles.rowTitle, { color: c.text }]} numberOfLines={1}>
                    {a.name}
                  </Text>
                  {a.album_count != null && (
                    <Text style={[styles.rowSub, { color: c.textSecondary }]}>{a.album_count} albums</Text>
                  )}
                </View>
                <Ionicons name="chevron-forward" size={18} color={c.textSecondary} />
              </Pressable>
            ))}
          {tab === 'albums' &&
            albums.map((a) => (
              <Pressable
                key={a.id}
                style={[styles.row, { borderBottomColor: c.border, backgroundColor: c.surface }]}
                onPress={() => openAlbum(a.id, a.title)}
              >
                <Ionicons name="disc-outline" size={22} color={c.textSecondary} />
                <View style={styles.rowText}>
                  <Text style={[styles.rowTitle, { color: c.text }]} numberOfLines={1}>
                    {a.title}
                  </Text>
                  {!!a.artist && (
                    <Text style={[styles.rowSub, { color: c.textSecondary }]} numberOfLines={1}>
                      {a.artist}
                    </Text>
                  )}
                </View>
                <Ionicons name="chevron-forward" size={18} color={c.textSecondary} />
              </Pressable>
            ))}
          {tab === 'playlists' &&
            playlists.map((p) => (
              <Pressable
                key={p.id}
                style={[styles.row, { borderBottomColor: c.border, backgroundColor: c.surface }]}
                onPress={() => openPlaylist(p.id, p.name)}
              >
                <Ionicons name="list-outline" size={22} color={c.textSecondary} />
                <View style={styles.rowText}>
                  <Text style={[styles.rowTitle, { color: c.text }]} numberOfLines={1}>
                    {p.name}
                  </Text>
                  {p.track_count != null && (
                    <Text style={[styles.rowSub, { color: c.textSecondary }]}>{p.track_count} tracks</Text>
                  )}
                </View>
                <Ionicons name="chevron-forward" size={18} color={c.textSecondary} />
              </Pressable>
            ))}
          {!loading && tab === 'artists' && artists.length === 0 && (
            <Text style={[styles.empty, { color: c.textSecondary }]}>No artists in cache. Pull refresh in header.</Text>
          )}
          {!loading && tab === 'albums' && albums.length === 0 && (
            <Text style={[styles.empty, { color: c.textSecondary }]}>No albums in cache. Pull refresh in header.</Text>
          )}
          {!loading && tab === 'playlists' && playlists.length === 0 && (
            <Text style={[styles.empty, { color: c.textSecondary }]}>No playlists in cache.</Text>
          )}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  headerActions: { flexDirection: 'row', alignItems: 'center' },
  sourceScroll: { maxHeight: 48, paddingHorizontal: 12, paddingTop: 8 },
  sourceChip: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 16,
    marginRight: 8,
  },
  searchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: 12,
    marginTop: 8,
    borderRadius: 10,
    borderWidth: StyleSheet.hairlineWidth,
    paddingHorizontal: 10,
  },
  searchInput: {
    flex: 1,
    minHeight: 40,
    fontSize: 16,
  },
  searchResults: {
    marginHorizontal: 12,
    marginTop: 8,
    paddingBottom: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  sectionLabel: {
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    marginBottom: 6,
  },
  linkRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 12,
    borderRadius: 8,
    marginBottom: 6,
  },
  linkTxt: { fontSize: 15, fontWeight: '600' },
  searchHit: { paddingVertical: 8, paddingHorizontal: 4 },
  tabs: {
    flexDirection: 'row',
    marginTop: 12,
    paddingHorizontal: 12,
    gap: 8,
  },
  tabBtn: {
    paddingVertical: 8,
    paddingHorizontal: 12,
  },
  tabLabel: { fontSize: 15, fontWeight: '600' },
  err: { marginHorizontal: 16, marginTop: 8, fontSize: 14 },
  hint: { marginHorizontal: 16, marginTop: 12, fontSize: 14, lineHeight: 20 },
  listPad: { paddingHorizontal: 12, paddingBottom: 120 },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderRadius: 10,
    marginBottom: 8,
    gap: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  rowText: { flex: 1, minWidth: 0 },
  rowTitle: { fontSize: 16, fontWeight: '600' },
  rowSub: { fontSize: 13, marginTop: 2 },
  empty: { marginTop: 24, textAlign: 'center', fontSize: 15 },
});
