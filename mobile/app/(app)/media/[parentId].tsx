import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  View,
  useColorScheme,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import {
  getAlbumsByArtist,
  getTracks,
  type MusicAlbum,
  type MusicLibraryResponse,
  type MusicTrack,
  type MusicTracksResponse,
} from '../../../src/api/media';
import { isApiError } from '../../../src/api/client';
import { getColors } from '../../../src/theme/colors';
import { useMediaPlayer } from '../../../src/context/MediaPlayerContext';
import { TrackItem } from '../../../src/components/media/TrackItem';
import { consumeMediaSearchTracksSeed } from '../../../src/session/mediaSearchSeed';
import { downloadTrack } from '../../../src/utils/mediaDownloadStore';
import { getCachedEntry, setCachedEntry, TTL_TRACKS_MS } from '../../../src/utils/mediaListCache';

function sortPodcastEpisodesNewestFirst(list: MusicTrack[]): MusicTrack[] {
  return [...list].sort((a, b) => {
    const ma = a.metadata as Record<string, unknown> | undefined;
    const mb = b.metadata as Record<string, unknown> | undefined;
    const ta = String(ma?.published_date ?? ma?.publishedAt ?? '');
    const tb = String(mb?.published_date ?? mb?.publishedAt ?? '');
    const da = ta ? new Date(ta).getTime() : NaN;
    const db = tb ? new Date(tb).getTime() : NaN;
    const na = Number.isFinite(da) ? da : 0;
    const nb = Number.isFinite(db) ? db : 0;
    return nb - na;
  });
}

export default function MediaDetailScreen() {
  const router = useRouter();
  const navigation = useNavigation();
  const params = useLocalSearchParams<{
    parentId: string;
    type: string;
    title?: string;
    serviceType?: string;
  }>();
  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const c = useMemo(() => getColors(scheme), [scheme]);
  const { replaceQueueAndPlay } = useMediaPlayer();

  const parentId = String(params.parentId ?? '');
  const detailType = String(params.type ?? 'album');
  const title = String(params.title ?? 'Media');
  const serviceType = params.serviceType ? String(params.serviceType) : null;
  const isAbs = serviceType === 'audiobookshelf';

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tracks, setTracks] = useState<MusicTrack[]>([]);
  const [artistAlbums, setArtistAlbums] = useState<MusicAlbum[]>([]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      if (detailType === 'search') {
        const seeded = consumeMediaSearchTracksSeed();
        if (!seeded.length) {
          setError('No search results. Go back and search again.');
          setTracks([]);
        } else {
          setTracks(seeded);
        }
        return;
      }
      if (detailType === 'artist') {
        const cacheKey = `library_artist_${parentId}_${serviceType ?? 'default'}`;
        const cached = await getCachedEntry<MusicLibraryResponse>(cacheKey, TTL_TRACKS_MS);
        if (cached) {
          setArtistAlbums(cached.data.albums ?? []);
          setTracks([]);
          setLoading(false);
          if (!cached.stale) return;
        }
        const lib = await getAlbumsByArtist(parentId, serviceType);
        setArtistAlbums(lib.albums ?? []);
        setTracks([]);
        await setCachedEntry(cacheKey, lib);
        return;
      }
      const pt = detailType === 'playlist' ? 'playlist' : 'album';
      const cacheKey = `tracks_${pt}_${parentId}_${serviceType ?? 'default'}`;
      const cached = await getCachedEntry<MusicTracksResponse>(cacheKey, TTL_TRACKS_MS);
      if (cached) {
        let t = cached.data.tracks ?? [];
        if (isAbs && pt === 'playlist') t = sortPodcastEpisodesNewestFirst(t);
        setTracks(t);
        setLoading(false);
        if (!cached.stale) return;
      }
      const res = await getTracks(parentId, pt, serviceType);
      let list = res.tracks ?? [];
      if (isAbs && pt === 'playlist') list = sortPodcastEpisodesNewestFirst(list);
      setTracks(list);
      await setCachedEntry(cacheKey, { ...res, tracks: list });
    } catch (e) {
      setError(isApiError(e) ? `Failed to load (${e.status})` : 'Failed to load');
      setTracks([]);
    } finally {
      setLoading(false);
    }
  }, [detailType, parentId, serviceType, isAbs]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    navigation.setOptions({
      title,
      headerShown: true,
      headerBackTitle: 'Back',
      headerStyle: { backgroundColor: c.surface },
      headerTintColor: c.text,
      headerTitleStyle: { color: c.text },
    });
  }, [navigation, title, c.surface, c.text]);

  const playAll = useCallback(
    async (list: MusicTrack[], startIndex = 0) => {
      if (!list.length) return;
      await replaceQueueAndPlay(list, startIndex, {
        serviceType,
        parentId: detailType === 'album' || detailType === 'playlist' ? parentId : null,
      });
    },
    [replaceQueueAndPlay, serviceType, parentId, detailType]
  );

  const shuffleAll = useCallback(async () => {
    if (!tracks.length) return;
    const shuffled = [...tracks];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    await replaceQueueAndPlay(shuffled, 0, {
      serviceType,
      parentId: detailType === 'album' || detailType === 'playlist' ? parentId : null,
    });
  }, [tracks, replaceQueueAndPlay, serviceType, parentId, detailType]);

  const downloadAll = useCallback(async () => {
    for (const t of tracks) {
      try {
        await downloadTrack(t, {
          serviceType,
          parentId: detailType === 'album' || detailType === 'playlist' ? parentId : null,
        });
      } catch {
        // continue other tracks
      }
    }
  }, [tracks, serviceType, parentId, detailType]);

  if (detailType === 'artist') {
    return (
      <View style={[styles.root, { backgroundColor: c.background }]}>
        <Text style={[styles.h2, { color: c.textSecondary }]}>{isAbs ? 'Books' : 'Albums'}</Text>
        <FlatList
          data={artistAlbums}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.listPad}
          ListEmptyComponent={
            loading ? (
              <ActivityIndicator color={c.textSecondary} style={{ marginTop: 24 }} />
            ) : (
              <Text style={[styles.empty, { color: c.textSecondary }]}>
                {isAbs ? 'No books for this author.' : 'No albums for this artist.'}
              </Text>
            )
          }
          renderItem={({ item }) => (
            <Pressable
              style={[styles.row, { backgroundColor: c.surface, borderColor: c.border }]}
              onPress={() =>
                router.push({
                  pathname: '/(app)/media/[parentId]',
                  params: {
                    parentId: item.id,
                    type: 'album',
                    title: item.title,
                    serviceType: serviceType ?? '',
                  },
                })
              }
            >
              <Ionicons name="disc-outline" size={22} color={c.textSecondary} />
              <View style={{ flex: 1, minWidth: 0 }}>
                <Text style={[styles.rowTitle, { color: c.text }]} numberOfLines={1}>
                  {item.title}
                </Text>
                {!!item.artist && (
                  <Text style={[styles.rowSub, { color: c.textSecondary }]} numberOfLines={1}>
                    {item.artist}
                  </Text>
                )}
              </View>
              <Ionicons name="chevron-forward" size={18} color={c.textSecondary} />
            </Pressable>
          )}
        />
      </View>
    );
  }

  const trackSectionHeading =
    detailType === 'search'
      ? null
      : detailType === 'playlist' && isAbs
        ? 'Episodes'
        : detailType === 'album' && isAbs
          ? 'Chapters'
          : 'Tracks';

  return (
    <View style={[styles.root, { backgroundColor: c.background }]}>
      {trackSectionHeading ? (
        <Text style={[styles.h2, { color: c.textSecondary }]}>{trackSectionHeading}</Text>
      ) : null}
      <View style={[styles.toolbar, { borderBottomColor: c.border }]}>
        <Pressable
          style={[styles.tbBtn, { backgroundColor: c.chipBg }]}
          onPress={() => void playAll(tracks, 0)}
          disabled={!tracks.length}
        >
          <Ionicons name="play" size={18} color={c.text} />
          <Text style={[styles.tbTxt, { color: c.text }]}>Play all</Text>
        </Pressable>
        <Pressable
          style={[styles.tbBtn, { backgroundColor: c.chipBg }]}
          onPress={() => void shuffleAll()}
          disabled={!tracks.length}
        >
          <Ionicons name="shuffle" size={18} color={c.text} />
          <Text style={[styles.tbTxt, { color: c.text }]}>Shuffle all</Text>
        </Pressable>
        <Pressable
          style={[styles.tbBtn, { backgroundColor: c.chipBg }]}
          onPress={() => void downloadAll()}
          disabled={!tracks.length}
        >
          <Ionicons name="download-outline" size={18} color={c.text} />
          <Text style={[styles.tbTxt, { color: c.text }]}>Download all</Text>
        </Pressable>
      </View>

      {error && <Text style={[styles.err, { color: c.danger }]}>{error}</Text>}

      {loading ? (
        <ActivityIndicator style={{ marginTop: 24 }} color={c.textSecondary} />
      ) : (
        <FlatList
          data={tracks}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.listPad}
          renderItem={({ item, index }) => (
            <TrackItem
              track={item}
              index={index}
              serviceType={serviceType}
              parentId={detailType === 'album' || detailType === 'playlist' ? parentId : null}
              onPlay={(i) => void playAll(tracks, i)}
            />
          )}
          ListEmptyComponent={
            <Text style={[styles.empty, { color: c.textSecondary }]}>
              {detailType === 'playlist' && isAbs
                ? 'No episodes.'
                : detailType === 'album' && isAbs
                  ? 'No chapters.'
                  : 'No tracks.'}
            </Text>
          }
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  h2: {
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    paddingHorizontal: 16,
    paddingTop: 12,
  },
  toolbar: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  tbBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 20,
  },
  tbTxt: { fontSize: 14, fontWeight: '600' },
  err: { margin: 12, fontSize: 14 },
  listPad: { paddingHorizontal: 12, paddingBottom: 120 },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderRadius: 10,
    marginBottom: 8,
    borderWidth: StyleSheet.hairlineWidth,
    gap: 10,
  },
  rowTitle: { fontSize: 16, fontWeight: '600' },
  rowSub: { fontSize: 13, marginTop: 2 },
  empty: { textAlign: 'center', marginTop: 24, fontSize: 15 },
});
