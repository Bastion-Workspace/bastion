import { useCallback, useEffect, useLayoutEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  useColorScheme,
} from 'react-native';
import { Image } from 'expo-image';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { buildEmbyImageUrl, getEmbyEpisodes, getEmbySeasons, type EmbyItem } from '../../../src/api/emby';
import { assertApiBaseUrl } from '../../../src/api/config';
import { getStoredToken } from '../../../src/session/tokenStore';
import { getColors } from '../../../src/theme/colors';

function formatSeasonEpisodeLabel(item: EmbyItem): string {
  const pi = item.ParentIndexNumber;
  const ix = item.IndexNumber;
  if (typeof pi === 'number' && typeof ix === 'number') {
    return `S${String(pi).padStart(2, '0')}E${String(ix).padStart(2, '0')}`;
  }
  if (typeof ix === 'number') return `E${ix}`;
  return '';
}

export default function EmbySeriesScreen() {
  const router = useRouter();
  const navigation = useNavigation();
  const params = useLocalSearchParams<{ seriesId?: string; seriesName?: string }>();
  const seriesId = typeof params.seriesId === 'string' ? params.seriesId : params.seriesId?.[0] ?? '';
  const seriesName =
    typeof params.seriesName === 'string' ? params.seriesName : params.seriesName?.[0] ?? 'Series';

  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const c = useMemo(() => getColors(scheme), [scheme]);
  const [seasons, setSeasons] = useState<EmbyItem[]>([]);
  const [openSeasonId, setOpenSeasonId] = useState<string | null>(null);
  const [episodesBySeason, setEpisodesBySeason] = useState<Record<string, EmbyItem[]>>({});
  const [loadingSeasonId, setLoadingSeasonId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [apiAuth, setApiAuth] = useState<{ base: string; token: string } | null>(null);

  useLayoutEffect(() => {
    navigation.setOptions({ title: seriesName });
  }, [navigation, seriesName]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!seriesId) {
        setError('Missing series');
        setLoading(false);
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const base = assertApiBaseUrl();
        const token = (await getStoredToken()) ?? '';
        if (cancelled) return;
        setApiAuth({ base, token });
        const data = await getEmbySeasons(seriesId);
        if (cancelled) return;
        setSeasons(data.Items ?? []);
      } catch {
        if (!cancelled) {
          setError('Could not load seasons');
          setSeasons([]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [seriesId]);

  const onPressSeason = useCallback((season: EmbyItem) => {
    const sid = season.Id;
    if (!sid) return;
    setOpenSeasonId((prev) => (prev === sid ? null : sid));
  }, []);

  useEffect(() => {
    if (!openSeasonId || !seriesId) return;
    if (Object.prototype.hasOwnProperty.call(episodesBySeason, openSeasonId)) return;
    let cancelled = false;
    setLoadingSeasonId(openSeasonId);
    void getEmbyEpisodes(seriesId, openSeasonId)
      .then((data) => {
        if (!cancelled) {
          setEpisodesBySeason((p) => ({ ...p, [openSeasonId]: data.Items ?? [] }));
        }
      })
      .catch(() => {
        if (!cancelled) {
          setEpisodesBySeason((p) => ({ ...p, [openSeasonId]: [] }));
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingSeasonId(null);
      });
    return () => {
      cancelled = true;
    };
  }, [openSeasonId, seriesId, episodesBySeason]);

  const openEpisode = useCallback(
    (ep: EmbyItem) => {
      router.push({
        pathname: '/(app)/media/video',
        params: {
          itemId: ep.Id,
          title: ep.Name ?? 'Episode',
          startTimeTicks: String(ep.UserData?.PlaybackPositionTicks ?? 0),
        },
      });
    },
    [router]
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
    <ScrollView style={[styles.root, { backgroundColor: c.background }]} contentContainerStyle={styles.scrollPad}>
      {seasons.map((season) => {
        const sid = season.Id;
        const expanded = sid != null && openSeasonId === sid;
        const eps = sid != null ? episodesBySeason[sid] : undefined;
        const seasonUri =
          apiAuth?.token && season.Id
            ? buildEmbyImageUrl(season.Id, apiAuth.base, apiAuth.token, { maxWidth: 120 })
            : null;
        return (
          <View key={season.Id} style={styles.seasonBlock}>
            <Pressable
              onPress={() => onPressSeason(season)}
              style={[styles.seasonRow, { backgroundColor: c.surface, borderColor: c.border }]}
            >
              {seasonUri ? (
                <Image source={{ uri: seasonUri }} style={styles.seasonThumb} contentFit="cover" cachePolicy="disk" />
              ) : (
                <View style={[styles.seasonThumb, { backgroundColor: c.chipBg }]} />
              )}
              <View style={styles.seasonText}>
                <Text style={[styles.seasonTitle, { color: c.text }]} numberOfLines={1}>
                  {season.Name ?? 'Season'}
                </Text>
                {season.UserData?.PlayedPercentage != null &&
                  season.UserData.PlayedPercentage > 0 &&
                  season.UserData.PlayedPercentage < 100 && (
                    <View style={[styles.miniProgressBg, { backgroundColor: c.border }]}>
                      <View
                        style={[
                          styles.miniProgressFill,
                          { width: `${season.UserData.PlayedPercentage}%`, backgroundColor: c.link },
                        ]}
                      />
                    </View>
                  )}
              </View>
              <Ionicons name={expanded ? 'chevron-up' : 'chevron-down'} size={22} color={c.textSecondary} />
            </Pressable>
            {expanded && sid != null && loadingSeasonId === sid ? (
              <ActivityIndicator style={{ marginVertical: 12 }} color={c.textSecondary} />
            ) : null}
            {expanded && eps
              ? eps.map((ep) => {
                  const epUri =
                    apiAuth?.token && ep.Id
                      ? buildEmbyImageUrl(ep.Id, apiAuth.base, apiAuth.token, { maxWidth: 160 })
                      : null;
                  const pct = ep.UserData?.PlayedPercentage;
                  const label = formatSeasonEpisodeLabel(ep);
                  return (
                    <Pressable
                      key={ep.Id}
                      onPress={() => openEpisode(ep)}
                      style={[styles.epRow, { borderBottomColor: c.border, backgroundColor: c.background }]}
                    >
                      {epUri ? (
                        <Image source={{ uri: epUri }} style={styles.epThumb} contentFit="cover" cachePolicy="disk" />
                      ) : (
                        <View style={[styles.epThumb, { backgroundColor: c.chipBg }]} />
                      )}
                      <View style={styles.epText}>
                        {!!label && (
                          <Text style={[styles.epMeta, { color: c.textSecondary }]} numberOfLines={1}>
                            {label}
                          </Text>
                        )}
                        <Text style={[styles.epTitle, { color: c.text }]} numberOfLines={2}>
                          {ep.Name ?? ep.Id}
                        </Text>
                        {typeof pct === 'number' && pct > 0 && pct < 100 && (
                          <View style={[styles.miniProgressBg, { backgroundColor: c.border, marginTop: 6 }]}>
                            <View style={[styles.miniProgressFill, { width: `${pct}%`, backgroundColor: c.link }]} />
                          </View>
                        )}
                      </View>
                      <Ionicons name="play-circle-outline" size={28} color={c.link} />
                    </Pressable>
                  );
                })
              : null}
          </View>
        );
      })}
      {seasons.length === 0 ? (
        <Text style={[styles.empty, { color: c.textSecondary }]}>No seasons found.</Text>
      ) : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  scrollPad: { paddingBottom: 32 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  seasonBlock: { marginBottom: 4 },
  seasonRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    marginHorizontal: 12,
    marginTop: 8,
    borderRadius: 10,
    borderWidth: StyleSheet.hairlineWidth,
    gap: 12,
  },
  seasonThumb: { width: 48, height: 72, borderRadius: 6 },
  seasonText: { flex: 1, minWidth: 0 },
  seasonTitle: { fontSize: 16, fontWeight: '600' },
  epRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 12,
    marginLeft: 24,
    marginRight: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
    gap: 10,
  },
  epThumb: { width: 88, height: 50, borderRadius: 6 },
  epText: { flex: 1, minWidth: 0 },
  epMeta: { fontSize: 12, fontWeight: '600', marginBottom: 2 },
  epTitle: { fontSize: 14, fontWeight: '600' },
  miniProgressBg: { height: 3, borderRadius: 2, overflow: 'hidden', marginTop: 6 },
  miniProgressFill: { height: '100%' },
  empty: { textAlign: 'center', marginTop: 24, fontSize: 15 },
});
