import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  useColorScheme,
} from 'react-native';
import { Image } from 'expo-image';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import {
  buildEmbyImageUrl,
  getEmbyLatestItems,
  getEmbyLibraries,
  getEmbyResumeItems,
  type EmbyItem,
  type EmbyLibrary,
} from '../../api/emby';
import { navigateFromEmbyItem } from '../../media/embyItemNavigation';
import { getColors } from '../../theme/colors';

type Props = {
  baseUrl: string;
  token: string;
};

export function EmbyHomeSections({ baseUrl, token }: Props) {
  const router = useRouter();
  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const c = useMemo(() => getColors(scheme), [scheme]);
  const [loading, setLoading] = useState(true);
  const [resume, setResume] = useState<EmbyItem[]>([]);
  const [latest, setLatest] = useState<EmbyItem[]>([]);
  const [libraryParentId, setLibraryParentId] = useState<string | null>(null);
  const [libraries, setLibraries] = useState<EmbyLibrary[]>([]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const libs = await getEmbyLibraries();
      setLibraries(libs.libraries ?? []);
      const first = libs.libraries?.[0];
      const parentId = first?.id ?? first?.item_id ?? null;
      setLibraryParentId(parentId);
      const [res, lat] = await Promise.all([
        getEmbyResumeItems(24),
        parentId ? getEmbyLatestItems(parentId, 24) : Promise.resolve({ Items: [] as EmbyItem[] }),
      ]);
      setResume(res.Items ?? []);
      setLatest(lat.Items ?? []);
    } catch {
      setLibraries([]);
      setResume([]);
      setLatest([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const onPressItem = useCallback(
    (item: EmbyItem) => {
      navigateFromEmbyItem(item, router);
    },
    [router]
  );

  const libraryIcon = useCallback((collectionType?: string) => {
    const ct = (collectionType ?? '').toLowerCase();
    if (ct === 'movies' || ct === 'movie') return 'film-outline' as const;
    if (ct === 'tvshows' || ct === 'tv') return 'tv-outline' as const;
    if (ct === 'music' || ct === 'musicvideos') return 'musical-notes-outline' as const;
    return 'library-outline' as const;
  }, []);

  const onPressLibrary = useCallback(
    (lib: EmbyLibrary) => {
      const id = lib.id ?? lib.item_id ?? '';
      if (!id) return;
      router.push({
        pathname: '/(app)/media/emby-library',
        params: {
          libraryId: id,
          libraryName: lib.name ?? 'Library',
          collectionType: lib.collection_type ?? '',
        },
      });
    },
    [router]
  );

  const renderCard = useCallback(
    ({ item }: { item: EmbyItem }) => {
      const uri = buildEmbyImageUrl(item.Id, baseUrl, token, { maxWidth: 200 });
      const pct = item.UserData?.PlayedPercentage;
      return (
        <Pressable
          style={[styles.card, { backgroundColor: c.surface, borderColor: c.border }]}
          onPress={() => onPressItem(item)}
        >
          <Image source={{ uri }} style={styles.poster} contentFit="cover" cachePolicy="disk" />
          <Text style={[styles.cardTitle, { color: c.text }]} numberOfLines={2}>
            {item.SeriesName ? `${item.SeriesName}\n${item.Name ?? ''}` : item.Name ?? item.Id}
          </Text>
          {typeof pct === 'number' && pct > 0 && pct < 100 && (
            <View style={[styles.progressBg, { backgroundColor: c.border }]}>
              <View style={[styles.progressFill, { width: `${pct}%`, backgroundColor: c.link }]} />
            </View>
          )}
        </Pressable>
      );
    },
    [baseUrl, token, c.surface, c.border, c.text, c.link, onPressItem]
  );

  if (loading) {
    return (
      <View style={styles.loadingWrap}>
        <ActivityIndicator color={c.textSecondary} />
      </View>
    );
  }

  if (!resume.length && !latest.length && !libraryParentId) {
    return null;
  }

  return (
    <View style={styles.root}>
      {libraries.length > 0 && (
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, { color: c.textSecondary }]}>Libraries</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.libRow}>
            {libraries.map((lib) => (
              <Pressable
                key={lib.id}
                onPress={() => onPressLibrary(lib)}
                style={[styles.libChip, { backgroundColor: c.surface, borderColor: c.border }]}
              >
                <Ionicons name={libraryIcon(lib.collection_type)} size={18} color={c.text} />
                <Text style={[styles.libChipText, { color: c.text }]} numberOfLines={1}>
                  {lib.name}
                </Text>
              </Pressable>
            ))}
          </ScrollView>
        </View>
      )}
      {resume.length > 0 && (
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, { color: c.textSecondary }]}>Continue watching</Text>
          <FlatList
            horizontal
            data={resume}
            keyExtractor={(it) => it.Id}
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.rowPad}
            renderItem={renderCard}
          />
        </View>
      )}
      {latest.length > 0 && (
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, { color: c.textSecondary }]}>Recently added</Text>
          <FlatList
            horizontal
            data={latest}
            keyExtractor={(it) => it.Id}
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.rowPad}
            renderItem={renderCard}
          />
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { marginBottom: 8 },
  loadingWrap: { paddingVertical: 16, alignItems: 'center' },
  section: { marginBottom: 12, overflow: 'hidden' },
  sectionTitle: {
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    marginHorizontal: 12,
    marginBottom: 8,
  },
  rowPad: { paddingHorizontal: 8 },
  card: {
    width: 108,
    marginRight: 8,
    borderRadius: 10,
    borderWidth: StyleSheet.hairlineWidth,
    overflow: 'hidden',
    paddingBottom: 8,
  },
  poster: { width: '100%', height: 140, backgroundColor: '#222' },
  cardTitle: { fontSize: 12, fontWeight: '600', marginTop: 6, paddingHorizontal: 6, minHeight: 32 },
  progressBg: { height: 3, marginHorizontal: 6, marginTop: 4, borderRadius: 2, overflow: 'hidden' },
  progressFill: { height: '100%', borderRadius: 2 },
  libRow: { paddingHorizontal: 8, flexDirection: 'row', alignItems: 'center' },
  libChip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 20,
    borderWidth: StyleSheet.hairlineWidth,
    marginRight: 8,
    maxWidth: 220,
  },
  libChipText: { fontSize: 14, fontWeight: '600', flexShrink: 1, marginLeft: 8 },
});
