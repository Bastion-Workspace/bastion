import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  useColorScheme,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import {
  getEbooksSettings,
  putEbooksSettings,
  type OpdsCatalogEntryResponse,
  type RecentlyOpenedEntry,
} from '../../../src/api/ebooks';
import { getColors } from '../../../src/theme/colors';
import { OpdsNavigator } from '../../../src/components/ebooks/OpdsNavigator';
import { ebookCacheDelete } from '../../../src/utils/ebookFileStore';

export default function EbooksLibraryScreen() {
  const router = useRouter();
  const navigation = useNavigation();
  const scheme = useColorScheme();
  const c = useMemo(() => getColors(scheme === 'dark' ? 'dark' : 'light'), [scheme]);
  const [loading, setLoading] = useState(true);
  const [recent, setRecent] = useState<RecentlyOpenedEntry[]>([]);
  const [catalogs, setCatalogs] = useState<OpdsCatalogEntryResponse[]>([]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const s = await getEbooksSettings();
      setRecent(s.recently_opened || []);
      setCatalogs(s.catalogs || []);
    } catch {
      setRecent([]);
      setCatalogs([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load])
  );

  useEffect(() => {
    navigation.setOptions({
      headerRight: () => (
        <Pressable
          onPress={() => router.push('/(app)/ebooks/settings')}
          style={styles.headerBtn}
          accessibilityLabel="eBook settings"
        >
          <Ionicons name="settings-outline" size={24} color={c.text} />
        </Pressable>
      ),
    });
  }, [navigation, router, c.text]);

  const openReader = useCallback(
    (p: { catalogId: string; acquisitionUrl: string; title: string; digest?: string; format?: 'epub' | 'pdf' }) => {
      router.push({
        pathname: '/(app)/ebooks/reader',
        params: {
          catalogId: p.catalogId,
          acquisitionUrl: p.acquisitionUrl,
          title: p.title,
          digest: p.digest || '',
          format: p.format === 'pdf' ? 'pdf' : 'epub',
        },
      });
    },
    [router]
  );

  const removeRecent = useCallback(
    async (row: RecentlyOpenedEntry) => {
      if (row.digest) {
        try {
          await ebookCacheDelete(row.digest);
        } catch {
          // ignore
        }
      }
      const next = recent.filter((r) => {
        if (row.digest && r.digest) return r.digest !== row.digest;
        return !(r.acquisition_url === row.acquisition_url && r.catalog_id === row.catalog_id);
      });
      try {
        await putEbooksSettings({ recently_opened: next });
        setRecent(next);
      } catch {
        // ignore
      }
    },
    [recent]
  );

  return (
    <View style={[styles.root, { backgroundColor: c.background }]}>
      <ScrollView
        keyboardShouldPersistTaps="handled"
        nestedScrollEnabled
        contentContainerStyle={styles.topScroll}
      >
        {loading ? <ActivityIndicator color={c.text} style={styles.spinner} /> : null}

        <Text style={[styles.section, { color: c.textSecondary }]}>RECENT</Text>
        {recent.length === 0 ? (
          <Text style={[styles.muted, { color: c.textSecondary }]}>No recently opened ebooks.</Text>
        ) : (
          recent.map((r) => (
            <View key={r.digest || `${r.catalog_id}:${r.acquisition_url}`} style={styles.recentRow}>
              <Pressable
                style={[styles.recentMain, { backgroundColor: c.surface, borderColor: c.border }]}
                onPress={() => {
                  if (r.catalog_id && r.acquisition_url) {
                    openReader({
                      catalogId: String(r.catalog_id),
                      acquisitionUrl: String(r.acquisition_url),
                      title: String(r.title || 'Book'),
                      digest: r.digest ? String(r.digest) : undefined,
                      format: r.acquisition_format === 'pdf' ? 'pdf' : 'epub',
                    });
                  }
                }}
              >
                <Text style={[styles.recentTitle, { color: c.text }]} numberOfLines={2}>
                  {r.title || 'Book'}
                </Text>
              </Pressable>
              <Pressable onPress={() => void removeRecent(r)} accessibilityLabel="Remove from recent">
                <Ionicons name="trash-outline" size={22} color={c.danger} />
              </Pressable>
            </View>
          ))
        )}
      </ScrollView>
      <Text style={[styles.section, { color: c.textSecondary, paddingHorizontal: 16 }]}>OPDS</Text>
      <View style={[styles.opdsBox, { borderColor: c.border }]}>
        <OpdsNavigator catalogs={catalogs} onOpenEbook={(args) => openReader({ ...args })} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  topScroll: { padding: 16, paddingBottom: 8 },
  spinner: { marginVertical: 16 },
  section: { fontSize: 13, fontWeight: '700', marginTop: 16, letterSpacing: 0.5 },
  muted: { fontSize: 14, marginTop: 6 },
  recentRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 8 },
  recentMain: { flex: 1, borderWidth: StyleSheet.hairlineWidth, borderRadius: 10, padding: 12 },
  recentTitle: { fontSize: 16, fontWeight: '600' },
  opdsBox: { flex: 1, marginHorizontal: 16, marginBottom: 100, minHeight: 280, borderWidth: StyleSheet.hairlineWidth, borderRadius: 12, padding: 8 },
  headerBtn: { marginRight: 8, padding: 6 },
});
