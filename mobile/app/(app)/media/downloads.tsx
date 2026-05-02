import { useCallback, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  useColorScheme,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import {
  clearAllDownloads,
  listDownloads,
  removeDownload,
  type MediaDownloadEntry,
} from '../../../src/utils/mediaDownloadStore';
import { getColors } from '../../../src/theme/colors';

function formatBytes(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export default function MediaDownloadsScreen() {
  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const c = getColors(scheme);
  const [items, setItems] = useState<MediaDownloadEntry[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const list = await listDownloads();
      setItems(list);
    } finally {
      setLoading(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load])
  );

  const onRemove = useCallback(
    (id: string) => {
      Alert.alert('Remove download', 'Delete this file from local storage?', [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Remove',
          style: 'destructive',
          onPress: () => {
            void (async () => {
              await removeDownload(id);
              await load();
            })();
          },
        },
      ]);
    },
    [load]
  );

  const onClearAll = useCallback(() => {
    Alert.alert('Clear all', 'Remove all downloaded media from this device?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Clear',
        style: 'destructive',
        onPress: () => {
          void (async () => {
            await clearAllDownloads();
            await load();
          })();
        },
      },
    ]);
  }, [load]);

  return (
    <View style={[styles.root, { backgroundColor: c.background }]}>
      <View style={[styles.toolbar, { borderBottomColor: c.border }]}>
        <Text style={[styles.hint, { color: c.textSecondary }]}>
          Long-press a row to remove it from this device.
        </Text>
        <Pressable
          onPress={onClearAll}
          disabled={!items.length}
          style={[styles.clearBtn, { opacity: items.length ? 1 : 0.4 }]}
        >
          <Text style={[styles.clearTxt, { color: c.danger }]}>Clear all</Text>
        </Pressable>
      </View>
      {loading ? (
        <ActivityIndicator style={{ marginTop: 24 }} color={c.textSecondary} />
      ) : (
        <ScrollView contentContainerStyle={styles.list}>
          {items.map((e) => (
            <Pressable
              key={e.track_id}
              onLongPress={() => onRemove(e.track_id)}
              delayLongPress={450}
              style={[styles.row, { backgroundColor: c.surface, borderColor: c.border }]}
              accessibilityLabel={`${e.title}, saved on device. Long press to remove.`}
              accessibilityRole="button"
            >
              <View style={styles.rowMid}>
                <Text style={[styles.title, { color: c.text }]} numberOfLines={2}>
                  {e.title}
                </Text>
                <Text style={[styles.meta, { color: c.textSecondary }]} numberOfLines={2}>
                  {[e.artist, formatBytes(e.file_size), 'Saved on device'].filter(Boolean).join(' · ')}
                </Text>
              </View>
            </Pressable>
          ))}
          {!items.length && (
            <Text style={[styles.empty, { color: c.textSecondary }]}>No downloads yet.</Text>
          )}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  toolbar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  hint: { flex: 1, fontSize: 12, lineHeight: 16 },
  clearBtn: { paddingVertical: 8, paddingHorizontal: 12 },
  clearTxt: { fontSize: 15, fontWeight: '600' },
  list: { padding: 16, paddingBottom: 120, gap: 10 },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    borderRadius: 10,
    borderWidth: StyleSheet.hairlineWidth,
    gap: 12,
  },
  rowMid: { flex: 1, minWidth: 0 },
  title: { fontSize: 16, fontWeight: '600' },
  meta: { fontSize: 13, marginTop: 4 },
  empty: { textAlign: 'center', marginTop: 32, fontSize: 15 },
});
