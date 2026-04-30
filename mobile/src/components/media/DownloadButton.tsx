import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, useColorScheme } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type { MusicTrack } from '../../api/media';
import { getColors } from '../../theme/colors';
import { downloadTrack, isDownloaded, removeDownload, type DownloadProgress } from '../../utils/mediaDownloadStore';

type Props = {
  track: MusicTrack;
  serviceType?: string | null;
  parentId?: string | null;
  size?: number;
};

export function DownloadButton({ track, serviceType, parentId, size = 22 }: Props) {
  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const c = getColors(scheme);
  const [downloaded, setDownloaded] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const d = await isDownloaded(track.id);
      setDownloaded(d);
      setError(false);
    } catch {
      setDownloaded(false);
    } finally {
      setLoading(false);
    }
  }, [track.id]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const onPress = useCallback(async () => {
    if (busy) return;
    setError(false);
    if (downloaded) {
      setBusy(true);
      try {
        await removeDownload(track.id);
        setDownloaded(false);
      } finally {
        setBusy(false);
      }
      return;
    }
    setBusy(true);
    setProgress({ loaded: 0, total: 0, percent: 0 });
    try {
      await downloadTrack(track, {
        serviceType,
        parentId,
        onProgress: (p) => setProgress(p),
      });
      setDownloaded(true);
      setProgress(null);
    } catch {
      setError(true);
      setProgress(null);
    } finally {
      setBusy(false);
    }
  }, [busy, downloaded, parentId, serviceType, track]);

  if (loading) {
    return <ActivityIndicator size="small" color={c.textSecondary} />;
  }

  const icon = error ? 'alert-circle-outline' : downloaded ? 'checkmark-circle' : 'download-outline';
  const color = error ? c.danger : downloaded ? c.link : c.textSecondary;

  return (
    <Pressable
      onPress={() => void onPress()}
      disabled={busy}
      style={styles.hit}
      accessibilityLabel={downloaded ? 'Remove download' : 'Download'}
      accessibilityRole="button"
    >
      {busy && progress ? (
        <Text style={[styles.pct, { color: c.textSecondary }]}>{progress.percent}%</Text>
      ) : busy ? (
        <ActivityIndicator size="small" color={c.textSecondary} />
      ) : (
        <Ionicons name={icon as keyof typeof Ionicons.glyphMap} size={size} color={color} />
      )}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  hit: {
    minWidth: 36,
    minHeight: 36,
    alignItems: 'center',
    justifyContent: 'center',
  },
  pct: {
    fontSize: 11,
    fontWeight: '600',
  },
});
