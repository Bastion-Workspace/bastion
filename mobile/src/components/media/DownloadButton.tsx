import { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Pressable,
  StyleSheet,
  Text,
  useColorScheme,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type { MusicTrack } from '../../api/media';
import { getColors } from '../../theme/colors';
import {
  downloadTrack,
  isDownloaded,
  removeDownload,
  type DownloadProgress,
} from '../../utils/mediaDownloadStore';

type Props = {
  track: MusicTrack;
  serviceType?: string | null;
  parentId?: string | null;
  size?: number;
  /** When true, show a short status line next to the icon (e.g. full player). */
  showCaption?: boolean;
  /**
   * When true and the track is already on device, a single tap does not delete;
   * use long-press and confirm to remove (avoids accidental removal).
   */
  removeRequiresLongPress?: boolean;
  /** Notified whenever local "downloaded" state changes. */
  onDownloadedChange?: (downloaded: boolean) => void;
};

export function DownloadButton({
  track,
  serviceType,
  parentId,
  size = 22,
  showCaption = false,
  removeRequiresLongPress = true,
  onDownloadedChange,
}: Props) {
  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const c = getColors(scheme);
  const [downloaded, setDownloaded] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState(false);

  const setDownloadedAndNotify = useCallback(
    (v: boolean) => {
      setDownloaded(v);
      onDownloadedChange?.(v);
    },
    [onDownloadedChange]
  );

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const d = await isDownloaded(track.id);
      setDownloadedAndNotify(d);
      setError(false);
    } catch {
      setDownloadedAndNotify(false);
    } finally {
      setLoading(false);
    }
  }, [setDownloadedAndNotify, track.id]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const runRemove = useCallback(async () => {
    setBusy(true);
    try {
      await removeDownload(track.id);
      setDownloadedAndNotify(false);
    } finally {
      setBusy(false);
    }
  }, [setDownloadedAndNotify, track.id]);

  const confirmRemove = useCallback(() => {
    Alert.alert(
      'Remove from device',
      'Delete this downloaded file from local storage?',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Remove', style: 'destructive', onPress: () => void runRemove() },
      ]
    );
  }, [runRemove]);

  const onPress = useCallback(async () => {
    if (busy) return;
    if (downloaded) {
      if (removeRequiresLongPress) {
        return;
      }
      void runRemove();
      return;
    }
    setError(false);
    setBusy(true);
    setProgress({ loaded: 0, total: 0, percent: 0 });
    try {
      await downloadTrack(track, {
        serviceType,
        parentId,
        onProgress: (p) => setProgress(p),
      });
      setDownloadedAndNotify(true);
      setProgress(null);
    } catch {
      setError(true);
      setProgress(null);
    } finally {
      setBusy(false);
    }
  }, [
    busy,
    downloaded,
    parentId,
    removeRequiresLongPress,
    runRemove,
    serviceType,
    setDownloadedAndNotify,
    track,
  ]);

  if (loading) {
    return showCaption ? (
      <View style={styles.captionRow}>
        <ActivityIndicator size="small" color={c.textSecondary} />
        <Text style={[styles.caption, { color: c.textSecondary }]}>Checking…</Text>
      </View>
    ) : (
      <ActivityIndicator size="small" color={c.textSecondary} />
    );
  }

  const icon = error
    ? 'alert-circle-outline'
    : downloaded
      ? 'checkmark-circle'
      : 'download-outline';
  const color = error ? c.danger : downloaded ? c.link : c.textSecondary;

  let caption = '';
  if (error) {
    caption = 'Download failed · tap to retry';
  } else if (busy && progress) {
    caption = `Downloading… ${progress.percent}%`;
  } else if (busy) {
    caption = 'Downloading…';
  } else if (downloaded) {
    caption = removeRequiresLongPress
      ? 'Saved on device · hold icon to remove'
      : 'Saved on device';
  } else {
    caption = 'Save for offline · tap icon';
  }

  const a11yLabel = downloaded
    ? removeRequiresLongPress
      ? 'Downloaded, long press to remove'
      : 'Remove download'
    : 'Download for offline';

  const iconPressable = (
    <Pressable
      onPress={() => void onPress()}
      onLongPress={() => {
        if (downloaded && removeRequiresLongPress) confirmRemove();
      }}
      disabled={busy}
      style={styles.hit}
      accessibilityLabel={a11yLabel}
      accessibilityHint={
        downloaded && removeRequiresLongPress
          ? 'Long press to remove from this device'
          : undefined
      }
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

  if (!showCaption) {
    return iconPressable;
  }

  return (
    <View style={styles.captionRow}>
      {iconPressable}
      <Text style={[styles.caption, { color: c.textSecondary }]} numberOfLines={2}>
        {caption}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  captionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginTop: 12,
    alignSelf: 'stretch',
  },
  caption: {
    flex: 1,
    fontSize: 13,
    lineHeight: 18,
  },
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
