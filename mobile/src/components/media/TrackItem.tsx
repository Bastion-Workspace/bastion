import { useMemo } from 'react';
import { Pressable, StyleSheet, Text, View, useColorScheme } from 'react-native';
import type { MusicTrack } from '../../api/media';
import { getColors } from '../../theme/colors';
import { DownloadButton } from './DownloadButton';

function formatDuration(seconds?: number | null): string {
  if (seconds == null || Number.isNaN(seconds)) return '';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

type Props = {
  track: MusicTrack;
  index: number;
  serviceType?: string | null;
  parentId?: string | null;
  onPlay: (index: number) => void;
};

export function TrackItem({ track, index, serviceType, parentId, onPlay }: Props) {
  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const c = useMemo(() => getColors(scheme), [scheme]);
  const dur = formatDuration(track.duration);

  return (
    <Pressable
      style={[styles.row, { borderBottomColor: c.border }]}
      onPress={() => onPlay(index)}
      accessibilityRole="button"
      accessibilityLabel={`Play ${track.title}`}
    >
      <Text style={[styles.num, { color: c.textSecondary }]}>{index + 1}</Text>
      <View style={styles.mid}>
        <Text style={[styles.title, { color: c.text }]} numberOfLines={1}>
          {track.title}
        </Text>
        {(track.artist || dur) && (
          <Text style={[styles.sub, { color: c.textSecondary }]} numberOfLines={1}>
            {[track.artist, dur].filter(Boolean).join(' · ')}
          </Text>
        )}
      </View>
      <DownloadButton track={track} serviceType={serviceType} parentId={parentId} />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 4,
    borderBottomWidth: StyleSheet.hairlineWidth,
    gap: 8,
  },
  num: {
    width: 28,
    fontSize: 13,
    textAlign: 'right',
  },
  mid: {
    flex: 1,
    minWidth: 0,
  },
  title: {
    fontSize: 16,
    fontWeight: '500',
  },
  sub: {
    fontSize: 12,
    marginTop: 2,
  },
});
