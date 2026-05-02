import { useMemo, useState } from 'react';
import { Pressable, StyleSheet, Text, View, useColorScheme } from 'react-native';
import type { MusicTrack } from '../../api/media';
import { getColors } from '../../theme/colors';
import { DownloadButton } from './DownloadButton';

function formatDuration(seconds?: number | null): string {
  if (seconds == null || seconds === 0 || Number.isNaN(seconds)) return '';
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
  const [onDevice, setOnDevice] = useState(false);

  const meta = track.metadata as Record<string, unknown> | null | undefined;
  const rawPub = meta?.published_date ?? meta?.publishedAt;
  let dateLabel: string | null = null;
  if (typeof rawPub === 'string' && rawPub.trim()) {
    const d = new Date(rawPub);
    if (!Number.isNaN(d.getTime())) {
      dateLabel = d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    }
  }
  const isAbs = serviceType === 'audiobookshelf';
  const isAbsEpisodeRow = isAbs && Boolean(dateLabel);
  const chapterLead =
    isAbs && !isAbsEpisodeRow && track.track_number != null
      ? `Chapter ${track.track_number}`
      : null;

  const subParts: (string | null | undefined)[] = [];
  if (isAbs) {
    if (isAbsEpisodeRow) subParts.push(dateLabel, track.artist, dur);
    else subParts.push(chapterLead, track.artist, dur);
  } else {
    subParts.push(track.artist, dur);
  }
  if (onDevice) subParts.push('On device');
  const subFiltered = subParts.filter((x): x is string => Boolean(x && String(x).trim()));

  return (
    <Pressable
      style={[styles.row, { borderBottomColor: c.border }]}
      onPress={() => onPlay(index)}
      accessibilityRole="button"
      accessibilityLabel={`Play ${track.title}`}
    >
      <Text style={[styles.num, { color: c.textSecondary }]}>{index + 1}</Text>
      <View style={styles.mid}>
        <Text style={[styles.title, { color: c.text }]} numberOfLines={2}>
          {track.title}
        </Text>
        {subFiltered.length > 0 && (
          <Text style={[styles.sub, { color: c.textSecondary }]} numberOfLines={2}>
            {subFiltered.join(' · ')}
          </Text>
        )}
      </View>
      <View style={styles.downloadWrap}>
        <DownloadButton
          track={track}
          serviceType={serviceType}
          parentId={parentId}
          onDownloadedChange={setOnDevice}
        />
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: 8,
    paddingHorizontal: 4,
    borderBottomWidth: StyleSheet.hairlineWidth,
    gap: 8,
  },
  num: {
    width: 26,
    fontSize: 12,
    textAlign: 'right',
    paddingTop: 1,
  },
  mid: {
    flex: 1,
    minWidth: 0,
  },
  title: {
    fontSize: 14,
    fontWeight: '500',
  },
  sub: {
    fontSize: 11,
    marginTop: 2,
  },
  downloadWrap: {
    alignSelf: 'center',
  },
});
