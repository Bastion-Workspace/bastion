import { useCallback, useMemo, useState } from 'react';
import {
  FlatList,
  Image,
  Modal,
  Pressable,
  StyleSheet,
  Text,
  View,
  useColorScheme,
  useWindowDimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { State, useProgress } from 'react-native-track-player';
import TrackPlayer from 'react-native-track-player';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useMediaPlayer } from '../../context/MediaPlayerContext';
import { getColors } from '../../theme/colors';
import { DownloadButton } from './DownloadButton';

export function FullPlayerModal() {
  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const c = useMemo(() => getColors(scheme), [scheme]);
  const insets = useSafeAreaInsets();
  const { width } = useWindowDimensions();
  const {
    fullPlayerVisible,
    setFullPlayerVisible,
    activeTrack,
    playbackState,
    playMeta,
    pause,
    resume,
    skipToNext,
    skipToPrevious,
    seekTo,
    stop,
  } = useMediaPlayer();
  const progress = useProgress(400);
  const [tab, setTab] = useState<'now' | 'queue'>('now');
  const [queue, setQueue] = useState<Awaited<ReturnType<typeof TrackPlayer.getQueue>>>([]);
  const [seekBarW, setSeekBarW] = useState(0);

  const position = fullPlayerVisible ? progress.position : 0;
  const duration = fullPlayerVisible ? progress.duration : 0;

  const loadQueue = useCallback(async () => {
    try {
      const q = await TrackPlayer.getQueue();
      setQueue(q);
    } catch {
      setQueue([]);
    }
  }, []);

  const playing = playbackState === State.Playing;
  const art = activeTrack && typeof activeTrack.artwork === 'string' ? activeTrack.artwork : undefined;
  const seekW = Math.max(0, width - 48);
  const fillW = duration > 0 && seekBarW > 0 ? (position / duration) * seekBarW : 0;

  const onSeekBarPress = useCallback(
    (locationX: number) => {
      if (!duration || seekBarW <= 0) return;
      const ratio = Math.max(0, Math.min(1, locationX / seekBarW));
      void seekTo(ratio * duration);
    },
    [duration, seekBarW, seekTo]
  );

  if (!fullPlayerVisible) {
    return null;
  }

  return (
    <Modal
      visible
      animationType="slide"
      presentationStyle="fullScreen"
      onShow={() => void loadQueue()}
      onRequestClose={() => setFullPlayerVisible(false)}
    >
      <View style={[styles.root, { backgroundColor: c.background, paddingTop: insets.top + 8 }]}>
        <View style={styles.headerRow}>
          <Pressable onPress={() => setFullPlayerVisible(false)} accessibilityLabel="Close" accessibilityRole="button">
            <Ionicons name="chevron-down" size={28} color={c.text} />
          </Pressable>
          <Text style={[styles.headerTitle, { color: c.textSecondary }]}>Now playing</Text>
          <View style={{ width: 28 }} />
        </View>

        <View style={styles.tabRow}>
          <Pressable
            onPress={() => setTab('now')}
            style={[styles.tabHit, tab === 'now' && { borderBottomColor: c.link, borderBottomWidth: 2 }]}
          >
            <Text style={[styles.tabTxt, { color: tab === 'now' ? c.text : c.textSecondary }]}>Player</Text>
          </Pressable>
          <Pressable
            onPress={() => {
              setTab('queue');
              void loadQueue();
            }}
            style={[styles.tabHit, tab === 'queue' && { borderBottomColor: c.link, borderBottomWidth: 2 }]}
          >
            <Text style={[styles.tabTxt, { color: tab === 'queue' ? c.text : c.textSecondary }]}>Queue</Text>
          </Pressable>
        </View>

        {tab === 'now' && activeTrack && (
          <View style={styles.nowBody}>
            {art ? (
              <Image source={{ uri: art }} style={[styles.bigArt, { width: seekW }]} />
            ) : (
              <View
                style={[
                  styles.bigArtPlaceholder,
                  { width: seekW, backgroundColor: c.surfaceMuted },
                ]}
              >
                <Ionicons name="musical-notes-outline" size={80} color={c.textSecondary} />
              </View>
            )}
            <Text style={[styles.trackTitle, { color: c.text }]}>{activeTrack.title}</Text>
            <Text style={[styles.trackArtist, { color: c.textSecondary }]}>{activeTrack.artist ?? ''}</Text>

            {activeTrack.id ? (
              <View style={styles.dlRow}>
                <Text style={[styles.dlLabel, { color: c.textSecondary }]}>Offline</Text>
                <DownloadButton
                  track={{
                    id: String(activeTrack.id),
                    title: String(activeTrack.title ?? ''),
                    artist: activeTrack.artist ?? undefined,
                    album: activeTrack.album ?? undefined,
                    duration: activeTrack.duration,
                    service_type: playMeta.serviceType ?? undefined,
                  }}
                  serviceType={playMeta.serviceType}
                  parentId={playMeta.parentId}
                />
              </View>
            ) : null}

            <Pressable
              onLayout={(e) => setSeekBarW(e.nativeEvent.layout.width)}
              onPress={(ev) => onSeekBarPress(ev.nativeEvent.locationX)}
              style={[styles.seekTrack, { width: seekW, backgroundColor: c.border }]}
              accessibilityLabel="Seek"
              accessibilityRole="adjustable"
            >
              <View style={[styles.seekFill, { width: fillW, backgroundColor: c.link }]} />
            </Pressable>
            <View style={styles.timeRow}>
              <Text style={[styles.timeTxt, { color: c.textSecondary }]}>{formatClock(position)}</Text>
              <Text style={[styles.timeTxt, { color: c.textSecondary }]}>{formatClock(duration)}</Text>
            </View>

            <View style={styles.controls}>
              <Pressable onPress={() => void skipToPrevious()} accessibilityLabel="Previous">
                <Ionicons name="play-skip-back" size={36} color={c.text} />
              </Pressable>
              <Pressable onPress={() => void (playing ? pause() : resume())} accessibilityLabel={playing ? 'Pause' : 'Play'}>
                <Ionicons name={playing ? 'pause-circle' : 'play-circle'} size={64} color={c.text} />
              </Pressable>
              <Pressable onPress={() => void skipToNext()} accessibilityLabel="Next">
                <Ionicons name="play-skip-forward" size={36} color={c.text} />
              </Pressable>
            </View>

            <View style={styles.bottomRow}>
              <Pressable onPress={() => void stop()} accessibilityLabel="Stop and clear queue">
                <Text style={[styles.stopTxt, { color: c.danger }]}>Stop</Text>
              </Pressable>
            </View>
          </View>
        )}

        {tab === 'queue' && (
          <FlatList
            data={queue}
            keyExtractor={(item, i) => `${String(item.id ?? i)}`}
            contentContainerStyle={{ paddingBottom: insets.bottom + 24 }}
            renderItem={({ item, index }) => (
              <Pressable
                style={[styles.qRow, { borderBottomColor: c.border }]}
                onPress={() => {
                  void TrackPlayer.skip(index);
                }}
              >
                <Text style={[styles.qTitle, { color: c.text }]} numberOfLines={1}>
                  {item.title}
                </Text>
                <Text style={[styles.qSub, { color: c.textSecondary }]} numberOfLines={1}>
                  {item.artist ?? ''}
                </Text>
              </Pressable>
            )}
            ListEmptyComponent={
              <Text style={[styles.empty, { color: c.textSecondary }]}>Queue is empty.</Text>
            }
          />
        )}
      </View>
    </Modal>
  );
}

function formatClock(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '0:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    paddingHorizontal: 16,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  headerTitle: {
    fontSize: 14,
    fontWeight: '600',
  },
  tabRow: {
    flexDirection: 'row',
    gap: 20,
    marginBottom: 16,
  },
  tabHit: {
    paddingBottom: 6,
  },
  tabTxt: {
    fontSize: 15,
    fontWeight: '600',
  },
  nowBody: {
    alignItems: 'center',
  },
  bigArt: {
    height: 280,
    borderRadius: 12,
    marginBottom: 20,
  },
  bigArtPlaceholder: {
    height: 280,
    borderRadius: 12,
    marginBottom: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  trackTitle: {
    fontSize: 22,
    fontWeight: '700',
    textAlign: 'center',
    alignSelf: 'stretch',
  },
  trackArtist: {
    fontSize: 16,
    marginTop: 6,
    textAlign: 'center',
  },
  dlRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginTop: 16,
    alignSelf: 'flex-start',
  },
  dlLabel: {
    fontSize: 14,
  },
  seekTrack: {
    height: 6,
    borderRadius: 3,
    marginTop: 24,
    overflow: 'hidden',
    justifyContent: 'center',
  },
  seekFill: {
    height: 6,
    borderRadius: 3,
    position: 'absolute',
    left: 0,
    top: 0,
  },
  timeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignSelf: 'stretch',
    marginTop: 8,
  },
  timeTxt: {
    fontSize: 12,
  },
  controls: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 36,
    marginTop: 28,
  },
  bottomRow: {
    marginTop: 32,
  },
  stopTxt: {
    fontSize: 16,
    fontWeight: '600',
  },
  qRow: {
    paddingVertical: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  qTitle: {
    fontSize: 16,
    fontWeight: '500',
  },
  qSub: {
    fontSize: 13,
    marginTop: 4,
  },
  empty: {
    textAlign: 'center',
    marginTop: 40,
    fontSize: 15,
  },
});
