import { useMemo } from 'react';
import { Image, Pressable, StyleSheet, Text, View, useColorScheme } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { State } from 'react-native-track-player';
import { MINI_PLAYER_STRIP_HEIGHT } from '../../constants/dock';
import { useMediaPlayer } from '../../context/MediaPlayerContext';
import { getColors } from '../../theme/colors';

export function MiniPlayer() {
  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const c = useMemo(() => getColors(scheme), [scheme]);
  const {
    hasActiveSession,
    activeTrack,
    playbackState,
    pause,
    resume,
    skipToNext,
    setFullPlayerVisible,
  } = useMediaPlayer();

  if (!hasActiveSession || !activeTrack) {
    return null;
  }

  const playing = playbackState === State.Playing;
  const art = typeof activeTrack.artwork === 'string' ? activeTrack.artwork : undefined;

  return (
    <Pressable
      style={[styles.wrap, { backgroundColor: c.surface, borderTopColor: c.border }]}
      onPress={() => setFullPlayerVisible(true)}
      accessibilityRole="button"
      accessibilityLabel="Open now playing"
    >
      {art ? (
        <Image source={{ uri: art }} style={styles.art} />
      ) : (
        <View style={[styles.artPlaceholder, { backgroundColor: c.chipBg }]}>
          <Ionicons name="musical-notes" size={22} color={c.textSecondary} />
        </View>
      )}
      <View style={styles.textCol}>
        <Text style={[styles.title, { color: c.text }]} numberOfLines={1}>
          {activeTrack.title}
        </Text>
        <Text style={[styles.sub, { color: c.textSecondary }]} numberOfLines={1}>
          {activeTrack.artist ?? ''}
        </Text>
      </View>
      <Pressable
        onPress={(e) => {
          e.stopPropagation();
          void (playing ? pause() : resume());
        }}
        style={styles.iconBtn}
        accessibilityLabel={playing ? 'Pause' : 'Play'}
        accessibilityRole="button"
      >
        <Ionicons name={playing ? 'pause' : 'play'} size={28} color={c.text} />
      </Pressable>
      <Pressable
        onPress={(e) => {
          e.stopPropagation();
          void skipToNext();
        }}
        style={styles.iconBtn}
        accessibilityLabel="Next track"
        accessibilityRole="button"
      >
        <Ionicons name="play-skip-forward" size={24} color={c.text} />
      </Pressable>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  wrap: {
    flexDirection: 'row',
    alignItems: 'center',
    minHeight: MINI_PLAYER_STRIP_HEIGHT,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderTopWidth: StyleSheet.hairlineWidth,
    gap: 10,
  },
  art: {
    width: 44,
    height: 44,
    borderRadius: 6,
  },
  artPlaceholder: {
    width: 44,
    height: 44,
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
  },
  textCol: {
    flex: 1,
    minWidth: 0,
  },
  title: {
    fontSize: 15,
    fontWeight: '600',
  },
  sub: {
    fontSize: 12,
    marginTop: 2,
  },
  iconBtn: {
    padding: 6,
  },
});
