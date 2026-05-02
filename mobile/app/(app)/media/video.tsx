import { useEvent, useEventListener } from 'expo';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
  useColorScheme,
} from 'react-native';
import { useVideoPlayer, VideoView } from 'expo-video';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { assertApiBaseUrl, getApiBaseUrl } from '../../../src/api/config';
import {
  buildEmbyHlsUrl,
  buildEmbyVideoStreamUrl,
  pickMobileSourceAndMode,
  postEmbyPlaybackInfo,
  reportEmbyPlaybackProgress,
  reportEmbyPlaybackStart,
  reportEmbyPlaybackStopped,
  type MobileSourcePick,
} from '../../../src/api/emby';
import { getStoredToken } from '../../../src/session/tokenStore';
import { stopEmbyPlaybackIfActive } from '../../../src/media/embySession';
import { getColors } from '../../../src/theme/colors';

function ticksToMillis(ticks: number): number {
  if (!Number.isFinite(ticks) || ticks <= 0) return 0;
  return Math.floor(ticks / 10_000);
}

function millisToTicks(ms: number): number {
  if (!Number.isFinite(ms) || ms < 0) return 0;
  return Math.floor(ms * 10_000);
}

type Session = {
  itemId: string;
  mediaSourceId: string;
  playSessionId: string;
  playMethod: string;
  audioStreamIndex: number | null;
};

export default function EmbyVideoScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const c = useMemo(() => getColors(scheme), [scheme]);
  const params = useLocalSearchParams<{ itemId?: string; title?: string; startTimeTicks?: string }>();
  const itemId = typeof params.itemId === 'string' ? params.itemId : params.itemId?.[0] ?? '';
  const title = typeof params.title === 'string' ? params.title : params.title?.[0] ?? 'Video';
  const startTicksRaw = typeof params.startTimeTicks === 'string' ? params.startTimeTicks : params.startTimeTicks?.[0];
  const startTicks = startTicksRaw != null ? Math.max(0, parseInt(startTicksRaw, 10) || 0) : 0;

  const sessionRef = useRef<Session | null>(null);
  const lastProgressReportRef = useRef(Date.now());
  const didSeekResumeRef = useRef(false);
  const pickRef = useRef<MobileSourcePick | null>(null);
  const hlsFallbackAttemptedRef = useRef(false);

  const [phase, setPhase] = useState<'loading' | 'ready' | 'error'>('loading');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [uri, setUri] = useState<string | null>(null);
  const [pick, setPick] = useState<MobileSourcePick | null>(null);

  const startMillis = useMemo(() => ticksToMillis(startTicks), [startTicks]);

  const player = useVideoPlayer(null, (p) => {
    p.timeUpdateEventInterval = 10;
  });

  const { isPlaying } = useEvent(player, 'playingChange', { isPlaying: player.playing });
  const isPlayingRef = useRef(isPlaying);
  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    pickRef.current = pick;
  }, [pick]);

  useEffect(() => {
    const base = getApiBaseUrl();
    if (base?.startsWith('http://')) {
      console.warn(
        'Video: API base URL uses HTTP; iOS App Transport Security may block playback without ATS exceptions.'
      );
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!itemId) {
        setPhase('error');
        setErrorMsg('Missing item');
        return;
      }
      await stopEmbyPlaybackIfActive();
      try {
        const base = assertApiBaseUrl();
        const token = (await getStoredToken()) ?? '';
        if (!token) {
          setPhase('error');
          setErrorMsg('Not signed in');
          return;
        }
        const info = await postEmbyPlaybackInfo(itemId, {
          start_time_ticks: startTicks,
        });
        hlsFallbackAttemptedRef.current = false;
        const parsed = pickMobileSourceAndMode(info as Record<string, unknown>, {
          preferDirectOverHls: Platform.OS === 'ios',
        });
        if (!parsed) {
          setPhase('error');
          setErrorMsg('Could not start playback');
          return;
        }
        sessionRef.current = {
          itemId,
          mediaSourceId: parsed.mediaSourceId,
          playSessionId: parsed.playSessionId,
          playMethod: parsed.playMethod,
          audioStreamIndex: parsed.audioStreamIndex,
        };
        await reportEmbyPlaybackStart({
          ItemId: itemId,
          MediaSourceId: parsed.mediaSourceId,
          PlaySessionId: parsed.playSessionId,
          PositionTicks: startTicks,
          PlayMethod: parsed.playMethod,
          IsPaused: false,
          ...(parsed.audioStreamIndex != null ? { AudioStreamIndex: parsed.audioStreamIndex } : {}),
        });
        const streamUrl = parsed.useHls
          ? buildEmbyHlsUrl(itemId, base, token, {
              mediaSourceId: parsed.mediaSourceId,
              playSessionId: parsed.playSessionId,
              audioStreamIndex: parsed.audioStreamIndex,
            })
          : buildEmbyVideoStreamUrl(itemId, base, token, {
              mediaSourceId: parsed.mediaSourceId,
              playSessionId: parsed.playSessionId,
              startTimeTicks: startTicks,
              static: true,
              audioStreamIndex: parsed.audioStreamIndex,
            });
        if (cancelled) return;
        lastProgressReportRef.current = Date.now();
        setPick(parsed);
        setUri(streamUrl);
        setPhase('ready');
      } catch {
        if (cancelled) return;
        setPhase('error');
        setErrorMsg('Could not load video');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [itemId, startTicks]);

  useEffect(() => {
    if (!uri || phase !== 'ready') return;
    player.replace(uri);
    player.play();
  }, [uri, phase, player]);

  useEventListener(player, 'statusChange', ({ status, error }) => {
    if (status === 'error') {
      const msg = error != null && typeof (error as { message?: string }).message === 'string'
        ? (error as { message: string }).message
        : 'Playback error';
      const p = pickRef.current;
      const couldTryDirectFallback =
        Platform.OS === 'ios' &&
        p?.useHls === true &&
        !hlsFallbackAttemptedRef.current &&
        itemId;
      if (couldTryDirectFallback) {
        hlsFallbackAttemptedRef.current = true;
        void (async () => {
          try {
            const base = assertApiBaseUrl();
            const token = (await getStoredToken()) ?? '';
            const directUrl = buildEmbyVideoStreamUrl(itemId, base, token, {
              mediaSourceId: p.mediaSourceId,
              playSessionId: p.playSessionId,
              startTimeTicks: startTicks,
              static: true,
              audioStreamIndex: p.audioStreamIndex,
            });
            const newPick: MobileSourcePick = {
              ...p,
              useHls: false,
              playMethod: 'DirectStream',
            };
            pickRef.current = newPick;
            setPick(newPick);
            sessionRef.current = sessionRef.current
              ? { ...sessionRef.current, playMethod: 'DirectStream' }
              : sessionRef.current;
            setUri(directUrl);
            setPhase('ready');
            setErrorMsg(null);
            didSeekResumeRef.current = false;
          } catch {
            setPhase('error');
            setErrorMsg(msg);
          }
        })();
        return;
      }
      setPhase('error');
      setErrorMsg(msg);
    }
    if (status === 'readyToPlay' && !didSeekResumeRef.current && startMillis > 0) {
      didSeekResumeRef.current = true;
      try {
        player.currentTime = startMillis / 1000;
      } catch {
        // ignore seek errors
      }
    }
  });

  useEventListener(player, 'timeUpdate', ({ currentTime }) => {
    const p = pickRef.current;
    const s = sessionRef.current;
    if (!p || !s) return;
    const now = Date.now();
    if (now - lastProgressReportRef.current < 10_000) return;
    if (!isPlayingRef.current) return;
    lastProgressReportRef.current = now;
    void reportEmbyPlaybackProgress({
      ItemId: s.itemId,
      MediaSourceId: s.mediaSourceId,
      PlaySessionId: s.playSessionId,
      PositionTicks: millisToTicks(currentTime * 1000),
      PlayMethod: p.playMethod,
      IsPaused: false,
      EventName: 'TimeUpdate',
      ...(p.audioStreamIndex != null ? { AudioStreamIndex: p.audioStreamIndex } : {}),
    }).catch(() => {});
  });

  const reportStopped = useCallback(async (positionSeconds: number) => {
    const s = sessionRef.current;
    if (!s) return;
    sessionRef.current = null;
    try {
      await reportEmbyPlaybackStopped({
        ItemId: s.itemId,
        MediaSourceId: s.mediaSourceId,
        PlaySessionId: s.playSessionId,
        PositionTicks: millisToTicks(positionSeconds * 1000),
      });
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    return () => {
      let pos = 0;
      try {
        pos = player.currentTime;
      } catch {
        pos = 0;
      }
      void reportStopped(pos);
    };
  }, [player, reportStopped]);

  if (phase === 'error') {
    return (
      <View style={[styles.center, { backgroundColor: '#000', paddingTop: insets.top }]}>
        <Pressable
          onPress={() => router.back()}
          style={[styles.closeBtn, { top: insets.top + 8 }]}
          accessibilityLabel="Close"
        >
          <Ionicons name="close" size={28} color="#fff" />
        </Pressable>
        <Text style={styles.errorText}>{errorMsg ?? 'Error'}</Text>
        <Pressable onPress={() => router.back()} style={styles.backLink}>
          <Text style={{ color: c.link, fontSize: 16 }}>Go back</Text>
        </Pressable>
      </View>
    );
  }

  if (phase === 'loading' || !uri || !pick) {
    return (
      <View style={[styles.center, { backgroundColor: '#000', paddingTop: insets.top }]}>
        <Pressable
          onPress={() => router.back()}
          style={[styles.closeBtn, { top: insets.top + 8 }]}
          accessibilityLabel="Close"
        >
          <Ionicons name="close" size={28} color="#fff" />
        </Pressable>
        <ActivityIndicator color="#fff" size="large" />
        <Text style={styles.loadingTitle} numberOfLines={1}>
          {title}
        </Text>
      </View>
    );
  }

  return (
    <View style={[styles.root, { paddingTop: insets.top, backgroundColor: '#000' }]}>
      <Pressable
        onPress={() => router.back()}
        style={[styles.closeBtn, { top: insets.top + 8 }]}
        accessibilityLabel="Close"
      >
        <Ionicons name="close" size={28} color="#fff" />
      </Pressable>
      <VideoView
        style={styles.video}
        player={player}
        nativeControls
        contentFit="contain"
        allowsFullscreen
      />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 24 },
  video: { flex: 1, width: '100%' },
  closeBtn: {
    position: 'absolute',
    right: 12,
    zIndex: 10,
    padding: 8,
  },
  loadingTitle: { color: '#aaa', marginTop: 16, fontSize: 15, maxWidth: '90%' },
  errorText: { color: '#fff', fontSize: 16, textAlign: 'center', marginBottom: 16 },
  backLink: { padding: 12 },
});
