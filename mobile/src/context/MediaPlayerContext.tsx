import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import * as FileSystem from 'expo-file-system';
import TrackPlayer, {
  AppKilledPlaybackBehavior,
  Capability,
  Event,
  State,
  useActiveTrack,
  usePlaybackState,
  type Track,
} from 'react-native-track-player';
import type { MusicTrack } from '../api/media';
import { getCoverArtUrl, getStreamProxyUrl } from '../api/media';
import { reportEmbyPlaybackStopped } from '../api/emby';
import {
  beginEmbyPlaybackForItem,
  embySessionRef,
  reportEmbyProgressFromPlayer,
  stopEmbyPlaybackIfActive,
} from '../media/embySession';
import { getLocalPath } from '../utils/mediaDownloadStore';

const POSITIONS_REL = 'media/positions.json';

async function positionsFilePath(): Promise<string> {
  const base = FileSystem.documentDirectory;
  if (!base) throw new Error('documentDirectory is not available');
  return `${base}${POSITIONS_REL}`;
}

async function readPositions(): Promise<Record<string, number>> {
  try {
    const p = await positionsFilePath();
    const info = await FileSystem.getInfoAsync(p);
    if (!info.exists) return {};
    const raw = await FileSystem.readAsStringAsync(p);
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, number>;
    }
    return {};
  } catch {
    return {};
  }
}

async function writePositionsFile(data: Record<string, number>): Promise<void> {
  const base = FileSystem.documentDirectory;
  if (!base) throw new Error('documentDirectory is not available');
  const dir = `${base}media/`;
  const dInfo = await FileSystem.getInfoAsync(dir);
  if (!dInfo.exists) {
    await FileSystem.makeDirectoryAsync(dir, { intermediates: true });
  }
  await FileSystem.writeAsStringAsync(await positionsFilePath(), JSON.stringify(data), {
    encoding: FileSystem.EncodingType.UTF8,
  });
}

export type PlayQueueMeta = {
  serviceType?: string | null;
  parentId?: string | null;
};

function normalizePlaybackState(raw: State | { state: State | undefined } | object | undefined): State {
  if (raw == null) return State.None;
  if (typeof raw === 'string' || typeof raw === 'number') return raw as State;
  if (typeof raw === 'object' && 'state' in raw) {
    const s = (raw as { state: State | undefined }).state;
    if (s == null) return State.None;
    return s;
  }
  return State.None;
}

type MediaPlayerContextValue = {
  ready: boolean;
  activeTrack: Track | undefined;
  /** Last queue metadata (for downloads / stream options). */
  playMeta: PlayQueueMeta;
  /** True when a track is loaded (queue may be paused). */
  hasActiveSession: boolean;
  replaceQueueAndPlay: (tracks: MusicTrack[], startIndex: number, meta: PlayQueueMeta) => Promise<void>;
  pause: () => Promise<void>;
  resume: () => Promise<void>;
  seekTo: (seconds: number) => Promise<void>;
  skipToNext: () => Promise<void>;
  skipToPrevious: () => Promise<void>;
  stop: () => Promise<void>;
  fullPlayerVisible: boolean;
  setFullPlayerVisible: (v: boolean) => void;
  playbackState: State;
  shuffleEnabled: boolean;
  toggleShuffle: () => void;
};

const MediaPlayerContext = createContext<MediaPlayerContextValue | null>(null);

export function MediaPlayerProvider({ children }: { children: React.ReactNode }) {
  const [ready, setReady] = useState(false);
  const [fullPlayerVisible, setFullPlayerVisible] = useState(false);
  const [shuffleEnabled, setShuffleEnabled] = useState(false);
  const [playMeta, setPlayMeta] = useState<PlayQueueMeta>({});
  const playMetaRef = useRef<PlayQueueMeta>({});
  const positionsRef = useRef<Record<string, number>>({});
  const lastEmbyProgressAtRef = useRef(0);
  const activeTrack = useActiveTrack();
  const rawState = usePlaybackState();
  const playbackState = normalizePlaybackState(rawState as State | { state: State | undefined });

  playMetaRef.current = playMeta;

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await TrackPlayer.setupPlayer({
          autoHandleInterruptions: true,
        });
        await TrackPlayer.updateOptions({
          capabilities: [
            Capability.Play,
            Capability.Pause,
            Capability.SkipToNext,
            Capability.SkipToPrevious,
            Capability.SeekTo,
            Capability.Stop,
          ],
          compactCapabilities: [Capability.Play, Capability.Pause, Capability.SkipToNext],
          progressUpdateEventInterval: 2,
          android: {
            appKilledPlaybackBehavior: AppKilledPlaybackBehavior.ContinuePlayback,
          },
        });
        positionsRef.current = await readPositions();
      } catch (e) {
        console.warn('TrackPlayer setup failed', e);
      }
      if (!cancelled) setReady(true);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!ready) return;
    const sub = TrackPlayer.addEventListener(Event.PlaybackActiveTrackChanged, async (ev) => {
      if (playMetaRef.current.serviceType !== 'emby') return;
      const lastId = ev.lastTrack?.id != null ? String(ev.lastTrack.id) : undefined;
      const newId = ev.track?.id != null ? String(ev.track.id) : undefined;
      const session = embySessionRef.current;
      if (lastId && session && session.itemId === lastId && newId && newId !== lastId) {
        embySessionRef.current = null;
        try {
          await reportEmbyPlaybackStopped({
            ItemId: lastId,
            MediaSourceId: session.mediaSourceId,
            PlaySessionId: session.playSessionId,
            PositionTicks: Math.floor(Math.max(0, ev.lastPosition) * 10_000_000),
          });
        } catch {
          // ignore
        }
      }
      if (newId && playMetaRef.current.serviceType === 'emby') {
        void beginEmbyPlaybackForItem(newId, 0);
      }
      if (!newId && playMetaRef.current.serviceType === 'emby') {
        embySessionRef.current = null;
      }
    });
    return () => {
      sub.remove();
    };
  }, [ready]);

  useEffect(() => {
    if (!ready) return;
    if (playbackState !== State.Playing) return;
    const id = setInterval(async () => {
      try {
        const track = await TrackPlayer.getActiveTrack();
        const progress = await TrackPlayer.getProgress();
        if (track?.id != null) {
          positionsRef.current[track.id] = progress.position;
          await writePositionsFile(positionsRef.current);
        }
        if (playMetaRef.current.serviceType === 'emby' && embySessionRef.current) {
          const now = Date.now();
          if (now - lastEmbyProgressAtRef.current >= 10_000) {
            lastEmbyProgressAtRef.current = now;
            void reportEmbyProgressFromPlayer(false);
          }
        }
      } catch {
        // ignore
      }
    }, 5000);
    return () => clearInterval(id);
  }, [ready, playbackState]);

  const replaceQueueAndPlay = useCallback(async (tracks: MusicTrack[], startIndex: number, meta: PlayQueueMeta) => {
    if (playMetaRef.current.serviceType === 'emby') {
      await stopEmbyPlaybackIfActive();
    }
    setShuffleEnabled(false);
    setPlayMeta(meta);
    playMetaRef.current = meta;
    await TrackPlayer.reset();
    const queue = await Promise.all(
      tracks.map(async (t) => {
        const local = await getLocalPath(t.id);
        const url =
          local ??
          (await getStreamProxyUrl(t.id, {
            serviceType: meta.serviceType ?? t.service_type ?? undefined,
            parentId: meta.parentId ?? undefined,
          }));
        const artwork = t.cover_art_id
          ? await getCoverArtUrl(t.cover_art_id, {
              serviceType: meta.serviceType ?? t.service_type ?? undefined,
              size: 300,
            })
          : undefined;
        return {
          id: t.id,
          url,
          title: t.title,
          artist: t.artist ?? undefined,
          album: t.album ?? undefined,
          duration: t.duration ?? undefined,
          artwork,
          cover_art_id: t.cover_art_id ?? undefined,
        } as Track;
      })
    );
    await TrackPlayer.add(queue);
    const idx = Math.max(0, Math.min(startIndex, Math.max(0, queue.length - 1)));
    if (queue.length > 0) {
      await TrackPlayer.skip(idx);
      const tid = queue[idx]?.id;
      if (tid) {
        const saved = positionsRef.current[tid];
        if (typeof saved === 'number' && saved > 2) {
          await TrackPlayer.seekTo(saved);
        }
      }
      await TrackPlayer.play();
      if (meta.serviceType === 'emby' && queue[idx]?.id != null) {
        void beginEmbyPlaybackForItem(String(queue[idx].id), 0);
      }
    }
  }, []);

  const pause = useCallback(async () => {
    if (playMetaRef.current.serviceType === 'emby' && embySessionRef.current) {
      void reportEmbyProgressFromPlayer(true);
    }
    await TrackPlayer.pause();
  }, []);

  const resume = useCallback(async () => {
    await TrackPlayer.play();
  }, []);

  const seekTo = useCallback(async (seconds: number) => {
    await TrackPlayer.seekTo(seconds);
  }, []);

  const skipToNext = useCallback(async () => {
    await TrackPlayer.skipToNext();
  }, []);

  const skipToPrevious = useCallback(async () => {
    await TrackPlayer.skipToPrevious();
  }, []);

  const stop = useCallback(async () => {
    if (playMetaRef.current.serviceType === 'emby') {
      await stopEmbyPlaybackIfActive();
    }
    await TrackPlayer.reset();
    setPlayMeta({});
    playMetaRef.current = {};
    setShuffleEnabled(false);
  }, []);

  const toggleShuffle = useCallback(() => {
    setShuffleEnabled((prev) => {
      const next = !prev;
      if (next) {
        void (async () => {
          try {
            const q = await TrackPlayer.getQueue();
            const idx = (await TrackPlayer.getActiveTrackIndex()) ?? 0;
            const upcoming = q.slice(idx + 1);
            const shuffled = [...upcoming];
            for (let i = shuffled.length - 1; i > 0; i--) {
              const j = Math.floor(Math.random() * (i + 1));
              [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
            }
            await TrackPlayer.removeUpcomingTracks();
            if (shuffled.length) await TrackPlayer.add(shuffled);
          } catch {
            // ignore
          }
        })();
      }
      return next;
    });
  }, []);

  const value = useMemo<MediaPlayerContextValue>(
    () => ({
      ready,
      activeTrack: activeTrack ?? undefined,
      playMeta,
      hasActiveSession: Boolean(activeTrack),
      replaceQueueAndPlay,
      pause,
      resume,
      seekTo,
      skipToNext,
      skipToPrevious,
      stop,
      fullPlayerVisible,
      setFullPlayerVisible,
      playbackState,
      shuffleEnabled,
      toggleShuffle,
    }),
    [
      ready,
      activeTrack,
      playMeta,
      replaceQueueAndPlay,
      pause,
      resume,
      seekTo,
      skipToNext,
      skipToPrevious,
      stop,
      fullPlayerVisible,
      playbackState,
      shuffleEnabled,
      toggleShuffle,
    ]
  );

  return <MediaPlayerContext.Provider value={value}>{children}</MediaPlayerContext.Provider>;
}

export function useMediaPlayer(): MediaPlayerContextValue {
  const ctx = useContext(MediaPlayerContext);
  if (!ctx) {
    throw new Error('useMediaPlayer must be used within MediaPlayerProvider');
  }
  return ctx;
}
