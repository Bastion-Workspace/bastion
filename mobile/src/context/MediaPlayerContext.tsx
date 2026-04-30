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
  State,
  useActiveTrack,
  usePlaybackState,
  type Track,
} from 'react-native-track-player';
import type { MusicTrack } from '../api/media';
import { getStreamProxyUrl } from '../api/media';
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

function normalizePlaybackState(raw: State | { state: State } | object | undefined): State {
  if (raw == null) return State.None;
  if (typeof raw === 'number') return raw as State;
  if (typeof raw === 'object' && 'state' in raw && typeof (raw as { state: State }).state === 'number') {
    return (raw as { state: State }).state;
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
};

const MediaPlayerContext = createContext<MediaPlayerContextValue | null>(null);

export function MediaPlayerProvider({ children }: { children: React.ReactNode }) {
  const [ready, setReady] = useState(false);
  const [fullPlayerVisible, setFullPlayerVisible] = useState(false);
  const [playMeta, setPlayMeta] = useState<PlayQueueMeta>({});
  const positionsRef = useRef<Record<string, number>>({});
  const activeTrack = useActiveTrack();
  const rawState = usePlaybackState();
  const playbackState = normalizePlaybackState(rawState as State | { state: State });

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
    if (playbackState !== State.Playing) return;
    const id = setInterval(async () => {
      try {
        const track = await TrackPlayer.getActiveTrack();
        const progress = await TrackPlayer.getProgress();
        if (track?.id != null) {
          positionsRef.current[track.id] = progress.position;
          await writePositionsFile(positionsRef.current);
        }
      } catch {
        // ignore
      }
    }, 5000);
    return () => clearInterval(id);
  }, [ready, playbackState]);

  const replaceQueueAndPlay = useCallback(async (tracks: MusicTrack[], startIndex: number, meta: PlayQueueMeta) => {
    setPlayMeta(meta);
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
        return {
          id: t.id,
          url,
          title: t.title,
          artist: t.artist ?? undefined,
          album: t.album ?? undefined,
          duration: t.duration ?? undefined,
        };
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
    }
  }, []);

  const pause = useCallback(async () => {
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
    await TrackPlayer.reset();
    setPlayMeta({});
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
