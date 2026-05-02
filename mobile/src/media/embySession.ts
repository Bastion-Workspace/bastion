import TrackPlayer from 'react-native-track-player';
import {
  pickMobileSourceAndMode,
  postEmbyPlaybackInfo,
  reportEmbyPlaybackProgress,
  reportEmbyPlaybackStart,
  reportEmbyPlaybackStopped,
} from '../api/emby';

export type EmbySession = {
  itemId: string;
  mediaSourceId: string;
  playSessionId: string;
  playMethod?: string;
  audioStreamIndex?: number | null;
};

export const embySessionRef: { current: EmbySession | null } = { current: null };

function secondsToTicks(sec: number): number {
  if (!Number.isFinite(sec) || sec < 0) return 0;
  return Math.floor(sec * 10_000_000);
}

export async function stopEmbyPlaybackIfActive(positionSeconds?: number): Promise<void> {
  const s = embySessionRef.current;
  if (!s) return;
  embySessionRef.current = null;
  let pos = positionSeconds;
  if (pos == null) {
    try {
      pos = (await TrackPlayer.getProgress()).position;
    } catch {
      pos = 0;
    }
  }
  try {
    await reportEmbyPlaybackStopped({
      ItemId: s.itemId,
      MediaSourceId: s.mediaSourceId,
      PlaySessionId: s.playSessionId,
      PositionTicks: secondsToTicks(pos ?? 0),
    });
  } catch {
    // ignore
  }
}

/** Idempotent: skips if session already matches itemId. */
export async function beginEmbyPlaybackForItem(itemId: string, startTimeSeconds = 0): Promise<void> {
  if (embySessionRef.current?.itemId === itemId) return;
  if (embySessionRef.current && embySessionRef.current.itemId !== itemId) {
    await stopEmbyPlaybackIfActive();
  }
  try {
    const info = await postEmbyPlaybackInfo(itemId, {
      start_time_ticks: secondsToTicks(startTimeSeconds),
    });
    const pick = pickMobileSourceAndMode(info as Record<string, unknown>);
    if (!pick) return;
    embySessionRef.current = {
      itemId,
      mediaSourceId: pick.mediaSourceId,
      playSessionId: pick.playSessionId,
      playMethod: pick.playMethod,
      audioStreamIndex: pick.audioStreamIndex,
    };
    await reportEmbyPlaybackStart({
      ItemId: itemId,
      MediaSourceId: pick.mediaSourceId,
      PlaySessionId: pick.playSessionId,
      PositionTicks: secondsToTicks(startTimeSeconds),
      PlayMethod: pick.playMethod,
      IsPaused: false,
      ...(pick.audioStreamIndex != null ? { AudioStreamIndex: pick.audioStreamIndex } : {}),
    });
  } catch {
    embySessionRef.current = null;
  }
}

export async function reportEmbyProgressFromPlayer(isPaused: boolean): Promise<void> {
  const s = embySessionRef.current;
  if (!s) return;
  try {
    const progress = await TrackPlayer.getProgress();
    await reportEmbyPlaybackProgress({
      ItemId: s.itemId,
      MediaSourceId: s.mediaSourceId,
      PlaySessionId: s.playSessionId,
      PositionTicks: secondsToTicks(progress.position),
      PlayMethod: s.playMethod ?? 'DirectStream',
      IsPaused: isPaused,
      EventName: 'TimeUpdate',
      ...(s.audioStreamIndex != null ? { AudioStreamIndex: s.audioStreamIndex } : {}),
    });
  } catch {
    // ignore
  }
}
