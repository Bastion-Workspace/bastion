import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';
import apiService from '../services/apiService';
import VideoPlayerDialog from '../components/video/VideoPlayerDialog';

const VideoContext = createContext(null);

export const useVideo = () => {
  const ctx = useContext(VideoContext);
  if (!ctx) {
    throw new Error('useVideo must be used within VideoProvider');
  }
  return ctx;
};

function itemId(item) {
  return item?.Id || item?.id;
}

/** Default audio stream index + codec from Emby MediaSource (PascalCase or camelCase). */
function defaultAudioFromMediaSource(ms) {
  const streams = ms?.MediaStreams || ms?.mediaStreams || [];
  const audioStreams = streams.filter((s) => (s.Type || s.type) === 'Audio');
  if (audioStreams.length === 0) {
    return { index: null, codec: null };
  }
  const defIdx = ms?.DefaultAudioStreamIndex ?? ms?.defaultAudioStreamIndex;
  const chosen =
    defIdx != null
      ? audioStreams.find((s) => (s.Index ?? s.index) === defIdx) || audioStreams[0]
      : audioStreams[0];
  const idx = chosen?.Index ?? chosen?.index;
  const codec = chosen?.Codec ?? chosen?.codec ?? null;
  return {
    index: idx != null && Number.isFinite(Number(idx)) ? Number(idx) : null,
    codec: codec != null ? String(codec) : null,
  };
}

/**
 * Codecs commonly muxed in MP4 that Chromium/Safari often cannot decode in <video> direct stream.
 * Prefer HLS transcode (AAC) when Emby offers it.
 */
function browserLikelyDecodesMp4Audio(codec) {
  if (codec == null || codec === '') return true;
  const c = String(codec).toLowerCase();
  if (c.includes('aac') || c === 'mp3' || c === 'opus') return true;
  if (
    c === 'ac3' ||
    c === 'eac3' ||
    c === 'dts' ||
    c === 'dca' ||
    c.includes('truehd') ||
    c.includes('dts-hd')
  ) {
    return false;
  }
  return true;
}

function pickSourceAndMode(playbackInfo) {
  const sources = playbackInfo?.MediaSources || [];
  const ms = sources.find((s) => s.IsDefault) || sources[0];
  if (!ms) return null;
  const sessionId =
    playbackInfo?.PlaySessionId ||
    playbackInfo?.playSessionId ||
    ms.PlaySessionId ||
    ms.playSessionId ||
    (typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : null);
  if (!sessionId) return null;
  const trans = (ms.TranscodingUrl || ms?.transcodingUrl || '').toLowerCase();
  const hasHls = trans.includes('m3u8');
  const { index: audioStreamIndex, codec: audioCodec } = defaultAudioFromMediaSource(ms);
  const directAudioOk = browserLikelyDecodesMp4Audio(audioCodec);
  const useHls = hasHls && (ms.SupportsDirectPlay !== true || !directAudioOk);
  return {
    mediaSourceId: ms.Id,
    playSessionId: sessionId,
    useHls,
    mediaSource: ms,
    audioStreamIndex,
    audioCodec,
  };
}

export const VideoProvider = ({ children }) => {
  const [open, setOpen] = useState(false);
  const [item, setItem] = useState(null);
  const [pick, setPick] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const progressTimerRef = useRef(null);
  const sessionRef = useRef({ itemId: null, mediaSourceId: null, playSessionId: null });

  const clearProgressTimer = useCallback(() => {
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
    }
  }, []);

  const reportStopped = useCallback(
    async (positionTicks) => {
      const s = sessionRef.current;
      if (!s.itemId) return;
      try {
        await apiService.emby.reportPlaybackStopped({
          ItemId: s.itemId,
          MediaSourceId: s.mediaSourceId,
          PlaySessionId: s.playSessionId,
          PositionTicks: positionTicks,
        });
      } catch (e) {
        console.warn('Emby playback stopped report failed', e);
      }
    },
    []
  );

  const closeVideo = useCallback(
    async (positionTicks = 0) => {
      clearProgressTimer();
      await reportStopped(positionTicks);
      sessionRef.current = { itemId: null, mediaSourceId: null, playSessionId: null };
      setOpen(false);
      setItem(null);
      setPick(null);
      setError(null);
    },
    [clearProgressTimer, reportStopped]
  );

  const openVideo = useCallback(
    async (rawItem) => {
      setError(null);
      setLoading(true);
      clearProgressTimer();
      try {
        const id = itemId(rawItem);
        if (!id) {
          setError('Missing item id');
          return;
        }
        const detail =
          rawItem?.Name && rawItem?.Type
            ? rawItem
            : await apiService.emby.getItemDetail(id);
        const pi = await apiService.emby.getPlaybackInfo(itemId(detail));
        const chosen = pickSourceAndMode(pi);
        if (!chosen) {
          setError('No playable media source');
          return;
        }
        sessionRef.current = {
          itemId: itemId(detail),
          mediaSourceId: chosen.mediaSourceId,
          playSessionId: chosen.playSessionId,
        };
        setItem(detail);
        setPick(chosen);
        setOpen(true);
        await apiService.emby.reportPlaybackStart({
          ItemId: itemId(detail),
          MediaSourceId: chosen.mediaSourceId,
          PlaySessionId: chosen.playSessionId,
          PositionTicks: 0,
          PlayMethod: chosen.useHls ? 'Transcode' : 'DirectStream',
          IsPaused: false,
          ...(chosen.audioStreamIndex != null
            ? { AudioStreamIndex: chosen.audioStreamIndex }
            : {}),
        });
      } catch (e) {
        console.error('openVideo failed', e);
        setError(e?.message || 'Playback failed');
      } finally {
        setLoading(false);
      }
    },
    [clearProgressTimer]
  );

  const startProgressLoop = useCallback(
    (getCurrentTicks) => {
      clearProgressTimer();
      progressTimerRef.current = setInterval(async () => {
        const s = sessionRef.current;
        if (!s.itemId) return;
        try {
          const ticks = getCurrentTicks();
          await apiService.emby.reportPlaybackProgress({
            ItemId: s.itemId,
            MediaSourceId: s.mediaSourceId,
            PlaySessionId: s.playSessionId,
            PositionTicks: ticks,
            EventName: 'TimeUpdate',
            IsPaused: false,
          });
        } catch (e) {
          console.warn('Emby progress report failed', e);
        }
      }, 10_000);
    },
    [clearProgressTimer]
  );

  useEffect(
    () => () => {
      clearProgressTimer();
    },
    [clearProgressTimer]
  );

  const value = {
    openVideo,
    closeVideo,
    loading,
    error,
    isOpen: open,
  };

  return (
    <VideoContext.Provider value={value}>
      {children}
      <VideoPlayerDialog
        open={open}
        item={item}
        pick={pick}
        error={error}
        onClose={closeVideo}
        onStartProgressLoop={startProgressLoop}
        clearProgressTimer={clearProgressTimer}
      />
    </VideoContext.Provider>
  );
};
