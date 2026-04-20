import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import {
  Box,
  Dialog,
  IconButton,
  Slider,
  Stack,
  Typography,
  Tooltip,
} from '@mui/material';
import {
  Close,
  Pause,
  PlayArrow,
  PictureInPicture,
  Fullscreen,
  FullscreenExit,
} from '@mui/icons-material';
import Hls from 'hls.js';
import apiService from '../../services/apiService';

function itemId(it) {
  return it?.Id || it?.id;
}

function titleForItem(it) {
  if (!it) return '';
  const series = it.SeriesName || it.seriesName;
  const sn = it.SeasonName || it.seasonName;
  const idx = it.IndexNumber;
  if (series && idx != null) {
    return `${series} — S${it.ParentIndexNumber ?? ''}E${idx} — ${it.Name || it.name || ''}`;
  }
  return it.Name || it.name || 'Video';
}

export default function VideoPlayerDialog({
  open,
  item,
  pick,
  error: contextError,
  onClose,
  onStartProgressLoop,
  clearProgressTimer,
}) {
  const videoRef = useRef(null);
  const hlsRef = useRef(null);
  const hideControlsTimer = useRef(null);
  const volumeRef = useRef(1);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [current, setCurrent] = useState(0);
  const [volume, setVolume] = useState(1);
  const [showControls, setShowControls] = useState(true);
  const [pip, setPip] = useState(false);
  const [fs, setFs] = useState(false);
  const [loadError, setLoadError] = useState(null);

  volumeRef.current = volume;

  const ticksFromSeconds = (s) => Math.floor(Math.max(0, s) * 10_000_000);

  const teardownMedia = useCallback(() => {
    clearProgressTimer();
    if (hlsRef.current) {
      hlsRef.current.destroy();
      hlsRef.current = null;
    }
    const v = videoRef.current;
    if (v) {
      v.removeAttribute('src');
      v.load();
    }
  }, [clearProgressTimer]);

  useLayoutEffect(() => {
    if (!open || !item || !pick) {
      teardownMedia();
      return undefined;
    }

    const id = itemId(item);
    setLoadError(null);
    let cancelled = false;
    let raf1 = 0;
    let raf2 = 0;

    const tryAttach = () => {
      const v = videoRef.current;
      if (!v) return null;

      let blocked = false;
      const streamExtra = { static: 'true' };
      if (pick.audioStreamIndex != null && Number.isFinite(Number(pick.audioStreamIndex))) {
        streamExtra.audio_stream_index = String(pick.audioStreamIndex);
      }
      const hlsExtra = {};
      if (pick.audioStreamIndex != null && Number.isFinite(Number(pick.audioStreamIndex))) {
        hlsExtra.audio_stream_index = String(pick.audioStreamIndex);
      }
      const url = pick.useHls
        ? apiService.emby.getHlsMasterUrl(id, pick.mediaSourceId, pick.playSessionId, hlsExtra)
        : apiService.emby.getVideoStreamUrl(id, pick.mediaSourceId, pick.playSessionId, streamExtra);

      if (pick.useHls) {
        if (v.canPlayType('application/vnd.apple.mpegurl')) {
          v.src = url;
        } else if (Hls.isSupported()) {
          const hls = new Hls({ enableWorker: true });
          hlsRef.current = hls;
          hls.loadSource(url);
          hls.attachMedia(v);
          hls.on(Hls.Events.ERROR, (_, data) => {
            if (data.fatal) {
              setLoadError(data.type || 'HLS error');
            }
          });
        } else {
          setLoadError('HLS not supported in this browser');
          blocked = true;
        }
      } else {
        v.src = url;
      }

      v.defaultMuted = false;
      v.muted = false;
      v.volume = volumeRef.current;

      const onMeta = () => {
        if (cancelled) return;
        setDuration(v.duration || 0);
        const ud = item.UserData || item.userData;
        const pos = ud?.PlaybackPositionTicks;
        if (pos && v.duration) {
          v.currentTime = pos / 10_000_000;
        }
      };
      v.addEventListener('loadedmetadata', onMeta);

      const onPlay = () => {
        const el = videoRef.current;
        if (el) {
          el.muted = false;
          el.volume = volumeRef.current;
        }
        setPlaying(true);
        onStartProgressLoop(() => ticksFromSeconds(videoRef.current?.currentTime || 0));
      };
      const onPause = () => setPlaying(false);
      const onTime = () => setCurrent(videoRef.current?.currentTime || 0);
      const onEnded = async () => {
        const ticks = ticksFromSeconds(videoRef.current?.duration || 0);
        await onClose(ticks);
      };

      v.addEventListener('play', onPlay);
      v.addEventListener('pause', onPause);
      v.addEventListener('timeupdate', onTime);
      v.addEventListener('ended', onEnded);

      if (!blocked) {
        v.play()
          .then(() => {
            if (cancelled) return;
            v.muted = false;
            v.volume = volumeRef.current;
          })
          .catch(() => {});
      }

      return () => {
        v.removeEventListener('loadedmetadata', onMeta);
        v.removeEventListener('play', onPlay);
        v.removeEventListener('pause', onPause);
        v.removeEventListener('timeupdate', onTime);
        v.removeEventListener('ended', onEnded);
      };
    };

    let detachListeners = tryAttach();
    if (!detachListeners) {
      raf1 = requestAnimationFrame(() => {
        if (cancelled) return;
        detachListeners = tryAttach();
        if (!detachListeners) {
          raf2 = requestAnimationFrame(() => {
            if (cancelled) return;
            detachListeners = tryAttach();
          });
        }
      });
    }

    return () => {
      cancelled = true;
      if (raf1) cancelAnimationFrame(raf1);
      if (raf2) cancelAnimationFrame(raf2);
      if (typeof detachListeners === 'function') detachListeners();
      teardownMedia();
    };
  }, [open, item, pick, onClose, onStartProgressLoop, teardownMedia]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return undefined;
    const onPip = () => setPip(document.pictureInPictureElement === v);
    const onFs = () => setFs(Boolean(document.fullscreenElement));
    v.addEventListener('enterpictureinpicture', onPip);
    v.addEventListener('leavepictureinpicture', onPip);
    document.addEventListener('fullscreenchange', onFs);
    return () => {
      v.removeEventListener('enterpictureinpicture', onPip);
      v.removeEventListener('leavepictureinpicture', onPip);
      document.removeEventListener('fullscreenchange', onFs);
    };
  }, [open]);

  const bumpControls = () => {
    setShowControls(true);
    if (hideControlsTimer.current) clearTimeout(hideControlsTimer.current);
    hideControlsTimer.current = setTimeout(() => setShowControls(false), 3000);
  };

  const handleClose = async () => {
    const v = videoRef.current;
    const ticks = ticksFromSeconds(v?.currentTime || 0);
    await onClose(ticks);
  };

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play();
    else v.pause();
    bumpControls();
  };

  const seek = (_, value) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = value;
    setCurrent(value);
    bumpControls();
  };

  const fmt = (s) => {
    if (!s || Number.isNaN(s)) return '0:00';
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  const togglePip = async () => {
    const v = videoRef.current;
    if (!v || !document.pictureInPictureEnabled) return;
    try {
      if (document.pictureInPictureElement) {
        await document.exitPictureInPicture();
      } else {
        await v.requestPictureInPicture();
      }
    } catch (e) {
      console.warn('PiP failed', e);
    }
    bumpControls();
  };

  const toggleFs = async () => {
    const v = videoRef.current;
    if (!v) return;
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await v.requestFullscreen();
      }
    } catch (e) {
      console.warn('Fullscreen failed', e);
    }
    bumpControls();
  };

  const displayError = contextError || loadError;

  return (
    <Dialog
      fullScreen
      open={open}
      onClose={handleClose}
      PaperProps={{
        sx: { bgcolor: 'black', m: 0 },
        onMouseMove: bumpControls,
      }}
    >
      <Box
        sx={{
          position: 'relative',
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <video
          ref={videoRef}
          playsInline
          muted={false}
          style={{ width: '100%', height: '100%', maxHeight: '100vh' }}
        />

        {displayError && (
          <Box
            sx={{
              position: 'absolute',
              top: 80,
              left: 16,
              right: 16,
              color: 'error.light',
            }}
          >
            <Typography>{displayError}</Typography>
          </Box>
        )}

        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            p: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: showControls ? 'linear-gradient(rgba(0,0,0,0.7), transparent)' : 'transparent',
            opacity: showControls ? 1 : 0,
            transition: 'opacity 0.2s',
          }}
        >
          <Typography variant="subtitle1" sx={{ color: 'white', px: 1, flex: 1 }} noWrap>
            {titleForItem(item)}
          </Typography>
          <IconButton onClick={handleClose} sx={{ color: 'white' }} size="large" aria-label="Close">
            <Close />
          </IconButton>
        </Box>

        <Box
          sx={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            p: 2,
            background: showControls ? 'linear-gradient(transparent, rgba(0,0,0,0.85))' : 'transparent',
            opacity: showControls ? 1 : 0,
            transition: 'opacity 0.2s',
          }}
        >
          <Stack spacing={1}>
            <Stack direction="row" alignItems="center" spacing={2}>
              <Typography variant="caption" sx={{ color: 'white', minWidth: 90 }}>
                {fmt(current)} / {fmt(duration)}
              </Typography>
              <Slider
                size="small"
                min={0}
                max={duration || 1}
                value={Math.min(current, duration || 1)}
                onChange={seek}
                sx={{ color: 'primary.light' }}
              />
            </Stack>
            <Stack direction="row" alignItems="center" spacing={1}>
              <IconButton sx={{ color: 'white' }} onClick={togglePlay} size="large">
                {playing ? <Pause /> : <PlayArrow />}
              </IconButton>
              <Typography variant="caption" sx={{ color: 'white', mr: 1 }}>
                Volume
              </Typography>
              <Slider
                size="small"
                min={0}
                max={1}
                step={0.05}
                value={volume}
                onChange={(_, v2) => {
                  setVolume(v2);
                  if (videoRef.current) videoRef.current.volume = v2;
                }}
                sx={{ width: 120, color: 'white' }}
              />
              <Tooltip title="Picture-in-picture">
                <span>
                  <IconButton sx={{ color: 'white' }} onClick={togglePip} disabled={!document.pictureInPictureEnabled}>
                    <PictureInPicture />
                  </IconButton>
                </span>
              </Tooltip>
              <Tooltip title={fs ? 'Exit fullscreen' : 'Fullscreen'}>
                <IconButton sx={{ color: 'white' }} onClick={toggleFs}>
                  {fs ? <FullscreenExit /> : <Fullscreen />}
                </IconButton>
              </Tooltip>
            </Stack>
          </Stack>
        </Box>
      </Box>
    </Dialog>
  );
}
