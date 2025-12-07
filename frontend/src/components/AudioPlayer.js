/**
 * Audio Player Component
 * Reusable audio player with standard controls for playback
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  IconButton,
  Slider,
  Typography,
  Paper,
  Tooltip,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  VolumeUp,
  VolumeOff,
  Download,
} from '@mui/icons-material';

const AudioPlayer = ({ src, filename, onError }) => {
  const audioRef = useRef(null);
  const blobUrlRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [error, setError] = useState(null);
  const [audioBlobUrl, setAudioBlobUrl] = useState(null);
  const [loading, setLoading] = useState(true);

  // Load audio file with authentication
  useEffect(() => {
    if (!src) return;

    let blobUrlToCleanup = null;

    const loadAudio = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Check if src is already a blob URL or data URL
        if (src.startsWith('blob:') || src.startsWith('data:')) {
          setAudioBlobUrl(src);
          setLoading(false);
          return;
        }

        // Fetch with authentication
        // Handle relative URLs by constructing full URL
        const fetchUrl = src.startsWith('/') ? `${window.location.origin}${src}` : src;
        const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
        const response = await fetch(fetchUrl, {
          headers: token ? { 'Authorization': `Bearer ${token}` } : {}
        });

        if (!response.ok) {
          throw new Error(`Failed to load audio: ${response.status} ${response.statusText}`);
        }

        const blob = await response.blob();
        const blobUrl = URL.createObjectURL(blob);
        // Cleanup previous blob URL if it exists
        if (blobUrlRef.current && blobUrlRef.current.startsWith('blob:')) {
          URL.revokeObjectURL(blobUrlRef.current);
        }
        blobUrlRef.current = blobUrl;
        setAudioBlobUrl(blobUrl);
      } catch (err) {
        console.error('Failed to load audio file:', err);
        setError('Failed to load audio file');
        if (onError) onError(err);
      } finally {
        setLoading(false);
      }
    };

    loadAudio();

    // Cleanup blob URL on unmount or src change
    return () => {
      if (blobUrlRef.current && blobUrlRef.current.startsWith('blob:')) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, [src, onError]);

  // Update current time as audio plays
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !audioBlobUrl) return;

    const updateTime = () => setCurrentTime(audio.currentTime);
    const updateDuration = () => setDuration(audio.duration);
    const handleEnded = () => setIsPlaying(false);
    const handleError = (e) => {
      setError('Failed to load audio file');
      setIsPlaying(false);
      if (onError) onError(e);
    };

    audio.addEventListener('timeupdate', updateTime);
    audio.addEventListener('loadedmetadata', updateDuration);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('error', handleError);

    return () => {
      audio.removeEventListener('timeupdate', updateTime);
      audio.removeEventListener('loadedmetadata', updateDuration);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('error', handleError);
    };
  }, [audioBlobUrl, onError]);

  // Sync volume with audio element
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = isMuted ? 0 : volume;
    }
  }, [volume, isMuted]);

  const togglePlayPause = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play().catch((e) => {
        setError('Failed to play audio');
        if (onError) onError(e);
      });
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (event, newValue) => {
    const audio = audioRef.current;
    if (!audio) return;

    audio.currentTime = newValue;
    setCurrentTime(newValue);
  };

  const handleVolumeChange = (event, newValue) => {
    setVolume(newValue);
    setIsMuted(newValue === 0);
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
  };

  const formatTime = (seconds) => {
    if (!isFinite(seconds) || isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleDownload = async () => {
    if (!src) return;

    try {
      // If we have a blob URL, use it directly
      if (audioBlobUrl && audioBlobUrl.startsWith('blob:')) {
        const link = document.createElement('a');
        link.href = audioBlobUrl;
        link.download = filename || 'audio';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } else {
        // Otherwise, fetch with auth and download
        const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
        const response = await fetch(src, {
          headers: token ? { 'Authorization': `Bearer ${token}` } : {}
        });
        
        if (!response.ok) {
          throw new Error('Failed to download audio');
        }
        
        const blob = await response.blob();
        const blobUrl = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = blobUrl;
        link.download = filename || 'audio';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(blobUrl);
      }
    } catch (err) {
      console.error('Failed to download audio:', err);
      setError('Failed to download audio');
    }
  };

  return (
    <Paper
      elevation={1}
      sx={{
        p: 2,
        borderRadius: 2,
        backgroundColor: 'background.paper',
        maxWidth: '100%',
      }}
    >
      {error && (
        <Typography variant="caption" color="error" sx={{ mb: 1, display: 'block' }}>
          {error}
        </Typography>
      )}

      {loading && (
        <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
          Loading audio...
        </Typography>
      )}

      <audio ref={audioRef} src={audioBlobUrl || ''} preload="metadata" />

      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        {/* Play/Pause Button */}
        <IconButton
          onClick={togglePlayPause}
          color="primary"
          size="small"
          sx={{ flexShrink: 0 }}
        >
          {isPlaying ? <Pause /> : <PlayArrow />}
        </IconButton>

        {/* Time Display */}
        <Typography variant="caption" sx={{ minWidth: '80px', textAlign: 'center' }}>
          {formatTime(currentTime)} / {formatTime(duration)}
        </Typography>

        {/* Seek Bar */}
        <Slider
          value={currentTime}
          min={0}
          max={duration || 0}
          onChange={handleSeek}
          size="small"
          sx={{ flexGrow: 1, mx: 1 }}
          disabled={!duration}
        />

        {/* Volume Control */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: '100px' }}>
          <Tooltip title={isMuted ? 'Unmute' : 'Mute'}>
            <IconButton onClick={toggleMute} size="small">
              {isMuted ? <VolumeOff fontSize="small" /> : <VolumeUp fontSize="small" />}
            </IconButton>
          </Tooltip>
          <Slider
            value={isMuted ? 0 : volume}
            min={0}
            max={1}
            step={0.01}
            onChange={handleVolumeChange}
            size="small"
            sx={{ width: '60px' }}
          />
        </Box>

        {/* Download Button */}
        <Tooltip title="Download">
          <IconButton onClick={handleDownload} size="small">
            <Download fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>

      {filename && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
          {filename}
        </Typography>
      )}
    </Paper>
  );
};

export default AudioPlayer;

