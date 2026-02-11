import React, { useRef, useEffect, useState } from 'react';
import {
  Box,
  IconButton,
  Slider,
  Typography,
  Tooltip,
} from '@mui/material';
import {
  SkipPrevious,
  PlayArrow,
  Pause,
  SkipNext,
  Repeat,
  RepeatOne,
  Shuffle,
  VolumeUp,
  VolumeOff,
  Close,
} from '@mui/icons-material';
import { useMusic } from '../../contexts/MediaContext';

const MusicStatusBarControls = () => {
  const {
    currentTrack,
    isPlaying,
    currentTime,
    duration,
    volume,
    isMuted,
    repeatMode,
    shuffleMode,
    togglePlayPause,
    handleNext,
    handlePrevious,
    handleSeek,
    handleVolumeChange,
    toggleMute,
    toggleRepeat,
    toggleShuffle,
    clearQueue,
    formatTime,
  } = useMusic();

  const trackTitleRef = useRef(null);
  const containerRef = useRef(null);
  const [needsScrolling, setNeedsScrolling] = useState(false);
  const [scrollDistance, setScrollDistance] = useState(0);

  // Calculate track title early so it can be used in useEffect
  const trackTitle = currentTrack ? `${currentTrack.title}${currentTrack.artist ? ` - ${currentTrack.artist}` : ''}` : '';

  // Check if text overflows and needs scrolling
  useEffect(() => {
    if (!currentTrack || !trackTitle) return;
    
    // Small delay to ensure DOM is ready
    const timeoutId = setTimeout(() => {
      if (trackTitleRef.current && containerRef.current) {
        const containerElement = containerRef.current;
        
        // Use a temporary measurement to get full text width
        const tempSpan = document.createElement('span');
        tempSpan.style.visibility = 'hidden';
        tempSpan.style.position = 'absolute';
        tempSpan.style.whiteSpace = 'nowrap';
        tempSpan.style.fontSize = '0.7rem';
        tempSpan.textContent = trackTitle;
        document.body.appendChild(tempSpan);
        
        const textWidth = tempSpan.offsetWidth;
        const containerWidth = containerElement.offsetWidth;
        
        document.body.removeChild(tempSpan);
        
        const isOverflowing = textWidth > containerWidth;
        setNeedsScrolling(isOverflowing);
        
        if (isOverflowing) {
          // Calculate how much we need to scroll (text width - container width + some padding)
          const distance = textWidth - containerWidth + 20;
          setScrollDistance(distance);
          
          // Create dynamic keyframes
          const styleId = 'music-title-scroll-animation';
          let styleElement = document.getElementById(styleId);
          if (!styleElement) {
            styleElement = document.createElement('style');
            styleElement.id = styleId;
            document.head.appendChild(styleElement);
          }
          styleElement.textContent = `
            @keyframes musicTitleScroll {
              0%, 25% {
                transform: translateX(0);
              }
              50%, 75% {
                transform: translateX(-${distance}px);
              }
              100% {
                transform: translateX(0);
              }
            }
          `;
        } else {
          // Remove animation style if not needed
          const styleElement = document.getElementById('music-title-scroll-animation');
          if (styleElement) {
            styleElement.remove();
          }
        }
      }
    }, 100);
    
    return () => {
      clearTimeout(timeoutId);
      // Cleanup animation style on unmount
      const styleElement = document.getElementById('music-title-scroll-animation');
      if (styleElement) {
        styleElement.remove();
      }
    };
  }, [currentTrack, trackTitle]);

  if (!currentTrack) {
    return null;
  }

  const getRepeatIcon = () => {
    if (repeatMode === 'track') {
      return <RepeatOne />;
    }
    return <Repeat />;
  };

  const getRepeatTooltip = () => {
    if (repeatMode === 'off') return 'Repeat: Off';
    if (repeatMode === 'track') return 'Repeat: Track';
    return 'Repeat: Album/Playlist';
  };

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        flex: 1,
        justifyContent: 'center',
        px: 1,
        minWidth: 0, // Allow shrinking
      }}
    >
      {/* Previous */}
      <Tooltip title="Previous track">
        <IconButton size="small" onClick={handlePrevious}>
          <SkipPrevious fontSize="small" />
        </IconButton>
      </Tooltip>

      {/* Play/Pause */}
      <Tooltip title={isPlaying ? 'Pause' : 'Play'}>
        <IconButton size="small" onClick={togglePlayPause}>
          {isPlaying ? <Pause fontSize="small" /> : <PlayArrow fontSize="small" />}
        </IconButton>
      </Tooltip>

      {/* Next */}
      <Tooltip title="Next track">
        <IconButton size="small" onClick={handleNext}>
          <SkipNext fontSize="small" />
        </IconButton>
      </Tooltip>

      {/* Track Info - fixed width so prev/play/next buttons stay static when title length varies */}
      <Box
        ref={containerRef}
        sx={{
          width: { xs: 100, sm: 170, md: 220 },
          minWidth: { xs: 100, sm: 170, md: 220 },
          flexShrink: 0,
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        <Typography
          ref={trackTitleRef}
          variant="caption"
          sx={{
            fontSize: '0.7rem',
            whiteSpace: 'nowrap',
            display: 'inline-block',
            ...(needsScrolling && scrollDistance > 0 && {
              animation: 'musicTitleScroll 8s ease-in-out infinite',
            }),
          }}
        >
          {trackTitle}
        </Typography>
      </Box>

      {/* Progress Slider */}
      <Box sx={{ width: { xs: 60, sm: 100, md: 150 }, minWidth: 60 }}>
        <Slider
          size="small"
          value={currentTime}
          max={duration || 100}
          onChange={(_, value) => handleSeek(value)}
          sx={{
            height: 4,
            '& .MuiSlider-thumb': {
              width: 8,
              height: 8,
            },
          }}
        />
      </Box>

      {/* Time Display */}
      <Typography variant="caption" sx={{ fontSize: '0.7rem', minWidth: '40px' }}>
        {formatTime(currentTime)} / {formatTime(duration)}
      </Typography>

      {/* Volume */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, width: { xs: 60, sm: 80 } }}>
        <Tooltip title={isMuted ? 'Unmute' : 'Mute'}>
          <IconButton size="small" onClick={toggleMute}>
            {isMuted ? <VolumeOff fontSize="small" /> : <VolumeUp fontSize="small" />}
          </IconButton>
        </Tooltip>
        <Slider
          size="small"
          value={isMuted ? 0 : volume * 100}
          onChange={(_, value) => handleVolumeChange(value / 100)}
          sx={{
            height: 4,
            '& .MuiSlider-thumb': {
              width: 8,
              height: 8,
            },
          }}
        />
      </Box>

      {/* Repeat */}
      <Tooltip title={getRepeatTooltip()}>
        <IconButton
          size="small"
          onClick={toggleRepeat}
          color={repeatMode !== 'off' ? 'primary' : 'default'}
        >
          {getRepeatIcon()}
        </IconButton>
      </Tooltip>

      {/* Shuffle */}
      <Tooltip title={shuffleMode ? 'Shuffle: On' : 'Shuffle: Off'}>
        <IconButton
          size="small"
          onClick={toggleShuffle}
          color={shuffleMode ? 'primary' : 'default'}
        >
          <Shuffle fontSize="small" />
        </IconButton>
      </Tooltip>

      {/* Close/Stop */}
      <Tooltip title="Stop playback">
        <IconButton
          size="small"
          onClick={clearQueue}
          color="default"
        >
          <Close fontSize="small" />
        </IconButton>
      </Tooltip>
    </Box>
  );
};

export default MusicStatusBarControls;

