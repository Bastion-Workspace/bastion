import React, { createContext, useContext, useState, useRef, useEffect, useCallback } from 'react';
import apiService from '../services/apiService';

const MusicContext = createContext(null);

export const useMusic = () => {
  const context = useContext(MusicContext);
  if (!context) {
    throw new Error('useMusic must be used within a MusicProvider');
  }
  return context;
};

export const MusicProvider = ({ children }) => {
  const [currentTrack, setCurrentTrack] = useState(null);
  const [queue, setQueue] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  
  // Load repeat and shuffle modes from localStorage
  const [repeatMode, setRepeatMode] = useState(() => {
    try {
      const saved = localStorage.getItem('musicRepeatMode');
      return saved || 'off';
    } catch (error) {
      console.error('Failed to load repeat mode from localStorage:', error);
      return 'off';
    }
  });
  
  const [shuffleMode, setShuffleMode] = useState(() => {
    try {
      const saved = localStorage.getItem('musicShuffleMode');
      return saved === 'true';
    } catch (error) {
      console.error('Failed to load shuffle mode from localStorage:', error);
      return false;
    }
  });
  const [originalQueue, setOriginalQueue] = useState([]); // Store original queue for shuffle
  const [currentParentId, setCurrentParentId] = useState(null); // For repeat album/playlist
  
  const audioRef = useRef(null);

  // Track if we're waiting to play (audio loading)
  const shouldPlayRef = useRef(false);
  const isLoadingTrackRef = useRef(false); // Track when we're loading a new track
  const handleTrackEndRef = useRef(null);
  const handleNextRef = useRef(null);

  // Initialize audio element (only once)
  useEffect(() => {
    if (!audioRef.current) {
      audioRef.current = new Audio();
      audioRef.current.preload = 'auto';
    }

    const audio = audioRef.current;

    const updateTime = () => {
      if (audio && !isNaN(audio.currentTime)) {
        setCurrentTime(audio.currentTime);
      }
    };
    const updateDuration = () => {
      if (audio && !isNaN(audio.duration) && isFinite(audio.duration)) {
        setDuration(audio.duration);
      }
    };
    const handleError = (e) => {
      const audio = e.target;
      const error = audio.error;
      let errorMsg = 'Unknown audio error';
      
      if (error) {
        switch (error.code) {
          case error.MEDIA_ERR_ABORTED:
            errorMsg = 'Audio playback aborted';
            break;
          case error.MEDIA_ERR_NETWORK:
            errorMsg = 'Network error loading audio';
            break;
          case error.MEDIA_ERR_DECODE:
            errorMsg = 'Audio decode error';
            break;
          case error.MEDIA_ERR_SRC_NOT_SUPPORTED:
            errorMsg = 'Audio format not supported';
            break;
          default:
            errorMsg = `Audio error code: ${error.code}`;
        }
      }
      
      // Only clear shouldPlayRef for real errors, not during track loading
      // "Empty src attribute" happens during blob URL cleanup, ignore it
      if (error && error.code === 4 && error.message === 'MEDIA_ELEMENT_ERROR: Empty src attribute') {
        // Just log, don't clear shouldPlayRef for cleanup-related errors
        console.log('Audio src cleared during track change (expected)');
      } else {
        console.error('Audio error:', errorMsg, e);
        setIsPlaying(false);
        shouldPlayRef.current = false;
        // Don't auto-advance on error - let user handle it
      }
    };
    const handleCanPlay = () => {
      console.log('Audio can play event fired');
      // Update duration when audio can play
      if (audio && !isNaN(audio.duration) && isFinite(audio.duration)) {
        setDuration(audio.duration);
        console.log('Duration set to:', audio.duration);
      }
      // Audio is ready to play - if we're supposed to be playing, start playback
      if (shouldPlayRef.current && currentTrack) {
        console.log('Attempting to play audio (shouldPlayRef is true)');
        audio.play().then(() => {
          console.log('Audio play() promise resolved');
        }).catch(err => {
          console.error('Failed to play audio:', err);
          setIsPlaying(false);
          shouldPlayRef.current = false;
        });
      }
    };
    const handlePlay = () => {
      console.log('Audio play event fired');
      setIsPlaying(true);
      shouldPlayRef.current = false; // Clear flag once playing
      isLoadingTrackRef.current = false; // Track loading is complete
      // Immediately update currentTime when playback starts
      if (audio && !isNaN(audio.currentTime)) {
        setCurrentTime(audio.currentTime);
      }
    };
    const handlePause = () => {
      console.log('Audio pause event fired, isLoadingTrack:', isLoadingTrackRef.current);
      setIsPlaying(false);
      // Don't clear shouldPlayRef if we're loading a track - it's just a pause during transition
      if (!isLoadingTrackRef.current) {
        shouldPlayRef.current = false;
      }
    };
    const handleEnded = () => {
      console.log('Audio ended event fired, currentTrack:', currentTrack?.id, 'isPlaying:', isPlaying);
      // Handle ended event - don't check isPlaying because it might be false when track ends naturally
      if (currentTrack && handleTrackEndRef.current) {
        console.log('Calling handleTrackEnd callback');
        handleTrackEndRef.current();
      }
    };
    const handleLoadedMetadata = () => {
      // Update duration when metadata loads
      if (audio && !isNaN(audio.duration) && isFinite(audio.duration)) {
        setDuration(audio.duration);
      }
    };

    audio.addEventListener('timeupdate', updateTime);
    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('durationchange', updateDuration);
    audio.addEventListener('error', handleError);
    audio.addEventListener('canplay', handleCanPlay);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', updateTime);
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('durationchange', updateDuration);
      audio.removeEventListener('error', handleError);
      audio.removeEventListener('canplay', handleCanPlay);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
    };
  }, []); // Only run once on mount

  // Update handlers when currentTrack or isPlaying changes
  useEffect(() => {
    if (!audioRef.current) return;
    
    const audio = audioRef.current;
    
    const handleCanPlay = () => {
      console.log('canplay event fired for track:', currentTrack?.id);
      // Update duration when audio can play
      if (audio && !isNaN(audio.duration) && isFinite(audio.duration)) {
        setDuration(audio.duration);
        console.log('Duration set to:', audio.duration);
      }
      // Audio is ready to play - if we're supposed to be playing, start playback
      if (shouldPlayRef.current && currentTrack) {
        console.log('Attempting to play audio (shouldPlayRef is true)');
        audio.play().then(() => {
          console.log('Audio play() promise resolved');
        }).catch(err => {
          console.error('Failed to play audio:', err);
          setIsPlaying(false);
          shouldPlayRef.current = false;
        });
      } else {
        console.log('Not playing - shouldPlayRef:', shouldPlayRef.current, 'currentTrack:', currentTrack?.id);
      }
    };
    
    const handleEnded = () => {
      console.log('Audio ended event fired (track change handler), currentTrack:', currentTrack?.id);
      // Handle ended event - don't check isPlaying because it might be false when track ends naturally
      if (currentTrack && handleTrackEndRef.current) {
        console.log('Calling handleTrackEnd callback (track change handler)');
        handleTrackEndRef.current();
      }
    };
    
    const handlePlay = () => {
      console.log('play event fired');
      // Immediately update currentTime when playback starts
      if (audio && !isNaN(audio.currentTime)) {
        setCurrentTime(audio.currentTime);
      }
    };
    
    const handleLoadedData = () => {
      console.log('loadeddata event fired, readyState:', audio.readyState);
      // Check if we should play when data is loaded
      if (shouldPlayRef.current && currentTrack && audio.readyState >= 2) {
        console.log('Audio has data, attempting to play');
        audio.play().catch(err => {
          console.error('Failed to play after loadeddata:', err);
        });
      }
    };

    audio.addEventListener('canplay', handleCanPlay);
    audio.addEventListener('canplaythrough', handleCanPlay);
    audio.addEventListener('loadeddata', handleLoadedData);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('play', handlePlay);

    // Check if audio is already ready to play (might have loaded before listener was added)
    if (shouldPlayRef.current && currentTrack && audio.readyState >= 3) {
      console.log('Audio already ready (readyState >= 3), attempting immediate play');
      setTimeout(() => {
        if (audioRef.current && shouldPlayRef.current && currentTrack) {
          audioRef.current.play().catch(err => {
            console.error('Failed to play already-ready audio:', err);
          });
        }
      }, 100);
    }

    return () => {
      if (audioRef.current) {
        audioRef.current.removeEventListener('canplay', handleCanPlay);
        audioRef.current.removeEventListener('canplaythrough', handleCanPlay);
        audioRef.current.removeEventListener('loadeddata', handleLoadedData);
        audioRef.current.removeEventListener('ended', handleEnded);
        audioRef.current.removeEventListener('play', handlePlay);
      }
    };
  }, [currentTrack, isPlaying]);

  // Sync volume with audio element
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = isMuted ? 0 : volume;
    }
  }, [volume, isMuted]);

  // Periodic time sync when playing (backup for timeupdate event)
  useEffect(() => {
    if (!isPlaying || !audioRef.current) return;

    const interval = setInterval(() => {
      if (audioRef.current && !isNaN(audioRef.current.currentTime)) {
        setCurrentTime(audioRef.current.currentTime);
      }
    }, 100); // Update every 100ms

    return () => clearInterval(interval);
  }, [isPlaying]);

  // Load track when currentTrack changes
  useEffect(() => {
    if (!currentTrack || !audioRef.current) return;

    const loadTrack = async () => {
      try {
        const audio = audioRef.current;
        // Save shouldPlay state before pausing (pause event will try to clear it)
        const shouldAutoPlay = shouldPlayRef.current;
        
        console.log('Loading track:', currentTrack.id, currentTrack.title, 'shouldAutoPlay:', shouldAutoPlay);
        
        // Mark that we're loading a track (prevents pause handler from clearing shouldPlayRef)
        isLoadingTrackRef.current = true;
        
        // Cleanup previous blob URL BEFORE pausing to avoid "Empty src attribute" error
        if (audio.src && audio.src.startsWith('blob:')) {
          URL.revokeObjectURL(audio.src);
          audio.src = ''; // Clear src before pausing
        }
        
        // Pause and reset audio to prevent false 'ended' events
        audio.pause();
        audio.currentTime = 0;
        setCurrentTime(0);
        setDuration(0); // Reset duration until new track loads
        
        // Restore shouldPlay state after pause (pause event handler won't clear it now)
        shouldPlayRef.current = shouldAutoPlay;
        
        // Get stream URL (proxy endpoint) - pass service_type if available
        const serviceType = currentTrack.service_type || null;
        const streamUrl = await apiService.music.getStreamUrl(currentTrack.id, serviceType);
        console.log('Stream URL received:', streamUrl, 'for service_type:', serviceType);
        
        // Fetch audio with authentication and create blob URL
        // This is necessary because the proxy streaming has issues with format detection
        const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
        console.log('Fetching audio stream with authentication...');
        
        const response = await fetch(streamUrl, {
          headers: token ? { 'Authorization': `Bearer ${token}` } : {}
        });
        
        if (!response.ok) {
          throw new Error(`Failed to load audio stream: ${response.status} ${response.statusText}`);
        }
        
        // Get Content-Type from response for proper format handling
        const contentType = response.headers.get('Content-Type') || 'audio/mpeg';
        console.log('Audio Content-Type:', contentType);
        
        // Create blob URL from response with correct type
        const blob = await response.blob();
        const typedBlob = new Blob([blob], { type: contentType });
        const blobUrl = URL.createObjectURL(typedBlob);
        console.log('Blob URL created, setting audio source, shouldPlayRef:', shouldPlayRef.current);
        
        // Set crossOrigin for CORS if needed
        audio.crossOrigin = 'anonymous';
        audio.src = blobUrl;
        
        // Add error listener for this specific load
        const handleLoadError = (e) => {
          console.error('Error loading audio source:', e);
          const error = audio.error;
          if (error) {
            console.error('Audio error details:', {
              code: error.code,
              message: error.message
            });
            // Only clear shouldPlayRef for real errors, not cleanup errors
            if (error.code !== 4 || error.message !== 'MEDIA_ELEMENT_ERROR: Empty src attribute') {
              shouldPlayRef.current = false;
            }
          }
        };
        
        audio.addEventListener('error', handleLoadError, { once: true });
        
        audio.load();
        console.log('Audio load() called, readyState:', audio.readyState, 'shouldPlayRef:', shouldPlayRef.current);
        
        // If audio is already loaded and we want to play, try playing immediately
        if (shouldPlayRef.current && audio.readyState >= 3) { // HAVE_FUTURE_DATA or higher
          console.log('Audio already loaded, attempting immediate play');
          setTimeout(() => {
            if (audioRef.current && shouldPlayRef.current) {
              audioRef.current.play().catch(err => {
                console.error('Failed to play already-loaded audio:', err);
              });
            }
          }, 50);
        }
        
        // Force an initial time update after a short delay to ensure state is synced
        setTimeout(() => {
          if (audioRef.current && !isNaN(audioRef.current.currentTime)) {
            setCurrentTime(audioRef.current.currentTime);
          }
          // Mark that track loading is complete
          isLoadingTrackRef.current = false;
        }, 100);
      } catch (error) {
        console.error('Failed to load track:', error);
        setIsPlaying(false);
        shouldPlayRef.current = false;
        isLoadingTrackRef.current = false;
      }
    };

    loadTrack();
    
    // Cleanup blob URL when track changes or component unmounts
    return () => {
      if (audioRef.current && audioRef.current.src && audioRef.current.src.startsWith('blob:')) {
        URL.revokeObjectURL(audioRef.current.src);
        audioRef.current.src = '';
      }
    };
  }, [currentTrack]);

  const handleTrackEnd = useCallback(() => {
    if (repeatMode === 'track' && currentTrack) {
      // Repeat current track
      if (audioRef.current) {
        audioRef.current.currentTime = 0;
        shouldPlayRef.current = true;
        audioRef.current.play().catch(console.error);
      }
      return;
    }

    if (repeatMode === 'album' && currentParentId) {
      // Repeat album/playlist - restart from beginning
      if (queue.length > 0) {
        setCurrentIndex(0);
        setCurrentTrack(queue[0]);
        shouldPlayRef.current = true;
        setIsPlaying(true);
        return;
      }
    }

    // Move to next track
    if (handleNextRef.current) {
      handleNextRef.current();
    }
  }, [repeatMode, currentTrack, currentParentId, queue]);

  // Update refs when callbacks change
  useEffect(() => {
    handleTrackEndRef.current = handleTrackEnd;
  }, [handleTrackEnd]);

  const playTrack = useCallback((track, tracks = null, parentId = null) => {
    if (tracks && tracks.length > 0) {
      // Store original queue order
      setOriginalQueue(tracks);
      setCurrentParentId(parentId);
      
      // If shuffle is enabled, handle shuffle logic
      if (shuffleMode && tracks.length > 1) {
        // Check if the selected track exists in the tracks array
        const trackIndex = tracks.findIndex(t => t.id === track.id);
        
        if (trackIndex >= 0) {
          // Specific track was clicked: play it first, then shuffle the rest
          const remainingTracks = tracks.filter(t => t.id !== track.id);
          const shuffled = [...remainingTracks].sort(() => Math.random() - 0.5);
          const newQueue = [track, ...shuffled];
          setQueue(newQueue);
          setCurrentIndex(0);
          setCurrentTrack(track);
        } else {
          // Track not found in array (shouldn't happen, but fallback to random)
          const shuffled = [...tracks].sort(() => Math.random() - 0.5);
          setQueue(shuffled);
          const randomIndex = Math.floor(Math.random() * shuffled.length);
          setCurrentIndex(randomIndex);
          setCurrentTrack(shuffled[randomIndex]);
        }
      } else {
        // Normal mode: play in order starting with selected track
        setQueue(tracks);
        const index = tracks.findIndex(t => t.id === track.id);
        if (index >= 0) {
          setCurrentIndex(index);
          setCurrentTrack(track);
        } else {
          // Track not found, start with first track
          setCurrentIndex(0);
          setCurrentTrack(tracks[0]);
        }
      }
      shouldPlayRef.current = true; // Mark that we want to play when ready
      setIsPlaying(true);
    } else {
      // Play single track
      setQueue([track]);
      setOriginalQueue([track]);
      setCurrentIndex(0);
      setCurrentTrack(track);
      shouldPlayRef.current = true; // Mark that we want to play when ready
      setIsPlaying(true);
    }
  }, [shuffleMode]);

  const togglePlayPause = useCallback(() => {
    if (!audioRef.current || !currentTrack) return;

    if (isPlaying) {
      audioRef.current.pause();
      shouldPlayRef.current = false;
    } else {
      // If audio is ready, play immediately; otherwise wait for canplay event
      if (audioRef.current.readyState >= 2) { // HAVE_CURRENT_DATA or higher
        audioRef.current.play().catch(err => {
          console.error('Failed to play audio:', err);
          setIsPlaying(false);
          shouldPlayRef.current = false;
        });
      } else {
        // Audio not ready yet, set flag and wait for canplay
        shouldPlayRef.current = true;
        setIsPlaying(true);
      }
    }
  }, [isPlaying, currentTrack]);

  const handleNext = useCallback(() => {
    if (queue.length === 0) return;

    let nextIndex;
    if (shuffleMode && queue.length > 1) {
      // Shuffle: pick random from remaining queue
      const remaining = queue.filter((_, idx) => idx !== currentIndex);
      if (remaining.length === 0) {
        // All tracks played, reshuffle original queue
        const shuffled = [...originalQueue].sort(() => Math.random() - 0.5);
        setQueue(shuffled);
        setCurrentIndex(0);
        setCurrentTrack(shuffled[0]);
        shouldPlayRef.current = true;
        setIsPlaying(true);
        return;
      }
      const randomTrack = remaining[Math.floor(Math.random() * remaining.length)];
      nextIndex = queue.findIndex(t => t.id === randomTrack.id);
    } else {
      // Normal: next track
      nextIndex = (currentIndex + 1) % queue.length;
    }

    if (nextIndex >= 0 && nextIndex < queue.length) {
      setCurrentIndex(nextIndex);
      setCurrentTrack(queue[nextIndex]);
      shouldPlayRef.current = true;
      setIsPlaying(true);
    }
  }, [queue, currentIndex, shuffleMode, originalQueue]);

  // Update handleNext ref when callback changes
  useEffect(() => {
    handleNextRef.current = handleNext;
  }, [handleNext]);

  const handlePrevious = useCallback(() => {
    if (queue.length === 0) return;

    let prevIndex;
    if (shuffleMode) {
      // In shuffle mode, previous is less meaningful, just go to previous in queue
      prevIndex = currentIndex > 0 ? currentIndex - 1 : queue.length - 1;
    } else {
      prevIndex = currentIndex > 0 ? currentIndex - 1 : queue.length - 1;
    }

    if (prevIndex >= 0 && prevIndex < queue.length) {
      setCurrentIndex(prevIndex);
      setCurrentTrack(queue[prevIndex]);
      shouldPlayRef.current = true;
      setIsPlaying(true);
    }
  }, [queue, currentIndex, shuffleMode]);

  const handleSeek = useCallback((newValue) => {
    if (audioRef.current) {
      audioRef.current.currentTime = newValue;
      setCurrentTime(newValue);
    }
  }, []);

  const handleVolumeChange = useCallback((newValue) => {
    setVolume(newValue);
    setIsMuted(newValue === 0);
  }, []);

  const toggleMute = useCallback(() => {
    setIsMuted(!isMuted);
  }, [isMuted]);

  const toggleRepeat = useCallback(() => {
    // Cycle: off -> track -> album -> off
    let newMode;
    if (repeatMode === 'off') {
      newMode = 'track';
    } else if (repeatMode === 'track') {
      newMode = 'album';
    } else {
      newMode = 'off';
    }
    setRepeatMode(newMode);
    // Persist to localStorage
    try {
      localStorage.setItem('musicRepeatMode', newMode);
    } catch (error) {
      console.error('Failed to save repeat mode to localStorage:', error);
    }
  }, [repeatMode]);

  const toggleShuffle = useCallback(() => {
    const newShuffleMode = !shuffleMode;
    
    if (newShuffleMode) {
      // Enable shuffle - shuffle remaining tracks
      if (queue.length > 1) {
        const remaining = queue.slice(currentIndex + 1);
        const shuffled = [...remaining].sort(() => Math.random() - 0.5);
        const newQueue = [...queue.slice(0, currentIndex + 1), ...shuffled];
        setQueue(newQueue);
      }
    } else {
      // Disable shuffle - restore original queue order
      setQueue(originalQueue);
      // Find current track in original queue
      if (currentTrack) {
        const index = originalQueue.findIndex(t => t.id === currentTrack.id);
        if (index >= 0) {
          setCurrentIndex(index);
        }
      }
    }
    
    setShuffleMode(newShuffleMode);
    // Persist to localStorage
    try {
      localStorage.setItem('musicShuffleMode', String(newShuffleMode));
    } catch (error) {
      console.error('Failed to save shuffle mode to localStorage:', error);
    }
  }, [shuffleMode, queue, originalQueue, currentIndex, currentTrack]);

  const clearQueue = useCallback(() => {
    setQueue([]);
    setOriginalQueue([]);
    setCurrentTrack(null);
    setCurrentIndex(-1);
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    shouldPlayRef.current = false;
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = '';
    }
  }, []);

  const formatTime = useCallback((seconds) => {
    if (!isFinite(seconds) || isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }, []);

  // Media Session API integration for hardware media keys
  useEffect(() => {
    if (!('mediaSession' in navigator)) {
      console.log('Media Session API not supported');
      return;
    }

    // Update metadata when track changes
    if (currentTrack) {
      navigator.mediaSession.metadata = new MediaMetadata({
        title: currentTrack.title || 'Unknown Track',
        artist: currentTrack.artist || 'Unknown Artist',
        album: currentTrack.album || 'Unknown Album',
        artwork: currentTrack.artwork ? [
          { src: currentTrack.artwork, sizes: '512x512', type: 'image/jpeg' }
        ] : []
      });
    } else {
      navigator.mediaSession.metadata = null;
    }

    // Register action handlers for hardware media keys
    navigator.mediaSession.setActionHandler('play', () => {
      console.log('Media Session: play action');
      if (!isPlaying && audioRef.current && currentTrack) {
        togglePlayPause();
      }
    });

    navigator.mediaSession.setActionHandler('pause', () => {
      console.log('Media Session: pause action');
      if (isPlaying && audioRef.current) {
        togglePlayPause();
      }
    });

    navigator.mediaSession.setActionHandler('previoustrack', () => {
      console.log('Media Session: previoustrack action');
      if (queue.length > 0) {
        handlePrevious();
      }
    });

    navigator.mediaSession.setActionHandler('nexttrack', () => {
      console.log('Media Session: nexttrack action');
      if (queue.length > 0) {
        handleNext();
      }
    });

    // Optional: seek forward/backward (if you want to support these later)
    navigator.mediaSession.setActionHandler('seekbackward', (details) => {
      console.log('Media Session: seekbackward action');
      const skipTime = details.seekOffset || 10;
      if (audioRef.current) {
        audioRef.current.currentTime = Math.max(0, audioRef.current.currentTime - skipTime);
      }
    });

    navigator.mediaSession.setActionHandler('seekforward', (details) => {
      console.log('Media Session: seekforward action');
      const skipTime = details.seekOffset || 10;
      if (audioRef.current) {
        audioRef.current.currentTime = Math.min(duration, audioRef.current.currentTime + skipTime);
      }
    });

    // Cleanup: remove handlers when component unmounts
    return () => {
      if ('mediaSession' in navigator) {
        navigator.mediaSession.setActionHandler('play', null);
        navigator.mediaSession.setActionHandler('pause', null);
        navigator.mediaSession.setActionHandler('previoustrack', null);
        navigator.mediaSession.setActionHandler('nexttrack', null);
        navigator.mediaSession.setActionHandler('seekbackward', null);
        navigator.mediaSession.setActionHandler('seekforward', null);
      }
    };
  }, [currentTrack, isPlaying, queue, handleNext, handlePrevious, togglePlayPause, duration]);

  const value = {
    // State
    currentTrack,
    queue,
    currentIndex,
    isPlaying,
    currentTime,
    duration,
    volume,
    isMuted,
    repeatMode,
    shuffleMode,
    currentParentId,
    
    // Actions
    playTrack,
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
  };

  return (
    <MusicContext.Provider value={value}>
      {children}
    </MusicContext.Provider>
  );
};

