import React, { createContext, useContext, useState, useRef, useEffect, useCallback } from 'react';
import apiService from '../services/apiService';
import { devLog } from '../utils/devConsole';

const MusicContext = createContext(null);

export const useMusic = () => {
  const context = useContext(MusicContext);
  if (!context) {
    throw new Error('useMusic must be used within a MusicProvider');
  }
  return context;
};

const STORAGE_KEY = 'musicPlayerState';
const CURRENT_TIME_SAVE_INTERVAL_MS = 2000;

/** Only use CORS mode when stream URL is cross-origin; same-origin + anonymous can confuse some browsers. */
function applyStreamUrlToAudioElement(audio, streamUrl) {
  try {
    const streamOrigin = new URL(streamUrl, window.location.href).origin;
    if (streamOrigin === window.location.origin) {
      audio.removeAttribute('crossorigin');
    } else {
      audio.crossOrigin = 'anonymous';
    }
  } catch {
    audio.crossOrigin = 'anonymous';
  }
  audio.src = streamUrl;
}

function serializeTrack(track) {
  if (!track) return null;
  return {
    id: track.id,
    title: track.title,
    artist: track.artist,
    album: track.album,
    artwork: track.artwork,
    service_type: track.service_type || null,
    parent_id: track.metadata?.parent_id || track.parent_id || null
  };
}

function loadPlayerState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (!data || (!data.queue && !data.currentTrack)) return null;
    const queue = Array.isArray(data.queue) ? data.queue : [];
    const currentIndex = typeof data.currentIndex === 'number' && data.currentIndex >= 0 && data.currentIndex < queue.length
      ? data.currentIndex
      : 0;
    const currentTrack = queue[currentIndex] || data.currentTrack || null;
    return {
      queue,
      originalQueue: queue,
      currentIndex: queue.length ? currentIndex : -1,
      currentTrack: queue.length ? (currentTrack ? { ...currentTrack } : null) : null,
      currentTime: typeof data.currentTime === 'number' && data.currentTime >= 0 ? data.currentTime : 0,
      volume: typeof data.volume === 'number' && data.volume >= 0 && data.volume <= 1 ? data.volume : 1,
      isMuted: Boolean(data.isMuted),
      restoreTime: typeof data.currentTime === 'number' && data.currentTime > 0 ? data.currentTime : null
    };
  } catch (e) {
    console.error('Failed to load music player state:', e);
    return null;
  }
}

function savePlayerState(state) {
  try {
    const payload = {
      queue: (state.queue || []).map(serializeTrack),
      currentIndex: state.currentIndex,
      currentTime: state.currentTime,
      volume: state.volume,
      isMuted: state.isMuted
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch (e) {
    console.error('Failed to save music player state:', e);
  }
}

function useInitialPlayerState() {
  const ref = useRef(undefined);
  if (ref.current === undefined) ref.current = loadPlayerState();
  return ref.current;
}

export const MusicProvider = ({ children }) => {
  const seekToOnLoadRef = useRef(null);
  const initial = useInitialPlayerState();

  const [currentTrack, setCurrentTrack] = useState(() => initial?.currentTrack ?? null);
  const [queue, setQueue] = useState(() => initial?.queue ?? []);
  const [currentIndex, setCurrentIndex] = useState(() =>
    initial && initial.queue?.length
      ? Math.max(0, Math.min(initial.currentIndex, initial.queue.length - 1))
      : -1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(() =>
    (typeof initial?.currentTime === 'number' && initial.currentTime >= 0) ? initial.currentTime : 0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(() =>
    (typeof initial?.volume === 'number' && initial.volume >= 0 && initial.volume <= 1) ? initial.volume : 1);
  const [isMuted, setIsMuted] = useState(() => Boolean(initial?.isMuted));
  
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
  const [originalQueue, setOriginalQueue] = useState(() =>
    (initial?.queue?.length ? (initial.originalQueue ?? initial.queue) : []));
  const [currentParentId, setCurrentParentId] = useState(null); // For repeat album/playlist
  
  const audioRef = useRef(null);
  const nextTrackAudioRef = useRef(null);

  useEffect(() => {
    if (initial?.restoreTime != null) seekToOnLoadRef.current = initial.restoreTime;
  }, []); // eslint-disable-line react-hooks/exhaustive-deps -- set seek-on-load once from initial restore

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
        devLog('Audio src cleared during track change (expected)');
      } else {
        console.error('Audio error:', errorMsg, e);
        setIsPlaying(false);
        shouldPlayRef.current = false;
        // Don't auto-advance on error - let user handle it
      }
    };
    const handleCanPlay = () => {
      devLog('Audio can play event fired');
      // Update duration when audio can play
      if (audio && !isNaN(audio.duration) && isFinite(audio.duration)) {
        setDuration(audio.duration);
        devLog('Duration set to:', audio.duration);
      }
      // Audio is ready to play - if we're supposed to be playing, start playback
      if (shouldPlayRef.current && currentTrack) {
        devLog('Attempting to play audio (shouldPlayRef is true)');
        audio.play().then(() => {
          devLog('Audio play() promise resolved');
        }).catch(err => {
          console.error('Failed to play audio:', err);
          setIsPlaying(false);
          shouldPlayRef.current = false;
        });
      }
    };
    const handlePlay = () => {
      devLog('Audio play event fired');
      setIsPlaying(true);
      shouldPlayRef.current = false; // Clear flag once playing
      isLoadingTrackRef.current = false; // Track loading is complete
      // Immediately update currentTime when playback starts
      if (audio && !isNaN(audio.currentTime)) {
        setCurrentTime(audio.currentTime);
      }
    };
    const handlePause = () => {
      devLog('Audio pause event fired, isLoadingTrack:', isLoadingTrackRef.current);
      setIsPlaying(false);
      // Don't clear shouldPlayRef if we're loading a track - it's just a pause during transition
      if (!isLoadingTrackRef.current) {
        shouldPlayRef.current = false;
      }
    };
    const handleEnded = () => {
      devLog('Audio ended event fired, currentTrack:', currentTrack?.id, 'isPlaying:', isPlaying);
      // Handle ended event - don't check isPlaying because it might be false when track ends naturally
      if (currentTrack && handleTrackEndRef.current) {
        devLog('Calling handleTrackEnd callback');
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
      devLog('canplay event fired for track:', currentTrack?.id);
      // Update duration when audio can play
      if (audio && !isNaN(audio.duration) && isFinite(audio.duration)) {
        setDuration(audio.duration);
        devLog('Duration set to:', audio.duration);
      }
      // Audio is ready to play - if we're supposed to be playing, start playback
      if (shouldPlayRef.current && currentTrack) {
        devLog('Attempting to play audio (shouldPlayRef is true)');
        audio.play().then(() => {
          devLog('Audio play() promise resolved');
        }).catch(err => {
          console.error('Failed to play audio:', err);
          setIsPlaying(false);
          shouldPlayRef.current = false;
        });
      } else {
        devLog('Not playing - shouldPlayRef:', shouldPlayRef.current, 'currentTrack:', currentTrack?.id);
      }
    };
    
    const handleEnded = () => {
      devLog('Audio ended event fired (track change handler), currentTrack:', currentTrack?.id);
      // Handle ended event - don't check isPlaying because it might be false when track ends naturally
      if (currentTrack && handleTrackEndRef.current) {
        devLog('Calling handleTrackEnd callback (track change handler)');
        handleTrackEndRef.current();
      }
    };
    
    const handlePlay = () => {
      devLog('play event fired');
      // Immediately update currentTime when playback starts
      if (audio && !isNaN(audio.currentTime)) {
        setCurrentTime(audio.currentTime);
      }
    };
    
    const handleLoadedData = () => {
      devLog('loadeddata event fired, readyState:', audio.readyState);
      // Check if we should play when data is loaded
      if (shouldPlayRef.current && currentTrack && audio.readyState >= 2) {
        devLog('Audio has data, attempting to play');
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
      devLog('Audio already ready (readyState >= 3), attempting immediate play');
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

  // Persist player state to localStorage (track, queue, volume, mute; throttle currentTime while playing)
  useEffect(() => {
    if (!currentTrack && queue.length === 0) return;
    savePlayerState({
      queue,
      currentIndex,
      currentTime,
      volume,
      isMuted
    });
  }, [currentTrack, queue, currentIndex, volume, isMuted]);

  const lastSavedTimeRef = useRef(0);
  useEffect(() => {
    if (!isPlaying || !currentTrack) return;
    const interval = setInterval(() => {
      if (audioRef.current && typeof audioRef.current.currentTime === 'number') {
        const t = audioRef.current.currentTime;
        if (Date.now() - lastSavedTimeRef.current >= CURRENT_TIME_SAVE_INTERVAL_MS) {
          lastSavedTimeRef.current = Date.now();
          savePlayerState({
            queue,
            currentIndex,
            currentTime: t,
            volume,
            isMuted
          });
        }
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [isPlaying, currentTrack, queue, currentIndex, volume, isMuted]);

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

    const loadTrack = () => {
      try {
        const audio = audioRef.current;
        const shouldAutoPlay = shouldPlayRef.current;

        isLoadingTrackRef.current = true;

        // Clear previous source (no blob URLs; direct stream URLs only)
        audio.pause();
        audio.removeAttribute('src');
        audio.load();
        audio.currentTime = 0;
        setCurrentTime(0);
        setDuration(0);

        shouldPlayRef.current = shouldAutoPlay;

        const serviceType = currentTrack.service_type || null;
        const parentId = currentTrack.metadata?.parent_id || currentTrack.parent_id || null;
        const streamUrl = apiService.music.getStreamUrl(currentTrack.id, serviceType, parentId);

        applyStreamUrlToAudioElement(audio, streamUrl);

        const handleLoadError = (e) => {
          const err = audio.error;
          if (err && (err.code === 4 && err.message === 'MEDIA_ELEMENT_ERROR: Empty src attribute')) return;
          console.error('Error loading audio source:', e);
          if (err) {
            console.error('Audio error details:', { code: err.code, message: err.message });
            shouldPlayRef.current = false;
          }
        };

        audio.addEventListener('error', handleLoadError, { once: true });
        audio.load();

        const seekTo = seekToOnLoadRef.current;
        if (seekTo != null && seekTo > 0) {
          const applySeek = () => {
            if (audioRef.current && !isNaN(seekTo)) {
              audioRef.current.currentTime = seekTo;
              setCurrentTime(seekTo);
            }
            seekToOnLoadRef.current = null;
          };
          audio.addEventListener('canplay', applySeek, { once: true });
          if (audio.readyState >= 3) setTimeout(applySeek, 0);
        }

        if (shouldPlayRef.current && audio.readyState >= 3) {
          setTimeout(() => {
            if (audioRef.current && shouldPlayRef.current) {
              audioRef.current.play().catch(err => {
                console.error('Failed to play already-loaded audio:', err);
              });
            }
          }, 50);
        }

        setTimeout(() => {
          if (audioRef.current && !isNaN(audioRef.current.currentTime)) {
            setCurrentTime(audioRef.current.currentTime);
          }
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
  }, [currentTrack]);

  // Preload next track in a hidden audio element for near-gapless transition
  useEffect(() => {
    const nextIndex = currentIndex >= 0 && queue.length > 0 ? currentIndex + 1 : -1;
    const hasNext = nextIndex >= 0 && nextIndex < queue.length;
    if (!hasNext || !queue[nextIndex]) {
      if (nextTrackAudioRef.current) {
        nextTrackAudioRef.current.removeAttribute('src');
        nextTrackAudioRef.current.load();
      }
      return;
    }
    const nextTrack = queue[nextIndex];
    if (!nextTrack.id) return;
    if (!nextTrackAudioRef.current) {
      nextTrackAudioRef.current = new Audio();
      nextTrackAudioRef.current.preload = 'auto';
    }
    const nextParentId = nextTrack.metadata?.parent_id || nextTrack.parent_id || null;
    const url = apiService.music.getStreamUrl(nextTrack.id, nextTrack.service_type || null, nextParentId);
    const resolvedUrl = new URL(url, window.location.href).href;
    if (nextTrackAudioRef.current.src !== resolvedUrl) {
      applyStreamUrlToAudioElement(nextTrackAudioRef.current, url);
      nextTrackAudioRef.current.load();
    }
  }, [queue, currentIndex]);

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

    if (repeatMode === 'album' && currentParentId && queue.length > 0) {
      // Repeat album/playlist: only restart from beginning when we just finished the last track
      const isLastTrack = currentIndex === queue.length - 1;
      if (isLastTrack) {
        setCurrentIndex(0);
        setCurrentTrack(queue[0]);
        shouldPlayRef.current = true;
        setIsPlaying(true);
        return;
      }
    }

    // Move to next track (repeat off, or repeat album but not at end yet)
    if (handleNextRef.current) {
      handleNextRef.current();
    }
  }, [repeatMode, currentTrack, currentParentId, queue, currentIndex]);

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
    seekToOnLoadRef.current = null;
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = '';
    }
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      console.error('Failed to clear music player state:', e);
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
      devLog('Media Session API not supported');
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
      devLog('Media Session: play action');
      if (!isPlaying && audioRef.current && currentTrack) {
        togglePlayPause();
      }
    });

    navigator.mediaSession.setActionHandler('pause', () => {
      devLog('Media Session: pause action');
      if (isPlaying && audioRef.current) {
        togglePlayPause();
      }
    });

    navigator.mediaSession.setActionHandler('previoustrack', () => {
      devLog('Media Session: previoustrack action');
      if (queue.length > 0) {
        handlePrevious();
      }
    });

    navigator.mediaSession.setActionHandler('nexttrack', () => {
      devLog('Media Session: nexttrack action');
      if (queue.length > 0) {
        handleNext();
      }
    });

    // Optional: seek forward/backward (if you want to support these later)
    navigator.mediaSession.setActionHandler('seekbackward', (details) => {
      devLog('Media Session: seekbackward action');
      const skipTime = details.seekOffset || 10;
      if (audioRef.current) {
        audioRef.current.currentTime = Math.max(0, audioRef.current.currentTime - skipTime);
      }
    });

    navigator.mediaSession.setActionHandler('seekforward', (details) => {
      devLog('Media Session: seekforward action');
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

