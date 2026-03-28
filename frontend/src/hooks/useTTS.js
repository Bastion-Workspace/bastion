import { useCallback, useEffect, useRef, useState } from 'react';
import apiService from '../services/apiService';
import { useVoiceAvailability } from '../contexts/VoiceAvailabilityContext';
import {
  mimeForStreamFormat,
  splitTextForTts,
  TTS_CHUNK_THRESHOLD_CHARS,
} from '../utils/ttsStreamUtils';

const PREFERRED_BROWSER_VOICE_KEYS = ['tts_voice_uri', 'ttsVoiceURI', 'voice_uri'];
const PREFERRED_RATE_KEYS = ['tts_rate', 'ttsRate'];
const PREFERRED_PITCH_KEYS = ['tts_pitch', 'ttsPitch'];
const PREFERRED_VOLUME_KEYS = ['tts_volume', 'ttsVolume'];
const PREFERRED_PROVIDER_KEYS = ['tts_provider', 'ttsProvider'];
const PREFERRED_SERVER_VOICE_KEYS = ['tts_voice_id', 'ttsVoiceId'];

const readNumberFromStorage = (keys, fallback, min, max) => {
  for (const key of keys) {
    const raw = localStorage.getItem(key);
    if (raw == null) continue;
    const value = Number(raw);
    if (Number.isFinite(value)) {
      return Math.min(max, Math.max(min, value));
    }
  }
  return fallback;
};

const readStringFromStorage = (keys) => {
  for (const key of keys) {
    const value = localStorage.getItem(key);
    if (value && value.trim()) return value.trim();
  }
  return '';
};

const resolveBrowserVoice = (speechSynthesis) => {
  const voices = speechSynthesis.getVoices() || [];
  if (!voices.length) return null;

  const preferredVoiceUri = readStringFromStorage(PREFERRED_BROWSER_VOICE_KEYS);
  if (preferredVoiceUri) {
    const explicitMatch = voices.find((v) => v.voiceURI === preferredVoiceUri);
    if (explicitMatch) return explicitMatch;
  }

  const preferredLang = (localStorage.getItem('tts_lang') || navigator.language || '').toLowerCase();
  if (preferredLang) {
    const langPrefix = preferredLang.slice(0, 2);
    const langMatch = voices.find((v) => (v.lang || '').toLowerCase().startsWith(langPrefix));
    if (langMatch) return langMatch;
  }

  return voices.find((v) => v.default) || voices[0] || null;
};

const waitSourceBufferIdle = (sb) =>
  new Promise((resolve) => {
    if (!sb.updating) {
      resolve();
      return;
    }
    sb.addEventListener('updateend', () => resolve(), { once: true });
  });

/** When using account voice settings (BYOK path), ignore localStorage provider/voice. */
const ttsLocalStorageFallbacks = (serverPrefs) => {
  if (serverPrefs?.byokTtsActive) {
    return { preferredProvider: '', preferredVoiceId: '' };
  }
  return {
    preferredProvider: readStringFromStorage(PREFERRED_PROVIDER_KEYS),
    preferredVoiceId: readStringFromStorage(PREFERRED_SERVER_VOICE_KEYS),
  };
};

export const useTTS = ({ stopEventName = 'tts-stop-all' } = {}) => {
  const {
    serverAvailable,
    serverTtsPrefs,
    prefsLoaded,
    refreshPrefs,
  } = useVoiceAvailability();

  const [isSpeaking, setIsSpeaking] = useState(false);

  const utteranceRef = useRef(null);
  const audioRef = useRef(null);
  const objectUrlRef = useRef(null);
  const modeRef = useRef(null);
  const ttsSessionRef = useRef(0);
  const abortControllerRef = useRef(null);

  const supportsBrowserSpeech =
    typeof window !== 'undefined' &&
    'speechSynthesis' in window &&
    'SpeechSynthesisUtterance' in window;

  const cleanupObjectUrl = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    try {
      abortControllerRef.current?.abort();
    } catch {
      /* ignore */
    }
    abortControllerRef.current = null;
    ttsSessionRef.current += 1;
    if (supportsBrowserSpeech) {
      window.speechSynthesis.cancel();
    }
    const audio = audioRef.current;
    if (audio) {
      audio.onplay = null;
      audio.onended = null;
      audio.onerror = null;
      audio.pause();
      audio.removeAttribute('src');
      audio.load();
      audioRef.current = null;
    }
    cleanupObjectUrl();
    utteranceRef.current = null;
    modeRef.current = null;
    setIsSpeaking(false);
  }, [cleanupObjectUrl, supportsBrowserSpeech]);

  const speakWithBrowser = useCallback(
    (text, session) =>
      new Promise((resolve, reject) => {
        if (!supportsBrowserSpeech) {
          reject(new Error('Browser speech synthesis is not available'));
          return;
        }

        const speechSynthesis = window.speechSynthesis;
        speechSynthesis.cancel();

        const utterance = new window.SpeechSynthesisUtterance(text);
        utterance.voice = resolveBrowserVoice(speechSynthesis);
        utterance.rate = readNumberFromStorage(PREFERRED_RATE_KEYS, 1, 0.5, 2);
        utterance.pitch = readNumberFromStorage(PREFERRED_PITCH_KEYS, 1, 0.5, 2);
        utterance.volume = readNumberFromStorage(PREFERRED_VOLUME_KEYS, 1, 0, 1);

        utterance.onstart = () => {
          if (session !== ttsSessionRef.current) return;
          utteranceRef.current = utterance;
          modeRef.current = 'browser';
          setIsSpeaking(true);
        };
        utterance.onend = () => {
          utteranceRef.current = null;
          modeRef.current = null;
          setIsSpeaking(false);
          if (session !== ttsSessionRef.current) {
            resolve();
            return;
          }
          resolve();
        };
        utterance.onerror = (event) => {
          utteranceRef.current = null;
          modeRef.current = null;
          setIsSpeaking(false);
          if (session !== ttsSessionRef.current) {
            resolve();
            return;
          }
          reject(new Error(event?.error || 'Browser TTS failed'));
        };

        speechSynthesis.speak(utterance);
      }),
    [supportsBrowserSpeech]
  );

  /**
   * Streamed server TTS via MediaSource. Uses X-Audio-Format for MIME (OGG/Opus or MP3).
   * Returns true on success, false if session superseded, null to fall back to buffered TTS.
   */
  const speakWithServerStream = useCallback(
    async (text, options = {}, session, serverPrefs) => {
      if (typeof window === 'undefined' || typeof MediaSource === 'undefined') {
        return null;
      }

      const { preferredProvider, preferredVoiceId } = ttsLocalStorageFallbacks(serverPrefs);
      const eff = serverPrefs || {
        effective_server_provider: '',
        effective_server_voice_id: '',
        byokTtsActive: false,
        prefer_browser_tts: false,
      };

      const explicitFormat = options.outputFormat
        ? String(options.outputFormat).toLowerCase()
        : '';
      if (explicitFormat === 'wav') {
        return null;
      }

      const streamOpts = {
        provider:
          options.provider ||
          eff.effective_server_provider ||
          preferredProvider ||
          '',
        voiceId:
          options.voiceId ||
          eff.effective_server_voice_id ||
          preferredVoiceId ||
          '',
        outputFormat:
          explicitFormat && explicitFormat !== 'ogg' ? explicitFormat : 'ogg',
        ...(options.signal ? { signal: options.signal } : {}),
      };

      let response;
      try {
        response = await apiService.voice.fetchSynthesizeStream(text, streamOpts);
      } catch (err) {
        if (err?.name === 'AbortError') {
          return false;
        }
        return null;
      }

      const headerFmt = (response.headers.get('X-Audio-Format') || 'ogg').toLowerCase();
      const streamMime = mimeForStreamFormat(headerFmt);
      if (!streamMime || !MediaSource.isTypeSupported(streamMime)) {
        try {
          response.body?.cancel?.();
        } catch {
          /* ignore */
        }
        return null;
      }

      if (session !== ttsSessionRef.current) {
        return false;
      }

      const body = response.body;
      if (!body || typeof body.getReader !== 'function') {
        return null;
      }

      return new Promise((resolve, reject) => {
        if (session !== ttsSessionRef.current) {
          resolve(false);
          return;
        }

        const mediaSource = new MediaSource();
        const objectUrl = URL.createObjectURL(mediaSource);
        objectUrlRef.current = objectUrl;
        const audio = new Audio(objectUrl);
        audioRef.current = audio;

        const teardownReject = (err) => {
          audioRef.current = null;
          modeRef.current = null;
          setIsSpeaking(false);
          cleanupObjectUrl();
          if (err?.name === 'AbortError') {
            resolve(false);
            return;
          }
          if (session !== ttsSessionRef.current) {
            resolve(false);
            return;
          }
          reject(err);
        };

        audio.onplay = () => {
          if (session !== ttsSessionRef.current) return;
          modeRef.current = 'server';
          setIsSpeaking(true);
        };
        audio.onended = () => {
          audioRef.current = null;
          modeRef.current = null;
          setIsSpeaking(false);
          cleanupObjectUrl();
          resolve(true);
        };
        audio.onerror = () => {
          audioRef.current = null;
          modeRef.current = null;
          setIsSpeaking(false);
          cleanupObjectUrl();
          if (session !== ttsSessionRef.current) {
            resolve(false);
            return;
          }
          reject(new Error('Server TTS streaming playback failed'));
        };

        mediaSource.addEventListener('sourceopen', () => {
          let sourceBuffer;
          try {
            sourceBuffer = mediaSource.addSourceBuffer(streamMime);
          } catch (e) {
            teardownReject(e);
            return;
          }

          const reader = body.getReader();

          const pump = async () => {
            try {
              while (true) {
                if (session !== ttsSessionRef.current) {
                  try {
                    await reader.cancel();
                  } catch {
                    /* ignore */
                  }
                  try {
                    reader.releaseLock();
                  } catch {
                    /* ignore */
                  }
                  cleanupObjectUrl();
                  resolve(false);
                  return;
                }
                let readResult;
                try {
                  readResult = await reader.read();
                } catch (readErr) {
                  if (readErr?.name === 'AbortError') {
                    try {
                      await reader.cancel();
                    } catch {
                      /* ignore */
                    }
                    cleanupObjectUrl();
                    resolve(false);
                    return;
                  }
                  teardownReject(readErr);
                  return;
                }
                const { done, value } = readResult;
                if (done) {
                  break;
                }
                if (value && value.byteLength > 0) {
                  await waitSourceBufferIdle(sourceBuffer);
                  if (session !== ttsSessionRef.current) {
                    try {
                      await reader.cancel();
                    } catch {
                      /* ignore */
                    }
                    try {
                      reader.releaseLock();
                    } catch {
                      /* ignore */
                    }
                    cleanupObjectUrl();
                    resolve(false);
                    return;
                  }
                  const copy = value.buffer.slice(
                    value.byteOffset,
                    value.byteOffset + value.byteLength
                  );
                  sourceBuffer.appendBuffer(copy);
                }
              }
              await waitSourceBufferIdle(sourceBuffer);
              if (mediaSource.readyState === 'open') {
                mediaSource.endOfStream();
              }
            } catch (e) {
              if (e?.name === 'AbortError') {
                cleanupObjectUrl();
                resolve(false);
                return;
              }
              teardownReject(e);
            }
          };

          pump();
        });

        audio.play().catch((err) => {
          if (err?.name === 'AbortError') {
            cleanupObjectUrl();
            resolve(false);
            return;
          }
          if (session !== ttsSessionRef.current) {
            cleanupObjectUrl();
            resolve(false);
            return;
          }
          teardownReject(err);
        });
      });
    },
    [cleanupObjectUrl]
  );

  const speakWithServer = useCallback(
    async (text, options = {}, session, serverPrefs) => {
      const { preferredProvider, preferredVoiceId } = ttsLocalStorageFallbacks(serverPrefs);
      const eff = serverPrefs || {
        effective_server_provider: '',
        effective_server_voice_id: '',
        byokTtsActive: false,
        prefer_browser_tts: false,
      };

      const synthOpts = {
        provider:
          options.provider ||
          eff.effective_server_provider ||
          preferredProvider ||
          '',
        voiceId:
          options.voiceId ||
          eff.effective_server_voice_id ||
          preferredVoiceId ||
          '',
        outputFormat: options.outputFormat || 'mp3',
        ...(options.signal ? { signal: options.signal } : {}),
      };

      let blob;
      try {
        const result = await apiService.voice.synthesize(text, synthOpts);
        blob = result.blob;
      } catch (err) {
        if (err?.name === 'AbortError') {
          return false;
        }
        throw err;
      }

      if (session !== ttsSessionRef.current) {
        return false;
      }

      return new Promise((resolve, reject) => {
        if (session !== ttsSessionRef.current) {
          resolve(false);
          return;
        }

        const objectUrl = URL.createObjectURL(blob);
        objectUrlRef.current = objectUrl;

        const audio = new Audio(objectUrl);
        audioRef.current = audio;

        audio.onplay = () => {
          if (session !== ttsSessionRef.current) return;
          modeRef.current = 'server';
          setIsSpeaking(true);
        };
        audio.onended = () => {
          audioRef.current = null;
          modeRef.current = null;
          setIsSpeaking(false);
          cleanupObjectUrl();
          resolve(true);
        };
        audio.onerror = () => {
          audioRef.current = null;
          modeRef.current = null;
          setIsSpeaking(false);
          cleanupObjectUrl();
          if (session !== ttsSessionRef.current) {
            resolve(false);
            return;
          }
          reject(new Error('Server TTS playback failed'));
        };

        audio.play().catch((err) => {
          if (session !== ttsSessionRef.current) {
            cleanupObjectUrl();
            resolve(false);
            return;
          }
          audioRef.current = null;
          modeRef.current = null;
          setIsSpeaking(false);
          cleanupObjectUrl();
          reject(err);
        });
      });
    },
    [cleanupObjectUrl]
  );

  const playOneServerSegment = useCallback(
    async (segmentText, options, session, serverPrefs) => {
      const streamed = await speakWithServerStream(
        segmentText,
        options,
        session,
        serverPrefs
      );
      if (streamed === false) {
        return false;
      }
      if (streamed === true) {
        return true;
      }
      const played = await speakWithServer(segmentText, options, session, serverPrefs);
      return played !== false;
    },
    [speakWithServer, speakWithServerStream]
  );

  const speak = useCallback(
    async (text, options = {}) => {
      const cleanText = String(text || '').trim();
      if (!cleanText) return;

      stop();
      window.dispatchEvent(new CustomEvent(stopEventName));
      const session = ++ttsSessionRef.current;
      const ac = new AbortController();
      abortControllerRef.current = ac;
      const ttsOptions = { ...options, signal: ac.signal };

      const serverPrefs = prefsLoaded ? serverTtsPrefs : await refreshPrefs();

      if (serverPrefs.prefer_browser_tts) {
        if (supportsBrowserSpeech) {
          await speakWithBrowser(cleanText, session);
          return;
        }
        throw new Error(
          'Browser TTS is selected in settings but speech synthesis is not available'
        );
      }

      if (serverAvailable) {
        try {
          const chunks = splitTextForTts(cleanText, TTS_CHUNK_THRESHOLD_CHARS);
          const useChunks = chunks.length > 1;

          if (useChunks) {
            for (const segment of chunks) {
              if (session !== ttsSessionRef.current) {
                return;
              }
              const seg = segment.trim();
              if (!seg) continue;
              const ok = await playOneServerSegment(seg, ttsOptions, session, serverPrefs);
              if (!ok) {
                return;
              }
            }
            return;
          }

          const streamed = await speakWithServerStream(
            cleanText,
            ttsOptions,
            session,
            serverPrefs
          );
          if (streamed === false) {
            return;
          }
          if (streamed === true) {
            if (session !== ttsSessionRef.current) {
              return;
            }
            return;
          }
          const played = await speakWithServer(
            cleanText,
            ttsOptions,
            session,
            serverPrefs
          );
          if (played === false) {
            return;
          }
          if (session !== ttsSessionRef.current) {
            return;
          }
          return;
        } catch (err) {
          if (session !== ttsSessionRef.current) {
            return;
          }
          if (err?.name === 'AbortError') {
            return;
          }
          console.warn('Server TTS failed, falling back to browser speech:', err);
        }
      }

      if (session !== ttsSessionRef.current) {
        return;
      }

      if (supportsBrowserSpeech) {
        await speakWithBrowser(cleanText, session);
        return;
      }

      throw new Error('No TTS path available');
    },
    [
      prefsLoaded,
      serverTtsPrefs,
      refreshPrefs,
      serverAvailable,
      playOneServerSegment,
      speakWithBrowser,
      speakWithServer,
      speakWithServerStream,
      stop,
      stopEventName,
      supportsBrowserSpeech,
    ]
  );

  useEffect(() => {
    const handleStop = () => stop();
    window.addEventListener(stopEventName, handleStop);
    return () => {
      window.removeEventListener(stopEventName, handleStop);
      stop();
    };
  }, [stop, stopEventName]);

  return {
    isSpeaking,
    serverAvailable,
    supportsBrowserSpeech,
    canSpeak: serverTtsPrefs.prefer_browser_tts
      ? supportsBrowserSpeech
      : serverAvailable || supportsBrowserSpeech,
    speak,
    stop,
  };
};

export default useTTS;
