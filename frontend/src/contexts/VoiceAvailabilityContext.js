import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import apiService from '../services/apiService';

const defaultPrefs = {
  effective_server_voice_id: '',
  effective_server_provider: '',
  byokTtsActive: false,
  prefer_browser_tts: true,
};

const VoiceAvailabilityContext = createContext({
  serverAvailable: false,
  serverTtsPrefs: defaultPrefs,
  prefsLoaded: true,
  refreshAvailability: async () => {},
  refreshPrefs: async () => defaultPrefs,
});

export function VoiceAvailabilityProvider({ children }) {
  const [serverAvailable, setServerAvailable] = useState(() => {
    const cached = apiService.voice.getAvailabilityFromCache?.();
    return cached === null || cached === undefined ? false : cached;
  });
  const [serverTtsPrefs, setServerTtsPrefs] = useState(defaultPrefs);
  const [prefsLoaded, setPrefsLoaded] = useState(false);

  const refreshPrefs = useCallback(async () => {
    try {
      const s = await apiService.getUserVoiceSettings();
      const byokTtsActive = s?.use_admin_tts === false;
      const next = {
        effective_server_voice_id: s?.effective_server_voice_id || '',
        effective_server_provider: s?.effective_server_provider || '',
        byokTtsActive,
        prefer_browser_tts: !!s?.prefer_browser_tts,
      };
      setServerTtsPrefs(next);
      return next;
    } catch {
      setServerTtsPrefs(defaultPrefs);
      return defaultPrefs;
    } finally {
      setPrefsLoaded(true);
    }
  }, []);

  const refreshAvailability = useCallback(async (forceRefresh = false) => {
    try {
      const result = await apiService.voice.checkAvailability(forceRefresh);
      setServerAvailable(!!result?.available);
    } catch {
      setServerAvailable(false);
    }
  }, []);

  useEffect(() => {
    refreshAvailability(false);
    refreshPrefs();
  }, [refreshAvailability, refreshPrefs]);

  useEffect(() => {
    const onPrefsChanged = () => {
      refreshPrefs();
    };
    window.addEventListener('bastion-voice-settings-changed', onPrefsChanged);
    return () => {
      window.removeEventListener('bastion-voice-settings-changed', onPrefsChanged);
    };
  }, [refreshPrefs]);

  const value = useMemo(
    () => ({
      serverAvailable,
      serverTtsPrefs,
      prefsLoaded,
      refreshAvailability,
      refreshPrefs,
    }),
    [serverAvailable, serverTtsPrefs, prefsLoaded, refreshAvailability, refreshPrefs]
  );

  return (
    <VoiceAvailabilityContext.Provider value={value}>
      {children}
    </VoiceAvailabilityContext.Provider>
  );
}

export function useVoiceAvailability() {
  return useContext(VoiceAvailabilityContext);
}
