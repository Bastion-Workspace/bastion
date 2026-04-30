import * as SecureStore from 'expo-secure-store';
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { Appearance, useColorScheme } from 'react-native';

const KEY = 'app:appearancePreference';

export type AppearancePreference = 'system' | 'light' | 'dark';

type Ctx = {
  preference: AppearancePreference;
  setPreference: (next: AppearancePreference) => Promise<void>;
  hydrated: boolean;
  resolvedScheme: 'light' | 'dark';
};

const AppearancePreferenceContext = createContext<Ctx | null>(null);

export function AppearancePreferenceProvider({ children }: { children: ReactNode }) {
  const [preference, setPreferenceState] = useState<AppearancePreference>('system');
  const [hydrated, setHydrated] = useState(false);
  const scheme = useColorScheme();

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const raw = await SecureStore.getItemAsync(KEY);
        if (cancelled) return;
        if (raw === 'light' || raw === 'dark' || raw === 'system') {
          setPreferenceState(raw);
        }
      } catch {
        /* keep default */
      } finally {
        if (!cancelled) setHydrated(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    if (preference === 'system') {
      Appearance.setColorScheme(null);
    } else {
      Appearance.setColorScheme(preference);
    }
  }, [hydrated, preference]);

  const setPreference = useCallback(async (next: AppearancePreference) => {
    setPreferenceState(next);
    try {
      await SecureStore.setItemAsync(KEY, next);
    } catch {
      /* ignore */
    }
  }, []);

  const resolvedScheme: 'light' | 'dark' = scheme === 'dark' ? 'dark' : 'light';

  const value = useMemo(
    () => ({ preference, setPreference, hydrated, resolvedScheme }),
    [preference, setPreference, hydrated, resolvedScheme]
  );

  return (
    <AppearancePreferenceContext.Provider value={value}>{children}</AppearancePreferenceContext.Provider>
  );
}

export function useAppearancePreference(): Ctx {
  const v = useContext(AppearancePreferenceContext);
  if (!v) {
    throw new Error('useAppearancePreference must be used within AppearancePreferenceProvider');
  }
  return v;
}
