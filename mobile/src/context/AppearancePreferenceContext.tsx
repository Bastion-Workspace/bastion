import * as SecureStore from 'expo-secure-store';
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { Appearance, useColorScheme } from 'react-native';

/** Must match SecureStore key rules: alphanumeric, `.`, `-`, `_` only (no `:`). */
const KEY = 'app.appearancePreference';

export type AppearancePreference = 'system' | 'light' | 'dark';

type Ctx = {
  preference: AppearancePreference;
  setPreference: (next: AppearancePreference) => Promise<void>;
  hydrated: boolean;
  resolvedScheme: 'light' | 'dark';
};

const AppearancePreferenceContext = createContext<Ctx | null>(null);

function readSyncPreference(): AppearancePreference {
  try {
    const raw = SecureStore.getItem(KEY);
    if (raw === 'light' || raw === 'dark' || raw === 'system') {
      return raw;
    }
  } catch {
    /* keep default */
  }
  return 'system';
}

function applyAppearanceFromPreference(pref: AppearancePreference): void {
  if (pref === 'system') {
    Appearance.setColorScheme(null);
  } else {
    Appearance.setColorScheme(pref);
  }
}

const initialPreference = readSyncPreference();
applyAppearanceFromPreference(initialPreference);

export function AppearancePreferenceProvider({ children }: { children: ReactNode }) {
  const [preference, setPreferenceState] = useState<AppearancePreference>(initialPreference);
  const [hydrated] = useState(true);
  const scheme = useColorScheme();

  useEffect(() => {
    applyAppearanceFromPreference(preference);
  }, [preference]);

  const setPreference = useCallback(async (next: AppearancePreference) => {
    setPreferenceState(next);
    applyAppearanceFromPreference(next);
    await SecureStore.setItemAsync(KEY, next);
  }, []);

  const resolvedScheme: 'light' | 'dark' = useMemo(() => {
    if (preference === 'light') return 'light';
    if (preference === 'dark') return 'dark';
    return scheme === 'dark' ? 'dark' : 'light';
  }, [preference, scheme]);

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
