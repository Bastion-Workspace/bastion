import React, { createContext, useContext, useState, useEffect, useMemo, useCallback } from 'react';

const ThemeContext = createContext(null);

export const ACCENT_IDS = ['blue', 'teal', 'green', 'purple', 'orange', 'indigo', 'rose', 'cyan', 'amber', 'red'];
export const DEFAULT_ACCENT_ID = 'blue';

const STORAGE_THEME_PREF = 'themePreference';
const STORAGE_LEGACY_DARK = 'darkMode';
const STORAGE_KEY_ACCENT = 'accentTheme';

/** @typedef {'light' | 'dark' | 'system'} ThemePreference */

function readInitialPreference() {
  try {
    const pref = localStorage.getItem(STORAGE_THEME_PREF);
    if (pref === 'light' || pref === 'dark' || pref === 'system') {
      return pref;
    }
    const legacy = localStorage.getItem(STORAGE_LEGACY_DARK);
    if (legacy !== null) {
      return JSON.parse(legacy) ? 'dark' : 'light';
    }
  } catch {
    // ignore
  }
  return 'system';
}

function readSystemPrefersDark() {
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

export const ThemeProvider = ({ children }) => {
  const [themePreference, setThemePreferenceState] = useState(readInitialPreference);
  const [systemPrefersDark, setSystemPrefersDark] = useState(readSystemPrefersDark);

  const [accentId, setAccentIdState] = useState(() => {
    const saved = localStorage.getItem(STORAGE_KEY_ACCENT);
    if (saved && ACCENT_IDS.includes(saved)) {
      return saved;
    }
    return DEFAULT_ACCENT_ID;
  });

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = (e) => setSystemPrefersDark(e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  const darkMode = useMemo(() => {
    if (themePreference === 'system') {
      return systemPrefersDark;
    }
    return themePreference === 'dark';
  }, [themePreference, systemPrefersDark]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_THEME_PREF, themePreference);
    } catch {
      // ignore
    }
  }, [themePreference]);

  useEffect(() => {
    document.documentElement.classList.toggle('dark-mode', darkMode);
  }, [darkMode]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY_ACCENT, accentId);
  }, [accentId]);

  const setAccentId = (id) => {
    if (ACCENT_IDS.includes(id)) {
      setAccentIdState(id);
    }
  };

  const setThemePreference = useCallback((pref) => {
    if (pref === 'light' || pref === 'dark' || pref === 'system') {
      setThemePreferenceState(pref);
    }
  }, []);

  const setDarkMode = useCallback((value) => {
    setThemePreferenceState(value ? 'dark' : 'light');
  }, []);

  const toggleDarkMode = useCallback(() => {
    setThemePreferenceState((prev) => {
      if (prev === 'system') {
        return prev;
      }
      return prev === 'dark' ? 'light' : 'dark';
    });
  }, []);

  const setAppearance = ({ mode, accentId: nextAccentId }) => {
    if (mode !== undefined) {
      setThemePreferenceState(!!mode ? 'dark' : 'light');
    }
    if (nextAccentId !== undefined && ACCENT_IDS.includes(nextAccentId)) {
      setAccentIdState(nextAccentId);
    }
  };

  const value = {
    darkMode,
    isDarkMode: darkMode,
    themePreference,
    systemPrefersDark,
    setThemePreference,
    accentId,
    toggleDarkMode,
    setDarkMode,
    setAccentId,
    setAppearance,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};
