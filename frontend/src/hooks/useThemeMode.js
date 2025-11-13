import { useEffect, useState } from 'react';
import { useTheme } from '../contexts/ThemeContext';

export const useThemeMode = () => {
  const { darkMode, toggleDarkMode, setDarkMode } = useTheme();
  const [systemPrefersDark, setSystemPrefersDark] = useState(false);

  // Detect system preference
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    setSystemPrefersDark(mediaQuery.matches);

    const handleChange = (e) => {
      setSystemPrefersDark(e.matches);
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  // Sync with system preference
  const syncWithSystem = () => {
    setDarkMode(systemPrefersDark);
  };

  // Check if current theme matches system preference
  const isSystemTheme = darkMode === systemPrefersDark;

  return {
    darkMode,
    toggleDarkMode,
    setDarkMode,
    systemPrefersDark,
    syncWithSystem,
    isSystemTheme,
    themeMode: darkMode ? 'dark' : 'light'
  };
}; 