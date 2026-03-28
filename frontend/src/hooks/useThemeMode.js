import { useTheme } from '../contexts/ThemeContext';

/**
 * Theme helpers; preference and system state live in ThemeContext.
 */
export const useThemeMode = () => {
  const {
    darkMode,
    toggleDarkMode,
    setDarkMode,
    themePreference,
    setThemePreference,
    systemPrefersDark,
  } = useTheme();

  const syncWithSystem = () => {
    setThemePreference('system');
  };

  const isSystemTheme = themePreference === 'system';

  return {
    darkMode,
    toggleDarkMode,
    setDarkMode,
    themePreference,
    setThemePreference,
    systemPrefersDark,
    syncWithSystem,
    isSystemTheme,
    themeMode: darkMode ? 'dark' : 'light',
  };
};
