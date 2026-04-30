import { DarkTheme, DefaultTheme, ThemeProvider } from '@react-navigation/native';
import type { ReactNode } from 'react';
import { AppearancePreferenceProvider, useAppearancePreference } from '../context/AppearancePreferenceContext';

type Props = { children: ReactNode };

function NavigationThemeBridge({ children }: Props) {
  const { resolvedScheme } = useAppearancePreference();
  const theme = resolvedScheme === 'dark' ? DarkTheme : DefaultTheme;
  return <ThemeProvider value={theme}>{children}</ThemeProvider>;
}

export function AppearanceThemeRoot({ children }: Props) {
  return (
    <AppearancePreferenceProvider>
      <NavigationThemeBridge>{children}</NavigationThemeBridge>
    </AppearancePreferenceProvider>
  );
}
