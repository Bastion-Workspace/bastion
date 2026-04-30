export type ColorSchemeName = 'light' | 'dark';

export type AppColors = {
  background: string;
  surface: string;
  surfaceMuted: string;
  text: string;
  textSecondary: string;
  border: string;
  link: string;
  danger: string;
  chipBg: string;
  chipBgActive: string;
  chipText: string;
  chipTextActive: string;
};

export function getColors(scheme: ColorSchemeName): AppColors {
  if (scheme === 'dark') {
    return {
      background: '#121212',
      surface: '#1e1e1e',
      surfaceMuted: '#2a2a2a',
      text: '#ececec',
      textSecondary: '#a0a0a0',
      border: '#3a3a3a',
      link: '#90caf9',
      danger: '#ef9a9a',
      chipBg: '#2a2a2a',
      chipBgActive: '#3949ab',
      chipText: '#ccc',
      chipTextActive: '#fff',
    };
  }
  return {
    background: '#fff',
    surface: '#f5f5fa',
    surfaceMuted: '#f0f0f8',
    text: '#1a1a2e',
    textSecondary: '#666',
    border: '#e0e0e0',
    link: '#1a5090',
    danger: '#c00',
    chipBg: '#eee',
    chipBgActive: '#1a1a2e',
    chipText: '#424242',
    chipTextActive: '#fff',
  };
}
