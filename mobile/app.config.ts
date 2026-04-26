import type { ExpoConfig } from 'expo/config';

const config: ExpoConfig = {
  name: 'Bastion',
  slug: 'bastion-mobile',
  version: '0.1.0',
  orientation: 'portrait',
  scheme: 'bastion',
  userInterfaceStyle: 'automatic',
  newArchEnabled: false,
  ios: {
    supportsTablet: true,
    bundleIdentifier: 'com.bastion.mobile',
  },
  android: {
    adaptiveIcon: {
      backgroundColor: '#1a1a2e',
    },
    package: 'com.bastion.mobile',
  },
  plugins: ['expo-router'],
  experiments: {
    typedRoutes: true,
  },
  extra: {
    apiBaseUrl: process.env.EXPO_PUBLIC_API_BASE_URL ?? '',
  },
};

export default config;
