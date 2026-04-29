import fs from 'fs';
import path from 'path';
import type { ExpoConfig } from 'expo/config';

/**
 * Launcher / adaptive icon source.
 *
 * Expo (via @expo/image-utils) only accepts **png | jpg | webp | gif** — not `.ico`, so we cannot
 * point `icon` at `favicon.ico` directly.
 *
 * To use the **same artwork as the site favicon**, add a PNG next to the ICO, e.g. export
 * `frontend/public/images/favicon.png` (1024×1024 square is ideal). If that file exists, we use it;
 * otherwise we fall back to the Bastion mark PNG.
 */
const bastionSplash = '../frontend/public/images/bastion.png';
const faviconPngRel = '../frontend/public/images/favicon.png';
const bastionSmallRel = '../frontend/public/images/bastion-small.png';

function resolveAppIcon(): string {
  const cwd = process.cwd();
  const favAbs = path.join(cwd, '..', 'frontend', 'public', 'images', 'favicon.png');
  return fs.existsSync(favAbs) ? faviconPngRel : bastionSmallRel;
}

const appIcon = resolveAppIcon();

const config: ExpoConfig = {
  name: 'Bastion',
  slug: 'bastion-mobile',
  version: '0.70.8',
  orientation: 'portrait',
  scheme: 'bastion',
  userInterfaceStyle: 'automatic',
  newArchEnabled: false,
  icon: appIcon,
  splash: {
    image: bastionSplash,
    resizeMode: 'contain',
    backgroundColor: '#1a1a2e',
  },
  ios: {
    supportsTablet: true,
    bundleIdentifier: 'com.bastion.mobile',
    infoPlist: {
      NSMicrophoneUsageDescription: 'Used to record voice instructions for Bastion.',
    },
  },
  android: {
    adaptiveIcon: {
      foregroundImage: appIcon,
      backgroundColor: '#1a1a2e',
    },
    package: 'com.bastion.mobile',
    permissions: ['RECORD_AUDIO'],
  },
  plugins: [
    'expo-router',
    './plugins/withAndroidEmbeddedDebugBundle',
    [
      'expo-build-properties',
      {
        android: {
          ndkVersion: '27.1.12297006',
        },
      },
    ],
  ],
  experiments: {
    typedRoutes: true,
  },
  extra: {
    apiBaseUrl: process.env.EXPO_PUBLIC_API_BASE_URL ?? '',
  },
};

export default config;
