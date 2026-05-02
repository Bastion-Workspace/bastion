import fs from 'fs';
import path from 'path';
import type { ExpoConfig } from 'expo/config';

/**
 * Launcher / adaptive icon source.
 *
 * Expo (via @expo/image-utils) only accepts **png | jpg | webp | gif** — not `.ico`, so we cannot
 * point `icon` at `favicon.ico` directly.
 *
 * Resolution order:
 * 1. `mobile/assets/app-icon.png` — dedicated mobile launcher art (square, ~1024×1024 recommended).
 * 2. `frontend/public/images/favicon.png` if present — same as web favicon export.
 * 3. `frontend/public/images/bastion-small.png` — default mark.
 *
 * Android adaptive icons: `foregroundImage` is drawn on top of `android.adaptiveIcon.backgroundColor`.
 * A 1024×1024 canvas is not enough if the PNG has transparency (logo smaller than the square, rounded
 * icon with clear corners, etc.) — those pixels show the background color as a ring or halo. Fix by
 * exporting an opaque square (full-bleed background) or set `backgroundColor` below to match the
 * transparent areas of your artwork.
 *
 * Splash (via `expo-splash-screen` plugin): same marks as web PWA — `bastion.png` (light) and
 * `bastion-dark.png` (dark), with backgrounds aligned to `frontend/index.html` theme-color values.
 */
const bastionSplashLight = '../frontend/public/images/bastion.png';
const bastionSplashDark = '../frontend/public/images/bastion-dark.png';
const splashBackgroundLight = '#f5f5f5';
const splashBackgroundDark = '#121212';
const faviconPngRel = '../frontend/public/images/favicon.png';
const bastionSmallRel = '../frontend/public/images/bastion-small.png';
const mobileAppIconRel = './assets/app-icon.png';

function resolveAppIcon(): string {
  const cwd = process.cwd();
  const mobileIconAbs = path.join(cwd, 'assets', 'app-icon.png');
  if (fs.existsSync(mobileIconAbs)) {
    return mobileAppIconRel;
  }
  const favAbs = path.join(cwd, '..', 'frontend', 'public', 'images', 'favicon.png');
  return fs.existsSync(favAbs) ? faviconPngRel : bastionSmallRel;
}

const appIcon = resolveAppIcon();

const config: ExpoConfig = {
  name: 'Bastion',
  slug: 'bastion-mobile',
  version: '0.70.14',
  orientation: 'portrait',
  scheme: 'bastion',
  userInterfaceStyle: 'automatic',
  newArchEnabled: false,
  icon: appIcon,
  ios: {
    supportsTablet: true,
    bundleIdentifier: 'com.bastion.mobile',
    infoPlist: {
      NSMicrophoneUsageDescription: 'Used to record voice instructions for Bastion.',
      UIBackgroundModes: ['audio'],
    },
    /**
     * Dark/light splash art for the system splash (separate from the expo-splash-screen plugin).
     * Without `ios.splash.dark` / `android.splash.dark`, SDK 52+ prebuild can omit night resources so
     * dark mode still shows the light splash image on some devices (see expo/expo#32860).
     */
    splash: {
      image: bastionSplashLight,
      resizeMode: 'contain',
      backgroundColor: splashBackgroundLight,
      dark: {
        image: bastionSplashDark,
        backgroundColor: splashBackgroundDark,
      },
    },
  },
  android: {
    adaptiveIcon: {
      foregroundImage: appIcon,
      backgroundColor: '#1a1a2e',
    },
    package: 'com.bastion.mobile',
    permissions: [
      'RECORD_AUDIO',
      'FOREGROUND_SERVICE',
      'FOREGROUND_SERVICE_MEDIA_PLAYBACK',
      'WAKE_LOCK',
    ],
    splash: {
      image: bastionSplashLight,
      resizeMode: 'contain',
      backgroundColor: splashBackgroundLight,
      dark: {
        image: bastionSplashDark,
        backgroundColor: splashBackgroundDark,
      },
    },
  },
  plugins: [
    'expo-router',
    [
      'expo-splash-screen',
      {
        image: bastionSplashLight,
        resizeMode: 'contain',
        backgroundColor: splashBackgroundLight,
        imageWidth: 200,
        dark: {
          image: bastionSplashDark,
          backgroundColor: splashBackgroundDark,
        },
      },
    ],
    'expo-notifications',
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
    eas: {
      projectId: '25772977-b277-4038-9d1b-ea3f55619fff',
    },
  },
};

export default config;
