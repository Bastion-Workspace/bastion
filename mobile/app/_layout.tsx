import { Stack, router } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { useEffect } from 'react';
import * as Notifications from 'expo-notifications';
import { AuthProvider } from '../src/context/AuthContext';
import { useAppearancePreference } from '../src/context/AppearancePreferenceContext';
import { RssPrefsProvider } from '../src/context/RssPrefsContext';
import { AppearanceThemeRoot } from '../src/components/AppearanceThemeRoot';
import { NotificationSurfaceBootstrap } from '../src/components/NotificationSurfaceBootstrap';
import TrackPlayer from 'react-native-track-player';

dayjs.extend(relativeTime);

const rntpGlobal = globalThis as { __bastionMediaPlaybackRegistered?: boolean };
if (!rntpGlobal.__bastionMediaPlaybackRegistered) {
  rntpGlobal.__bastionMediaPlaybackRegistered = true;
  TrackPlayer.registerPlaybackService(() => require('../src/media/playbackService').default);
}

function PushNavigationEffect() {
  useEffect(() => {
    const sub = Notifications.addNotificationResponseReceivedListener((response) => {
      const data = response.notification.request.content.data as Record<string, unknown>;
      const cid = data?.conversation_id;
      if (cid && typeof cid === 'string') {
        router.push({ pathname: '/(app)/chat', params: { conversationId: cid } });
      }
    });
    return () => sub.remove();
  }, []);
  return null;
}

function RootStatusBar() {
  const { resolvedScheme } = useAppearancePreference();
  const isDark = resolvedScheme === 'dark';
  return <StatusBar style={isDark ? 'light' : 'dark'} />;
}

export default function RootLayout() {
  return (
    <AuthProvider>
      <AppearanceThemeRoot>
        <RssPrefsProvider>
          <NotificationSurfaceBootstrap />
          <PushNavigationEffect />
          <SafeAreaProvider>
            <RootStatusBar />
            <Stack screenOptions={{ headerBackTitle: 'Back' }}>
              <Stack.Screen name="index" options={{ headerShown: false }} />
              <Stack.Screen name="(auth)" options={{ headerShown: false }} />
              <Stack.Screen name="(app)" options={{ headerShown: false, title: 'Bastion' }} />
            </Stack>
          </SafeAreaProvider>
        </RssPrefsProvider>
      </AppearanceThemeRoot>
    </AuthProvider>
  );
}
