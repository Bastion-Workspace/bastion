import { Redirect, Stack } from 'expo-router';
import { StyleSheet, View } from 'react-native';
import { AppLauncherSheet } from '../../src/components/AppLauncherSheet';
import { BottomDock } from '../../src/components/BottomDock';
import { FullPlayerModal } from '../../src/components/media/FullPlayerModal';
import { MiniPlayer } from '../../src/components/media/MiniPlayer';
import { AppLauncherProvider } from '../../src/context/AppLauncherContext';
import { useAuth } from '../../src/context/AuthContext';
import { MediaPlayerProvider, useMediaPlayer } from '../../src/context/MediaPlayerContext';
import { MINI_PLAYER_STRIP_HEIGHT } from '../../src/constants/dock';
import { VoiceModalProvider } from '../../src/voice/VoiceModalContext';
import { VoiceShortcutProvider } from '../../src/voice/VoiceShortcutContext';

function AppShellWithPlayer() {
  const { hasActiveSession } = useMediaPlayer();
  return (
    <View style={styles.root}>
      <View style={[styles.stackWrap, hasActiveSession ? { paddingBottom: MINI_PLAYER_STRIP_HEIGHT } : undefined]}>
        <Stack
          initialRouteName="chat"
          screenOptions={{
            headerShown: true,
            headerBackTitle: 'Back',
          }}
        >
          <Stack.Screen name="chat" options={{ title: 'Bastion Chat' }} />
          <Stack.Screen name="todos" options={{ title: 'ToDos' }} />
          <Stack.Screen name="documents" options={{ headerShown: false, title: 'Documents' }} />
          <Stack.Screen name="messages" options={{ headerShown: false, title: 'Messages' }} />
          <Stack.Screen name="rss" options={{ headerShown: false }} />
          <Stack.Screen name="ebooks" options={{ headerShown: false, title: 'eBooks' }} />
          <Stack.Screen name="media" options={{ headerShown: false, title: 'Media' }} />
          <Stack.Screen name="home" options={{ title: 'Settings' }} />
          <Stack.Screen name="shortcut-send" options={{ headerShown: false, title: 'Shortcut send' }} />
          <Stack.Screen name="voice" options={{ headerShown: false, title: 'Voice shortcut' }} />
        </Stack>
      </View>
      <MiniPlayer />
      <BottomDock />
      <AppLauncherSheet />
      <FullPlayerModal />
    </View>
  );
}

export default function AppLayout() {
  const { token, isReady, apiConfigured } = useAuth();

  if (!isReady) {
    return null;
  }
  if (!apiConfigured || !token) {
    return <Redirect href="/(auth)/login" />;
  }

  return (
    <MediaPlayerProvider>
      <VoiceShortcutProvider>
        <VoiceModalProvider>
          <AppLauncherProvider>
            <AppShellWithPlayer />
          </AppLauncherProvider>
        </VoiceModalProvider>
      </VoiceShortcutProvider>
    </MediaPlayerProvider>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  stackWrap: { flex: 1 },
});
