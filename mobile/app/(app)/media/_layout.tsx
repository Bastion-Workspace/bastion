import { Stack } from 'expo-router';

export default function MediaLayout() {
  return (
    <Stack
      screenOptions={{
        headerBackTitle: 'Back',
      }}
    >
      <Stack.Screen name="index" options={{ headerShown: true, title: 'Media' }} />
      <Stack.Screen name="[parentId]" options={{ headerShown: true }} />
      <Stack.Screen name="emby-library" options={{ headerShown: true }} />
      <Stack.Screen name="emby-series" options={{ headerShown: true }} />
      <Stack.Screen
        name="video"
        options={{ presentation: 'fullScreenModal', headerShown: false, animation: 'fade' }}
      />
      <Stack.Screen name="downloads" options={{ headerShown: true, title: 'Downloads' }} />
    </Stack>
  );
}
