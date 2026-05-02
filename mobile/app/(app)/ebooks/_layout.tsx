import { Stack } from 'expo-router';

export default function EbooksLayout() {
  return (
    <Stack
      screenOptions={{
        headerShown: true,
        headerBackTitle: 'Back',
      }}
    >
      <Stack.Screen name="index" options={{ title: 'eBooks' }} />
      <Stack.Screen name="settings" options={{ title: 'eBooks & OPDS' }} />
      <Stack.Screen
        name="reader"
        options={{ headerShown: false, title: 'Reader', gestureEnabled: false }}
      />
    </Stack>
  );
}
