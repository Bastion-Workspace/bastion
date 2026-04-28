import { Stack } from 'expo-router';

export default function MessagesLayout() {
  return (
    <Stack screenOptions={{ headerShown: false }}>
      <Stack.Screen name="index" />
      <Stack.Screen name="[roomId]" options={{ headerShown: true, title: 'Room' }} />
    </Stack>
  );
}
