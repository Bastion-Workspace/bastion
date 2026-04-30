import { Stack } from 'expo-router';

export default function RssLayout() {
  return (
    <Stack screenOptions={{ headerShown: true, headerBackTitle: 'Back' }}>
      <Stack.Screen name="index" options={{ title: 'RSS' }} />
    </Stack>
  );
}
