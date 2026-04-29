import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { AuthProvider } from '../src/context/AuthContext';

dayjs.extend(relativeTime);

export default function RootLayout() {
  return (
    <AuthProvider>
      <SafeAreaProvider>
      <StatusBar style="auto" />
      <Stack screenOptions={{ headerBackTitle: 'Back' }}>
        <Stack.Screen name="index" options={{ headerShown: false }} />
        <Stack.Screen name="(auth)/login" options={{ title: 'Sign in' }} />
        <Stack.Screen name="(auth)/server" options={{ title: 'Server' }} />
        <Stack.Screen name="(app)" options={{ headerShown: false, title: 'Bastion' }} />
      </Stack>
      </SafeAreaProvider>
    </AuthProvider>
  );
}
