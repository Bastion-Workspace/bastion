import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { AuthProvider } from '../src/context/AuthContext';

export default function RootLayout() {
  return (
    <AuthProvider>
      <StatusBar style="auto" />
      <Stack screenOptions={{ headerBackTitle: 'Back' }}>
        <Stack.Screen name="index" options={{ headerShown: false }} />
        <Stack.Screen name="(auth)/login" options={{ title: 'Sign in' }} />
        <Stack.Screen name="(auth)/server" options={{ title: 'Server' }} />
        <Stack.Screen name="(app)" options={{ headerShown: false, title: 'Bastion' }} />
      </Stack>
    </AuthProvider>
  );
}
