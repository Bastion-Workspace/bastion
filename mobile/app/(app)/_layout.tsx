import { Stack, Redirect } from 'expo-router';
import { useAuth } from '../../src/context/AuthContext';

export default function AppLayout() {
  const { token, isReady, apiConfigured } = useAuth();

  if (!isReady) {
    return null;
  }
  if (!apiConfigured || !token) {
    return <Redirect href="/(auth)/login" />;
  }

  return <Stack screenOptions={{ headerShown: true }} />;
}
