import { Redirect } from 'expo-router';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { useAuth } from '../src/context/AuthContext';

export default function Index() {
  const { token, isReady, apiConfigured } = useAuth();

  if (!isReady) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  if (!apiConfigured) {
    return (
      <View style={styles.center}>
        <Text style={styles.warn}>Set EXPO_PUBLIC_API_BASE_URL to your Bastion server.</Text>
      </View>
    );
  }

  if (!token) {
    return <Redirect href="/(auth)/login" />;
  }
  return <Redirect href="/home" />;
}

const styles = StyleSheet.create({
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
  warn: { textAlign: 'center', fontSize: 16 },
});
