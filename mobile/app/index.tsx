import { Redirect } from 'expo-router';
import { ActivityIndicator, StyleSheet, View } from 'react-native';
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
    return <Redirect href="/(auth)/server" />;
  }

  if (!token) {
    return <Redirect href="/(auth)/login" />;
  }
  return <Redirect href="/home" />;
}

const styles = StyleSheet.create({
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
});
