import { useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { Redirect, useRouter } from 'expo-router';
import { getApiBaseUrl } from '../../src/api/config';
import { useAuth } from '../../src/context/AuthContext';

export default function LoginScreen() {
  const { token, login, apiConfigured, isReady } = useAuth();
  const router = useRouter();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);

  if (isReady && token) {
    return <Redirect href="/home" />;
  }

  if (isReady && !apiConfigured) {
    return <Redirect href="/(auth)/server" />;
  }

  async function onSubmit() {
    if (!username.trim() || !password) {
      Alert.alert('Missing fields', 'Enter username and password.');
      return;
    }
    setBusy(true);
    try {
      await login(username.trim(), password);
      router.replace('/home');
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Login failed';
      Alert.alert('Sign in failed', msg);
    } finally {
      setBusy(false);
    }
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <Text style={styles.title}>Bastion</Text>
      <Text style={styles.serverLine} numberOfLines={2}>
        {getApiBaseUrl()}
      </Text>
      <Pressable onPress={() => router.push('/(auth)/server')} style={styles.linkWrap}>
        <Text style={styles.link}>Change server</Text>
      </Pressable>
      <TextInput
        style={styles.input}
        placeholder="Username"
        autoCapitalize="none"
        autoCorrect={false}
        value={username}
        onChangeText={setUsername}
      />
      <TextInput
        style={styles.input}
        placeholder="Password"
        secureTextEntry
        value={password}
        onChangeText={setPassword}
      />
      <Pressable style={[styles.button, busy && styles.buttonDisabled]} onPress={onSubmit} disabled={busy}>
        {busy ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>Sign in</Text>}
      </Pressable>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', padding: 24, backgroundColor: '#f5f5f5' },
  title: { fontSize: 28, fontWeight: '700', marginBottom: 8, textAlign: 'center' },
  serverLine: { textAlign: 'center', fontSize: 13, color: '#555', marginBottom: 4, paddingHorizontal: 8 },
  linkWrap: { alignSelf: 'center', marginBottom: 20 },
  link: { fontSize: 15, color: '#1a1a2e', fontWeight: '600', textDecorationLine: 'underline' },
  input: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#ddd',
    fontSize: 16,
  },
  button: {
    backgroundColor: '#1a1a2e',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 8,
  },
  buttonDisabled: { opacity: 0.7 },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
