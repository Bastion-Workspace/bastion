import { useEffect, useState } from 'react';
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
import { useRouter } from 'expo-router';
import { getApiBaseUrl } from '../../src/api/config';
import { useAuth } from '../../src/context/AuthContext';

export default function ServerScreen() {
  const { isReady, apiConfigured, setServerUrl } = useAuth();
  const router = useRouter();
  const [url, setUrl] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (isReady) {
      setUrl(getApiBaseUrl() || '');
    }
  }, [isReady]);

  async function onSave() {
    setBusy(true);
    try {
      await setServerUrl(url);
      router.replace('/(auth)/login');
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Could not save URL';
      Alert.alert('Invalid server URL', msg);
    } finally {
      setBusy(false);
    }
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <Text style={styles.title}>Bastion server</Text>
      <Text style={styles.sub}>
        Enter the same origin you use in the browser (scheme, host, and port if any). No path after the host — for
        example <Text style={styles.mono}>https://bastion.example.com</Text> or{' '}
        <Text style={styles.mono}>http://10.0.2.2:3051</Text> from the Android emulator to your machine.
      </Text>
      <TextInput
        style={styles.input}
        placeholder="https://your-bastion-host"
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
        value={url}
        onChangeText={setUrl}
        editable={!busy}
      />
      <Pressable style={[styles.button, busy && styles.buttonDisabled]} onPress={onSave} disabled={busy}>
        {busy ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>Save & continue</Text>}
      </Pressable>
      {apiConfigured ? (
        <Pressable style={styles.secondary} onPress={() => router.replace('/(auth)/login')} disabled={busy}>
          <Text style={styles.secondaryText}>Back to sign in</Text>
        </Pressable>
      ) : null}
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', padding: 24, backgroundColor: '#f5f5f5' },
  title: { fontSize: 24, fontWeight: '700', marginBottom: 12, textAlign: 'center' },
  sub: { fontSize: 14, color: '#444', marginBottom: 20, lineHeight: 20 },
  mono: { fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' }) },
  input: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 14,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#ddd',
    fontSize: 16,
  },
  button: {
    backgroundColor: '#1a1a2e',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonDisabled: { opacity: 0.7 },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  secondary: { marginTop: 16, padding: 12, alignItems: 'center' },
  secondaryText: { color: '#1a1a2e', fontSize: 15, fontWeight: '600' },
});
