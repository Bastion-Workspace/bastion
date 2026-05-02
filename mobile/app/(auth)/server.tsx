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
import { useAppearancePreference } from '../../src/context/AppearancePreferenceContext';
import { useAuth } from '../../src/context/AuthContext';
import { getColors } from '../../src/theme/colors';

export default function ServerScreen() {
  const { resolvedScheme } = useAppearancePreference();
  const c = getColors(resolvedScheme);
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
      style={[styles.container, { backgroundColor: c.background }]}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <Text style={[styles.title, { color: c.text }]}>Server URL</Text>
      <Text style={[styles.sub, { color: c.textSecondary }]}>
        Enter the same origin you use in the browser (scheme, host, and port if any). No path after the host — for
        example <Text style={styles.mono}>https://bastion.example.com</Text> or{' '}
        <Text style={styles.mono}>http://10.0.2.2:3051</Text> from the Android emulator to your machine.
      </Text>
      <TextInput
        style={[styles.input, { backgroundColor: c.surface, borderColor: c.border, color: c.text }]}
        placeholder="https://your-bastion-host"
        placeholderTextColor={c.textSecondary}
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
        value={url}
        onChangeText={setUrl}
        editable={!busy}
      />
      <Pressable
        style={[styles.button, { backgroundColor: c.text }, busy && styles.buttonDisabled]}
        onPress={onSave}
        disabled={busy}
      >
        {busy ? <ActivityIndicator color={c.background} /> : <Text style={[styles.buttonText, { color: c.background }]}>Save & continue</Text>}
      </Pressable>
      {apiConfigured ? (
        <Pressable style={styles.secondary} onPress={() => router.replace('/(auth)/login')} disabled={busy}>
          <Text style={[styles.secondaryText, { color: c.link }]}>Back to sign in</Text>
        </Pressable>
      ) : null}
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', padding: 24 },
  title: { fontSize: 22, fontWeight: '700', marginBottom: 12, textAlign: 'center' },
  sub: { fontSize: 14, marginBottom: 20, lineHeight: 20 },
  mono: { fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' }) },
  input: {
    borderRadius: 8,
    padding: 14,
    marginBottom: 16,
    borderWidth: 1,
    fontSize: 16,
  },
  button: {
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonDisabled: { opacity: 0.7 },
  buttonText: { fontSize: 16, fontWeight: '600' },
  secondary: { marginTop: 16, padding: 12, alignItems: 'center' },
  secondaryText: { fontSize: 15, fontWeight: '600' },
});
