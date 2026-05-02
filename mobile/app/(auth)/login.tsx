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
import { Redirect, useRouter } from 'expo-router';
import { getApiBaseUrl } from '../../src/api/config';
import { useAppearancePreference } from '../../src/context/AppearancePreferenceContext';
import { useAuth } from '../../src/context/AuthContext';
import { loadLastAppRoute } from '../../src/session/lastAppRouteStore';
import { getColors } from '../../src/theme/colors';

export default function LoginScreen() {
  const { resolvedScheme } = useAppearancePreference();
  const c = getColors(resolvedScheme);
  const { token, login, apiConfigured, isReady } = useAuth();
  const router = useRouter();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [authedHref, setAuthedHref] = useState<string | null>(null);

  useEffect(() => {
    if (!isReady || !token) {
      return;
    }
    let cancelled = false;
    void (async () => {
      const href = await loadLastAppRoute();
      if (!cancelled) {
        setAuthedHref(href);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [isReady, token]);

  if (isReady && token) {
    if (authedHref === null) {
      return (
        <View style={[styles.centeredLoader, { backgroundColor: c.background }]}>
          <ActivityIndicator size="large" />
        </View>
      );
    }
    return <Redirect href={authedHref} />;
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
      const next = await loadLastAppRoute();
      router.replace(next);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Login failed';
      Alert.alert('Sign in failed', msg);
    } finally {
      setBusy(false);
    }
  }

  return (
    <KeyboardAvoidingView
      style={[styles.container, { backgroundColor: c.background }]}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <Text style={[styles.subtitle, { color: c.textSecondary }]}>Sign in to your account</Text>
      <Text style={[styles.serverLine, { color: c.textSecondary }]} numberOfLines={2}>
        {getApiBaseUrl()}
      </Text>
      <Pressable onPress={() => router.push('/(auth)/server')} style={styles.linkWrap}>
        <Text style={[styles.link, { color: c.link }]}>Change server</Text>
      </Pressable>
      <TextInput
        style={[styles.input, { backgroundColor: c.surface, borderColor: c.border, color: c.text }]}
        placeholder="Username"
        placeholderTextColor={c.textSecondary}
        autoCapitalize="none"
        autoCorrect={false}
        value={username}
        onChangeText={setUsername}
      />
      <TextInput
        style={[styles.input, { backgroundColor: c.surface, borderColor: c.border, color: c.text }]}
        placeholder="Password"
        placeholderTextColor={c.textSecondary}
        secureTextEntry
        value={password}
        onChangeText={setPassword}
      />
      <Pressable
        style={[styles.button, { backgroundColor: c.text }, busy && styles.buttonDisabled]}
        onPress={onSubmit}
        disabled={busy}
      >
        {busy ? <ActivityIndicator color={c.background} /> : <Text style={[styles.buttonText, { color: c.background }]}>Sign in</Text>}
      </Pressable>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', padding: 24 },
  subtitle: { fontSize: 16, fontWeight: '600', marginBottom: 12, textAlign: 'center' },
  serverLine: { textAlign: 'center', fontSize: 13, marginBottom: 4, paddingHorizontal: 8 },
  linkWrap: { alignSelf: 'center', marginBottom: 20 },
  link: { fontSize: 15, fontWeight: '600', textDecorationLine: 'underline' },
  input: {
    borderRadius: 8,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1,
    fontSize: 16,
  },
  button: {
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 8,
  },
  buttonDisabled: { opacity: 0.7 },
  buttonText: { fontSize: 16, fontWeight: '600' },
  centeredLoader: { flex: 1, justifyContent: 'center', alignItems: 'center' },
});
