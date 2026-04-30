import { useRouter, useLocalSearchParams } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';
import { quickSendToDefaultAgent, QUICK_SEND_MAX_MESSAGE_CHARS } from '../../src/api/quickSend';

type SendStatus = 'sending' | 'done' | 'error';

function normalizeMessageParam(raw: string | string[] | undefined): string {
  if (raw === undefined) return '';
  const s = typeof raw === 'string' ? raw : raw[0] ?? '';
  if (!s) return '';
  try {
    return decodeURIComponent(s.replace(/\+/g, ' '));
  } catch {
    return s;
  }
}

/**
 * Deep link: bastion://shortcut-send?m=<url-encoded text>
 * Fire-and-forget to the default chat model; then navigate to Chat.
 */
export default function ShortcutSendScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ m?: string | string[] }>();
  const [status, setStatus] = useState<SendStatus>('sending');
  const [errorMessage, setErrorMessage] = useState('');
  const ranRef = useRef(false);

  useEffect(() => {
    if (ranRef.current) return;
    ranRef.current = true;

    const text = normalizeMessageParam(params.m).trim();
    if (!text) {
      setErrorMessage(
        'Missing message. Use bastion://shortcut-send?m=' + encodeURIComponent('your text here')
      );
      setStatus('error');
      return;
    }
    if (text.length > QUICK_SEND_MAX_MESSAGE_CHARS) {
      setErrorMessage(`Message too long (max ${QUICK_SEND_MAX_MESSAGE_CHARS} characters).`);
      setStatus('error');
      return;
    }

    void (async () => {
      try {
        await quickSendToDefaultAgent(text, {
          title: 'Shortcut',
          sessionId: 'bastion-mobile-shortcut',
        });
        setStatus('done');
        setTimeout(() => {
          router.replace('/(app)/chat');
        }, 1200);
      } catch (e) {
        setErrorMessage(e instanceof Error ? e.message : 'Send failed');
        setStatus('error');
      }
    })();
  }, [params.m, router]);

  return (
    <View style={styles.container}>
      {status === 'sending' ? (
        <>
          <ActivityIndicator size="large" color="#1a1a2e" />
          <Text style={styles.hint}>Sending to Bastion…</Text>
        </>
      ) : null}
      {status === 'done' ? (
        <>
          <Text style={styles.done}>Sent to Bastion</Text>
          <Text style={styles.sub}>Opening Chat…</Text>
        </>
      ) : null}
      {status === 'error' ? (
        <>
          <Text style={styles.error}>{errorMessage}</Text>
          <Pressable style={styles.btn} onPress={() => router.replace('/(app)/chat')}>
            <Text style={styles.btnText}>Close</Text>
          </Pressable>
        </>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
    backgroundColor: '#f5f5f5',
    gap: 16,
  },
  hint: { fontSize: 16, color: '#444' },
  done: { fontSize: 18, fontWeight: '700', color: '#2e7d32' },
  sub: { fontSize: 14, color: '#666' },
  error: { fontSize: 15, color: '#b71c1c', textAlign: 'center' },
  btn: {
    marginTop: 8,
    paddingVertical: 12,
    paddingHorizontal: 24,
    backgroundColor: '#1a1a2e',
    borderRadius: 8,
  },
  btnText: { color: '#fff', fontWeight: '600', fontSize: 16 },
});
