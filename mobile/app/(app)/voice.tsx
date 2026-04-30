import { useRouter } from 'expo-router';
import { useEffect, useRef } from 'react';
import { View } from 'react-native';
import { useVoiceShortcut } from '../../src/voice/VoiceShortcutContext';

/**
 * Deep link: bastion://voice
 * Opens the global voice capture modal (same flow as the dock mic button).
 */
export default function ExternalVoiceShortcutScreen() {
  const router = useRouter();
  const { requestVoiceOpen } = useVoiceShortcut();
  const ranRef = useRef(false);

  useEffect(() => {
    if (ranRef.current) return;
    ranRef.current = true;
    requestVoiceOpen();
    router.replace('/(app)/chat');
  }, [requestVoiceOpen, router]);

  return <View style={{ flex: 1, backgroundColor: '#f5f5f5' }} />;
}
