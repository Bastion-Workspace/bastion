import { Ionicons } from '@expo/vector-icons';
import { Audio } from 'expo-av';
import { useCallback, useEffect, useRef, useState, type ReactElement } from 'react';
import {
  ActivityIndicator,
  Modal,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { quickSendToDefaultAgent } from '../api/quickSend';
import { transcribeAudio } from '../api/stt';
import { useVoiceShortcut } from './VoiceShortcutContext';

type ExpoAvRecording = {
  stopAndUnloadAsync(): Promise<void>;
  getURI(): string | null;
};

type VoicePhase = 'idle' | 'preparing' | 'recording' | 'transcribing' | 'sending' | 'done' | 'error';

/**
 * Voice capture modal state and UI. Used only inside VoiceModalProvider.
 */
export function useVoiceModalController(): {
  openVoice: () => void;
  modalElement: ReactElement;
} {
  const [modalVisible, setModalVisible] = useState(false);
  const [phase, setPhase] = useState<VoicePhase>('idle');
  const [elapsedSec, setElapsedSec] = useState(0);
  const [errorMessage, setErrorMessage] = useState('');
  const recordingRef = useRef<ExpoAvRecording | null>(null);
  const doneTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const phaseRef = useRef<VoicePhase>('idle');
  const modalVisibleRef = useRef(false);
  const lastHandledVoiceRequestRef = useRef(0);
  const { voiceOpenRequestId } = useVoiceShortcut();
  const insets = useSafeAreaInsets();

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  useEffect(() => {
    modalVisibleRef.current = modalVisible;
  }, [modalVisible]);

  useEffect(() => {
    if (phase !== 'recording') {
      setElapsedSec(0);
      return;
    }
    setElapsedSec(0);
    const t = setInterval(() => setElapsedSec((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [phase]);

  useEffect(() => {
    return () => {
      if (doneTimerRef.current) clearTimeout(doneTimerRef.current);
      void (async () => {
        const rec = recordingRef.current;
        recordingRef.current = null;
        if (rec) {
          try {
            await rec.stopAndUnloadAsync();
          } catch {
            /* ignore */
          }
        }
      })();
    };
  }, []);

  const closeModal = useCallback(() => {
    if (doneTimerRef.current) {
      clearTimeout(doneTimerRef.current);
      doneTimerRef.current = null;
    }
    setModalVisible(false);
    setPhase('idle');
    setErrorMessage('');
    setElapsedSec(0);
  }, []);

  const startSession = useCallback(async () => {
    setErrorMessage('');
    setPhase('preparing');
    try {
      const perm = await Audio.requestPermissionsAsync();
      if (!perm.granted) {
        setErrorMessage('Microphone access is required to record a voice note.');
        setPhase('error');
        return;
      }
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });
      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      recordingRef.current = recording;
      setPhase('recording');
    } catch {
      setErrorMessage('Could not start microphone recording.');
      setPhase('error');
    }
  }, []);

  useEffect(() => {
    if (voiceOpenRequestId === 0 || voiceOpenRequestId === lastHandledVoiceRequestRef.current) {
      return;
    }
    lastHandledVoiceRequestRef.current = voiceOpenRequestId;
    if (modalVisibleRef.current) {
      return;
    }
    setModalVisible(true);
    void startSession();
  }, [voiceOpenRequestId, startSession]);

  const openVoice = useCallback(() => {
    setModalVisible(true);
    void startSession();
  }, [startSession]);

  const discardAndClose = useCallback(async () => {
    if (doneTimerRef.current) {
      clearTimeout(doneTimerRef.current);
      doneTimerRef.current = null;
    }
    const rec = recordingRef.current;
    recordingRef.current = null;
    if (rec) {
      try {
        await rec.stopAndUnloadAsync();
      } catch {
        /* ignore */
      }
    }
    closeModal();
  }, [closeModal]);

  const stopAndSubmit = useCallback(async () => {
    const rec = recordingRef.current;
    recordingRef.current = null;
    if (!rec) {
      closeModal();
      return;
    }

    setPhase('transcribing');

    try {
      await rec.stopAndUnloadAsync();
    } catch {
      setErrorMessage('Could not stop recording.');
      setPhase('error');
      return;
    }

    const uri = rec.getURI();
    if (!uri) {
      setErrorMessage('No audio file was produced.');
      setPhase('error');
      return;
    }

    let text: string;
    try {
      text = await transcribeAudio(uri);
    } catch (e) {
      setErrorMessage(e instanceof Error ? e.message : 'Transcription failed.');
      setPhase('error');
      return;
    }

    const trimmed = text.trim();
    if (!trimmed) {
      setErrorMessage('No speech was detected.');
      setPhase('error');
      return;
    }

    setPhase('sending');
    try {
      await quickSendToDefaultAgent(trimmed, {
        title: 'Voice Note',
        sessionId: 'bastion-mobile-voice',
      });
    } catch (e) {
      setErrorMessage(e instanceof Error ? e.message : 'Could not send to Bastion.');
      setPhase('error');
      return;
    }

    setPhase('done');
    doneTimerRef.current = setTimeout(() => {
      doneTimerRef.current = null;
      closeModal();
    }, 1500);
  }, [closeModal]);

  const onTryAgain = useCallback(() => {
    recordingRef.current = null;
    setErrorMessage('');
    void startSession();
  }, [startSession]);

  const modalElement = (
    <Modal
      visible={modalVisible}
      animationType="fade"
      transparent
      onRequestClose={() => {
        const p = phaseRef.current;
        if (p === 'preparing' || p === 'recording') {
          void discardAndClose();
        } else if (p === 'error' || p === 'done') {
          closeModal();
        }
      }}
    >
      <View
        style={[
          styles.modalRoot,
          { paddingTop: 24 + insets.top, paddingBottom: 24 + insets.bottom },
        ]}
        pointerEvents="box-none"
      >
        <Pressable
          style={styles.modalBackdropFill}
          onPress={() => {
            const p = phaseRef.current;
            if (p === 'done') closeModal();
            else if (p === 'preparing' || p === 'recording') void discardAndClose();
          }}
        />
        <View style={styles.modalCard}>
          <Text style={styles.modalTitle}>Voice input</Text>

          {phase === 'preparing' ? (
            <View style={styles.modalBody}>
              <ActivityIndicator size="large" color="#1a1a2e" />
              <Text style={styles.modalHint}>Starting microphone…</Text>
              <Pressable style={styles.secondaryBtn} onPress={() => void discardAndClose()}>
                <Text style={styles.secondaryBtnText}>Cancel</Text>
              </Pressable>
            </View>
          ) : null}

          {phase === 'recording' ? (
            <View style={styles.modalBody}>
              <View style={styles.recordingDot} />
              <Text style={styles.timerText}>
                {Math.floor(elapsedSec / 60)}:{String(elapsedSec % 60).padStart(2, '0')}
              </Text>
              <Pressable style={styles.stopBtn} onPress={() => void stopAndSubmit()}>
                <Text style={styles.stopBtnText}>Stop</Text>
              </Pressable>
            </View>
          ) : null}

          {phase === 'transcribing' ? (
            <View style={styles.modalBody}>
              <ActivityIndicator size="large" color="#1a1a2e" />
              <Text style={styles.modalHint}>Transcribing…</Text>
            </View>
          ) : null}

          {phase === 'sending' ? (
            <View style={styles.modalBody}>
              <ActivityIndicator size="large" color="#1a1a2e" />
              <Text style={styles.modalHint}>Sending…</Text>
            </View>
          ) : null}

          {phase === 'done' ? (
            <View style={styles.modalBody}>
              <Ionicons name="checkmark-circle" size={48} color="#2e7d32" />
              <Text style={styles.doneText}>Sent to Bastion</Text>
              <Text style={styles.modalSub}>Open Chat from the launcher to read the reply.</Text>
            </View>
          ) : null}

          {phase === 'error' ? (
            <View style={styles.modalBody}>
              <Text style={styles.errorText}>{errorMessage}</Text>
              <View style={styles.errorActions}>
                <Pressable style={styles.secondaryBtn} onPress={closeModal}>
                  <Text style={styles.secondaryBtnText}>Close</Text>
                </Pressable>
                <Pressable style={styles.primaryBtn} onPress={onTryAgain}>
                  <Text style={styles.primaryBtnText}>Try again</Text>
                </Pressable>
              </View>
            </View>
          ) : null}

          {phase === 'recording' ? (
            <Pressable style={styles.cancelLink} onPress={() => void discardAndClose()}>
              <Text style={styles.cancelLinkText}>Cancel</Text>
            </Pressable>
          ) : null}
        </View>
      </View>
    </Modal>
  );

  return { openVoice, modalElement };
}

const styles = StyleSheet.create({
  modalRoot: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 24,
  },
  modalBackdropFill: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.45)',
  },
  modalCard: {
    width: '100%',
    maxWidth: 360,
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    zIndex: 1,
    elevation: 4,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#111',
    marginBottom: 16,
    textAlign: 'center',
  },
  modalBody: {
    alignItems: 'center',
    gap: 16,
    paddingVertical: 8,
  },
  modalHint: { fontSize: 15, color: '#555' },
  modalSub: { fontSize: 13, color: '#888', textAlign: 'center' },
  recordingDot: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: '#c62828',
  },
  timerText: { fontSize: 28, fontWeight: '700', color: '#1a1a2e' },
  stopBtn: {
    marginTop: 8,
    backgroundColor: '#1a1a2e',
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 8,
  },
  stopBtnText: { color: '#fff', fontWeight: '700', fontSize: 16 },
  doneText: { fontSize: 17, fontWeight: '700', color: '#2e7d32' },
  errorText: { fontSize: 15, color: '#b71c1c', textAlign: 'center' },
  errorActions: { flexDirection: 'row', gap: 12, marginTop: 8 },
  secondaryBtn: {
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ccc',
  },
  secondaryBtnText: { fontSize: 15, fontWeight: '600', color: '#333' },
  primaryBtn: {
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    backgroundColor: '#1a1a2e',
  },
  primaryBtnText: { fontSize: 15, fontWeight: '600', color: '#fff' },
  cancelLink: { alignSelf: 'center', marginTop: 12, padding: 8 },
  cancelLinkText: { fontSize: 15, color: '#1a5090', fontWeight: '600' },
});
