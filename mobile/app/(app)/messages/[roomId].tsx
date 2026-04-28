import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import dayjs from 'dayjs';
import { useLocalSearchParams } from 'expo-router';
import {
  getRoomMessages,
  openRoomWebSocket,
  sendRoomMessage,
  sendTyping,
  type MessagingMessage,
} from '../../../src/api/messaging';
import { useAuth } from '../../../src/context/AuthContext';

export default function RoomChatScreen() {
  const { roomId } = useLocalSearchParams<{ roomId: string }>();
  const { user } = useAuth();
  const [messages, setMessages] = useState<MessagingMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const listRef = useRef<FlatList<MessagingMessage>>(null);

  const myUserId = user?.user_id != null ? String(user.user_id) : '';

  const load = useCallback(async () => {
    if (!roomId) return;
    const res = await getRoomMessages(roomId, 80);
    const list = res.messages ?? [];
    setMessages(list);
  }, [roomId]);

  useEffect(() => {
    if (!roomId) return;
    let ws: WebSocket | null = null;
    void (async () => {
      try {
        await load();
        ws = await openRoomWebSocket(roomId, {
          onMessage: (msg) => {
            setMessages((prev) => {
              if (prev.some((m) => m.message_id === msg.message_id)) return prev;
              return [...prev, msg];
            });
          },
        });
        wsRef.current = ws;
      } finally {
        setLoading(false);
      }
    })();
    return () => {
      ws?.close();
      wsRef.current = null;
    };
  }, [roomId, load]);

  async function onSend() {
    const text = input.trim();
    if (!text || !roomId || sending) return;
    setSending(true);
    setInput('');
    try {
      const msg = await sendRoomMessage(roomId, text);
      setMessages((prev) => {
        if (prev.some((m) => m.message_id === msg.message_id)) return prev;
        return [...prev, msg];
      });
    } catch {
      setInput(text);
    } finally {
      setSending(false);
    }
  }

  if (loading && messages.length === 0) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={80}
    >
      <FlatList
        ref={listRef}
        data={messages}
        keyExtractor={(item) => item.message_id}
        contentContainerStyle={styles.list}
        onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
        renderItem={({ item }) => {
          const isMe =
            myUserId.length > 0 && item.user_id != null && String(item.user_id) === myUserId;
          return (
            <View
              style={[
                styles.bubble,
                isMe ? styles.bubbleMe : styles.bubbleOther,
              ]}
            >
              <Text style={[styles.msg, isMe && styles.msgMe]}>{item.content}</Text>
              <Text style={[styles.time, isMe && styles.timeMe]}>
                {dayjs(item.created_at).format('h:mm A')}
              </Text>
            </View>
          );
        }}
      />
      <View style={styles.composer}>
        <TextInput
          style={styles.input}
          placeholder="Message"
          value={input}
          onChangeText={(t) => {
            setInput(t);
            if (wsRef.current) sendTyping(wsRef.current, t.length > 0);
          }}
          multiline
        />
        <Pressable style={styles.sendBtn} onPress={() => void onSend()} disabled={sending}>
          <Text style={styles.sendText}>{sending ? '…' : 'Send'}</Text>
        </Pressable>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: '#f0f0f0' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  list: { padding: 12, paddingBottom: 8 },
  bubble: {
    maxWidth: '90%',
    padding: 10,
    borderRadius: 10,
    marginBottom: 8,
  },
  bubbleOther: {
    alignSelf: 'flex-start',
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  bubbleMe: {
    alignSelf: 'flex-end',
    backgroundColor: '#1a1a2e',
  },
  msg: { fontSize: 15, color: '#111' },
  msgMe: { color: '#fff' },
  time: { fontSize: 10, color: '#888', marginTop: 4 },
  timeMe: { color: '#ccc' },
  composer: { flexDirection: 'row', padding: 8, gap: 8, borderTopWidth: 1, borderColor: '#ddd' },
  input: {
    flex: 1,
    minHeight: 40,
    maxHeight: 120,
    backgroundColor: '#fff',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderWidth: 1,
    borderColor: '#ccc',
  },
  sendBtn: { justifyContent: 'center', paddingHorizontal: 16, backgroundColor: '#1a1a2e', borderRadius: 8 },
  sendText: { color: '#fff', fontWeight: '600' },
});
