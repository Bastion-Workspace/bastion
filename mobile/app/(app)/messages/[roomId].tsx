import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Image,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  useWindowDimensions,
  View,
} from 'react-native';
import dayjs from 'dayjs';
import Markdown from 'react-native-markdown-display';
import { useLocalSearchParams } from 'expo-router';
import { absolutizeMessageMediaRefs, resolveAbsoluteApiUrl } from '../../../src/api/config';
import {
  getRoomMessages,
  openRoomWebSocket,
  sendRoomMessage,
  sendTyping,
  type MessagingMessage,
} from '../../../src/api/messaging';
import { getStoredToken } from '../../../src/session/tokenStore';
import { useAuth } from '../../../src/context/AuthContext';

export default function RoomChatScreen() {
  const { roomId } = useLocalSearchParams<{ roomId: string }>();
  const { user } = useAuth();
  const { width: windowWidth } = useWindowDimensions();
  const [messages, setMessages] = useState<MessagingMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [imageAuthHeaders, setImageAuthHeaders] = useState<Record<string, string>>({});
  const wsRef = useRef<WebSocket | null>(null);
  const listRef = useRef<FlatList<MessagingMessage>>(null);
  const hasScrolledInitial = useRef(false);

  useEffect(() => {
    hasScrolledInitial.current = false;
  }, [roomId]);

  const myUserId = user?.user_id != null ? String(user.user_id) : '';

  const imageMaxWidth = Math.min(windowWidth * 0.82, 360);

  useEffect(() => {
    void (async () => {
      const t = await getStoredToken();
      setImageAuthHeaders(t ? { Authorization: `Bearer ${t}` } : {});
    })();
  }, []);

  const markdownRules = useMemo(
    () => ({
      image: (node: { key: string; attributes: { src?: string; alt?: string } }, _children: unknown, _parent: unknown, mdStyles: { image?: object }) => {
        const raw = node.attributes?.src ?? '';
        const uri = resolveAbsoluteApiUrl(raw);
        if (!uri) return null;
        return (
          <Image
            key={node.key}
            accessibilityLabel={node.attributes?.alt || 'Attachment'}
            source={Object.keys(imageAuthHeaders).length ? { uri, headers: imageAuthHeaders } : { uri }}
            style={[mdStyles.image, styles.inlineImage, { maxWidth: imageMaxWidth }]}
            resizeMode="contain"
          />
        );
      },
    }),
    [imageAuthHeaders, imageMaxWidth]
  );

  const mdStylesOther = useMemo(
    () => ({
      body: { color: '#111', fontSize: 15, lineHeight: 22 },
      paragraph: { marginTop: 0, marginBottom: 4 },
      bullet_list: { marginBottom: 4 },
      ordered_list: { marginBottom: 4 },
      link: { color: '#1565c0' },
      image: { marginVertical: 6, borderRadius: 8, width: imageMaxWidth, minHeight: 80 },
      code_inline: {
        backgroundColor: '#f0f0f0',
        fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' }),
      },
    }),
    [imageMaxWidth]
  );

  const mdStylesMe = useMemo(
    () => ({
      ...mdStylesOther,
      body: { ...mdStylesOther.body, color: '#fff' },
      link: { color: '#bbdefb' },
      code_inline: { ...mdStylesOther.code_inline, backgroundColor: 'rgba(255,255,255,0.15)', color: '#fff' },
    }),
    [mdStylesOther]
  );

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

  useEffect(() => {
    if (loading || messages.length === 0) return;
    if (hasScrolledInitial.current) return;
    hasScrolledInitial.current = true;
    const id = requestAnimationFrame(() => {
      listRef.current?.scrollToEnd({ animated: false });
    });
    return () => cancelAnimationFrame(id);
  }, [loading, messages.length]);

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

  function formatMessageTime(iso: string): string {
    const d = dayjs(iso);
    if (!d.isValid()) return '';
    return d.format('MMM D, YYYY · h:mm A');
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
        contentContainerStyle={[
          styles.list,
          messages.length > 0 ? styles.listGrowEnd : undefined,
        ]}
        onContentSizeChange={() => {
          listRef.current?.scrollToEnd({ animated: true });
        }}
        renderItem={({ item }) => {
          const isMe =
            myUserId.length > 0 && item.user_id != null && String(item.user_id) === myUserId;
          const body = absolutizeMessageMediaRefs(item.content ?? '');
          return (
            <View
              style={[
                styles.bubble,
                isMe ? styles.bubbleMe : styles.bubbleOther,
              ]}
            >
              <Markdown style={isMe ? mdStylesMe : mdStylesOther} rules={markdownRules}>
                {body.trim() ? body : '\u00a0'}
              </Markdown>
              <Text style={[styles.time, isMe && styles.timeMe]}>{formatMessageTime(item.created_at)}</Text>
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
  listGrowEnd: { flexGrow: 1, justifyContent: 'flex-end' },
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
  inlineImage: {
    alignSelf: 'flex-start',
  },
  time: { fontSize: 11, color: '#666', marginTop: 6 },
  timeMe: { color: '#bbb' },
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
