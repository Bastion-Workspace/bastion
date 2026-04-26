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
import {
  addUserMessage,
  createConversation,
  getConversationMessages,
  listConversations,
  type ConversationMessage,
  type ConversationSummary,
} from '../../src/api/conversations';
import { streamOrchestrator } from '../../src/api/orchestratorStream';

type Row = { id: string; role: string; content: string };

export default function AiChatScreen() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [rows, setRows] = useState<Row[]>([]);
  const [input, setInput] = useState('');
  const [loadingList, setLoadingList] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const loadConversations = useCallback(async () => {
    const res = await listConversations(0, 30);
    setConversations(res.conversations ?? []);
  }, []);

  useEffect(() => {
    void (async () => {
      try {
        await loadConversations();
      } finally {
        setLoadingList(false);
      }
    })();
  }, [loadConversations]);

  const loadMessages = useCallback(async (cid: string) => {
    const res = await getConversationMessages(cid, 0, 100);
    const msgs = res.messages ?? [];
    setRows(
      msgs.map((m: ConversationMessage) => ({
        id: m.message_id,
        role: m.message_type,
        content: m.content,
      }))
    );
  }, []);

  useEffect(() => {
    if (conversationId) {
      void loadMessages(conversationId);
    } else {
      setRows([]);
    }
  }, [conversationId, loadMessages]);

  async function pickConversation(c: ConversationSummary) {
    setConversationId(c.conversation_id);
  }

  async function newChat() {
    const res = await createConversation({ title: 'Mobile chat', initial_message: null });
    const id = res.conversation?.conversation_id;
    if (id) {
      setConversationId(id);
      await loadConversations();
    }
  }

  async function onSend() {
    const q = input.trim();
    if (!q || streaming) return;
    setInput('');
    let cid = conversationId;
    if (!cid) {
      const res = await createConversation({ title: null, initial_message: q });
      cid = res.conversation?.conversation_id ?? null;
      if (!cid) return;
      setConversationId(cid);
      await loadConversations();
      await loadMessages(cid);
    } else {
      await addUserMessage(cid, q);
      setRows((prev) => [...prev, { id: `local-${Date.now()}`, role: 'user', content: q }]);
    }

    const assistantId = `asst-${Date.now()}`;
    setRows((prev) => [...prev, { id: assistantId, role: 'assistant', content: '' }]);
    setStreaming(true);
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    let acc = '';
    try {
      await streamOrchestrator({
        query: q,
        conversation_id: cid,
        session_id: 'bastion-mobile',
        signal: abortRef.current.signal,
        onChunk: (chunk) => {
          if (chunk.type === 'content' && typeof chunk.content === 'string') {
            acc += chunk.content;
            setRows((prev) =>
              prev.map((r) => (r.id === assistantId ? { ...r, content: acc } : r))
            );
          }
          if (chunk.type === 'complete' && typeof chunk.content === 'string') {
            acc += chunk.content;
            setRows((prev) =>
              prev.map((r) => (r.id === assistantId ? { ...r, content: acc } : r))
            );
          }
        },
      });
    } catch {
      setRows((prev) =>
        prev.map((r) =>
          r.id === assistantId ? { ...r, content: r.content || '(Stream failed)' } : r
        )
      );
    } finally {
      setStreaming(false);
      if (cid) void loadMessages(cid);
    }
  }

  if (loadingList) {
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
      <View style={styles.picker}>
        <Pressable style={styles.newBtn} onPress={() => void newChat()}>
          <Text style={styles.newBtnText}>New chat</Text>
        </Pressable>
        <FlatList
          horizontal
          data={conversations}
          keyExtractor={(c) => c.conversation_id}
          renderItem={({ item }) => (
            <Pressable
              style={[
                styles.convChip,
                item.conversation_id === conversationId && styles.convChipActive,
              ]}
              onPress={() => void pickConversation(item)}
            >
              <Text numberOfLines={1} style={styles.convChipText}>
                {item.title || 'Untitled'}
              </Text>
            </Pressable>
          )}
        />
      </View>
      <FlatList
        data={rows}
        keyExtractor={(r) => r.id}
        contentContainerStyle={styles.list}
        renderItem={({ item }) => (
          <View
            style={[
              styles.bubble,
              item.role === 'user' ? styles.bubbleUser : styles.bubbleAsst,
            ]}
          >
            <Text style={[styles.role, item.role === 'user' && styles.msgUser]}>{item.role}</Text>
            <Text style={[styles.msg, item.role === 'user' && styles.msgUser]}>{item.content}</Text>
          </View>
        )}
      />
      <View style={styles.composer}>
        <TextInput
          style={styles.input}
          placeholder="Ask Bastion…"
          value={input}
          onChangeText={setInput}
          multiline
        />
        <Pressable style={styles.sendBtn} onPress={() => void onSend()} disabled={streaming}>
          <Text style={styles.sendText}>{streaming ? '…' : 'Send'}</Text>
        </Pressable>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: '#f5f5f5' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  picker: { paddingVertical: 8, borderBottomWidth: 1, borderColor: '#ddd' },
  newBtn: {
    alignSelf: 'flex-start',
    marginHorizontal: 12,
    marginBottom: 8,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#1a1a2e',
    borderRadius: 6,
  },
  newBtnText: { color: '#fff', fontWeight: '600' },
  convChip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    marginLeft: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 16,
    maxWidth: 160,
  },
  convChipActive: { backgroundColor: '#cce' },
  convChipText: { fontSize: 13 },
  list: { padding: 12 },
  bubble: {
    padding: 10,
    borderRadius: 10,
    marginBottom: 10,
    maxWidth: '92%',
  },
  bubbleUser: { alignSelf: 'flex-end', backgroundColor: '#1a1a2e' },
  bubbleAsst: { alignSelf: 'flex-start', backgroundColor: '#fff', borderWidth: 1, borderColor: '#ddd' },
  role: { fontSize: 10, color: '#888', marginBottom: 4 },
  msg: { fontSize: 15, color: '#111' },
  msgUser: { color: '#fff' },
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
