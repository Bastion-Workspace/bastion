import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import Markdown from 'react-native-markdown-display';
import { useLocalSearchParams, useRouter } from 'expo-router';
import {
  createConversation,
  getConversationMessages,
  listConversations,
  type ConversationMessage,
  type ConversationSummary,
} from '../../src/api/conversations';
import { setActiveConversationForNotifications } from '../../src/session/activeConversationRef';
import { getEnabledModels, getModelRoles, setUserChatModelRole, type EnabledModel } from '../../src/api/models';
import { streamOrchestrator } from '../../src/api/orchestratorStream';
import { useModalSheetBottomPadding } from '../../src/components/ScreenShell';

dayjs.extend(relativeTime);

type Row = { id: string; role: string; content: string; statusText?: string };

const DOC_SNIPPET_MAX = 12_000;

const assistantMarkdownStyles = {
  body: { color: '#111', fontSize: 15, lineHeight: 22 },
  paragraph: { marginTop: 0, marginBottom: 8 },
  bullet_list: { marginBottom: 8 },
  ordered_list: { marginBottom: 8 },
  code_inline: {
    backgroundColor: '#eee',
    fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' }),
  },
  code_block: {
    backgroundColor: '#f0f0f0',
    padding: 8,
    fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' }),
    fontSize: 13,
  },
  fence: {
    backgroundColor: '#f0f0f0',
    padding: 8,
    fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' }),
    fontSize: 13,
  },
  link: { color: '#1a5090' },
  heading1: { fontSize: 20, fontWeight: '700', marginBottom: 8 },
  heading2: { fontSize: 18, fontWeight: '700', marginBottom: 6 },
  heading3: { fontSize: 16, fontWeight: '700', marginBottom: 4 },
};

function shortModelLabel(modelId: string): string {
  if (!modelId) return 'Default';
  const slash = modelId.lastIndexOf('/');
  return slash >= 0 ? modelId.slice(slash + 1) : modelId.slice(-24);
}

function buildActiveEditor(
  documentId: string,
  title: string,
  snippet: string,
  frontmatter: Record<string, unknown> = {}
): Record<string, unknown> {
  const filename = title.match(/\.(md|org|txt|pdf|docx|srt|vtt)$/i) ? title : `${title}.md`;
  return {
    is_editable: false,
    filename,
    language: 'markdown',
    content: snippet,
    content_length: snippet.length,
    document_id: documentId,
    cursor_offset: -1,
    selection_start: -1,
    selection_end: -1,
    frontmatter: { document_id: documentId, source: 'bastion-mobile', ...frontmatter },
  };
}

export default function BastionChatScreen() {
  const navigation = useNavigation();
  const router = useRouter();
  const params = useLocalSearchParams<{
    docId?: string;
    docTitle?: string;
    docSnippet?: string;
    /** Present when opening from document FAB so each open gets a fresh conversation. */
    docSession?: string;
    /** JSON-serialized Record from document YAML frontmatter (FAB flow). */
    docFrontmatter?: string;
    conversationId?: string;
  }>();

  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [rows, setRows] = useState<Row[]>([]);
  const [input, setInput] = useState('');
  const [loadingList, setLoadingList] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const listRef = useRef<FlatList<Row>>(null);

  const [historyOpen, setHistoryOpen] = useState(false);
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  const [enabledModels, setEnabledModels] = useState<EnabledModel[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const modalSheetBottomPad = useModalSheetBottomPadding(16);

  const [docContext, setDocContext] = useState<{
    documentId: string;
    title: string;
    snippet: string;
    frontmatter: Record<string, unknown>;
  } | null>(null);
  const docBootstrapRef = useRef<string | null>(null);

  const loadConversations = useCallback(async () => {
    const res = await listConversations(0, 30);
    setConversations(res.conversations ?? []);
  }, []);

  useEffect(() => {
    void (async () => {
      try {
        await loadConversations();
        const [roles, models] = await Promise.all([getModelRoles(), getEnabledModels()]);
        setEnabledModels(models);
        const role = (roles.user_chat_model ?? '').trim();
        if (role) {
          setSelectedModelId(role);
        } else if (models.length > 0) {
          setSelectedModelId(models[0].model_id);
        }
      } finally {
        setLoadingList(false);
      }
    })();
  }, [loadConversations]);

  useEffect(() => {
    setActiveConversationForNotifications(conversationId);
    return () => setActiveConversationForNotifications(null);
  }, [conversationId]);

  useLayoutEffect(() => {
    navigation.setOptions({
      headerRight: () => (
        <Pressable
          onPress={() => setModelPickerOpen(true)}
          style={styles.headerModelBtn}
          hitSlop={10}
          accessibilityRole="button"
          accessibilityLabel="Chat model"
          accessibilityHint="Opens the list of models for this chat"
        >
          <Text style={styles.headerModelLabel} numberOfLines={1}>
            {shortModelLabel(selectedModelId)}
          </Text>
          <Ionicons name="chevron-down" size={16} color="#1a5090" />
        </Pressable>
      ),
    });
  }, [navigation, selectedModelId]);

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
    if (!conversationId) setRows([]);
  }, [conversationId]);

  useEffect(() => {
    const fromRoute =
      typeof params.conversationId === 'string' ? params.conversationId.trim() : '';
    if (fromRoute) {
      setConversationId(fromRoute);
      void loadMessages(fromRoute);
    }
  }, [params.conversationId, loadMessages]);

  const recentConversations = useMemo(() => {
    const list = [...conversations];
    list.sort((a, b) => {
      const ta = a.last_message_at ? new Date(a.last_message_at).getTime() : 0;
      const tb = b.last_message_at ? new Date(b.last_message_at).getTime() : 0;
      return tb - ta;
    });
    return list.slice(0, 10);
  }, [conversations]);

  useEffect(() => {
    const docId = typeof params.docId === 'string' ? params.docId.trim() : '';
    const docTitle = typeof params.docTitle === 'string' ? params.docTitle.trim() : '';
    const docSession = typeof params.docSession === 'string' ? params.docSession.trim() : '';
    const rawSnippet = typeof params.docSnippet === 'string' ? params.docSnippet : '';
    if (!docId || !docTitle || !docSession) return;

    const fmRaw = typeof params.docFrontmatter === 'string' ? params.docFrontmatter.trim() : '';
    let parsedFm: Record<string, unknown> = {};
    if (fmRaw) {
      try {
        const o = JSON.parse(fmRaw) as unknown;
        if (o && typeof o === 'object' && !Array.isArray(o)) {
          parsedFm = o as Record<string, unknown>;
        }
      } catch {
        /* ignore invalid JSON */
      }
    }
    const bootKey = `${docId}:${docSession}`;
    if (docBootstrapRef.current === bootKey) return;
    docBootstrapRef.current = bootKey;
    const snippet = rawSnippet.slice(0, DOC_SNIPPET_MAX);
    setDocContext({ documentId: docId, title: docTitle, snippet, frontmatter: parsedFm });
  }, [params.docId, params.docTitle, params.docSnippet, params.docSession, params.docFrontmatter]);

  function clearDocRouteParams() {
    router.setParams({
      docId: '',
      docTitle: '',
      docSnippet: '',
      docSession: '',
      docFrontmatter: '',
    });
  }

  function dismissDocBanner() {
    setDocContext(null);
    docBootstrapRef.current = null;
    clearDocRouteParams();
  }

  async function pickConversation(c: ConversationSummary) {
    setConversationId(c.conversation_id);
    setHistoryOpen(false);
    await loadMessages(c.conversation_id);
  }

  async function newChat() {
    setDocContext(null);
    docBootstrapRef.current = null;
    clearDocRouteParams();
    const res = await createConversation({ title: 'Bastion Chat', initial_message: null });
    const id = res.conversation?.conversation_id;
    if (id) {
      setConversationId(id);
      await loadConversations();
      setRows([]);
    }
  }

  const activeEditorPayload = useMemo(() => {
    if (!docContext) return null;
    return buildActiveEditor(
      docContext.documentId,
      docContext.title,
      docContext.snippet,
      docContext.frontmatter
    );
  }, [docContext]);

  async function onSend() {
    const q = input.trim();
    if (!q || streaming) return;
    setInput('');
    let cid = conversationId;

    if (!cid) {
      const title = docContext ? `Document: ${docContext.title}` : null;
      try {
        const res = await createConversation({ title, initial_message: null });
        cid = res.conversation?.conversation_id ?? null;
      } catch {
        return;
      }
      if (!cid) return;
      setConversationId(cid);
      void loadConversations();
    }

    setRows((prev) => [...prev, { id: `local-${Date.now()}`, role: 'user', content: q }]);
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
        user_chat_model: selectedModelId || undefined,
        active_editor: activeEditorPayload,
        editor_preference: activeEditorPayload ? 'prefer' : undefined,
        signal: abortRef.current.signal,
        onChunk: (chunk) => {
          if (chunk.type === 'status') {
            const msg = String(chunk.message || chunk.content || '').trim();
            if (msg && !acc) {
              setRows((prev) =>
                prev.map((r) => (r.id === assistantId ? { ...r, statusText: msg } : r))
              );
            }
          }
          if (chunk.type === 'content' && typeof chunk.content === 'string') {
            acc += chunk.content;
            setRows((prev) =>
              prev.map((r) =>
                r.id === assistantId ? { ...r, content: acc, statusText: undefined } : r
              )
            );
          }
          if (
            chunk.type === 'complete' &&
            typeof chunk.content === 'string' &&
            chunk.content &&
            !acc
          ) {
            acc = chunk.content;
            setRows((prev) =>
              prev.map((r) =>
                r.id === assistantId ? { ...r, content: acc, statusText: undefined } : r
              )
            );
          }
        },
      });
    } catch {
      setRows((prev) =>
        prev.map((r) =>
          r.id === assistantId
            ? { ...r, content: r.content || '(Stream failed)', statusText: undefined }
            : r
        )
      );
    } finally {
      setStreaming(false);
      if (cid && !acc) void loadMessages(cid);
    }
  }

  async function onPickModel(m: EnabledModel) {
    setSelectedModelId(m.model_id);
    setModelPickerOpen(false);
    try {
      await setUserChatModelRole(m.model_id);
    } catch {
      /* still use locally for this session */
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
      {docContext ? (
        <View style={styles.docBanner}>
          <Text style={styles.docBannerText} numberOfLines={1}>
            Context: {docContext.title}
          </Text>
          <Pressable onPress={dismissDocBanner} hitSlop={12} accessibilityLabel="Clear document context">
            <Text style={styles.docBannerDismiss}>×</Text>
          </Pressable>
        </View>
      ) : null}

      <View style={styles.toolbar}>
        <Pressable
          style={styles.newBtn}
          onPress={() => void newChat()}
          onLongPress={() => setHistoryOpen(true)}
          delayLongPress={400}
        >
          <Text style={styles.newBtnText}>New chat</Text>
        </Pressable>
        <Text style={styles.hint}>Hold for recent chats</Text>
      </View>

      <FlatList
        ref={listRef}
        data={rows}
        keyExtractor={(r) => r.id}
        contentContainerStyle={styles.list}
        onContentSizeChange={() => {
          requestAnimationFrame(() => {
            listRef.current?.scrollToEnd({ animated: true });
          });
        }}
        renderItem={({ item }) => (
          <View
            style={[
              styles.bubble,
              item.role === 'user' ? styles.bubbleUser : styles.bubbleAsst,
            ]}
          >
            {item.role === 'assistant' ? (
              item.content.trim() ? (
                <Markdown style={assistantMarkdownStyles}>{item.content}</Markdown>
              ) : item.statusText ? (
                <Text style={styles.statusText}>{item.statusText}</Text>
              ) : (
                <Text style={assistantMarkdownStyles.body}>{'\u00a0'}</Text>
              )
            ) : (
              <Text style={[styles.msg, styles.msgUser]}>{item.content}</Text>
            )}
          </View>
        )}
      />

      <View style={styles.composer}>
        <TextInput
          style={styles.input}
          placeholder="Ask Bastion…"
          placeholderTextColor="#888"
          value={input}
          onChangeText={setInput}
          multiline
          blurOnSubmit={false}
          accessibilityHint="Return adds a new line. Use the Send button to send."
        />
        <Pressable
          style={styles.sendBtn}
          onPress={() => void onSend()}
          disabled={streaming}
          accessibilityRole="button"
          accessibilityLabel="Send message"
        >
          <Text style={styles.sendText}>{streaming ? '…' : 'Send'}</Text>
        </Pressable>
      </View>

      <Modal visible={historyOpen} animationType="slide" transparent onRequestClose={() => setHistoryOpen(false)}>
        <Pressable style={styles.modalBackdrop} onPress={() => setHistoryOpen(false)}>
          <Pressable
            style={[styles.sheet, { paddingBottom: modalSheetBottomPad }]}
            onPress={(e) => e.stopPropagation()}
          >
            <Text style={styles.sheetTitle}>Recent chats</Text>
            <FlatList
              data={recentConversations}
              keyExtractor={(c) => c.conversation_id}
              style={styles.sheetList}
              ListEmptyComponent={<Text style={styles.sheetEmpty}>No conversations yet.</Text>}
              renderItem={({ item }) => (
                <Pressable
                  style={styles.sheetRow}
                  onPress={() => void pickConversation(item)}
                >
                  <Text style={styles.sheetRowTitle} numberOfLines={1}>
                    {item.title || 'Untitled'}
                  </Text>
                  {item.last_message_at ? (
                    <Text style={styles.sheetRowSub}>{dayjs(item.last_message_at).fromNow()}</Text>
                  ) : null}
                </Pressable>
              )}
            />
            <Pressable style={styles.sheetClose} onPress={() => setHistoryOpen(false)}>
              <Text style={styles.sheetCloseText}>Close</Text>
            </Pressable>
          </Pressable>
        </Pressable>
      </Modal>

      <Modal visible={modelPickerOpen} animationType="fade" transparent onRequestClose={() => setModelPickerOpen(false)}>
        <Pressable style={styles.modalBackdrop} onPress={() => setModelPickerOpen(false)}>
          <Pressable
            style={[styles.sheet, { paddingBottom: modalSheetBottomPad }]}
            onPress={(e) => e.stopPropagation()}
          >
            <Text style={styles.sheetTitle}>Chat model</Text>
            <ScrollView style={styles.sheetList} keyboardShouldPersistTaps="handled">
              {enabledModels.length === 0 ? (
                <Text style={styles.sheetEmpty}>
                  No enabled models. Configure providers in the web app under Settings.
                </Text>
              ) : (
                enabledModels.map((m) => (
                  <Pressable
                    key={m.model_id}
                    style={[styles.sheetRow, m.model_id === selectedModelId && styles.sheetRowActive]}
                    onPress={() => void onPickModel(m)}
                  >
                    <Text style={styles.sheetRowTitle} numberOfLines={2}>
                      {m.display_name}
                    </Text>
                    <Text style={styles.sheetRowSub} numberOfLines={1}>
                      {m.model_id}
                    </Text>
                  </Pressable>
                ))
              )}
            </ScrollView>
            <Pressable style={styles.sheetClose} onPress={() => setModelPickerOpen(false)}>
              <Text style={styles.sheetCloseText}>Close</Text>
            </Pressable>
          </Pressable>
        </Pressable>
      </Modal>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: '#f5f5f5' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  docBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: '#e8eaf6',
    borderBottomWidth: 1,
    borderColor: '#c5cae9',
  },
  docBannerText: { flex: 1, fontSize: 14, fontWeight: '600', color: '#1a1a2e', marginRight: 8 },
  docBannerDismiss: { fontSize: 22, color: '#333', lineHeight: 24 },
  toolbar: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderBottomWidth: 1,
    borderColor: '#ddd',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  newBtn: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#1a1a2e',
    borderRadius: 6,
  },
  newBtnText: { color: '#fff', fontWeight: '600' },
  hint: { fontSize: 12, color: '#666', flex: 1 },
  list: { padding: 12 },
  bubble: {
    padding: 10,
    borderRadius: 10,
    marginBottom: 10,
    maxWidth: '92%',
  },
  bubbleUser: { alignSelf: 'flex-end', backgroundColor: '#1a1a2e' },
  bubbleAsst: { alignSelf: 'flex-start', backgroundColor: '#fff', borderWidth: 1, borderColor: '#ddd' },
  msg: { fontSize: 15, color: '#111' },
  msgUser: { color: '#fff' },
  statusText: { fontSize: 14, color: '#888', fontStyle: 'italic' },
  composer: { flexDirection: 'row', padding: 8, alignItems: 'flex-end', borderTopWidth: 1, borderColor: '#ddd', gap: 8 },
  headerModelBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    maxWidth: 160,
    marginRight: 4,
    paddingVertical: 6,
    paddingHorizontal: 8,
    gap: 2,
  },
  headerModelLabel: { fontSize: 13, fontWeight: '700', color: '#1a5090', flexShrink: 1 },
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
  sendBtn: {
    justifyContent: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#1a1a2e',
    borderRadius: 8,
  },
  sendText: { color: '#fff', fontWeight: '600' },
  modalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'flex-end',
  },
  sheet: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    maxHeight: '55%',
  },
  sheetTitle: { fontSize: 18, fontWeight: '700', padding: 16, borderBottomWidth: 1, borderColor: '#eee' },
  sheetList: { maxHeight: 320 },
  sheetEmpty: { padding: 24, textAlign: 'center', color: '#666' },
  sheetRow: { paddingVertical: 14, paddingHorizontal: 16, borderBottomWidth: 1, borderColor: '#f0f0f0' },
  sheetRowActive: { backgroundColor: '#e8f5e9' },
  sheetRowTitle: { fontSize: 16, fontWeight: '600', color: '#111' },
  sheetRowSub: { fontSize: 12, color: '#888', marginTop: 4 },
  sheetClose: { marginTop: 8, alignSelf: 'center', paddingVertical: 10, paddingHorizontal: 24 },
  sheetCloseText: { fontSize: 16, color: '#1a5090', fontWeight: '600' },
});
