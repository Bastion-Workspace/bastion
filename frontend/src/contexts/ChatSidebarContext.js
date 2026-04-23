import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useLayoutEffect,
  useMemo,
  useCallback,
  useReducer,
  useRef,
} from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useLocation } from 'react-router-dom';
import { useAuth } from './AuthContext';
import apiService from '../services/apiService';
import BackgroundJobService from '../services/backgroundJobService';
import tabNotificationManager from '../utils/tabNotification';
import browserNotificationManager from '../utils/browserNotification';
import { documentDiffStore } from '../services/documentDiffStore';
import { createAgentStatusWebSocket } from '../utils/agentStatusTypes';
import { buildMessageTree, getActivePath, getNextSibling, extendToLinearLeaf } from '../utils/messageTreeUtils';
import { devLog } from '../utils/devConsole';
import {
  LEGACY_CHAT_CONVERSATION_STORAGE_KEY,
  persistedActiveConversationLocalKey,
  activeConversationSessionStorageKey,
  readPersistedChatModelForUser,
  readPersistedUserEditorPreferenceForUser,
  writePersistedChatModelForUser,
  writePersistedUserEditorPreferenceForUser,
  clearLegacyGlobalChatPreferenceKeys,
} from '../utils/chatSelectionStorage';

// Format agent type to display name
const formatAgentName = (agentType) => {
  if (!agentType) return 'AI';
  const formatted = agentType
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
  return formatted;
};

/** Stable key for deduping optimistic UI messages against server rows (role + content prefix). */
function buildMessageMergeKey(msg) {
  const role = msg.role || msg.type || '';
  const text = (msg.content || '').trim().slice(0, 500);
  return `${role}|${text}`;
}

function appendStreamingTailToBase(base, prevMessages) {
  const tail = prevMessages.filter(
    (msg) =>
      (msg.isStreaming || msg.isPending) &&
      !base.some(
        (p) => String(p.message_id || p.id) === String(msg.message_id || msg.id)
      )
  );
  return tail.length ? [...base, ...tail] : base;
}

/** After streaming completes, refetch can return before new rows exist; keep local messages not yet on server. */
function appendOptimisticTailToBase(base, prevMessages) {
  const baseIds = new Set(
    base.map((p) => String(p.message_id || p.id)).filter((x) => x && x !== 'undefined')
  );
  const baseKeys = new Set(base.map(buildMessageMergeKey));
  const extra = prevMessages.filter((msg) => {
    if (msg.isStreaming || msg.isPending) return false;
    const id = msg.message_id ?? msg.id;
    if (id != null && id !== undefined && baseIds.has(String(id))) return false;
    const key = buildMessageMergeKey(msg);
    if (key === '|') return false;
    if (baseKeys.has(key)) return false;
    return true;
  });
  return extra.length ? [...base, ...extra] : base;
}

function mergeFetchedPathWithPrevious(pathForDisplay, prevMessages) {
  const withStreaming = appendStreamingTailToBase(pathForDisplay, prevMessages);
  return appendOptimisticTailToBase(withStreaming, prevMessages);
}

const ARTIFACT_DRAWER_INITIAL = { active: null, history: [] };

function artifactDrawerReducer(state, action) {
  switch (action.type) {
    case 'open': {
      if (!action.payload) return state;
      const next = action.payload;
      if (!state.active) {
        return { ...state, active: next };
      }
      if (state.active.code !== next.code) {
        return { active: next, history: [...state.history, state.active] };
      }
      return { ...state, active: next };
    }
    case 'close':
      return ARTIFACT_DRAWER_INITIAL;
    case 'revert': {
      const idx = action.index;
      if (idx < 0 || idx >= state.history.length) return state;
      return {
        active: state.history[idx],
        history: state.history.slice(0, idx),
      };
    }
    case 'stream_replace_if_open': {
      const next = action.payload;
      if (!next || state.active == null) return state;
      if (state.active.code !== next.code) {
        return { active: next, history: [...state.history, state.active] };
      }
      return { ...state, active: next };
    }
    default:
      return state;
  }
}

const ChatSidebarContext = createContext();

export const useChatSidebar = () => {
  const context = useContext(ChatSidebarContext);
  if (!context) {
    throw new Error('useChatSidebar must be used within a ChatSidebarProvider');
  }
  return context;
};

export const ChatSidebarProvider = ({ children }) => {
  const location = useLocation();
  const { isAuthenticated, user, loading: authLoading } = useAuth();
  // Note: EditorProvider is a child of ChatSidebarProvider, so we can't use useEditor() here
  // We'll check localStorage directly with strict validation instead
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(420);
  const [isFullWidth, setIsFullWidth] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  // Restored per-user after auth resolves (see restore effect); avoid global localStorage on first paint.
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  /** Full message tree (all branches) when loaded with include_tree */
  const [allMessages, setAllMessages] = useState([]);
  /** Active leaf message_id from server */
  const [currentNodeId, setCurrentNodeId] = useState(null);
  const [query, setQuery] = useState('');
  const [replyToMessage, setReplyToMessage] = useState(null); // Message being replied to
  const [selectedModel, setSelectedModel] = useState('');
  const [backgroundJobService, setBackgroundJobService] = useState(null);
  /** When set, chat follow-ups route to this agent line (CEO) until @auto */
  const [activeLineRouting, setActiveLineRouting] = useState(null);
  /** Chat artifact drawer: active payload + session-only version stack. */
  const [artifactDrawer, dispatchArtifactDrawer] = useReducer(
    artifactDrawerReducer,
    ARTIFACT_DRAWER_INITIAL
  );
  const activeArtifact = artifactDrawer.active;
  const artifactHistory = artifactDrawer.history;

  const [artifactCollapsed, setArtifactCollapsed] = useState(false);

  const prevArtifactCodeRef = useRef(null);
  useEffect(() => {
    const code = activeArtifact?.code ?? null;
    if (code != null && code !== prevArtifactCodeRef.current) {
      setArtifactCollapsed(false);
    }
    prevArtifactCodeRef.current = code;
  }, [activeArtifact]);

  useEffect(() => {
    const uid = user?.user_id;
    try {
      if (uid && currentConversationId) {
        const sk = activeConversationSessionStorageKey(uid);
        if (sk) sessionStorage.setItem(sk, currentConversationId);
      } else if (uid && !currentConversationId) {
        const sk = activeConversationSessionStorageKey(uid);
        if (sk) sessionStorage.removeItem(sk);
      }
    } catch (_) {
      /* ignore quota or private mode */
    }
  }, [currentConversationId, user?.user_id]);

  const setActiveArtifact = useCallback((art) => {
    if (art == null) {
      setArtifactCollapsed(false);
      dispatchArtifactDrawer({ type: 'close' });
    } else {
      dispatchArtifactDrawer({ type: 'open', payload: art });
    }
  }, []);

  const openArtifact = useCallback(
    (art) => {
      if (art == null) return;
      if (
        artifactCollapsed &&
        activeArtifact &&
        activeArtifact.code === art.code
      ) {
        setArtifactCollapsed(false);
        return;
      }
      dispatchArtifactDrawer({ type: 'open', payload: art });
    },
    [artifactCollapsed, activeArtifact]
  );

  const revertArtifact = useCallback((index) => {
    dispatchArtifactDrawer({ type: 'revert', index });
  }, []);
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);

  const messageTree = useMemo(() => buildMessageTree(allMessages), [allMessages]);

  const normalizeServerMessage = useCallback((message) => ({
    id: message.message_id || message.id,
    message_id: message.message_id,
    role: message.message_type || message.role,
    type: message.message_type || message.role,
    content: message.content,
    timestamp: message.created_at || message.timestamp,
    created_at: message.created_at,
    sequence_number: message.sequence_number,
    parent_message_id: message.parent_message_id,
    branch_id: message.branch_id,
    citations: message.citations || [],
    metadata: message.metadata_json || message.metadata || {},
    editor_operations: (message.metadata_json || message.metadata || {})?.editor_operations || message.editor_operations || [],
    editor_document_id: (message.metadata_json || message.metadata || {})?.editor_document_id ?? message.editor_document_id,
    editor_filename: (message.metadata_json || message.metadata || {})?.editor_filename ?? message.editor_filename,
    ...message,
  }), []);
  // LangGraph is the only system - no toggles needed
  const useLangGraphSystem = true; // Always use LangGraph
  const messagesConversationIdRef = React.useRef(null); // Track which conversation the current messages belong to
  /** Latest action implementations for stable context wrappers (avoids new context value every render). */
  const chatSidebarActionsRef = React.useRef({});
  /** Active streaming fetch AbortController; Stop aborts read and backend uses Redis cancel when run_id is known. */
  const streamAbortControllerRef = React.useRef(null);
  /** Assistant placeholder message id for the in-flight stream (Stop / abort UI). */
  const activeStreamMessageIdRef = React.useRef(null);
  const lastCreatedConversationIdRef = React.useRef(null); // Skip redundant refetch after creating new conversation
  const isLoadingFromMetadataRef = React.useRef(false); // Track when we're loading preferences from metadata to prevent save loop
  const currentConversationIdRef = React.useRef(null);
  const lastRestoreAttemptUserRef = React.useRef(null);
  const lastHydratedChatPrefsUserRef = React.useRef(null);

  // **CONVERSATION-SCOPED ACTIVITY STATE**: Isolate loading indicators, job IDs, and executing plans per conversation
  // Map<conversationId, { isLoading, currentJobId, executingPlans }>
  const [conversationActivityState, setConversationActivityState] = useState(new Map());
  
  // Helper to get current conversation's activity state
  const getCurrentActivityState = React.useCallback(() => {
    if (!currentConversationId) {
      return { isLoading: false, currentJobId: null, executingPlans: new Set() };
    }
    return conversationActivityState.get(currentConversationId) || {
      isLoading: false,
      currentJobId: null,
      executingPlans: new Set()
    };
  }, [currentConversationId, conversationActivityState]);
  
  // Helper to update current conversation's activity state
  const updateCurrentActivityState = React.useCallback((updates) => {
    if (!currentConversationId) return;
    
    setConversationActivityState(prev => {
      const newMap = new Map(prev);
      const currentState = newMap.get(currentConversationId) || {
        isLoading: false,
        currentJobId: null,
        executingPlans: new Set()
      };
      
      // Merge updates with current state
      newMap.set(currentConversationId, { ...currentState, ...updates });
      
      return newMap;
    });
  }, [currentConversationId]);
  
  // Expose as computed values for backward compatibility
  const isLoading = getCurrentActivityState().isLoading;
  const currentJobId = getCurrentActivityState().currentJobId;
  const executingPlans = getCurrentActivityState().executingPlans;
  
  // **ROOSEVELT'S PREFERENCE MANAGEMENT**: Store user preferences separately from what gets sent to backend
  // User preference: what the user actually toggled (persists across navigation)
  const [userEditorPreference, setUserEditorPreference] = useState('prefer');

  // Active preference: what actually gets sent to the backend (context-aware)
  const [editorPreference, setEditorPreference] = useState(() => {
    try {
      const pathname = typeof window !== 'undefined' ? window.location.pathname : '';
      return pathname.startsWith('/documents') ? 'prefer' : 'ignore';
    } catch {
      return 'ignore';
    }
  });

  // **ROOSEVELT'S SMART PREFERENCE SYSTEM**: 
  // User preferences persist across navigation, but active preferences are context-aware
  // This ensures preferences are remembered while keeping agents isolated to their pages
  useEffect(() => {
    const onDocumentsPage = location.pathname.startsWith('/documents');
    
    // Apply user's editor preference ONLY when on documents page, otherwise force 'ignore'
    if (onDocumentsPage) {
      if (editorPreference !== userEditorPreference) {
        devLog('On documents page - applying user editor preference:', userEditorPreference);
        setEditorPreference(userEditorPreference);
      }
    } else {
      if (editorPreference !== 'ignore') {
        devLog('Not on documents page - disabling editor preference');
        setEditorPreference('ignore');
      }
    }
  }, [location.pathname, userEditorPreference, editorPreference]);

  const [dataWorkspacePreference, setDataWorkspacePreferenceState] = useState(() => {
    try { return localStorage.getItem('dataWorkspacePreference') || 'auto'; } catch { return 'auto'; }
  });
  const setDataWorkspacePreference = React.useCallback((pref) => {
    setDataWorkspacePreferenceState(pref);
    try { localStorage.setItem('dataWorkspacePreference', pref); } catch {}
  }, []);

  // WebSocket: line sub-agent messages persisted during CEO run (out-of-band from SSE)
  useEffect(() => {
    if (!currentConversationId) return undefined;
    const token = localStorage.getItem('auth_token');
    if (!token) return undefined;
    const ctrl = createAgentStatusWebSocket({
      conversationId: currentConversationId,
      token,
      onMessage: (msg) => {
        if (msg.type !== 'line_agent_message') return;
        if (msg.conversation_id && msg.conversation_id !== currentConversationId) return;
        const mid = msg.message_id || `line-${Date.now()}`;
        setMessages((prev) => {
          if (prev.some((m) => m.message_id === mid || m.id === mid)) return prev;
          return [
            ...prev,
            {
              id: mid,
              message_id: mid,
              role: 'assistant',
              type: 'assistant',
              content: msg.content || '',
              timestamp: msg.timestamp || new Date().toISOString(),
              metadata: msg.metadata || {},
            },
          ];
        });
      },
    });
    return () => {
      try {
        ctrl.close();
      } catch (_) {}
    };
  }, [currentConversationId]);

  const queryClient = useQueryClient();

  useEffect(() => {
    currentConversationIdRef.current = currentConversationId;
  }, [currentConversationId]);

  const clearConversationWorkspace = useCallback(() => {
    const id = currentConversationIdRef.current;
    if (backgroundJobService) {
      backgroundJobService.setCurrentConversationId(null);
      backgroundJobService.disconnectAll();
      backgroundJobService.clearCompletedJobs();
    }
    setCurrentConversationId(null);
    setMessages([]);
    setAllMessages([]);
    setCurrentNodeId(null);
    setQuery('');
    messagesConversationIdRef.current = null;
    if (id) {
      queryClient.removeQueries(['conversation', id]);
      queryClient.removeQueries(['conversationMessages', id]);
    }
    queryClient.invalidateQueries(['conversations']);
  }, [queryClient, backgroundJobService]);

  const forgetPersistedActiveThreadForUser = useCallback((userId) => {
    if (!userId) return;
    try {
      const lk = persistedActiveConversationLocalKey(userId);
      if (lk) localStorage.removeItem(lk);
      const sk = activeConversationSessionStorageKey(userId);
      if (sk) sessionStorage.removeItem(sk);
    } catch (_) {
      /* ignore */
    }
  }, []);

  const discardInaccessibleConversation = useCallback(() => {
    const uid = user?.user_id;
    clearConversationWorkspace();
    forgetPersistedActiveThreadForUser(uid);
  }, [clearConversationWorkspace, forgetPersistedActiveThreadForUser, user?.user_id]);

  const authChatSessionRef = useRef(null);
  // Layout: reset session on logout/switch and hydrate model/editor before child useEffects run
  // (avoids ChatInputArea keeping the previous user's model when it is still in the enabled list).
  useLayoutEffect(() => {
    if (authLoading) return;
    const snapshot = {
      authed: !!(isAuthenticated && user?.user_id),
      userId: user?.user_id ?? null,
    };
    const prev = authChatSessionRef.current;
    if (prev != null) {
      const hadUser = prev.authed && prev.userId;
      const hasUser = snapshot.authed && snapshot.userId;
      const loggedOut = hadUser && !hasUser;
      const switchedUser = hadUser && hasUser && prev.userId !== snapshot.userId;
      if (loggedOut || switchedUser) {
        if (prev.userId) {
          try {
            const sk = activeConversationSessionStorageKey(prev.userId);
            if (sk) sessionStorage.removeItem(sk);
          } catch (_) {
            /* ignore */
          }
        }
        clearLegacyGlobalChatPreferenceKeys();
        lastRestoreAttemptUserRef.current = null;
        lastHydratedChatPrefsUserRef.current = null;
        clearConversationWorkspace();
        dispatchArtifactDrawer({ type: 'close' });
        setReplyToMessage(null);
        setActiveLineRouting(null);
        setSelectedModel('');
        setUserEditorPreference('prefer');
        setEditorPreference(location.pathname.startsWith('/documents') ? 'prefer' : 'ignore');
      }
    }
    authChatSessionRef.current = snapshot;

    const uid = user?.user_id;
    if (!uid) return;

    const ed = readPersistedUserEditorPreferenceForUser(uid);
    const onDocuments = location.pathname.startsWith('/documents');

    if (lastHydratedChatPrefsUserRef.current !== uid) {
      lastHydratedChatPrefsUserRef.current = uid;
      const model = readPersistedChatModelForUser(uid);
      setSelectedModel(model);
      if (model) {
        apiService.selectModel(model).catch(() => {});
      }
      setUserEditorPreference(ed);
    }
    setEditorPreference(onDocuments ? ed : 'ignore');
  }, [
    authLoading,
    isAuthenticated,
    user?.user_id,
    clearConversationWorkspace,
    location.pathname,
  ]);

  useEffect(() => {
    if (authLoading || !isAuthenticated || !user?.user_id) return;
    const uid = user.user_id;
    if (currentConversationId) return;
    if (lastRestoreAttemptUserRef.current === uid) return;
    lastRestoreAttemptUserRef.current = uid;

    const key = persistedActiveConversationLocalKey(uid);
    let saved = key ? localStorage.getItem(key) : null;
    if (!saved || saved === 'null') {
      const legacy = localStorage.getItem(LEGACY_CHAT_CONVERSATION_STORAGE_KEY);
      if (legacy && legacy !== 'null' && key) {
        try {
          localStorage.setItem(key, legacy);
          localStorage.removeItem(LEGACY_CHAT_CONVERSATION_STORAGE_KEY);
          saved = legacy;
        } catch (_) {
          /* ignore */
        }
      }
    }
    if (saved && saved !== 'null') {
      devLog('💾 Restoring last active conversation for user from localStorage:', saved);
      setCurrentConversationId(saved);
    }
  }, [authLoading, isAuthenticated, user?.user_id, currentConversationId]);

  // Preference update function for saving to conversation metadata
  const updateConversationPreference = React.useCallback(async (key, value) => {
    if (!currentConversationId) {
      devLog('No conversation - only updating global preference');
      return;
    }
    
    try {
      // Save to conversation metadata
      await apiService.patch(`/api/conversations/${currentConversationId}/metadata`, {
        metadata: {
          [key]: value
        }
      });
      
      devLog(`Saved ${key} to conversation ${currentConversationId}:`, value);
      
      // Invalidate conversation query to refresh
      queryClient.invalidateQueries(['conversation', currentConversationId]);
    } catch (error) {
      console.error(`Failed to save ${key} to conversation:`, error);
    }
  }, [currentConversationId, queryClient]);

  // Handle editor preference change - save to both global and conversation
  const handleEditorPreferenceChange = React.useCallback(async (newPreference) => {
    // Update user preference (persists across navigation)
    setUserEditorPreference(newPreference);
    if (user?.user_id) {
      writePersistedUserEditorPreferenceForUser(user.user_id, newPreference);
    }
    
    // Update active preference (context-aware)
    if (location.pathname.startsWith('/documents')) {
      setEditorPreference(newPreference);
      
      // Save to current conversation
      if (currentConversationId) {
        await updateConversationPreference('editor_preference', newPreference);
      }
    }
  }, [currentConversationId, updateConversationPreference, location.pathname, user?.user_id]);

  // Initialize background job service
  useEffect(() => {
    devLog('🔄 Initializing background job service...');
    const service = new BackgroundJobService(apiService);
    devLog('✅ Background job service created:', service);
    setBackgroundJobService(service);
    
    return () => {
      if (service) {
        devLog('🧹 Cleaning up background job service...');
        service.disconnectAll();
      }
    };
  }, []);

  // CRITICAL: Log conversation ID changes for debugging
  useEffect(() => {
    devLog('🔄 ChatSidebarContext: currentConversationId changed to:', currentConversationId);
  }, [currentConversationId]);

  // Load sidebar layout from localStorage (device-level). Model/editor hydrate per user after auth.
  useEffect(() => {
    const savedCollapsed = localStorage.getItem('chatSidebarCollapsed');
    const savedWidth = localStorage.getItem('chatSidebarWidth');
    const savedFullWidth = localStorage.getItem('chatSidebarFullWidth');
    
    if (savedCollapsed !== null) {
      setIsCollapsed(JSON.parse(savedCollapsed));
    }
    
    if (savedWidth !== null) {
      setSidebarWidth(JSON.parse(savedWidth));
    }
    
    if (savedFullWidth !== null) {
      setIsFullWidth(JSON.parse(savedFullWidth));
    }
    // Auto-collapse on small screens
    try {
      if (window && window.matchMedia && window.matchMedia('(max-width: 900px)').matches) {
        setIsCollapsed(true);
        setIsFullWidth(false);
      }
    } catch {}
  }, []);

  // Save sidebar preferences to localStorage
  useEffect(() => {
    localStorage.setItem('chatSidebarCollapsed', JSON.stringify(isCollapsed));
  }, [isCollapsed]);

  useEffect(() => {
    localStorage.setItem('chatSidebarWidth', JSON.stringify(sidebarWidth));
  }, [sidebarWidth]);

  useEffect(() => {
    localStorage.setItem('chatSidebarFullWidth', JSON.stringify(isFullWidth));
  }, [isFullWidth]);

  useEffect(() => {
    const uid = user?.user_id;
    if (!uid) return;
    try {
      writePersistedUserEditorPreferenceForUser(uid, userEditorPreference);
    } catch (_) {
      /* ignore */
    }
  }, [user?.user_id, userEditorPreference]);

  // Save selected model to localStorage and conversation metadata
  useEffect(() => {
    // Skip saving if we're currently loading from metadata (prevents circular updates)
    if (isLoadingFromMetadataRef.current) {
      return;
    }
    
    const uid = user?.user_id;
    if (uid && selectedModel) {
      writePersistedChatModelForUser(uid, selectedModel);
      
      // Also save to conversation metadata if we have a conversation
      if (currentConversationId) {
        updateConversationPreference('user_chat_model', selectedModel).catch(err => {
          console.error('Failed to save model to conversation:', err);
        });
      }
      
      // Notify backend of model selection
      apiService.selectModel(selectedModel).catch(err => {
        console.error('Failed to notify backend of model selection:', err);
      });
    } else if (uid && !selectedModel) {
      writePersistedChatModelForUser(uid, '');
    }
  }, [selectedModel, currentConversationId, updateConversationPreference, user?.user_id]);

  // Per-user: last active thread survives logout/login on the same browser profile.
  useEffect(() => {
    const uid = user?.user_id;
    if (!uid) return;
    const key = persistedActiveConversationLocalKey(uid);
    if (!key) return;
    try {
      if (currentConversationId) {
        localStorage.setItem(key, currentConversationId);
        devLog('💾 Persisted conversation to localStorage:', currentConversationId, 'for user', uid);
      } else {
        localStorage.removeItem(key);
        devLog('💾 Cleared persisted conversation for user', uid);
      }
    } catch (error) {
      console.error('Failed to persist current conversation to localStorage:', error);
    }
  }, [currentConversationId, user?.user_id]);

  // PRIORITY: Use unified chat service for conversation loading
  const { data: conversationData, isLoading: conversationLoading, refetch: refetchConversation } = useQuery(
    ['conversation', currentConversationId],
    () => currentConversationId ? apiService.getConversation(currentConversationId) : null,
    {
      enabled: !!currentConversationId,
      refetchOnWindowFocus: false,
      refetchOnMount: true, // Always refetch when component mounts (e.g., switching conversations)
      staleTime: 0, // Always consider data stale to get latest metadata
      onSuccess: (data) => {
        devLog('✅ ChatSidebarContext: Conversation data loaded:', {
          conversationId: currentConversationId,
          hasMessages: !!data?.messages,
          messageCount: data?.messages?.length || 0,
          hasMetadata: !!data?.metadata_json,
          metadataKeys: data?.metadata_json ? Object.keys(data.metadata_json) : [],
          userChatModel: data?.metadata_json?.user_chat_model
        });
      },
      onError: (error) => {
        console.error('❌ ChatSidebarContext: Failed to load conversation:', error);
        const status = error?.response?.status;
        if (status === 403 || status === 404) {
          discardInaccessibleConversation();
        }
      }
    }
  );
  
  // Refetch conversation when switching to ensure we get latest metadata (skip when we just created it; cache is primed)
  useEffect(() => {
    if (!currentConversationId || !refetchConversation) return;
    if (lastCreatedConversationIdRef.current === currentConversationId) {
      lastCreatedConversationIdRef.current = null;
      return;
    }
    devLog('🔄 Conversation switched, refetching to get latest metadata:', currentConversationId);
    refetchConversation();
  }, [currentConversationId, refetchConversation]);

  // Load conversation-specific preferences from metadata
  useEffect(() => {
    const uid = user?.user_id;
    if (!currentConversationId) {
      setActiveLineRouting(null);
      if (uid) {
        setSelectedModel(readPersistedChatModelForUser(uid) || '');
      }
      return;
    }
    
    // Mark that we're loading from metadata to prevent save effect from running
    isLoadingFromMetadataRef.current = true;
    
    if (conversationData?.metadata_json) {
      const metadata = conversationData.metadata_json;

      if (metadata.active_line_id) {
        setActiveLineRouting({
          id: metadata.active_line_id,
          name: metadata.active_line_name || '',
        });
      } else {
        setActiveLineRouting(null);
      }
      
      // Load model preference
      if (metadata.user_chat_model) {
        devLog('🔄 Loading conversation model preference:', metadata.user_chat_model, 'for conversation:', currentConversationId);
        setSelectedModel(metadata.user_chat_model);
      } else if (uid) {
        const scopedModel = readPersistedChatModelForUser(uid);
        if (scopedModel) {
          devLog('🔄 No conversation model preference, using user default:', scopedModel);
          setSelectedModel(scopedModel);
        }
      }
      
      // Load editor preference (only on documents page)
      if (metadata.editor_preference && location.pathname.startsWith('/documents')) {
        devLog('🔄 Loading conversation editor preference:', metadata.editor_preference);
        setEditorPreference(metadata.editor_preference);
      } else if (uid) {
        const scopedEditor = readPersistedUserEditorPreferenceForUser(uid);
        if (location.pathname.startsWith('/documents')) {
          setEditorPreference(scopedEditor);
        }
      }
    } else if (conversationData) {
      setActiveLineRouting(null);
      devLog('🔄 Conversation loaded but no metadata, using user defaults');
      if (uid) {
        const scopedModel = readPersistedChatModelForUser(uid);
        if (scopedModel) setSelectedModel(scopedModel);
        const scopedEditor = readPersistedUserEditorPreferenceForUser(uid);
        if (location.pathname.startsWith('/documents')) {
          setEditorPreference(scopedEditor);
        }
      }
    }
    
    // Reset the flag after a short delay to allow state updates to complete
    setTimeout(() => {
      isLoadingFromMetadataRef.current = false;
    }, 100);
  }, [conversationData, currentConversationId, location.pathname, user?.user_id]);

  // PRIORITY: Load messages using unified chat service
  const { data: messagesData, isLoading: messagesLoading, refetch: refetchMessages } = useQuery(
    ['conversationMessages', currentConversationId],
    () => currentConversationId
      ? apiService.getConversationMessages(
          currentConversationId,
          0,
          500,
          lastCreatedConversationIdRef.current === currentConversationId,
          true
        )
      : null,
    {
      enabled: !!currentConversationId,
      refetchOnWindowFocus: false,
      staleTime: 300000, // 5 minutes
      onSuccess: (data) => {
        devLog('✅ ChatSidebarContext: Messages loaded:', {
          conversationId: currentConversationId,
          messageCount: data?.messages?.length || 0,
          hasMore: data?.has_more || false
        });
        if (data?.messages) {
          const normalizedMessages = data.messages.map(normalizeServerMessage);
          setAllMessages(normalizedMessages);
          const nodeId =
            data.current_node_message_id ||
            data.currentNodeMessageId ||
            null;
          const tree = buildMessageTree(normalizedMessages);
          const effectiveNodeId =
            nodeId && tree.byId.has(nodeId)
              ? extendToLinearLeaf(tree, nodeId)
              : nodeId;
          setCurrentNodeId(effectiveNodeId ?? null);
          let pathMsgs = effectiveNodeId
            ? getActivePath(tree, effectiveNodeId)
            : [];
          if (effectiveNodeId && pathMsgs.length === 0) {
            const orphan = normalizedMessages.find(
              (m) => (m.message_id || m.id) === effectiveNodeId
            );
            if (orphan) {
              pathMsgs = [orphan];
            }
          }
          const pathForDisplay =
            pathMsgs.length > 0
              ? pathMsgs.map((m) => normalizeServerMessage(m))
              : normalizedMessages;

          setMessages((prevMessages) => {
            if (messagesConversationIdRef.current !== currentConversationId) {
              messagesConversationIdRef.current = currentConversationId;
            }
            return mergeFetchedPathWithPrevious(pathForDisplay, prevMessages);
          });
        }
      },
      onError: (error) => {
        console.error('❌ ChatSidebarContext: Failed to load messages:', error);
        const status = error?.response?.status;
        if (status === 403 || status === 404) {
          discardInaccessibleConversation();
        }
      }
    }
  );

  // Update messages when conversation data changes (fallback)
  useEffect(() => {
    const embedded = conversationData?.messages;
    if (
      Array.isArray(embedded) &&
      embedded.length > 0 &&
      !messagesData
    ) {
      devLog('🔄 ChatSidebarContext: Using fallback conversation messages');
      // Normalize message format for consistent frontend handling
      const normalizedMessages = embedded.map(message => ({
        id: message.message_id || message.id,
        message_id: message.message_id,
        role: message.role,
        type: message.role, // Add type field for components that expect it
        content: message.content,
        timestamp: message.created_at || message.timestamp,
        created_at: message.created_at,
        sequence_number: message.sequence_number,
        citations: message.citations || [],
        metadata: message.metadata || {},
        // ✅ CRITICAL FIX: Extract editor_operations from metadata to top level
        editor_operations: message.metadata?.editor_operations || message.editor_operations || [],
        editor_document_id: message.metadata?.editor_document_id ?? message.editor_document_id,
        editor_filename: message.metadata?.editor_filename ?? message.editor_filename,
        // Preserve any other fields
        ...message
      }));
      setMessages(normalizedMessages);
    }
  }, [conversationData, messagesData]);

  // Create new conversation mutation
  const createConversationMutation = useMutation(
    () => apiService.createConversation(),
    {
      onSuccess: (newConversation) => {
        const id = newConversation.conversation.conversation_id;
        // Prime cache so GET conversation and GET messages are skipped (faster new-chat UX)
        queryClient.setQueryData(['conversation', id], newConversation);
        queryClient.setQueryData(['conversationMessages', id], { messages: [], total_count: 0, has_more: false });
        lastCreatedConversationIdRef.current = id;
        setCurrentConversationId(id);
        setMessages([]);
        setAllMessages([]);
        setCurrentNodeId(null);
        messagesConversationIdRef.current = null;
        setQuery('');
        queryClient.invalidateQueries(['conversations']);
        // Clear "just created" ref after a short window so future refetches run normally
        setTimeout(() => { lastCreatedConversationIdRef.current = null; }, 30000);
      },
      onError: (error) => {
        console.error('Failed to create conversation:', error);
      },
    }
  );

  // Handle background job progress updates
  const handleBackgroundJobProgress = (jobData) => {
    devLog('🔄 Background job progress:', jobData);
    
    // Update the execution message with progress
    setMessages(prev => prev.map(msg => {
      if (msg.jobId === jobData.job_id || msg.jobId === `research_plan_${currentConversationId}`) {
        return {
          ...msg,
          content: `🔄 **Executing research plan...** \n\n${jobData.message}\n\nProgress: ${jobData.progress}%`,
          progress: jobData.progress,
          currentTool: jobData.current_tool,
          currentIteration: jobData.current_iteration,
          maxIterations: jobData.max_iterations
        };
      }
      return msg;
    }));
  };

  // Handle background job completion
  const handleBackgroundJobCompleted = (jobData) => {
    devLog('✅ Background job completed:', jobData);
    devLog('🔍 Job details:', {
      jobId: jobData.job_id,
      conversationId: jobData.conversation_id,
      currentConversationId: currentConversationId,
      hasResult: !!jobData.result,
      hasAnswer: !!(jobData.result && jobData.result.answer),
      hasPlan: !!(jobData.result && jobData.result.research_plan)
    });
    
    // CRITICAL: Refresh messages from database to ensure we get the complete conversation
    // This ensures research plan results are properly displayed
    if (currentConversationId && jobData.conversation_id === currentConversationId) {
      devLog('🔄 Refreshing messages from database after job completion');
      refetchMessages(); // This will reload messages from the API
      
      // Also invalidate the conversation query to ensure fresh data
      queryClient.invalidateQueries(['conversation', currentConversationId]);
      queryClient.invalidateQueries(['conversationMessages', currentConversationId]);
    }
    
    // Find and update the pending message (fallback for immediate UI update)
    setMessages(prev => {
      let hasNewContent = false;
      const updated = prev.map(msg => {
        if (msg.jobId === jobData.job_id) {
          const newContent = jobData.result?.answer || jobData.answer || 'Job completed successfully';
          // Check if this is actually new content (not just a status update)
          if (newContent && newContent.trim().length > 0 && 
              (!msg.content || msg.content !== newContent)) {
            hasNewContent = true;
          }
          
          const updatedMessage = {
            ...msg,
            content: newContent,
            isResearchJob: false,
            timestamp: new Date().toISOString(),
          };
          
          // Add research plan data if present
          if (jobData.result?.research_plan) {
            updatedMessage.research_plan = jobData.result.research_plan;
            updatedMessage.planApproved = jobData.result.plan_approved || false;
          }
          
          return updatedMessage;
        }
        return msg;
      });
      
      // Flash tab if new content was added and tab is hidden
      if (hasNewContent) {
        tabNotificationManager.startFlashing('New message');
      }
      
      return updated;
    });
    
    // Remove from executing plans
    updateCurrentActivityState({
      executingPlans: (() => {
        const currentState = getCurrentActivityState();
        const newSet = new Set(currentState.executingPlans);
        newSet.delete(jobData.job_id);
        return newSet;
      })()
    });
    
    // Invalidate conversations to refresh the list
    queryClient.invalidateQueries(['conversations']);
    
    // Clear current job ID when job completes
    if (currentJobId === jobData.job_id) {
      updateCurrentActivityState({ currentJobId: null });
    }
    
    devLog('✅ Background job completion handling completed');
  };

  // Check if a query is a HITL permission response
  const isHITLPermissionResponse = (query) => {
    const lowerQuery = query.toLowerCase().trim();
    return (
      lowerQuery === 'yes' ||
      lowerQuery === 'y' ||
      lowerQuery === 'ok' ||
      lowerQuery === 'okay' ||
      lowerQuery === 'sure' ||
      lowerQuery === 'go ahead' ||
      lowerQuery === 'proceed' ||
      lowerQuery === 'approved' ||
      lowerQuery === 'approve' ||
      lowerQuery === 'allow' ||
      lowerQuery === 'no' ||
      lowerQuery === 'n' ||
      lowerQuery === 'deny' ||
      lowerQuery === 'decline' ||
      lowerQuery === 'cancel'
    );
  };

  // Find the most recent HITL permission request message
  const findRecentPermissionRequest = () => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      if (message.role === 'assistant' && message.content && (
        message.content.includes('🔍 Research Permission Required') ||
        message.content.includes('Permission Required') ||
        message.content.includes('May I proceed') ||
        message.content.includes('Do you approve') ||
        message.content.includes('web search') ||
        message.content.includes('search the web') ||
        message.content.includes('external search')
      )) {
        return message;
      }
    }
    return null;
  };

  const toggleSidebar = () => {
    setIsCollapsed((c) => !c);
  };

  const selectConversation = (conversationId) => {
    devLog('🔄 ChatSidebarContext: Selecting conversation:', conversationId);
    
    // Clear current state to prevent cross-conversation contamination
    setMessages([]);
    setAllMessages([]);
    setCurrentNodeId(null);
    setQuery('');
    // Activity state is now automatically isolated by conversationId
    // No need to manually clear - each conversation has its own state
    
    // Reset the messages conversation ID ref to trigger fresh load
    messagesConversationIdRef.current = null;
    
    // Update background job service with new conversation ID
    if (backgroundJobService) {
      devLog('🔄 ChatSidebarContext: Updating background job service conversation ID');
      backgroundJobService.setCurrentConversationId(conversationId);
      backgroundJobService.disconnectAll(); // Clear any existing connections
      backgroundJobService.clearCompletedJobs();
    }
    
    // Set the new conversation ID (this will trigger React Query to load messages)
    setCurrentConversationId(conversationId);
    
    // Invalidate queries to ensure fresh data
    queryClient.invalidateQueries(['conversation', conversationId]);
    queryClient.invalidateQueries(['conversationMessages', conversationId]);
    
    devLog('✅ ChatSidebarContext: Conversation selection completed for:', conversationId);
  };

  const createNewConversation = () => {
    createConversationMutation.mutate();
  };

  // Poll task status for async orchestrator
  // pollTaskStatus removed - no longer needed since we removed async fallback

  const sendMessage = async (executionMode = 'auto', overrideQuery = null) => {
    // ROOSEVELT'S HITL SUPPORT: Allow override query for direct API calls without state dependency
    let actualQuery = overrideQuery || query.trim();
    
    // Handle reply: prepend quoted message if replying
    if (replyToMessage && !overrideQuery) {
      const quotedContent = replyToMessage.content || '';
      const quotedPreview = quotedContent.length > 100 ? quotedContent.substring(0, 100) + '...' : quotedContent;
      const replyPrefix = `> ${quotedPreview}\n\n`;
      actualQuery = replyPrefix + actualQuery;
    }
    
    devLog('🔄 sendMessage called with:', { query: actualQuery, overrideQuery: !!overrideQuery, backgroundJobService: !!backgroundJobService, executionMode, hasReply: !!replyToMessage });
    
    if (!actualQuery || !backgroundJobService) {
      devLog('❌ sendMessage early return:', { hasQuery: !!actualQuery, hasService: !!backgroundJobService });
      return;
    }

    const currentQuery = actualQuery;
    
    // Clear reply state after using it
    if (replyToMessage && !overrideQuery) {
      setReplyToMessage(null);
    }
    let conversationId = currentConversationId;

    // Clear input immediately for better UX (only if not using override query)
    if (!overrideQuery) {
      setQuery('');
    }

    // ROOSEVELT'S HITL PRIORITY: Check for HITL permission responses FIRST
    if (isHITLPermissionResponse(currentQuery)) {
      const recentPermissionRequest = findRecentPermissionRequest();
      if (recentPermissionRequest) {
        devLog('🛡️ Detected HITL permission response, continuing LangGraph flow:', currentQuery);
        // Continue with normal LangGraph flow - it will handle the permission response
        // Don't return early - let the normal flow handle it
      }
    }

    // Add user message immediately
    const userMessage = {
      id: Date.now(),
      role: 'user',
      type: 'user', // Add type field for consistency
      content: currentQuery,
      timestamp: new Date().toISOString(),
    };

    // Handle conversation creation if needed
    if (!conversationId) {
      try {
        const newConversation = await apiService.createConversation({
          initial_message: currentQuery
        });
        devLog('🔍 Full conversation response:', newConversation);
        devLog('🔍 Response keys:', Object.keys(newConversation));
        conversationId = newConversation.conversation.conversation_id; // Access conversation_id through the conversation field
        setCurrentConversationId(conversationId);
        queryClient.invalidateQueries(['conversations']);
        devLog('✅ Created new conversation:', conversationId);
      } catch (error) {
        console.error('❌ Failed to create conversation:', error);
        setQuery(currentQuery); // Restore query on failure
        return;
      }
    }

    // Add user message to UI immediately
    setMessages(prev => [...prev, userMessage]);
    updateCurrentActivityState({ isLoading: true });

    // LangGraph is the only system - always use it
      // Use LangGraph system
      try {
        devLog('🔄 Using LangGraph system');
        
        // User message already added above, no need to add again
        
        // 🌊 STREAMING-FIRST POLICY: Stream everything for optimal UX!
        devLog('🌊 Using streaming for ALL queries');
        
        // Use streaming endpoint for ALL real-time responses
        await handleStreamingResponse(currentQuery, conversationId, sessionId);
        
      } catch (error) {
        console.error('❌ LangGraph failed:', error);
        setMessages(prev => [...prev, {
          id: Date.now(),
          role: 'system',
          type: 'system', // Add type field for consistency
          content: `❌ LangGraph failed: ${error.message}`,
          timestamp: new Date().toISOString(),
          isError: true,
        }]);
      } finally {
        updateCurrentActivityState({ isLoading: false });
      }
  };

  // 🌊 ROOSEVELT'S UNIVERSAL STREAMING POLICY: All queries deserve real-time responses!

  // Handle streaming response from orchestrator
  const handleStreamingResponse = async (query, conversationId, sessionId, streamOptions = {}) => {
    const { isBranchResend, branchMessageId } = streamOptions;
    devLog('🌊 Starting streaming response for:', query);
    
    try {
      updateCurrentActivityState({ currentJobId: null });

      // Add streaming message placeholder (server assigns run_id in first SSE run_started)
      const streamingMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        type: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        isStreaming: true,
        jobId: null,
        metadata: {
          streaming: true,
          agent_type: null
        }
      };
      
      setMessages(prev => [...prev, streamingMessage]);
      activeStreamMessageIdRef.current = streamingMessage.id;

      // Track if we've notified for this message to avoid multiple notifications during streaming
      let hasNotified = false;
      
      // Create EventSource for Server-Sent Events  
      const token = localStorage.getItem('auth_token'); // Match apiService token key
      devLog('🔑 Using auth_token for streaming:', token ? 'TOKEN_PRESENT' : 'NO_TOKEN');
      
      // Attach active_editor when preference is not ignore and cache shows an open doc with content (.md/.org editable, or .txt/.pdf/.docx read-only).
      // EditorProvider is a child of ChatSidebarProvider, so we read editor_ctx_cache from localStorage.
      let activeEditorPayload = null;
      
      // Check editor preference first - if set to 'ignore', don't send editor context
      if (editorPreference !== 'ignore') {
        try {
          // Request fresh frontmatter parsing before reading cache (ensures frontmatter is current)
          // Use a promise-based approach to ensure refresh completes
          const refreshPromise = new Promise((resolve) => {
            const handleRefreshComplete = () => {
              window.removeEventListener('editorCacheRefreshed', handleRefreshComplete);
              resolve();
            };
            window.addEventListener('editorCacheRefreshed', handleRefreshComplete, { once: true });
            
            window.dispatchEvent(new CustomEvent('refreshEditorCache'));
            
            setTimeout(() => {
              window.removeEventListener('editorCacheRefreshed', handleRefreshComplete);
              // Do not clear editor_ctx_cache on timeout: a slow editor mount or busy main thread
              // can miss the refresh event; clearing here used to drop valid active_editor.
              resolve();
            }, 400);
          });
          
          await refreshPromise;
          
          // Get editor state from localStorage (updated by DocumentViewer when editor is open)
          const editorCtx = JSON.parse(localStorage.getItem('editor_ctx_cache') || 'null');
          
          // DEBUG: Log what we got from localStorage
          devLog('🔍 EDITOR_CTX_CACHE DEBUG:', {
            exists: !!editorCtx,
            isEditable: editorCtx?.isEditable,
            filename: editorCtx?.filename,
            hasContent: !!(editorCtx?.content && editorCtx.content.trim().length > 0),
            contentLength: editorCtx?.content?.length || 0,
            frontmatterType: editorCtx?.frontmatter?.type,
            frontmatterKeys: editorCtx?.frontmatter ? Object.keys(editorCtx.frontmatter) : [],
            fullFrontmatter: editorCtx?.frontmatter,
            canonicalPath: editorCtx?.canonicalPath,
            documentId: editorCtx?.documentId,
            rawEditorCtxKeys: editorCtx ? Object.keys(editorCtx) : []
          });
          
          // DEBUG: Log the raw cache value
          devLog('🔍 RAW EDITOR_CTX_CACHE:', localStorage.getItem('editor_ctx_cache'));
          
          // Transport: editable .md/.org, or read-only .txt/.pdf/.docx with non-empty content.
          const filenameLower = editorCtx?.filename?.toLowerCase() || '';
          const hasContent = !!(editorCtx?.content && editorCtx.content.trim().length > 0);
          const EDITABLE_EXTENSIONS = ['.md', '.org'];
          const READONLY_TEXT_EXTENSIONS = ['.txt', '.pdf', '.docx', '.srt', '.vtt'];
          const isEditableEditorState =
            editorCtx &&
            editorCtx.isEditable === true &&
            editorCtx.filename &&
            EDITABLE_EXTENSIONS.some((ext) => filenameLower.endsWith(ext)) &&
            hasContent;
          const isReadonlyEditorState =
            editorCtx &&
            editorCtx.isEditable === false &&
            editorCtx.filename &&
            READONLY_TEXT_EXTENSIONS.some((ext) => filenameLower.endsWith(ext)) &&
            hasContent;
          const hasValidEditorState = isEditableEditorState || isReadonlyEditorState;

          devLog('🔍 EDITOR STATE VALIDATION:', {
            hasValidEditorState,
            isEditableEditorState,
            isReadonlyEditorState,
            passedCheck1_editorCtxExists: !!editorCtx,
            passedCheck2_hasContent: hasContent,
            filename: editorCtx?.filename,
          });

          if (hasValidEditorState) {
            const language =
              editorCtx.language ||
              (filenameLower.endsWith('.org')
                ? 'org'
                : filenameLower.endsWith('.md')
                  ? 'markdown'
                  : filenameLower.endsWith('.txt')
                    ? 'plaintext'
                    : filenameLower.endsWith('.pdf')
                      ? 'pdf'
                      : filenameLower.endsWith('.docx')
                        ? 'docx'
                        : filenameLower.endsWith('.srt')
                          ? 'srt'
                          : filenameLower.endsWith('.vtt')
                            ? 'vtt'
                            : 'markdown');
            activeEditorPayload = {
              is_editable: editorCtx.isEditable === true,
              filename: editorCtx.filename,
              language: language,
              content: editorCtx.content,
              content_length: editorCtx.contentLength || editorCtx.content.length,
              frontmatter: editorCtx.frontmatter || {},
              cursor_offset: typeof editorCtx.cursorOffset === 'number' ? editorCtx.cursorOffset : -1,
              selection_start: typeof editorCtx.selectionStart === 'number' ? editorCtx.selectionStart : -1,
              selection_end: typeof editorCtx.selectionEnd === 'number' ? editorCtx.selectionEnd : -1,
              canonical_path: editorCtx.canonicalPath || null,
              document_id: editorCtx.documentId || null,
              folder_id: editorCtx.folderId || null,
            };
            devLog(
              '✅ Editor context — sending active_editor:',
              editorCtx.filename,
              editorCtx.isEditable === true ? '(editable)' : '(read-only)'
            );
          } else {
            if (!editorCtx) {
              devLog('🚫 NO EDITOR STATE IN CACHE - no editor tab is open');
            } else if (!editorCtx.filename) {
              devLog('🚫 NO FILENAME in editor cache');
            } else if (!hasContent) {
              devLog('🚫 NO EDITOR CONTENT - editor state exists but content is empty');
            } else if (editorCtx.isEditable === true) {
              devLog(
                '🚫 EDITABLE PATH NOT MET - expected .md/.org with content; filename:',
                editorCtx.filename
              );
            } else if (editorCtx.isEditable === false) {
              devLog(
                '🚫 READ-ONLY PATH NOT MET - expected .txt/.pdf/.docx with content; filename:',
                editorCtx.filename
              );
            } else {
              devLog('🚫 EDITOR CACHE isEditable not true/false - filename:', editorCtx.filename);
            }
            // CRITICAL: Explicitly set to null - never send stale data
            activeEditorPayload = null;
          }
        } catch (e) {
          console.error('❌ Error checking editor state:', e);
          // On any error, don't send active_editor
          activeEditorPayload = null;
        }
      } else {
        devLog('🚫 Editor preference is "ignore" - not sending editor context');
        activeEditorPayload = null;
      }

      let codeWorkspaceIdPayload = null;
      try {
        const cw = JSON.parse(localStorage.getItem('code_workspace_ctx_cache') || 'null');
        if (cw && cw.code_workspace_id) {
          const s = String(cw.code_workspace_id).trim();
          codeWorkspaceIdPayload = s || null;
        }
      } catch (e) {
        codeWorkspaceIdPayload = null;
      }

      let activeDataWorkspacePayload = null;
      if (dataWorkspacePreference !== 'ignore') {
        try {
          const dwCtx = JSON.parse(localStorage.getItem('data_workspace_ctx_cache') || 'null');
          const schema = dwCtx?.schema;
          const hasSchema = Array.isArray(schema) && schema.length > 0;
          if (dwCtx && dwCtx.table_id && hasSchema) {
            activeDataWorkspacePayload = {
              workspace_id: dwCtx.workspace_id || '',
              workspace_name: dwCtx.workspace_name || '',
              database_id: dwCtx.database_id || '',
              database_name: dwCtx.database_name || '',
              table_id: dwCtx.table_id,
              table_name: dwCtx.table_name || '',
              row_count: typeof dwCtx.row_count === 'number' ? dwCtx.row_count : 0,
              schema: schema,
              visible_rows: Array.isArray(dwCtx.visible_rows) ? dwCtx.visible_rows : [],
              visible_row_count: typeof dwCtx.visible_row_count === 'number' ? dwCtx.visible_row_count : 0
            };
          }
        } catch (e) {
          activeDataWorkspacePayload = null;
        }
      }

      let activeArtifactPayload = null;
      if (activeArtifact && activeArtifact.code) {
        activeArtifactPayload = {
          artifact_type: activeArtifact.artifact_type || activeArtifact.type || '',
          title: activeArtifact.title || '',
          code: activeArtifact.code,
          language: activeArtifact.language || null,
        };
      }

      const streamAbort = new AbortController();
      streamAbortControllerRef.current = streamAbort;

      const response = await fetch('/api/async/orchestrator/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        signal: streamAbort.signal,
        body: JSON.stringify({
          query: query,
          conversation_id: conversationId,
          session_id: sessionId,
          active_editor: activeEditorPayload,
          editor_preference: editorPreference,
          active_data_workspace: activeDataWorkspacePayload,
          data_workspace_preference: dataWorkspacePreference,
          active_artifact: activeArtifactPayload,
          user_chat_model: selectedModel || undefined,
          is_branch_resend: !!isBranchResend,
          branch_message_id: branchMessageId || undefined,
          code_workspace_id: codeWorkspaceIdPayload || undefined,
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedContent = '';

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                devLog('🌊 Stream data:', data);

                if (data.type === 'run_started' && data.run_id) {
                  updateCurrentActivityState({ currentJobId: data.run_id });
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === streamingMessage.id
                        ? {
                            ...msg,
                            jobId: data.run_id,
                            metadata: {
                              ...(msg.metadata || {}),
                              job_id: data.run_id,
                              streaming: true,
                            },
                          }
                        : msg
                    )
                  );
                } else if (data.type === 'cancelled') {
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === streamingMessage.id
                        ? {
                            ...msg,
                            content: msg.content?.trim()
                              ? `${msg.content}\n\n_(Stopped)_`
                              : '_(Stopped)_',
                            isStreaming: false,
                            isCancelled: true,
                          }
                        : msg
                    )
                  );
                  updateCurrentActivityState({ currentJobId: null, isLoading: false });
                  streamAbortControllerRef.current = null;
                  queryClient.invalidateQueries(['conversationMessages', conversationId]);
                  window.dispatchEvent(new CustomEvent('agentStreamComplete'));
                  break;
                } else if (data.type === 'title') {
                  // Handle conversation title update - update immediately in UI
                  if (data.message && conversationId) {
                    devLog('🔤 Received title update:', data.message);
                    // Optimistically update the conversation title in React Query cache
                    // Update both the specific conversation query and the conversations list
                    queryClient.setQueryData(['conversation', conversationId], (old) => {
                      if (!old?.conversation) return old;
                      return { ...old, conversation: { ...old.conversation, title: data.message } };
                    });
                    // Also update the conversations list cache to update sidebar immediately
                    queryClient.setQueryData(['conversations'], (old) => {
                      if (!old?.conversations) return old;
                      return {
                        ...old,
                        conversations: old.conversations.map(conv =>
                          conv.conversation_id === conversationId
                            ? { ...conv, title: data.message }
                            : conv
                        )
                      };
                    });
                    // Invalidate to ensure fresh data on next fetch
                    queryClient.invalidateQueries(['conversations']);
                    queryClient.invalidateQueries(['conversation', conversationId]);
                  }
                } else if (data.type === 'status') {
                  // Update message with status and capture agent_type if available
                  setMessages(prev => prev.map(msg => {
                    if (msg.id === streamingMessage.id) {
                      const currentMetadata = msg.metadata || {};
                      const updateData = { content: `${data.message}` };
                      const agentType = data.agent_type || data.agent || data.node;
                      if (agentType) {
                        updateData.metadata = { ...currentMetadata, agent_type: agentType };
                      }
                      if (data.agent_display_name) {
                        updateData.metadata = { ...(updateData.metadata || currentMetadata), agent_display_name: data.agent_display_name };
                      }
                      if (data.persona_ai_name) {
                        updateData.metadata = {
                          ...(updateData.metadata || currentMetadata),
                          persona_ai_name: data.persona_ai_name,
                        };
                      }
                      return { ...msg, ...updateData };
                    }
                    return msg;
                  }));
                } else if (data.type === 'tool_status') {
                  // ROOSEVELT'S TOOL STATUS STREAMING: Handle tool execution status updates
                  const statusIcon = data.status_type === 'tool_start' ? '🔧' :
                                   data.status_type === 'tool_complete' ? '✅' : '❌';
                  const statusMessage = `${statusIcon} ${data.message}`;
                  
                  setMessages(prev => prev.map(msg => {
                    if (msg.id === streamingMessage.id) {
                      const currentMetadata = msg.metadata || {};
                      const updateData = { content: statusMessage, isToolStatus: true };
                      // Check for agent info in various field names: agent_type, agent, node
                      const agentType = data.agent_type || data.agent || data.node;
                      if (agentType) {
                        updateData.metadata = { ...currentMetadata, agent_type: agentType };
                      }
                      return { ...msg, ...updateData };
                    }
                    return msg;
                  }));
                } else if (data.type === 'progress') {
                  // Handle progress messages that may include agent/node information
                  setMessages(prev => prev.map(msg => {
                    if (msg.id === streamingMessage.id) {
                      const currentMetadata = msg.metadata || {};
                      const updateData = {};
                      // Check for agent info in various field names: agent_type, agent, node
                      const agentType = data.agent_type || data.agent || data.node;
                      if (agentType) {
                        updateData.metadata = { ...currentMetadata, agent_type: agentType };
                      }
                      // Update with progress message if provided
                      if (data.message) {
                        updateData.content = data.message;
                      }
                      return { ...msg, ...updateData };
                    }
                    return msg;
                  }));
                } else if (data.type === 'content_stream') {
                  // Real-time streaming content
                  accumulatedContent += data.content;
                  
                  // Flash tab if this is the first content chunk and tab is hidden
                  if (!hasNotified && accumulatedContent.trim().length > 0) {
                    hasNotified = true;
                    tabNotificationManager.startFlashing('New message');
                  }
                  
                  setMessages(prev => prev.map(msg => 
                    msg.id === streamingMessage.id 
                      ? { ...msg, content: accumulatedContent, isStreaming: true }
                      : msg
                  ));
                } else if (data.type === 'content') {
                  accumulatedContent += data.content;
                  
                  // Flash tab if this is the first content chunk and tab is hidden
                  if (!hasNotified && accumulatedContent.trim().length > 0) {
                    hasNotified = true;
                    tabNotificationManager.startFlashing('New message');
                  }
                  
                  setMessages(prev => prev.map(msg => {
                    if (msg.id === streamingMessage.id) {
                      const updated = { ...msg, content: accumulatedContent, isStreaming: true };
                      if (data.agent_display_name) {
                        updated.metadata = { ...(msg.metadata || {}), agent_display_name: data.agent_display_name };
                      }
                      if (data.persona_ai_name) {
                        updated.metadata = { ...(updated.metadata || msg.metadata || {}), persona_ai_name: data.persona_ai_name };
                      }
                      return updated;
                    }
                    return msg;
                  }));
                } else if (data.type === 'citations') {
                  // **ROOSEVELT'S CITATION CAVALRY**: Capture citations from research agent!
                  devLog('🔗 Citations received:', data.citations);
                  devLog('🔗 streamingMessage.id:', streamingMessage.id);
                  devLog('🔗 Current messages count:', messages.length);
                  const citations = Array.isArray(data.citations) ? data.citations : [];
                  setMessages(prev => {
                    devLog('🔗 Updating messages, looking for id:', streamingMessage.id);
                    const updated = prev.map(msg => {
                      if (msg.id === streamingMessage.id) {
                        devLog('✅ FOUND streaming message, adding citations!');
                        return { 
                          ...msg, 
                          citations: citations,
                          metadata: {
                            ...(msg.metadata || {}),
                            citations: citations
                          }
                        };
                      }
                      return msg;
                    });
                    devLog('🔗 Messages after citation update:', updated.map(m => ({ id: m.id, hasCitations: !!m.citations, citationCount: m.citations?.length })));
                    return updated;
                  });
                  devLog(`✅ Added ${citations.length} citations to streaming message`);
                } else if (data.type === 'permission_request') {
                  // ROOSEVELT'S HITL: Permission request detected
                  devLog('🛡️ Permission request received:', data);
                  
                  setMessages(prev => prev.map(msg => 
                    msg.id === streamingMessage.id 
                      ? { 
                          ...msg, 
                          content: data.content,
                          isStreaming: false,
                          isPermissionRequest: true,  // Tag for special handling
                          requiresApproval: data.requires_approval,
                          conversationId: data.conversation_id,
                          timestamp: new Date().toISOString()
                        }
                      : msg
                  ));
                  
                  devLog('✅ Permission request message updated');
                  
                } else if (data.type === 'notification') {
                  // Signal Corps: Spontaneous notification/alert
                  devLog('📢 Notification received:', data);
                  
                  const notification = {
                    id: `note_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    role: 'system',
                    type: 'notification',
                    severity: data.severity || 'info', // info, success, warning, error
                    content: data.message,
                    ephemeral: true, // Mark as ephemeral - not saved to long-term history
                    timestamp: data.timestamp || new Date().toISOString(),
                    agent_name: data.agent || data.agent_name || 'system'
                  };
                  
                  setMessages(prev => [...prev, notification]);
                  
                  // Show browser notification for warnings, errors, or if explicitly requested
                  // Check metadata for browser_notify flag (allows agents to explicitly request browser notifications)
                  const browserNotify = data.browser_notify === true || 
                                       data.metadata?.browser_notify === 'true' ||
                                       data.metadata?.browser_notify === true;
                  
                  browserNotificationManager.showNotification({
                    message: data.message,
                    severity: data.severity || 'info',
                    agent: data.agent || data.agent_name || 'system',
                    timestamp: data.timestamp || new Date().toISOString(),
                    browser_notify: browserNotify
                  }).catch(err => {
                    console.error('Error showing browser notification:', err);
                  });
                  
                  // Auto-remove temporary notifications after 10 seconds
                  if (data.temporary) {
                    setTimeout(() => {
                      setMessages(prev => prev.filter(m => m.id !== notification.id));
                    }, 10000);
                  }
                  
                  devLog('✅ Notification added to messages');
                  
                } else if (data.type === 'complete_hitl') {
                  // HITL completion - awaiting user permission response
                  devLog('🛡️ HITL completion - awaiting permission response');
                  
                  // Don't clear job ID yet - we're waiting for user input
                  updateCurrentActivityState({ isLoading: false });
                  
                  // Refresh conversations to ensure state is saved
                  queryClient.invalidateQueries(['conversations']);
                  queryClient.invalidateQueries(['conversation', conversationId]);
                  break;
                  
                } else if (data.type === 'complete') {
                  // Normal streaming complete
                  setMessages(prev => {
                    const updated = prev.map(msg => 
                      msg.id === streamingMessage.id 
                        ? { 
                            ...msg, 
                            content: data.final_content || accumulatedContent || msg.content,
                            isStreaming: false,
                            timestamp: new Date().toISOString(),
                            // **ROOSEVELT'S CITATION PRESERVATION**: Don't overwrite citations!
                            // Citations and metadata are already in msg from previous citation event
                            metadata: {
                              ...(msg.metadata || {}),
                              ...(data.metadata || {})
                            }
                          }
                        : msg
                    );
                    devLog('✅ Streaming completed - final message state:', 
                      updated.find(m => m.id === streamingMessage.id)?.citations ? 
                        `HAS ${updated.find(m => m.id === streamingMessage.id)?.citations?.length} CITATIONS` : 
                        'NO CITATIONS'
                    );
                    return updated;
                  });

                  if (data.metadata?.artifact || data.metadata?.artifacts) {
                    let parsed = null;
                    const rawList = data.metadata.artifacts;
                    if (rawList) {
                      let list = rawList;
                      if (typeof list === 'string') {
                        try {
                          list = JSON.parse(list);
                        } catch {
                          list = null;
                        }
                      }
                      if (Array.isArray(list) && list.length > 0 && list[0]?.artifact_type) {
                        parsed = list[0];
                      }
                    }
                    if (!parsed && data.metadata?.artifact) {
                      parsed = data.metadata.artifact;
                      if (typeof parsed === 'string') {
                        try {
                          parsed = JSON.parse(parsed);
                        } catch {
                          parsed = null;
                        }
                      }
                    }
                    if (parsed && typeof parsed === 'object' && parsed.artifact_type) {
                      dispatchArtifactDrawer({ type: 'open', payload: parsed });
                    }
                  }
                  
                  devLog('✅ Streaming completed successfully');
                  
                  updateCurrentActivityState({ currentJobId: null });
                  streamAbortControllerRef.current = null;
                  activeStreamMessageIdRef.current = null;
                  
                  // Refresh conversations - title may have been updated from "New Conversation"
                  // Force a refetch to ensure we get the latest title
                  queryClient.invalidateQueries(['conversations']);
                  queryClient.invalidateQueries(['conversation', conversationId]);
                  
                  // CRITICAL: Also invalidate and refetch messages to ensure the saved message appears
                  // This fixes the issue where messages don't appear until page refresh
                  queryClient.invalidateQueries(['conversationMessages', conversationId]);
                  queryClient.refetchQueries(['conversationMessages', conversationId]);
                  
                  // Also refetch the conversation list to ensure title updates are visible
                  queryClient.refetchQueries(['conversations']);
                  window.dispatchEvent(new CustomEvent('agentStreamComplete'));
                  break;
                } else if (data.type === 'done') {
                  streamAbortControllerRef.current = null;
                  activeStreamMessageIdRef.current = null;
                  if (data.active_line_id) {
                    setActiveLineRouting({
                      id: data.active_line_id,
                      name: data.active_line_name || '',
                    });
                  }
                  // Streaming complete - check if conversation was updated (title generation)
                  if (data.conversation_updated) {
                    devLog('🔄 Conversation updated - refreshing to get new title');
                    // Invalidate and refetch conversations to get updated title
                    queryClient.invalidateQueries(['conversations']);
                    queryClient.invalidateQueries(['conversation', conversationId]);
                    queryClient.refetchQueries(['conversations']);
                  }
                  
                  // CRITICAL: Always invalidate and refetch messages when streaming is done
                  // This ensures the saved message appears even if the 'complete' event was missed
                  queryClient.invalidateQueries(['conversationMessages', conversationId]);
                  queryClient.refetchQueries(['conversationMessages', conversationId]);
                  window.dispatchEvent(new CustomEvent('agentStreamComplete'));
                  break;
                } else if (data.type === 'error') {
                  throw new Error(data.message || 'Streaming error');
                }
              } catch (parseError) {
                console.warn('⚠️ Failed to parse stream data:', line, parseError);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
        streamAbortControllerRef.current = null;
      }
      
    } catch (error) {
      streamAbortControllerRef.current = null;
      activeStreamMessageIdRef.current = null;
      if (error?.name === 'AbortError') {
        updateCurrentActivityState({ currentJobId: null, isLoading: false });
        return;
      }
      console.error('❌ Streaming failed:', error);
      
      // Update message to show error
      updateCurrentActivityState({ currentJobId: null });
      activeStreamMessageIdRef.current = null;
      setMessages(prev => prev.map(msg => 
        msg.isStreaming 
          ? { 
              ...msg, 
              content: `❌ Streaming failed: ${error.message}`,
              isStreaming: false,
              isError: true,
              timestamp: new Date().toISOString()
            }
          : msg
      ));
    }
  };

  const editAndResendMessage = async (messageId, newContent) => {
    const text = (newContent || '').trim();
    if (!currentConversationId || !text || !messageId) return;
    updateCurrentActivityState({ isLoading: true });
    try {
      const result = await apiService.editAndBranch(currentConversationId, messageId, text);
      const bid = result?.message?.message_id;
      await refetchMessages();
      if (bid) {
        await handleStreamingResponse(text, currentConversationId, sessionId, {
          isBranchResend: true,
          branchMessageId: bid,
        });
      }
    } catch (error) {
      console.error('editAndResendMessage failed:', error);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: 'system',
          type: 'system',
          content: `Could not start branch: ${error.message}`,
          isError: true,
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      updateCurrentActivityState({ isLoading: false });
    }
  };

  const switchBranch = async (messageId, direction) => {
    if (!currentConversationId) return;
    const nextId = getNextSibling(messageTree, messageId, direction);
    if (!nextId) return;
    try {
      const res = await apiService.switchBranch(currentConversationId, nextId);
      setCurrentNodeId(res.current_node_message_id);
      if (res.active_path && res.active_path.length) {
        setMessages(res.active_path.map(normalizeServerMessage));
      } else {
        const tree = buildMessageTree(allMessages);
        const path = getActivePath(tree, res.current_node_message_id);
        setMessages(path.map(normalizeServerMessage));
      }
      await refetchMessages();
    } catch (error) {
      console.error('switchBranch failed:', error);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setAllMessages([]);
    setCurrentNodeId(null);
    messagesConversationIdRef.current = null; // Reset ref when clearing chat
    setQuery('');
    // Activity state is conversation-scoped, so clearing conversation clears its state
    if (currentConversationId) {
      updateCurrentActivityState({ currentJobId: null, isLoading: false });
    }
  };

  const cancelCurrentJob = async () => {
    const runId = currentJobId;
    const streamMid = activeStreamMessageIdRef.current;
    const isUuidRun =
      typeof runId === 'string' &&
      /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
        runId.trim()
      );

    try {
      if (isUuidRun) {
        devLog('Stopping stream run_id:', runId);
        await apiService.cancelUnifiedJob(runId.trim());
      }
    } catch (error) {
      console.error('Failed to cancel stream run:', error);
    }

    streamAbortControllerRef.current?.abort();
    streamAbortControllerRef.current = null;

    if (streamMid != null) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === streamMid && msg.isStreaming
            ? {
                ...msg,
                content: (msg.content || '').trim()
                  ? `${msg.content}\n\n_(Stopped)_`
                  : '_(Stopped)_',
                isCancelled: true,
                isStreaming: false,
              }
            : msg
        )
      );
    }

    updateCurrentActivityState({ currentJobId: null, isLoading: false });
    activeStreamMessageIdRef.current = null;
  };

  chatSidebarActionsRef.current = {
    toggleSidebar,
    selectConversation,
    createNewConversation,
    sendMessage,
    clearChat,
    editAndResendMessage,
    switchBranch,
    cancelCurrentJob,
  };

  const stableToggleSidebar = useCallback(() => {
    chatSidebarActionsRef.current.toggleSidebar();
  }, []);
  const stableSelectConversation = useCallback((conversationId) => {
    chatSidebarActionsRef.current.selectConversation(conversationId);
  }, []);
  const stableCreateNewConversation = useCallback(() => {
    chatSidebarActionsRef.current.createNewConversation();
  }, []);
  const stableSendMessage = useCallback((executionMode, overrideQuery) => {
    return chatSidebarActionsRef.current.sendMessage(executionMode, overrideQuery);
  }, []);
  const stableClearChat = useCallback(() => {
    chatSidebarActionsRef.current.clearChat();
  }, []);
  const stableEditAndResendMessage = useCallback((messageId, newContent) => {
    return chatSidebarActionsRef.current.editAndResendMessage(messageId, newContent);
  }, []);
  const stableSwitchBranch = useCallback((messageId, direction) => {
    return chatSidebarActionsRef.current.switchBranch(messageId, direction);
  }, []);
  const stableCancelCurrentJob = useCallback(() => {
    return chatSidebarActionsRef.current.cancelCurrentJob();
  }, []);

  const value = useMemo(
    () => ({
      // State
      isCollapsed,
      sidebarWidth,
      setSidebarWidth, // Export setSidebarWidth for resize functionality
      isFullWidth,
      setIsFullWidth,
      isResizing,
      setIsResizing,
      currentConversationId,
      messages,
      allMessages,
      currentNodeId,
      messageTree,
      setMessages,
      query,
      setQuery,
      replyToMessage,
      setReplyToMessage,
      isLoading,
      selectedModel,
      setSelectedModel,
      backgroundJobService,
      sessionId,
      executingPlans,
      currentJobId, // Add current job ID for cancellation

      // Actions (stable wrappers delegate to latest implementations)
      toggleSidebar: stableToggleSidebar,
      selectConversation: stableSelectConversation,
      createNewConversation: stableCreateNewConversation,
      sendMessage: stableSendMessage,
      clearChat: stableClearChat,
      editAndResendMessage: stableEditAndResendMessage,
      switchBranch: stableSwitchBranch,

      cancelCurrentJob: stableCancelCurrentJob, // Add cancellation function

      // **ROOSEVELT**: Editor preference (active = sent to backend, user = checkbox state)
      editorPreference, // Active preference sent to backend (context-aware)
      setEditorPreference: setUserEditorPreference, // UI checkboxes modify user preference
      handleEditorPreferenceChange, // Handler that saves to conversation metadata

      dataWorkspacePreference,
      setDataWorkspacePreference,

      // Conversation preference management
      updateConversationPreference, // Save preferences to conversation metadata

      activeLineRouting,

      activeArtifact,
      setActiveArtifact,
      artifactHistory,
      openArtifact,
      revertArtifact,
      artifactCollapsed,
      setArtifactCollapsed,

      /** True while GET /conversations/:id/messages is in flight (for empty-state UI). */
      conversationMessagesLoading: messagesLoading,
    }),
    [
      isCollapsed,
      sidebarWidth,
      isFullWidth,
      isResizing,
      currentConversationId,
      messages,
      allMessages,
      currentNodeId,
      messageTree,
      query,
      replyToMessage,
      isLoading,
      selectedModel,
      backgroundJobService,
      sessionId,
      executingPlans,
      currentJobId,
      editorPreference,
      handleEditorPreferenceChange,
      dataWorkspacePreference,
      setDataWorkspacePreference,
      updateConversationPreference,
      activeLineRouting,
      activeArtifact,
      artifactHistory,
      openArtifact,
      revertArtifact,
      artifactCollapsed,
      setArtifactCollapsed,
      messagesLoading,
      stableToggleSidebar,
      stableSelectConversation,
      stableCreateNewConversation,
      stableSendMessage,
      stableClearChat,
      stableEditAndResendMessage,
      stableSwitchBranch,
      stableCancelCurrentJob,
      setActiveArtifact,
    ]
  );

  return (
    <ChatSidebarContext.Provider value={value}>
      {children}
    </ChatSidebarContext.Provider>
  );
}; 