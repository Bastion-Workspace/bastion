import React, { createContext, useContext, useState, useEffect, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useLocation } from 'react-router-dom';
import apiService from '../services/apiService';
import BackgroundJobService from '../services/backgroundJobService';
import tabNotificationManager from '../utils/tabNotification';
import browserNotificationManager from '../utils/browserNotification';
import { documentDiffStore } from '../services/documentDiffStore';
import { createAgentStatusWebSocket } from '../utils/agentStatusTypes';

// Format agent type to display name
const formatAgentName = (agentType) => {
  if (!agentType) return 'AI';
  const formatted = agentType
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
  return formatted;
};

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
  // Note: EditorProvider is a child of ChatSidebarProvider, so we can't use useEditor() here
  // We'll check localStorage directly with strict validation instead
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(420);
  const [isFullWidth, setIsFullWidth] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  // ROOSEVELT: Load current conversation from localStorage for session persistence
  const [currentConversationId, setCurrentConversationId] = useState(() => {
    try {
      const saved = localStorage.getItem('chatSidebarCurrentConversation');
      const conversationId = saved && saved !== 'null' ? saved : null;
      console.log('💾 Page refresh - loading conversation from localStorage:', conversationId);
      if (conversationId) {
        console.log('🔄 Will restore conversation automatically on page load');
      } else {
        console.log('🆕 No saved conversation - starting fresh');
      }
      return conversationId;
    } catch (error) {
      console.error('Failed to load current conversation from localStorage:', error);
      return null;
    }
  });
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const [replyToMessage, setReplyToMessage] = useState(null); // Message being replied to
  const [selectedModel, setSelectedModel] = useState(() => {
    try {
      const saved = localStorage.getItem('chatSidebarSelectedModel');
      return saved && saved !== 'null' ? saved : '';
    } catch {
      return '';
    }
  });
  const [backgroundJobService, setBackgroundJobService] = useState(null);
  /** When set, chat follow-ups route to this agent line (CEO) until @auto */
  const [activeLineRouting, setActiveLineRouting] = useState(null);
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  // LangGraph is the only system - no toggles needed
  const useLangGraphSystem = true; // Always use LangGraph
  const messagesConversationIdRef = React.useRef(null); // Track which conversation the current messages belong to
  const lastCreatedConversationIdRef = React.useRef(null); // Skip redundant refetch after creating new conversation
  const isLoadingFromMetadataRef = React.useRef(false); // Track when we're loading preferences from metadata to prevent save loop
  
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
  const [userEditorPreference, setUserEditorPreference] = useState(() => {
    try { return localStorage.getItem('userEditorPreference') || 'prefer'; } catch { return 'prefer'; }
  });
  
  // Active preference: what actually gets sent to the backend (context-aware)
  // Initialize based on current location - if on documents page, use user preference, otherwise 'ignore'
  const [editorPreference, setEditorPreference] = useState(() => {
    // Use location from useLocation hook (available in component)
    try {
      const pathname = typeof window !== 'undefined' ? window.location.pathname : '';
      const onDocumentsPage = pathname.startsWith('/documents');
      return onDocumentsPage ? (localStorage.getItem('userEditorPreference') || 'prefer') : 'ignore';
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
        console.log('On documents page - applying user editor preference:', userEditorPreference);
        setEditorPreference(userEditorPreference);
      }
    } else {
      if (editorPreference !== 'ignore') {
        console.log('Not on documents page - disabling editor preference');
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

  // WebSocket: line sub-agent messages during CEO run (out-of-band from SSE)
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
      } catch (_) { /* ignore */ }
    };
  }, [currentConversationId]);

  // Preference update function for saving to conversation metadata
  const updateConversationPreference = React.useCallback(async (key, value) => {
    if (!currentConversationId) {
      console.log('No conversation - only updating global preference');
      return;
    }
    
    try {
      // Save to conversation metadata
      await apiService.patch(`/api/conversations/${currentConversationId}/metadata`, {
        metadata: {
          [key]: value
        }
      });
      
      console.log(`Saved ${key} to conversation ${currentConversationId}:`, value);
      
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
    localStorage.setItem('userEditorPreference', newPreference);
    
    // Update active preference (context-aware)
    if (location.pathname.startsWith('/documents')) {
      setEditorPreference(newPreference);
      
      // Save to current conversation
      if (currentConversationId) {
        await updateConversationPreference('editor_preference', newPreference);
      }
    }
  }, [currentConversationId, updateConversationPreference, location.pathname]);

  // Initialize background job service
  useEffect(() => {
    console.log('🔄 Initializing background job service...');
    const service = new BackgroundJobService(apiService);
    console.log('✅ Background job service created:', service);
    setBackgroundJobService(service);
    
    return () => {
      if (service) {
        console.log('🧹 Cleaning up background job service...');
        service.disconnectAll();
      }
    };
  }, []);

  // CRITICAL: Log conversation ID changes for debugging
  useEffect(() => {
    console.log('🔄 ChatSidebarContext: currentConversationId changed to:', currentConversationId);
  }, [currentConversationId]);

  // Load sidebar preferences and model selection from localStorage
  useEffect(() => {
    const savedCollapsed = localStorage.getItem('chatSidebarCollapsed');
    const savedWidth = localStorage.getItem('chatSidebarWidth');
    const savedModel = localStorage.getItem('chatSidebarSelectedModel');
    const savedFullWidth = localStorage.getItem('chatSidebarFullWidth');
    
    if (savedCollapsed !== null) {
      setIsCollapsed(JSON.parse(savedCollapsed));
    }
    
    if (savedWidth !== null) {
      setSidebarWidth(JSON.parse(savedWidth));
    }
    
    if (savedModel !== null) {
      setSelectedModel(savedModel);
      // Immediately notify backend of the saved model selection
      console.log('🔄 App loaded - notifying backend of saved model:', savedModel);
      apiService.selectModel(savedModel).catch(error => {
        console.warn('⚠️ Failed to notify backend of saved model on app load:', error);
      });
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

  // **ROOSEVELT**: Save USER preferences to localStorage (these persist across navigation)
  useEffect(() => {
    try { localStorage.setItem('userEditorPreference', userEditorPreference); } catch {}
  }, [userEditorPreference]);

  // Save selected model to localStorage and conversation metadata
  useEffect(() => {
    // Skip saving if we're currently loading from metadata (prevents circular updates)
    if (isLoadingFromMetadataRef.current) {
      return;
    }
    
    if (selectedModel) {
      localStorage.setItem('chatSidebarSelectedModel', selectedModel);
      
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
    }
  }, [selectedModel, currentConversationId, updateConversationPreference]);

  // ROOSEVELT: Save current conversation to localStorage for session persistence
  useEffect(() => {
    try {
      if (currentConversationId) {
        localStorage.setItem('chatSidebarCurrentConversation', currentConversationId);
        console.log('💾 Persisted conversation to localStorage:', currentConversationId);
      } else {
        localStorage.removeItem('chatSidebarCurrentConversation');
        console.log('💾 Cleared conversation from localStorage');
      }
    } catch (error) {
      console.error('Failed to persist current conversation to localStorage:', error);
    }
  }, [currentConversationId]);

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
        console.log('✅ ChatSidebarContext: Conversation data loaded:', {
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
    console.log('🔄 Conversation switched, refetching to get latest metadata:', currentConversationId);
    refetchConversation();
  }, [currentConversationId, refetchConversation]);

  // Load conversation-specific preferences from metadata
  useEffect(() => {
    if (!currentConversationId) {
      setActiveLineRouting(null);
      // No conversation - use global preferences
      const globalModel = localStorage.getItem('chatSidebarSelectedModel');
      if (globalModel) setSelectedModel(globalModel);
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
        console.log('🔄 Loading conversation model preference:', metadata.user_chat_model, 'for conversation:', currentConversationId);
        setSelectedModel(metadata.user_chat_model);
      } else {
        // Fall back to global preference
        const globalModel = localStorage.getItem('chatSidebarSelectedModel');
        if (globalModel) {
          console.log('🔄 No conversation model preference, using global:', globalModel);
          setSelectedModel(globalModel);
        }
      }
      
      // Load editor preference (only on documents page)
      if (metadata.editor_preference && location.pathname.startsWith('/documents')) {
        console.log('🔄 Loading conversation editor preference:', metadata.editor_preference);
        setEditorPreference(metadata.editor_preference);
      } else {
        // Fall back to global user preference
        const globalEditorPref = localStorage.getItem('userEditorPreference') || 'prefer';
        if (location.pathname.startsWith('/documents')) {
          setEditorPreference(globalEditorPref);
        }
      }
    } else if (conversationData) {
      setActiveLineRouting(null);
      // Conversation loaded but no metadata yet - use global preferences
      console.log('🔄 Conversation loaded but no metadata, using global preferences');
      const globalModel = localStorage.getItem('chatSidebarSelectedModel');
      if (globalModel) setSelectedModel(globalModel);
      
      const globalEditorPref = localStorage.getItem('userEditorPreference') || 'prefer';
      if (location.pathname.startsWith('/documents')) {
        setEditorPreference(globalEditorPref);
      }
    }
    
    // Reset the flag after a short delay to allow state updates to complete
    setTimeout(() => {
      isLoadingFromMetadataRef.current = false;
    }, 100);
  }, [conversationData, currentConversationId, location.pathname]);

  // PRIORITY: Load messages using unified chat service
  const { data: messagesData, isLoading: messagesLoading, refetch: refetchMessages } = useQuery(
    ['conversationMessages', currentConversationId],
    () => currentConversationId
      ? apiService.getConversationMessages(currentConversationId, 0, 100, lastCreatedConversationIdRef.current === currentConversationId)
      : null,
    {
      enabled: !!currentConversationId,
      refetchOnWindowFocus: false,
      staleTime: 300000, // 5 minutes
      onSuccess: (data) => {
        console.log('✅ ChatSidebarContext: Messages loaded:', {
          conversationId: currentConversationId,
          messageCount: data?.messages?.length || 0,
          hasMore: data?.has_more || false
        });
        if (data?.messages) {
          // Normalize message format for consistent frontend handling
          const normalizedMessages = data.messages.map(message => ({
            id: message.message_id || message.id,
            message_id: message.message_id,
            role: message.message_type || message.role, // ✅ Fix: API returns message_type, frontend expects role
            type: message.message_type || message.role, // Add type field for components that expect it
            content: message.content,
            timestamp: message.created_at || message.timestamp,
            created_at: message.created_at,
            sequence_number: message.sequence_number,
            citations: message.citations || [],
            metadata: message.metadata_json || message.metadata || {},
            // ✅ CRITICAL FIX: Extract editor_operations from metadata to top level
            editor_operations: (message.metadata_json || message.metadata || {})?.editor_operations || message.editor_operations || [],
            editor_document_id: (message.metadata_json || message.metadata || {})?.editor_document_id ?? message.editor_document_id,
            editor_filename: (message.metadata_json || message.metadata || {})?.editor_filename ?? message.editor_filename,
            // Preserve any other fields
            ...message
          }));
          
          // CRITICAL FIX: Always load messages when conversation changes, only prevent overwrite for same conversation
          setMessages(prevMessages => {
            // Check if we're switching to a different conversation (or initial load)
            const isConversationSwitch = messagesConversationIdRef.current !== currentConversationId;
            
            // If switching conversations or initial load, always load database messages
            if (isConversationSwitch) {
              console.log('✅ Loading messages for conversation:', {
                previousConversationId: messagesConversationIdRef.current,
                newConversationId: currentConversationId,
                messageCount: normalizedMessages.length,
                isInitialLoad: messagesConversationIdRef.current === null
              });
              messagesConversationIdRef.current = currentConversationId;
              return normalizedMessages;
            }
            
            // Same conversation: only prevent overwrite if we have truly unsaved messages (pending/streaming, not system)
            const hasUnsavedMessages = prevMessages.some(msg => 
              msg.isPending || msg.isStreaming
            );
            
            // Only prevent overwrite if we have unsaved messages AND database has fewer messages
            // This protects against losing messages during active streaming/pending operations
            if (hasUnsavedMessages && normalizedMessages.length < prevMessages.length) {
              console.log('🔄 Keeping unsaved messages to prevent loss during active conversation:', {
                localCount: prevMessages.length,
                dbCount: normalizedMessages.length,
                hasUnsaved: hasUnsavedMessages
              });
              return prevMessages;
            }
            
            // For same conversation with no unsaved messages, or when database has more messages, use database
            console.log('✅ Loading messages from database:', {
              messageCount: normalizedMessages.length,
              conversationId: currentConversationId,
              isSameConversation: !isConversationSwitch
            });
            return normalizedMessages;
          });
        }
      },
      onError: (error) => {
        console.error('❌ ChatSidebarContext: Failed to load messages:', error);
      }
    }
  );

  // Update messages when conversation data changes (fallback)
  useEffect(() => {
    if (conversationData?.messages && !messagesData) {
      console.log('🔄 ChatSidebarContext: Using fallback conversation messages');
      // Normalize message format for consistent frontend handling
      const normalizedMessages = conversationData.messages.map(message => ({
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
    console.log('🔄 Background job progress:', jobData);
    
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
    console.log('✅ Background job completed:', jobData);
    console.log('🔍 Job details:', {
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
      console.log('🔄 Refreshing messages from database after job completion');
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
    
    console.log('✅ Background job completion handling completed');
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
    setIsCollapsed(!isCollapsed);
  };

  const selectConversation = (conversationId) => {
    console.log('🔄 ChatSidebarContext: Selecting conversation:', conversationId);
    
    // Clear current state to prevent cross-conversation contamination
    setMessages([]);
    setQuery('');
    // Activity state is now automatically isolated by conversationId
    // No need to manually clear - each conversation has its own state
    
    // Reset the messages conversation ID ref to trigger fresh load
    messagesConversationIdRef.current = null;
    
    // Update background job service with new conversation ID
    if (backgroundJobService) {
      console.log('🔄 ChatSidebarContext: Updating background job service conversation ID');
      backgroundJobService.setCurrentConversationId(conversationId);
      backgroundJobService.disconnectAll(); // Clear any existing connections
      backgroundJobService.clearCompletedJobs();
    }
    
    // Set the new conversation ID (this will trigger React Query to load messages)
    setCurrentConversationId(conversationId);
    
    // Invalidate queries to ensure fresh data
    queryClient.invalidateQueries(['conversation', conversationId]);
    queryClient.invalidateQueries(['conversationMessages', conversationId]);
    
    console.log('✅ ChatSidebarContext: Conversation selection completed for:', conversationId);
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
    
    console.log('🔄 sendMessage called with:', { query: actualQuery, overrideQuery: !!overrideQuery, backgroundJobService: !!backgroundJobService, executionMode, hasReply: !!replyToMessage });
    
    if (!actualQuery || !backgroundJobService) {
      console.log('❌ sendMessage early return:', { hasQuery: !!actualQuery, hasService: !!backgroundJobService });
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
        console.log('🛡️ Detected HITL permission response, continuing LangGraph flow:', currentQuery);
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
        console.log('🔍 Full conversation response:', newConversation);
        console.log('🔍 Response keys:', Object.keys(newConversation));
        conversationId = newConversation.conversation.conversation_id; // Access conversation_id through the conversation field
        setCurrentConversationId(conversationId);
        queryClient.invalidateQueries(['conversations']);
        console.log('✅ Created new conversation:', conversationId);
      } catch (error) {
        console.error('❌ Failed to create conversation:', error);
        setQuery(currentQuery); // Restore query on failure
        return;
      }
    }

    // Add user message to UI immediately
    setMessages(prev => [...prev, userMessage]);
    updateCurrentActivityState({ isLoading: true });

    // NEWS CHAT QUICK PATH: If user asked for headlines, fetch and present as cards
    try {
      const lowerQ = currentQuery.toLowerCase();
      if (lowerQ.includes('latest headlines') || lowerQ.startsWith('headlines') || lowerQ.includes('news headlines')) {
        // Ensure conversation exists so the quick-path exchange is persisted
        if (!conversationId) {
          try {
            const newConversation = await apiService.createConversation({ initial_message: currentQuery });
            conversationId = newConversation.conversation.conversation_id;
            setCurrentConversationId(conversationId);
            queryClient.invalidateQueries(['conversations']);
          } catch (error) {
            console.error('❌ Failed to create conversation for headlines quick path:', error);
            // Continue without persistence
          }
        } else {
          // Persist the user message to the conversation if API supports it
          try {
            await apiService.addMessageToConversation(conversationId, { content: currentQuery, role: 'user' });
          } catch (e) {
            // Non-fatal
          }
        }

        const resp = await apiService.get('/api/news/headlines');
        const headlines = (resp && resp.headlines) || [];
        const newsMsg = {
          id: Date.now() + 2,
          role: 'assistant',
          type: 'assistant',
          content: '**Latest Headlines**',
          news_results: headlines,
          timestamp: new Date().toISOString(),
        };

        // Persist assistant message if conversation exists
        if (conversationId) {
          try {
            await apiService.addMessageToConversation(conversationId, {
              content: newsMsg.content,
              role: 'assistant',
              metadata: { news_results: headlines }
            });
          } catch (e) {
            // Non-fatal
          }
        }

        setMessages(prev => [...prev, newsMsg]);
        updateCurrentActivityState({ isLoading: false });
        return;
      }
    } catch (e) {
      // fall through to normal flow
    }

    // LangGraph is the only system - always use it
      // Use LangGraph system
      try {
        console.log('🔄 Using LangGraph system');
        
        // User message already added above, no need to add again
        
        // 🌊 STREAMING-FIRST POLICY: Stream everything for optimal UX!
        console.log('🌊 Using streaming for ALL queries');
        
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
  const handleStreamingResponse = async (query, conversationId, sessionId) => {
    console.log('🌊 Starting streaming response for:', query);
    
    try {
      // ROOSEVELT'S CANCEL BUTTON FIX: Create streaming job ID for cancel functionality
      const streamingJobId = `streaming_${Date.now()}`;
      updateCurrentActivityState({ currentJobId: streamingJobId });
      
      // Add streaming message placeholder
      const streamingMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        type: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        isStreaming: true,
        jobId: streamingJobId,
        metadata: {
          streaming: true,
          job_id: streamingJobId,
          agent_type: null
        }
      };
      
      setMessages(prev => [...prev, streamingMessage]);
      
      // Track if we've notified for this message to avoid multiple notifications during streaming
      let hasNotified = false;
      
      // Create EventSource for Server-Sent Events  
      const token = localStorage.getItem('auth_token'); // Match apiService token key
      console.log('🔑 Using auth_token for streaming:', token ? 'TOKEN_PRESENT' : 'NO_TOKEN');
      
      // Attach active_editor when preference is not ignore and cache shows an open editable .md/.org with content.
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
          console.log('🔍 EDITOR_CTX_CACHE DEBUG:', {
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
          console.log('🔍 RAW EDITOR_CTX_CACHE:', localStorage.getItem('editor_ctx_cache'));
          
          // Transport checks only: open editable .md/.org with content. How to use it is Agent Factory / orchestrator.
          const filenameLower = editorCtx?.filename?.toLowerCase() || '';
          const hasValidEditorState = editorCtx && 
                                      editorCtx.isEditable === true && 
                                      editorCtx.filename && 
                                      (filenameLower.endsWith('.md') || filenameLower.endsWith('.org')) &&
                                      editorCtx.content &&
                                      editorCtx.content.trim().length > 0;
          
          console.log('🔍 EDITOR STATE VALIDATION:', {
            hasValidEditorState,
            passedCheck1_editorCtxExists: !!editorCtx,
            passedCheck2_isEditableTrue: editorCtx?.isEditable === true,
            passedCheck3_filenameEndsMdOrOrg: filenameLower.endsWith('.md') || filenameLower.endsWith('.org'),
            passedCheck4_hasContent: !!(editorCtx?.content && editorCtx.content.trim().length > 0)
          });
          
          if (hasValidEditorState) {
            const language = filenameLower.endsWith('.org') ? 'org' : (editorCtx.language || 'markdown');
            activeEditorPayload = {
              is_editable: true,
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
            console.log('✅ Editor tab is open and editable - sending active_editor:', editorCtx.filename);
          } else {
            // Editor is NOT open or NOT editable - be very explicit about why
            if (!editorCtx) {
              console.log('🚫 NO EDITOR STATE IN CACHE - no editor tab is open');
            } else if (editorCtx.isEditable !== true) {
              console.log('🚫 EDITOR NOT EDITABLE (isEditable=' + editorCtx.isEditable + ') - viewing PDF or document, not editing');
            } else if (!editorCtx.filename || (!editorCtx.filename.toLowerCase().endsWith('.md') && !editorCtx.filename.toLowerCase().endsWith('.org'))) {
              console.log('🚫 NO VALID MARKDOWN OR ORG FILE - filename:', editorCtx.filename);
            } else if (!editorCtx.content || !editorCtx.content.trim()) {
              console.log('🚫 NO EDITOR CONTENT - editor state exists but content is empty');
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
        console.log('🚫 Editor preference is "ignore" - not sending editor context');
        activeEditorPayload = null;
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

      const response = await fetch('/api/async/orchestrator/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          query: query,
          conversation_id: conversationId,
          session_id: sessionId,
          active_editor: activeEditorPayload,
          editor_preference: editorPreference,
          active_data_workspace: activeDataWorkspacePayload,
          data_workspace_preference: dataWorkspacePreference,
          user_chat_model: selectedModel || undefined
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
                console.log('🌊 Stream data:', data);

                if (data.type === 'title') {
                  // Handle conversation title update - update immediately in UI
                  if (data.message && conversationId) {
                    console.log('🔤 Received title update:', data.message);
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
                      return updated;
                    }
                    return msg;
                  }));
                } else if (data.type === 'citations') {
                  // **ROOSEVELT'S CITATION CAVALRY**: Capture citations from research agent!
                  console.log('🔗 Citations received:', data.citations);
                  console.log('🔗 streamingMessage.id:', streamingMessage.id);
                  console.log('🔗 Current messages count:', messages.length);
                  const citations = Array.isArray(data.citations) ? data.citations : [];
                  setMessages(prev => {
                    console.log('🔗 Updating messages, looking for id:', streamingMessage.id);
                    const updated = prev.map(msg => {
                      if (msg.id === streamingMessage.id) {
                        console.log('✅ FOUND streaming message, adding citations!');
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
                    console.log('🔗 Messages after citation update:', updated.map(m => ({ id: m.id, hasCitations: !!m.citations, citationCount: m.citations?.length })));
                    return updated;
                  });
                  console.log(`✅ Added ${citations.length} citations to streaming message`);
                } else if (data.type === 'permission_request') {
                  // ROOSEVELT'S HITL: Permission request detected
                  console.log('🛡️ Permission request received:', data);
                  
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
                  
                  console.log('✅ Permission request message updated');
                  
                } else if (data.type === 'notification') {
                  // Signal Corps: Spontaneous notification/alert
                  console.log('📢 Notification received:', data);
                  
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
                  
                  console.log('✅ Notification added to messages');
                  
                } else if (data.type === 'complete_hitl') {
                  // HITL completion - awaiting user permission response
                  console.log('🛡️ HITL completion - awaiting permission response');
                  
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
                    console.log('✅ Streaming completed - final message state:', 
                      updated.find(m => m.id === streamingMessage.id)?.citations ? 
                        `HAS ${updated.find(m => m.id === streamingMessage.id)?.citations?.length} CITATIONS` : 
                        'NO CITATIONS'
                    );
                    return updated;
                  });
                  
                  console.log('✅ Streaming completed successfully');
                  
                  // ROOSEVELT'S CANCEL BUTTON FIX: Clear job ID when streaming completes
                  updateCurrentActivityState({ currentJobId: null });
                  
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
                  if (data.active_line_id) {
                    setActiveLineRouting({
                      id: data.active_line_id,
                      name: data.active_line_name || '',
                    });
                  }
                  // Streaming complete - check if conversation was updated (title generation)
                  if (data.conversation_updated) {
                    console.log('🔄 Conversation updated - refreshing to get new title');
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
      }
      
    } catch (error) {
      console.error('❌ Streaming failed:', error);
      
      // Update message to show error
      updateCurrentActivityState({ currentJobId: null });
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

  const clearChat = () => {
    setMessages([]);
    messagesConversationIdRef.current = null; // Reset ref when clearing chat
    setQuery('');
    // Activity state is conversation-scoped, so clearing conversation clears its state
    if (currentConversationId) {
      updateCurrentActivityState({ currentJobId: null, isLoading: false });
    }
  };

  const cancelCurrentJob = async () => {
    if (!currentJobId) return;
    
    try {
      console.log('🛑 Cancelling job:', currentJobId);
      await apiService.cancelUnifiedJob(currentJobId);
      
      // Update the pending message to show cancellation
      setMessages(prev => prev.map(msg => 
        msg.jobId === currentJobId 
          ? { ...msg, content: '❌ **Cancelled by user**', isCancelled: true }
          : msg
      ));
      
      updateCurrentActivityState({ currentJobId: null, isLoading: false });
      
      console.log('✅ Job cancelled successfully');
    } catch (error) {
      console.error('❌ Failed to cancel job:', error);
    }
  };

  const value = {
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

    // Actions
    toggleSidebar,
    selectConversation,
    createNewConversation,
    sendMessage,
    clearChat,

    cancelCurrentJob, // Add cancellation function

    // **ROOSEVELT**: Editor preference (active = sent to backend, user = checkbox state)
    editorPreference, // Active preference sent to backend (context-aware)
    setEditorPreference: setUserEditorPreference, // UI checkboxes modify user preference
    handleEditorPreferenceChange, // Handler that saves to conversation metadata

    dataWorkspacePreference,
    setDataWorkspacePreference,

    // Conversation preference management
    updateConversationPreference, // Save preferences to conversation metadata

    activeLineRouting,
  };

  return (
    <ChatSidebarContext.Provider value={value}>
      {children}
    </ChatSidebarContext.Provider>
  );
}; 