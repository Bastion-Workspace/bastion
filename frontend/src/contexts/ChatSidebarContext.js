import React, { createContext, useContext, useState, useEffect, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useLocation } from 'react-router-dom';
import apiService from '../services/apiService';
import BackgroundJobService from '../services/backgroundJobService';
import tabNotificationManager from '../utils/tabNotification';

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
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('');
  const [backgroundJobService, setBackgroundJobService] = useState(null);
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const [executingPlans, setExecutingPlans] = useState(new Set());
  // LangGraph is the only system - no toggles needed
  const useLangGraphSystem = true; // Always use LangGraph
  const [currentJobId, setCurrentJobId] = useState(null); // Track current job for cancellation
  const [activeTasks, setActiveTasks] = useState(new Map()); // Track active async tasks
  const messagesConversationIdRef = React.useRef(null); // Track which conversation the current messages belong to
  
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

  const queryClient = useQueryClient();

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

  // NEWS NOTIFICATIONS: Poll headlines and raise browser notifications for breaking/urgent
  useEffect(() => {
    let intervalId;
    let lastNotified = new Set();
    const POLL_MS = 60000; // 60s
    const requestPermission = async () => {
      try {
        if (typeof Notification !== 'undefined' && Notification.permission === 'default') {
          await Notification.requestPermission();
        }
      } catch {}
    };
    const notify = (headline) => {
      try {
        if (typeof Notification === 'undefined') return;
        if (Notification.permission !== 'granted') return;
        const n = new Notification(headline.title || 'Breaking News', {
          body: headline.summary || '',
          tag: headline.id,
        });
        n.onclick = () => {
          try { window.focus(); } catch {}
          window.location.href = `/news/${headline.id}`;
        };
      } catch {}
    };
    const poll = async () => {
      try {
        const res = await apiService.get('/api/news/headlines');
        const headlines = (res && res.headlines) || [];
        for (const h of headlines) {
          if ((h.severity === 'breaking' || h.severity === 'urgent') && !lastNotified.has(h.id)) {
            notify(h);
            lastNotified.add(h.id);
            if (lastNotified.size > 200) {
              // Prevent unbounded growth
              lastNotified = new Set(Array.from(lastNotified).slice(-100));
            }
          }
        }
      } catch {}
    };
    requestPermission();
    poll();
    intervalId = setInterval(poll, POLL_MS);
    return () => intervalId && clearInterval(intervalId);
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

  // Save selected model to localStorage
  useEffect(() => {
    if (selectedModel) {
      localStorage.setItem('chatSidebarSelectedModel', selectedModel);
    }
  }, [selectedModel]);

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
  const { data: conversationData, isLoading: conversationLoading } = useQuery(
    ['conversation', currentConversationId],
    () => currentConversationId ? apiService.getConversation(currentConversationId) : null,
    {
      enabled: !!currentConversationId,
      refetchOnWindowFocus: false,
      staleTime: 300000, // 5 minutes
      onSuccess: (data) => {
        console.log('✅ ChatSidebarContext: Conversation data loaded:', {
          conversationId: currentConversationId,
          hasMessages: !!data?.messages,
          messageCount: data?.messages?.length || 0
        });
      },
      onError: (error) => {
        console.error('❌ ChatSidebarContext: Failed to load conversation:', error);
      }
    }
  );

  // PRIORITY: Load messages using unified chat service
  const { data: messagesData, isLoading: messagesLoading, refetch: refetchMessages } = useQuery(
    ['conversationMessages', currentConversationId],
    () => currentConversationId ? apiService.getConversationMessages(currentConversationId) : null,
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
            metadata: message.metadata || {},
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
        setCurrentConversationId(newConversation.conversation.conversation_id); // Access conversation_id through the conversation field
        setMessages([]);
        messagesConversationIdRef.current = null; // Reset ref for new conversation
        setQuery('');
        queryClient.invalidateQueries(['conversations']);
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
    setExecutingPlans(prev => {
      const newSet = new Set(prev);
      newSet.delete(jobData.job_id);
      return newSet;
    });
    
    // Invalidate conversations to refresh the list
    queryClient.invalidateQueries(['conversations']);
    
    // Clear current job ID when job completes
    if (currentJobId === jobData.job_id) {
      setCurrentJobId(null);
    }
    
    console.log('✅ Background job completion handling completed');
  };

  // Handle plan execution
  // ROOSEVELT: Execute Plan logic removed - pure LangGraph uses simple Yes/No responses
  const handleExecutePlan = async (message) => {
    console.log('🔍 Attempting to execute plan for message:', {
      messageId: message.id,
      hasResearchPlan: !!message.research_plan,
      hasJobId: !!message.jobId,
      hasMetadataJobId: !!message.metadata?.job_id,
      content: message.content?.substring(0, 100),
      metadata: message.metadata
    });
    
    // Check for research plan in multiple ways to handle page refresh scenarios
    const hasResearchPlan = message.research_plan || 
                           (message.content && message.content.includes('## Research Plan')) ||
                           (message.content && message.content.includes('**Research Plan**')) ||
                           (message.content && message.content.includes('### Research Plan')) ||
                           (message.content && message.content.includes('Research Plan:')) ||
                           (message.content && message.content.includes('Step') && message.content.includes('Research'));
    
    const hasJobId = message.jobId || message.metadata?.job_id;
    
    console.log('🔍 Plan execution validation:', {
      hasResearchPlan,
      hasJobId,
      researchPlanContent: message.research_plan?.substring(0, 50),
      messageContent: message.content?.substring(0, 50)
    });
    
    if (!hasResearchPlan || !hasJobId) {
      console.warn('❌ Cannot execute plan - missing research plan or job ID:', {
        hasResearchPlan,
        hasJobId,
        messageId: message.id,
        content: message.content?.substring(0, 100)
      });
      return;
    }
    
    try {
      console.log('🚀 Executing research plan for job:', hasJobId);
      
      // Mark this plan as being executed
      setExecutingPlans(prev => new Set([...prev, hasJobId]));
      
      setMessages(prev => prev.map(msg => 
        msg.id === message.id 
          ? { ...msg, planApproved: true, isExecuting: true }
          : msg
      ));
      
      // Get the original query from the conversation messages
      // Look for the most recent user message to get the original query
      let originalQuery = 'Execute research plan';
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === 'user') {
          originalQuery = messages[i].content;
          break;
        }
      }
      
      console.log('🔍 Found original query:', originalQuery);
      
      console.log('🔬 Calling plan execution endpoint with original query:', originalQuery);
      
      const response = await apiService.executePlan({
        query: originalQuery,
        conversation_id: currentConversationId,
        session_id: sessionId,
        parent_job_id: hasJobId
      });
      
      console.log('✅ Plan execution response:', response);
      
      if (response.success) {
        // Add the execution job as a new message
        const executionMessage = {
          id: Date.now(),
          role: 'assistant',
          content: `🔄 **Executing research plan...** \n\nJob ID: \`${response.job_id}\`\n\nResearch tools are running to gather information. Results will appear when complete.`,
          timestamp: new Date().toISOString(),
          isResearchJob: true,
          jobId: response.job_id,
          executionMode: 'execute',
          query: originalQuery,
          metadata: {
            job_id: response.job_id,
            query: originalQuery,
            execution_mode: 'execute',
            session_id: sessionId,
            conversation_id: currentConversationId,
            parent_job_id: hasJobId
          }
        };
        
        setMessages(prev => [...prev, executionMessage]);
        
        // Connect to the execution job's WebSocket
        backgroundJobService.connectToJobProgress(response.job_id, {
          onProgress: (jobId, progress) => {
            console.log('🔄 Plan execution progress:', jobId, progress);
            handleBackgroundJobProgress(progress);
          },
          onCompletion: (jobData) => {
            console.log('✅ Plan execution completed:', jobData);
            handleBackgroundJobCompleted(jobData);
          },
          onError: (error) => {
            console.error('❌ Plan execution WebSocket error:', error);
          }
        }, currentConversationId);
        
      } else {
        console.error('❌ Plan execution failed:', response.error);
        // Remove from executing plans on failure
        setExecutingPlans(prev => {
          const newSet = new Set(prev);
          newSet.delete(hasJobId);
          return newSet;
        });
        
        // Update message to show error
        setMessages(prev => prev.map(msg => 
          msg.id === message.id 
            ? { ...msg, planApproved: false, isExecuting: false }
            : msg
        ));
      }
      
    } catch (error) {
      console.error('❌ Plan execution error:', error);
      
      // Remove from executing plans on error
      setExecutingPlans(prev => {
        const newSet = new Set(prev);
        newSet.delete(hasJobId);
        return newSet;
      });
      
      // Update message to show error
      setMessages(prev => prev.map(msg => 
        msg.id === message.id 
          ? { ...msg, planApproved: false, isExecuting: false }
          : msg
      ));
    }
  };

  // Check if a message is a follow-up command that should execute a plan
  const isFollowUpCommand = (query) => {
    const lowerQuery = query.toLowerCase().trim();
    return (
      lowerQuery === 'yes' ||
      lowerQuery === 'execute' ||
      lowerQuery === 'execute the plan' ||
      lowerQuery === 'run the plan' ||
      lowerQuery === 'go ahead' ||
      lowerQuery === 'proceed' ||
      lowerQuery === 'do it' ||
      lowerQuery.startsWith('yes,') ||
      lowerQuery.startsWith('execute') ||
      lowerQuery.includes('execute the plan') ||
      lowerQuery.includes('run the plan')
    );
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

  // Find the most recent research plan message
  const findRecentResearchPlan = () => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      if (message.role === 'assistant' && 
          (message.research_plan || 
           (message.content && (
             message.content.includes('## Research Plan') ||
             message.content.includes('**Research Plan**') ||
             message.content.includes('### Research Plan') ||
             message.content.includes('Research Plan:')
           )))) {
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
    setExecutingPlans(new Set());
    setQuery('');
    setIsLoading(false); // ✅ Clear loading state to prevent cross-conversation thinking wheel
    
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
  const pollTaskStatus = async (taskId, conversationId) => {
    const pollInterval = 2000; // Poll every 2 seconds
    const maxAttempts = 150; // 5 minutes maximum (150 * 2s)
    let attempts = 0;

    const poll = async () => {
      try {
        attempts++;
        const statusResponse = await apiService.get(`/api/async/orchestrator/status/${taskId}`);
        
        console.log(`🔍 Task ${taskId} status:`, statusResponse.status);
        
        // Update progress in messages
        setMessages(prev => prev.map(msg => {
          if (msg.taskId === taskId && msg.isPending) {
            return {
              ...msg,
              progress: statusResponse.progress || msg.progress,
              content: statusResponse.progress?.message || msg.content
            };
          }
          return msg;
        }));

        if (statusResponse.status === 'SUCCESS') {
          // Task completed successfully
          const result = statusResponse.result;
          console.log('✅ Async task completed:', result);
          
          // Flash tab if tab is hidden when async response completes
          if (result.response && result.response.trim().length > 0) {
            tabNotificationManager.startFlashing('New message');
          }
          
          // Replace pending message with final result
          setMessages(prev => prev.map(msg => {
            if (msg.taskId === taskId && msg.isPending) {
              return {
                ...msg,
                isPending: false,
                content: result.response || 'Task completed successfully',
                timestamp: new Date().toISOString(),
                metadata: {
                  ...msg.metadata,
                  async_completed: true,
                  delegated_agent: result.delegated_agent,
                  orchestrator_decision: result.orchestrator_decision
                }
              };
            }
            return msg;
          }));
          
          // Remove from active tasks
          setActiveTasks(prev => {
            const newMap = new Map(prev);
            newMap.delete(taskId);
            return newMap;
          });
          
          // Refresh conversations
          queryClient.invalidateQueries(['conversations']);
          
        } else if (statusResponse.status === 'FAILURE') {
          // Task failed
          console.error('❌ Async task failed:', statusResponse.error);
          
          setMessages(prev => prev.map(msg => {
            if (msg.taskId === taskId && msg.isPending) {
              return {
                ...msg,
                isPending: false,
                content: `Error: ${statusResponse.error || 'Task failed'}`,
                isError: true,
                timestamp: new Date().toISOString()
              };
            }
            return msg;
          }));
          
          // Remove from active tasks
          setActiveTasks(prev => {
            const newMap = new Map(prev);
            newMap.delete(taskId);
            return newMap;
          });
          
        } else if (attempts < maxAttempts) {
          // Continue polling
          setTimeout(poll, pollInterval);
        } else {
          // Max attempts reached
          console.warn('⚠️ Max polling attempts reached for task:', taskId);
          setMessages(prev => prev.map(msg => {
            if (msg.taskId === taskId && msg.isPending) {
              return {
                ...msg,
                isPending: false,
                content: 'Task is taking longer than expected. Please check back later.',
                isWarning: true,
                timestamp: new Date().toISOString()
              };
            }
            return msg;
          }));
        }
        
      } catch (error) {
        console.error('❌ Error polling task status:', error);
        if (attempts < maxAttempts) {
          setTimeout(poll, pollInterval * 2); // Slower retry on error
        }
      }
    };

    // Start polling
    setTimeout(poll, pollInterval);
    
    // Track active task
    setActiveTasks(prev => new Map(prev).set(taskId, {
      taskId,
      conversationId,
      startedAt: new Date(),
      attempts: 0
    }));
  };

  const sendMessage = async (executionMode = 'auto', overrideQuery = null) => {
    // ROOSEVELT'S HITL SUPPORT: Allow override query for direct API calls without state dependency
    const actualQuery = overrideQuery || query.trim();
    console.log('🔄 sendMessage called with:', { query: actualQuery, overrideQuery: !!overrideQuery, backgroundJobService: !!backgroundJobService, executionMode });
    
    if (!actualQuery || !backgroundJobService) {
      console.log('❌ sendMessage early return:', { hasQuery: !!actualQuery, hasService: !!backgroundJobService });
      return;
    }

    const currentQuery = actualQuery;
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
    // Check if this is a follow-up command to execute a research plan (ONLY if not HITL)
    else if (isFollowUpCommand(currentQuery)) {
      const recentPlan = findRecentResearchPlan();
      if (recentPlan) {
        console.log('🎯 Detected follow-up command, executing recent plan:', recentPlan);
        await handleExecutePlan(recentPlan);
        return;
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
    setIsLoading(true);

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
        setIsLoading(false);
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
        
        try {
          // Use streaming endpoint for ALL real-time responses
          await handleStreamingResponse(currentQuery, conversationId, sessionId);
        } catch (streamingError) {
          console.log('🔄 Streaming failed, falling back to async:', streamingError.message);
          // Fallback to async orchestrator only if streaming fails
          const result = await apiService.post('/api/async/orchestrator/start', {
            query: currentQuery,
            conversation_id: conversationId,
            session_id: sessionId,
            priority: 'normal'
          });
        
          if (result.success) {
          const taskId = result.task_id;
          console.log('🔍 Async orchestrator task started:', {
            taskId,
            status: result.status,
            conversationId: result.conversation_id,
            estimatedCompletion: result.estimated_completion
          });
          
          // ROOSEVELT'S CANCEL BUTTON FIX: Set currentJobId for cancel button functionality
          setCurrentJobId(taskId);
          
          // Add pending message to indicate task is processing
          const pendingMessage = {
            id: Date.now() + 1,
            role: 'assistant',
            type: 'assistant',
            content: '🔄 Processing your request in the background...',
            timestamp: new Date().toISOString(),
            isPending: true,
            taskId: taskId,
            progress: { percentage: 0, message: 'Task queued and starting...' },
            metadata: {
              async_task: true,
              task_id: taskId,
              estimated_completion: result.estimated_completion
            }
          };
          
          setMessages(prev => [...prev, pendingMessage]);
          
          // Start polling for task status
          pollTaskStatus(taskId, conversationId);
          
          // Invalidate conversations to refresh the list
          queryClient.invalidateQueries(['conversations']);
          queryClient.invalidateQueries(['conversation', conversationId]);
          
            console.log('✅ LangGraph message processed successfully');
          } else {
            throw new Error(result.error || 'Failed to process LangGraph message');
          }
        }
        
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
        setIsLoading(false);
      }
  };

  // 🌊 ROOSEVELT'S UNIVERSAL STREAMING POLICY: All queries deserve real-time responses!

  // Handle streaming response from orchestrator
  const handleStreamingResponse = async (query, conversationId, sessionId) => {
    console.log('🌊 Starting streaming response for:', query);
    
    try {
      // ROOSEVELT'S CANCEL BUTTON FIX: Create streaming job ID for cancel functionality
      const streamingJobId = `streaming_${Date.now()}`;
      setCurrentJobId(streamingJobId);
      
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
      
      // Attach active editor payload ONLY if an editable markdown file is actually open in an editor tab
      // CRITICAL: Be very strict - check localStorage with strict validation
      // Note: EditorProvider is a child of ChatSidebarProvider, so we check localStorage directly
      let activeEditorPayload = null;
      try {
        // Get editor state from localStorage (updated by DocumentViewer when editor is open)
        const editorCtx = JSON.parse(localStorage.getItem('editor_ctx_cache') || 'null');
        
        // STRICT VALIDATION: Must have ALL of these conditions met:
        // 1. editorCtx exists
        // 2. isEditable is EXACTLY true (not truthy, not undefined)
        // 3. filename exists and ends with .md
        // 4. content exists (not empty/null)
        // 5. frontmatter.type is in allowed list
        const hasValidEditorState = editorCtx && 
                                    editorCtx.isEditable === true && 
                                    editorCtx.filename && 
                                    editorCtx.filename.toLowerCase().endsWith('.md') &&
                                    editorCtx.content &&
                                    editorCtx.content.trim().length > 0;
        
        if (hasValidEditorState) {
          // All validation passed - editor is actually open and editable
          const fmType = (editorCtx.frontmatter && editorCtx.frontmatter.type || '').toLowerCase().trim();
          const allowedTypes = ['fiction','non-fiction','nonfiction','article','rules','outline','character','style','sysml','podcast','substack','blog','electronics','project'];
          
          if (allowedTypes.includes(fmType)) {
            activeEditorPayload = {
              is_editable: true,
              filename: editorCtx.filename,
              language: 'markdown',
              content: editorCtx.content,
              content_length: editorCtx.contentLength || editorCtx.content.length,
              frontmatter: editorCtx.frontmatter || {},
              cursor_offset: typeof editorCtx.cursorOffset === 'number' ? editorCtx.cursorOffset : -1,
              selection_start: typeof editorCtx.selectionStart === 'number' ? editorCtx.selectionStart : -1,
              selection_end: typeof editorCtx.selectionEnd === 'number' ? editorCtx.selectionEnd : -1,
              canonical_path: editorCtx.canonicalPath || null,
            };
            console.log('✅ Editor tab is open and editable - sending active_editor:', editorCtx.filename);
          } else {
            console.log('🚫 Editor open but frontmatter.type not in allowed list:', fmType);
            activeEditorPayload = null;
          }
        } else {
          // Editor is NOT open or NOT editable - be very explicit about why
          if (!editorCtx) {
            console.log('🚫 NO EDITOR STATE IN CACHE - no editor tab is open');
          } else if (editorCtx.isEditable !== true) {
            console.log('🚫 EDITOR NOT EDITABLE (isEditable=' + editorCtx.isEditable + ') - viewing PDF or document, not editing');
          } else if (!editorCtx.filename || !editorCtx.filename.toLowerCase().endsWith('.md')) {
            console.log('🚫 NO VALID MARKDOWN FILE - filename:', editorCtx.filename);
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
          editor_preference: editorPreference
        })
      });

      if (!response.ok) {
        if (response.status === 401) {
          console.log('🔒 Authentication failed for streaming, falling back to async');
          throw new Error('Authentication failed - falling back to async');
        }
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
                      // Check for agent info in various field names: agent_type, agent, node
                      const agentType = data.agent_type || data.agent || data.node;
                      if (agentType) {
                        updateData.metadata = { ...currentMetadata, agent_type: agentType };
                        // If content is empty, show formatted agent name
                        if (!data.message || data.message.trim() === '') {
                          updateData.content = formatAgentName(agentType);
                        }
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
                        // If content is empty, show formatted agent name
                        if (!msg.content || msg.content.trim() === '') {
                          updateData.content = formatAgentName(agentType);
                        }
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
                  // ROOSEVELT'S NEWLINE FIX: Don't add spaces between chunks!
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
                } else if (data.type === 'editor_operations') {
                  // ROOSEVELT'S HITL EDIT PREVIEW: Capture editor operations payload
                  const ops = Array.isArray(data.operations) ? data.operations : [];
                  const mEdit = data.manuscript_edit || null;
                  setMessages(prev => prev.map(msg => 
                    msg.id === streamingMessage.id 
                      ? { 
                          ...msg, 
                          editor_operations: ops,
                          manuscript_edit: mEdit,
                          hasEditorOps: ops.length > 0
                        }
                      : msg
                  ));
                  
                  // Emit event for live diff display in editor
                  if (ops.length > 0) {
                    console.log('🔍 ChatSidebarContext emitting editorOperationsLive:', {
                      operationsCount: ops.length,
                      messageId: streamingMessage.id,
                      firstOp: ops[0]
                    });
                    window.dispatchEvent(new CustomEvent('editorOperationsLive', {
                      detail: { 
                        operations: ops,
                        manuscriptEdit: mEdit,
                        messageId: streamingMessage.id
                      }
                    }));
                  }
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
                  
                } else if (data.type === 'complete_hitl') {
                  // HITL completion - awaiting user permission response
                  console.log('🛡️ HITL completion - awaiting permission response');
                  
                  // Don't clear job ID yet - we're waiting for user input
                  setIsLoading(false);
                  
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
                            timestamp: new Date().toISOString()
                            // **ROOSEVELT'S CITATION PRESERVATION**: Don't overwrite citations!
                            // Citations and metadata are already in msg from previous citation event
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
                  setCurrentJobId(null);
                  
                  // Refresh conversations - title may have been updated from "New Conversation"
                  // Force a refetch to ensure we get the latest title
                  queryClient.invalidateQueries(['conversations']);
                  queryClient.invalidateQueries(['conversation', conversationId]);
                  
                  // Also refetch the conversation list to ensure title updates are visible
                  queryClient.refetchQueries(['conversations']);
                  break;
                } else if (data.type === 'done') {
                  // Streaming complete - check if conversation was updated (title generation)
                  if (data.conversation_updated) {
                    console.log('🔄 Conversation updated - refreshing to get new title');
                    // Invalidate and refetch conversations to get updated title
                    queryClient.invalidateQueries(['conversations']);
                    queryClient.invalidateQueries(['conversation', conversationId]);
                    queryClient.refetchQueries(['conversations']);
                  }
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
      
      // If streaming failed due to auth or other issues, fall back to async
      if (error.message.includes('Authentication failed') || error.message.includes('401')) {
        console.log('🔄 Falling back to async orchestrator due to streaming failure');
        
        // Remove the streaming message
        setMessages(prev => prev.filter(msg => !msg.isStreaming));
        
        // Call async orchestrator as fallback
        try {
          const result = await apiService.post('/api/async/orchestrator/start', {
            query: query,
            conversation_id: conversationId,
            session_id: sessionId,
            priority: 'normal'
          });
          
          if (result.success) {
            const taskId = result.task_id;
            console.log('🔄 Fallback async task started:', taskId);
            
            // ROOSEVELT'S CANCEL BUTTON FIX: Set currentJobId for cancel button functionality
            setCurrentJobId(taskId);
            
            // Add pending message for async processing
            const pendingMessage = {
              id: Date.now() + 1,
              role: 'assistant',
              type: 'assistant',
              content: '🔄 Processing your request in the background...',
              timestamp: new Date().toISOString(),
              isPending: true,
              taskId: taskId,
              progress: { percentage: 0, message: 'Streaming failed, using async processing...' },
              metadata: {
                async_task: true,
                task_id: taskId,
                fallback: true
              }
            };
            
            setMessages(prev => [...prev, pendingMessage]);
            
            // Start polling for task status
            pollTaskStatus(taskId, conversationId);
            
            // Refresh conversations
            queryClient.invalidateQueries(['conversations']);
            queryClient.invalidateQueries(['conversation', conversationId]);
          }
        } catch (asyncError) {
          console.error('❌ Async fallback also failed:', asyncError);
          // ROOSEVELT'S CANCEL BUTTON FIX: Clear job ID on complete failure
          setCurrentJobId(null);
          setMessages(prev => [...prev, {
            id: Date.now(),
            role: 'system',
            type: 'system',
            content: `❌ Both streaming and async processing failed: ${asyncError.message}`,
            timestamp: new Date().toISOString(),
            isError: true,
          }]);
        }
      } else {
        // Update message to show error for non-auth issues
        // ROOSEVELT'S CANCEL BUTTON FIX: Clear job ID on streaming failure
        setCurrentJobId(null);
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
    }
  };

  const clearChat = () => {
    setMessages([]);
    messagesConversationIdRef.current = null; // Reset ref when clearing chat
    setQuery('');
    setCurrentJobId(null); // Clear current job when clearing chat
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
      
      setCurrentJobId(null);
      setIsLoading(false);
      
      console.log('✅ Job cancelled successfully');
    } catch (error) {
      console.error('❌ Failed to cancel job:', error);
    }
  };

  // Cancel async orchestrator task
  const cancelAsyncTask = async (taskId) => {
    try {
      console.log('🛑 Attempting to cancel async task:', taskId);
      await apiService.post(`/api/async/orchestrator/cancel/${taskId}`);
      
      // Update message to show cancellation
      setMessages(prev => prev.map(msg => {
        if (msg.taskId === taskId && msg.isPending) {
          return {
            ...msg,
            isPending: false,
            content: '❌ **Task cancelled by user**',
            isCancelled: true,
            timestamp: new Date().toISOString()
          };
        }
        return msg;
      }));
      
      // Remove from active tasks
      setActiveTasks(prev => {
        const newMap = new Map(prev);
        newMap.delete(taskId);
        return newMap;
      });
      
      console.log('✅ Async task cancellation requested');
    } catch (error) {
      console.error('❌ Failed to cancel async task:', error);
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
    isLoading,
    selectedModel,
    setSelectedModel,
    backgroundJobService,
    sessionId,
    executingPlans,
    currentJobId, // Add current job ID for cancellation
    activeTasks, // Add active async tasks tracking
    
    // Actions
    toggleSidebar,
    selectConversation,
    createNewConversation,
    sendMessage,
    clearChat,

    cancelCurrentJob, // Add cancellation function
    cancelAsyncTask, // Add async task cancellation
    
    // **ROOSEVELT**: Editor preference (active = sent to backend, user = checkbox state)
    editorPreference, // Active preference sent to backend (context-aware)
    setEditorPreference: setUserEditorPreference, // UI checkboxes modify user preference
  };

  return (
    <ChatSidebarContext.Provider value={value}>
      {children}
    </ChatSidebarContext.Provider>
  );
}; 