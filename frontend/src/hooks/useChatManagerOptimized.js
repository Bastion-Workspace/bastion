import { useEffect } from 'react';
import { useQueryClient } from 'react-query';
import apiService from '../services/apiService';
import { useChatState } from './useChatState';
import { useChatActions } from './useChatActions';
import messageDeduplicationService from '../services/messageDeduplicationService';

export const useChatManagerOptimized = () => {
  const queryClient = useQueryClient();
  
  // Get state from useChatState
  const state = useChatState();
  
  // Get actions from useChatActions
  const actions = useChatActions(state);

  // Auto-scroll handled by ChatMessagesArea.js - no duplicate scrolling here

  // Focus on input field when component mounts
  useEffect(() => {
    state.textFieldRef.current?.focus();
  }, []);

  // Initialize background job service
  useEffect(() => {
    actions.initializeBackgroundJobService();
    
    // Cleanup on unmount
    return () => {
      if (state.backgroundJobService) {
        state.backgroundJobService.disconnectAll();
      }
    };
  }, [state.backgroundJobService]);

  // Simple intent classification with immediate UI feedback
  const getExecutionMode = async (query, conversationHistory) => {
    // If force mode is set, use it immediately
    if (state.forceExecutionMode) {
      return state.forceExecutionMode;
    }
    
    // Use the existing LLM classification but don't block UI
    try {
      const classification = await apiService.classifyIntent(query.trim(), conversationHistory);
      return classification.execution_mode;
    } catch (error) {
      console.error('❌ Intent classification failed, defaulting to plan mode:', error);
      return 'plan'; // Default to plan mode for better reliability
    }
  };

  // Optimized message sending with immediate UI feedback
  const handleSendMessage = async () => {
    if (!state.query.trim() || state.isLoading) return;

    const currentQuery = state.query.trim();
    let conversationId = state.currentConversationId;

    // Lock the conversation ID at the start to prevent race conditions
    const lockedConversationId = conversationId;

    // Clear input immediately for better UX
    state.setQuery('');

    // Add user message immediately
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: currentQuery,
      timestamp: new Date(),
    };

    // Handle conversation creation if needed
    if (!lockedConversationId) {
      conversationId = await actions.createNewConversation(currentQuery);
      if (conversationId) {
        state.setCurrentConversationId(conversationId);
      } else {
        console.error('❌ Failed to create conversation');
        state.setQuery(currentQuery); // Restore query on failure
        return;
      }
    } else {
      conversationId = lockedConversationId;
      if (state.messages.length === 0) {
        try {
          await apiService.updateConversation(conversationId, null, {
            initial_message: currentQuery
          });
          queryClient.invalidateQueries(['conversations']);
        } catch (error) {
          console.error('❌ Failed to update conversation title:', error);
        }
      }
    }

    // Add user message to UI immediately
    state.setMessages(prev => [...prev, userMessage]);
    state.setIsLoading(true);

    // Get execution mode with optimization
    let executionMode = 'plan';
    
    try {
      const conversationContext = state.messages.slice(-4).map(msg => ({
        query: msg.type === 'user' ? msg.content : '',
        answer: msg.type === 'bot' ? msg.content : '',
      })).filter(item => item.query || item.answer);

      executionMode = await getExecutionMode(currentQuery, conversationContext);
      
      // Clear force execution mode after use
      if (state.forceExecutionMode) {
        state.setForceExecutionMode(null);
      }
    } catch (error) {
      console.error('❌ Execution mode determination failed:', error);
      executionMode = 'plan';
    }
    
    try {
      // Set the current conversation ID in the background job service
      if (state.backgroundJobService) {
        state.backgroundJobService.setCurrentConversationId(conversationId);
      }
      
      const jobId = await state.backgroundJobService.submitBackgroundJob(
        currentQuery,
        state.sessionId,
        conversationId,
        executionMode
      );
      
      // Show appropriate pending message for ALL execution modes
      let pendingContent;
      if (executionMode === 'plan') {
        pendingContent = `🔄 **Analyzing available data and planning research...** \n\nJob ID: \`${jobId}\`\n\nI'm checking what information is available and will either provide an answer or create a research plan for your approval.`;
      } else if (executionMode === 'execute') {
        pendingContent = `🔄 **Executing research plan...** \n\nJob ID: \`${jobId}\`\n\nResearch tools are running to gather information. Results will appear when complete.`;
      } else if (executionMode === 'chat') {
        pendingContent = `🔄 **Processing your request...** \n\nJob ID: \`${jobId}\`\n\nGenerating response using available knowledge. You can navigate away and return to see the result.`;
      } else {
        pendingContent = `🔄 **Processing...** \n\nJob ID: \`${jobId}\`\n\nWorking on your request in the background.`;
      }
      
      const pendingMessage = {
        id: Date.now(),
        type: 'bot',
        content: pendingContent,
        timestamp: new Date(),
        isResearchJob: true,
        jobId: jobId,
        executionMode: executionMode,
        query: currentQuery,
        metadata: {
          job_id: jobId,
          query: currentQuery,
          execution_mode: executionMode,
          session_id: state.sessionId,
          conversation_id: conversationId
        }
      };
      
      // Add pending message - use the locked conversation ID for consistency
      state.setMessages(prev => [...prev, pendingMessage]);
      state.setIsLoading(false);
      
      // Connect to job WebSocket to receive completion notification
      if (state.backgroundJobService) {
        state.backgroundJobService.connectToJobProgress(jobId, {
          onProgress: (jobId, progress) => {
            console.log('🔄 Job progress:', jobId, progress);
          },
          onCompletion: (jobData) => {
            console.log('✅ Job completed:', jobData);
            actions.handleBackgroundJobCompleted(jobData);
          },
          onError: (error) => {
            console.error('❌ Job WebSocket error:', error);
          }
        }, conversationId); // Pass conversation ID for validation
      }
      
    } catch (error) {
      console.error('❌ Failed to submit research job:', error);
      state.setMessages(prev => [...prev, {
        id: Date.now(),
        type: 'error',
        content: `❌ Failed to start research: ${error.message}`,
        timestamp: new Date(),
      }]);
      state.setIsLoading(false);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  return {
    // State
    query: state.query,
    setQuery: state.setQuery,
    messages: state.messages,
    setMessages: state.setMessages,
    currentConversationId: state.currentConversationId,
    sessionId: state.sessionId,
    isLoading: state.isLoading,
    sidebarCollapsed: state.sidebarCollapsed,
    setSidebarCollapsed: state.setSidebarCollapsed,
    savingNoteFor: state.savingNoteFor,
    copiedMessageId: state.copiedMessageId,
    forceExecutionMode: state.forceExecutionMode,
    setForceExecutionMode: state.setForceExecutionMode,
    backgroundJobService: state.backgroundJobService,
    executingPlans: state.executingPlans,
    setExecutingPlans: state.setExecutingPlans,
    
    // Refs
    messagesEndRef: state.messagesEndRef,
    textFieldRef: state.textFieldRef,
    lastLoadedConversationRef: state.lastLoadedConversationRef,
    
    // Handlers
    handleSendMessage,
    handleKeyPress,
    handleCopyMessage: actions.handleCopyMessage,
    handleSaveAsNote: actions.handleSaveAsNote,

    handleConversationSelect: actions.handleConversationSelect,
    handleClearChat: actions.handleClearChat,
    handleNewConversation: actions.handleNewConversation,
    handleBackgroundJobCompleted: actions.handleBackgroundJobCompleted,
  };
};
