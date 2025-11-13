import { useEffect } from 'react';
import { useQueryClient } from 'react-query';
import apiService from '../services/apiService';
import { useChatState } from './useChatState';
import { useChatActions } from './useChatActions';
import messageDeduplicationService from '../services/messageDeduplicationService';

export const useChatManager = () => {
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

  // Handle sending messages
  const handleSendMessage = async () => {
    if (!state.query.trim() || state.isLoading) return;

    const currentQuery = state.query.trim();
    let conversationId = state.currentConversationId;

    if (!conversationId) {
      conversationId = await actions.createNewConversation(currentQuery);
      if (conversationId) {
        state.setCurrentConversationId(conversationId);
      } else {
        console.error('❌ Failed to create conversation');
        return;
      }
    } else {
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

    // CRITICAL: Check if conversation has changed during async operations
    if (state.currentConversationId !== conversationId) {
      console.warn(`⚠️ Conversation changed during message submission. Original: ${conversationId}, Current: ${state.currentConversationId}. Aborting message submission.`);
      return;
    }

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: state.query.trim(),
      timestamp: new Date(),
    };

    // CRITICAL: Only add message if we're still in the same conversation
    if (state.currentConversationId === conversationId) {
      state.setMessages(prev => [...prev, userMessage]);
      state.setIsLoading(true);
    } else {
      console.warn(`⚠️ Conversation changed before adding user message. Aborting.`);
      return;
    }

    let executionMode = 'plan';
    
    if (state.forceExecutionMode) {
      executionMode = state.forceExecutionMode;
      state.setForceExecutionMode(null);
    } else {
      try {
        const conversationContext = state.messages.slice(-4).map(msg => ({
          query: msg.type === 'user' ? msg.content : '',
          answer: msg.type === 'bot' ? msg.content : '',
        })).filter(item => item.query || item.answer);

        const classification = await apiService.classifyIntent(state.query.trim(), conversationContext);
        executionMode = classification.execution_mode;
      } catch (error) {
        console.error('❌ Intent classification failed, defaulting to plan mode:', error);
        executionMode = 'plan';
      }
    }
    
    try {
      // CRITICAL: Check conversation hasn't changed before job submission
      if (state.currentConversationId !== conversationId) {
        console.warn(`⚠️ Conversation changed before job submission. Aborting.`);
        state.setIsLoading(false);
        return;
      }

      // Set the current conversation ID in the background job service
      if (state.backgroundJobService) {
        state.backgroundJobService.setCurrentConversationId(conversationId);
      }
      
      const jobId = await state.backgroundJobService.submitBackgroundJob(
        state.query.trim(),
        state.sessionId,
        conversationId,
        executionMode
      );
      
      // CRITICAL: Check conversation hasn't changed after job submission
      if (state.currentConversationId !== conversationId) {
        console.warn(`⚠️ Conversation changed after job submission. Aborting pending message.`);
        state.setIsLoading(false);
        return;
      }
      
      // Only show pending message for plan and execute modes, not for quick responses
      if (executionMode === 'plan' || executionMode === 'execute') {
        let pendingContent;
        if (executionMode === 'plan') {
          pendingContent = `🔄 **Analyzing available data and planning research...** \n\nJob ID: \`${jobId}\`\n\nI'm checking what information is available and will either provide an answer or create a research plan for your approval.`;
        } else if (executionMode === 'execute') {
          pendingContent = `🔄 **Executing research plan...** \n\nJob ID: \`${jobId}\`\n\nResearch tools are running to gather information. Results will appear when complete.`;
        }
        
        const pendingMessage = {
          id: Date.now(),
          type: 'bot',
          content: pendingContent,
          timestamp: new Date(),
          isResearchJob: true,
          jobId: jobId,
          executionMode: executionMode,
          query: state.query.trim(),
          metadata: {
            job_id: jobId,
            query: state.query.trim(),
            execution_mode: executionMode,
            session_id: state.sessionId,
            conversation_id: conversationId
          }
        };
        
        // CRITICAL: Only add pending message if still in same conversation
        if (state.currentConversationId === conversationId) {
          state.setMessages(prev => [...prev, pendingMessage]);
        } else {
          console.warn(`⚠️ Conversation changed before adding pending message. Aborting.`);
        }
      }
      state.setIsLoading(false);
      
      // Connect to job WebSocket to receive completion notification
      if (state.backgroundJobService && state.currentConversationId === conversationId) {
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
      // CRITICAL: Only add error message if still in same conversation
      if (state.currentConversationId === conversationId) {
        state.setMessages(prev => [...prev, {
          id: Date.now(),
          type: 'error',
          content: `❌ Failed to start research: ${error.message}`,
          timestamp: new Date(),
        }]);
      }
      state.setIsLoading(false);
    }

    state.setQuery('');
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
