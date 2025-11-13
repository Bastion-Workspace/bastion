import { useMutation, useQueryClient } from 'react-query';
import apiService from '../services/apiService';
import ResearchJobService from '../services/backgroundJobService';
import { copyToClipboard, copyAsPlainText, smartCopy } from '../utils/chatUtils';
import messageDeduplicationService from '../services/messageDeduplicationService';

export const useChatActions = (state) => {
  const queryClient = useQueryClient();
  const {
    setMessages,
    setCurrentConversationId,
    setSessionId,
    setIsLoading,
    setExecutingPlans,
    setCopiedMessageId,
    setSavingNoteFor,
    setForceExecutionMode,
    setBackgroundJobService,
    setQuery,
    messagesEndRef,
    lastLoadedConversationRef,
    currentConversationId,
    sessionId,
    backgroundJobService,
    executingPlans,
    messages,
  } = state;

  // Initialize background job service
  const initializeBackgroundJobService = () => {
    if (!backgroundJobService && apiService) {
      const researchService = new ResearchJobService(apiService);
      setBackgroundJobService(researchService);
    }
  };

  // Handle completed research jobs
  const handleBackgroundJobCompleted = (job) => {
    console.log('🎯 Research job completed:', job);
    console.log('🔍 Job result details:', {
      hasResult: !!job.result,
      hasAnswer: !!(job.result && job.result.answer),
      hasPlan: !!(job.result && job.result.research_plan),
      conversationId: job.conversation_id,
      currentConversationId: currentConversationId
    });
    
    // CRITICAL: More flexible conversation validation
    // Allow processing if:
    // 1. Current conversation is null (new conversation scenario)
    // 2. Job conversation matches current conversation
    // 3. Current conversation is in transition (null/undefined values)
    const shouldProcessJob = !currentConversationId || 
                           job.conversation_id === currentConversationId ||
                           !currentConversationId || 
                           currentConversationId === 'null' || 
                           currentConversationId === 'undefined';
    
    if (!shouldProcessJob) {
      console.warn(`⚠️ Skipping job completion ${job.job_id} - belongs to conversation ${job.conversation_id}, not ${currentConversationId}`);
      return;
    }
    
    // If currentConversationId is null but we have a job with a conversation ID,
    // update the current conversation ID to match the job
    let effectiveConversationId = currentConversationId;
    if (currentConversationId === null && job.conversation_id) {
      console.log(`🔄 Setting current conversation ID to ${job.conversation_id} from completed job`);
      effectiveConversationId = job.conversation_id;
      setCurrentConversationId(job.conversation_id);
    }
    
    // ADDITIONAL CHECK: Verify the WebSocket connection is still valid for this conversation
    // BUT: Allow completion if currentConversationId is null and we're setting it from the job
    if (backgroundJobService && backgroundJobService.isJobWebSocketActive(job.job_id)) {
      const expectedConversationId = backgroundJobService.getJobWebSocketConversationId(job.job_id);
      if (expectedConversationId && expectedConversationId !== currentConversationId && currentConversationId !== null) {
        console.warn(`⚠️ Skipping job completion ${job.job_id} - WebSocket expects conversation ${expectedConversationId}, but current is ${currentConversationId}`);
        return;
      } else if (expectedConversationId && currentConversationId === null) {
        console.log(`✅ Allowing job completion ${job.job_id} - WebSocket expects conversation ${expectedConversationId}, current is null (will be set)`);
      }
    }
    
    // Process the job result - be more flexible about what constitutes a valid result
    const hasValidResult = job.result && (
      job.result.answer || 
      job.result.research_plan || 
      job.result.error || 
      job.result.status
    );
    
    if (hasValidResult) {
      setMessages(prev => {
        // Check if we already have a result message for this job
        const hasExistingResult = prev.some(msg => 
          msg.jobId === job.job_id && msg.isResearchResult && !msg.isResearchJob
        );
        
        if (hasExistingResult) {
          console.log(`📋 Result already exists for job ${job.job_id}, skipping duplicate`);
          return prev;
        }
        
        let messageReplaced = false;
        const updated = prev.map(msg => {
          // Replace pending message if it exists
          if (msg.jobId === job.job_id && msg.isResearchJob && !msg.isResearchResult) {
            messageReplaced = true;
            let content, isResearchPlan = false;
            
            // Determine content type based on result
            if (job.result.answer) {
              // This is an execution result or immediate answer
              content = job.result.answer;
              isResearchPlan = false;
            } else if (job.result.research_plan) {
              // This is a research plan
              content = job.result.research_plan;
              isResearchPlan = true;
            } else if (job.result.error) {
              // This is an error result
              content = `❌ **Error during processing:**\n\n${job.result.error}`;
              isResearchPlan = false;
            } else {
              // Fallback content
              content = job.result.status || 'Processing completed';
              isResearchPlan = false;
            }
            
            console.log(`🔄 Replacing pending message for job ${job.job_id} with ${isResearchPlan ? 'plan' : 'answer'}`);
            
            return {
              id: `job_result_${job.job_id}`,
              type: 'bot',
              content: content,
              citations: job.result.citations || [],
              timestamp: new Date().toISOString(),
              processing_time: job.result.processing_time || 0,
              execution_mode: job.result.execution_mode || 'plan',
              isResearchResult: true,
              isResearchPlan: isResearchPlan,
              planApproved: job.result.plan_approved || false,
              jobId: job.job_id,
              query: job.query,
              research_plan: job.result.research_plan,
              metadata: {
                job_id: job.job_id,
                query: job.query,
                execution_mode: job.result.execution_mode || 'plan',
                plan_approved: job.result.plan_approved || false,
                processing_time: job.result.processing_time || 0,
                session_id: sessionId,
                conversation_id: effectiveConversationId
              }
            };
          }
          return msg;
        });
        
        // If no pending message was found to replace, add a new result message
        if (!messageReplaced) {
          let content, isResearchPlan = false;
          
          // Determine content type based on result
          if (job.result.answer) {
            // This is an execution result or immediate answer
            content = job.result.answer;
            isResearchPlan = false;
          } else if (job.result.research_plan) {
            // This is a research plan
            content = job.result.research_plan;
            isResearchPlan = true;
          } else if (job.result.error) {
            // This is an error result
            content = `❌ **Error during processing:**\n\n${job.result.error}`;
            isResearchPlan = false;
          } else {
            // Fallback content
            content = job.result.status || 'Processing completed';
            isResearchPlan = false;
          }
          
          console.log(`📋 Adding new result message for job ${job.job_id} (${isResearchPlan ? 'plan' : 'answer'})`);
          
          const newResultMessage = {
            id: `job_result_${job.job_id}`,
            type: 'bot',
            content: content,
            citations: job.result.citations || [],
            timestamp: new Date().toISOString(),
            processing_time: job.result.processing_time || 0,
            execution_mode: job.result.execution_mode || 'plan',
            isResearchResult: true,
            isResearchPlan: isResearchPlan,
            planApproved: job.result.plan_approved || false,
            jobId: job.job_id,
            query: job.query,
            research_plan: job.result.research_plan,
            metadata: {
              job_id: job.job_id,
              query: job.query,
              execution_mode: job.result.execution_mode || 'plan',
              plan_approved: job.result.plan_approved || false,
              processing_time: job.result.processing_time || 0,
              session_id: sessionId,
              conversation_id: effectiveConversationId
            }
          };
          
          // Check for duplicates before adding
          if (!messageDeduplicationService.isDuplicateMessage(newResultMessage, effectiveConversationId, updated)) {
            updated.push(newResultMessage);
            messageDeduplicationService.registerMessage(newResultMessage, effectiveConversationId);
          } else {
            console.log(`🔍 Skipping duplicate result message for job ${job.job_id}`);
          }
        }
        
        // Simple ordering without aggressive deduplication since PostgreSQL is the source of truth
        const orderedMessages = messageDeduplicationService.orderMessages(updated, effectiveConversationId);
        return orderedMessages;
      });
      
      // Remove from executing plans when job completes
      setExecutingPlans(prev => {
        const newSet = new Set(prev);
        newSet.delete(job.job_id);
        return newSet;
      });
      
      // Auto-scroll handled by ChatMessagesArea.js - no duplicate scrolling here
    } else {
      console.warn(`⚠️ Job ${job.job_id} completed but has no valid result to process`);
    }
  };

  // Handle plan execution
  // ROOSEVELT: Execute Plan logic removed - pure LangGraph uses simple Yes/No responses

  // Handle copy message with smart formatting
  const handleCopyMessage = async (message, asPlainText = false) => {
    let success;
    
    if (asPlainText) {
      // User explicitly requested plain text
      success = await copyAsPlainText(message.content);
    } else {
      // Use smart copy for rich text (HTML) when possible
      success = await smartCopy(message.content);
    }
    
    if (success) {
      setCopiedMessageId(message.id);
      setTimeout(() => {
        setCopiedMessageId(null);
      }, 2000);
    } else {
      alert(`Failed to copy message ${asPlainText ? 'as plain text' : ''} to clipboard`);
    }
  };

  // Handle copy message as plain text (for office applications)
  const handleCopyAsPlainText = async (message) => {
    return await handleCopyMessage(message, true);
  };

  // Handle save as markdown
  const handleSaveAsMarkdown = async (message) => {
    if (!currentConversationId || !message.message_id) {
      console.error('Cannot save message: missing conversation ID or message ID');
      return;
    }

    try {
      setSavingNoteFor(message.id);
      
      // Get the conversation details to use as context
      const conversation = await apiService.getConversation(currentConversationId);
      const conversationTitle = conversation?.title || 'Chat Conversation';
      
      // Create a filename based on conversation title and message timestamp
      const timestamp = new Date().toISOString().split('T')[0];
      const sanitizedTitle = conversationTitle.replace(/[^a-zA-Z0-9\s-]/g, '').replace(/\s+/g, '-');
      const filename = `${sanitizedTitle}-${timestamp}.md`;
      
      // Create markdown content
      const markdownContent = `# ${conversationTitle}

**Date:** ${new Date().toLocaleDateString()}
**Time:** ${new Date().toLocaleTimeString()}
**Message Type:** ${message.role === 'user' ? 'User Question' : 'Assistant Response'}

## Message Content

${message.content}

---
*Saved from conversation: ${conversationTitle}*
`;

      // Create the markdown file using the existing note creation API
      const noteData = {
        title: filename.replace('.md', ''),
        content: markdownContent,
        category: 'chat-export',
        tags: ['chat', 'export', message.role]
      };
      
      const result = await apiService.createNote(noteData);
      console.log('Message saved as markdown:', result);
      alert('Message saved as markdown file successfully!');
    } catch (error) {
      console.error('Failed to save message as markdown:', error);
      alert('Failed to save message as markdown. Please try again.');
    } finally {
      setSavingNoteFor(null);
    }
  };

  // Handle conversation selection
  const handleConversationSelect = (conversation) => {
    const conversationId = conversation?.conversation_id || null;
    console.log('🔄 Selecting conversation:', conversationId);
    
    // Clear deduplication data for the previous conversation
    if (currentConversationId) {
      messageDeduplicationService.clearConversation(currentConversationId);
    }
    
    // CRITICAL: Disconnect all WebSocket connections before switching
    if (backgroundJobService) {
      console.log('🔌 Disconnecting all WebSocket connections before conversation switch');
      backgroundJobService.disconnectAll();
      backgroundJobService.clearCompletedJobs();
      // Set the new conversation ID in the background job service
      backgroundJobService.setCurrentConversationId(conversationId);
    }
    
    // Clear state immediately to prevent cross-chat message display
    setMessages([]);
    setExecutingPlans(new Set());
    
    // Set new conversation ID after cleanup (null is valid for clearing current conversation)
    setCurrentConversationId(conversationId);
    lastLoadedConversationRef.current = null;
    
    // Add a small delay to ensure WebSocket disconnections are processed
    setTimeout(() => {
      console.log('🔄 Conversation switch completed for:', conversationId);
    }, 100);
  };

  // Clear chat
  const handleClearChat = () => {
    // CRITICAL: Disconnect all WebSocket connections before clearing
    if (backgroundJobService) {
      console.log('🔌 Disconnecting all WebSocket connections before clearing chat');
      backgroundJobService.disconnectAll();
      backgroundJobService.clearCompletedJobs();
    }
    
    // Clear state immediately
    setMessages([]);
    setExecutingPlans(new Set());
    setCurrentConversationId(null);
    lastLoadedConversationRef.current = null;
  };

  // Create new conversation
  const handleNewConversation = async () => {
    if (state.isLoading) return;

    try {
      setIsLoading(true);
      const response = await apiService.createConversation({
        title: 'New Conversation',
        description: 'New chat conversation'
      });
      
      const conversationId = response.conversation.conversation_id;
      setCurrentConversationId(conversationId);
      setMessages([]);
      setQuery('');
      setSessionId(`session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
      lastLoadedConversationRef.current = conversationId;
      
      // Clear executing plans when creating new conversation
      setExecutingPlans(new Set());
      
      queryClient.invalidateQueries(['conversations']);
    } catch (error) {
      console.error('❌ Failed to create new conversation:', error);
      setCurrentConversationId(null);
      setMessages([]);
      setQuery('');
      setSessionId(`session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
      lastLoadedConversationRef.current = null;
    } finally {
      setIsLoading(false);
    }
  };

  // Fallback helper function
  const createNewConversation = async (initialMessage) => {
    try {
      const response = await apiService.createConversation({
        title: 'New Conversation',
        description: 'New chat conversation'
      });
      return response.conversation.conversation_id;
    } catch (error) {
      console.error('Failed to create conversation:', error);
      return null;
    }
  };

  return {
    initializeBackgroundJobService,
    handleBackgroundJobCompleted,

    handleCopyMessage,
    handleCopyAsPlainText,
    handleSaveAsMarkdown,
    handleConversationSelect,
    handleClearChat,
    handleNewConversation,
    createNewConversation,
  };
};
