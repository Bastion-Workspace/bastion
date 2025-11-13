import { useEffect, useRef, useCallback } from 'react';
import { useQuery } from 'react-query';
import apiService from '../../services/apiService';
import messageOrderingService from '../../services/messageOrderingService';

export const useImprovedConversationLoader = (
  currentConversationId, 
  setMessages, 
  setExecutingPlans, 
  backgroundJobService, 
  sessionId
) => {
  const loadingRef = useRef(false);
  const lastLoadedConversationRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Load conversation messages with improved error handling and ordering
  const { data: conversationMessages, isLoading: messagesLoading, error: messagesError, refetch: refetchMessages } = useQuery(
    ['conversationMessages', currentConversationId],
    async () => {
      if (!currentConversationId) return null;
      
      // Abort any previous request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      
      // Create new abort controller
      abortControllerRef.current = new AbortController();
      
      try {
        console.log(`🔄 Loading messages for conversation: ${currentConversationId}`);
        messageOrderingService.setLoadingState(currentConversationId, true);
        
        const data = await apiService.getConversationMessages(currentConversationId);
        
        console.log(`✅ Loaded ${data?.messages?.length || 0} messages from API`);
        return data;
      } catch (error) {
        if (error.name === 'AbortError') {
          console.log('🚫 Request aborted');
          return null;
        }
        throw error;
      } finally {
        messageOrderingService.setLoadingState(currentConversationId, false);
      }
    },
    {
      enabled: !!currentConversationId,
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
      staleTime: 30000, // Consider data fresh for 30 seconds
      cacheTime: 300000, // Keep in cache for 5 minutes
      retry: (failureCount, error) => {
        // Don't retry if it's an abort error
        if (error.name === 'AbortError') return false;
        return failureCount < 2;
      },
      onSuccess: async (data) => {
        if (!data?.messages || !currentConversationId) return;
        
        console.log(`🔄 Processing ${data.messages.length} messages for conversation ${currentConversationId}`);
        
        try {
          // Process messages through the ordering service
          const processedMessages = messageOrderingService.processConversationMessages(
            data.messages.map(msg => ({
              id: msg.message_id,
              message_id: msg.message_id,
              type: msg.message_type === 'user' ? 'user' : 'assistant',
              content: msg.content,
              citations: msg.citations || [],
              timestamp: new Date(msg.created_at),
              created_at: msg.created_at,
              sequence_number: msg.sequence_number,
              queryTime: msg.query_time,
              sessionId: currentConversationId,
              model_used: msg.model_used,
              metadata: msg.metadata_json || {}
            })),
            currentConversationId
          );
          
          // Set messages using the ordering service
          setMessages(prevMessages => {
            // If this is a different conversation, replace entirely
            if (lastLoadedConversationRef.current !== currentConversationId) {
              console.log(`🔄 Switching to conversation ${currentConversationId}, replacing messages`);
              lastLoadedConversationRef.current = currentConversationId;
              return processedMessages;
            }
            
            // Otherwise, merge with existing messages (for pending messages, etc.)
            const pendingMessages = prevMessages.filter(msg => 
              msg.isPending || msg.isResearchJob || !msg.message_id
            );
            
            if (pendingMessages.length > 0) {
              console.log(`🔄 Merging ${pendingMessages.length} pending messages with loaded messages`);
              return messageOrderingService.mergeMessages(
                processedMessages, 
                pendingMessages, 
                currentConversationId
              );
            }
            
            return processedMessages;
          });
          
          // Check for ongoing background jobs after a short delay
          setTimeout(async () => {
            if (currentConversationId === lastLoadedConversationRef.current && backgroundJobService) {
              await checkOngoingJobs();
            }
          }, 500);
          
        } catch (error) {
          console.error('❌ Error processing conversation messages:', error);
        }
      },
      onError: (error) => {
        console.error('❌ Failed to load conversation messages:', error);
        messageOrderingService.setLoadingState(currentConversationId, false);
      }
    }
  );

  // Check for ongoing background jobs
  const checkOngoingJobs = useCallback(async () => {
    if (!backgroundJobService || !currentConversationId) return;
    
    try {
      console.log('🔍 Checking for ongoing jobs...');
      await backgroundJobService.checkAndReconnectToOngoingJobs(
        currentConversationId,
        (job) => {
          console.log('🔍 Found ongoing job:', job);
          
          // Only process jobs that belong to the current conversation
          if (job.conversation_id === currentConversationId) {
            restorePlanStateFromJob(job);
          } else {
            console.warn(`⚠️ Skipping job ${job.job_id} - belongs to different conversation`);
          }
        }
      );
    } catch (error) {
      console.error('❌ Error checking ongoing jobs:', error);
    }
  }, [backgroundJobService, currentConversationId]);

  // Restore plan state from background job
  const restorePlanStateFromJob = useCallback((job) => {
    console.log(`🔄 Restoring plan state from job:`, {
      jobId: job.job_id,
      status: job.status,
      conversationId: job.conversation_id
    });
    
    if (job.conversation_id !== currentConversationId) {
      console.warn(`⚠️ Job ${job.job_id} belongs to different conversation, skipping`);
      return;
    }
    
    setMessages(prevMessages => {
      const updatedMessages = prevMessages.map(msg => {
        // Try to match the message to this job
        const isJobMatch = (
          (job.result?.research_plan && msg.content.includes(job.result.research_plan.substring(0, 100))) ||
          (job.result?.answer && msg.content.includes(job.result.answer.substring(0, 100))) ||
          (msg.metadata?.job_id === job.job_id)
        );
        
        if (isJobMatch) {
          console.log(`🔄 Updating message ${msg.id} with job state`);
          return {
            ...msg,
            isResearchResult: true,
            isResearchPlan: !!job.result?.research_plan && !job.result?.answer,
            planApproved: job.result?.plan_approved || false,
            isExecuting: job.status === 'running' || job.status === 'queued',
            jobId: job.job_id,
            query: job.query,
            execution_mode: job.result?.execution_mode || 'plan',
            metadata: {
              ...msg.metadata,
              job_id: job.job_id,
              plan_approved: job.result?.plan_approved || false,
              execution_mode: job.result?.execution_mode || 'plan'
            }
          };
        }
        return msg;
      });
      
      // Update executing plans state
      if (job.status === 'running' || job.status === 'queued') {
        setExecutingPlans(prev => new Set([...prev, job.job_id]));
      } else if (job.status === 'completed') {
        setExecutingPlans(prev => {
          const newSet = new Set(prev);
          newSet.delete(job.job_id);
          return newSet;
        });
      }
      
      return updatedMessages;
    });
  }, [currentConversationId, setExecutingPlans]);

  // Clear cache when conversation changes
  useEffect(() => {
    if (currentConversationId && lastLoadedConversationRef.current !== currentConversationId) {
      console.log(`🔄 Conversation changed to ${currentConversationId}, clearing cache`);
      
      // Clear cache for previous conversation
      if (lastLoadedConversationRef.current) {
        messageOrderingService.clearCache(lastLoadedConversationRef.current);
      }
      
      // Abort any ongoing request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    }
  }, [currentConversationId]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // Refetch messages when conversation changes
  useEffect(() => {
    if (currentConversationId && refetchMessages) {
      console.log('🔄 Triggering message refetch for conversation:', currentConversationId);
      refetchMessages();
    }
  }, [currentConversationId, refetchMessages]);

  return {
    conversationMessages,
    messagesLoading: messagesLoading || messageOrderingService.isLoading(currentConversationId),
    messagesError,
    refetchMessages,
    isLoadingState: loadingRef.current
  };
};
