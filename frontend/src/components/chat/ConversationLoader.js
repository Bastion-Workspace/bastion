import { useEffect } from 'react';
import { useQuery, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';

export const useConversationLoader = (currentConversationId, setMessages, setExecutingPlans, backgroundJobService, sessionId, lastLoadedConversationRef, messages) => {
  const queryClient = useQueryClient();

  // SIMPLIFIED: Only handle background job reconnection
  // Message loading is now handled by ChatSidebarContext
  
  useEffect(() => {
    if (backgroundJobService && currentConversationId) {
      console.log('🔄 ConversationLoader: Checking for ongoing jobs for conversation:', currentConversationId);
      
      backgroundJobService.checkAndReconnectToOngoingJobs(
                currentConversationId,
                (job) => {
          console.log('🔍 ConversationLoader: Found ongoing job:', job);
          // Only process jobs that belong to the current conversation
                  if (job.conversation_id === currentConversationId) {
            console.log('✅ ConversationLoader: Reconnecting to job:', job.job_id);
            // The job will be handled by the background job service
                  } else {
            console.warn(`⚠️ ConversationLoader: Skipping job ${job.job_id} - belongs to conversation ${job.conversation_id}, not ${currentConversationId}`);
          }
        }
      );
    }
  }, [currentConversationId, backgroundJobService]);

  // Return minimal interface to maintain compatibility
  return {
    messagesLoading: false,
    messagesError: null,
    refetchMessages: () => {
      console.log('🔄 ConversationLoader: Refetch requested - delegating to ChatSidebarContext');
      // Invalidate queries to trigger refresh in ChatSidebarContext
      queryClient.invalidateQueries(['conversationMessages', currentConversationId]);
      queryClient.invalidateQueries(['conversation', currentConversationId]);
    }
  };
};
