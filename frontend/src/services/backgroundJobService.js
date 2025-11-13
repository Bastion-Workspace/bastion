/**
 * Background Job Service - Handles background research jobs and reconnection
 */

import apiService from './apiService';

class BackgroundJobService {
  constructor(apiService) {
    this.apiService = apiService;
    this.jobWebSockets = new Map();
    this.jobCallbacks = new Map();
    this.jobConversationIds = new Map();
    this.activeJobs = new Map();
    this.completedJobs = new Set(); // Track completed jobs to prevent polling
    this.currentConversationId = null;
  }

  /**
   * Submit a research job for background processing
   */
  async submitBackgroundJob(query, sessionId, conversationId, executionMode = 'execute') {
    try {
      const response = await apiService.post('/api/background-chat/submit', {
        query,
        session_id: sessionId,
        conversation_id: conversationId,
        execution_mode: executionMode
      });

      const jobId = response.job_id;
      
      // Track the job
      this.activeJobs.set(jobId, {
        jobId,
        query,
        sessionId,
        conversationId,
        executionMode,
        status: 'submitted',
        submittedAt: new Date()
      });

      console.log('🎯 Background job submitted:', jobId);
      return jobId;
    } catch (error) {
      console.error('❌ Failed to submit background job:', error);
      throw error;
    }
  }

  /**
   * Get status of a specific background job
   */
  async getJobStatus(jobId) {
    try {
      return await apiService.get(`/api/background-chat/job/${jobId}`);
    } catch (error) {
      console.error(`❌ Failed to get job status for ${jobId}:`, error);
      return null;
    }
  }

  /**
   * Get ongoing jobs for a conversation
   */
  async getOngoingJobsForConversation(conversationId) {
    try {
      const response = await apiService.get(`/api/background-chat/conversation/${conversationId}/ongoing`);
      return response.ongoing_jobs || [];
    } catch (error) {
      console.error(`❌ Failed to get ongoing jobs for conversation ${conversationId}:`, error);
      return [];
    }
  }

  /**
   * Cancel a background job
   */
  async cancelJob(jobId) {
    try {
      const response = await apiService.post(`/api/background-chat/job/${jobId}/cancel`);
      
      // Remove from active jobs
      this.activeJobs.delete(jobId);
      
      // Close WebSocket if connected
      if (this.jobWebSockets.has(jobId)) {
        this.jobWebSockets.get(jobId).close();
        this.jobWebSockets.delete(jobId);
      }

      return response;
    } catch (error) {
      console.error(`❌ Failed to cancel job ${jobId}:`, error);
      throw error;
    }
  }

  /**
   * Connect to job progress WebSocket
   */
  connectToJobProgress(jobId, callbacks = {}, expectedConversationId = null) {
    try {
      // Close existing connection if any
      if (this.jobWebSockets.has(jobId)) {
        const existingWs = this.jobWebSockets.get(jobId);
        console.log(`🔌 Closing existing WebSocket for job ${jobId}`);
        existingWs.close();
        this.jobWebSockets.delete(jobId);
      }

      // Validate conversation context before creating connection
      if (expectedConversationId && !this.shouldProcessJobCompletion(expectedConversationId)) {
        console.warn(`⚠️ Refusing to create WebSocket for job ${jobId} - conversation context mismatch`);
        if (callbacks.onError) callbacks.onError(jobId, 'Conversation context mismatch');
        return;
      }

      // Get authentication token
      const token = this.apiService.getToken();
      if (!token) {
        console.error('❌ No authentication token available for WebSocket connection');
        if (callbacks.onError) callbacks.onError(jobId, 'No authentication token');
        return;
      }

      const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws/job-progress/${jobId}?token=${encodeURIComponent(token)}`;
      const ws = new WebSocket(wsUrl);

      // Store the expected conversation ID and creation timestamp with the WebSocket
      ws.expectedConversationId = expectedConversationId;
      ws.createdAt = new Date();
      ws.jobId = jobId;

      // Set up connection timeout
      let connectionTimeout = setTimeout(() => {
        console.warn(`⚠️ WebSocket connection timeout for job ${jobId}, starting polling fallback`);
        startPolling();
      }, 10000); // 10 second timeout

      // Set up polling fallback in case WebSocket fails
      let pollingInterval = null;
      let pollingAttempts = 0;
      const maxPollingAttempts = 60; // 5 minutes with 5-second intervals
      
      const startPolling = () => {
        console.log(`🔄 Starting polling fallback for job ${jobId}`);
        pollingInterval = setInterval(async () => {
          pollingAttempts++;
          try {
            const jobStatus = await this.getJobStatus(jobId);
            if (jobStatus && jobStatus.status === 'completed' && jobStatus.result) {
              console.log(`✅ Job ${jobId} completed via polling fallback`);
              clearInterval(pollingInterval);
              
              // Mark job as completed to prevent further polling
              this.completedJobs.add(jobId);
              
              // Process completion
              if (callbacks.onCompletion) {
                const jobData = {
                  job_id: jobId,
                  result: jobStatus.result,
                  query: jobStatus.query || this.activeJobs.get(jobId)?.query,
                  conversation_id: jobStatus.conversation_id
                };
                
                if (jobData.result && (jobData.result.answer || jobData.result.research_plan)) {
                  callbacks.onCompletion(jobData);
                }
              }
            } else if (pollingAttempts >= maxPollingAttempts) {
              console.warn(`⚠️ Polling timeout for job ${jobId}`);
              clearInterval(pollingInterval);
              this.completedJobs.add(jobId); // Mark as completed to prevent further attempts
              if (callbacks.onError) callbacks.onError(jobId, 'Polling timeout');
            }
          } catch (error) {
            console.error(`❌ Polling error for job ${jobId}:`, error);
            if (pollingAttempts >= maxPollingAttempts) {
              clearInterval(pollingInterval);
              this.completedJobs.add(jobId); // Mark as completed to prevent further attempts
              if (callbacks.onError) callbacks.onError(jobId, 'Polling failed');
            }
          }
        }, 5000); // Poll every 5 seconds
      };

      ws.onopen = () => {
        console.log(`📡 Connected to job progress for ${jobId}${expectedConversationId ? ` (conversation: ${expectedConversationId})` : ''}`);
        console.log(`📡 WebSocket ready state: ${ws.readyState}`);
        clearTimeout(connectionTimeout); // Clear connection timeout
        if (callbacks.onConnect) callbacks.onConnect(jobId);
      };

      ws.onerror = (error) => {
        console.error(`❌ WebSocket error for job ${jobId}:`, error);
        clearTimeout(connectionTimeout); // Clear connection timeout
        // Start polling fallback when WebSocket fails
        startPolling();
        if (callbacks.onError) callbacks.onError(jobId, error);
      };

      ws.onclose = (event) => {
        console.log(`📡 Disconnected from job progress for ${jobId}. Code: ${event.code}, Reason: ${event.reason}`);
        clearTimeout(connectionTimeout); // Clear connection timeout
        
        // Only start polling if WebSocket closed unexpectedly and we haven't already completed
        if (event.code !== 1000 && !this.completedJobs.has(jobId)) {
          console.log(`🔄 WebSocket closed unexpectedly, starting polling fallback for job ${jobId}`);
          startPolling();
        } else {
          console.log(`✅ WebSocket closed normally for job ${jobId}, no polling needed`);
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log(`📡 Received job progress for ${jobId}:`, data);
          
          if (data.type === 'job_completed') {
            console.log(`✅ Job ${jobId} completed via WebSocket`);
            clearTimeout(connectionTimeout); // Clear connection timeout
            
            // Mark job as completed to prevent polling
            this.completedJobs.add(jobId);
            
            // Process completion
            if (callbacks.onCompletion) {
              const jobData = {
                job_id: jobId,
                result: data.result,
                query: data.query || this.activeJobs.get(jobId)?.query,
                conversation_id: data.conversation_id
              };
              
              if (jobData.result && (jobData.result.answer || jobData.result.research_plan)) {
                callbacks.onCompletion(jobData);
              }
            }
            
            // Close WebSocket after processing
            ws.close(1000, 'Job completed');
          } else if (data.type === 'job_progress') {
            if (callbacks.onProgress) {
              callbacks.onProgress(jobId, data.progress);
            }
          } else if (data.type === 'job_error') {
            console.error(`❌ Job ${jobId} error via WebSocket:`, data.error);
            clearTimeout(connectionTimeout); // Clear connection timeout
            
            // Mark job as completed to prevent polling
            this.completedJobs.add(jobId);
            
            if (callbacks.onError) {
              callbacks.onError(jobId, data.error);
            }
            
            // Close WebSocket after processing
            ws.close(1000, 'Job error');
          }
        } catch (error) {
          console.error(`❌ Failed to parse WebSocket message for job ${jobId}:`, error);
        }
      };

      // Store WebSocket connection
      this.jobWebSockets.set(jobId, ws);
      
      // Store the expected conversation ID for validation
      this.jobConversationIds.set(jobId, expectedConversationId);
      
      // Store job info for reference
      this.activeJobs.set(jobId, {
        query: callbacks.query || 'Unknown query',
        conversationId: expectedConversationId,
        createdAt: new Date()
      });

      this.jobCallbacks.set(jobId, callbacks);

      return ws;
    } catch (error) {
      console.error(`❌ Failed to connect to job progress for ${jobId}:`, error);
      throw error;
    }
  }

  /**
   * Connect to conversation jobs WebSocket
   */
  connectToConversationJobs(conversationId, callbacks = {}) {
    try {
      const token = this.apiService.getToken();
      if (!token) {
        console.error('❌ No authentication token available for conversation WebSocket');
        if (callbacks.onError) callbacks.onError(conversationId, 'No authentication token');
        return;
      }
      
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws/conversation-jobs/${conversationId}?token=${encodeURIComponent(token)}`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log(`📡 Connected to conversation jobs for ${conversationId}`);
        if (callbacks.onConnect) callbacks.onConnect(conversationId);
      };

      ws.onmessage = (event) => {
        try {
          const update = JSON.parse(event.data);
          
          if (update.type === 'ongoing_jobs') {
            console.log(`📋 Ongoing jobs for conversation ${conversationId}:`, update.jobs);
            if (callbacks.onOngoingJobs) callbacks.onOngoingJobs(conversationId, update.jobs);
          } else if (update.type === 'background_job_completed') {
            console.log(`✅ Job completed in conversation ${conversationId}:`, update.job_id);
            if (callbacks.onJobCompleted) callbacks.onJobCompleted(conversationId, update);
          }
        } catch (error) {
          console.error('❌ Error parsing conversation WebSocket message:', error);
        }
      };

      ws.onclose = () => {
        console.log(`📡 Disconnected from conversation jobs for ${conversationId}`);
        if (callbacks.onDisconnect) callbacks.onDisconnect(conversationId);
      };

      ws.onerror = (error) => {
        console.error(`❌ Conversation WebSocket error for ${conversationId}:`, error);
        if (callbacks.onError) callbacks.onError(conversationId, error);
      };

      return ws;
    } catch (error) {
      console.error(`❌ Failed to connect to conversation jobs for ${conversationId}:`, error);
      throw error;
    }
  }

  /**
   * Check for ongoing jobs when returning to a conversation
   */
  async checkAndReconnectToOngoingJobs(conversationId, onJobFound) {
    try {
      console.log(`🔍 Checking for jobs in conversation ${conversationId}`);
      
      // First check for ongoing jobs
      const ongoingJobs = await apiService.get(`/api/background-chat/conversation/${conversationId}/ongoing`);
      
      if (ongoingJobs.length > 0) {
        console.log(`🔄 Found ${ongoingJobs.length} ongoing jobs for conversation ${conversationId}`);
        
        for (const job of ongoingJobs) {
          // CRITICAL: Validate that job belongs to the current conversation
          if (job.conversation_id !== conversationId) {
            console.warn(`⚠️ Skipping ongoing job ${job.job_id} - belongs to conversation ${job.conversation_id}, not ${conversationId}`);
            continue;
          }
          
          // Notify caller about found job
          if (onJobFound) {
            onJobFound(job);
          }
          
          // Connect to job progress with conversation context
          this.connectToJobProgress(job.job_id, {
            onCompletion: (jobData) => {
              // Job completed while we were away - notify caller with full job data
              const completedJob = {
                ...job,
                status: 'completed',
                result: jobData.result,
                progress: jobData.progress
              };
              console.log('✅ Job completed during reconnection:', completedJob);
              if (onJobFound) {
                onJobFound(completedJob);
              }
            },
            onProgress: (jobId, progress) => {
              console.log(`🔄 Reconnected job progress for ${jobId}:`, progress);
            }
          }, conversationId); // Pass conversation ID for validation
        }
      }
      
      return ongoingJobs;
    } catch (error) {
      console.error(`❌ Failed to check jobs for conversation ${conversationId}:`, error);
      return [];
    }
  }

  /**
   * Disconnect from all job WebSockets
   */
  disconnectAll() {
    console.log(`🔌 Disconnecting ${this.jobWebSockets.size} WebSocket connections`);
    for (const [jobId, ws] of this.jobWebSockets.entries()) {
      console.log(`🔌 Closing WebSocket for job ${jobId}`);
      ws.close();
    }
    this.jobWebSockets.clear();
    this.jobCallbacks.clear();
    this.activeJobs.clear();
    this.completedJobs.clear(); // Clear completed jobs tracking
  }

  /**
   * Check if a job WebSocket is still active
   */
  isJobWebSocketActive(jobId) {
    return this.jobWebSockets.has(jobId);
  }

  /**
   * Get the expected conversation ID for a job WebSocket
   */
  getJobWebSocketConversationId(jobId) {
    const ws = this.jobWebSockets.get(jobId);
    return ws ? ws.expectedConversationId : null;
  }

  /**
   * Set the current active conversation ID
   */
  setCurrentConversationId(conversationId) {
    console.log(`🔄 BackgroundJobService: Setting current conversation to ${conversationId}`);
    this.currentConversationId = conversationId;
  }

  /**
   * Get the current active conversation ID
   */
  getCurrentConversationId() {
    return this.currentConversationId;
  }

  /**
   * Check if a job completion should be processed based on current conversation context
   */
  shouldProcessJobCompletion(jobConversationId) {
    // If no current conversation ID, allow processing (new conversation scenario)
    if (!this.currentConversationId) {
      console.log(`✅ Allowing job completion for conversation ${jobConversationId} - no current conversation set`);
      return true;
    }
    
    // If job conversation matches current conversation, allow processing
    if (jobConversationId === this.currentConversationId) {
      console.log(`✅ Allowing job completion for conversation ${jobConversationId} - matches current conversation`);
      return true;
    }
    
    // RELAXED: Allow processing if current conversation is null or undefined
    // This handles cases where the conversation context might be in transition
    if (!this.currentConversationId || this.currentConversationId === 'null' || this.currentConversationId === 'undefined') {
      console.log(`✅ Allowing job completion for conversation ${jobConversationId} - current conversation is null/undefined`);
      return true;
    }
    
    // RELAXED: Allow processing if the job conversation is a substring of current conversation
    // This handles cases where conversation IDs might have slight variations
    if (this.currentConversationId.includes(jobConversationId) || jobConversationId.includes(this.currentConversationId)) {
      console.log(`✅ Allowing job completion for conversation ${jobConversationId} - substring match with current ${this.currentConversationId}`);
      return true;
    }
    
    console.warn(`⚠️ Rejecting job completion for conversation ${jobConversationId} - doesn't match current ${this.currentConversationId}`);
    return false;
  }

  /**
   * Get user's job history
   */
  async getJobHistory(limit = 50) {
    try {
      const response = await apiService.get(`/api/background-chat/history?limit=${limit}`);
      return response.job_history || [];
    } catch (error) {
      console.error('❌ Failed to get job history:', error);
      return [];
    }
  }

  /**
   * Clear completed jobs tracking (useful when switching conversations)
   */
  clearCompletedJobs() {
    this.completedJobs.clear();
  }

  /**
   * Mark a job as completed to prevent further polling
   */
  markJobCompleted(jobId) {
    this.completedJobs.add(jobId);
    console.log(`✅ Marked job ${jobId} as completed to prevent further polling`);
  }
}

export default BackgroundJobService;