import React, { useState, useCallback, useRef } from 'react';
import { backgroundJobService } from '../services/backgroundJobService';
import { messageDeduplicationService } from '../services/messageDeduplicationService';

/**
 * Unified Chat Manager Hook
 * Simplified chat management using the unified background processing API
 * All messages go through background service with integrated classification
 */
export const useChatManagerUnified = (conversationId, sessionId) => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [pendingJobs, setPendingJobs] = useState(new Map());
  const [error, setError] = useState(null);
  
  // WebSocket connection for job notifications
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  
  // Initialize WebSocket connection for job notifications
  const initializeWebSocket = useCallback(() => {
    if (!sessionId) return;
    
    try {
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws/${sessionId}`;
      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        console.log('🔌 WebSocket connected for unified chat');
        setError(null);
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (e) {
          console.error('❌ Failed to parse WebSocket message:', e);
        }
      };
      
      wsRef.current.onclose = () => {
        console.log('🔌 WebSocket disconnected, attempting reconnect...');
        // Attempt reconnection after 3 seconds
        reconnectTimeoutRef.current = setTimeout(initializeWebSocket, 3000);
      };
      
      wsRef.current.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
      };
      
    } catch (error) {
      console.error('❌ Failed to initialize WebSocket:', error);
    }
  }, [sessionId]);
  
  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback((data) => {
    if (data.type === 'background_job_completed') {
      handleJobCompletion(data);
    } else if (data.type === 'background_job_progress') {
      handleJobProgress(data);
    }
  }, []);
  
  // Handle job completion
  const handleJobCompletion = useCallback((data) => {
    const { job_id, result, query } = data;
    
    console.log('✅ Job completed:', job_id, result);
    
    // Remove from pending jobs
    setPendingJobs(prev => {
      const updated = new Map(prev);
      updated.delete(job_id);
      return updated;
    });
    
    // Add assistant response to messages
    if (result && result.success) {
      const assistantMessage = {
        id: `assistant_${Date.now()}`,
        type: 'assistant',
        content: result.answer || result.research_plan || 'Response completed',
        citations: result.citations || [],
        timestamp: new Date().toISOString(),
        execution_mode: result.execution_mode,
        processing_time: result.processing_time
      };
      
      setMessages(prev => {
        // Check for duplicates
        if (messageDeduplicationService.isDuplicate(assistantMessage, prev)) {
          console.log('🔍 Duplicate assistant message detected, skipping');
          return prev;
        }
        
        return [...prev, assistantMessage];
      });
    } else {
      // Handle error case
      const errorMessage = {
        id: `error_${Date.now()}`,
        type: 'assistant',
        content: `Error: ${result?.error || 'Unknown error occurred'}`,
        citations: [],
        timestamp: new Date().toISOString(),
        isError: true
      };
      
      setMessages(prev => [...prev, errorMessage]);
    }
    
    setIsLoading(false);
  }, []);
  
  // Handle job progress updates
  const handleJobProgress = useCallback((data) => {
    const { job_id, progress } = data;
    
    setPendingJobs(prev => {
      const updated = new Map(prev);
      const job = updated.get(job_id);
      if (job) {
        updated.set(job_id, {
          ...job,
          progress: progress
        });
      }
      return updated;
    });
  }, []);
  
  // Send message using unified API
  const sendMessage = useCallback(async (query) => {
    if (!query.trim()) return;
    
    setError(null);
    setIsLoading(true);
    
    // Add user message immediately
    const userMessage = {
      id: `user_${Date.now()}`,
      type: 'user',
      content: query,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => {
      // Check for duplicates
      if (messageDeduplicationService.isDuplicate(userMessage, prev)) {
        console.log('🔍 Duplicate user message detected, skipping');
        return prev;
      }
      
      return [...prev, userMessage];
    });
    
    try {
      // Get recent conversation context for classification
      const conversationContext = messages.slice(-6).map(msg => ({
        query: msg.type === 'user' ? msg.content : '',
        answer: msg.type === 'assistant' ? msg.content : ''
      })).filter(ctx => ctx.query || ctx.answer);
      
      // Submit to unified API
      const response = await fetch('/api/unified-chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          conversation_id: conversationId,
          session_id: sessionId,
          conversation_context: conversationContext,
          priority: 5
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.success) {
        console.log(`🎯 Job submitted: ${result.job_id} (${result.execution_mode})`);
        
        // Add to pending jobs
        setPendingJobs(prev => new Map(prev).set(result.job_id, {
          id: result.job_id,
          query,
          execution_mode: result.execution_mode,
          confidence: result.confidence,
          reasoning: result.reasoning,
          submitted_at: new Date().toISOString(),
          progress: null
        }));
        
        // Add pending message
        const pendingMessage = {
          id: `pending_${result.job_id}`,
          type: 'assistant',
          content: getPendingMessage(result.execution_mode, result.reasoning),
          isPending: true,
          job_id: result.job_id,
          execution_mode: result.execution_mode,
          timestamp: new Date().toISOString()
        };
        
        setMessages(prev => [...prev, pendingMessage]);
        
      } else {
        throw new Error(result.error || 'Failed to submit message');
      }
      
    } catch (error) {
      console.error('❌ Failed to send message:', error);
      setError(error.message);
      setIsLoading(false);
      
      // Add error message
      const errorMessage = {
        id: `error_${Date.now()}`,
        type: 'assistant',
        content: `Error: ${error.message}`,
        citations: [],
        timestamp: new Date().toISOString(),
        isError: true
      };
      
      setMessages(prev => [...prev, errorMessage]);
    }
  }, [conversationId, sessionId, messages]);
  
  // Get appropriate pending message based on execution mode
  const getPendingMessage = (executionMode, reasoning) => {
    switch (executionMode) {
      case 'chat':
        return '💬 Generating response...';
      case 'direct':
        return '🔍 Searching knowledge base...';
      case 'plan':
        return '📋 Analyzing and planning research...';
      case 'execute':
        return '🚀 Executing research plan...';
      default:
        return `🔄 Processing (${reasoning})...`;
    }
  };
  
  // Clear messages
  const clearMessages = useCallback(() => {
    setMessages([]);
    setPendingJobs(new Map());
    setError(null);
  }, []);
  
  // Cancel a pending job
  const cancelJob = useCallback(async (jobId) => {
    try {
      const response = await fetch(`/api/jobs/${jobId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        setPendingJobs(prev => {
          const updated = new Map(prev);
          updated.delete(jobId);
          return updated;
        });
        
        // Remove pending message
        setMessages(prev => prev.filter(msg => msg.job_id !== jobId));
      }
    } catch (error) {
      console.error('❌ Failed to cancel job:', error);
    }
  }, []);
  
  // Initialize WebSocket on mount
  React.useEffect(() => {
    initializeWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [initializeWebSocket]);
  
  return {
    messages,
    isLoading,
    error,
    pendingJobs: Array.from(pendingJobs.values()),
    sendMessage,
    clearMessages,
    cancelJob
  };
};
