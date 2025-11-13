import ApiServiceBase from '../base/ApiServiceBase';

class ChatService extends ApiServiceBase {
  // Chat-specific methods
  queryKnowledgeBase = async (query, sessionId = 'default', maxResults = 10, conversationId = null) => {
    return this.post('/api/query', {
      query,
      session_id: sessionId,
      conversation_id: conversationId,
      max_results: maxResults,
    });
  }

  mcpQueryKnowledgeBase = async (query, sessionId = 'default', executionMode = 'plan', conversationId = null) => {
    console.log('🔍 ApiService MCP call - conversationId:', conversationId);
    const payload = {
      query,
      session_id: sessionId,
      conversation_id: conversationId,
      execution_mode: executionMode,
    };
    console.log('🔍 ApiService MCP payload:', payload);
    return this.post('/api/mcp-query', payload);
  }

  classifyIntent = async (query, conversationContext = []) => {
    return this.post('/api/classify-intent', {
      query,
      conversation_context: conversationContext,
    });
  }

  // Chat methods
  sendMessage = async (conversationId, message, executionMode = 'auto') => {
    return this.post('/api/v2/chat/legacy/query', {
      query: message,
      conversation_id: conversationId,
      execution_mode: executionMode,
      session_id: 'default'
    });
  }

  // New unified chat method with local-first strategy
  sendUnifiedMessage = async (conversationId, message, sessionId = 'default', executionMode = 'auto') => {
    const payload = {
      query: message,
      conversation_id: conversationId,
      session_id: sessionId,
      execution_mode: executionMode
    };
    console.log('🔍 sendUnifiedMessage payload:', payload);
    return this.post('/api/v2/chat/unified', payload);
  }

  // LangGraph chat method
  sendLangGraphMessage = async (conversationId, message, sessionId = 'default', executionMode = 'auto') => {
    const payload = {
      query: message,
      conversation_id: conversationId,
      session_id: sessionId,
      execution_mode: executionMode
    };
    console.log('🔍 sendLangGraphMessage payload:', payload);
    return this.post('/api/langgraph/chat', payload);
  }

  // Unified background processing
  sendUnifiedMessageBackground = async (conversationId, message, sessionId = 'default') => {
    return this.post('/api/v2/chat/unified/background', {
      query: message,
      conversation_id: conversationId,
      session_id: sessionId
    });
  }

  // Cancel unified job
  cancelUnifiedJob = async (jobId) => {
    return this.post(`/api/v2/chat/unified/job/${jobId}/cancel`);
  }

  // ROOSEVELT: Execute Plan removed - LangGraph uses simple Yes/No responses
}

export default new ChatService();
