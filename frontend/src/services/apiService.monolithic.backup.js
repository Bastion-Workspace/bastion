// API Service for frontend-backend communication
const API_BASE = process.env.REACT_APP_API_URL || '';

class ApiService {
  constructor() {
    this.baseURL = API_BASE;
  }

  async request(url, options = {}) {
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    const fullURL = `${this.baseURL}${url}`;
    
    console.log('🌐 API Request:', {
      url: fullURL,
      method: config.method || 'GET',
      headers: Object.keys(config.headers),
      hasBody: !!config.body
    });

    const response = await fetch(fullURL, config);

    console.log('🌐 API Response:', {
      status: response.status,
      ok: response.ok,
      url: response.url
    });

    if (!response.ok) {
      const error = new Error(`HTTP error! status: ${response.status}`);
      try {
        const errorData = await response.json();
        error.response = { data: errorData };
        console.error('🌐 API Error:', errorData);
      } catch (e) {
        error.response = { data: { detail: response.statusText } };
        console.error('🌐 API Error (no JSON):', response.statusText);
      }
      throw error;
    }

    const result = await response.json();
    console.log('🌐 API Success:', result);
    return result;
  },

  get = async (url, options = {}) => {
    return this.request(url, { method: 'GET', ...options });
  },

  post = async (url, data, options = {}) => {
    return this.request(url, {
      method: 'POST',
      body: JSON.stringify(data),
      ...options,
    });
  },

  put = async (url, data, options = {}) => {
    return this.request(url, {
      method: 'PUT',
      body: JSON.stringify(data),
      ...options,
    });
  },

  delete = async (url, options = {}) => {
    return this.request(url, { method: 'DELETE', ...options });
  },

  // Authentication methods
  getToken = () => {
    return localStorage.getItem('auth_token');
  },

  login = async (username, password) => {
    return this.post('/api/auth/login', {
      username,
      password,
    });
  },

  logout = async () => {
    return this.post('/api/auth/logout');
  },

  getCurrentUser = async () => {
    return this.get('/api/auth/me');
  },

  register = async (userData) => {
    return this.post('/api/auth/register', userData);
  },

  changePassword = async (currentPassword, newPassword) => {
    return this.post('/api/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
  },

  // Admin change password for any user
  adminChangePassword = async (userId, passwordData) => {
    return this.post(`/api/admin/users/${userId}/change-password`, passwordData);
  },

  // Chat-specific methods
  queryKnowledgeBase = async (query, sessionId = 'default', maxResults = 10, conversationId = null) => {
    return this.post('/api/query', {
      query,
      session_id: sessionId,
      conversation_id: conversationId,
      max_results: maxResults,
    });
  },

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
  },

  classifyIntent = async (query, conversationContext = []) => {
    return this.post('/api/classify-intent', {
      query,
      conversation_context: conversationContext,
    });
  },

  // Document methods
  getDocuments = async () => {
    return this.get('/api/documents');
  },

  getUserDocuments = async (offset = 0, limit = 100) => {
    return this.get(`/api/user/documents?offset=${offset}&limit=${limit}`);
  },

  uploadDocument = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    return this.request('/api/documents/upload', {
      method: 'POST',
      body: formData,
      headers: {} // Let browser set Content-Type for FormData
    });
  },

  uploadUserDocument = async (file, userId) => {
    const formData = new FormData();
    formData.append('file', file);
    if (userId) formData.append('user_id', userId);
    
    return this.request('/api/user/documents/upload', {
      method: 'POST',
      body: formData,
      headers: {} // Let browser set Content-Type for FormData
    });
  },

  importFromUrl = async (url) => {
    return this.post('/api/documents/import-url', { url });
  },

  deleteDocument = async (documentId) => {
    return this.delete(`/api/documents/${documentId}`);
  },

  reprocessDocument = async (documentId) => {
    return this.post(`/api/documents/${documentId}/reprocess`);
  },

  reprocessUserDocument = async (documentId) => {
    return this.post(`/api/user/documents/${documentId}/reprocess`);
  },

  updateDocument = async (documentId, updates) => {
    return this.put(`/api/documents/${documentId}`, updates);
  },

  updateDocumentMetadata = async (documentId, metadata) => {
    return this.put(`/api/documents/${documentId}/metadata`, metadata);
  },

  submitDocumentToGlobal = async (documentId, reason = null) => {
    return this.post(`/api/user/documents/${documentId}/submit`, {
      document_id: documentId,
      reason: reason
    });
  },

  // Document content retrieval
  getDocumentContent = async (documentId) => {
    try {
      const response = await this.request(`/api/documents/${documentId}/content`);
      return response;
    } catch (error) {
      console.error('Failed to get document content:', error);
      throw error;
    }
  },

  // Document management
  // Settings methods
  getSettings = async () => {
    return this.get('/api/settings');
  },

  updateSetting = async (key, value) => {
    return this.post('/api/settings', { key, value });
  },

  getAvailableModels = async () => {
    return this.get('/api/models/available');
  },

  getEnabledModels = async () => {
    return this.get('/api/models/enabled');
  },

  setEnabledModels = async (modelIds) => {
    return this.post('/api/models/enabled', { model_ids: modelIds });
  },

  getCurrentModel = async () => {
    return this.get('/api/models/current');
  },

  selectModel = async (modelName) => {
    return this.post('/api/models/select', { model_name: modelName });
  },


  clearQdrantDatabase = async () => {
    return this.post('/api/admin/clear-qdrant');
  },

  clearNeo4jDatabase = async () => {
    return this.post('/api/admin/clear-neo4j');
  },

  clearAllDocuments = async () => {
    return this.post('/api/admin/clear-documents');
  },

  uploadMultipleDocuments = async (files) => {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });
    
    return this.request('/api/documents/upload-multiple', {
      method: 'POST',
      body: formData,
      headers: {} // Let browser set Content-Type for FormData
    });
  },

  // Conversation methods
  createConversation = async (data = {}) => {
    return this.post('/api/conversations', {
      title: data.title || null,
      description: data.description || null,
      tags: data.tags || [],
      folder_id: data.folder_id || null,
      initial_message: data.initial_message || null
    });
  },

  reorderConversations = async (conversationIds, orderLocked = false) => {
    return this.post('/api/conversations/reorder', {
      conversation_ids: conversationIds,
      order_locked: orderLocked
    });
  },

  listConversations = async (skip = 0, limit = 50) => {
    return this.get(`/api/conversations?skip=${skip}&limit=${limit}`);
  },

  getConversation = async (conversationId) => {
    return this.get(`/api/conversations/${conversationId}`);
  },

  getConversationMessages = async (conversationId, skip = 0, limit = 100) => {
    return this.get(`/api/conversations/${conversationId}/messages?skip=${skip}&limit=${limit}`);
  },

  addMessageToConversation = async (conversationId, messageData) => {
    return this.post(`/api/conversations/${conversationId}/messages`, messageData);
  },

  updateConversation = async (conversationId, title, updates = {}) => {
    return this.put(`/api/conversations/${conversationId}`, { title, ...updates });
  },

  deleteConversation = async (conversationId) => {
    console.log('🌐 API: Deleting conversation:', conversationId);
    try {
      const result = await this.delete(`/api/conversations/${conversationId}`);
      console.log('🌐 API: Delete conversation result:', result);
      return result;
    } catch (error) {
      console.error('🌐 API: Delete conversation error:', error);
      throw error;
    }
  },

  deleteAllConversations = async () => {
    console.log('🌐 API: Deleting ALL conversations');
    try {
      const result = await this.delete('/api/conversations');
      console.log('🌐 API: Delete all conversations result:', result);
      return result;
    } catch (error) {
      console.error('🌐 API: Delete all conversations error:', error);
      throw error;
    }
  },

  // Chat methods
  sendMessage = async (conversationId, message, executionMode = 'auto') => {
    return this.post('/api/v2/chat/legacy/query', {
      query: message,
      conversation_id: conversationId,
      execution_mode: executionMode,
      session_id: 'default'
    });
  },

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
  },

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
  },

  // Unified background processing
  sendUnifiedMessageBackground = async (conversationId, message, sessionId = 'default') => {
    return this.post('/api/v2/chat/unified/background', {
      query: message,
      conversation_id: conversationId,
      session_id: sessionId
    });
  },

  // Cancel unified job
  cancelUnifiedJob = async (jobId) => {
    return this.post(`/api/v2/chat/unified/job/${jobId}/cancel`);
  },

  executePlan = async (data) => {
    return this.post('/api/execute-plan', data);
  },

  // Notes methods
  getNotes = async (skip = 0, limit = 100, category = null, tag = null, search = null) => {
    let url = `/api/notes?skip=${skip}&limit=${limit}`;
    if (category) url += `&category=${encodeURIComponent(category)}`;
    if (tag) url += `&tag=${encodeURIComponent(tag)}`;
    if (search) url += `&search=${encodeURIComponent(search)}`;
    return this.get(url);
  },

  getNote = async (noteId) => {
    return this.get(`/api/notes/${noteId}`);
  },

  createNote = async (noteData) => {
    return this.post('/api/notes', noteData);
  },

  updateNote = async (noteId, noteData) => {
    return this.put(`/api/notes/${noteId}`, noteData);
  },

  deleteNote = async (noteId) => {
    return this.delete(`/api/notes/${noteId}`);
  },

  searchNotes = async (query, limit = 100) => {
    return this.post('/api/notes/search', {
      query: query,
      limit: limit
    });
  },

  getNoteTags = async () => {
    return this.get('/api/notes/categories');
  },

  exportNotes = async () => {
    const response = await fetch(`${this.baseURL}/api/notes/export`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('auth_token') || ''}`
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.blob();
  },

  saveMessageAsNote = async (conversationId, messageId) => {
    return this.post(`/api/conversations/${conversationId}/messages/${messageId}/save-as-note`);
  },

  // User management methods
  getUsers = async () => {
    return this.get('/api/admin/users');
  },

  createUser = async (userData) => {
    return this.post('/api/admin/users', userData);
  },

  updateUser = async (userId, userData) => {
    return this.put(`/api/admin/users/${userId}`, userData);
  },

  deleteUser = async (userId) => {
    return this.delete(`/api/admin/users/${userId}`);
  },

  changePassword = async (userId, passwordData) => {
    return this.post(`/api/admin/users/${userId}/change-password`, passwordData);
  },

  // Pending submissions methods
  getPendingSubmissions = async () => {
    return this.get('/api/admin/pending-submissions');
  },

  reviewSubmission = async (documentId, action, comment = '') => {
    return this.post(`/api/admin/submissions/${documentId}/review`, {
      action,
      comment
    });
  },


  // ===== FOLDER MANAGEMENT METHODS =====

  getFolderTree = async (collectionType = 'user') => {
    return this.get(`/api/folders/tree?collection_type=${collectionType}`);
  },

  getFolderContents = async (folderId) => {
    return this.get(`/api/folders/${folderId}/contents`);
  },

  createFolder = async (folderData) => {
    return this.post('/api/folders', folderData);
  },

  updateFolder = async (folderId, folderData) => {
    return this.put(`/api/folders/${folderId}`, folderData);
  },

  deleteFolder = async (folderId, recursive = false) => {
    return this.delete(`/api/folders/${folderId}?recursive=${recursive}`);
  },

  moveFolder = async (folderId, newParentId = null) => {
    return this.post(`/api/folders/${folderId}/move`, { new_parent_id: newParentId });
  },

  createDefaultFolders = async () => {
    return this.post('/api/folders/default');
  },

  // ===== DOCUMENT CREATION METHODS =====

  createDocument = async (documentData) => {
    return this.post('/api/documents/create', documentData);
  },

  updateDocument = async (documentId, documentData) => {
    return this.put(`/api/documents/${documentId}`, documentData);
  },

  // User timezone methods
  getUserTimezone = async () => {
    return this.get('/api/settings/user/timezone');
  },

  setUserTimezone = async (timezoneData) => {
    return this.put('/api/settings/user/timezone', timezoneData);
  },

  // Prompt settings methods
  getPromptSettings = async () => {
    return this.get('/api/settings/prompt');
  },

  updatePromptSettings = async (settings) => {
    return this.post('/api/settings/prompt', settings);
  },

  getPromptOptions = async () => {
    return this.get('/api/settings/prompt/options');
  },

  // ===== TEMPLATE MANAGEMENT API =====
  
  // Get user's templates
  getUserTemplates = async () => {
    return this.get('/api/templates/');
  },

  // Get public/system templates
  getPublicTemplates = async () => {
    return this.get('/api/templates/public');
  },

  // Get specific template by ID
  getTemplate = async (templateId) => {
    return this.get(`/api/templates/${templateId}`);
  },

  // Create new template
  createTemplate = async (templateData) => {
    return this.post('/api/templates/', templateData);
  },

  // Update existing template
  updateTemplate = async (templateId, updates) => {
    return this.put(`/api/templates/${templateId}`, updates);
  },

  // Delete template
  deleteTemplate = async (templateId) => {
    return this.delete(`/api/templates/${templateId}`);
  },

  // Duplicate template
  duplicateTemplate = async (templateId, duplicateData) => {
    return this.post(`/api/templates/${templateId}/duplicate`, duplicateData);
  },

  // Search templates by keywords
  searchTemplates = async (keywords) => {
    return this.get(`/api/templates/search?keywords=${encodeURIComponent(keywords)}`);
  },

  // Get template statistics (admin only)
  getTemplateStats = async () => {
    return this.get('/api/templates/stats/overview');
  },

  // Get available field types
  getAvailableFieldTypes = async () => {
    return this.get('/api/templates/field-types/available');
  },

  // Validate template structure
  validateTemplate = async (templateData) => {
    return this.post('/api/templates/validate', templateData);
  },

  // Template execution methods
  executeTemplateResearch = async (conversationId, templateId, query, customInstructions = null) => {
    return this.post('/api/template-execution/execute', {
      conversation_id: conversationId,
      template_id: templateId,
      query: query,
      custom_instructions: customInstructions
    });
  },

  confirmTemplateUsage = async (conversationId, templateId, action, modifications = null) => {
    return this.post('/api/template-execution/confirm', {
      conversation_id: conversationId,
      template_id: templateId,
      action: action,
      modifications: modifications
    });
  },

  getTemplateExecutionStatus = async (conversationId) => {
    return this.get(`/api/template-execution/status/${conversationId}`);
  },

  generateTemplatePlan = async (templateId, query) => {
    return this.get(`/api/template-execution/plan/${templateId}?query=${encodeURIComponent(query)}`);
  }
}

const apiService = new ApiService();
export default apiService;
