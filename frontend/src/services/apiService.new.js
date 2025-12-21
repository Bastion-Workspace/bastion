// Unified API Service - Domain Service Orchestrator
import authService from './auth/AuthService';
import documentService from './document/DocumentService';
import chatService from './chat/ChatService';
import conversationService from './conversation/ConversationService';

// Import remaining methods from original service temporarily
// This allows gradual migration without breaking existing functionality
import { 
  // Notes methods
  getNotes,
  getNote,
  createNote,
  updateNote,
  deleteNote,
  searchNotes,
  getNoteTags,
  exportNotes,
  saveMessageAsNote,
  
  // Settings methods
  getSettings,
  updateSetting,
  getAvailableModels,
  getEnabledModels,
  setEnabledModels,
  getCurrentModel,
  selectModel,
  
  // Admin methods
  clearQdrantDatabase,
  clearNeo4jDatabase,
  clearAllDocuments,
  getUsers,
  createUser,
  updateUser,
  deleteUser,
  changePassword,
  
  // Folder methods
  getFolderTree,
  getFolderContents,
  createFolder,
  updateFolder,
  deleteFolder,
  moveFolder,
  createDefaultFolders,
  
  // User settings methods
  getUserTimezone,
  setUserTimezone,
  getPromptSettings,
  updatePromptSettings,
  getPromptOptions,
  
  // Template methods
  getUserTemplates,
  getPublicTemplates,
  getTemplate,
  createTemplate,
  updateTemplate as updateTemplateMethod,
  deleteTemplate,
  duplicateTemplate,
  searchTemplates,
  getTemplateStats,
  getAvailableFieldTypes,
  validateTemplate,
  executeTemplateResearch,
  confirmTemplateUsage,
  getTemplateExecutionStatus,
  generateTemplatePlan
} from './apiService';

class ApiService {
  constructor() {
    // Domain services
    this.auth = authService;
    this.documents = documentService;
    this.chat = chatService;
    this.conversations = conversationService;
  }

  // Temporarily expose remaining methods directly for backward compatibility
  // These will be moved to domain services in future phases

  // Notes methods
  getNotes = getNotes;
  getNote = getNote;
  createNote = createNote;
  updateNote = updateNote;
  deleteNote = deleteNote;
  searchNotes = searchNotes;
  getNoteTags = getNoteTags;
  exportNotes = exportNotes;
  saveMessageAsNote = saveMessageAsNote;

  // Settings methods  
  getSettings = getSettings;
  updateSetting = updateSetting;
  getAvailableModels = getAvailableModels;
  getEnabledModels = getEnabledModels;
  setEnabledModels = setEnabledModels;
  getCurrentModel = getCurrentModel;
  selectModel = selectModel;

  // Admin methods
  clearQdrantDatabase = clearQdrantDatabase;
  clearNeo4jDatabase = clearNeo4jDatabase;
  clearAllDocuments = clearAllDocuments;
  getUsers = getUsers;
  createUser = createUser;
  updateUser = updateUser;
  deleteUser = deleteUser;
  changePassword = changePassword;

  // Folder methods
  getFolderTree = getFolderTree;
  getFolderContents = getFolderContents;
  createFolder = createFolder;
  updateFolder = updateFolder;
  deleteFolder = deleteFolder;
  moveFolder = moveFolder;
  createDefaultFolders = createDefaultFolders;

  // User settings methods
  getUserTimezone = getUserTimezone;
  setUserTimezone = setUserTimezone;
  getPromptSettings = getPromptSettings;
  updatePromptSettings = updatePromptSettings;
  getPromptOptions = getPromptOptions;

  // Template methods
  getUserTemplates = getUserTemplates;
  getPublicTemplates = getPublicTemplates;
  getTemplate = getTemplate;
  createTemplate = createTemplate;
  updateTemplate = updateTemplateMethod;
  deleteTemplate = deleteTemplate;
  duplicateTemplate = duplicateTemplate;
  searchTemplates = searchTemplates;
  getTemplateStats = getTemplateStats;
  getAvailableFieldTypes = getAvailableFieldTypes;
  validateTemplate = validateTemplate;
  executeTemplateResearch = executeTemplateResearch;
  confirmTemplateUsage = confirmTemplateUsage;
  getTemplateExecutionStatus = getTemplateExecutionStatus;
  generateTemplatePlan = generateTemplatePlan;

  // Legacy method proxies for backward compatibility
  // These delegate to the appropriate domain service

  // Auth methods (delegate to auth service)
  getToken = () => this.auth.getToken();
  login = (username, password) => this.auth.login(username, password);
  logout = () => this.auth.logout();
  getCurrentUser = () => this.auth.getCurrentUser();
  register = (userData) => this.auth.register(userData);
  adminChangePassword = (userId, passwordData) => this.auth.adminChangePassword(userId, passwordData);

  // Document methods (delegate to document service)
  getDocuments = () => this.documents.getDocuments();
  getUserDocuments = (offset, limit) => this.documents.getUserDocuments(offset, limit);
  uploadDocument = (file) => this.documents.uploadDocument(file);
  uploadUserDocument = (file, userId) => this.documents.uploadUserDocument(file, userId);
  uploadMultipleDocuments = (files) => this.documents.uploadMultipleDocuments(files);
  importFromUrl = (url) => this.documents.importFromUrl(url);
  deleteDocument = (documentId) => this.documents.deleteDocument(documentId);
  reprocessDocument = (documentId) => this.documents.reprocessDocument(documentId);
  reprocessUserDocument = (documentId) => this.documents.reprocessUserDocument(documentId);
  updateDocument = (documentId, updates) => this.documents.updateDocument(documentId, updates);
  updateDocumentMetadata = (documentId, metadata) => this.documents.updateDocumentMetadata(documentId, metadata);
  getDocumentContent = (documentId) => this.documents.getDocumentContent(documentId);
  createDocument = (documentData) => this.documents.createDocument(documentData);

  // Chat methods (delegate to chat service)
  queryKnowledgeBase = (query, sessionId, maxResults, conversationId) => 
    this.chat.queryKnowledgeBase(query, sessionId, maxResults, conversationId);
  mcpQueryKnowledgeBase = (query, sessionId, executionMode, conversationId) => 
    this.chat.mcpQueryKnowledgeBase(query, sessionId, executionMode, conversationId);
  classifyIntent = (query, conversationContext) => this.chat.classifyIntent(query, conversationContext);
  sendMessage = (conversationId, message, executionMode) => 
    this.chat.sendMessage(conversationId, message, executionMode);
  sendUnifiedMessage = (conversationId, message, sessionId, executionMode) => 
    this.chat.sendUnifiedMessage(conversationId, message, sessionId, executionMode);
  sendLangGraphMessage = (conversationId, message, sessionId, executionMode) => 
    this.chat.sendLangGraphMessage(conversationId, message, sessionId, executionMode);
  sendUnifiedMessageBackground = (conversationId, message, sessionId) => 
    this.chat.sendUnifiedMessageBackground(conversationId, message, sessionId);
  cancelUnifiedJob = (jobId) => this.chat.cancelUnifiedJob(jobId);
  executePlan = (data) => this.chat.executePlan(data);

  // Conversation methods (delegate to conversation service)
  createConversation = (data) => this.conversations.createConversation(data);
  reorderConversations = (conversationIds, orderLocked) => 
    this.conversations.reorderConversations(conversationIds, orderLocked);
  listConversations = (skip, limit) => this.conversations.listConversations(skip, limit);
  getConversation = (conversationId) => this.conversations.getConversation(conversationId);
  getConversationMessages = (conversationId, skip, limit) => 
    this.conversations.getConversationMessages(conversationId, skip, limit);
  addMessageToConversation = (conversationId, messageData) => 
    this.conversations.addMessageToConversation(conversationId, messageData);
  updateConversation = (conversationId, title, updates) => 
    this.conversations.updateConversation(conversationId, title, updates);
  deleteConversation = (conversationId) => this.conversations.deleteConversation(conversationId);
  deleteAllConversations = () => this.conversations.deleteAllConversations();
}

const apiService = new ApiService();
export default apiService;
