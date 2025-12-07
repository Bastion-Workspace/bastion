// Unified API Service - Complete Domain Service Orchestrator
// Replaces the monolithic apiService.js with organized domain services

import authService from './auth/AuthService';
import documentService from './document/DocumentService';
import chatService from './chat/ChatService';
import conversationService from './conversation/ConversationService';
import settingsService from './settings/SettingsService';
import adminService from './admin/AdminService';
import folderService from './folder/FolderService';
import templateService from './template/TemplateService';
import integrationService from './integration/IntegrationService';
import audioService from './audio/AudioService';
import orgService from './org/OrgService';

class ApiService {
  constructor() {
    // Domain services - New organized approach
    this.auth = authService;
    this.documents = documentService;
    this.chat = chatService;
    this.conversations = conversationService;
    this.settings = settingsService;
    this.admin = adminService;
    this.folders = folderService;
    this.templates = templateService;
    this.integrations = integrationService;
    this.audio = audioService;
    this.org = orgService;
  }

  // ===== CORE HTTP METHODS =====
  // Expose base HTTP methods for services like rssService that need direct access
  request = (url, options = {}) => this.auth.request(url, options);
  get = (url, options = {}) => this.auth.get(url, options);
  post = (url, data, options = {}) => this.auth.post(url, data, options);
  put = (url, data, options = {}) => this.auth.put(url, data, options);
  delete = (url, options = {}) => this.auth.delete(url, options);

  // Legacy method proxies for backward compatibility
  // These delegate to the appropriate domain service to maintain existing functionality

  // ===== AUTH METHODS =====
  getToken = () => this.auth.getToken();
  login = (username, password) => this.auth.login(username, password);
  logout = () => this.auth.logout();
  getCurrentUser = () => this.auth.getCurrentUser();
  register = (userData) => this.auth.register(userData);
  changePassword = (currentPassword, newPassword) => this.auth.changePassword(currentPassword, newPassword);
  adminChangePassword = (userId, passwordData) => this.auth.adminChangePassword(userId, passwordData);

  // ===== DOCUMENT METHODS =====
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
  renameDocument = (documentId, newFilename) => this.documents.renameDocument(documentId, newFilename);
  moveDocument = (documentId, newFolderId, userId) => this.documents.moveDocument(documentId, newFolderId, userId);
  submitDocumentToGlobal = (documentId, reason) => this.documents.submitDocumentToGlobal(documentId, reason);
  getDocumentContent = (documentId) => this.documents.getDocumentContent(documentId);
  updateDocumentContent = (documentId, content) => this.documents.updateDocumentContent(documentId, content);
  createDocument = (documentData) => this.documents.createDocument(documentData);
  createDocumentFromContent = (args) => this.documents.createDocumentFromContent(args);
  exemptDocument = (documentId) => this.documents.exemptDocument(documentId);
  removeDocumentExemption = (documentId) => this.documents.removeDocumentExemption(documentId);

  // ===== CHAT METHODS =====
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
  // ROOSEVELT: Execute Plan removed - LangGraph uses simple Yes/No responses

  // ===== CONVERSATION METHODS =====
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

  // ===== NOTES METHODS =====
  getNotes = (skip, limit, category, tag, search) => this.notes.getNotes(skip, limit, category, tag, search);
  getNote = (noteId) => this.notes.getNote(noteId);
  createNote = (noteData) => this.notes.createNote(noteData);
  updateNote = (noteId, noteData) => this.notes.updateNote(noteId, noteData);
  deleteNote = (noteId) => this.notes.deleteNote(noteId);
  searchNotes = (query, limit) => this.notes.searchNotes(query, limit);
  getNoteTags = () => this.notes.getNoteTags();
  exportNotes = () => this.notes.exportNotes();


  // ===== SETTINGS METHODS =====
  getSettings = () => this.settings.getSettings();
  updateSetting = (key, value) => this.settings.updateSetting(key, value);
  setSetting = (key, payload) => this.settings.setSettingValue(key, payload);
  getAvailableModels = () => this.settings.getAvailableModels();
  getEnabledModels = () => this.settings.getEnabledModels();
  setEnabledModels = (modelIds) => this.settings.setEnabledModels(modelIds);
  getCurrentModel = () => this.settings.getCurrentModel();
  selectModel = (modelName) => this.settings.selectModel(modelName);
  getUserTimezone = () => this.settings.getUserTimezone();
  setUserTimezone = (timezoneData) => this.settings.setUserTimezone(timezoneData);
  getPromptSettings = () => this.settings.getPromptSettings();
  updatePromptSettings = (settings) => this.settings.updatePromptSettings(settings);
  getPromptOptions = () => this.settings.getPromptOptions();

  // ===== ADMIN METHODS =====
  clearQdrantDatabase = () => this.admin.clearQdrantDatabase();
  clearNeo4jDatabase = () => this.admin.clearNeo4jDatabase();
  clearAllDocuments = () => this.admin.clearAllDocuments();
  getUsers = () => this.admin.getUsers();
  createUser = (userData) => this.admin.createUser(userData);
  updateUser = (userId, userData) => this.admin.updateUser(userId, userData);
  deleteUser = (userId) => this.admin.deleteUser(userId);
  getPendingSubmissions = () => this.admin.getPendingSubmissions();
  reviewSubmission = (documentId, action, comment) => this.admin.reviewSubmission(documentId, action, comment);

  // ===== FOLDER METHODS =====
  getFolderTree = (collectionType) => this.folders.getFolderTree(collectionType);
  getFolderContents = (folderId) => this.folders.getFolderContents(folderId);
  createFolder = (folderData) => this.folders.createFolder(folderData);
  updateFolder = (folderId, folderData) => this.folders.updateFolder(folderId, folderData);
  deleteFolder = (folderId, recursive) => this.folders.deleteFolder(folderId, recursive);
  moveFolder = (folderId, newParentId) => this.folders.moveFolder(folderId, newParentId);
  exemptFolder = (folderId) => this.folders.exemptFolder(folderId);
  removeFolderExemption = (folderId) => this.folders.removeFolderExemption(folderId);
  overrideFolderExemption = (folderId) => this.folders.overrideFolderExemption(folderId);
  createDefaultFolders = () => this.folders.createDefaultFolders();

  // ===== PROJECT METHODS =====
  createProject = async (parentFolderId, projectName, projectType) => {
    const response = await this.post('/api/projects/create', {
      parent_folder_id: parentFolderId,
      project_name: projectName,
      project_type: projectType
    });
    return response;
  };

  // ===== TEMPLATE METHODS =====
  getUserTemplates = () => this.templates.getUserTemplates();
  getPublicTemplates = () => this.templates.getPublicTemplates();
  getTemplate = (templateId) => this.templates.getTemplate(templateId);
  createTemplate = (templateData) => this.templates.createTemplate(templateData);
  updateTemplate = (templateId, updates) => this.templates.updateTemplate(templateId, updates);
  deleteTemplate = (templateId) => this.templates.deleteTemplate(templateId);
  duplicateTemplate = (templateId, duplicateData) => this.templates.duplicateTemplate(templateId, duplicateData);
  searchTemplates = (keywords) => this.templates.searchTemplates(keywords);
  getTemplateStats = () => this.templates.getTemplateStats();
  getAvailableFieldTypes = () => this.templates.getAvailableFieldTypes();
  validateTemplate = (templateData) => this.templates.validateTemplate(templateData);
  executeTemplateResearch = (conversationId, templateId, query, customInstructions) => 
    this.templates.executeTemplateResearch(conversationId, templateId, query, customInstructions);
  confirmTemplateUsage = (conversationId, templateId, action, modifications) => 
    this.templates.confirmTemplateUsage(conversationId, templateId, action, modifications);
  getTemplateExecutionStatus = (conversationId) => this.templates.getTemplateExecutionStatus(conversationId);
  generateTemplatePlan = (templateId, query) => this.templates.generateTemplatePlan(templateId, query);

  // ===== INTEGRATION METHODS =====
}

const apiService = new ApiService();
export default apiService;
