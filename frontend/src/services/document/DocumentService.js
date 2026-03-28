import ApiServiceBase from '../base/ApiServiceBase';

class DocumentService extends ApiServiceBase {
  // Document methods
  getDocuments = async () => {
    return this.get('/api/documents');
  }

  getUserDocuments = async (offset = 0, limit = 100) => {
    return this.get(`/api/user/documents?offset=${offset}&limit=${limit}`);
  }

  /** Lightweight check for org files (no document list). Use for sidebar org tools visibility. */
  getHasOrgDocuments = async () => {
    return this.get('/api/user/documents/has-org');
  }

  uploadDocument = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    return this.request('/api/documents/upload', {
      method: 'POST',
      body: formData,
      headers: {} // Let browser set Content-Type for FormData
    });
  }

  uploadUserDocument = async (file, userId) => {
    const formData = new FormData();
    formData.append('file', file);
    if (userId) formData.append('user_id', userId);
    
    return this.request('/api/user/documents/upload', {
      method: 'POST',
      body: formData,
      headers: {} // Let browser set Content-Type for FormData
    });
  }

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
  }

  importFromUrl = async (url) => {
    return this.post('/api/documents/import-url', { url });
  }

  importImage = async (imageUrl, filename = null, folderId = null) => {
    return this.post('/api/documents/import-image', {
      image_url: imageUrl,
      filename: filename,
      folder_id: folderId
    });
  }

  deleteDocument = async (documentId) => {
    return this.delete(`/api/documents/${documentId}`);
  }

  reprocessDocument = async (documentId) => {
    return this.post(`/api/documents/${documentId}/reprocess`);
  }

  reprocessUserDocument = async (documentId) => {
    return this.post(`/api/user/documents/${documentId}/reprocess`);
  }

  exemptDocument = async (documentId) => {
    return this.post(`/api/documents/${documentId}/exempt`);
  }

  removeDocumentExemption = async (documentId, inherit = false) => {
    const queryParam = inherit ? '?inherit=true' : '';
    return this.delete(`/api/documents/${documentId}/exempt${queryParam}`);
  }

  updateDocument = async (documentId, updates) => {
    return this.put(`/api/documents/${documentId}`, updates);
  }

  updateDocumentMetadata = async (documentId, metadata) => {
    return this.put(`/api/documents/${documentId}/metadata`, metadata);
  }

  getPendingProposals = async (documentId, options = {}) => {
    return this.get(`/api/documents/${documentId}/pending-proposals`, options);
  }

  applyDocumentEditProposal = async (proposalId, selectedOperationIndices = null) => {
    return this.post('/api/documents/edit-proposals/apply', {
      proposal_id: proposalId,
      selected_operation_indices: selectedOperationIndices
    });
  }

  rejectDocumentEditProposal = async (proposalId) => {
    return this.post('/api/documents/edit-proposals/reject', { proposal_id: proposalId });
  }

  markDocumentEditProposalApplied = async (proposalId) => {
    return this.post('/api/documents/edit-proposals/mark-applied', { proposal_id: proposalId });
  }

  renameDocument = async (documentId, newFilename) => {
    // Use FileManager API which also normalizes extension and renames disk file
    return this.post('/api/file-manager/rename-file', {
      document_id: documentId,
      new_filename: newFilename
    });
  }

  moveDocument = async (documentId, newFolderId, userId = null) => {
    // Use FileManager API to move files between folders with websocket updates
    return this.post('/api/file-manager/move-file', {
      document_id: documentId,
      new_folder_id: newFolderId,
      user_id: userId || undefined
    });
  }

  /**
   * Search documents (hybrid vector + full-text, semantic-only, or fulltext-only).
   * Results include highlighted_snippet when available from full-text search.
   */
  searchDocuments = async (query, { searchMode = 'hybrid', limit = 20, folderId = null, fileTypes = null } = {}) => {
    const body = {
      query: String(query || '').trim(),
      search_mode: searchMode,
      limit: limit
    };
    if (folderId) body.folder_id = folderId;
    if (fileTypes && fileTypes.length) body.file_types = fileTypes;
    return this.post('/api/user/documents/search', body);
  }

  // Document content retrieval
  getDocumentContent = async (documentId) => {
    try {
      const response = await this.request(`/api/documents/${documentId}/content`);
      return response;
    } catch (error) {
      console.error('Failed to get document content:', error);
      throw error;
    }
  }

  // Document content update
  updateDocumentContent = async (documentId, content) => {
    try {
      return await this.put(`/api/documents/${documentId}/content`, { content });
    } catch (error) {
      console.error('Failed to update document content:', error);
      throw error;
    }
  }

  // Document version history
  getDocumentVersions = async (documentId, skip = 0, limit = 100) => {
    return this.get(`/api/documents/${documentId}/versions?skip=${skip}&limit=${limit}`);
  }

  getVersionContent = async (documentId, versionId) => {
    return this.get(`/api/documents/${documentId}/versions/${versionId}/content`);
  }

  diffVersions = async (documentId, fromVersionId, toVersionId) => {
    return this.get(`/api/documents/${documentId}/versions/diff?from=${encodeURIComponent(fromVersionId)}&to=${encodeURIComponent(toVersionId)}`);
  }

  rollbackToVersion = async (documentId, versionId) => {
    return this.post(`/api/documents/${documentId}/versions/${versionId}/rollback`);
  }

  // Document creation methods
  createDocumentFromContent = async ({ content, title, filename, userId, folderId, docType = 'org' }) => {
    // Use FileManager API to place a text document into a specific folder
    const payload = {
      content,
      title,
      filename: filename || `${title}.${docType === 'md' ? 'md' : docType === 'org' ? 'org' : 'txt'}`,
      source_type: 'manual',
      doc_type: docType,
      user_id: userId,
      collection_type: 'user',
      target_folder_id: folderId,
      process_immediately: true,
    };
    return this.post('/api/file-manager/place-file', payload);
  }
}

export default new DocumentService();
