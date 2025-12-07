import ApiServiceBase from '../base/ApiServiceBase';

class FolderService extends ApiServiceBase {
  // ===== FOLDER MANAGEMENT METHODS =====

  getFolderTree = async (collectionType = 'user') => {
    return this.get(`/api/folders/tree?collection_type=${collectionType}`);
  }

  getFolderContents = async (folderId) => {
    return this.get(`/api/folders/${folderId}/contents`);
  }

  createFolder = async (folderData) => {
    return this.post('/api/folders', folderData);
  }

  updateFolder = async (folderId, folderData) => {
    return this.put(`/api/folders/${folderId}`, folderData);
  }

  deleteFolder = async (folderId, recursive = false) => {
    return this.delete(`/api/folders/${folderId}?recursive=${recursive}`);
  }

  moveFolder = async (folderId, newParentId = null) => {
    const qp = newParentId ? `?new_parent_id=${encodeURIComponent(newParentId)}` : '';
    return this.post(`/api/folders/${folderId}/move${qp}`, {});
  }

  createDefaultFolders = async () => {
    return this.post('/api/folders/default');
  }

  exemptFolder = async (folderId) => {
    return this.post(`/api/folders/${folderId}/exempt`);
  }

  removeFolderExemption = async (folderId) => {
    return this.delete(`/api/folders/${folderId}/exempt`);
  }

  overrideFolderExemption = async (folderId) => {
    return this.post(`/api/folders/${folderId}/exempt/override`);
  }
}

export default new FolderService();
