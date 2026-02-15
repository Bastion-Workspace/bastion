import ApiServiceBase from '../base/ApiServiceBase';

class FolderService extends ApiServiceBase {
  // ===== FOLDER MANAGEMENT METHODS =====

  getFolderTree = async (collectionType = 'user') => {
    return this.get(`/api/folders/tree?collection_type=${collectionType}`);
  }

  getFolderContents = async (folderId, limit = 250, offset = 0) => {
    const params = new URLSearchParams();
    if (limit != null) params.set('limit', String(limit));
    if (offset != null && offset > 0) params.set('offset', String(offset));
    const qs = params.toString();
    return this.get(`/api/folders/${folderId}/contents${qs ? `?${qs}` : ''}`);
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

  /**
   * Download the user's library (My Documents) as a zip file.
   * Fetches as blob and triggers browser download.
   */
  downloadLibrary = async () => {
    const token = localStorage.getItem('auth_token');
    const baseUrl = process.env.REACT_APP_API_URL || '';
    const resp = await fetch(`${baseUrl}/api/folders/library/download`, {
      method: 'GET',
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      credentials: 'include',
    });
    if (!resp.ok) {
      throw new Error(resp.statusText || 'Download failed');
    }
    const blob = await resp.blob();
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'my-library.zip';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  }
}

export default new FolderService();
