import ApiServiceBase from '../base/ApiServiceBase';

/** Must match `FolderContentsBatchRequest.folder_ids` max_length in backend/models/api_models.py */
const FOLDER_CONTENTS_BATCH_MAX_IDS = 100;

class FolderService extends ApiServiceBase {
  // ===== FOLDER MANAGEMENT METHODS =====

  getFolderTree = async (collectionType = 'user', shallow = true) => {
    const s = shallow ? 'true' : 'false';
    return this.get(`/api/folders/tree?collection_type=${collectionType}&shallow=${s}`);
  }

  getFolderContents = async (folderId, limit = 250, offset = 0) => {
    const params = new URLSearchParams();
    if (limit != null) params.set('limit', String(limit));
    if (offset != null && offset > 0) params.set('offset', String(offset));
    const qs = params.toString();
    return this.get(`/api/folders/${folderId}/contents${qs ? `?${qs}` : ''}`);
  }

  /**
   * Load multiple folders (one or more HTTP requests, chunked to API limit).
   * Returns { contents: { [folderId]: ... }, errors: { ... } } merged across chunks.
   */
  getFolderContentsBatch = async (folderIds, limit = 250, offset = 0, maxConcurrent = null) => {
    const raw = Array.isArray(folderIds) ? folderIds : [];
    const seen = new Set();
    const ids = [];
    for (const x of raw) {
      if (typeof x !== 'string') continue;
      const s = x.trim();
      if (!s || seen.has(s)) continue;
      seen.add(s);
      ids.push(s);
    }
    if (ids.length === 0) {
      return { contents: {}, errors: {} };
    }
    const merged = { contents: {}, errors: {} };
    for (let i = 0; i < ids.length; i += FOLDER_CONTENTS_BATCH_MAX_IDS) {
      const chunk = ids.slice(i, i + FOLDER_CONTENTS_BATCH_MAX_IDS);
      const body = {
        folder_ids: chunk,
        limit,
        offset,
      };
      if (maxConcurrent != null && maxConcurrent > 0) {
        body.max_concurrent = maxConcurrent;
      }
      const part = await this.post('/api/folders/contents/batch', body);
      if (part?.contents && typeof part.contents === 'object') {
        Object.assign(merged.contents, part.contents);
      }
      if (part?.errors && typeof part.errors === 'object') {
        Object.assign(merged.errors, part.errors);
      }
    }
    return merged;
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
    const baseUrl = import.meta.env.VITE_API_URL || '';
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
