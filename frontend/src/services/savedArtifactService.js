/**
 * Saved chat artifacts API (library, dashboard embeds, share, export).
 */
import apiService from './apiService';
import authService from './auth/AuthService';

const API_BASE = import.meta.env.VITE_API_URL || '';

const savedArtifactService = {
  list: () => apiService.get('/api/saved-artifacts'),

  create: (body) => apiService.post('/api/saved-artifacts', body),

  get: (id) => apiService.get(`/api/saved-artifacts/${encodeURIComponent(id)}`),

  patch: (id, body) => apiService.patch(`/api/saved-artifacts/${encodeURIComponent(id)}`, body),

  delete: (id) => apiService.delete(`/api/saved-artifacts/${encodeURIComponent(id)}`),

  share: (id) => apiService.post(`/api/saved-artifacts/${encodeURIComponent(id)}/share`, {}),

  unshare: (id) => apiService.delete(`/api/saved-artifacts/${encodeURIComponent(id)}/share`),

  /**
   * Public artifact JSON (no auth). Used by share page and embeds.
   */
  fetchPublic: async (shareToken) => {
    const url = `${API_BASE}/api/public/artifacts/${encodeURIComponent(shareToken)}?format=json`;
    const res = await fetch(url, { method: 'GET', headers: { Accept: 'application/json' } });
    if (!res.ok) {
      const err = new Error(`Failed to load shared artifact (${res.status})`);
      err.status = res.status;
      throw err;
    }
    return res.json();
  },

  /**
   * Download standalone HTML export (auth required).
   */
  downloadExport: async (artifactId) => {
    const token = authService.getToken();
    const url = `${API_BASE}/api/saved-artifacts/${encodeURIComponent(artifactId)}/export`;
    const res = await fetch(url, {
      method: 'GET',
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!res.ok) {
      throw new Error(`Export failed (${res.status})`);
    }
    const blob = await res.blob();
    const cd = res.headers.get('Content-Disposition');
    let filename = 'artifact.html';
    if (cd) {
      const m = /filename="([^"]+)"/.exec(cd);
      if (m) filename = m[1];
    }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  },
};

export default savedArtifactService;
