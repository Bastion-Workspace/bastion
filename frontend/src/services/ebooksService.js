/**
 * Ebooks / OPDS / KoSync API client.
 */
import apiService from './apiService';

const ebooksService = {
  getSettings: () => apiService.get('/api/ebooks/settings'),
  putSettings: (body) => apiService.put('/api/ebooks/settings', body),
  putKosyncSettings: (body) => apiService.put('/api/ebooks/kosync/settings', body),
  fetchOpds: (body) =>
    body?.want === 'binary'
      ? apiService.post('/api/ebooks/opds/fetch', body, { responseType: 'arraybuffer' })
      : apiService.post('/api/ebooks/opds/fetch', body),
  kosyncTest: (body) => apiService.post('/api/ebooks/kosync/test', body),
  kosyncHealth: () => apiService.get('/api/ebooks/kosync/health'),
  kosyncRegister: (body) => apiService.post('/api/ebooks/kosync/register', body),
  getProgress: (document) =>
    apiService.get(`/api/ebooks/kosync/progress/${encodeURIComponent(document)}`),
  putProgress: (body) => apiService.put('/api/ebooks/kosync/progress', body),
};

export default ebooksService;
