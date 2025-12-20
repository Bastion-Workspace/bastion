import ApiServiceBase from './base/ApiServiceBase';

class EntertainmentSyncService extends ApiServiceBase {
  // Configuration management
  async getConfigs() {
    return this.get('/api/entertainment/sync/config');
  }

  async getConfig(configId) {
    return this.get(`/api/entertainment/sync/config/${configId}`);
  }

  async createConfig(config) {
    return this.post('/api/entertainment/sync/config', config);
  }

  async updateConfig(configId, updates) {
    return this.put(`/api/entertainment/sync/config/${configId}`, updates);
  }

  async deleteConfig(configId) {
    return this.delete(`/api/entertainment/sync/config/${configId}`);
  }

  // Connection testing
  async testConnection(configId) {
    return this.post(`/api/entertainment/sync/config/${configId}/test`);
  }

  // Sync operations
  async triggerSync(configId) {
    return this.post(`/api/entertainment/sync/config/${configId}/trigger`);
  }

  async getSyncStatus(configId) {
    return this.get(`/api/entertainment/sync/config/${configId}/status`);
  }

  // Synced items
  async getSyncedItems(configId, filters = {}) {
    const params = new URLSearchParams();
    if (filters.external_type) {
      params.append('external_type', filters.external_type);
    }
    if (filters.limit) {
      params.append('limit', filters.limit);
    }
    if (filters.skip) {
      params.append('skip', filters.skip);
    }
    
    const queryString = params.toString();
    const url = `/api/entertainment/sync/config/${configId}/items${queryString ? '?' + queryString : ''}`;
    return this.get(url);
  }
}

export default new EntertainmentSyncService();







