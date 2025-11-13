import ApiServiceBase from '../base/ApiServiceBase';

class IntegrationService extends ApiServiceBase {
  // Calibre integration methods
  getCalibreStatus = async () => {
    return this.get('/api/calibre/status');
  }

  toggleCalibreIntegration = async (enabled) => {
    return this.post('/api/calibre/settings/toggle', { enabled });
  }

  updateCalibreSettings = async (settings) => {
    return this.post('/api/calibre/settings/update', { settings });
  }

  searchCalibreLibrary = async (query, limit = 20) => {
    return this.post('/api/calibre/search', { query, limit });
  }

  getCalibreFilters = async () => {
    return this.get('/api/calibre/filters');
  }

  // Future integrations can be added here:
  // - RSS feeds
  // - External APIs
  // - Third-party services
}

export default new IntegrationService();
