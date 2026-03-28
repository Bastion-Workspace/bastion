import ApiServiceBase from '../base/ApiServiceBase';

class SettingsService extends ApiServiceBase {
  // Settings methods
  getSettings = async () => {
    return this.get('/api/settings');
  }

  getSettingsByCategory = async (category) => {
    return this.get(`/api/settings/${category}`);
  }

  setSettingValue = async (key, { value, description = '', category = 'general' }) => {
    return this.put(`/api/settings/${encodeURIComponent(key)}`, {
      key,
      value,
      description,
      category
    });
  }

  updateSetting = async (key, value) => {
    return this.post('/api/settings', { key, value });
  }

  // Model management methods
  getAvailableModels = async () => {
    return this.get('/api/models/available');
  }

  getEnabledModels = async () => {
    return this.get('/api/models/enabled');
  }

  setEnabledModels = async (modelIds) => {
    return this.post('/api/models/enabled', { model_ids: modelIds });
  }

  getCurrentModel = async () => {
    return this.get('/api/models/current');
  }

  selectModel = async (modelName) => {
    return this.post('/api/models/select', { model_name: modelName });
  }

  invalidateCatalogSlice = async () => {
    return this.post('/api/models/catalog-slice/invalidate', {});
  }

  // User timezone methods
  getUserTimezone = async () => {
    return this.get('/api/settings/user/timezone');
  }

  setUserTimezone = async (timezoneData) => {
    return this.put('/api/settings/user/timezone', timezoneData);
  }

  // User zip code methods
  getUserZipCode = async () => {
    return this.get('/api/settings/user/zip-code');
  }

  setUserZipCode = async (zipCodeData) => {
    return this.put('/api/settings/user/zip-code', zipCodeData);
  }

  // User time format methods
  getUserTimeFormat = async () => {
    return this.get('/api/settings/user/time-format');
  }

  setUserTimeFormat = async (timeFormatData) => {
    return this.put('/api/settings/user/time-format', timeFormatData);
  }

  // Prompt settings methods
  getPromptSettings = async () => {
    return this.get('/api/settings/prompt');
  }

  updatePromptSettings = async (settings) => {
    return this.post('/api/settings/prompt', settings);
  }

  getPromptOptions = async () => {
    return this.get('/api/settings/prompt/options');
  }

  // Personas (built-in + custom; default persona)
  getPersonas = async (params = {}) => {
    const q = new URLSearchParams(params).toString();
    return this.get(`/api/personas${q ? `?${q}` : ''}`);
  }

  getPersona = async (personaId) => {
    return this.get(`/api/personas/${encodeURIComponent(personaId)}`);
  }

  createPersona = async (data) => {
    return this.post('/api/personas', data);
  }

  updatePersona = async (personaId, data) => {
    return this.put(`/api/personas/${encodeURIComponent(personaId)}`, data);
  }

  deletePersona = async (personaId) => {
    return this.delete(`/api/personas/${encodeURIComponent(personaId)}`);
  }

  getDefaultPersona = async () => {
    return this.get('/api/settings/default-persona');
  }

  setDefaultPersona = async (personaId) => {
    return this.post('/api/settings/default-persona', { persona_id: personaId ?? null });
  }

  getDefaultChatAgentProfile = async () => {
    return this.get('/api/settings/default-chat-agent-profile');
  }

  setDefaultChatAgentProfile = async (agentProfileId) => {
    return this.post('/api/settings/default-chat-agent-profile', {
      agent_profile_id: agentProfileId ?? null,
    });
  }

  // User preferred name methods
  getUserPreferredName = async () => {
    return this.get('/api/settings/user/preferred-name');
  }

  setUserPreferredName = async (preferredNameData) => {
    return this.put('/api/settings/user/preferred-name', preferredNameData);
  }

  // User phone number methods
  getUserPhoneNumber = async () => {
    return this.get('/api/settings/user/phone-number');
  }

  setUserPhoneNumber = async (phoneNumberData) => {
    return this.put('/api/settings/user/phone-number', phoneNumberData);
  }

  // User birthday methods
  getUserBirthday = async () => {
    return this.get('/api/settings/user/birthday');
  }

  setUserBirthday = async (birthdayData) => {
    return this.put('/api/settings/user/birthday', birthdayData);
  }

  // User AI context methods
  getUserAiContext = async () => {
    return this.get('/api/settings/user/ai-context');
  }

  setUserAiContext = async (aiContextData) => {
    return this.put('/api/settings/user/ai-context', aiContextData);
  }

  getUserFacts = async () => {
    return this.get('/api/settings/user/facts');
  }

  addUserFact = async (data) => {
    return this.post('/api/settings/user/facts', data);
  }

  deleteUserFact = async (factKey) => {
    return this.delete(`/api/settings/user/facts/${encodeURIComponent(factKey)}`);
  }

  getFactsPreferences = async () => {
    return this.get('/api/settings/user/facts-preferences');
  }

  setFactsPreferences = async (data) => {
    return this.post('/api/settings/user/facts-preferences', data);
  }

  getEpisodes = async (params = {}) => {
    const q = new URLSearchParams(params).toString();
    return this.get(`/api/settings/user/episodes${q ? `?${q}` : ''}`);
  }

  deleteEpisode = async (episodeId) => {
    return this.delete(`/api/settings/user/episodes/${episodeId}`);
  }

  deleteAllEpisodes = async () => {
    return this.delete('/api/settings/user/episodes');
  }

  getEpisodesPreferences = async () => {
    return this.get('/api/settings/user/episodes-preferences');
  }

  setEpisodesPreferences = async (data) => {
    return this.post('/api/settings/user/episodes-preferences', data);
  }

  getPendingFacts = async () => {
    return this.get('/api/settings/user/facts/pending');
  }

  resolvePendingFact = async (historyId, action) => {
    return this.post(`/api/settings/user/facts/pending/${historyId}/resolve`, { action });
  }

  getFactHistory = async (params = {}) => {
    const q = new URLSearchParams(params).toString();
    return this.get(`/api/settings/user/facts/history${q ? `?${q}` : ''}`);
  }

  // Vision features methods
  getVisionFeaturesEnabled = async () => {
    return this.get('/api/settings/user/vision-features');
  }

  setVisionFeaturesEnabled = async (enabled) => {
    return this.put('/api/settings/user/vision-features', { enabled });
  }

  getVisionServiceStatus = async () => {
    return this.get('/api/vision/service-status');
  }
}

export default new SettingsService();
