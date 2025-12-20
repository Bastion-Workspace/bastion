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

  // User preferred name methods
  getUserPreferredName = async () => {
    return this.get('/api/settings/user/preferred-name');
  }

  setUserPreferredName = async (preferredNameData) => {
    return this.put('/api/settings/user/preferred-name', preferredNameData);
  }

  // User AI context methods
  getUserAiContext = async () => {
    return this.get('/api/settings/user/ai-context');
  }

  setUserAiContext = async (aiContextData) => {
    return this.put('/api/settings/user/ai-context', aiContextData);
  }
}

export default new SettingsService();
