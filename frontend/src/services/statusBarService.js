import apiService from './apiService';

/**
 * Service for fetching status bar data
 */
class StatusBarService {
  /**
   * Fetch status bar data from backend
   * @returns {Promise<Object>} Status bar data including time, weather, and version
   */
  async getStatusBarData() {
    try {
      const response = await apiService.get('/api/status-bar/data');
      // apiService.get() returns the JSON directly, not wrapped in data
      return response || {};
    } catch (error) {
      console.error('Failed to fetch status bar data:', error);
      // Return fallback data
      const now = new Date();
      return {
        current_time: now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        date_formatted: now.toLocaleDateString('en-US', { month: '2-digit', day: '2-digit', year: 'numeric' }),
        weather: null,
        app_version: 'dev'
      };
    }
  }
}

export default new StatusBarService();

