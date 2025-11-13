/**
 * ROOSEVELT'S ORG-MODE SERVICE
 * 
 * **BULLY!** Handles all org-mode specific API operations!
 * 
 * Domain service for:
 * - Searching org files
 * - Fetching agenda items
 * - Managing TODO lists
 * - Document lookup by filename
 */

import ApiServiceBase from '../base/ApiServiceBase';

class OrgService extends ApiServiceBase {
  /**
   * Search across all org files
   * 
   * @param {string} query - Search query text
   * @param {Object} options - Search options
   * @param {string[]} options.tags - Filter by tags
   * @param {string[]} options.todoStates - Filter by TODO states
   * @param {boolean} options.includeContent - Include content in search
   * @param {number} options.limit - Max results
   * @returns {Promise<Object>} Search results
   */
  async searchOrgFiles(query, options = {}) {
    const params = new URLSearchParams();
    params.append('query', query);
    
    if (options.tags && options.tags.length > 0) {
      params.append('tags', options.tags.join(','));
    }
    
    if (options.todoStates && options.todoStates.length > 0) {
      params.append('todo_states', options.todoStates.join(','));
    }
    
    if (options.includeContent !== undefined) {
      params.append('include_content', options.includeContent);
    }
    
    if (options.limit) {
      params.append('limit', options.limit);
    }
    
    return this.get(`/api/org/search?${params.toString()}`);
  }

  /**
   * Get all TODO items across org files
   * 
   * @param {Object} options - Filter options
   * @param {string[]} options.states - Filter by TODO states
   * @param {string[]} options.tags - Filter by tags
   * @param {number} options.limit - Max results
   * @returns {Promise<Object>} TODO items
   */
  async getAllTodos(options = {}) {
    const params = new URLSearchParams();
    
    if (options.states && options.states.length > 0) {
      params.append('states', options.states.join(','));
    }
    
    if (options.tags && options.tags.length > 0) {
      params.append('tags', options.tags.join(','));
    }
    
    if (options.limit) {
      params.append('limit', options.limit);
    }
    
    return this.get(`/api/org/todos?${params.toString()}`);
  }

  /**
   * Get agenda items (scheduled and deadline)
   * 
   * @param {Object} options - Agenda options
   * @param {number} options.daysAhead - Number of days to look ahead
   * @param {boolean} options.includeScheduled - Include SCHEDULED items
   * @param {boolean} options.includeDeadlines - Include DEADLINE items
   * @returns {Promise<Object>} Agenda items
   */
  async getAgenda(options = {}) {
    const params = new URLSearchParams();
    
    if (options.daysAhead) {
      params.append('days_ahead', options.daysAhead);
    }
    
    if (options.includeScheduled !== undefined) {
      params.append('include_scheduled', options.includeScheduled);
    }
    
    if (options.includeDeadlines !== undefined) {
      params.append('include_deadlines', options.includeDeadlines);
    }
    
    return this.get(`/api/org/agenda?${params.toString()}`);
  }

  /**
   * Look up a document by filename
   * 
   * **BULLY!** Find documents for navigation!
   * 
   * @param {string} filename - Filename to search for (e.g., "tasks.org")
   * @returns {Promise<Object>} Document metadata
   */
  async lookupDocument(filename) {
    const params = new URLSearchParams();
    params.append('filename', filename);
    
    return this.get(`/api/org/lookup-document?${params.toString()}`);
  }
}

// Export singleton instance
const orgService = new OrgService();
export default orgService;

