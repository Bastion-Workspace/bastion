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
   * List org headline tags with usage counts (GET /api/org/tags).
   * @param {Object} options
   * @param {boolean} options.includeArchives - Include _archive.org files
   */
  async listOrgTags(options = {}) {
    const params = new URLSearchParams();
    if (options.includeArchives) {
      params.append('include_archives', 'true');
    }
    const q = params.toString();
    return this.get(q ? `/api/org/tags?${q}` : '/api/org/tags');
  }

  /**
   * Search across all org files
   * 
   * @param {string} query - Search query text
   * @param {Object} options - Search options
   * @param {string[]} options.tags - Filter by tags
   * @param {'any'|'all'} options.tagsMatch - OR vs AND when multiple tags
   * @param {boolean} options.includeArchives - Include _archive.org files
   * @param {string[]} options.todoStates - Filter by TODO states
   * @param {boolean} options.includeContent - Include content in search
   * @param {number} options.limit - Max results
   * @returns {Promise<Object>} Search results
   */
  async searchOrgFiles(query, options = {}) {
    const params = new URLSearchParams();
    params.append('query', query ?? '');
    
    if (options.tags && options.tags.length > 0) {
      params.append('tags', options.tags.join(','));
    }

    if (options.tagsMatch === 'all' || options.tagsMatch === 'any') {
      params.append('tags_match', options.tagsMatch);
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

    if (options.includeArchives) {
      params.append('include_archives', 'true');
    }
    
    return this.get(`/api/org/search?${params.toString()}`);
  }

  /**
   * Get all TODO items across org files (legacy endpoint)
   * @param {Object} options - Filter options
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
   * List todos via universal todo API (0-based line_number). Use for listing and for inline toggle/update.
   * @param {Object} options - scope (all|inbox|file path), states, tags, query, limit
   * @returns {Promise<Object>} { success, results, count, files_searched }
   */
  async listTodos(options = {}) {
    const params = new URLSearchParams();
    params.append('scope', options.scope || 'all');
    if (options.states && options.states.length > 0) {
      params.append('states', options.states.join(','));
    }
    if (options.tags && options.tags.length > 0) {
      params.append('tags', options.tags.join(','));
    }
    if (options.query) {
      params.append('query', options.query);
    }
    if (options.limit) {
      params.append('limit', options.limit);
    }
    if (options.includeArchives) {
      params.append('include_archives', 'true');
    }
    return this.get(`/api/todos?${params.toString()}`);
  }

  /**
   * Toggle a todo (TODO <-> DONE). Uses 0-based line_number.
   * @param {string} filePath - Full or relative org file path
   * @param {number} lineNumber - 0-based line index
   * @param {string} [headingText] - Optional heading text for verification
   * @returns {Promise<Object>} { success, file_path, line_number, new_line, error }
   */
  async toggleTodo(filePath, lineNumber, headingText = null) {
    return this.post('/api/todos/toggle', {
      file_path: filePath,
      line_number: lineNumber,
      ...(headingText && { heading_text: headingText }),
    });
  }

  /**
   * Update a todo (state, text, tags). Uses 0-based line_number.
   * @param {string} filePath - Full or relative org file path
   * @param {number} lineNumber - 0-based line index
   * @param {Object} updates - { new_state, new_text, add_tags, remove_tags, scheduled, deadline, priority }
   * @param {string} [headingText] - Optional heading text for verification
   * @returns {Promise<Object>} { success, file_path, line_number, new_line, error }
   */
  async updateTodo(filePath, lineNumber, updates, headingText = null) {
    return this.post('/api/todos/update', {
      file_path: filePath,
      line_number: lineNumber,
      ...(headingText && { heading_text: headingText }),
      ...updates,
    });
  }

  /**
   * Get agenda items (scheduled and deadline)
   *
   * @param {Object} options - Agenda options
   * @param {number} options.daysAhead - Number of days to look ahead
   * @param {boolean} options.includeScheduled - Include SCHEDULED items
   * @param {boolean} options.includeDeadlines - Include DEADLINE items
   * @param {string[]} options.includeOrgFiles - Org filenames to include (e.g. ['calendar.org']). Omit = all; [] = none.
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

    if (options.includeAppointments !== undefined) {
      params.append('include_appointments', options.includeAppointments);
    }

    if (options.includeOrgFiles !== undefined) {
      options.includeOrgFiles.forEach((f) => params.append('include_org_files', f));
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

