/**
 * Contacts API service for Contacts view (O365 + org-mode).
 * List connections, O365 contacts (CRUD), and org contacts.
 */

import ApiServiceBase from '../base/ApiServiceBase';

class ContactService extends ApiServiceBase {
  /**
   * List contacts-capable connections (Microsoft O365).
   * @returns {Promise<{ connections: Array<{ id: number, provider: string, display_name: string }>, error?: string }>}
   */
  async getConnections() {
    return this.get('/api/contacts/connections');
  }

  /**
   * Get O365 contacts for a connection.
   * @param {number} [connectionId] - Connection ID (default: first Microsoft)
   * @param {string} [folderId] - Contact folder ID (empty = default)
   * @param {number} [top=100] - Max contacts
   * @returns {Promise<{ contacts: Array, total_count: number, error?: string }>}
   */
  async getO365Contacts(connectionId = null, folderId = '', top = 100) {
    const params = new URLSearchParams();
    if (connectionId != null) params.append('connection_id', connectionId);
    if (folderId) params.append('folder_id', folderId);
    params.append('top', String(top));
    return this.get(`/api/contacts/o365?${params.toString()}`);
  }

  /**
   * Create an O365 contact.
   * @param {Object} data - Contact fields (display_name, given_name, surname, email_addresses, phone_numbers, etc.)
   * @param {number} [connectionId] - Connection ID
   * @returns {Promise<{ success: boolean, contact_id?: string, error?: string }>}
   */
  async createContact(data, connectionId = null) {
    const params = connectionId != null ? `?connection_id=${connectionId}` : '';
    return this.post(`/api/contacts/o365${params}`, data);
  }

  /**
   * Update an O365 contact.
   * @param {string} contactId - Contact ID
   * @param {Object} data - Fields to update (partial)
   * @param {number} [connectionId] - Connection ID
   * @returns {Promise<{ success: boolean, error?: string }>}
   */
  async updateContact(contactId, data, connectionId = null) {
    const params = connectionId != null ? `?connection_id=${connectionId}` : '';
    return this.patch(`/api/contacts/o365/${encodeURIComponent(contactId)}${params}`, data);
  }

  /**
   * Delete an O365 contact.
   * @param {string} contactId - Contact ID
   * @param {number} [connectionId] - Connection ID
   * @returns {Promise<{ success: boolean, error?: string }>}
   */
  async deleteContact(contactId, connectionId = null) {
    const params = connectionId != null ? `?connection_id=${connectionId}` : '';
    return this.delete(`/api/contacts/o365/${encodeURIComponent(contactId)}${params}`);
  }

  /**
   * Get org-mode contacts (from org files).
   * @param {string} [category] - Filter by category (tag or parent heading)
   * @param {number} [limit=500] - Max results
   * @returns {Promise<{ success: boolean, results: Array, count: number, error?: string }>}
   */
  async getOrgContacts(category = null, limit = 500) {
    const params = new URLSearchParams();
    if (category) params.append('category', category);
    params.append('limit', String(limit));
    return this.get(`/api/contacts/org?${params.toString()}`);
  }
}

const contactService = new ContactService();
export default contactService;
