/**
 * Calendar API service for Agenda view (O365, future CalDAV).
 * List connections, calendars, and events in a date range.
 */

import ApiServiceBase from '../base/ApiServiceBase';

class CalendarService extends ApiServiceBase {
  /**
   * List calendar-capable connections for the current user.
   * @returns {Promise<{ connections: Array<{ id: number, provider: string, display_name: string }>, error?: string }>}
   */
  async getConnections() {
    return this.get('/api/calendar/connections');
  }

  /**
   * List calendars for a connection (or first Microsoft).
   * @param {number} [connectionId] - Connection ID
   * @returns {Promise<{ calendars: Array<{ id, name, color, is_default, can_edit }>, error?: string }>}
   */
  async getCalendars(connectionId = null) {
    const params = new URLSearchParams();
    if (connectionId != null) params.append('connection_id', connectionId);
    return this.get(`/api/calendar/calendars?${params.toString()}`);
  }

  /**
   * Get calendar events in a date range.
   * @param {string} start - ISO 8601 start (e.g. 2026-02-20T00:00:00)
   * @param {string} end - ISO 8601 end (e.g. 2026-02-27T23:59:59)
   * @param {number} [connectionId] - Connection ID
   * @param {string} [calendarId] - Calendar ID (empty = default)
   * @param {number} [top=50] - Max events
   * @returns {Promise<{ events: Array, total_count: number, error?: string }>}
   */
  async getEvents(start, end, connectionId = null, calendarId = '', top = 50) {
    const params = new URLSearchParams();
    params.append('start', start);
    params.append('end', end);
    if (connectionId != null) params.append('connection_id', connectionId);
    if (calendarId) params.append('calendar_id', calendarId);
    params.append('top', String(top));
    return this.get(`/api/calendar/events?${params.toString()}`);
  }
}

const calendarService = new CalendarService();
export default calendarService;
