/**
 * Browser Notification Utility
 * Handles browser-level notifications for Signal Corps notifications
 */

class BrowserNotificationManager {
  constructor() {
    this.permissionRequested = false;
    this.notificationTags = new Set();
    this.maxTags = 200; // Prevent unbounded growth
  }

  /**
   * Request notification permission if not already granted
   */
  async requestPermission() {
    if (this.permissionRequested) {
      return Notification.permission;
    }

    try {
      if (typeof Notification === 'undefined') {
        return 'unsupported';
      }

      if (Notification.permission === 'default') {
        await Notification.requestPermission();
      }

      this.permissionRequested = true;
      return Notification.permission;
    } catch (error) {
      console.error('Error requesting notification permission:', error);
      return 'denied';
    }
  }

  /**
   * Show a browser notification
   * @param {Object} notificationData - Notification data from Signal Corps
   * @param {string} notificationData.message - Notification message
   * @param {string} notificationData.severity - Severity level (info, success, warning, error)
   * @param {string} notificationData.agent - Agent name that emitted the notification
   * @param {string} notificationData.timestamp - Timestamp
   * @param {boolean} notificationData.browser_notify - Whether to show browser notification (from metadata)
   */
  async showNotification(notificationData) {
    try {
      // Check if browser notifications are supported
      if (typeof Notification === 'undefined') {
        return false;
      }

      // Request permission if needed
      const permission = await this.requestPermission();
      if (permission !== 'granted') {
        return false;
      }

      // Determine if we should show browser notification
      // Show for: error, warning, or if explicitly requested via metadata
      const severity = notificationData.severity || 'info';
      const shouldNotify = 
        severity === 'error' || 
        severity === 'warning' || 
        notificationData.browser_notify === true;

      if (!shouldNotify) {
        return false;
      }

      // Create notification title based on severity and agent
      const agentName = notificationData.agent || 'System';
      const title = severity === 'error' 
        ? `⚠️ Alert from ${agentName}`
        : severity === 'warning'
        ? `⚠️ Warning from ${agentName}`
        : `ℹ️ ${agentName}`;

      // Create unique tag to prevent duplicates
      const tag = `${agentName}_${notificationData.timestamp || Date.now()}`;
      
      // Check if we've already shown this notification
      if (this.notificationTags.has(tag)) {
        return false;
      }

      // Create and show notification
      const notification = new Notification(title, {
        body: notificationData.message,
        tag: tag,
        icon: '/favicon.ico', // Use app icon if available
        badge: '/favicon.ico',
        requireInteraction: severity === 'error', // Keep error notifications visible until clicked
      });

      // Handle click - focus window
      notification.onclick = () => {
        try {
          window.focus();
          notification.close();
        } catch (error) {
          console.error('Error handling notification click:', error);
        }
      };

      // Track notification tag
      this.notificationTags.add(tag);
      
      // Clean up old tags to prevent memory growth
      if (this.notificationTags.size > this.maxTags) {
        const tagsArray = Array.from(this.notificationTags);
        this.notificationTags = new Set(tagsArray.slice(-100));
      }

      return true;
    } catch (error) {
      console.error('Error showing browser notification:', error);
      return false;
    }
  }

  /**
   * Clear all tracked notification tags
   */
  clearTags() {
    this.notificationTags.clear();
  }
}

// Export singleton instance
const browserNotificationManager = new BrowserNotificationManager();
export default browserNotificationManager;
