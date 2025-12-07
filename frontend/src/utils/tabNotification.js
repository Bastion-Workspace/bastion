/**
 * Tab Notification Utility
 * Flashes the browser tab title when new messages arrive and the tab is not visible
 */

class TabNotificationManager {
  constructor() {
    this.originalTitle = document.title;
    this.flashInterval = null;
    this.isFlashing = false;
    this.flashCount = 0;
    this.flashMessage = 'New message';
    this.maxFlashCount = 20; // Stop flashing after 20 cycles
    this.flashDelay = 1000; // Flash every 1 second
    
    // Listen for visibility changes
    document.addEventListener('visibilitychange', this.handleVisibilityChange.bind(this));
  }

  handleVisibilityChange() {
    if (document.visibilityState === 'visible') {
      // Tab is now visible - stop flashing and restore title
      this.stopFlashing();
    }
  }

  startFlashing(message = 'New message') {
    // Only flash if tab is hidden
    if (document.visibilityState === 'visible') {
      return;
    }

    // If already flashing, update the message but don't restart
    if (this.isFlashing) {
      // Update the flash message for the next cycle
      this.flashMessage = message;
      return;
    }

    this.isFlashing = true;
    this.flashCount = 0;
    this.flashMessage = message;
    this.originalTitle = document.title;

    // Start flashing
    this.flashInterval = setInterval(() => {
      if (document.visibilityState === 'visible') {
        // Tab became visible - stop flashing
        this.stopFlashing();
        return;
      }

      // Toggle between original title and notification
      if (document.title === this.originalTitle) {
        document.title = `🔔 ${this.flashMessage}`;
      } else {
        document.title = this.originalTitle;
      }

      this.flashCount++;
      
      // Stop after max cycles
      if (this.flashCount >= this.maxFlashCount) {
        this.stopFlashing();
      }
    }, this.flashDelay);
  }

  stopFlashing() {
    if (this.flashInterval) {
      clearInterval(this.flashInterval);
      this.flashInterval = null;
    }
    
    // Restore original title
    if (document.title !== this.originalTitle) {
      document.title = this.originalTitle;
    }
    
    this.isFlashing = false;
    this.flashCount = 0;
  }

  destroy() {
    this.stopFlashing();
    document.removeEventListener('visibilitychange', this.handleVisibilityChange.bind(this));
  }
}

// Create singleton instance
const tabNotificationManager = new TabNotificationManager();

export default tabNotificationManager;

