/**
 * Centralized Document Diff Store
 * 
 * Manages persistent storage of editor diffs across tab switches.
 * Provides document-aware diff tracking, smart invalidation, and
 * notification system for UI updates.
 */

class DocumentDiffStore {
  constructor() {
    this.diffs = {}; // { [documentId]: { operations: [], messageId, timestamp, contentHash } }
    this.listeners = new Set();
    this.storageKey = 'document_diffs_store';
    this.loadFromStorage();
  }

  /**
   * Set diffs for a document
   * @param {string} documentId - Document identifier
   * @param {Array} operations - Array of diff operations
   * @param {string} messageId - Message ID that generated these diffs
   * @param {string} contentSnapshot - Current document content (for validation)
   */
  setDiffs(documentId, operations, messageId, contentSnapshot) {
    if (!documentId) {
      console.warn('DocumentDiffStore: Cannot set diffs without documentId');
      return;
    }

    const contentHash = this._hashContent(contentSnapshot || '');
    
    this.diffs[documentId] = {
      operations: Array.isArray(operations) ? operations : [],
      messageId: messageId || null,
      timestamp: Date.now(),
      contentHash: contentHash
    };

    this.saveToStorage();
    this.notify(documentId, 'set');
  }

  /**
   * Get diffs for a document
   * @param {string} documentId - Document identifier
   * @returns {Object|null} Diff data or null if not found
   */
  getDiffs(documentId) {
    if (!documentId) return null;
    return this.diffs[documentId] || null;
  }

  /**
   * Clear all diffs for a document
   * @param {string} documentId - Document identifier
   */
  clearDiffs(documentId) {
    if (!documentId) return;
    
    if (this.diffs[documentId]) {
      delete this.diffs[documentId];
      this.saveToStorage();
      this.notify(documentId, 'clear');
    }
  }

  /**
   * Remove a specific diff operation
   * @param {string} documentId - Document identifier
   * @param {string} operationId - Operation ID to remove
   */
  removeDiff(documentId, operationId) {
    if (!documentId || !operationId) return;

    const docDiffs = this.diffs[documentId];
    if (!docDiffs || !Array.isArray(docDiffs.operations)) return;

    const initialLength = docDiffs.operations.length;
    docDiffs.operations = docDiffs.operations.filter(op => {
      // Operation ID can be in various formats
      const opId = op.operationId || op.id || op.start + '-' + op.end;
      return opId !== operationId && String(opId) !== String(operationId);
    });

    if (docDiffs.operations.length !== initialLength) {
      this.saveToStorage();
      this.notify(documentId, 'remove');
    }
  }

  /**
   * Validate diffs against current document content
   * Removes diffs that are invalid due to content changes
   * @param {string} documentId - Document identifier
   * @param {string} currentContent - Current document content
   * @returns {Array} Array of invalidated operation IDs
   */
  validateDiffs(documentId, currentContent) {
    if (!documentId) return [];

    const docDiffs = this.diffs[documentId];
    if (!docDiffs) return [];

    const currentHash = this._hashContent(currentContent || '');
    const invalidated = [];

    // If content hash changed significantly, all diffs may be stale
    // For now, we keep them and let the plugin handle position validation
    // This method can be extended for more sophisticated validation

    return invalidated;
  }

  /**
   * Invalidate diffs that overlap with a manual edit range
   * @param {string} documentId - Document identifier
   * @param {number} editStart - Start position of manual edit
   * @param {number} editEnd - End position of manual edit
   * @returns {Array} Array of invalidated operation IDs
   */
  invalidateOverlappingDiffs(documentId, editStart, editEnd) {
    if (!documentId) return [];

    const docDiffs = this.diffs[documentId];
    if (!docDiffs || !Array.isArray(docDiffs.operations)) return [];

    const invalidated = [];
    const remaining = [];

    docDiffs.operations.forEach(op => {
      const opStart = op.from !== undefined ? op.from : (op.start !== undefined ? op.start : 0);
      const opEnd = op.to !== undefined ? op.to : (op.end !== undefined ? op.end : opStart);

      // Check for overlap
      const overlaps = !(editEnd < opStart || editStart > opEnd);

      if (overlaps) {
        const opId = op.operationId || op.id || `${opStart}-${opEnd}`;
        invalidated.push(opId);
      } else {
        remaining.push(op);
      }
    });

    if (invalidated.length > 0) {
      docDiffs.operations = remaining;
      this.saveToStorage();
      this.notify(documentId, 'invalidate');
    }

    return invalidated;
  }

  /**
   * Save diffs to localStorage
   */
  saveToStorage() {
    try {
      const serialized = JSON.stringify(this.diffs);
      localStorage.setItem(this.storageKey, serialized);
    } catch (error) {
      console.error('DocumentDiffStore: Failed to save to storage:', error);
    }
  }

  /**
   * Load diffs from localStorage
   */
  loadFromStorage() {
    try {
      const stored = localStorage.getItem(this.storageKey);
      if (stored) {
        this.diffs = JSON.parse(stored);
        // Clean up old diffs (older than 24 hours)
        const now = Date.now();
        const maxAge = 24 * 60 * 60 * 1000; // 24 hours
        
        Object.keys(this.diffs).forEach(docId => {
          const diff = this.diffs[docId];
          if (diff.timestamp && (now - diff.timestamp) > maxAge) {
            delete this.diffs[docId];
          }
        });
        
        if (Object.keys(this.diffs).length !== Object.keys(JSON.parse(stored)).length) {
          this.saveToStorage();
        }
      }
    } catch (error) {
      console.error('DocumentDiffStore: Failed to load from storage:', error);
      this.diffs = {};
    }
  }

  /**
   * Subscribe to diff changes
   * @param {Function} callback - Callback function (documentId, changeType) => void
   */
  subscribe(callback) {
    if (typeof callback === 'function') {
      this.listeners.add(callback);
    }
  }

  /**
   * Unsubscribe from diff changes
   * @param {Function} callback - Callback function to remove
   */
  unsubscribe(callback) {
    this.listeners.delete(callback);
  }

  /**
   * Notify all listeners of a change
   * @param {string} documentId - Document identifier
   * @param {string} changeType - Type of change: 'set', 'clear', 'remove', 'invalidate'
   */
  notify(documentId, changeType) {
    this.listeners.forEach(callback => {
      try {
        callback(documentId, changeType);
      } catch (error) {
        console.error('DocumentDiffStore: Listener error:', error);
      }
    });
  }

  /**
   * Hash content for validation (simple hash)
   * @param {string} content - Content to hash
   * @returns {string} Hash string
   * @private
   */
  _hashContent(content) {
    if (!content) return '';
    // Simple hash: first 100 chars + length + last 100 chars
    const start = content.substring(0, 100);
    const end = content.substring(Math.max(0, content.length - 100));
    return `${start.length}_${content.length}_${end.length}`;
  }

  /**
   * Get all document IDs with pending diffs
   * @returns {Array} Array of document IDs
   */
  getDocumentsWithDiffs() {
    return Object.keys(this.diffs).filter(docId => {
      const diff = this.diffs[docId];
      return diff && Array.isArray(diff.operations) && diff.operations.length > 0;
    });
  }

  /**
   * Get diff count for a document
   * @param {string} documentId - Document identifier
   * @returns {number} Number of pending diffs
   */
  getDiffCount(documentId) {
    const docDiffs = this.getDiffs(documentId);
    return docDiffs && Array.isArray(docDiffs.operations) ? docDiffs.operations.length : 0;
  }
}

// Export singleton instance
export const documentDiffStore = new DocumentDiffStore();

