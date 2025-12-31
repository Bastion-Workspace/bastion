/**
 * Centralized Document Diff Store
 * 
 * Manages persistent storage of editor diffs across tab switches.
 * Provides document-aware diff tracking, smart invalidation, and
 * notification system for UI updates.
 */

class DocumentDiffStore {
  constructor() {
    // Get current user from localStorage (set by auth system)
    const currentUser = localStorage.getItem('user_id') || 'anonymous';
    this.diffs = {}; // { [documentId]: { operations: [], messageId, timestamp, contentHash } }
    this.listeners = new Set();
    this.storageKey = `document_diffs_store_${currentUser}`; // ✅ User-specific storage
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
    console.log('🔍 removeDiff called:', { documentId, operationId });
    
    if (!documentId || !operationId) {
      console.warn('⚠️ removeDiff: Missing documentId or operationId');
      return;
    }

    const docDiffs = this.diffs[documentId];
    if (!docDiffs || !Array.isArray(docDiffs.operations)) {
      console.warn('⚠️ removeDiff: No diffs found for document:', documentId);
      return;
    }

    console.log('🔍 removeDiff: Current operations:', docDiffs.operations.length, 
                'Operations:', docDiffs.operations.map(op => ({
                  operationId: op.operationId,
                  id: op.id,
                  start: op.start,
                  end: op.end,
                  fallbackId: op.start + '-' + op.end
                })));

    const initialLength = docDiffs.operations.length;
    docDiffs.operations = docDiffs.operations.filter(op => {
      // Operation ID can be in various formats
      const opId = op.operationId || op.id || op.start + '-' + op.end;
      const matches = opId === operationId || String(opId) === String(operationId);
      console.log('🔍 Checking operation:', { opId, operationId, matches });
      return !matches;
    });

    console.log('🔍 removeDiff: After filter:', docDiffs.operations.length, 'removed:', initialLength - docDiffs.operations.length);

    if (docDiffs.operations.length !== initialLength) {
      // ✅ If no operations left, clear the entire document entry
      if (docDiffs.operations.length === 0) {
        console.log('🗑️ No diffs remaining for document, clearing entry:', documentId);
        delete this.diffs[documentId];
      }
      
      this.saveToStorage();
      this.notify(documentId, 'remove');
      
      console.log('✅ Removed diff, remaining count:', docDiffs.operations.length);
    } else {
      console.warn('⚠️ removeDiff: No operations were removed! operationId not found:', operationId);
    }
  }

  /**
   * Validate diffs against current document content
   * Marks diffs as stale when content hash changes significantly
   * @param {string} documentId - Document identifier
   * @param {string} currentContent - Current document content
   * @returns {Object} { invalidated: Array of operation IDs, isStale: boolean }
   */
  validateDiffs(documentId, currentContent) {
    if (!documentId) return { invalidated: [], isStale: false };

    const docDiffs = this.diffs[documentId];
    if (!docDiffs) return { invalidated: [], isStale: false };

    const currentHash = this._hashContent(currentContent || '');
    const storedHash = docDiffs.contentHash || '';
    const invalidated = [];
    let isStale = false;

    // Compare hashes: if they differ, content has changed
    if (currentHash !== storedHash) {
      // Parse hash components to determine severity
      const currentParts = currentHash.split('_');
      const storedParts = storedHash.split('_');
      
      if (currentParts.length === storedParts.length && currentParts.length >= 2) {
        const currentLength = parseInt(currentParts[0], 10) || 0;
        const storedLength = parseInt(storedParts[0], 10) || 0;
        
        // Calculate length change percentage
        const lengthChange = Math.abs(currentLength - storedLength);
        const maxLength = Math.max(currentLength, storedLength, 1);
        const changePercent = (lengthChange / maxLength) * 100;
        
        // Check if sample hashes match (indicates where change occurred)
        let matchingSamples = 0;
        for (let i = 1; i < Math.min(currentParts.length, storedParts.length); i++) {
          if (currentParts[i] === storedParts[i]) {
            matchingSamples++;
          }
        }
        
        // Mark as stale if:
        // - Length changed by more than 5%
        // - OR less than 3 out of 5 samples match (major structural change)
        if (changePercent > 5 || matchingSamples < 3) {
          isStale = true;
          console.warn(`⚠️ Document content has drifted significantly:`, {
            documentId,
            lengthChange: `${changePercent.toFixed(1)}%`,
            matchingSamples: `${matchingSamples}/${currentParts.length - 1}`,
            storedLength,
            currentLength
          });
        }
      } else {
        // Hash format changed - assume stale
        isStale = true;
      }
      
      // Update stored hash
      docDiffs.contentHash = currentHash;
      this.saveToStorage();
    }

    // Store stale status for UI to display
    docDiffs.isStale = isStale;

    return { invalidated, isStale };
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
      console.log('📝 DocumentDiffStore: Listener subscribed, total listeners:', this.listeners.size);
    }
  }

  /**
   * Unsubscribe from diff changes
   * @param {Function} callback - Callback function to remove
   */
  unsubscribe(callback) {
    this.listeners.delete(callback);
    console.log('📝 DocumentDiffStore: Listener unsubscribed, total listeners:', this.listeners.size);
  }

  /**
   * Notify all listeners of a change
   * @param {string} documentId - Document identifier
   * @param {string} changeType - Type of change: 'set', 'clear', 'remove', 'invalidate'
   */
  notify(documentId, changeType) {
    console.log('📢 DocumentDiffStore: Notifying listeners', { 
      documentId, 
      changeType, 
      listenerCount: this.listeners.size,
      remainingDiffs: this.diffs[documentId]?.operations?.length || 0
    });
    
    this.listeners.forEach(callback => {
      try {
        callback(documentId, changeType);
      } catch (error) {
        console.error('DocumentDiffStore: Listener error:', error);
      }
    });
  }

  /**
   * Hash content for validation using multi-point sampling
   * Samples content at multiple positions to detect drift even when middle sections change
   * @param {string} content - Content to hash
   * @returns {string} Hash string
   * @private
   */
  _hashContent(content) {
    if (!content) return '';
    
    const length = content.length;
    if (length === 0) return '0_0_0_0_0_0';
    
    // Sample at multiple points: 0%, 25%, 50%, 75%, 100%
    // Sample size: 50 chars at each position (or available chars if near boundaries)
    const sampleSize = 50;
    const positions = [
      0, // Start
      Math.floor(length * 0.25), // Quarter
      Math.floor(length * 0.5), // Middle
      Math.floor(length * 0.75), // Three-quarters
      Math.max(0, length - sampleSize) // End
    ];
    
    // Extract samples and compute simple hash for each
    const samples = positions.map(pos => {
      const end = Math.min(length, pos + sampleSize);
      const sample = content.substring(pos, end);
      // Simple hash: sum of char codes (fast, not cryptographic)
      let hash = 0;
      for (let i = 0; i < sample.length; i++) {
        hash = (hash * 31 + sample.charCodeAt(i)) >>> 0;
      }
      return hash.toString(16);
    });
    
    // Combine: length + all sample hashes
    return `${length}_${samples.join('_')}`;
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

