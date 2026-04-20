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
   * Merge proposal diffs for a document (multi-proposal support).
   * Replaces only ops belonging to this proposalId; preserves ops from other proposals.
   * @param {string} documentId - Document identifier
   * @param {Array} operations - Array of diff operations for this proposal
   * @param {string} proposalId - Proposal ID (messageId) for these ops
   * @param {string} contentSnapshot - Current document content (for validation)
   */
  mergeProposalDiff(documentId, operations, proposalId, contentSnapshot) {
    if (!documentId || !proposalId) return;
    const existing = this.diffs[documentId];
    const existingOps = existing?.operations || [];
    const filtered = existingOps.filter(op => op.messageId !== proposalId);
    const tagged = (Array.isArray(operations) ? operations : []).map((op, idx) => ({
      ...op,
      messageId: proposalId,
      operationId: op.operationId || op.id || `${proposalId}-${idx}-${op.start}-${op.end}`
    }));
    const merged = [...filtered, ...tagged];
    if (import.meta.env.DEV) {
      for (let i = 0; i < merged.length; i++) {
        for (let j = i + 1; j < merged.length; j++) {
          const a = merged[i];
          const b = merged[j];
          if (a.messageId === b.messageId) continue;
          const as = a.start;
          const ae = a.end;
          const bs = b.start;
          const be = b.end;
          if (as == null || ae == null || bs == null || be == null) continue;
          if (!(ae <= bs || be <= as)) {
            console.warn('Overlapping ops from different proposals:', a, b);
          }
        }
      }
    }
    this.diffs[documentId] = {
      operations: merged,
      messageId: proposalId,
      timestamp: Date.now(),
      contentHash: this._hashContent(contentSnapshot || '')
    };
    this.saveToStorage();
    this.notify(documentId, 'set');
  }

  /**
   * Save a full snapshot of operations (plugin save-back). No top-level messageId.
   * @param {string} documentId - Document identifier
   * @param {Array} operations - Full array of operations (each op has its own messageId)
   * @param {string} contentSnapshot - Current document content (for validation)
   */
  saveSnapshot(documentId, operations, contentSnapshot) {
    if (!documentId) return;
    this.diffs[documentId] = {
      operations: Array.isArray(operations) ? operations : [],
      messageId: null,
      timestamp: Date.now(),
      contentHash: this._hashContent(contentSnapshot || '')
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
   * Remove all diffs for a proposal (batch clear). One filter, one save, one notify.
   * @param {string} documentId - Document identifier
   * @param {string} proposalId - Proposal ID (messageId) to remove
   */
  removeProposalDiffs(documentId, proposalId) {
    if (!documentId || !proposalId) return;
    const docDiffs = this.diffs[documentId];
    if (!docDiffs || !Array.isArray(docDiffs.operations)) return;
    const before = docDiffs.operations.length;
    docDiffs.operations = docDiffs.operations.filter(op => op.messageId !== proposalId);
    if (docDiffs.operations.length === before) return;
    if (docDiffs.operations.length === 0) {
      delete this.diffs[documentId];
    }
    this.saveToStorage();
    this.notify(documentId, 'remove');
  }

  /**
   * Remove a specific diff operation
   * @param {string} documentId - Document identifier
   * @param {string} operationId - Operation ID to remove
   */
  removeDiff(documentId, operationId) {
    if (!documentId || !operationId) {
      console.warn('DocumentDiffStore removeDiff: Missing documentId or operationId');
      return;
    }

    const docDiffs = this.diffs[documentId];
    if (!docDiffs || !Array.isArray(docDiffs.operations)) {
      console.warn('DocumentDiffStore removeDiff: No diffs found for document:', documentId);
      return;
    }

    const initialLength = docDiffs.operations.length;
    docDiffs.operations = docDiffs.operations.filter(op => {
      const opId = op.operationId || op.id || `${op.start}-${op.end}`;
      if (opId === operationId || String(opId) === String(operationId)) return false;

      // Backward compat: legacy ops without operationId/id may have been shown with a
      // synthetic plugin id: messageId-idx-start-end. Match prefix + range suffix.
      if (!op.operationId && !op.id && op.messageId != null) {
        const prefix = String(op.messageId) + '-';
        const suffix = '-' + op.start + '-' + op.end;
        if (String(operationId).startsWith(prefix) && String(operationId).endsWith(suffix)) {
          return false;
        }
      }
      return true;
    });

    if (docDiffs.operations.length !== initialLength) {
      if (docDiffs.operations.length === 0) {
        delete this.diffs[documentId];
      }
      this.saveToStorage();
      this.notify(documentId, 'remove');
    } else {
      console.warn('DocumentDiffStore removeDiff: No operations were removed, operationId not found:', operationId);
    }
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
        const parsed = JSON.parse(stored);
        const originalKeyCount = Object.keys(parsed).length;
        this.diffs = parsed;
        // Clean up old diffs (older than 24 hours)
        const now = Date.now();
        const maxAge = 24 * 60 * 60 * 1000; // 24 hours
        
        Object.keys(this.diffs).forEach(docId => {
          const diff = this.diffs[docId];
          if (diff.timestamp && (now - diff.timestamp) > maxAge) {
            delete this.diffs[docId];
          }
        });

        let migrated = false;
        Object.values(this.diffs).forEach(diff => {
          if (diff?.operations && Array.isArray(diff.operations)) {
            const indexByMessage = {};
            diff.operations.forEach(op => {
              if (!op.operationId && !op.id) {
                const mid = op.messageId || 'op';
                const idx = indexByMessage[mid] ?? 0;
                indexByMessage[mid] = idx + 1;
                op.operationId = `${mid}-${idx}-${op.start}-${op.end}`;
                migrated = true;
              }
            });
          }
        });

        const needsSave = Object.keys(this.diffs).length !== originalKeyCount || migrated;
        if (needsSave) {
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

