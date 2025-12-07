/**
 * Editor Diff Coordinator
 * 
 * Validates and coordinates edit operations before displaying as live diffs.
 * Provides utilities for checking operation validity, detecting conflicts,
 * and filtering stale operations.
 */

/**
 * Compute simple hash for text (matches backend pre_hash validation)
 * @param {string} text - Text to hash
 * @returns {string} Hex hash string
 */
export function computeHash(text) {
  let h = 0;
  for (let i = 0; i < text.length; i++) {
    h = (h * 31 + text.charCodeAt(i)) >>> 0;
  }
  return h.toString(16);
}

/**
 * Validate that an operation is still valid for the current document
 * @param {Object} operation - Operation object with start, end, pre_hash, etc.
 * @param {string} currentDocText - Current document text
 * @returns {{valid: boolean, reason?: string}} Validation result
 */
export function validateOperation(operation, currentDocText) {
  const start = Number(operation.start || 0);
  const end = Number(operation.end || start);
  const opType = operation.op_type || 'replace_range';
  
  // Check range validity
  if (start < 0 || end < start || end > currentDocText.length) {
    return {
      valid: false,
      reason: `Invalid range: [${start}, ${end}] for document length ${currentDocText.length}`
    };
  }
  
  // Check pre_hash if provided
  if (operation.pre_hash && operation.pre_hash.length > 0) {
    const originalSlice = currentDocText.slice(start, end);
    const computedHash = computeHash(originalSlice);
    if (computedHash !== operation.pre_hash) {
      return {
        valid: false,
        reason: `Pre-hash mismatch: document changed since operation was generated`
      };
    }
  }
  
  // For replace_range, check if original_text matches
  if (opType === 'replace_range' && operation.original_text) {
    const currentText = currentDocText.slice(start, end);
    if (currentText !== operation.original_text) {
      return {
        valid: false,
        reason: `Original text mismatch: document changed`
      };
    }
  }
  
  return { valid: true };
}

/**
 * Detect conflicts between operations (overlapping ranges)
 * @param {Array<Object>} operations - Array of operation objects
 * @returns {Array<{op1: number, op2: number, reason: string}>} Array of conflicts
 */
export function detectConflicts(operations) {
  const conflicts = [];
  
  for (let i = 0; i < operations.length; i++) {
    for (let j = i + 1; j < operations.length; j++) {
      const op1 = operations[i];
      const op2 = operations[j];
      
      const start1 = Number(op1.start || 0);
      const end1 = Number(op1.end || start1);
      const start2 = Number(op2.start || 0);
      const end2 = Number(op2.end || start2);
      
      // Check for overlap
      if (!(end1 <= start2 || end2 <= start1)) {
        conflicts.push({
          op1: i,
          op2: j,
          reason: `Operations overlap: [${start1}, ${end1}] and [${start2}, ${end2}]`
        });
      }
    }
  }
  
  return conflicts;
}

/**
 * Filter out stale operations that are no longer valid
 * @param {Array<Object>} operations - Array of operation objects
 * @param {string} currentDocText - Current document text
 * @returns {Array<Object>} Filtered array of valid operations
 */
export function filterStaleOperations(operations, currentDocText) {
  return operations.filter(op => {
    const validation = validateOperation(op, currentDocText);
    if (!validation.valid) {
      console.warn('Filtering stale operation:', validation.reason, op);
      return false;
    }
    return true;
  });
}

/**
 * Sort operations by position (highest index first for safe batch application)
 * @param {Array<Object>} operations - Array of operation objects
 * @returns {Array<Object>} Sorted operations
 */
export function sortOperationsForApplication(operations) {
  return operations.slice().sort((a, b) => {
    const startA = Number(a.start || 0);
    const startB = Number(b.start || 0);
    return startB - startA; // Descending order
  });
}

/**
 * Check if cursor position is within any operation range
 * @param {number} cursorPos - Cursor position
 * @param {Array<Object>} operations - Array of operation objects
 * @returns {Object|null} Operation containing cursor, or null
 */
export function findOperationAtCursor(cursorPos, operations) {
  for (const op of operations) {
    const start = Number(op.start || 0);
    const end = Number(op.end || start);
    if (cursorPos >= start && cursorPos <= end) {
      return op;
    }
  }
  return null;
}


