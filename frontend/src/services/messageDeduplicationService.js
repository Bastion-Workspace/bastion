/**
 * Message Deduplication and Ordering Service
 * SIMPLIFIED VERSION - PostgreSQL is the single source of truth
 * This service now only handles ordering and basic validation
 */

class MessageDeduplicationService {
  constructor() {
    this.conversationMessages = new Map(); // Track messages per conversation for debugging
    this.pendingOperations = new Map(); // Track pending async operations
  }

  /**
   * Simple message ordering without aggressive deduplication
   * Since PostgreSQL is the single source of truth, we only need basic ordering
   */
  orderMessages(messages, conversationId) {
    console.log(`🔄 Ordering ${messages.length} messages for conversation ${conversationId}`);
    
    // Sort messages by timestamp and sequence
    const sortedMessages = [...messages].sort((a, b) => {
      // First sort by sequence number if available
      const seqA = a.sequence_number || 0;
      const seqB = b.sequence_number || 0;
      if (seqA !== seqB) {
        return seqA - seqB;
      }
      
      // Then by timestamp
      const timeA = new Date(a.timestamp || a.created_at || 0);
      const timeB = new Date(b.timestamp || b.created_at || 0);
      return timeA - timeB;
    });

    console.log(`✅ Ordered ${sortedMessages.length} messages`);
    return sortedMessages;
  }

  /**
   * DEPRECATED: Legacy method for backward compatibility
   * Now just calls orderMessages without deduplication
   */
  deduplicateAndOrderMessages(messages, conversationId) {
    console.log(`🔄 Legacy deduplication call - using simple ordering for ${messages.length} messages`);
    return this.orderMessages(messages, conversationId);
  }

  /**
   * Check if a message might be a duplicate (for debugging only)
   * This is now only used for logging, not for removal
   */
  isDuplicateMessage(message, conversationId, existingMessages = []) {
    // Only check for exact duplicates with same ID
    const exactDuplicate = existingMessages.some(existing => 
      existing.id === message.id || 
      existing.message_id === message.message_id
    );
    
    if (exactDuplicate) {
      console.log(`🔍 Exact duplicate detected: ${message.id || message.message_id}`);
      return true;
    }
    
    return false;
  }

  /**
   * Register a message (for debugging tracking only)
   */
  registerMessage(message, conversationId) {
    // Track conversation messages for debugging
    if (!this.conversationMessages.has(conversationId)) {
      this.conversationMessages.set(conversationId, new Set());
    }
    this.conversationMessages.get(conversationId).add(message.id || message.message_id);
  }

  /**
   * Clear tracking data for a conversation
   */
  clearConversation(conversationId) {
    console.log(`🧹 Clearing tracking data for conversation ${conversationId}`);
    this.conversationMessages.delete(conversationId);
    this.pendingOperations.delete(conversationId);
  }

  /**
   * Clear all tracking data
   */
  clearAll() {
    console.log(`🧹 Clearing all tracking data`);
    this.conversationMessages.clear();
    this.pendingOperations.clear();
  }

  /**
   * Track a pending operation to prevent race conditions
   */
  trackPendingOperation(conversationId, operationType, operationId) {
    if (!this.pendingOperations.has(conversationId)) {
      this.pendingOperations.set(conversationId, new Map());
    }
    
    this.pendingOperations.get(conversationId).set(operationType, {
      operationId,
      timestamp: new Date()
    });
  }

  /**
   * Check if an operation is already pending
   */
  isPendingOperation(conversationId, operationType, operationId) {
    const conversationOps = this.pendingOperations.get(conversationId);
    if (!conversationOps) return false;
    
    const pending = conversationOps.get(operationType);
    return pending && pending.operationId === operationId;
  }

  /**
   * Complete a pending operation
   */
  completePendingOperation(conversationId, operationType) {
    const conversationOps = this.pendingOperations.get(conversationId);
    if (conversationOps) {
      conversationOps.delete(operationType);
    }
  }

  /**
   * Get statistics for debugging
   */
  getStats() {
    return {
      conversationsTracked: this.conversationMessages.size,
      pendingOperations: Array.from(this.pendingOperations.entries()).map(([convId, ops]) => ({
        conversationId: convId,
        operations: Array.from(ops.keys())
      }))
    };
  }
}

// Export singleton instance
export default new MessageDeduplicationService();
