/**
 * Message Ordering Service
 * Handles proper message ordering and state consistency
 */

class MessageOrderingService {
  constructor() {
    this.messageCache = new Map(); // Cache messages by conversation
    this.loadingStates = new Map(); // Track loading states
  }

  /**
   * Sort messages by timestamp and sequence number
   */
  sortMessages(messages) {
    return [...messages].sort((a, b) => {
      // First, sort by sequence number if available
      if (a.sequence_number && b.sequence_number) {
        return a.sequence_number - b.sequence_number;
      }
      
      // Fallback to timestamp sorting
      const timeA = new Date(a.timestamp || a.created_at || 0);
      const timeB = new Date(b.timestamp || b.created_at || 0);
      
      if (timeA.getTime() !== timeB.getTime()) {
        return timeA - timeB;
      }
      
      // If timestamps are identical, use message ID for consistent ordering
      return (a.id || a.message_id || '').localeCompare(b.id || b.message_id || '');
    });
  }

  /**
   * Normalize message format for consistent handling
   */
  normalizeMessage(message) {
    return {
      id: message.id || message.message_id || `temp_${Date.now()}_${Math.random()}`,
      message_id: message.message_id || message.id,
      type: message.type || (message.message_type === 'user' ? 'user' : 'assistant'),
      content: message.content || '',
      timestamp: message.timestamp || message.created_at || new Date().toISOString(),
      created_at: message.created_at || message.timestamp,
      sequence_number: message.sequence_number || 0,
      citations: message.citations || [],
      metadata: message.metadata || {},
      // Preserve all other properties
      ...message
    };
  }

  /**
   * Merge new messages with existing ones, avoiding duplicates
   */
  mergeMessages(existingMessages, newMessages, conversationId) {
    console.log(`🔄 Merging messages for conversation ${conversationId}:`, {
      existing: existingMessages.length,
      new: newMessages.length
    });

    // Normalize all messages
    const normalizedExisting = existingMessages.map(msg => this.normalizeMessage(msg));
    const normalizedNew = newMessages.map(msg => this.normalizeMessage(msg));

    // Create a map of existing messages by ID for fast lookup
    const existingMap = new Map();
    normalizedExisting.forEach(msg => {
      existingMap.set(msg.id, msg);
      if (msg.message_id && msg.message_id !== msg.id) {
        existingMap.set(msg.message_id, msg);
      }
    });

    // Add new messages that don't already exist
    const mergedMessages = [...normalizedExisting];
    
    for (const newMsg of normalizedNew) {
      const isDuplicate = existingMap.has(newMsg.id) || 
                         (newMsg.message_id && existingMap.has(newMsg.message_id)) ||
                         this.isContentDuplicate(newMsg, normalizedExisting);
      
      if (!isDuplicate) {
        mergedMessages.push(newMsg);
        existingMap.set(newMsg.id, newMsg);
        if (newMsg.message_id && newMsg.message_id !== newMsg.id) {
          existingMap.set(newMsg.message_id, newMsg);
        }
      } else {
        console.log(`🔍 Skipping duplicate message: ${newMsg.id}`);
      }
    }

    // Sort the merged messages
    const sortedMessages = this.sortMessages(mergedMessages);
    
    console.log(`✅ Message merge completed: ${sortedMessages.length} total messages`);
    return sortedMessages;
  }

  /**
   * Check if a message is a content duplicate
   */
  isContentDuplicate(newMessage, existingMessages) {
    // Skip content duplicate check for very short messages
    if (!newMessage.content || newMessage.content.length < 10) {
      return false;
    }

    return existingMessages.some(existing => {
      // Exact content match
      if (existing.content === newMessage.content) {
        return true;
      }

      // Similar content check for longer messages
      if (existing.content && newMessage.content && 
          existing.content.length > 50 && newMessage.content.length > 50) {
        const similarity = this.calculateSimilarity(existing.content, newMessage.content);
        return similarity > 0.95;
      }

      return false;
    });
  }

  /**
   * Calculate content similarity
   */
  calculateSimilarity(str1, str2) {
    const shorter = str1.length < str2.length ? str1 : str2;
    const longer = str1.length >= str2.length ? str1 : str2;
    
    if (shorter.length === 0) return 0;
    
    // Simple similarity based on common characters
    let commonChars = 0;
    for (let i = 0; i < shorter.length; i++) {
      if (longer.includes(shorter[i])) {
        commonChars++;
      }
    }
    
    return commonChars / shorter.length;
  }

  /**
   * Filter out invalid or corrupted messages
   */
  filterValidMessages(messages) {
    return messages.filter(msg => {
      // Must have basic required fields
      if (!msg.content && !msg.isPending) {
        console.warn(`⚠️ Filtering out message with no content: ${msg.id}`);
        return false;
      }

      // Must have a valid type
      if (!msg.type || !['user', 'assistant', 'bot'].includes(msg.type)) {
        console.warn(`⚠️ Filtering out message with invalid type: ${msg.type}`);
        return false;
      }

      return true;
    });
  }

  /**
   * Process messages for a conversation with full ordering and deduplication
   */
  processConversationMessages(messages, conversationId) {
    console.log(`🔄 Processing ${messages.length} messages for conversation ${conversationId}`);

    // Filter out invalid messages
    const validMessages = this.filterValidMessages(messages);
    
    // Normalize message format
    const normalizedMessages = validMessages.map(msg => this.normalizeMessage(msg));
    
    // Sort messages properly
    const sortedMessages = this.sortMessages(normalizedMessages);
    
    // Cache the processed messages
    this.messageCache.set(conversationId, sortedMessages);
    
    console.log(`✅ Processed ${sortedMessages.length} valid messages for conversation ${conversationId}`);
    return sortedMessages;
  }

  /**
   * Get cached messages for a conversation
   */
  getCachedMessages(conversationId) {
    return this.messageCache.get(conversationId) || [];
  }

  /**
   * Clear cache for a conversation
   */
  clearCache(conversationId) {
    this.messageCache.delete(conversationId);
    this.loadingStates.delete(conversationId);
  }

  /**
   * Clear all caches
   */
  clearAllCaches() {
    this.messageCache.clear();
    this.loadingStates.clear();
  }

  /**
   * Set loading state for a conversation
   */
  setLoadingState(conversationId, isLoading) {
    this.loadingStates.set(conversationId, isLoading);
  }

  /**
   * Check if conversation is loading
   */
  isLoading(conversationId) {
    return this.loadingStates.get(conversationId) || false;
  }

  /**
   * Add a single message to a conversation
   */
  addMessage(conversationId, newMessage) {
    const existingMessages = this.getCachedMessages(conversationId);
    const normalizedMessage = this.normalizeMessage(newMessage);
    
    // Check for duplicates
    const isDuplicate = existingMessages.some(existing => 
      existing.id === normalizedMessage.id || 
      existing.message_id === normalizedMessage.message_id ||
      (existing.content === normalizedMessage.content && existing.type === normalizedMessage.type)
    );
    
    if (!isDuplicate) {
      const updatedMessages = this.sortMessages([...existingMessages, normalizedMessage]);
      this.messageCache.set(conversationId, updatedMessages);
      return updatedMessages;
    }
    
    return existingMessages;
  }

  /**
   * Remove a message from a conversation
   */
  removeMessage(conversationId, messageId) {
    const existingMessages = this.getCachedMessages(conversationId);
    const filteredMessages = existingMessages.filter(msg => 
      msg.id !== messageId && msg.message_id !== messageId
    );
    
    this.messageCache.set(conversationId, filteredMessages);
    return filteredMessages;
  }

  /**
   * Update a message in a conversation
   */
  updateMessage(conversationId, messageId, updates) {
    const existingMessages = this.getCachedMessages(conversationId);
    const updatedMessages = existingMessages.map(msg => {
      if (msg.id === messageId || msg.message_id === messageId) {
        return { ...msg, ...updates };
      }
      return msg;
    });
    
    this.messageCache.set(conversationId, updatedMessages);
    return updatedMessages;
  }
}

// Export singleton instance
export default new MessageOrderingService();
