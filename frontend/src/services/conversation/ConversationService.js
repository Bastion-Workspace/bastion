import ApiServiceBase from '../base/ApiServiceBase';

class ConversationService extends ApiServiceBase {
  // Conversation methods
  createConversation = async (data = {}) => {
    return this.post('/api/conversations', {
      title: data.title || null,
      description: data.description || null,
      tags: data.tags || [],
      folder_id: data.folder_id || null,
      initial_message: data.initial_message || null
    });
  }

  reorderConversations = async (conversationIds, orderLocked = false) => {
    return this.post('/api/conversations/reorder', {
      conversation_ids: conversationIds,
      order_locked: orderLocked
    });
  }

  listConversations = async (skip = 0, limit = 50) => {
    return this.get(`/api/conversations?skip=${skip}&limit=${limit}`);
  }

  getConversation = async (conversationId) => {
    return this.get(`/api/conversations/${conversationId}`);
  }

  getConversationMessages = async (conversationId, skip = 0, limit = 100) => {
    return this.get(`/api/conversations/${conversationId}/messages?skip=${skip}&limit=${limit}`);
  }

  addMessageToConversation = async (conversationId, messageData) => {
    return this.post(`/api/conversations/${conversationId}/messages`, messageData);
  }

  updateConversation = async (conversationId, title, updates = {}) => {
    return this.put(`/api/conversations/${conversationId}`, { title, ...updates });
  }

  deleteConversation = async (conversationId) => {
    console.log('🌐 API: Deleting conversation:', conversationId);
    try {
      const result = await this.delete(`/api/conversations/${conversationId}`);
      console.log('🌐 API: Delete conversation result:', result);
      return result;
    } catch (error) {
      console.error('🌐 API: Delete conversation error:', error);
      throw error;
    }
  }

  deleteAllConversations = async () => {
    console.log('🌐 API: Deleting ALL conversations');
    try {
      const result = await this.delete('/api/conversations');
      console.log('🌐 API: Delete all conversations result:', result);
      return result;
    } catch (error) {
      console.error('🌐 API: Delete all conversations error:', error);
      throw error;
    }
  }
}

export default new ConversationService();
