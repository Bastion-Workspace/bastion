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

  getConversationMessages = async (conversationId, skip = 0, limit = 100, skipCheckpoint = false) => {
    const params = new URLSearchParams({ skip, limit });
    if (skipCheckpoint) params.set('skip_checkpoint', 'true');
    return this.get(`/api/conversations/${conversationId}/messages?${params}`);
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

  // Conversation sharing methods
  shareConversation = async (conversationId, userId, shareType = 'read', expiresAt = null) => {
    return this.post(`/api/conversations/${conversationId}/share`, {
      shared_with_user_id: userId,
      share_type: shareType,
      expires_at: expiresAt
    });
  }

  unshareConversation = async (conversationId, shareId) => {
    return this.delete(`/api/conversations/${conversationId}/share/${shareId}`);
  }

  getConversationShares = async (conversationId) => {
    return this.get(`/api/conversations/${conversationId}/shares`);
  }

  getSharedConversations = async (skip = 0, limit = 50) => {
    return this.get(`/api/conversations/shared-with-me?skip=${skip}&limit=${limit}`);
  }

  getConversationParticipants = async (conversationId) => {
    return this.get(`/api/conversations/${conversationId}/participants`);
  }

  updateSharePermissions = async (conversationId, shareId, shareType) => {
    return this.put(`/api/conversations/${conversationId}/share/${shareId}`, {
      share_type: shareType
    });
  }

  // Attachment methods
  uploadAttachment = async (conversationId, messageId, file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    return this.post(
      `/api/conversations/${conversationId}/messages/${messageId}/attachments`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
  }

  getAttachmentUrl = (conversationId, messageId, attachmentId) => {
    return `/api/conversations/${conversationId}/messages/${messageId}/attachments/${attachmentId}`;
  }

  deleteAttachment = async (conversationId, messageId, attachmentId) => {
    return this.delete(`/api/conversations/${conversationId}/messages/${messageId}/attachments/${attachmentId}`);
  }

  listAttachments = async (conversationId, messageId) => {
    return this.get(`/api/conversations/${conversationId}/messages/${messageId}/attachments`);
  }
}

export default new ConversationService();
