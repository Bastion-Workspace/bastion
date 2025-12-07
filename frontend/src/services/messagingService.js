/**
 * Roosevelt's Messaging Service
 * Frontend API wrapper for user-to-user messaging
 * 
 * BULLY! A well-organized messaging service is like a well-organized cavalry charge!
 */

import apiService from './apiService';

class MessagingService {
  constructor() {
    this.wsConnections = new Map(); // room_id -> WebSocket
    this.messageHandlers = new Map(); // room_id -> callback
    this.presenceHandlers = new Set(); // Set of presence callbacks
    this.roomUpdateHandlers = new Set(); // Set of room update callbacks
    this.newRoomHandlers = new Set(); // Set of new room callbacks
    this.reconnectTimeouts = new Map(); // room_id -> timeout
    this.userWebSocket = null; // User-level WebSocket for all room notifications
  }

  // =====================
  // ROOM OPERATIONS
  // =====================

  async createRoom(participantIds, roomName = null) {
    try {
      const response = await apiService.post('/api/messaging/rooms', {
        participant_ids: participantIds,
        room_name: roomName
      });
      // ApiServiceBase returns JSON directly, not wrapped in .data
      return response;
    } catch (error) {
      console.error('❌ Failed to create room:', error);
      throw error;
    }
  }

  async getUserRooms(limit = 20) {
    try {
      const response = await apiService.get('/api/messaging/rooms', {
        params: { limit }
      });
      // ApiServiceBase returns JSON directly, not wrapped in .data
      return response.rooms;
    } catch (error) {
      console.error('❌ Failed to get user rooms:', error);
      throw error;
    }
  }

  async updateRoomName(roomId, roomName) {
    try {
      const response = await apiService.put(`/api/messaging/rooms/${roomId}/name`, {
        room_name: roomName
      });
      // ApiServiceBase returns JSON directly, not wrapped in .data
      return response;
    } catch (error) {
      console.error('❌ Failed to update room name:', error);
      throw error;
    }
  }

  async deleteRoom(roomId) {
    try {
      const response = await apiService.delete(`/api/messaging/rooms/${roomId}`);
      // ApiServiceBase returns JSON directly, not wrapped in .data
      return response;
    } catch (error) {
      console.error('❌ Failed to delete room:', error);
      throw error;
    }
  }

  async updateNotificationSettings(roomId, settings) {
    try {
      const response = await apiService.put(`/api/messaging/rooms/${roomId}/notifications`, {
        settings
      });
      return response;
    } catch (error) {
      console.error('❌ Failed to update notification settings:', error);
      throw error;
    }
  }

  async addParticipant(roomId, userId, shareHistory = false) {
    try {
      const response = await apiService.post(`/api/messaging/rooms/${roomId}/participants`, {
        user_id: userId,
        share_history: shareHistory
      });
      // ApiServiceBase returns JSON directly, not wrapped in .data
      return response;
    } catch (error) {
      console.error('❌ Failed to add participant:', error);
      throw error;
    }
  }

  // =====================
  // MESSAGE OPERATIONS
  // =====================

  async getRoomMessages(roomId, limit = 50, beforeMessageId = null) {
    try {
      const params = { limit };
      if (beforeMessageId) {
        params.before_message_id = beforeMessageId;
      }
      
      const response = await apiService.get(`/api/messaging/rooms/${roomId}/messages`, {
        params
      });
      // ApiServiceBase returns JSON directly, not wrapped in .data
      return response;
    } catch (error) {
      console.error('❌ Failed to get room messages:', error);
      throw error;
    }
  }

  async sendMessage(roomId, content, messageType = 'text', metadata = null) {
    try {
      const response = await apiService.post(`/api/messaging/rooms/${roomId}/messages`, {
        content,
        message_type: messageType,
        metadata
      });
      // ApiServiceBase returns JSON directly, not wrapped in .data
      return response;
    } catch (error) {
      console.error('❌ Failed to send message:', error);
      throw error;
    }
  }

  async deleteMessage(messageId, deleteFor = 'me') {
    try {
      const response = await apiService.delete(`/api/messaging/messages/${messageId}`, {
        params: { delete_for: deleteFor }
      });
      // ApiServiceBase returns JSON directly, not wrapped in .data
      return response;
    } catch (error) {
      console.error('❌ Failed to delete message:', error);
      throw error;
    }
  }

  // =====================
  // ATTACHMENT OPERATIONS
  // =====================

  async uploadAttachment(roomId, messageId, file) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await apiService.post(
        `/api/messaging/rooms/${roomId}/messages/${messageId}/attachments`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      return response;
    } catch (error) {
      console.error('Failed to upload attachment:', error);
      throw error;
    }
  }

  async getAttachment(attachmentId) {
    try {
      const response = await apiService.get(`/api/messaging/attachments/${attachmentId}`);
      return response;
    } catch (error) {
      console.error('Failed to get attachment:', error);
      throw error;
    }
  }

  getAttachmentUrl(attachmentId) {
    const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
    return `/api/messaging/attachments/${attachmentId}/file${token ? `?token=${token}` : ''}`;
  }

  async getMessageAttachments(messageId) {
    try {
      const response = await apiService.get(`/api/messaging/messages/${messageId}/attachments`);
      return response.attachments || [];
    } catch (error) {
      console.error('Failed to get message attachments:', error);
      throw error;
    }
  }

  // =====================
  // REACTION OPERATIONS
  // =====================

  async addReaction(messageId, emoji) {
    try {
      const response = await apiService.post(`/api/messaging/messages/${messageId}/reactions`, {
        emoji
      });
      // ApiServiceBase returns JSON directly, not wrapped in .data
      return response;
    } catch (error) {
      console.error('❌ Failed to add reaction:', error);
      throw error;
    }
  }

  async removeReaction(reactionId) {
    try {
      const response = await apiService.delete(`/api/messaging/reactions/${reactionId}`);
      // ApiServiceBase returns JSON directly, not wrapped in .data
      return response;
    } catch (error) {
      console.error('❌ Failed to remove reaction:', error);
      throw error;
    }
  }

  // =====================
  // PRESENCE OPERATIONS
  // =====================

  async updatePresence(status, statusMessage = null) {
    try {
      const response = await apiService.put('/api/messaging/presence', {
        status,
        status_message: statusMessage
      });
      // ApiServiceBase returns JSON directly, not wrapped in .data
      return response;
    } catch (error) {
      console.error('❌ Failed to update presence:', error);
      throw error;
    }
  }

  async getUserPresence(userId) {
    try {
      const response = await apiService.get(`/api/messaging/presence/${userId}`);
      // ApiServiceBase returns JSON directly, not wrapped in .data
      return response;
    } catch (error) {
      console.error('❌ Failed to get user presence:', error);
      throw error;
    }
  }

  async getRoomPresence(roomId) {
    try {
      const response = await apiService.get(`/api/messaging/rooms/${roomId}/presence`);
      // ApiServiceBase returns JSON directly, not wrapped in .data
      return response.presence;
    } catch (error) {
      console.error('❌ Failed to get room presence:', error);
      throw error;
    }
  }

  async getUnreadCounts() {
    try {
      const response = await apiService.get('/api/messaging/unread-counts');
      // ApiServiceBase returns JSON directly, not wrapped in .data
      return response;
    } catch (error) {
      console.error('❌ Failed to get unread counts:', error);
      throw error;
    }
  }

  // =====================
  // WEBSOCKET OPERATIONS
  // =====================

  connectToRoom(roomId, onMessage, onPresenceUpdate) {
    // Don't reconnect if already connected
    if (this.wsConnections.has(roomId)) {
      console.log(`💬 Already connected to room ${roomId}`);
      return;
    }

    const token = localStorage.getItem('auth_token');
    if (!token) {
      console.error('❌ No auth token available for WebSocket');
      return;
    }

    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/messaging/ws/${roomId}?token=${token}`;
    
    try {
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log(`💬 BULLY! Connected to room ${roomId}`);
        this.wsConnections.set(roomId, ws);
        
        // Start heartbeat
        this.startHeartbeat(roomId);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'new_message') {
            if (onMessage) {
              onMessage(data.message);
            }
          } else if (data.type === 'presence_update') {
            if (onPresenceUpdate) {
              onPresenceUpdate(data);
            }
            
            // Call global presence handlers
            this.presenceHandlers.forEach(handler => {
              handler(data);
            });
          } else if (data.type === 'room_updated') {
            // Call room update handlers
            this.roomUpdateHandlers.forEach(handler => {
              handler(data);
            });
          } else if (data.type === 'participant_added') {
            // Call room update handlers (participants are part of room state)
            this.roomUpdateHandlers.forEach(handler => {
              handler(data);
            });
          } else if (data.type === 'typing') {
            // Handle typing indicators if needed
            if (this.messageHandlers.has(roomId)) {
              this.messageHandlers.get(roomId)(data);
            }
          }
        } catch (error) {
          console.error('❌ Failed to parse WebSocket message:', error);
        }
      };
      
      ws.onclose = () => {
        console.log(`💬 Disconnected from room ${roomId}`);
        this.wsConnections.delete(roomId);
        
        // Attempt reconnection after delay
        const timeout = setTimeout(() => {
          console.log(`🔄 Attempting to reconnect to room ${roomId}...`);
          this.connectToRoom(roomId, onMessage, onPresenceUpdate);
        }, 3000);
        
        this.reconnectTimeouts.set(roomId, timeout);
      };
      
      ws.onerror = (error) => {
        console.error(`❌ WebSocket error for room ${roomId}:`, error);
      };
    } catch (error) {
      console.error(`❌ Failed to create WebSocket for room ${roomId}:`, error);
    }
  }

  disconnectFromRoom(roomId) {
    // Clear reconnect timeout if exists
    if (this.reconnectTimeouts.has(roomId)) {
      clearTimeout(this.reconnectTimeouts.get(roomId));
      this.reconnectTimeouts.delete(roomId);
    }
    
    // Close WebSocket if exists
    if (this.wsConnections.has(roomId)) {
      const ws = this.wsConnections.get(roomId);
      ws.close();
      this.wsConnections.delete(roomId);
      console.log(`💬 Disconnected from room ${roomId}`);
    }
    
    // Remove message handler
    this.messageHandlers.delete(roomId);
  }

  startHeartbeat(roomId) {
    const ws = this.wsConnections.get(roomId);
    if (!ws) return;
    
    // Send heartbeat every 25 seconds (before the 30s server threshold)
    const heartbeatInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'heartbeat' }));
      } else {
        clearInterval(heartbeatInterval);
      }
    }, 25000);
    
    // Store interval for cleanup
    ws.heartbeatInterval = heartbeatInterval;
  }

  sendTypingIndicator(roomId, isTyping) {
    const ws = this.wsConnections.get(roomId);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'typing',
        is_typing: isTyping
      }));
    }
  }

  registerPresenceHandler(callback) {
    this.presenceHandlers.add(callback);
    
    // Return unregister function
    return () => {
      this.presenceHandlers.delete(callback);
    };
  }

  registerRoomUpdateHandler(callback) {
    this.roomUpdateHandlers.add(callback);
    
    // Return unregister function
    return () => {
      this.roomUpdateHandlers.delete(callback);
    };
  }

  registerNewRoomHandler(callback) {
    this.newRoomHandlers.add(callback);
    
    // Return unregister function
    return () => {
      this.newRoomHandlers.delete(callback);
    };
  }

  // User-level WebSocket for notifications across all rooms
  connectUserWebSocket() {
    if (this.userWebSocket) {
      console.log('💬 User WebSocket already connected');
      return;
    }

    const token = localStorage.getItem('token');
    if (!token) {
      console.error('❌ No token available for user WebSocket');
      return;
    }

    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/messaging/ws/user?token=${token}`;
    
    console.log('💬 Connecting user WebSocket...');
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('✅ User WebSocket connected');
      this.userWebSocket = ws;
      
      // Start heartbeat
      const heartbeatInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'heartbeat' }));
        }
      }, 30000);
      ws.heartbeatInterval = heartbeatInterval;
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'new_message') {
          // Notify all handlers about new message
          this.roomUpdateHandlers.forEach(handler => {
            handler(data);
          });
        } else if (data.type === 'new_room') {
          // Notify about new room
          this.newRoomHandlers.forEach(handler => {
            handler(data);
          });
        } else if (data.type === 'room_updated') {
          // Room name or details updated
          this.roomUpdateHandlers.forEach(handler => {
            handler(data);
          });
        } else if (data.type === 'participant_added') {
          // Participant added to room
          this.roomUpdateHandlers.forEach(handler => {
            handler(data);
          });
        } else if (data.type === 'presence_update') {
          // Presence update
          this.presenceHandlers.forEach(handler => {
            handler(data);
          });
        }
      } catch (error) {
        console.error('❌ Failed to parse user WebSocket message:', error);
      }
    };
    
    ws.onclose = () => {
      console.log('💬 User WebSocket disconnected');
      if (ws.heartbeatInterval) {
        clearInterval(ws.heartbeatInterval);
      }
      this.userWebSocket = null;
      
      // Attempt reconnection after delay
      setTimeout(() => {
        console.log('🔄 Attempting to reconnect user WebSocket...');
        this.connectUserWebSocket();
      }, 3000);
    };
    
    ws.onerror = (error) => {
      console.error('❌ User WebSocket error:', error);
    };
  }

  disconnectUserWebSocket() {
    if (this.userWebSocket) {
      if (this.userWebSocket.heartbeatInterval) {
        clearInterval(this.userWebSocket.heartbeatInterval);
      }
      this.userWebSocket.close();
      this.userWebSocket = null;
      console.log('💬 User WebSocket disconnected');
    }
  }

  // =====================
  // CLEANUP
  // =====================

  disconnectAll() {
    // Clear all reconnect timeouts
    this.reconnectTimeouts.forEach(timeout => clearTimeout(timeout));
    this.reconnectTimeouts.clear();
    
    // Close all WebSocket connections
    this.wsConnections.forEach((ws, roomId) => {
      if (ws.heartbeatInterval) {
        clearInterval(ws.heartbeatInterval);
      }
      ws.close();
    });
    this.wsConnections.clear();
    
    // Close user WebSocket
    this.disconnectUserWebSocket();
    
    // Clear handlers
    this.messageHandlers.clear();
    this.presenceHandlers.clear();
    this.roomUpdateHandlers.clear();
    this.newRoomHandlers.clear();
    
    console.log('💬 All messaging connections closed');
  }
}

// Global instance
const messagingService = new MessagingService();

export default messagingService;

