/**
 * Roosevelt's Messaging Context
 * Global state management for user-to-user messaging
 * 
 * BULLY! Centralized state for the entire messaging cavalry!
 */

import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import messagingService from '../services/messagingService';
import { useAuth } from './AuthContext';
import tabNotificationManager from '../utils/tabNotification';

const MessagingContext = createContext();

export const useMessaging = () => {
  const context = useContext(MessagingContext);
  if (!context) {
    throw new Error('useMessaging must be used within a MessagingProvider');
  }
  return context;
};

export const MessagingProvider = ({ children }) => {
  const { user, isAuthenticated } = useAuth();
  
  // State
  const [rooms, setRooms] = useState([]);
  const [currentRoomId, setCurrentRoomId] = useState(null);
  const [messages, setMessages] = useState({}); // room_id -> messages array
  const [presence, setPresence] = useState({}); // user_id -> presence info
  const [unreadCounts, setUnreadCounts] = useState({}); // room_id -> count
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [isMessagingFullScreen, setIsMessagingFullScreen] = useState(() => {
    try {
      const saved = localStorage.getItem('messagingFullScreen');
      return saved !== null ? JSON.parse(saved) : false;
    } catch (error) {
      console.error('Failed to load messaging full-screen state from localStorage:', error);
      return false;
    }
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const presenceUpdateInterval = useRef(null);

  // =====================
  // ROOM OPERATIONS
  // =====================

  const loadRooms = useCallback(async () => {
    if (!isAuthenticated) return;
    
    try {
      setIsLoading(true);
      const userRooms = await messagingService.getUserRooms();
      setRooms(userRooms);
      
      // Update unread counts
      const counts = {};
      userRooms.forEach(room => {
        counts[room.room_id] = room.unread_count || 0;
      });
      setUnreadCounts(counts);
      
      setError(null);
    } catch (error) {
      console.error('❌ Failed to load rooms:', error);
      setError('Failed to load rooms');
    } finally {
      setIsLoading(false);
    }
  }, [isAuthenticated]);

  const createRoom = useCallback(async (participantIds, roomName = null) => {
    try {
      const newRoom = await messagingService.createRoom(participantIds, roomName);
      // Reload rooms to get full participant details and display names
      await loadRooms();
      return newRoom;
    } catch (error) {
      console.error('❌ Failed to create room:', error);
      throw error;
    }
  }, [loadRooms]);

  const updateRoomName = useCallback(async (roomId, roomName) => {
    try {
      await messagingService.updateRoomName(roomId, roomName);
      setRooms(prev => prev.map(room => 
        room.room_id === roomId ? { ...room, room_name: roomName, display_name: roomName } : room
      ));
    } catch (error) {
      console.error('❌ Failed to update room name:', error);
      throw error;
    }
  }, []);

  const deleteRoom = useCallback(async (roomId) => {
    try {
      await messagingService.deleteRoom(roomId);
      setRooms(prev => prev.filter(room => room.room_id !== roomId));
      setMessages(prev => {
        const newMessages = { ...prev };
        delete newMessages[roomId];
        return newMessages;
      });
      if (currentRoomId === roomId) {
        setCurrentRoomId(null);
      }
    } catch (error) {
      console.error('❌ Failed to delete room:', error);
      throw error;
    }
  }, [currentRoomId]);

  const addParticipantToRoom = useCallback(async (roomId, userId, shareHistory = false) => {
    try {
      await messagingService.addParticipant(roomId, userId, shareHistory);
      // Reload the room to get updated participant list
      await loadRooms();
    } catch (error) {
      console.error('❌ Failed to add participant:', error);
      throw error;
    }
  }, [loadRooms]);

  const toggleMute = useCallback(async (roomId, muted) => {
    try {
      await messagingService.updateNotificationSettings(roomId, { muted });
      // Update local state immediately
      setRooms(prev => prev.map(room =>
        room.room_id === roomId
          ? {
              ...room,
              notification_settings: {
                ...(room.notification_settings || {}),
                muted
              }
            }
          : room
      ));
    } catch (error) {
      console.error('❌ Failed to toggle mute:', error);
      throw error;
    }
  }, []);

  // =====================
  // MESSAGE OPERATIONS
  // =====================

  const loadMessages = useCallback(async (roomId, beforeMessageId = null) => {
    try {
      const response = await messagingService.getRoomMessages(roomId, 50, beforeMessageId);
      
      if (beforeMessageId) {
        // Append older messages
        setMessages(prev => ({
          ...prev,
          [roomId]: [...(prev[roomId] || []), ...response.messages]
        }));
      } else {
        // Replace with fresh messages
        setMessages(prev => ({
          ...prev,
          [roomId]: response.messages
        }));
      }
      
      // Mark room as read
      setUnreadCounts(prev => ({ ...prev, [roomId]: 0 }));
      
      return response;
    } catch (error) {
      console.error('❌ Failed to load messages:', error);
      throw error;
    }
  }, []);

  const sendMessage = useCallback(async (roomId, content, messageType = 'text', metadata = null) => {
    try {
      const newMessage = await messagingService.sendMessage(roomId, content, messageType, metadata);
      
      // Add message to local state
      setMessages(prev => ({
        ...prev,
        [roomId]: [...(prev[roomId] || []), newMessage]
      }));
      
      // Update room's last message time
      setRooms(prev => prev.map(room =>
        room.room_id === roomId
          ? { ...room, last_message_at: newMessage.created_at }
          : room
      ).sort((a, b) => new Date(b.last_message_at) - new Date(a.last_message_at)));
      
      return newMessage;
    } catch (error) {
      console.error('❌ Failed to send message:', error);
      throw error;
    }
  }, []);

  const deleteMessage = useCallback(async (roomId, messageId, deleteFor = 'me') => {
    try {
      await messagingService.deleteMessage(messageId, deleteFor);
      
      // Remove message from local state
      setMessages(prev => ({
        ...prev,
        [roomId]: (prev[roomId] || []).filter(msg => msg.message_id !== messageId)
      }));
    } catch (error) {
      console.error('❌ Failed to delete message:', error);
      throw error;
    }
  }, []);

  const addReaction = useCallback(async (roomId, messageId, emoji) => {
    try {
      const reaction = await messagingService.addReaction(messageId, emoji);
      
      // Update message reactions in local state
      setMessages(prev => ({
        ...prev,
        [roomId]: (prev[roomId] || []).map(msg =>
          msg.message_id === messageId
            ? {
                ...msg,
                reactions: [...(msg.reactions || []), {
                  ...reaction,
                  user_id: user.user_id,
                  emoji
                }]
              }
            : msg
        )
      }));
    } catch (error) {
      console.error('❌ Failed to add reaction:', error);
      throw error;
    }
  }, [user]);

  const removeReaction = useCallback(async (roomId, reactionId) => {
    try {
      await messagingService.removeReaction(reactionId);
      
      // Remove reaction from local state
      setMessages(prev => ({
        ...prev,
        [roomId]: (prev[roomId] || []).map(msg => ({
          ...msg,
          reactions: (msg.reactions || []).filter(r => r.reaction_id !== reactionId)
        }))
      }));
    } catch (error) {
      console.error('❌ Failed to remove reaction:', error);
      throw error;
    }
  }, []);

  // =====================
  // PRESENCE OPERATIONS
  // =====================

  const updatePresence = useCallback(async (status, statusMessage = null) => {
    if (!isAuthenticated) return;
    
    try {
      await messagingService.updatePresence(status, statusMessage);
    } catch (error) {
      console.error('❌ Failed to update presence:', error);
    }
  }, [isAuthenticated]);

  const loadRoomPresence = useCallback(async (roomId) => {
    try {
      const roomPresence = await messagingService.getRoomPresence(roomId);
      
      // Update global presence state
      setPresence(prev => ({ ...prev, ...roomPresence }));
    } catch (error) {
      console.error('❌ Failed to load room presence:', error);
    }
  }, []);

  // =====================
  // WEBSOCKET OPERATIONS
  // =====================

  const connectToRoom = useCallback((roomId) => {
    if (!isAuthenticated) return;
    
    const handleNewMessage = (message) => {
      setMessages(prev => ({
        ...prev,
        [roomId]: [...(prev[roomId] || []), message]
      }));
      
      // Increment unread count if not current room
      if (roomId !== currentRoomId) {
        setUnreadCounts(prev => ({
          ...prev,
          [roomId]: (prev[roomId] || 0) + 1
        }));
      }
      
      // Update room's last message time and check for notifications
      setRooms(prev => {
        const updated = prev.map(room =>
          room.room_id === roomId
            ? { ...room, last_message_at: message.created_at }
            : room
        ).sort((a, b) => new Date(b.last_message_at) - new Date(a.last_message_at));
        
        // Flash tab notification if not current room and not muted
        if (roomId !== currentRoomId) {
          const room = updated.find(r => r.room_id === roomId);
          const notificationSettings = room?.notification_settings || {};
          if (!notificationSettings.muted) {
            tabNotificationManager.startFlashing('New message');
          }
        }
        
        return updated;
      });
    };
    
    const handlePresenceUpdate = (presenceData) => {
      setPresence(prev => ({
        ...prev,
        [presenceData.user_id]: {
          status: presenceData.status,
          status_message: presenceData.status_message,
          last_seen_at: presenceData.timestamp
        }
      }));
    };
    
    messagingService.connectToRoom(roomId, handleNewMessage, handlePresenceUpdate);
  }, [isAuthenticated, currentRoomId]);

  const disconnectFromRoom = useCallback((roomId) => {
    messagingService.disconnectFromRoom(roomId);
  }, []);

  // =====================
  // ROOM SELECTION
  // =====================

  const selectRoom = useCallback((roomId) => {
    if (currentRoomId) {
      disconnectFromRoom(currentRoomId);
    }
    
    setCurrentRoomId(roomId);
    
    // Load messages if not already loaded
    if (!messages[roomId]) {
      loadMessages(roomId);
    }
    
    // Connect to room WebSocket
    connectToRoom(roomId);
    
    // Load presence
    loadRoomPresence(roomId);
    
    // Mark as read
    setUnreadCounts(prev => ({ ...prev, [roomId]: 0 }));
  }, [currentRoomId, messages, loadMessages, connectToRoom, loadRoomPresence, disconnectFromRoom]);

  // =====================
  // DRAWER MANAGEMENT
  // =====================

  const toggleDrawer = useCallback(() => {
    setIsDrawerOpen(prev => !prev);
  }, []);

  const openDrawer = useCallback(() => {
    setIsDrawerOpen(true);
  }, []);

  const closeDrawer = useCallback(() => {
    setIsDrawerOpen(false);
  }, []);

  // =====================
  // FULL-SCREEN MANAGEMENT
  // =====================

  const toggleFullScreen = useCallback(() => {
    setIsMessagingFullScreen(prev => !prev);
  }, []);

  // Load full-screen preference from localStorage
  useEffect(() => {
    const savedFullScreen = localStorage.getItem('messagingFullScreen');
    if (savedFullScreen !== null) {
      setIsMessagingFullScreen(JSON.parse(savedFullScreen));
    }
    // Auto-disable full-screen on small screens
    try {
      if (window && window.matchMedia && window.matchMedia('(max-width: 900px)').matches) {
        setIsMessagingFullScreen(false);
      }
    } catch {}
  }, []);

  // Save full-screen preference to localStorage
  useEffect(() => {
    localStorage.setItem('messagingFullScreen', JSON.stringify(isMessagingFullScreen));
  }, [isMessagingFullScreen]);

  // =====================
  // COMPUTED VALUES
  // =====================

  const totalUnreadCount = Object.values(unreadCounts).reduce((sum, count) => sum + count, 0);

  const currentRoom = rooms.find(r => r.room_id === currentRoomId);

  const currentMessages = messages[currentRoomId] || [];

  // =====================
  // EFFECTS
  // =====================

  // Load rooms on mount/auth change
  useEffect(() => {
    if (isAuthenticated) {
      loadRooms();
    } else {
      // Clear state on logout
      setRooms([]);
      setMessages({});
      setPresence({});
      setUnreadCounts({});
      setCurrentRoomId(null);
    }
  }, [isAuthenticated, loadRooms]);

  // Connect user WebSocket for global notifications
  useEffect(() => {
    if (isAuthenticated) {
      messagingService.connectUserWebSocket();
      
      return () => {
        messagingService.disconnectUserWebSocket();
      };
    }
  }, [isAuthenticated]);

  // Update presence on mount and periodically
  useEffect(() => {
    if (isAuthenticated) {
      updatePresence('online');
      
      // Update presence every 25 seconds
      presenceUpdateInterval.current = setInterval(() => {
        updatePresence('online');
      }, 25000);
      
      return () => {
        if (presenceUpdateInterval.current) {
          clearInterval(presenceUpdateInterval.current);
        }
        updatePresence('offline');
        messagingService.disconnectAll();
      };
    }
  }, [isAuthenticated, updatePresence]);

  // Register global presence handler
  useEffect(() => {
    const unregister = messagingService.registerPresenceHandler((presenceData) => {
      setPresence(prev => ({
        ...prev,
        [presenceData.user_id]: {
          status: presenceData.status,
          status_message: presenceData.status_message,
          last_seen_at: presenceData.timestamp
        }
      }));
    });
    
    return unregister;
  }, []);

  // Register room update handler
  useEffect(() => {
    const unregister = messagingService.registerRoomUpdateHandler((updateData) => {
      console.log('💬 Room update received:', updateData);
      
      if (updateData.type === 'room_updated') {
        // Room name changed
        setRooms(prev => prev.map(room =>
          room.room_id === updateData.room_id
            ? { ...room, room_name: updateData.room_name }
            : room
        ));
      } else if (updateData.type === 'participant_added') {
        // Participant added - reload room to get updated participant list
        loadRooms();
      } else if (updateData.type === 'new_message') {
        // New message received - update unread count and timestamp
        const message = updateData.message;
        const roomId = message.room_id;
        
        // Update room timestamp
        setRooms(prev => {
          const updated = prev.map(room =>
            room.room_id === roomId
              ? { ...room, last_message_at: message.created_at }
              : room
          ).sort((a, b) => new Date(b.last_message_at) - new Date(a.last_message_at));
          
          // Flash tab notification if not current room and not muted
          if (roomId !== currentRoomId) {
            const room = updated.find(r => r.room_id === roomId);
            const notificationSettings = room?.notification_settings || {};
            if (!notificationSettings.muted) {
              tabNotificationManager.startFlashing('New message');
            }
          }
          
          return updated;
        });
        
        // Increment unread count if not current room
        if (roomId !== currentRoomId) {
          setUnreadCounts(prev => ({
            ...prev,
            [roomId]: (prev[roomId] || 0) + 1
          }));
        }
      }
    });
    
    return unregister;
  }, [loadRooms, currentRoomId]);

  // Register new room handler
  useEffect(() => {
    const unregister = messagingService.registerNewRoomHandler((data) => {
      console.log('💬 New room received:', data);
      
      // Add new room to list
      setRooms(prev => [data.room, ...prev]);
      
      // Initialize unread count to 0
      setUnreadCounts(prev => ({
        ...prev,
        [data.room.room_id]: 0
      }));
    });
    
    return unregister;
  }, []);

  const value = {
    // State
    rooms,
    currentRoomId,
    currentRoom,
    messages: currentMessages,
    presence,
    unreadCounts,
    totalUnreadCount,
    isDrawerOpen,
    isMessagingFullScreen,
    isLoading,
    error,
    
    // Room operations
    loadRooms,
    createRoom,
    updateRoomName,
    deleteRoom,
    addParticipantToRoom,
    selectRoom,
    toggleMute,
    
    // Message operations
    loadMessages,
    sendMessage,
    deleteMessage,
    addReaction,
    removeReaction,
    
    // Presence operations
    updatePresence,
    loadRoomPresence,
    
    // WebSocket operations
    connectToRoom,
    disconnectFromRoom,
    
    // Drawer management
    toggleDrawer,
    openDrawer,
    closeDrawer,
    
    // Full-screen management
    setIsMessagingFullScreen,
    toggleFullScreen,
  };

  return (
    <MessagingContext.Provider value={value}>
      {children}
    </MessagingContext.Provider>
  );
};

