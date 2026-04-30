/**
 * Notification Context
 * State for agent-initiated in-app notifications (bell icon).
 * Persists agent-initiated notifications; no WebSocket management here.
 */

import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react';
import { useAuth } from './AuthContext';

const STORAGE_KEY_PREFIX = 'agent_notifications_';
const MAX_NOTIFICATIONS = 50;

const NotificationContext = createContext();

export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider');
  }
  return context;
};

const loadPersisted = (userId) => {
  if (!userId) return { unreadCount: 0, notifications: [] };
  try {
    const raw = localStorage.getItem(`${STORAGE_KEY_PREFIX}${userId}`);
    if (!raw) return { unreadCount: 0, notifications: [] };
    const data = JSON.parse(raw);
    return {
      unreadCount: typeof data.unreadCount === 'number' ? data.unreadCount : 0,
      notifications: Array.isArray(data.notifications) ? data.notifications.slice(0, MAX_NOTIFICATIONS) : [],
    };
  } catch {
    return { unreadCount: 0, notifications: [] };
  }
};

const persist = (userId, unreadCount, notifications) => {
  if (!userId) return;
  try {
    localStorage.setItem(
      `${STORAGE_KEY_PREFIX}${userId}`,
      JSON.stringify({
        unreadCount,
        notifications: notifications.slice(0, MAX_NOTIFICATIONS),
      })
    );
  } catch (e) {
    console.warn('Failed to persist notification state', e);
  }
};

export const NotificationProvider = ({ children }) => {
  const { user } = useAuth();
  const userId = user?.user_id || null;

  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCountState] = useState(0);
  const persistDirtyRef = useRef(false);

  useEffect(() => {
    const { unreadCount: savedCount, notifications: savedList } = loadPersisted(userId);
    setUnreadCountState(savedCount);
    setNotifications(savedList);
  }, [userId]);

  useEffect(() => {
    if (userId && persistDirtyRef.current) {
      persist(userId, unreadCount, notifications);
      persistDirtyRef.current = false;
    }
  }, [userId, unreadCount, notifications]);

  const addNotification = useCallback((payload) => {
    const item = {
      id: (payload.notification_id && String(payload.notification_id)) || `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      ...payload,
      read: false,
      timestamp: payload.timestamp || new Date().toISOString(),
    };
    setNotifications((prev) => [item, ...prev].slice(0, MAX_NOTIFICATIONS));
    setUnreadCountState((prev) => prev + 1);
    persistDirtyRef.current = true;
  }, []);

  const markOneRead = useCallback((id) => {
    setNotifications((prev) => {
      const wasUnread = prev.some((n) => n.id === id && !n.read);
      if (wasUnread) setUnreadCountState((c) => Math.max(0, c - 1));
      return prev.map((n) => (n.id === id && !n.read ? { ...n, read: true } : n));
    });
    persistDirtyRef.current = true;
  }, []);

  const markAllRead = useCallback(() => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
    setUnreadCountState(0);
    persistDirtyRef.current = true;
  }, []);

  const dismissNotification = useCallback((id) => {
    setNotifications((prev) => {
      const target = prev.find((n) => n.id === id);
      if (target && !target.read) {
        setUnreadCountState((c) => Math.max(0, c - 1));
      }
      return prev.filter((n) => n.id !== id);
    });
    persistDirtyRef.current = true;
  }, []);

  const clearNotifications = useCallback(() => {
    setNotifications([]);
    setUnreadCountState(0);
    persistDirtyRef.current = true;
  }, []);

  const value = {
    notifications,
    unreadCount,
    addNotification,
    markAllRead,
    markOneRead,
    dismissNotification,
    clearNotifications,
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  );
};
