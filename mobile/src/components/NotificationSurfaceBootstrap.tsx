import { useEffect, useRef } from 'react';
import { useAuth } from '../context/AuthContext';
import { startNotificationsSurfaceSocket } from '../api/notificationsSurface';
import { getActiveConversationForNotifications } from '../session/activeConversationRef';

/**
 * Keeps /api/ws/conversations open for surface signaling when authenticated.
 */
export function NotificationSurfaceBootstrap(): null {
  const { token, apiConfigured } = useAuth();
  const stopRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!token || !apiConfigured) {
      stopRef.current?.();
      stopRef.current = null;
      return;
    }
    const { close } = startNotificationsSurfaceSocket(() => getActiveConversationForNotifications());
    stopRef.current = close;
    return () => {
      close();
      stopRef.current = null;
    };
  }, [token, apiConfigured]);

  return null;
}
