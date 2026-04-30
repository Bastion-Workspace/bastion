import { AppState, type AppStateStatus } from 'react-native';
import { getStoredToken } from '../session/tokenStore';
import { assertApiBaseUrl, wsUrlFromHttpBase } from './config';
import { getOrCreateMobileSurfaceId } from '../session/surfaceIdStore';

type Unlisten = () => void;

/**
 * Lightweight WebSocket to /api/ws/conversations for surface_meta / surface_state only.
 */
export function startNotificationsSurfaceSocket(
  getActiveConversationId: () => string | null
): { close: Unlisten } {
  let ws: WebSocket | null = null;
  let heartbeat: ReturnType<typeof setInterval> | null = null;
  let appSub: { remove: () => void } | null = null;
  let surfaceId: string | null = null;

  const connect = async () => {
    const token = await getStoredToken();
    if (!token) return;
    const base = assertApiBaseUrl();
    surfaceId = await getOrCreateMobileSurfaceId();
    const url = wsUrlFromHttpBase('/api/ws/conversations', { token });
    ws = new WebSocket(url);
    ws.onopen = () => {
      ws?.send(
        JSON.stringify({
          type: 'surface_meta',
          surface_id: surfaceId,
          surface_type: 'mobile',
        })
      );
      const pushState = (status: AppStateStatus) => {
        if (!ws || ws.readyState !== WebSocket.OPEN || !surfaceId) return;
        const active = getActiveConversationId() || '';
        const state = status === 'active' ? 'active' : 'background';
        ws.send(
          JSON.stringify({
            type: 'surface_state',
            surface_id: surfaceId,
            state,
            active_conversation_id: active,
          })
        );
      };
      pushState(AppState.currentState);
      appSub = AppState.addEventListener('change', (next) => {
        pushState(next);
      });
      heartbeat = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'heartbeat' }));
        }
      }, 30000);
    };
    ws.onerror = () => {
      /* non-fatal */
    };
    ws.onclose = () => {
      if (heartbeat) clearInterval(heartbeat);
      heartbeat = null;
      appSub?.remove();
      appSub = null;
    };
  };

  void connect();

  return {
    close: () => {
      if (heartbeat) clearInterval(heartbeat);
      appSub?.remove();
      appSub = null;
      try {
        ws?.close();
      } catch {
        /* ignore */
      }
      ws = null;
    },
  };
}
