import { apiRequest } from './client';
import { wsUrlFromHttpBase } from './config';
import { getStoredToken } from '../session/tokenStore';

export type Room = {
  room_id: string;
  room_name?: string | null;
  name?: string | null;
  last_message_at?: string | null;
};

export async function getUserRooms(limit = 50): Promise<Room[]> {
  const res = await apiRequest<{ rooms?: Room[] }>(`/api/messaging/rooms?limit=${limit}`);
  return res.rooms ?? [];
}

export type MessagingMessage = {
  message_id: string;
  room_id: string;
  content: string;
  message_type?: string;
  created_at: string;
  user_id?: string;
};

export async function getRoomMessages(
  roomId: string,
  limit = 50,
  beforeMessageId?: string | null
): Promise<{ messages?: MessagingMessage[] }> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (beforeMessageId) params.set('before_message_id', beforeMessageId);
  return apiRequest(`/api/messaging/rooms/${roomId}/messages?${params.toString()}`);
}

export async function sendRoomMessage(
  roomId: string,
  content: string,
  messageType = 'text'
): Promise<MessagingMessage> {
  return apiRequest(`/api/messaging/rooms/${roomId}/messages`, {
    method: 'POST',
    body: JSON.stringify({
      content,
      message_type: messageType,
      metadata: {},
    }),
  });
}

export type RoomSocketHandlers = {
  onMessage?: (msg: MessagingMessage) => void;
  onPresence?: (data: unknown) => void;
  onTyping?: (data: { user_id?: string; is_typing?: boolean }) => void;
};

export async function openRoomWebSocket(
  roomId: string,
  handlers: RoomSocketHandlers
): Promise<WebSocket> {
  const token = await getStoredToken();
  if (!token) {
    throw new Error('Not authenticated');
  }
  const url = wsUrlFromHttpBase(`/api/messaging/ws/${roomId}`, { token });
  const ws = new WebSocket(url);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data as string) as {
        type?: string;
        message?: MessagingMessage;
        user_id?: string;
        is_typing?: boolean;
      };
      if (data.type === 'new_message' && data.message && handlers.onMessage) {
        handlers.onMessage(data.message);
      } else if (data.type === 'presence_update' && handlers.onPresence) {
        handlers.onPresence(data);
      } else if (data.type === 'typing' && handlers.onTyping) {
        handlers.onTyping(data);
      }
    } catch {
      /* ignore parse errors */
    }
  };

  const heartbeat = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'heartbeat' }));
    }
  }, 25000);

  ws.addEventListener('close', () => clearInterval(heartbeat));

  return ws;
}

export async function openUserWebSocket(onEvent: (data: unknown) => void): Promise<WebSocket> {
  const token = await getStoredToken();
  if (!token) {
    throw new Error('Not authenticated');
  }
  const url = wsUrlFromHttpBase('/api/messaging/ws/user', { token });
  const ws = new WebSocket(url);
  ws.onmessage = (event) => {
    try {
      onEvent(JSON.parse(event.data as string));
    } catch {
      /* ignore */
    }
  };
  const heartbeat = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'heartbeat' }));
    }
  }, 30000);
  ws.addEventListener('close', () => clearInterval(heartbeat));
  return ws;
}

export function sendTyping(ws: WebSocket, isTyping: boolean): void {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'typing', is_typing: isTyping }));
  }
}
