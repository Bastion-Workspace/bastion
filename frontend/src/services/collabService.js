import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';

const COLLAB_COLORS = ['#30bced', '#6eeb83', '#ffbc42', '#e84855', '#8ac926', '#ff6b6b'];

function wsOrigin() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}`;
}

function authToken() {
  return localStorage.getItem('auth_token') || localStorage.getItem('token') || '';
}

function hashToIndex(str) {
  let h = 0;
  const s = String(str || '');
  for (let i = 0; i < s.length; i += 1) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

/**
 * @param {string} documentId
 * @param {{ user_id?: string, display_name?: string, username?: string }} user
 * @param {{ onStatus?: (status: string) => void, onSynced?: (synced: boolean) => void }} [callbacks]
 */
export function createCollabSession(documentId, user, callbacks = {}) {
  const token = authToken();
  if (!token) {
    throw new Error('Not authenticated');
  }

  const ydoc = new Y.Doc();
  const ytext = ydoc.getText('content');
  const undoManager = new Y.UndoManager(ytext);

  const serverUrl = `${wsOrigin()}/api/ws/collab`;
  const provider = new WebsocketProvider(serverUrl, documentId, ydoc, {
    params: { token },
    connect: true,
  });

  const { awareness } = provider;
  const displayName = user?.display_name || user?.username || user?.user_id || 'User';
  const colorIdx = hashToIndex(user?.user_id || displayName) % COLLAB_COLORS.length;
  const color = COLLAB_COLORS[colorIdx];
  awareness.setLocalStateField('user', {
    name: displayName,
    color,
    colorLight: `${color}40`,
  });

  const onStatus = (payload) => {
    const arr = Array.isArray(payload) ? payload : [payload];
    const status = arr[0]?.status;
    if (status) callbacks.onStatus?.(status);
  };
  const onSynced = (args) => {
    const synced = Array.isArray(args) ? args[0] : args;
    callbacks.onSynced?.(!!synced);
  };

  provider.on('status', onStatus);
  provider.on('synced', onSynced);

  return {
    ydoc,
    ytext,
    provider,
    awareness,
    undoManager,
    documentId,
    _onStatus: onStatus,
    _onSynced: onSynced,
  };
}

/**
 * @param {ReturnType<typeof createCollabSession> | null | undefined} session
 */
export function destroyCollabSession(session) {
  if (!session) return;
  try {
    session.provider?.off('status', session._onStatus);
    session.provider?.off('synced', session._onSynced);
  } catch (_) {
    /* ignore */
  }
  try {
    session.provider?.destroy();
  } catch (_) {
    /* ignore */
  }
  try {
    session.ydoc?.destroy();
  } catch (_) {
    /* ignore */
  }
}

/**
 * Update websocket query params before next reconnect (e.g. refreshed JWT).
 * @param {ReturnType<typeof createCollabSession>} session
 */
export function updateCollabAuthToken(session) {
  if (session?.provider) {
    session.provider.params = { ...session.provider.params, token: authToken() };
  }
}
