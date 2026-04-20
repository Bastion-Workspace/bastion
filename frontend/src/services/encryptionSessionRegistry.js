/**
 * In-memory per-document encryption session tokens for multi-tab UX.
 * Tokens are not persisted. Central heartbeat keeps server sessions alive when
 * DocumentViewer is unmounted but the document tab remains open.
 */

import apiService from './apiService';
import { setEncryptionSessionTokenResolver } from './document/DocumentService';

const HEARTBEAT_MS = 60000;
const SESSION_EVENT = 'bastion-encryption-session-lost';

/** @type {Map<string, string>} */
const sessions = new Map();

let heartbeatIntervalId = null;

function ensureHeartbeat() {
  if (heartbeatIntervalId != null) return;
  heartbeatIntervalId = setInterval(() => {
    tickHeartbeats();
  }, HEARTBEAT_MS);
}

function maybeStopHeartbeat() {
  if (sessions.size === 0 && heartbeatIntervalId != null) {
    clearInterval(heartbeatIntervalId);
    heartbeatIntervalId = null;
  }
}

function dispatchSessionLost(documentId) {
  try {
    window.dispatchEvent(
      new CustomEvent(SESSION_EVENT, { detail: { document_id: documentId } })
    );
  } catch (_) {
    /* ignore */
  }
}

async function tickHeartbeats() {
  if (sessions.size === 0) return;
  const entries = Array.from(sessions.entries());
  await Promise.all(
    entries.map(async ([documentId, token]) => {
      try {
        await apiService.encryptionHeartbeat(documentId, token);
      } catch {
        sessions.delete(documentId);
        maybeStopHeartbeat();
        dispatchSessionLost(documentId);
      }
    })
  );
}

/**
 * @param {string} documentId
 * @param {string} sessionToken
 */
export function set(documentId, sessionToken) {
  if (!documentId || !sessionToken) return;
  sessions.set(documentId, sessionToken);
  ensureHeartbeat();
}

/**
 * @param {string} documentId
 * @returns {string|undefined}
 */
export function get(documentId) {
  if (!documentId) return undefined;
  return sessions.get(documentId);
}

/**
 * Remove session locally without calling the server (e.g. after failed heartbeat).
 * @param {string} documentId
 */
export function clear(documentId) {
  if (!documentId) return;
  sessions.delete(documentId);
  maybeStopHeartbeat();
}

/**
 * Server lock + remove from registry.
 * @param {string} documentId
 */
export async function lockAndRemove(documentId) {
  if (!documentId) return;
  const had = sessions.has(documentId);
  sessions.delete(documentId);
  maybeStopHeartbeat();
  if (had) {
    try {
      await apiService.lockEncryptedDocument(documentId);
    } catch {
      /* best-effort */
    }
  }
}

/**
 * Lock all registered sessions (e.g. page unload, logout).
 */
export async function lockAllRegistered() {
  const ids = Array.from(sessions.keys());
  if (heartbeatIntervalId != null) {
    clearInterval(heartbeatIntervalId);
    heartbeatIntervalId = null;
  }
  sessions.clear();
  await Promise.all(
    ids.map((documentId) =>
      apiService.lockEncryptedDocument(documentId).catch(() => {})
    )
  );
}

/**
 * Best-effort synchronous-style unload: fire locks without blocking navigation.
 */
export function lockAllRegisteredFireAndForget() {
  const ids = Array.from(sessions.keys());
  if (heartbeatIntervalId != null) {
    clearInterval(heartbeatIntervalId);
    heartbeatIntervalId = null;
  }
  sessions.clear();
  for (const documentId of ids) {
    void apiService.lockEncryptedDocument(documentId).catch(() => {});
  }
}

setEncryptionSessionTokenResolver((documentId) => get(documentId));

export const ENCRYPTION_SESSION_LOST_EVENT = SESSION_EVENT;

if (typeof window !== 'undefined' && !window.__bastionEncryptionBeforeUnloadHook) {
  window.__bastionEncryptionBeforeUnloadHook = true;
  window.addEventListener('beforeunload', () => {
    lockAllRegisteredFireAndForget();
  });
}
