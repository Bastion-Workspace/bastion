/** Legacy global key (pre per-user isolation). */
export const LEGACY_CHAT_CONVERSATION_STORAGE_KEY = 'chatSidebarCurrentConversation';

const SESSION_ACTIVE_PREFIX = 'bastion_ui_active_conversation_id';

export function persistedActiveConversationLocalKey(userId) {
  if (!userId) return null;
  return `${LEGACY_CHAT_CONVERSATION_STORAGE_KEY}:${userId}`;
}

export function activeConversationSessionStorageKey(userId) {
  if (!userId) return null;
  return `${SESSION_ACTIVE_PREFIX}:${userId}`;
}

/** Remove legacy + all per-user persisted active conversation ids (e.g. cache invalidation). */
export function clearAllPersistedChatConversationLocalKeys() {
  try {
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i += 1) {
      const k = localStorage.key(i);
      if (!k) continue;
      if (k === LEGACY_CHAT_CONVERSATION_STORAGE_KEY) {
        keysToRemove.push(k);
      } else if (k.startsWith(`${LEGACY_CHAT_CONVERSATION_STORAGE_KEY}:`)) {
        keysToRemove.push(k);
      }
    }
    keysToRemove.forEach((k) => localStorage.removeItem(k));
  } catch {
    /* ignore */
  }
}
