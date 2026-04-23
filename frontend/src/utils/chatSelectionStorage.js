/** Legacy global key (pre per-user isolation). */
export const LEGACY_CHAT_CONVERSATION_STORAGE_KEY = 'chatSidebarCurrentConversation';

/** Legacy global keys for chat UI preferences (pre per-user isolation). */
const LEGACY_CHAT_MODEL_STORAGE_KEY = 'chatSidebarSelectedModel';
const LEGACY_USER_EDITOR_PREF_STORAGE_KEY = 'userEditorPreference';

const SESSION_ACTIVE_PREFIX = 'bastion_ui_active_conversation_id';

export function persistedActiveConversationLocalKey(userId) {
  if (!userId) return null;
  return `${LEGACY_CHAT_CONVERSATION_STORAGE_KEY}:${userId}`;
}

export function activeConversationSessionStorageKey(userId) {
  if (!userId) return null;
  return `${SESSION_ACTIVE_PREFIX}:${userId}`;
}

export function persistedChatModelLocalKey(userId) {
  if (!userId) return null;
  return `${LEGACY_CHAT_MODEL_STORAGE_KEY}:${userId}`;
}

export function persistedUserEditorPreferenceLocalKey(userId) {
  if (!userId) return null;
  return `${LEGACY_USER_EDITOR_PREF_STORAGE_KEY}:${userId}`;
}

/**
 * Read scoped value; if missing, copy legacy global key into scoped key once and remove legacy.
 */
function readScopedWithLegacyMigrate(scopedKey, legacyKey) {
  if (!scopedKey) return '';
  try {
    let v = localStorage.getItem(scopedKey);
    if (v != null && v !== '' && v !== 'null') return v;
    const leg = localStorage.getItem(legacyKey);
    if (leg != null && leg !== '' && leg !== 'null') {
      localStorage.setItem(scopedKey, leg);
      localStorage.removeItem(legacyKey);
      return leg;
    }
    return '';
  } catch {
    return '';
  }
}

export function readPersistedChatModelForUser(userId) {
  const k = persistedChatModelLocalKey(userId);
  const v = readScopedWithLegacyMigrate(k, LEGACY_CHAT_MODEL_STORAGE_KEY);
  return v && v !== 'null' ? v : '';
}

export function readPersistedUserEditorPreferenceForUser(userId) {
  const k = persistedUserEditorPreferenceLocalKey(userId);
  const v = readScopedWithLegacyMigrate(k, LEGACY_USER_EDITOR_PREF_STORAGE_KEY);
  if (!v || v === 'null') return 'prefer';
  return v;
}

export function writePersistedChatModelForUser(userId, modelId) {
  const k = persistedChatModelLocalKey(userId);
  if (!k) return;
  try {
    if (modelId) localStorage.setItem(k, modelId);
    else localStorage.removeItem(k);
  } catch {
    /* ignore */
  }
}

export function writePersistedUserEditorPreferenceForUser(userId, pref) {
  const k = persistedUserEditorPreferenceLocalKey(userId);
  if (!k) return;
  try {
    if (pref) localStorage.setItem(k, pref);
    else localStorage.removeItem(k);
  } catch {
    /* ignore */
  }
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
