/**
 * Client-side presence display helpers.
 * PRESENCE_STALE_MS must stay aligned with backend config.PRESENCE_OFFLINE_THRESHOLD_SECONDS (default 90).
 */

export const PRESENCE_STALE_MS = 90 * 1000;

export function parseLastSeenMs(value) {
  if (value == null || value === '') return null;
  const t = new Date(value).getTime();
  return Number.isNaN(t) ? null : t;
}

/**
 * @param {{ status?: string, last_seen_at?: string|Date|null }|null|undefined} entry
 * @param {number} [nowMs]
 * @returns {'online'|'away'|'offline'}
 */
export function getEffectiveDisplayStatus(entry, nowMs = Date.now()) {
  if (!entry) return 'offline';
  const status = entry.status || 'offline';
  if (status === 'offline') return 'offline';
  const lastMs = parseLastSeenMs(entry.last_seen_at);
  if (lastMs == null) {
    if (status === 'online' || status === 'away') return status;
    return 'offline';
  }
  if (nowMs - lastMs > PRESENCE_STALE_MS) return 'offline';
  return status === 'away' ? 'away' : status === 'online' ? 'online' : 'offline';
}

/**
 * Prefer live messaging context entry; fall back to team REST member shape.
 * @param {object|undefined} contextPresence - from MessagingContext presence[userId]
 * @param {{ is_online?: boolean, last_seen?: string|null }} [restMember]
 */
export function mergePresenceFromContextAndRest(contextPresence, restMember) {
  if (
    contextPresence &&
    (contextPresence.status != null || contextPresence.last_seen_at != null)
  ) {
    return {
      status: contextPresence.status || 'offline',
      last_seen_at: contextPresence.last_seen_at ?? null,
      status_message: contextPresence.status_message ?? null,
    };
  }
  if (restMember) {
    return {
      status: restMember.is_online ? 'online' : 'offline',
      last_seen_at: restMember.last_seen || restMember.last_seen_at || null,
      status_message: null,
    };
  }
  return { status: 'offline', last_seen_at: null, status_message: null };
}

/**
 * @param {Array<{ user_id?: string }>} participants
 * @param {string|undefined} currentUserId
 * @param {Record<string, { status?: string, last_seen_at?: string|null }>} presenceMap
 * @param {number} [nowMs]
 */
export function summarizeTeamPresence(participants, currentUserId, presenceMap, nowMs = Date.now()) {
  const others = (participants || []).filter((p) => p.user_id && p.user_id !== currentUserId);
  let activeCount = 0;
  for (const p of others) {
    const st = getEffectiveDisplayStatus(presenceMap[p.user_id], nowMs);
    if (st === 'online' || st === 'away') activeCount += 1;
  }
  return { memberCount: others.length, activeCount, others };
}
