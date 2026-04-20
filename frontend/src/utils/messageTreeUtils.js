/**
 * Utilities for conversation message tree (branching / edit-and-resend).
 */

/**
 * @param {Array<object>} flatMessages
 * @returns {{ byId: Map<string, object>, roots: object[] }}
 */
export function buildMessageTree(flatMessages) {
  const byId = new Map();
  if (!Array.isArray(flatMessages)) {
    return { byId, roots: [] };
  }
  flatMessages.forEach((m) => {
    const id = m.message_id || m.id;
    if (!id) return;
    byId.set(id, { ...m, _children: [] });
  });
  const roots = [];
  byId.forEach((node) => {
    const p = node.parent_message_id;
    if (p && byId.has(p)) {
      byId.get(p)._children.push(node);
    } else {
      roots.push(node);
    }
  });
  return { byId, roots };
}

function stripInternal(node) {
  if (!node) return null;
  const { _children, ...rest } = node;
  return rest;
}

/**
 * Ordered messages from root to current leaf along parent chain.
 * @param {{ byId: Map<string, object> }} tree
 * @param {string|null} currentNodeId
 * @returns {object[]}
 */
export function getActivePath(tree, currentNodeId) {
  if (!tree?.byId?.size || !currentNodeId || !tree.byId.has(currentNodeId)) {
    return [];
  }
  const chain = [];
  let cur = tree.byId.get(currentNodeId);
  while (cur) {
    chain.push(cur);
    const p = cur.parent_message_id;
    cur = p && tree.byId.has(p) ? tree.byId.get(p) : null;
  }
  return chain.reverse().map((n) => stripInternal(n));
}

/**
 * @param {{ byId: Map<string, object> }} tree
 * @param {string} messageId
 * @returns {{ siblings: object[], index: number, total: number }|null}
 */
export function getSiblings(tree, messageId) {
  if (!tree?.byId?.has(messageId)) return null;
  const node = tree.byId.get(messageId);
  const parentId = node.parent_message_id ?? null;
  const siblings = [];
  tree.byId.forEach((n) => {
    const np = n.parent_message_id ?? null;
    if (np === parentId) {
      siblings.push(n);
    }
  });
  siblings.sort((a, b) => {
    const ta = new Date(a.created_at || a.timestamp || 0).getTime();
    const tb = new Date(b.created_at || b.timestamp || 0).getTime();
    if (ta !== tb) return ta - tb;
    return (a.sequence_number || 0) - (b.sequence_number || 0);
  });
  const index = siblings.findIndex((s) => (s.message_id || s.id) === messageId);
  return {
    siblings: siblings.map((s) => stripInternal(s)),
    index: index >= 0 ? index : 0,
    total: siblings.length,
  };
}

/**
 * @param {number} direction -1 or +1
 * @returns {string|null} next sibling message_id
 */
export function getNextSibling(tree, messageId, direction) {
  const info = getSiblings(tree, messageId);
  if (!info || info.total <= 1) return null;
  let idx = info.index + direction;
  if (idx < 0) idx = info.total - 1;
  if (idx >= info.total) idx = 0;
  const s = info.siblings[idx];
  return s ? s.message_id || s.id : null;
}

/**
 * From a message id, walk down only while there is exactly one child.
 * Use before getActivePath when current_node_message_id may lag (e.g. still on the
 * user message while a single assistant reply exists as child) so the path includes descendants.
 * Stops at a branch (multiple children) so fork navigation stays correct.
 * @param {{ byId: Map<string, object> }} tree
 * @param {string} messageId
 * @returns {string|null}
 */
export function extendToLinearLeaf(tree, messageId) {
  if (!tree?.byId?.has(messageId)) return messageId;
  let cur = tree.byId.get(messageId);
  while (cur._children && cur._children.length === 1) {
    cur = cur._children[0];
  }
  const id = cur.message_id || cur.id;
  return id != null ? String(id) : messageId;
}
