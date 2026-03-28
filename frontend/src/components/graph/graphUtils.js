/**
 * Graph utilities: community detection (Louvain), degree map, and color palette for communities.
 */
import Graph from 'graphology';
import louvain from 'graphology-communities-louvain';

/**
 * Run Louvain community detection on the given nodes and edges.
 * @param {Array<{ id: string }>} nodes - Array of node objects with at least id
 * @param {Array<{ source: string, target: string, weight?: number }>} edges - Array of edges
 * @returns {Record<string, number>} Map of node id -> community index (0-based)
 */
export function computeCommunities(nodes, edges) {
  if (!nodes?.length) return {};
  const g = new Graph({ multi: false, type: 'undirected' });
  nodes.forEach((n) => {
    if (n.id && !g.hasNode(n.id)) g.addNode(n.id);
  });
  edges.forEach((e) => {
    if (!e.source || !e.target) return;
    if (!g.hasNode(e.source) || !g.hasNode(e.target)) return;
    try {
      if (!g.hasEdge(e.source, e.target)) {
        g.addEdge(e.source, e.target, e.weight != null ? { weight: e.weight } : {});
      }
    } catch (_) {
      // ignore duplicate / invalid edges
    }
  });
  return louvain(g);
}

/** 12-color palette for community coloring (distinct, works in light and dark). */
const COMMUNITY_COLORS = [
  '#1976d2', '#2e7d32', '#ed6c02', '#9c27b0', '#0288d1', '#388e3c',
  '#d32f2f', '#7b1fa2', '#00796b', '#f57c00', '#5d4037', '#455a64',
];

/**
 * Get a stable color for a community index.
 * @param {number} communityIndex - 0-based community index from Louvain
 * @returns {string} Hex color
 */
export function communityColorPalette(communityIndex) {
  if (communityIndex == null || communityIndex < 0) return COMMUNITY_COLORS[0];
  return COMMUNITY_COLORS[communityIndex % COMMUNITY_COLORS.length];
}

/**
 * Compute degree (number of incident edges) for each node from the edge list.
 * @param {Array<{ source: string, target: string }>} edges - Array of edges
 * @returns {Record<string, number>} Map of node id -> degree
 */
export function degreeMap(edges) {
  const deg = {};
  (edges || []).forEach((e) => {
    if (e.source != null) deg[e.source] = (deg[e.source] || 0) + 1;
    if (e.target != null) deg[e.target] = (deg[e.target] || 0) + 1;
  });
  return deg;
}

/**
 * Scale edge width from weight for Cytoscape (minWidth to maxWidth).
 * @param {number} weight - Edge weight
 * @param {number} minW - Minimum weight in dataset
 * @param {number} maxW - Maximum weight in dataset
 * @param {number} outMin - Output min width (e.g. 1)
 * @param {number} outMax - Output max width (e.g. 6)
 * @returns {number} Width value
 */
export function edgeWeightScale(weight, minW, maxW, outMin = 1, outMax = 6) {
  if (weight == null || !Number.isFinite(weight)) return outMin;
  const range = maxW - minW;
  if (range <= 0) return outMax;
  const t = (weight - minW) / range;
  return outMin + t * (outMax - outMin);
}
