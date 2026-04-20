/**
 * Org chart as interactive Cytoscape graph (fcose layout). Nodes show agent name and role; click for details.
 */

import React, { useMemo, useState, useRef } from 'react';
import { Box, Typography, Popover } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import GraphCanvas from '../graph/GraphCanvas';

function addEdges(nodes, edges, parentId = null) {
  if (!nodes || !nodes.length) return;
  for (const n of nodes) {
    const id = n.id || n.membership_id || n.agent_profile_id;
    if (!id) continue;
    if (parentId) edges.push({ source: parentId, target: id });
    addEdges(n.children || [], edges, id);
  }
}

function normalizeId(value) {
  if (value == null || value === '') return null;
  const s = String(value).trim();
  return s || null;
}

function treeToNodesAndEdges(orgChart, activeAgentProfileId, highlightProfileIds = []) {
  const roots = Array.isArray(orgChart) ? orgChart : [];
  const nodes = [];
  const edges = [];
  const activeId = normalizeId(activeAgentProfileId);
  const highlightSet = new Set((highlightProfileIds || []).map((x) => normalizeId(x)).filter(Boolean));

  const collect = (nodeList) => {
    for (const n of nodeList || []) {
      const id = n.id || n.membership_id || n.agent_profile_id;
      if (!id) continue;
      const profileId = normalizeId(n.agent_profile_id);
      const isActive =
        activeId != null && profileId != null && activeId === profileId;
      const isHighlight = profileId != null && highlightSet.has(profileId);

      nodes.push({
        id,
        label: n.agent_name || n.agent_handle || 'Agent',
        role: n.role || 'worker',
        agent_name: n.agent_name,
        agent_handle: n.agent_handle,
        agent_profile_id: n.agent_profile_id,
        color: n.color || '#1976d2',
        labelLength: Math.min(30, (n.agent_name || n.agent_handle || 'Agent').length),
        isActive,
        isHighlight,
      });
      collect(n.children || []);
    }
  };
  collect(roots);
  addEdges(roots, edges);
  return { nodes, edges };
}

/** Flat peer list + no edges — circle layout for committee / round-robin / consensus. */
function peerNodesAndEdges(orgChart, activeAgentProfileId, highlightProfileIds = []) {
  const { nodes: treeNodes } = treeToNodesAndEdges(orgChart, activeAgentProfileId, highlightProfileIds);
  return { nodes: treeNodes, edges: [] };
}

const getOrgChartStyle = (theme) => [
  {
    selector: 'node',
    style: {
      label: 'data(label)',
      'text-valign': 'center',
      'text-halign': 'center',
      'font-size': 10,
      'text-max-width': 90,
      'text-wrap': 'ellipsis',
      'background-color': 'data(color)',
      'border-width': 1,
      'border-color': 'rgba(0,0,0,0.2)',
      'border-style': 'solid',
      width: 80,
      height: 36,
      shape: 'round-rectangle',
      color: '#fff',
      'text-margin-y': 0,
    },
  },
  {
    selector: 'edge',
    style: {
      'curve-style': 'bezier',
      'target-arrow-shape': 'triangle',
      'target-arrow-color': '#666',
      width: 1.5,
      'line-color': '#999',
    },
  },
  {
    selector: 'node.hovered',
    style: { 'border-width': 2, 'border-color': '#fff' },
  },
  {
    selector: 'node[?isActive]',
    style: {
      'border-width': 3,
      'border-color': theme.palette.primary.light,
      'overlay-color': theme.palette.primary.main,
      'overlay-opacity': 0.45,
      'overlay-padding': 6,
      'overlay-shape': 'round-rectangle',
    },
  },
  {
    selector: 'node[?isHighlight]',
    style: {
      'border-width': 3,
      'border-color': theme.palette.warning.light,
      'border-style': 'solid',
    },
  },
];

/**
 * @param {'tree'|'circle'} layoutMode — tree uses reporting lines; circle lays out peers in a ring.
 * @param {string[]} highlightAgentProfileIds — optional chair / current rotation leader emphasis.
 */
export default function OrgChartView({
  orgChart,
  activeAgentProfileId,
  layoutMode = 'tree',
  highlightAgentProfileIds = [],
}) {
  const theme = useTheme();
  const [anchorEl, setAnchorEl] = useState(null);
  const [popoverNode, setPopoverNode] = useState(null);

  const { nodes, edges } = useMemo(() => {
    if (layoutMode === 'circle') {
      return peerNodesAndEdges(orgChart, activeAgentProfileId, highlightAgentProfileIds);
    }
    return treeToNodesAndEdges(orgChart, activeAgentProfileId, highlightAgentProfileIds);
  }, [orgChart, activeAgentProfileId, layoutMode, highlightAgentProfileIds]);

  const layoutOptions = useMemo(
    () => (layoutMode === 'circle' ? { name: 'circle', padding: 32 } : {}),
    [layoutMode]
  );

  const chartStyle = useMemo(() => getOrgChartStyle(theme), [theme]);

  const containerRef = useRef(null);

  const handleNodeClick = (data) => {
    setPopoverNode(data);
    setAnchorEl(containerRef.current || document.body);
  };

  const handleClosePopover = () => {
    setAnchorEl(null);
    setPopoverNode(null);
  };

  if (nodes.length === 0) {
    return (
      <Box sx={{ py: 2, textAlign: 'center' }}>
        <Typography color="text.secondary">
          No members in org chart. Add agents and set reporting lines in Settings.
        </Typography>
      </Box>
    );
  }

  return (
    <Box ref={containerRef} sx={{ height: 320, position: 'relative' }}>
      <GraphCanvas
        nodes={nodes}
        edges={edges}
        style={chartStyle}
        layoutOptions={layoutOptions}
        onNodeClick={handleNodeClick}
        theme={theme}
        sx={{ minHeight: 320, bgcolor: theme.palette.mode === 'dark' ? 'background.default' : '#f5f5f5' }}
      />
      <Popover
        open={Boolean(popoverNode)}
        anchorEl={anchorEl}
        onClose={handleClosePopover}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        transformOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Box sx={{ p: 2, maxWidth: 280 }}>
          {popoverNode && (
            <>
              <Typography variant="subtitle2">{popoverNode.label}</Typography>
              {popoverNode.agent_handle && (
                <Typography variant="caption" color="text.secondary">@{popoverNode.agent_handle}</Typography>
              )}
              <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                Role: {popoverNode.role || 'worker'}
              </Typography>
            </>
          )}
        </Box>
      </Popover>
    </Box>
  );
}
