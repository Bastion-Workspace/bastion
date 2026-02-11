import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
  ReactFlowProvider,
  Handle,
  Position,
} from 'reactflow';
import 'reactflow/dist/style.css';
import * as d3Force from 'd3-force';
import {
  Box,
  Button,
  Dialog,
  IconButton,
  TextField,
  Typography,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip,
  CircularProgress,
  useTheme,
} from '@mui/material';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import FullscreenExitIcon from '@mui/icons-material/FullscreenExit';
import RefreshIcon from '@mui/icons-material/Refresh';
import CloseIcon from '@mui/icons-material/Close';
import SearchIcon from '@mui/icons-material/Search';
import apiService from '../services/apiService';

const NODE_WIDTH = 120;
const NODE_HEIGHT = 40;
const LAYOUT_WIDTH = 800;
const LAYOUT_HEIGHT = 600;

function mergeBidirectionalEdges(edgesData, nodeIds) {
  const directed = new Set();
  for (const e of edgesData || []) {
    if (!nodeIds.has(e.source) || !nodeIds.has(e.target)) continue;
    directed.add(`${e.source}|${e.target}`);
  }
  const seen = new Set();
  const merged = [];
  for (const e of edgesData || []) {
    if (!nodeIds.has(e.source) || !nodeIds.has(e.target)) continue;
    const [a, b] = [e.source, e.target].sort();
    const key = `${a}|${b}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const bidirectional = directed.has(`${a}|${b}`) && directed.has(`${b}|${a}`);
    merged.push({ source: a, target: b, bidirectional });
  }
  return merged;
}

const HANDLE_ANGLES = ['right', 'bottomRight', 'bottom', 'bottomLeft', 'left', 'topLeft', 'top', 'topRight'];
function getHandleForAngle(rad) {
  let deg = (rad * 180) / Math.PI;
  if (deg < 0) deg += 360;
  const index = Math.round(deg / 45) % 8;
  return HANDLE_ANGLES[index];
}

function computeLayout(nodesData, edgesData) {
  if (!nodesData?.length) return { nodes: [], edges: [] };
  const nodeIds = new Set(nodesData.map((n) => n.id));
  const mergedEdges = mergeBidirectionalEdges(edgesData, nodeIds);
  const links = mergedEdges.map((e) => ({ source: e.source, target: e.target }));
  const nodes = nodesData.map((n) => ({
    id: n.id,
    data: { label: n.label, ...n },
    position: { x: 0, y: 0 },
    type: 'fileNode',
  }));
  const linkForce = d3Force.forceLink(links).id((d) => d.id).distance(220);
  const sim = d3Force.forceSimulation(nodes)
    .force('link', linkForce)
    .force('charge', d3Force.forceManyBody().strength(-500))
    .force('center', d3Force.forceCenter(LAYOUT_WIDTH / 2, LAYOUT_HEIGHT / 2))
    .force('collision', d3Force.forceCollide().radius(90));
  for (let i = 0; i < 350; i++) sim.tick();
  const positioned = nodes.map((n) => ({
    ...n,
    position: { x: n.x ?? 0, y: n.y ?? 0 },
  }));
  const posById = Object.fromEntries(positioned.map((n) => [n.id, n.position]));
  const edges = mergedEdges.map((e, i) => {
    const srcPos = posById[e.source] || { x: 0, y: 0 };
    const tgtPos = posById[e.target] || { x: 0, y: 0 };
    const angleToTarget = Math.atan2(tgtPos.y - srcPos.y, tgtPos.x - srcPos.x);
    const angleToSource = Math.atan2(srcPos.y - tgtPos.y, srcPos.x - tgtPos.x);
    const sourceHandle = `source-${getHandleForAngle(angleToTarget)}`;
    const targetHandle = `target-${getHandleForAngle(angleToSource)}`;
    const edgeId = `e-${e.source}-${e.target}-${i}`;
    const base = {
      id: edgeId,
      source: e.source,
      target: e.target,
      sourceHandle,
      targetHandle,
      type: 'smoothstep',
      markerEnd: { type: MarkerType.ArrowClosed },
    };
    if (e.bidirectional) base.markerStart = { type: MarkerType.ArrowClosed };
    return base;
  });
  return { nodes: positioned, edges };
}

const cornerStyle = { width: 8, height: 8 };
function FileNode({ data, selected }) {
  return (
    <>
      <Handle type="target" position={Position.Top} id="target-top" style={cornerStyle} />
      <Handle type="target" position={Position.Top} id="target-topRight" style={{ ...cornerStyle, left: '85%' }} />
      <Handle type="target" position={Position.Right} id="target-right" style={cornerStyle} />
      <Handle type="target" position={Position.Right} id="target-bottomRight" style={{ ...cornerStyle, top: '85%' }} />
      <Handle type="target" position={Position.Bottom} id="target-bottom" style={cornerStyle} />
      <Handle type="target" position={Position.Bottom} id="target-bottomLeft" style={{ ...cornerStyle, left: '15%' }} />
      <Handle type="target" position={Position.Left} id="target-left" style={cornerStyle} />
      <Handle type="target" position={Position.Left} id="target-topLeft" style={{ ...cornerStyle, top: '15%' }} />
      <Handle type="source" position={Position.Top} id="source-top" style={cornerStyle} />
      <Handle type="source" position={Position.Top} id="source-topRight" style={{ ...cornerStyle, left: '85%' }} />
      <Handle type="source" position={Position.Right} id="source-right" style={cornerStyle} />
      <Handle type="source" position={Position.Right} id="source-bottomRight" style={{ ...cornerStyle, top: '85%' }} />
      <Handle type="source" position={Position.Bottom} id="source-bottom" style={cornerStyle} />
      <Handle type="source" position={Position.Bottom} id="source-bottomLeft" style={{ ...cornerStyle, left: '15%' }} />
      <Handle type="source" position={Position.Left} id="source-left" style={cornerStyle} />
      <Handle type="source" position={Position.Left} id="source-topLeft" style={{ ...cornerStyle, top: '15%' }} />
      <Box
        sx={{
          px: 1.5,
          py: 1,
          borderRadius: 1,
          border: '1px solid',
          borderColor: selected ? 'primary.main' : 'divider',
          bgcolor: 'background.paper',
          boxShadow: 1,
          minWidth: NODE_WIDTH,
          minHeight: NODE_HEIGHT,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Typography variant="body2" noWrap sx={{ maxWidth: NODE_WIDTH - 16 }}>
          {data?.label ?? data?.id ?? ''}
        </Typography>
      </Box>
    </>
  );
}

const nodeTypes = { fileNode: FileNode };

function GraphCanvas({ nodes: initialNodes, edges: initialEdges, onNodeClick, searchFilter, theme }) {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const palette = theme?.palette || {};
  const edgeStroke = palette.primary?.main ?? '#5c6bc0';
  const background = palette.background?.default ?? '#fafafa';
  const isDark = palette.mode === 'dark';
  const minimapColors = {
    org: palette.success?.main ?? '#2e7d32',
    markdown: palette.primary?.main ?? '#1565c0',
    other: palette.warning?.main ?? '#ed6c02',
    default: palette.text?.secondary ?? '#757575',
  };

  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  const { visibleNodes, visibleEdges } = useMemo(() => {
    if (!searchFilter?.trim()) return { visibleNodes: nodes, visibleEdges: edges };
    const q = searchFilter.trim().toLowerCase();
    const visibleIds = new Set(
      nodes.filter((n) => (n.data?.label || '').toLowerCase().includes(q)).map((n) => n.id)
    );
    const visibleNodes = nodes.filter((n) => visibleIds.has(n.id));
    const visibleEdges = edges.filter(
      (e) => visibleIds.has(e.source) && visibleIds.has(e.target)
    );
    return { visibleNodes, visibleEdges };
  }, [nodes, edges, searchFilter]);

  return (
    <Box sx={{ width: '100%', height: '100%', minHeight: 400 }}>
      <ReactFlow
        nodes={visibleNodes}
        edges={visibleEdges}
        nodeTypes={nodeTypes}
        defaultEdgeOptions={{ style: { stroke: edgeStroke, strokeWidth: 2 } }}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={(_, node) => {
          const docId = node.id;
          const docName = node.data?.label || docId;
          onNodeClick?.(docId, docName);
        }}
        fitView
        defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
        nodesDraggable
        nodesConnectable={false}
        elementsSelectable
        style={{ background }}
      >
        <Background color={palette.divider} gap={16} />
        <Controls />
        <MiniMap
          style={{ backgroundColor: background }}
          className={isDark ? 'file-graph-minimap file-graph-minimap-dark' : 'file-graph-minimap'}
          nodeColor={(n) => {
            const t = n.data?.type;
            if (t === 'org') return minimapColors.org;
            if (t === 'markdown') return minimapColors.markdown;
            if (t === 'other') return minimapColors.other;
            return minimapColors.default;
          }}
        />
      </ReactFlow>
    </Box>
  );
}

function FileRelationGraphInner({ scope: initialScope, folderId, onOpenDocument }) {
  const theme = useTheme();
  const [scope, setScope] = useState(initialScope || 'all');
  const [folderIdState, setFolderIdState] = useState(folderId || '');
  const [data, setData] = useState({ nodes: [], edges: [], unresolved_targets: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchFilter, setSearchFilter] = useState('');
  const [fullscreen, setFullscreen] = useState(false);
  const [layoutResult, setLayoutResult] = useState({ nodes: [], edges: [] });
  const [layoutPending, setLayoutPending] = useState(false);

  const fetchGraph = useCallback(async () => {
    setLoading(true);
    setError(null);
    const timeoutMs = 20000;
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Request timed out')), timeoutMs)
    );
    try {
      const res = await Promise.race([
        apiService.getLinkGraph(
          scope === 'folder' && folderIdState ? 'folder' : 'all',
          folderIdState || null
        ),
        timeoutPromise,
      ]);
      setData({
        nodes: res.nodes || [],
        edges: res.edges || [],
        unresolved_targets: res.unresolved_targets || [],
      });
    } catch (e) {
      setError(e?.message || 'Failed to load link graph');
      setData({ nodes: [], edges: [], unresolved_targets: [] });
    } finally {
      setLoading(false);
    }
  }, [scope, folderIdState]);

  useEffect(() => {
    fetchGraph();
  }, [fetchGraph]);

  useEffect(() => {
    if (!data.nodes?.length) {
      setLayoutResult({ nodes: [], edges: [] });
      setLayoutPending(false);
      return;
    }
    setLayoutPending(true);
    const id = setTimeout(() => {
      const result = computeLayout(data.nodes, data.edges);
      setLayoutResult(result);
      setLayoutPending(false);
    }, 0);
    return () => clearTimeout(id);
  }, [data.nodes, data.edges]);

  const layoutNodes = layoutResult.nodes;
  const layoutEdges = layoutResult.edges;
  const layoutEdgesWithStyle = useMemo(() => {
    const deg = {};
    for (const e of layoutEdges) {
      deg[e.source] = (deg[e.source] || 0) + 1;
      deg[e.target] = (deg[e.target] || 0) + 1;
    }
    let max = 0;
    let hub = null;
    for (const id of Object.keys(deg)) {
      if (deg[id] > max) {
        max = deg[id];
        hub = id;
      }
    }
    const incomingColor = theme.palette.primary.main;
    const outgoingColor = theme.palette.secondary?.main ?? theme.palette.info?.main ?? '#0288d1';
    const neutralColor = theme.palette.text.secondary;
    return layoutEdges.map((e) => {
      let stroke = neutralColor;
      if (hub) {
        if (e.target === hub) stroke = incomingColor;
        else if (e.source === hub) stroke = outgoingColor;
      }
      return {
        ...e,
        style: { stroke, strokeWidth: 2, strokeOpacity: 1 },
      };
    });
  }, [layoutEdges, theme.palette.primary.main, theme.palette.secondary?.main, theme.palette.info?.main, theme.palette.text.secondary]);

  const handleNodeClick = useCallback(
    (docId, docName) => {
      if (window.tabbedContentManagerRef?.openDocument) {
        window.tabbedContentManagerRef.openDocument(docId, docName);
      } else {
        onOpenDocument?.(docId, docName);
      }
    },
    [onOpenDocument]
  );

  const toolbar = (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        p: 1,
        borderBottom: '1px solid',
        borderColor: 'divider',
        flexWrap: 'wrap',
      }}
    >
      <TextField
        size="small"
        placeholder="Search files..."
        value={searchFilter}
        onChange={(e) => setSearchFilter(e.target.value)}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon fontSize="small" />
            </InputAdornment>
          ),
        }}
        sx={{ minWidth: 180 }}
      />
      <FormControl size="small" sx={{ minWidth: 120 }}>
        <InputLabel>Scope</InputLabel>
        <Select
          value={scope}
          label="Scope"
          onChange={(e) => setScope(e.target.value)}
        >
          <MenuItem value="all">All files</MenuItem>
          <MenuItem value="folder">Folder</MenuItem>
        </Select>
      </FormControl>
      {scope === 'folder' && (
        <TextField
          size="small"
          placeholder="Folder ID"
          value={folderIdState}
          onChange={(e) => setFolderIdState(e.target.value)}
          sx={{ minWidth: 200 }}
        />
      )}
      <Tooltip title="Refresh">
        <IconButton onClick={fetchGraph} disabled={loading}>
          <RefreshIcon />
        </IconButton>
      </Tooltip>
      <Tooltip title={fullscreen ? 'Exit fullscreen' : 'Fullscreen'}>
        <IconButton onClick={() => setFullscreen(!fullscreen)}>
          {fullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
        </IconButton>
      </Tooltip>
      <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
        {data.nodes.length} nodes, {layoutResult.edges.length} connections
        {data.unresolved_targets?.length > 0 &&
          `, ${data.unresolved_targets.length} unresolved`}
      </Typography>
      <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
        Edges: blue = into most-linked file, pink = out from it, gray = other
      </Typography>
    </Box>
  );

  const graphArea = (
    <Box sx={{ flex: 1, minHeight: 0, position: 'relative' }}>
      {loading && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'action.hover',
            zIndex: 10,
          }}
        >
          <CircularProgress />
        </Box>
      )}
      {error && (
        <Box sx={{ p: 2, color: 'error.main' }}>
          <Typography>{error}</Typography>
          <Button onClick={fetchGraph} size="small" sx={{ mt: 1 }}>
            Retry
          </Button>
        </Box>
      )}
      {!loading && !error && layoutPending && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'action.hover',
            zIndex: 5,
          }}
        >
          <Typography color="text.secondary" sx={{ mr: 1 }}>Computing layout…</Typography>
          <CircularProgress size={24} />
        </Box>
      )}
      {!loading && !error && !layoutPending && layoutNodes.length === 0 && (
        <Box sx={{ p: 3, textAlign: 'center', color: 'text.secondary' }}>
          <Typography>No documents with links yet. Edit .org or .md files and add links.</Typography>
        </Box>
      )}
      {!loading && !error && !layoutPending && layoutNodes.length > 0 && (
        <GraphCanvas
          nodes={layoutNodes}
          edges={layoutEdgesWithStyle}
          onNodeClick={handleNodeClick}
          searchFilter={searchFilter}
          theme={theme}
        />
      )}
    </Box>
  );

  const content = (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 400 }}>
      {toolbar}
      {graphArea}
    </Box>
  );

  if (fullscreen) {
    return (
      <Dialog fullScreen open onClose={() => setFullscreen(false)}>
        <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              p: 1,
              borderBottom: '1px solid',
              borderColor: 'divider',
            }}
          >
            <Typography variant="h6" sx={{ ml: 1 }}>
              File relation cloud
            </Typography>
            <IconButton onClick={() => setFullscreen(false)}>
              <CloseIcon />
            </IconButton>
          </Box>
          <Box sx={{ flex: 1, minHeight: 0 }}>{content}</Box>
        </Box>
      </Dialog>
    );
  }

  return content;
}

export default function FileRelationGraph(props) {
  return (
    <ReactFlowProvider>
      <FileRelationGraphInner {...props} />
    </ReactFlowProvider>
  );
}
