import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  MiniMap,
  useNodesState,
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

function getConnectedComponents(mergedEdges, nodeIds) {
  const adj = new Map();
  for (const id of nodeIds) adj.set(id, new Set());
  for (const e of mergedEdges) {
    if (!nodeIds.has(e.source) || !nodeIds.has(e.target)) continue;
    adj.get(e.source).add(e.target);
    adj.get(e.target).add(e.source);
  }
  const visited = new Set();
  const components = [];
  for (const id of nodeIds) {
    if (visited.has(id)) continue;
    const stack = [id];
    const compIds = new Set();
    while (stack.length) {
      const u = stack.pop();
      if (visited.has(u)) continue;
      visited.add(u);
      compIds.add(u);
      for (const v of adj.get(u) || []) if (!visited.has(v)) stack.push(v);
    }
    if (compIds.size) components.push(compIds);
  }
  return components;
}

function bbox(nodes) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const n of nodes) {
    const x = n.x ?? n.position?.x ?? 0;
    const y = n.y ?? n.position?.y ?? 0;
    minX = Math.min(minX, x); minY = Math.min(minY, y);
    maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
  }
  const pad = 50;
  return {
    minX: minX - pad,
    minY: minY - pad,
    maxX: maxX + pad,
    maxY: maxY + pad,
    width: Math.max(0, maxX - minX + 2 * pad),
    height: Math.max(0, maxY - minY + 2 * pad),
  };
}

function computeLayout(nodesData, edgesData) {
  if (!nodesData?.length) return { nodes: [], edges: [] };
  const nodeIds = new Set(nodesData.map((n) => n.id));
  const mergedEdges = mergeBidirectionalEdges(edgesData, nodeIds);
  const connectedIds = new Set();
  for (const e of mergedEdges) {
    connectedIds.add(e.source);
    connectedIds.add(e.target);
  }
  const connectedNodesData = nodesData.filter((n) => connectedIds.has(n.id));
  const unconnectedNodesData = nodesData.filter((n) => !connectedIds.has(n.id));

  const idToNode = new Map(connectedNodesData.map((n) => [n.id, n]));
  const components = getConnectedComponents(mergedEdges, connectedIds);
  const PAD = 120;
  const positionedConnected = [];
  const componentBoxes = [];

  for (let c = 0; c < components.length; c++) {
    const compIds = components[c];
    const compNodesData = [...compIds].map((id) => idToNode.get(id)).filter(Boolean);
    const compEdges = mergedEdges.filter((e) => compIds.has(e.source) && compIds.has(e.target));
    const links = compEdges.map((e) => ({ source: e.source, target: e.target }));
    const nodes = compNodesData.map((n) => ({
      id: n.id,
      data: { label: n.label, ...n },
      position: { x: 0, y: 0 },
      type: 'fileNode',
    }));
    if (nodes.length > 0 && links.length > 0) {
      const linkForce = d3Force.forceLink(links).id((d) => d.id).distance(220);
      const sim = d3Force.forceSimulation(nodes)
        .force('link', linkForce)
        .force('charge', d3Force.forceManyBody().strength(-400))
        .force('center', d3Force.forceCenter(0, 0))
        .force('collision', d3Force.forceCollide().radius(90));
      for (let i = 0; i < 350; i++) sim.tick();
    }
    const withPos = nodes.map((n) => ({
      ...n,
      position: { x: n.x ?? 0, y: n.y ?? 0 },
    }));
    const box = bbox(withPos);
    componentBoxes.push({ box, nodes: withPos });
  }

  const maxW = Math.max(200, ...componentBoxes.map(({ box }) => box.width));
  const maxH = Math.max(120, ...componentBoxes.map(({ box }) => box.height));
  const cols = Math.max(1, Math.ceil(Math.sqrt(componentBoxes.length)));
  const clusterBounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
  for (let i = 0; i < componentBoxes.length; i++) {
    const { box, nodes } = componentBoxes[i];
    const col = i % cols;
    const row = Math.floor(i / cols);
    const dx = col * (maxW + PAD) - box.minX;
    const dy = row * (maxH + PAD) - box.minY;
    for (const n of nodes) {
      const pos = { x: (n.position?.x ?? n.x ?? 0) + dx, y: (n.position?.y ?? n.y ?? 0) + dy };
      positionedConnected.push({ ...n, position: pos });
      clusterBounds.minX = Math.min(clusterBounds.minX, pos.x);
      clusterBounds.minY = Math.min(clusterBounds.minY, pos.y);
      clusterBounds.maxX = Math.max(clusterBounds.maxX, pos.x);
      clusterBounds.maxY = Math.max(clusterBounds.maxY, pos.y);
    }
  }
  if (positionedConnected.length > 0 && Number.isFinite(clusterBounds.minX)) {
    const cx = (clusterBounds.minX + clusterBounds.maxX) / 2;
    const cy = (clusterBounds.minY + clusterBounds.maxY) / 2;
    const tx = LAYOUT_WIDTH / 2 - cx;
    const ty = LAYOUT_HEIGHT / 2 - cy;
    for (const n of positionedConnected) {
      n.position.x += tx;
      n.position.y += ty;
    }
    clusterBounds.minX += tx;
    clusterBounds.maxX += tx;
    clusterBounds.minY += ty;
    clusterBounds.maxY += ty;
  }
  const hasCluster = positionedConnected.length > 0 && Number.isFinite(clusterBounds.minX);
  const orphanMargin = 160;
  const bandWidth = 220;
  const clusterMinX = hasCluster ? clusterBounds.minX - orphanMargin : 0;
  const clusterMaxX = hasCluster ? clusterBounds.maxX + orphanMargin : LAYOUT_WIDTH;
  const clusterMinY = hasCluster ? clusterBounds.minY - orphanMargin : 0;
  const clusterMaxY = hasCluster ? clusterBounds.maxY + orphanMargin : LAYOUT_HEIGHT;
  const defaultBand = { minX: -200, maxX: LAYOUT_WIDTH + 400, minY: -150, maxY: LAYOUT_HEIGHT + 350 };
  const clusterBands = hasCluster
    ? [
        { minX: -800, maxX: clusterMinX - bandWidth, minY: clusterMinY - 400, maxY: clusterMaxY + 400 },
        { minX: clusterMaxX + bandWidth, maxX: LAYOUT_WIDTH + 600, minY: clusterMinY - 400, maxY: clusterMaxY + 400 },
        { minX: clusterMinX - 400, maxX: clusterMaxX + 400, minY: -600, maxY: clusterMinY - bandWidth },
        { minX: clusterMinX - 400, maxX: clusterMaxX + 400, minY: clusterMaxY + bandWidth, maxY: LAYOUT_HEIGHT + 500 },
      ].filter((b) => b.minX < b.maxX && b.minY < b.maxY)
    : [];
  const bands = clusterBands.length > 0 ? clusterBands : [defaultBand];

  const positionedUnconnected = unconnectedNodesData.map((n, i) => {
    const band = bands[i % bands.length];
    const x = band.minX + Math.random() * (band.maxX - band.minX);
    const y = band.minY + Math.random() * (band.maxY - band.minY);
    return {
      id: n.id,
      data: { label: n.label, ...n },
      position: { x, y },
      type: 'fileNode',
    };
  });

  const positioned = [...positionedConnected, ...positionedUnconnected];
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

function computeEdgeHandles(nodes, baseEdges) {
  const posById = Object.fromEntries(
    nodes.map((n) => [n.id, n.position ?? { x: 0, y: 0 }])
  );
  return baseEdges.map((e) => {
    const srcPos = posById[e.source] || { x: 0, y: 0 };
    const tgtPos = posById[e.target] || { x: 0, y: 0 };
    const angleToTarget = Math.atan2(tgtPos.y - srcPos.y, tgtPos.x - srcPos.x);
    const angleToSource = Math.atan2(srcPos.y - tgtPos.y, srcPos.x - tgtPos.x);
    return {
      ...e,
      sourceHandle: `source-${getHandleForAngle(angleToTarget)}`,
      targetHandle: `target-${getHandleForAngle(angleToSource)}`,
    };
  });
}

function GraphCanvas({ nodes: initialNodes, edges: initialEdges, onNodeClick, searchFilter, theme, initialViewport, onViewportChange }) {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
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
  }, [initialNodes, setNodes]);

  const baseEdges = useMemo(
    () =>
      (initialEdges || []).map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        style: e.style,
        markerEnd: e.markerEnd,
        markerStart: e.markerStart,
      })),
    [initialEdges]
  );

  const computedEdges = useMemo(
    () => computeEdgeHandles(nodes, baseEdges),
    [nodes, baseEdges]
  );

  const { visibleNodes, visibleEdges } = useMemo(() => {
    if (!searchFilter?.trim()) return { visibleNodes: nodes, visibleEdges: computedEdges };
    const q = searchFilter.trim().toLowerCase();
    const matchingIds = new Set(
      nodes.filter((n) => (n.data?.label || '').toLowerCase().includes(q)).map((n) => n.id)
    );
    const visibleIds = new Set(matchingIds);
    for (const e of computedEdges) {
      if (matchingIds.has(e.source) || matchingIds.has(e.target)) {
        visibleIds.add(e.source);
        visibleIds.add(e.target);
      }
    }
    const visibleNodes = nodes.filter((n) => visibleIds.has(n.id));
    const visibleEdges = computedEdges.filter(
      (e) => visibleIds.has(e.source) && visibleIds.has(e.target)
    );
    return { visibleNodes, visibleEdges };
  }, [nodes, computedEdges, searchFilter]);

  const defaultViewport = initialViewport && typeof initialViewport.zoom === 'number'
    ? { x: initialViewport.x ?? 0, y: initialViewport.y ?? 0, zoom: initialViewport.zoom }
    : { x: 0, y: 0, zoom: 0.8 };
  const hasSavedViewport = initialViewport && typeof initialViewport.zoom === 'number';

  return (
    <Box sx={{ width: '100%', height: '100%', minHeight: 400 }}>
      <ReactFlow
        nodes={visibleNodes}
        edges={visibleEdges}
        nodeTypes={nodeTypes}
        defaultEdgeOptions={{ style: { stroke: edgeStroke, strokeWidth: 2 } }}
        onNodesChange={onNodesChange}
        onEdgesChange={() => {}}
        onNodeClick={(_, node) => {
          const docId = node.id;
          const docName = node.data?.label || docId;
          onNodeClick?.(docId, docName);
        }}
        fitView={!hasSavedViewport}
        defaultViewport={defaultViewport}
        onMoveEnd={(_, viewport) => onViewportChange?.(viewport)}
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

function FileRelationGraphInner({ scope: initialScope, folderId, onOpenDocument, persistedState, onStateChange }) {
  const theme = useTheme();
  const [scope, setScope] = useState(persistedState?.scope ?? initialScope ?? 'all');
  const [folderIdState, setFolderIdState] = useState(persistedState?.folderId ?? folderId ?? '');
  const [data, setData] = useState({ nodes: [], edges: [], unresolved_targets: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchFilter, setSearchFilter] = useState(persistedState?.searchFilter ?? '');
  const [fullscreen, setFullscreen] = useState(false);
  const [layoutResult, setLayoutResult] = useState({ nodes: [], edges: [] });
  const [layoutPending, setLayoutPending] = useState(false);
  const lastViewportRef = React.useRef(persistedState?.viewport ?? null);

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

  const onStateChangeRef = React.useRef(onStateChange);
  onStateChangeRef.current = onStateChange;
  useEffect(() => {
    const fn = onStateChangeRef.current;
    if (typeof fn !== 'function') return;
    fn({
      scope,
      folderId: folderIdState,
      searchFilter,
      viewport: lastViewportRef.current ?? undefined,
    });
  }, [scope, folderIdState, searchFilter]);

  const handleViewportChange = useCallback(
    (viewport) => {
      lastViewportRef.current = viewport;
      onStateChange?.({ viewport });
    },
    [onStateChange]
  );

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
          initialViewport={persistedState?.viewport}
          onViewportChange={handleViewportChange}
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
