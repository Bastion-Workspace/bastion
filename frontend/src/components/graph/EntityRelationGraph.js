/**
 * Entity (knowledge) graph using Cytoscape. Community + entity-type coloring, focus-dim, EntityDetailPanel.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Box,
  Button,
  Chip,
  Dialog,
  FormControl,
  IconButton,
  InputAdornment,
  InputLabel,
  MenuItem,
  Select,
  TextField,
  Typography,
  Tooltip,
  CircularProgress,
  useTheme,
} from '@mui/material';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import FullscreenExitIcon from '@mui/icons-material/FullscreenExit';
import RefreshIcon from '@mui/icons-material/Refresh';
import CloseIcon from '@mui/icons-material/Close';
import SearchIcon from '@mui/icons-material/Search';
import apiService from '../../services/apiService';
import { useChatSidebar } from '../../contexts/ChatSidebarContext';
import { computeCommunities, communityColorPalette } from './graphUtils';
import GraphCanvas, { FCOSE_OPTIONS_LOOSE } from './GraphCanvas';
import EntityDetailPanel from './EntityDetailPanel';

function entityTypeColor(entityType, palette) {
  const t = (entityType || 'MISC').toUpperCase();
  if (t === 'PERSON') return palette?.primary?.main ?? '#1565c0';
  if (t === 'ORG') return palette?.secondary?.main ?? '#7b1fa2';
  if (t === 'LOC' || t === 'LOCATION') return palette?.success?.main ?? '#2e7d32';
  return palette?.warning?.main ?? '#ed6c02';
}

export default function EntityRelationGraph({ onOpenDocument, persistedState, onStateChange }) {
  const theme = useTheme();
  const { setQuery, setIsCollapsed } = useChatSidebar();
  const [data, setData] = useState({ nodes: [], edges: [], entity_count: 0, document_count: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchFilter, setSearchFilter] = useState(persistedState?.searchFilter ?? '');
  const [entityTypeFilter, setEntityTypeFilter] = useState(persistedState?.entityTypeFilter ?? '');
  const [entityLimit, setEntityLimit] = useState(persistedState?.entityLimit ?? 100);
  const [fullscreen, setFullscreen] = useState(false);
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [entityDetail, setEntityDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const lastViewportRef = React.useRef(persistedState?.viewport ?? null);

  const fetchGraph = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.getEntityGraph(entityLimit);
      setData({
        nodes: res.nodes || [],
        edges: res.edges || [],
        entity_count: res.entity_count ?? 0,
        document_count: res.document_count ?? 0,
      });
    } catch (e) {
      setError(e?.message || 'Failed to load entity graph');
      setData({ nodes: [], edges: [], entity_count: 0, document_count: 0 });
    } finally {
      setLoading(false);
    }
  }, [entityLimit]);

  useEffect(() => {
    fetchGraph();
  }, [fetchGraph]);

  useEffect(() => {
    if (typeof onStateChange === 'function') {
      onStateChange({
        searchFilter,
        entityTypeFilter,
        entityLimit,
        viewport: lastViewportRef.current ?? undefined,
      });
    }
  }, [searchFilter, entityTypeFilter, entityLimit, onStateChange]);

  const handleViewportChange = useCallback(
    (viewport) => {
      lastViewportRef.current = viewport;
      onStateChange?.({ viewport });
    },
    [onStateChange]
  );

  const communities = useMemo(
    () => computeCommunities(data.nodes, data.edges),
    [data.nodes, data.edges]
  );

  const nodesWithColors = useMemo(() => {
    const palette = theme.palette;
    return (data.nodes || []).map((n) => {
      const color = communityColorPalette(communities[n.id]);
      const typeColor = n.type === 'entity' ? entityTypeColor(n.entity_type, palette) : undefined;
      return {
        ...n,
        color: n.type === 'entity' ? color : (palette?.background?.paper ?? '#fff'),
        typeColor: typeColor || palette?.divider,
      };
    });
  }, [data.nodes, communities, theme.palette]);

  const entityTypes = useMemo(() => {
    const types = new Set();
    nodesWithColors.forEach((n) => {
      if (n.type === 'entity' && n.entity_type) types.add((n.entity_type || '').toUpperCase());
    });
    return ['', ...[...types].sort()];
  }, [nodesWithColors]);

  const passTypeFilter = useCallback(
    (n) => {
      if (!entityTypeFilter) return true;
      if (n.type !== 'entity') return true;
      return (n.entity_type || '').toUpperCase() === entityTypeFilter.toUpperCase();
    },
    [entityTypeFilter]
  );

  const filteredByType = useMemo(
    () => nodesWithColors.filter(passTypeFilter),
    [nodesWithColors, passTypeFilter]
  );

  const filteredNodes = useMemo(() => {
    if (!searchFilter?.trim()) return filteredByType;
    const q = searchFilter.trim().toLowerCase();
    const matchingIds = new Set(
      filteredByType.filter((n) => (n.label || '').toLowerCase().includes(q)).map((n) => n.id)
    );
    const visibleIds = new Set(matchingIds);
    for (const e of data.edges || []) {
      if (matchingIds.has(e.source) || matchingIds.has(e.target)) {
        visibleIds.add(e.source);
        visibleIds.add(e.target);
      }
    }
    return filteredByType.filter((n) => visibleIds.has(n.id));
  }, [filteredByType, data.edges, searchFilter]);

  const filteredEdges = useMemo(() => {
    const visibleIds = new Set(filteredNodes.map((n) => n.id));
    return (data.edges || []).filter(
      (e) => visibleIds.has(e.source) && visibleIds.has(e.target)
    );
  }, [data.edges, filteredNodes]);

  const focusedNodeIds = useMemo(() => {
    if (!selectedEntity?.id) return null;
    const ids = new Set([selectedEntity.id]);
    for (const e of filteredEdges) {
      if (e.source === selectedEntity.id) ids.add(e.target);
      if (e.target === selectedEntity.id) ids.add(e.source);
    }
    return ids;
  }, [selectedEntity?.id, filteredEdges]);

  const handleNodeClick = useCallback(
    (nodeData) => {
      if (!nodeData) return;
      const isDocument = nodeData.type === 'document' || nodeData.node_type === 'document' || (typeof nodeData.id === 'string' && nodeData.id.startsWith('doc:'));
      const isEntity = nodeData.type === 'entity' || nodeData.node_type === 'entity' || (typeof nodeData.id === 'string' && nodeData.id.startsWith('entity:'));
      if (isDocument) {
        const docId = (nodeData.id || '').startsWith('doc:') ? nodeData.id.slice(4) : nodeData.id;
        const docName = nodeData.label ?? docId;
        if (docId && window.tabbedContentManagerRef?.openDocument) {
          window.tabbedContentManagerRef.openDocument(docId, docName);
        } else {
          onOpenDocument?.(docId, docName);
        }
        return;
      }
      if (isEntity) {
        const name = nodeData.label ?? nodeData.name ?? (typeof nodeData.id === 'string' && nodeData.id.startsWith('entity:') ? nodeData.id.slice(7) : '') ?? '';
        const id = nodeData.id ?? `entity:${name}`;
        setSelectedEntity({
          id,
          name,
          entity_type: nodeData.entity_type,
          doc_count: nodeData.doc_count,
        });
        setEntityDetail(null);
        setDetailLoading(true);
        apiService
          .getEntityDetail(name)
          .then((res) => setEntityDetail(res))
          .catch(() => setEntityDetail({ name, document_mentions: [], co_occurring_entities: [] }))
          .finally(() => setDetailLoading(false));
      }
    },
    [onOpenDocument]
  );

  const handleNodeDoubleClick = useCallback(
    (nodeData) => {
      if (nodeData?.type === 'document') {
        const docId = (nodeData.id || '').startsWith('doc:') ? nodeData.id.slice(4) : nodeData.id;
        const docName = nodeData.label ?? docId;
        if (docId && window.tabbedContentManagerRef?.openDocument) {
          window.tabbedContentManagerRef.openDocument(docId, docName);
        } else {
          onOpenDocument?.(docId, docName);
        }
      }
    },
    [onOpenDocument]
  );

  const handleClearSelection = useCallback(() => {
    setSelectedEntity(null);
    setEntityDetail(null);
  }, []);

  const handleAskAI = useCallback(
    (entityName) => {
      setQuery(`What do my documents say about ${entityName}?`);
      setIsCollapsed(false);
    },
    [setQuery, setIsCollapsed]
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
        placeholder="Search entities or documents..."
        value={searchFilter}
        onChange={(e) => setSearchFilter(e.target.value)}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon fontSize="small" />
            </InputAdornment>
          ),
        }}
        sx={{ minWidth: 220 }}
      />
      <FormControl size="small" sx={{ minWidth: 140 }}>
        <InputLabel id="entity-type-filter-label">Entity type</InputLabel>
        <Select
          labelId="entity-type-filter-label"
          value={entityTypeFilter}
          label="Entity type"
          onChange={(e) => setEntityTypeFilter(e.target.value)}
        >
          <MenuItem value="">All types</MenuItem>
          {entityTypes.filter((t) => t).map((t) => (
            <MenuItem key={t} value={t}>
              {t}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      <FormControl size="small" sx={{ minWidth: 120 }}>
        <InputLabel id="entity-limit-label">Show</InputLabel>
        <Select
          labelId="entity-limit-label"
          value={entityLimit}
          label="Show"
          onChange={(e) => setEntityLimit(Number(e.target.value))}
        >
          <MenuItem value={100}>100 entities</MenuItem>
          <MenuItem value={200}>200 entities</MenuItem>
          <MenuItem value={500}>500 entities</MenuItem>
          <MenuItem value={1000}>1000 entities</MenuItem>
        </Select>
      </FormControl>
      <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
        {data.entity_count} entities, {data.document_count} documents
      </Typography>
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
    <Box sx={{ flex: 1, minHeight: 0, position: 'relative', display: 'flex', flexDirection: 'column' }}>
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
      {!loading && !error && filteredNodes.length === 0 && (
        <Box sx={{ p: 3, textAlign: 'center', color: 'text.secondary' }}>
          <Typography>No entity graph data yet. Add documents and extract entities to see the knowledge map.</Typography>
        </Box>
      )}
      {!loading && !error && filteredNodes.length > 0 && (
        <Box sx={{ display: 'flex', flex: 1, minHeight: 0, gap: 0 }}>
          <Box sx={{ flex: 1, minHeight: 0 }}>
            <GraphCanvas
              nodes={filteredNodes}
              edges={filteredEdges}
              layoutOptions={FCOSE_OPTIONS_LOOSE}
              onNodeClick={handleNodeClick}
              onNodeDoubleClick={handleNodeDoubleClick}
              onPaneClick={handleClearSelection}
              focusedNodeIds={focusedNodeIds}
              theme={theme}
              initialViewport={persistedState?.viewport}
              onViewportChange={handleViewportChange}
            />
          </Box>
          <EntityDetailPanel
            selectedEntity={selectedEntity}
            entityDetail={entityDetail}
            detailLoading={detailLoading}
            onClose={handleClearSelection}
            onOpenDocument={(docId, docName) => {
              if (window.tabbedContentManagerRef?.openDocument) {
                window.tabbedContentManagerRef.openDocument(docId, docName);
              } else {
                onOpenDocument?.(docId, docName);
              }
            }}
            onAskAI={handleAskAI}
          />
        </Box>
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
              Entity graph (knowledge map)
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
