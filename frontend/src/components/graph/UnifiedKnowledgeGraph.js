/**
 * Unified knowledge graph: file links + entity graph in one canvas. Layer toggles for Files / Entities.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Box,
  Button,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
  CircularProgress,
  Tooltip,
  IconButton,
  useTheme,
} from '@mui/material';
import InsertDriveFileOutlinedIcon from '@mui/icons-material/InsertDriveFileOutlined';
import AccountTreeOutlinedIcon from '@mui/icons-material/AccountTreeOutlined';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import apiService from '../../services/apiService';
import { computeCommunities, communityColorPalette } from './graphUtils';
import GraphCanvas from './GraphCanvas';

function entityTypeColor(entityType, palette) {
  const t = (entityType || 'MISC').toUpperCase();
  if (t === 'PERSON') return palette?.primary?.main ?? '#1565c0';
  if (t === 'ORG') return palette?.secondary?.main ?? '#7b1fa2';
  if (t === 'LOC' || t === 'LOCATION') return palette?.success?.main ?? '#2e7d32';
  return palette?.warning?.main ?? '#ed6c02';
}

export default function UnifiedKnowledgeGraph({ onOpenDocument, persistedState, onStateChange }) {
  const theme = useTheme();
  const [data, setData] = useState({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showFiles, setShowFiles] = useState(true);
  const [showEntities, setShowEntities] = useState(true);
  const [entityLimit, setEntityLimit] = useState(100);

  const fetchGraph = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.getUnifiedGraph('files,entities', entityLimit);
      setData({ nodes: res.nodes || [], edges: res.edges || [] });
    } catch (e) {
      setError(e?.message || 'Failed to load unified graph');
      setData({ nodes: [], edges: [] });
    } finally {
      setLoading(false);
    }
  }, [entityLimit]);

  useEffect(() => {
    fetchGraph();
  }, [fetchGraph]);

  const communities = useMemo(
    () => computeCommunities(data.nodes, data.edges),
    [data.nodes, data.edges]
  );

  const palette = theme.palette;
  const nodesWithColors = useMemo(() => {
    return (data.nodes || []).map((n) => {
      const nodeType = n.node_type || n.type;
      const color = communityColorPalette(communities[n.id]);
      const typeColor = nodeType === 'entity' ? entityTypeColor(n.entity_type, palette) : undefined;
      return {
        ...n,
        type: nodeType === 'entity' ? 'entity' : 'file',
        color: nodeType === 'entity' ? color : (color || (palette?.background?.paper ?? '#fff')),
        typeColor: typeColor || palette?.divider,
      };
    });
  }, [data.nodes, communities, palette]);

  const visibleNodes = useMemo(() => {
    return nodesWithColors.filter((n) => {
      const nodeType = n.node_type || n.type;
      if (nodeType === 'file') return showFiles;
      if (nodeType === 'entity') return showEntities;
      return true;
    });
  }, [nodesWithColors, showFiles, showEntities]);

  const visibleNodeIds = useMemo(() => new Set(visibleNodes.map((n) => n.id)), [visibleNodes]);

  const visibleEdges = useMemo(() => {
    return (data.edges || []).filter((e) => {
      if (!visibleNodeIds.has(e.source) || !visibleNodeIds.has(e.target)) return false;
      const edgeType = e.edge_type || '';
      if (edgeType === 'file_link') return showFiles;
      if (edgeType === 'mentions' || edgeType === 'co_occurs') return showEntities;
      return true;
    });
  }, [data.edges, visibleNodeIds, showFiles, showEntities]);

  const handleLayerChange = useCallback((_, newLayers) => {
    if (newLayers === null) return;
    setShowFiles(newLayers.includes('files'));
    setShowEntities(newLayers.includes('entities'));
  }, []);

  const handleNodeClick = useCallback(
    (nodeData) => {
      const nodeType = nodeData?.node_type || nodeData?.type;
      if (nodeType === 'file') {
        const docId = nodeData.id;
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

  const layers = useMemo(() => {
    const arr = [];
    if (showFiles) arr.push('files');
    if (showEntities) arr.push('entities');
    return arr;
  }, [showFiles, showEntities]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 400 }}>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          p: 1,
          borderBottom: '1px solid',
          borderColor: 'divider',
        }}
      >
        <ToggleButtonGroup value={layers} onChange={handleLayerChange} size="small">
          <ToggleButton value="files" aria-label="Files layer">
            <InsertDriveFileOutlinedIcon sx={{ mr: 0.5 }} />
            Files
          </ToggleButton>
          <ToggleButton value="entities" aria-label="Entities layer">
            <AccountTreeOutlinedIcon sx={{ mr: 0.5 }} />
            Entities
          </ToggleButton>
        </ToggleButtonGroup>
        <Tooltip
          title="One graph combining document links (Files) and extracted people, orgs, places (Entities). Toggle layers to show files, entities, or both."
          placement="bottom"
        >
          <IconButton size="small" aria-label="What is Unified Knowledge Graph?">
            <InfoOutlinedIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Button size="small" onClick={fetchGraph} disabled={loading}>
          Refresh
        </Button>
      </Box>
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
        {!loading && !error && visibleNodes.length === 0 && (
          <Box sx={{ p: 3, textAlign: 'center', color: 'text.secondary' }}>
            <Typography>No graph data. Enable Files and/or Entities layers and add documents.</Typography>
          </Box>
        )}
        {!loading && !error && visibleNodes.length > 0 && (
          <GraphCanvas
            nodes={visibleNodes}
            edges={visibleEdges}
            onNodeClick={handleNodeClick}
            theme={theme}
          />
        )}
      </Box>
    </Box>
  );
}
