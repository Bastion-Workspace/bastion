/**
 * File relation graph (link cloud) using Cytoscape. Community coloring, degree-sized nodes, search/scope.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Box,
  Button,
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
import { computeCommunities, communityColorPalette } from './graphUtils';
import GraphCanvas from './GraphCanvas';

export default function FileRelationGraph({
  scope: initialScope = 'all',
  folderId,
  onOpenDocument,
  persistedState,
  onStateChange,
}) {
  const theme = useTheme();
  const [scope, setScope] = useState(persistedState?.scope ?? initialScope ?? 'all');
  const [folderIdState, setFolderIdState] = useState(persistedState?.folderId ?? folderId ?? '');
  const [data, setData] = useState({ nodes: [], edges: [], unresolved_targets: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchFilter, setSearchFilter] = useState(persistedState?.searchFilter ?? '');
  const [fullscreen, setFullscreen] = useState(false);
  const lastViewportRef = React.useRef(persistedState?.viewport ?? null);

  const fetchGraph = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.getLinkGraph(
        scope === 'folder' && folderIdState ? 'folder' : 'all',
        folderIdState || null
      );
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
    if (typeof onStateChange === 'function') {
      onStateChange({
        scope,
        folderId: folderIdState,
        searchFilter,
        viewport: lastViewportRef.current ?? undefined,
      });
    }
  }, [scope, folderIdState, searchFilter, onStateChange]);

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

  const nodesWithColor = useMemo(() => {
    return (data.nodes || []).map((n) => ({
      ...n,
      type: 'file',
      color: communityColorPalette(communities[n.id]),
    }));
  }, [data.nodes, communities]);

  const filteredNodes = useMemo(() => {
    if (!searchFilter?.trim()) return nodesWithColor;
    const q = searchFilter.trim().toLowerCase();
    const matchingIds = new Set(
      nodesWithColor.filter((n) => (n.label || '').toLowerCase().includes(q)).map((n) => n.id)
    );
    const visibleIds = new Set(matchingIds);
    for (const e of data.edges || []) {
      if (matchingIds.has(e.source) || matchingIds.has(e.target)) {
        visibleIds.add(e.source);
        visibleIds.add(e.target);
      }
    }
    return nodesWithColor.filter((n) => visibleIds.has(n.id));
  }, [nodesWithColor, data.edges, searchFilter]);

  const filteredEdges = useMemo(() => {
    const visibleIds = new Set(filteredNodes.map((n) => n.id));
    return (data.edges || []).filter(
      (e) => visibleIds.has(e.source) && visibleIds.has(e.target)
    );
  }, [data.edges, filteredNodes]);

  const handleNodeClick = useCallback(
    (nodeData) => {
      const docId = nodeData?.id;
      const docName = nodeData?.label ?? docId;
      if (docId && window.tabbedContentManagerRef?.openDocument) {
        window.tabbedContentManagerRef.openDocument(docId, docName);
      } else if (docId) {
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
        <Select value={scope} label="Scope" onChange={(e) => setScope(e.target.value)}>
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
      {!loading && !error && filteredNodes.length === 0 && (
        <Box sx={{ p: 3, textAlign: 'center', color: 'text.secondary' }}>
          <Typography>No documents with links yet. Edit .org or .md files and add links.</Typography>
        </Box>
      )}
      {!loading && !error && filteredNodes.length > 0 && (
        <GraphCanvas
          nodes={filteredNodes}
          edges={filteredEdges.map((e) => ({ ...e, edge_type: 'file_link' }))}
          onNodeClick={handleNodeClick}
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
