import React, { useState, useMemo, useEffect, useCallback } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TableContainer,
  Paper,
  Switch,
  IconButton,
  Tooltip,
  Chip,
  Alert,
  CircularProgress,
  Autocomplete,
  TextField,
} from '@mui/material';
import { Add, Refresh, DeleteOutline } from '@mui/icons-material';
import { useQuery, useQueryClient, useMutation } from 'react-query';
import rssService from '../services/rssService';
import RSSFeedManager from './RSSFeedManager';
import { useAuth } from '../contexts/AuthContext';
import apiService from '../services/apiService';
import { formatInstantDateTime } from '../utils/userTimeDisplay';

function flattenFolderTree(nodes, prefix = '') {
  const out = [];
  if (!Array.isArray(nodes)) return out;
  for (const n of nodes) {
    const name = n.name || 'Folder';
    const label = prefix ? `${prefix} / ${name}` : name;
    if (n.folder_id) {
      out.push({ folder_id: n.folder_id, label });
    }
    if (n.children && n.children.length > 0) {
      out.push(...flattenFolderTree(n.children, label));
    }
  }
  return out;
}

function formatInterval(seconds) {
  if (!seconds && seconds !== 0) return '—';
  const m = Math.round(seconds / 60);
  if (m < 60) return `${m} min`;
  const h = Math.round(m / 60);
  return `${h} h`;
}

export default function RSSFeedSettings() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const [addOpen, setAddOpen] = useState(false);
  const [busyId, setBusyId] = useState(null);

  const { data: feeds, isLoading, error, refetch } = useQuery(
    ['rss', 'feeds'],
    () => rssService.getFeeds(),
    { staleTime: 30_000 }
  );

  const { data: rssUnreadCounts = {} } = useQuery(
    ['rss', 'unread-counts'],
    () => rssService.getUnreadCounts(),
    { staleTime: 30_000 }
  );

  const list = Array.isArray(feeds) ? feeds : [];

  const { data: treeData } = useQuery(
    ['folders', 'tree', 'user', 'rss-import-picker'],
    () => apiService.getFolderTree('user'),
    { staleTime: 60_000 }
  );

  const { data: globalTreeData } = useQuery(
    ['folders', 'tree', 'global', 'rss-import-picker'],
    () => apiService.getFolderTree('global'),
    { staleTime: 60_000, enabled: user?.role === 'admin' }
  );

  const { data: importLoc, isLoading: importLocLoading } = useQuery(
    ['rss', 'import-location'],
    () => rssService.getImportLocation(),
    { staleTime: 30_000 }
  );

  const { data: userTimeFormatData } = useQuery(
    'userTimeFormat',
    () => apiService.settings.getUserTimeFormat(),
    { staleTime: 5 * 60 * 1000, refetchOnWindowFocus: false }
  );
  const { data: userTimezoneData } = useQuery(
    'userTimezone',
    () => apiService.getUserTimezone(),
    { staleTime: 5 * 60 * 1000, refetchOnWindowFocus: false }
  );
  const displayTimeFormat = userTimeFormatData?.time_format || '24h';
  const displayTimeZone = userTimezoneData?.timezone || undefined;

  const formatFeedLastCheck = useCallback(
    (iso) => {
      if (!iso) return '—';
      const s = formatInstantDateTime(iso, {
        timeFormat: displayTimeFormat,
        timeZone: displayTimeZone,
      });
      return s || '—';
    },
    [displayTimeFormat, displayTimeZone]
  );

  const folderOptions = useMemo(() => {
    const userFlat = flattenFolderTree(treeData?.folders || []);
    const base = [{ folder_id: '', label: 'Default — Web Sources / feed name' }];
    if (user?.role === 'admin' && globalTreeData?.folders?.length) {
      const globalFlat = flattenFolderTree(globalTreeData.folders).map((o) => ({
        ...o,
        label: `Global — ${o.label}`,
      }));
      const mine = userFlat.map((o) => ({ ...o, label: `My documents — ${o.label}` }));
      return [...base, ...globalFlat, ...mine];
    }
    return [...base, ...userFlat];
  }, [treeData, globalTreeData, user?.role]);

  const [importDraft, setImportDraft] = useState(null);

  useEffect(() => {
    if (importLocLoading) return;
    const fid = importLoc?.folder_id ? String(importLoc.folder_id) : '';
    const match = folderOptions.find((o) => o.folder_id === fid);
    setImportDraft(match || folderOptions[0] || null);
  }, [importLoc?.folder_id, importLocLoading, folderOptions]);

  const saveImportLocMutation = useMutation(
    (opt) => rssService.setImportLocation(opt?.folder_id ? opt.folder_id : null),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['rss', 'import-location']);
      },
    }
  );

  const savedImportFolderId = importLoc?.folder_id ? String(importLoc.folder_id) : '';
  const importLocationDirty =
    importDraft && (importDraft.folder_id || '') !== savedImportFolderId;

  const handleToggleActive = async (feed, next) => {
    const isGlobal = feed.user_id == null || feed.user_id === undefined;
    if (isGlobal && user?.role !== 'admin') {
      return;
    }
    setBusyId(feed.feed_id);
    try {
      await rssService.toggleFeedActive(feed.feed_id, next);
      await queryClient.invalidateQueries(['rss', 'feeds']);
      await queryClient.invalidateQueries(['rss', 'unread-counts']);
    } catch (e) {
      console.error('toggleFeedActive failed', e);
    } finally {
      setBusyId(null);
    }
  };

  const handleRefresh = async (feedId) => {
    setBusyId(feedId);
    try {
      await rssService.refreshFeed(feedId);
      await queryClient.invalidateQueries(['rss', 'feeds']);
    } catch (e) {
      console.error('refreshFeed failed', e);
    } finally {
      setBusyId(null);
    }
  };

  const handleDelete = async (feed) => {
    if (!window.confirm(`Remove feed "${feed.feed_name}"?`)) return;
    setBusyId(feed.feed_id);
    try {
      await rssService.deleteFeed(feed.feed_id);
      await queryClient.invalidateQueries(['rss', 'feeds']);
      await queryClient.invalidateQueries({ queryKey: ['folders', 'tree'], exact: false });
    } catch (e) {
      console.error('deleteFeed failed', e);
    } finally {
      setBusyId(null);
    }
  };

  return (
    <Box>
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" flexWrap="wrap" gap={2}>
            <Typography variant="h6">RSS Feeds</Typography>
            <Button variant="contained" startIcon={<Add />} onClick={() => setAddOpen(true)}>
              Add feed
            </Button>
          </Box>
          <Alert severity="info" sx={{ mt: 2 }}>
            Pausing a feed stops automatic polling. Use Refresh to fetch immediately. Global feeds can only be
            paused by an admin.
          </Alert>
        </CardContent>
      </Card>

      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Import location
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            When you import an article from a feed, it becomes a document in your library. By default it goes under{' '}
            <strong>Web Sources</strong> in a subfolder named after the feed. Choose a folder below to import into a
            specific place in <strong>My Documents</strong> instead. Admins may choose a folder in Global Documents.
          </Typography>
          <Box display="flex" flexWrap="wrap" alignItems="center" gap={2}>
            <Autocomplete
              sx={{ minWidth: 280, flex: '1 1 280px' }}
              options={folderOptions}
              getOptionLabel={(o) => o?.label || ''}
              isOptionEqualToValue={(a, b) => (a?.folder_id || '') === (b?.folder_id || '')}
              value={importDraft}
              onChange={(_, v) => setImportDraft(v || folderOptions[0])}
              loading={importLocLoading || saveImportLocMutation.isLoading}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Default import folder"
                  placeholder="Search folders"
                  InputProps={{
                    ...params.InputProps,
                    endAdornment: (
                      <>
                        {importLocLoading ? <CircularProgress color="inherit" size={20} /> : null}
                        {params.InputProps.endAdornment}
                      </>
                    ),
                  }}
                />
              )}
            />
            <Button
              variant="contained"
              disabled={!importLocationDirty || saveImportLocMutation.isLoading || !importDraft}
              onClick={() => saveImportLocMutation.mutate(importDraft)}
            >
              Save import location
            </Button>
          </Box>
          {saveImportLocMutation.isError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {saveImportLocMutation.error?.message || 'Could not save import location'}
            </Alert>
          )}
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error.message || 'Failed to load feeds'}
        </Alert>
      )}

      <Card>
        <CardContent>
          {isLoading ? (
            <Box display="flex" justifyContent="center" p={3}>
              <CircularProgress />
            </Box>
          ) : (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>URL</TableCell>
                    <TableCell>Scope</TableCell>
                    <TableCell>Interval</TableCell>
                    <TableCell align="center">Polling</TableCell>
                    <TableCell align="center">Unread</TableCell>
                    <TableCell>Last check</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {list.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={8}>
                        <Typography color="text.secondary">No feeds yet. Add one to get started.</Typography>
                      </TableCell>
                    </TableRow>
                  ) : (
                    list.map((feed) => {
                      const isGlobal = feed.user_id == null || feed.user_id === undefined;
                      const canToggle = !isGlobal || user?.role === 'admin';
                      const unread =
                        rssUnreadCounts[feed.feed_id] ??
                        rssUnreadCounts[String(feed.feed_id)] ??
                        0;
                      const active = feed.is_active !== false;
                      return (
                        <TableRow key={feed.feed_id}>
                          <TableCell>{feed.feed_name}</TableCell>
                          <TableCell sx={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {feed.feed_url}
                          </TableCell>
                          <TableCell>
                            {isGlobal ? <Chip size="small" label="Global" /> : <Chip size="small" label="Mine" variant="outlined" />}
                          </TableCell>
                          <TableCell>{formatInterval(feed.check_interval)}</TableCell>
                          <TableCell align="center">
                            <Switch
                              size="small"
                              checked={active}
                              disabled={!canToggle || busyId === feed.feed_id}
                              onChange={(_, v) => handleToggleActive(feed, v)}
                            />
                          </TableCell>
                          <TableCell align="center">{unread > 0 ? <Chip size="small" label={unread} color="primary" /> : '0'}</TableCell>
                          <TableCell>{formatFeedLastCheck(feed.last_check)}</TableCell>
                          <TableCell align="right">
                            <Tooltip title="Poll now">
                              <span>
                                <IconButton
                                  size="small"
                                  disabled={busyId === feed.feed_id}
                                  onClick={() => handleRefresh(feed.feed_id)}
                                >
                                  <Refresh fontSize="small" />
                                </IconButton>
                              </span>
                            </Tooltip>
                            <Tooltip title="Delete feed">
                              <span>
                                <IconButton
                                  size="small"
                                  color="error"
                                  disabled={busyId === feed.feed_id}
                                  onClick={() => handleDelete(feed)}
                                >
                                  <DeleteOutline fontSize="small" />
                                </IconButton>
                              </span>
                            </Tooltip>
                          </TableCell>
                        </TableRow>
                      );
                    })
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>

      <RSSFeedManager
        isOpen={addOpen}
        onClose={() => setAddOpen(false)}
        onFeedAdded={() => {
          queryClient.invalidateQueries(['rss', 'feeds']);
          queryClient.invalidateQueries(['rss', 'unread-counts']);
          setAddOpen(false);
        }}
      />
    </Box>
  );
}
