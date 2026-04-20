import React, { useCallback, useMemo, useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Tabs,
  Tab,
  TextField,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  IconButton,
  Divider,
} from '@mui/material';
import { ArrowBack, Refresh, DeleteOutline } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import ebooksService from '../services/ebooksService';
import { ebookCacheDelete } from '../services/ebookIndexedDb';

function linkLooksLikeOpdsCatalogNav(l) {
  if (!l?.href) return false;
  const rel = String(l.rel || '').toLowerCase();
  const typ = String(l.type || '').toLowerCase();
  if (rel.includes('opds-spec.org/acquisition') || (rel.includes('acquisition') && rel.includes('opds'))) {
    return false;
  }
  if (rel.includes('thumbnail') || rel.includes('cover') || rel.includes('opds-spec.org/image')) {
    return false;
  }
  if (rel.includes('subsection')) return true;
  if (rel.includes('opds-spec.org/catalog')) return true;
  if (rel.includes('opds-spec.org/group')) return true;
  if (typ.includes('profile=opds-catalog') || (typ.includes('opds-catalog') && typ.includes('atom'))) {
    return true;
  }
  if ((rel.includes('alternate') || rel.includes('related')) && typ.includes('opds-catalog')) {
    return true;
  }
  return false;
}

function pickNavigationHrefFromEntry(en) {
  const fromApi = (en.navigation_links || []).find((l) => l?.href);
  if (fromApi?.href) return String(fromApi.href).trim();
  const fromLinks = (en.links || []).find(linkLooksLikeOpdsCatalogNav);
  return fromLinks?.href ? String(fromLinks.href).trim() : '';
}

function pickAcquisitionHrefFromEntry(en) {
  if (en.acquisition_href) return String(en.acquisition_href).trim();
  const links = en.links || [];
  const acq = links.find((l) => {
    if (!l?.href) return false;
    const rel = String(l.rel || '').toLowerCase();
    const typ = String(l.type || '').toLowerCase();
    const isAcq =
      rel.includes('opds-spec.org/acquisition') || (rel.includes('acquisition') && rel.includes('opds'));
    if (!isAcq) return false;
    return typ.includes('epub') || String(l.href).toLowerCase().endsWith('.epub');
  });
  return acq?.href ? String(acq.href).trim() : '';
}

function TabPanel({ hidden, children }) {
  if (hidden) return null;
  return <Box sx={{ pt: 2 }}>{children}</Box>;
}

export default function OpdsHubTab({ onClose, onOpenEbook }) {
  const queryClient = useQueryClient();
  const [tab, setTab] = useState(0);
  const { data: settings, isLoading } = useQuery(['ebooks-settings'], () => ebooksService.getSettings(), {
    staleTime: 20_000,
  });

  const catalogs = settings?.catalogs || [];
  const recent = settings?.recently_opened || [];

  const [catalogId, setCatalogId] = useState('');
  useEffect(() => {
    if (!catalogId && catalogs.length > 0) {
      setCatalogId(catalogs[0].id);
    }
  }, [catalogId, catalogs]);

  const activeCatalog = useMemo(
    () => catalogs.find((c) => c.id === catalogId) || null,
    [catalogs, catalogId]
  );

  const [navStack, setNavStack] = useState([]);
  const [feed, setFeed] = useState(null);
  const [error, setError] = useState('');

  const fetchFeed = useCallback(
    async (url) => {
      if (!catalogId || !url) return;
      setError('');
      try {
        const res = await ebooksService.fetchOpds({ catalog_id: catalogId, url, want: 'atom' });
        setFeed(res.feed || null);
      } catch (e) {
        setFeed(null);
        setError(e?.message || 'Fetch failed');
      }
    },
    [catalogId]
  );

  useEffect(() => {
    if (tab !== 1 || !activeCatalog?.root_url) return undefined;
    const root = activeCatalog.root_url;
    setNavStack([root]);
    void fetchFeed(root);
    return undefined;
  }, [tab, catalogId, activeCatalog?.root_url, fetchFeed]);

  const saveRecentMutation = useMutation((recently_opened) => ebooksService.putSettings({ recently_opened }), {
    onSuccess: () => queryClient.invalidateQueries(['ebooks-settings']),
  });

  const onRemoveRecent = useCallback(
    async (row) => {
      if (row.digest) {
        try {
          await ebookCacheDelete(row.digest);
        } catch (_) {}
      }
      const next = recent.filter((r) => {
        if (row.digest && r.digest) return r.digest !== row.digest;
        return !(r.acquisition_url === row.acquisition_url && r.catalog_id === row.catalog_id);
      });
      saveRecentMutation.mutate(next);
    },
    [recent, saveRecentMutation]
  );

  const [searchQuery, setSearchQuery] = useState('');
  const onSearch = useCallback(() => {
    const tmpl = feed?.search_template;
    if (!tmpl || !searchQuery.trim()) return;
    let u = tmpl;
    if (u.includes('{searchTerms}')) {
      u = u.split('{searchTerms}').join(encodeURIComponent(searchQuery.trim()));
    } else {
      u = u.replace(/\{[qQ]\}/g, encodeURIComponent(searchQuery.trim()));
    }
    setNavStack([u]);
    setTab(1);
    void fetchFeed(u);
  }, [feed?.search_template, searchQuery, fetchFeed]);

  const currentBase = navStack.length > 0 ? navStack[navStack.length - 1] : activeCatalog?.root_url || '';

  const entries = feed?.entries || [];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', bgcolor: 'background.paper', p: 2 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="h6">OPDS</Typography>
        <Button size="small" onClick={onClose}>
          Close
        </Button>
      </Box>
      <Tabs value={tab} onChange={(_, v) => setTab(v)}>
        <Tab label="Recent" />
        <Tab label="Browse" />
        <Tab label="Search" />
      </Tabs>

      <TabPanel hidden={tab !== 0}>
        {isLoading ? (
          <Typography>Loading…</Typography>
        ) : recent.length === 0 ? (
          <Typography color="text.secondary">No recently opened ebooks.</Typography>
        ) : (
          <List dense>
            {recent.map((r) => (
              <ListItem
                key={r.digest || `${r.catalog_id}:${r.acquisition_url}`}
                disablePadding
                secondaryAction={
                  <IconButton edge="end" aria-label="Remove from recent" onClick={() => onRemoveRecent(r)}>
                    <DeleteOutline fontSize="small" />
                  </IconButton>
                }
              >
                <ListItemButton
                  onClick={() =>
                    onOpenEbook?.({
                      catalogId: r.catalog_id,
                      acquisitionUrl: r.acquisition_url,
                      title: r.title || 'Book',
                      digest: r.digest,
                    })
                  }
                >
                  <ListItemText primary={r.title || 'Untitled'} secondary={r.digest || ''} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        )}
      </TabPanel>

      <TabPanel hidden={tab !== 1}>
        <Box display="flex" gap={1} alignItems="center" flexWrap="wrap" mb={1}>
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Catalog</InputLabel>
            <Select label="Catalog" value={catalogId} onChange={(e) => setCatalogId(e.target.value)}>
              {catalogs.map((c) => (
                <MenuItem key={c.id} value={c.id}>
                  {c.title}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <IconButton
            size="small"
            disabled={navStack.length <= 1}
            onClick={() => {
              const nextStack = navStack.slice(0, -1);
              setNavStack(nextStack);
              const u = nextStack[nextStack.length - 1];
              if (u) void fetchFeed(u);
            }}
            aria-label="Back"
          >
            <ArrowBack />
          </IconButton>
          <IconButton size="small" onClick={() => currentBase && void fetchFeed(currentBase)} aria-label="Refresh">
            <Refresh />
          </IconButton>
        </Box>
        {error && (
          <Typography color="error" variant="body2" sx={{ mb: 1 }}>
            {error}
          </Typography>
        )}
        <Typography variant="subtitle2" gutterBottom>
          {feed?.feed_title}
        </Typography>
        <List dense sx={{ overflow: 'auto', flex: 1 }}>
          {entries.map((en) => {
            const acq = pickAcquisitionHrefFromEntry(en);
            const navHref = pickNavigationHrefFromEntry(en);
            return (
              <ListItemButton
                key={en.id || en.title}
                onClick={() => {
                  if (acq) {
                    const abs = new URL(acq, currentBase).href;
                    onOpenEbook?.({
                      catalogId,
                      acquisitionUrl: abs,
                      title: en.title || 'Book',
                    });
                  } else if (navHref) {
                    const next = new URL(navHref, currentBase).href;
                    setNavStack((s) => [...s, next]);
                    void fetchFeed(next);
                  } else {
                    setError(
                      'No catalog or EPUB link found on this entry (server may use an unsupported OPDS link shape).'
                    );
                  }
                }}
              >
                <ListItemText
                  primary={en.title || '(untitled)'}
                  secondary={
                    en.summary
                      ? String(en.summary).replace(/<[^>]+>/g, '').slice(0, 140)
                      : acq
                        ? 'EPUB'
                        : 'Navigation'
                  }
                />
              </ListItemButton>
            );
          })}
        </List>
      </TabPanel>

      <TabPanel hidden={tab !== 2}>
        <Typography variant="body2" color="text.secondary" paragraph>
          Search uses the OpenSearch template from the feed last loaded in Browse. Switch to Browse to load a catalog,
          then return here.
        </Typography>
        <TextField
          fullWidth
          size="small"
          label="Query"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          sx={{ mb: 1 }}
        />
        <Button variant="contained" disabled={!feed?.search_template} onClick={onSearch}>
          Run search
        </Button>
        <Divider sx={{ my: 2 }} />
        <Typography variant="caption" color="text.secondary">
          Template: {feed?.search_template || '(none)'}
        </Typography>
      </TabPanel>
    </Box>
  );
}
