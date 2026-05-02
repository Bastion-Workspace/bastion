import React, { useCallback, useMemo, useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
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
  Grid,
  Breadcrumbs,
  Link,
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

function pickAcquisitionLinkFromEntry(en) {
  if (en.acquisition_href) {
    const href = String(en.acquisition_href).trim();
    const t = (en.acquisition_type || '').toLowerCase();
    const format = t === 'pdf' ? 'pdf' : 'epub';
    return { href, format };
  }
  const links = en.links || [];
  const acq = links.find((l) => {
    if (!l?.href) return false;
    const rel = String(l.rel || '').toLowerCase();
    const typ = String(l.type || '').toLowerCase();
    const hrefL = String(l.href).toLowerCase();
    const isAcq =
      rel.includes('opds-spec.org/acquisition') || (rel.includes('acquisition') && rel.includes('opds'));
    if (!isAcq) return false;
    const isEpub = typ.includes('epub') || hrefL.endsWith('.epub');
    const isPdf = typ.includes('application/pdf') || hrefL.endsWith('.pdf');
    return isEpub || isPdf;
  });
  if (!acq?.href) return { href: '', format: 'epub' };
  const href = String(acq.href).trim();
  const typ = String(acq.type || '').toLowerCase();
  const hrefL = href.toLowerCase();
  const isEpub = typ.includes('epub') || hrefL.endsWith('.epub');
  const isPdf = typ.includes('application/pdf') || hrefL.endsWith('.pdf');
  const format = isEpub ? 'epub' : isPdf ? 'pdf' : 'epub';
  return { href, format };
}

function pickAcquisitionHrefFromEntry(en) {
  return pickAcquisitionLinkFromEntry(en).href;
}

function pickAcquisitionFormatFromEntry(en) {
  return pickAcquisitionLinkFromEntry(en).format;
}

function buildSearchUrl(template, rawQuery) {
  const q = encodeURIComponent(rawQuery.trim());
  let u = template;
  if (u.includes('{searchTerms}')) {
    u = u.split('{searchTerms}').join(q);
  } else {
    u = u.replace(/\{[qQ]\}/g, q);
  }
  if (u.includes('%s')) {
    u = u.split('%s').join(q);
  }
  return u;
}

function entrySecondaryLine(en, acq, navHref, format) {
  if (en.summary) {
    return String(en.summary).replace(/<[^>]+>/g, '').slice(0, 140);
  }
  if (acq) return format === 'pdf' ? 'PDF' : 'EPUB';
  const n = en.number_of_items;
  if (typeof n === 'number' && n >= 0) {
    return `${n} title${n === 1 ? '' : 's'}`;
  }
  if (navHref) return 'Catalog section';
  return undefined;
}

function TabPanel({ hidden, children, flexFill }) {
  if (hidden) return null;
  return (
    <Box
      sx={{
        pt: 2,
        ...(flexFill ? { flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' } : {}),
      }}
    >
      {children}
    </Box>
  );
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

  const [navTrail, setNavTrail] = useState([]);
  const [feed, setFeed] = useState(null);
  const [error, setError] = useState('');
  const [catalogSearchTemplate, setCatalogSearchTemplate] = useState('');

  useEffect(() => {
    setCatalogSearchTemplate('');
  }, [catalogId]);

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
    setNavTrail([{ url: root, title: activeCatalog.title || 'Catalog' }]);
    void fetchFeed(root);
    return undefined;
  }, [tab, catalogId, activeCatalog?.root_url, activeCatalog?.title, fetchFeed]);

  useEffect(() => {
    if (feed?.search_template) {
      setCatalogSearchTemplate(feed.search_template);
    }
  }, [feed?.search_template]);

  useEffect(() => {
    if (tab !== 1 || !feed?.feed_title) return;
    setNavTrail((prev) => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      if (last.title === feed.feed_title) return prev;
      const next = [...prev];
      next[next.length - 1] = { ...last, title: feed.feed_title };
      return next;
    });
  }, [feed?.feed_title, tab]);

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
  const effectiveSearchTemplate = feed?.search_template || catalogSearchTemplate;

  const onSearch = useCallback(() => {
    const tmpl = effectiveSearchTemplate;
    if (!tmpl || !searchQuery.trim()) return;
    const u = buildSearchUrl(tmpl, searchQuery);
    setNavTrail((prev) => {
      const root = prev[0] || { url: activeCatalog?.root_url || '', title: activeCatalog?.title || 'Catalog' };
      return [
        root,
        {
          url: u,
          title: `Search: ${searchQuery.trim()}`,
        },
      ];
    });
    setTab(1);
    void fetchFeed(u);
  }, [effectiveSearchTemplate, searchQuery, fetchFeed, activeCatalog?.root_url, activeCatalog?.title]);

  const currentBase = navTrail.length > 0 ? navTrail[navTrail.length - 1].url : activeCatalog?.root_url || '';

  const entries = feed?.entries || [];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', bgcolor: 'background.paper', p: 2 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="h6">OPDS</Typography>
        <Button size="small" onClick={onClose}>
          Close
        </Button>
      </Box>
      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ flexShrink: 0 }}>
        <Tab label="Recent" />
        <Tab label="Browse" />
        <Tab label="Search" />
      </Tabs>

      <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <TabPanel hidden={tab !== 0} flexFill>
          {isLoading ? (
            <Typography>Loading…</Typography>
          ) : recent.length === 0 ? (
            <Typography color="text.secondary">No recently opened ebooks.</Typography>
          ) : (
            <Box sx={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
              <Grid container spacing={0.5}>
                {recent.map((r) => (
                  <Grid item xs={12} sm={6} key={r.digest || `${r.catalog_id}:${r.acquisition_url}`}>
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'stretch',
                        border: 1,
                        borderColor: 'divider',
                        borderRadius: 1,
                        overflow: 'hidden',
                      }}
                    >
                      <ListItemButton
                        sx={{ flex: 1, py: 1, alignItems: 'flex-start' }}
                        onClick={() =>
                          onOpenEbook?.({
                            catalogId: r.catalog_id,
                            acquisitionUrl: r.acquisition_url,
                            title: r.title || 'Book',
                            digest: r.digest,
                            format: r.acquisition_format === 'pdf' ? 'pdf' : 'epub',
                            author: r.author || undefined,
                          })
                        }
                      >
                        <ListItemText
                          primary={r.title || 'Untitled'}
                          secondary={(r.author && String(r.author).trim()) || '\u2014'}
                          primaryTypographyProps={{ variant: 'subtitle2', fontWeight: 600 }}
                          secondaryTypographyProps={{
                            variant: 'caption',
                            color: 'text.secondary',
                            sx: { fontSize: 10, lineHeight: 1.35, mt: 0.35, display: 'block' },
                          }}
                        />
                      </ListItemButton>
                      <IconButton
                        aria-label="Remove from recent"
                        onClick={(e) => {
                          e.stopPropagation();
                          onRemoveRecent(r);
                        }}
                        sx={{ borderRadius: 0, flexShrink: 0 }}
                      >
                        <DeleteOutline fontSize="small" />
                      </IconButton>
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </Box>
          )}
        </TabPanel>

        <TabPanel hidden={tab !== 1} flexFill>
          <Box sx={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
            <Box display="flex" gap={1} alignItems="center" flexWrap="wrap" mb={1} sx={{ flexShrink: 0 }}>
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
                disabled={navTrail.length <= 1}
                onClick={() => {
                  const nextStack = navTrail.slice(0, -1);
                  setNavTrail(nextStack);
                  const u = nextStack[nextStack.length - 1]?.url;
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
              <Typography color="error" variant="body2" sx={{ mb: 1, flexShrink: 0 }}>
                {error}
              </Typography>
            )}
            <Box sx={{ flexShrink: 0, mb: 1 }}>
              <Breadcrumbs maxItems={6} aria-label="Browse path">
                {navTrail.map((crumb, i) => {
                  const isLast = i === navTrail.length - 1;
                  const label = crumb.title || '(untitled)';
                  if (isLast) {
                    return (
                      <Typography key={`${crumb.url}:${i}`} variant="body2" color="text.primary" component="span">
                        {label}
                      </Typography>
                    );
                  }
                  return (
                    <Link
                      key={`${crumb.url}:${i}`}
                      component="button"
                      variant="body2"
                      color="inherit"
                      underline="hover"
                      onClick={() => {
                        const next = navTrail.slice(0, i + 1);
                        setNavTrail(next);
                        const u = next[next.length - 1]?.url;
                        if (u) void fetchFeed(u);
                      }}
                      sx={{ cursor: 'pointer' }}
                    >
                      {label}
                    </Link>
                  );
                })}
              </Breadcrumbs>
            </Box>
            <Box sx={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
              <Grid container spacing={0.5}>
                {entries.map((en) => {
                  const acq = pickAcquisitionHrefFromEntry(en);
                  const acqFormat = pickAcquisitionFormatFromEntry(en);
                  const navHref = pickNavigationHrefFromEntry(en);
                  const secondary = entrySecondaryLine(en, acq, navHref, acqFormat);
                  return (
                    <Grid item xs={12} sm={6} key={en.id || `${en.title}:${navHref || acq}`}>
                      <ListItemButton
                        sx={{
                          border: 1,
                          borderColor: 'divider',
                          borderRadius: 1,
                          alignItems: 'flex-start',
                          height: '100%',
                        }}
                        onClick={() => {
                          if (acq) {
                            const abs = new URL(acq, currentBase).href;
                            onOpenEbook?.({
                              catalogId,
                              acquisitionUrl: abs,
                              title: en.title || 'Book',
                              format: acqFormat,
                              author: en.author || undefined,
                            });
                          } else if (navHref) {
                            const next = new URL(navHref, currentBase).href;
                            setNavTrail((s) => [...s, { url: next, title: en.title || 'Untitled' }]);
                            void fetchFeed(next);
                          } else {
                            setError(
                              'No catalog, EPUB, or PDF link found on this entry (server may use an unsupported OPDS link shape).'
                            );
                          }
                        }}
                      >
                        <ListItemText
                          primary={en.title || '(untitled)'}
                          secondary={secondary}
                          primaryTypographyProps={{ variant: 'body2', fontWeight: 500 }}
                          secondaryTypographyProps={{ variant: 'caption' }}
                        />
                      </ListItemButton>
                    </Grid>
                  );
                })}
              </Grid>
            </Box>
          </Box>
        </TabPanel>

        <TabPanel hidden={tab !== 2} flexFill>
          <Typography variant="body2" color="text.secondary" paragraph>
            Search uses the OpenSearch URL from the catalog. Load the root feed in Browse at least once per catalog so
            the template is available; it stays active while you drill into sections (Authors, etc.).
          </Typography>
          <TextField
            fullWidth
            size="small"
            label="Query"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && effectiveSearchTemplate && searchQuery.trim()) {
                e.preventDefault();
                onSearch();
              }
            }}
            sx={{ mb: 1 }}
          />
          <Button variant="contained" disabled={!effectiveSearchTemplate || !searchQuery.trim()} onClick={onSearch}>
            Run search
          </Button>
          <Divider sx={{ my: 2 }} />
          <Typography variant="caption" color="text.secondary" sx={{ wordBreak: 'break-all' }}>
            Template: {effectiveSearchTemplate || '(none — open Browse on this catalog first)'}
          </Typography>
        </TabPanel>
      </Box>
    </Box>
  );
}
