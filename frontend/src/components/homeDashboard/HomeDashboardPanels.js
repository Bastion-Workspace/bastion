import React, { useMemo } from 'react';
import { useQuery } from 'react-query';
import ReactMarkdown from 'react-markdown';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  CircularProgress,
  Divider,
  FormControl,
  FormControlLabel,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Switch,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import {
  Add,
  ArrowDownward,
  ArrowUpward,
  Delete,
} from '@mui/icons-material';
import {
  flattenFolderTree,
  loadRssHeadlines,
} from './homeDashboardUtils';
import { solidSurfaceBg } from '../../theme/wallpaperPaneSx';
import apiService from '../../services/apiService';
import savedArtifactService from '../../services/savedArtifactService';
import {
  OrgAgendaBlock,
  FolderShortcutsView,
  PinnedDocumentsBlockWithAdd,
} from './homeDashboardWidgetBlocks';
import { FolderImageSlideshowWidgetEditor } from './homeDashboardFolderImageSlideshow';

export function NavLinksView({ config, navigate }) {
  const items = config?.items || [];
  if (!items.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        No links yet. Edit layout to add shortcuts.
      </Typography>
    );
  }
  return (
    <Stack direction="row" flexWrap="wrap" gap={1}>
      {items.map((item, idx) => (
        <Button
          key={idx}
          variant="outlined"
          size="small"
          onClick={() => {
            if (item.path) navigate(item.path);
            else if (item.href) window.open(item.href, '_blank', 'noopener,noreferrer');
          }}
        >
          {item.label}
        </Button>
      ))}
    </Stack>
  );
}

export function MarkdownCardView({ config }) {
  const title = config?.title;
  const body = config?.body ?? '';
  return (
    <Box>
      {title ? (
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
      ) : null}
      <Box className="markdown-body" sx={{ '& p': { mt: 0 }, '& p + p': { mt: 1 } }}>
        <ReactMarkdown>{body || ' '}</ReactMarkdown>
      </Box>
    </Box>
  );
}

/** One useQuery instance per mounted widget (not inside a map callback). */
export function RssHeadlinesBlock({ config, navigate }) {
  const feedId = config?.feed_id || null;
  const limit = config?.limit ?? 8;
  const readFilter =
    config?.read_filter === 'unread' || config?.read_filter === 'read' || config?.read_filter === 'all'
      ? config.read_filter
      : 'all';
  const { data: articles, isLoading, error } = useQuery(
    ['homeDashboardRss', feedId, limit, readFilter],
    () => loadRssHeadlines(feedId, limit, readFilter),
    { staleTime: 60 * 1000 }
  );

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={2}>
        <CircularProgress size={28} />
      </Box>
    );
  }
  if (error) {
    return (
      <Typography color="error" variant="body2">
        Could not load headlines.
      </Typography>
    );
  }
  if (!articles?.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        No articles yet. Add RSS feeds in Settings.
      </Typography>
    );
  }
  return (
    <Stack divider={<Divider flexItem />} spacing={1}>
      {articles.map((a) => (
        <Box key={a.article_id}>
          <Typography
            component="button"
            type="button"
            onClick={() => {
              if (a?.feed_id) {
                const q = new URLSearchParams();
                q.set('rss_feed', a.feed_id);
                q.set('rss_feed_name', a.feed_name || 'RSS');
                if (a.article_id) q.set('rss_article', a.article_id);
                navigate(`/documents?${q.toString()}`);
                return;
              }
              navigate('/documents');
            }}
            sx={{
              border: 'none',
              background: 'none',
              padding: 0,
              cursor: 'pointer',
              textAlign: 'left',
              color: 'primary.main',
              font: 'inherit',
            }}
          >
            {a.title}
          </Typography>
          {a.feed_name ? (
            <Typography variant="caption" color="text.secondary" display="block">
              {a.feed_name}
            </Typography>
          ) : null}
        </Box>
      ))}
    </Stack>
  );
}

export function WidgetCard({
  title,
  children,
  editMode,
  onDelete,
  onUp,
  onDown,
  disableUp,
  disableDown,
  hideStackReorder,
  cardContentOverflow = 'auto',
}) {
  return (
    <Card
      variant="outlined"
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        bgcolor: (t) => solidSurfaceBg(t),
      }}
    >
      <CardHeader
        title={title}
        action={
          editMode ? (
            <Stack direction="row" alignItems="center">
              {!hideStackReorder ? (
                <>
                  <Tooltip title="Move up">
                    <span>
                      <IconButton size="small" onClick={onUp} disabled={disableUp}>
                        <ArrowUpward fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                  <Tooltip title="Move down">
                    <span>
                      <IconButton size="small" onClick={onDown} disabled={disableDown}>
                        <ArrowDownward fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                </>
              ) : null}
              <Tooltip title="Remove widget">
                <IconButton size="small" color="error" onClick={onDelete}>
                  <Delete fontSize="small" />
                </IconButton>
              </Tooltip>
            </Stack>
          ) : null
        }
      />
      <CardContent sx={{ pt: 0, flex: 1, overflow: cardContentOverflow, minHeight: 0 }}>
        {children}
      </CardContent>
    </Card>
  );
}

export function WidgetEditor({ widget, feeds, onChange }) {
  const updateConfig = (patch) => {
    onChange({ ...widget, config: { ...widget.config, ...patch } });
  };

  if (widget.type === 'nav_links') {
    const items = widget.config?.items || [];
    const setItems = (next) => updateConfig({ items: next });
    return (
      <Stack spacing={2}>
        {items.map((item, i) => (
          <Stack key={i} direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems="flex-start">
            <TextField
              size="small"
              label="Label"
              value={item.label}
              onChange={(e) => {
                const next = [...items];
                next[i] = { ...next[i], label: e.target.value };
                setItems(next);
              }}
              sx={{ minWidth: 140 }}
            />
            <TextField
              size="small"
              label="Path (e.g. /documents)"
              value={item.path || ''}
              onChange={(e) => {
                const next = [...items];
                next[i] = { ...next[i], path: e.target.value || null, href: null };
                setItems(next);
              }}
              sx={{ flex: 1 }}
            />
            <Typography variant="caption" sx={{ alignSelf: 'center' }}>
              or
            </Typography>
            <TextField
              size="small"
              label="External URL"
              value={item.href || ''}
              onChange={(e) => {
                const next = [...items];
                next[i] = { ...next[i], href: e.target.value || null, path: null };
                setItems(next);
              }}
              sx={{ flex: 1 }}
            />
            <IconButton size="small" onClick={() => setItems(items.filter((_, j) => j !== i))}>
              <Delete fontSize="small" />
            </IconButton>
          </Stack>
        ))}
        <Button
          size="small"
          startIcon={<Add />}
          onClick={() => setItems([...items, { label: 'New', path: '/documents', href: null }])}
        >
          Add link
        </Button>
      </Stack>
    );
  }

  if (widget.type === 'markdown_card') {
    return (
      <Stack spacing={2}>
        <TextField
          size="small"
          label="Title (optional)"
          value={widget.config?.title || ''}
          onChange={(e) => updateConfig({ title: e.target.value || null })}
        />
        <TextField
          label="Markdown body"
          value={widget.config?.body || ''}
          onChange={(e) => updateConfig({ body: e.target.value })}
          multiline
          minRows={4}
          fullWidth
        />
      </Stack>
    );
  }

  if (widget.type === 'scratchpad') {
    return (
      <Stack spacing={2}>
        <Typography variant="body2" color="text.secondary">
          Scratch pad content is shared across all dashboards. Edit it in view mode (exit Edit layout).
        </Typography>
        <FormControlLabel
          control={
            <Switch
              checked={widget.config?.show_labels !== false}
              onChange={(e) => updateConfig({ show_labels: e.target.checked })}
              size="small"
            />
          }
          label="Show pad labels on tabs"
        />
      </Stack>
    );
  }

  if (widget.type === 'rss_headlines') {
    const readFilter =
      widget.config?.read_filter === 'unread' ||
      widget.config?.read_filter === 'read' ||
      widget.config?.read_filter === 'all'
        ? widget.config.read_filter
        : 'all';
    return (
      <Stack spacing={2}>
        <Stack spacing={2} direction={{ xs: 'column', sm: 'row' }} flexWrap="wrap">
          <FormControl size="small" sx={{ minWidth: 220 }}>
            <InputLabel>Feed</InputLabel>
            <Select
              label="Feed"
              value={widget.config?.feed_id || ''}
              onChange={(e) => updateConfig({ feed_id: e.target.value || null })}
            >
              <MenuItem value="">
                <em>All recent (up to 4 feeds)</em>
              </MenuItem>
              {(feeds || []).map((f) => (
                <MenuItem key={f.feed_id} value={f.feed_id}>
                  {f.feed_name || f.feed_url}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Articles</InputLabel>
            <Select
              label="Articles"
              value={readFilter}
              onChange={(e) => updateConfig({ read_filter: e.target.value })}
            >
              <MenuItem value="unread">Unread only</MenuItem>
              <MenuItem value="all">All (read and unread)</MenuItem>
              <MenuItem value="read">Read only</MenuItem>
            </Select>
          </FormControl>
          <TextField
            size="small"
            type="number"
            label="Max items"
            inputProps={{ min: 1, max: 50 }}
            value={widget.config?.limit ?? 8}
            onChange={(e) =>
              updateConfig({ limit: Math.min(50, Math.max(1, parseInt(e.target.value, 10) || 8)) })
            }
            sx={{ width: 120 }}
          />
        </Stack>
      </Stack>
    );
  }

  if (widget.type === 'org_agenda') {
    return (
      <Stack spacing={2}>
        <TextField
          size="small"
          type="number"
          label="Days ahead"
          inputProps={{ min: 1, max: 30 }}
          value={widget.config?.days_ahead ?? 7}
          onChange={(e) =>
            updateConfig({
              days_ahead: Math.min(30, Math.max(1, parseInt(e.target.value, 10) || 7)),
            })
          }
          sx={{ maxWidth: 160 }}
        />
        <Stack spacing={0.5}>
          <label>
            <input
              type="checkbox"
              checked={widget.config?.include_scheduled !== false}
              onChange={(e) => updateConfig({ include_scheduled: e.target.checked })}
            />{' '}
            Scheduled
          </label>
          <label>
            <input
              type="checkbox"
              checked={widget.config?.include_deadlines !== false}
              onChange={(e) => updateConfig({ include_deadlines: e.target.checked })}
            />{' '}
            Deadlines
          </label>
          <label>
            <input
              type="checkbox"
              checked={widget.config?.include_appointments !== false}
              onChange={(e) => updateConfig({ include_appointments: e.target.checked })}
            />{' '}
            Appointments
          </label>
        </Stack>
      </Stack>
    );
  }

  if (widget.type === 'folder_shortcuts') {
    return <FolderShortcutsWidgetEditor widget={widget} onChange={onChange} />;
  }

  if (widget.type === 'pinned_documents') {
    return (
      <Stack spacing={2}>
        <TextField
          size="small"
          type="number"
          label="Max items shown"
          inputProps={{ min: 1, max: 50 }}
          value={widget.config?.limit ?? 10}
          onChange={(e) =>
            updateConfig({
              limit: Math.min(50, Math.max(1, parseInt(e.target.value, 10) || 10)),
            })
          }
          sx={{ width: 180 }}
        />
        <label>
          <input
            type="checkbox"
            checked={Boolean(widget.config?.show_preview)}
            onChange={(e) => updateConfig({ show_preview: e.target.checked })}
          />{' '}
          Show description preview (loads more data)
        </label>
      </Stack>
    );
  }

  if (widget.type === 'folder_images') {
    return <FolderImageSlideshowWidgetEditor widget={widget} onChange={onChange} />;
  }

  if (widget.type === 'artifact_embed') {
    return <ArtifactEmbedWidgetEditor widget={widget} onChange={onChange} />;
  }

  return null;
}

function ArtifactEmbedWidgetEditor({ widget, onChange }) {
  const updateConfig = (patch) => {
    onChange({ ...widget, config: { ...widget.config, ...patch } });
  };
  const { data, isLoading, error } = useQuery(
    ['savedArtifactsList'],
    () => savedArtifactService.list(),
    { staleTime: 60 * 1000 }
  );
  const artifacts = data?.artifacts || [];
  const current = widget.config?.artifact_id || '';

  return (
    <Stack spacing={2}>
      {isLoading ? (
        <Typography variant="body2" color="text.secondary">
          Loading saved artifacts…
        </Typography>
      ) : null}
      {error ? (
        <Typography variant="body2" color="error">
          Could not load saved artifacts.
        </Typography>
      ) : null}
      <FormControl size="small" fullWidth>
        <InputLabel>Saved artifact</InputLabel>
        <Select
          label="Saved artifact"
          value={current}
          onChange={(e) => updateConfig({ artifact_id: e.target.value || null })}
        >
          <MenuItem value="">
            <em>None</em>
          </MenuItem>
          {artifacts.map((a) => (
            <MenuItem key={a.id} value={a.id}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis' }}>{a.title}</span>
                <Chip size="small" label={String(a.artifact_type || '').toUpperCase()} sx={{ height: 22 }} />
              </Box>
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      {!artifacts.length && !isLoading ? (
        <Typography variant="caption" color="text.secondary">
          Save an artifact from chat (artifact panel) to your library first.
        </Typography>
      ) : null}
    </Stack>
  );
}

function FolderShortcutsWidgetEditor({ widget, onChange }) {
  const { data: treeRes, isLoading } = useQuery(
    ['homeDashboardFolderTreeEditor'],
    () => apiService.getFolderTree('user', false),
    { staleTime: 5 * 60 * 1000 }
  );

  const flat = useMemo(
    () => flattenFolderTree(treeRes?.folders || []),
    [treeRes?.folders]
  );

  const items = widget.config?.items || [];
  const updateConfig = (patch) => {
    onChange({ ...widget, config: { ...widget.config, ...patch } });
  };
  const setItems = (next) => updateConfig({ items: next });

  return (
    <Stack spacing={2}>
      {isLoading ? (
        <Typography variant="body2" color="text.secondary">
          Loading folders…
        </Typography>
      ) : null}
      {items.map((it, i) => (
        <Stack key={`${it.folder_id}-${i}`} direction={{ xs: 'column', sm: 'row' }} spacing={1}>
          <FormControl size="small" sx={{ minWidth: 220, flex: 1 }}>
            <InputLabel>Folder</InputLabel>
            <Select
              label="Folder"
              value={it.folder_id || ''}
              onChange={(e) => {
                const next = [...items];
                next[i] = { ...next[i], folder_id: e.target.value };
                setItems(next);
              }}
            >
              {flat.map((f) => (
                <MenuItem key={f.folder_id} value={f.folder_id}>
                  {f.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <TextField
            size="small"
            label="Label override"
            value={it.label || ''}
            onChange={(e) => {
              const next = [...items];
              next[i] = { ...next[i], label: e.target.value || null };
              setItems(next);
            }}
            sx={{ minWidth: 140 }}
          />
          <IconButton size="small" onClick={() => setItems(items.filter((_, j) => j !== i))}>
            <Delete fontSize="small" />
          </IconButton>
        </Stack>
      ))}
      <Button
        size="small"
        startIcon={<Add />}
        disabled={!flat.length}
        onClick={() => {
          const first = flat[0]?.folder_id;
          if (first) setItems([...items, { folder_id: first, label: null }]);
        }}
      >
        Add folder
      </Button>
    </Stack>
  );
}
