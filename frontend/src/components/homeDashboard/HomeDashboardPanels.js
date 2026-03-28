import React from 'react';
import { useQuery } from 'react-query';
import ReactMarkdown from 'react-markdown';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  CircularProgress,
  Divider,
  FormControl,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
  Stack,
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
import { loadRssHeadlines } from './homeDashboardUtils';

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
  const { data: articles, isLoading, error } = useQuery(
    ['homeDashboardRss', feedId, limit],
    () => loadRssHeadlines(feedId, limit),
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
        No articles yet. Add RSS feeds under News.
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
            onClick={() => navigate(`/news/${a.article_id}`)}
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

export function WidgetCard({ title, children, editMode, onDelete, onUp, onDown, disableUp, disableDown }) {
  return (
    <Card variant="outlined">
      <CardHeader
        title={title}
        action={
          editMode ? (
            <Stack direction="row" alignItems="center">
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
              <Tooltip title="Remove widget">
                <IconButton size="small" color="error" onClick={onDelete}>
                  <Delete fontSize="small" />
                </IconButton>
              </Tooltip>
            </Stack>
          ) : null
        }
      />
      <CardContent sx={{ pt: 0 }}>{children}</CardContent>
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

  if (widget.type === 'rss_headlines') {
    return (
      <Stack spacing={2} direction={{ xs: 'column', sm: 'row' }}>
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
    );
  }

  return null;
}
