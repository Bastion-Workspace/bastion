/**
 * Renders a JSON object as a collapsible, clickable tree.
 * Clicking a key path calls onPathSelect with dot-notation (e.g. results.data.items).
 * Used in connector test panel to set response_list_path.
 */

import React, { useState } from 'react';
import { Box, Typography, Paper, IconButton } from '@mui/material';
import { ExpandMore, ExpandLess } from '@mui/icons-material';

function JsonNode({ data, path = '', onPathSelect, depth = 0 }) {
  const [open, setOpen] = useState(depth < 2);
  const isRoot = path === '';

  if (data === null) {
    return (
      <Typography component="span" sx={{ fontFamily: 'monospace', color: 'text.secondary' }}>
        null
      </Typography>
    );
  }
  if (typeof data === 'boolean') {
    return (
      <Typography component="span" sx={{ fontFamily: 'monospace', color: 'warning.main' }}>
        {String(data)}
      </Typography>
    );
  }
  if (typeof data === 'number') {
    return (
      <Typography component="span" sx={{ fontFamily: 'monospace', color: 'info.main' }}>
        {data}
      </Typography>
    );
  }
  if (typeof data === 'string') {
    const preview = data.length > 60 ? data.slice(0, 60) + '…' : data;
    return (
      <Typography component="span" sx={{ fontFamily: 'monospace', color: 'success.main' }} title={data}>
        "{preview}"
      </Typography>
    );
  }
  if (Array.isArray(data)) {
    const key = path || '[]';
    return (
      <Box sx={{ pl: depth * 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap' }}>
          <IconButton size="small" onClick={() => setOpen((o) => !o)} sx={{ p: 0, mr: 0.25 }}>
            {open ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
          </IconButton>
          <Typography
            component="span"
            sx={{
              fontFamily: 'monospace',
              cursor: path ? 'pointer' : 'default',
              '&:hover': path ? { textDecoration: 'underline' } : {},
            }}
            onClick={() => path && onPathSelect && onPathSelect(path)}
          >
            [{data.length} items]
          </Typography>
        </Box>
        {open && (
          <Box sx={{ pl: 2 }}>
            {data.slice(0, 5).map((item, i) => (
              <Box key={i} sx={{ display: 'flex', gap: 0.5, alignItems: 'flex-start' }}>
                <Typography component="span" sx={{ fontFamily: 'monospace', color: 'text.secondary', minWidth: 24 }}>
                  {i}:
                </Typography>
                <JsonNode data={item} path={path ? `${path}[${i}]` : `[${i}]`} onPathSelect={onPathSelect} depth={depth + 1} />
              </Box>
            ))}
            {data.length > 5 && (
              <Typography component="span" sx={{ fontFamily: 'monospace', color: 'text.secondary' }}>
                … and {data.length - 5} more
              </Typography>
            )}
          </Box>
        )}
      </Box>
    );
  }
  if (typeof data === 'object') {
    const entries = Object.entries(data);
    return (
      <Box sx={{ pl: depth * 2 }}>
        {entries.map(([k, v]) => (
          <JsonObjectEntry
            key={k}
            keyName={k}
            value={v}
            path={path ? `${path}.${k}` : k}
            onPathSelect={onPathSelect}
            depth={depth}
          />
        ))}
      </Box>
    );
  }
  return (
    <Typography component="span" sx={{ fontFamily: 'monospace' }}>
      {String(data)}
    </Typography>
  );
}

function JsonObjectEntry({ keyName, value, path, onPathSelect, depth }) {
  const [open, setOpen] = useState(depth < 1);
  const isExpandable = value !== null && typeof value === 'object';

  return (
    <Box sx={{ mb: 0.25 }}>
      <Box sx={{ display: 'flex', alignItems: 'flex-start', flexWrap: 'wrap' }}>
        {isExpandable && (
          <IconButton size="small" onClick={() => setOpen((o) => !o)} sx={{ p: 0, mr: 0.25 }}>
            {open ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
          </IconButton>
        )}
        {!isExpandable && <Box component="span" sx={{ width: 24 }} />}
        <Typography
          component="span"
          sx={{
            fontFamily: 'monospace',
            fontWeight: 600,
            cursor: 'pointer',
            '&:hover': { textDecoration: 'underline' },
            mr: 0.5,
          }}
          onClick={() => onPathSelect && onPathSelect(path)}
        >
          {keyName}:
        </Typography>
        {!isExpandable && <JsonNode data={value} path={path} onPathSelect={onPathSelect} depth={depth + 1} />}
      </Box>
      {isExpandable && open && (
        <Box sx={{ pl: 2 }}>
          <JsonNode data={value} path={path} onPathSelect={onPathSelect} depth={depth + 1} />
        </Box>
      )}
    </Box>
  );
}

export default function JsonResponseViewer({ data, onPathSelect, maxHeight = 400 }) {
  if (data === undefined || data === null) return null;
  let parsed = data;
  if (typeof data === 'string') {
    try {
      parsed = JSON.parse(data);
    } catch {
      return (
        <Paper variant="outlined" sx={{ p: 2, fontFamily: 'monospace', fontSize: '0.875rem', whiteSpace: 'pre-wrap' }}>
          {data}
        </Paper>
      );
    }
  }
  return (
    <Paper
      variant="outlined"
      sx={{
        p: 2,
        maxHeight,
        overflow: 'auto',
        fontSize: '0.875rem',
      }}
    >
      <JsonNode data={parsed} onPathSelect={onPathSelect} />
    </Paper>
  );
}
