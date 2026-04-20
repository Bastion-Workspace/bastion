import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Button,
  IconButton,
  ListSubheader,
  Menu,
  MenuItem,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import ResizableRightDrawer from './ResizableRightDrawer';
import {
  Add,
  ArrowDownward,
  ArrowLeft,
  ArrowRight,
  ArrowUpward,
  Delete,
  FormatAlignCenter,
  FormatAlignLeft,
  FormatAlignRight,
  Sort,
} from '@mui/icons-material';
import { serializeMarkdownTable } from '../../utils/markdownTableUtils';

function cloneModel(m) {
  return {
    headers: [...(m.headers || [])],
    alignments: [...(m.alignments || [])],
    rows: (m.rows || []).map((r) => [...r]),
  };
}

function cycleAlign(current) {
  if (current === 'left') return 'center';
  if (current === 'center') return 'right';
  return 'left';
}

function alignIcon(align) {
  if (align === 'center') return <FormatAlignCenter fontSize="small" />;
  if (align === 'right') return <FormatAlignRight fontSize="small" />;
  return <FormatAlignLeft fontSize="small" />;
}

/**
 * @param {{
 *   open: boolean,
 *   onClose: () => void,
 *   initialModel: { headers: string[], alignments: string[], rows: string[][] },
 *   isEditing: boolean,
 *   onApply: (markdown: string) => void,
 * }} props
 */
export default function MarkdownTableEditor({ open, onClose, initialModel, isEditing, onApply }) {
  const [model, setModel] = useState(() => cloneModel(initialModel));
  const [sortAnchor, setSortAnchor] = useState(null);

  useEffect(() => {
    if (open) {
      setModel(cloneModel(initialModel));
    }
  }, [open, initialModel]);

  const colCount = model.headers.length;

  const ensureWidths = useCallback((m) => {
    const h = [...m.headers];
    const a = [...m.alignments];
    const rows = m.rows.map((r) => [...r]);
    while (a.length < h.length) a.push('left');
    a.length = h.length;
    for (let i = 0; i < rows.length; i++) {
      while (rows[i].length < h.length) rows[i].push('');
      rows[i].length = h.length;
    }
    return { headers: h, alignments: a, rows };
  }, []);

  const setHeader = (idx, val) => {
    setModel((prev) => {
      const next = cloneModel(prev);
      next.headers[idx] = val;
      return ensureWidths(next);
    });
  };

  const setCell = (rowIdx, colIdx, val) => {
    setModel((prev) => {
      const next = cloneModel(prev);
      if (!next.rows[rowIdx]) next.rows[rowIdx] = next.headers.map(() => '');
      next.rows[rowIdx][colIdx] = val;
      return ensureWidths(next);
    });
  };

  const toggleAlign = (colIdx) => {
    setModel((prev) => {
      const next = cloneModel(prev);
      const cur = next.alignments[colIdx] || 'left';
      next.alignments[colIdx] = cycleAlign(cur);
      return next;
    });
  };

  const addColumn = () => {
    setModel((prev) => {
      const next = cloneModel(prev);
      next.headers.push(`Column ${next.headers.length + 1}`);
      next.alignments.push('left');
      next.rows.forEach((r) => r.push(''));
      return next;
    });
  };

  const removeColumn = (colIdx) => {
    setModel((prev) => {
      if (prev.headers.length <= 1) return prev;
      const next = cloneModel(prev);
      next.headers.splice(colIdx, 1);
      next.alignments.splice(colIdx, 1);
      next.rows.forEach((r) => r.splice(colIdx, 1));
      return next;
    });
  };

  const moveColumn = (colIdx, delta) => {
    const j = colIdx + delta;
    setModel((prev) => {
      if (j < 0 || j >= prev.headers.length) return prev;
      const next = cloneModel(prev);
      const swap = (arr, a, b) => {
        const t = arr[a];
        arr[a] = arr[b];
        arr[b] = t;
      };
      swap(next.headers, colIdx, j);
      swap(next.alignments, colIdx, j);
      next.rows.forEach((r) => swap(r, colIdx, j));
      return next;
    });
  };

  const addRow = () => {
    setModel((prev) => {
      const next = cloneModel(prev);
      next.rows.push(prev.headers.map(() => ''));
      return ensureWidths(next);
    });
  };

  const removeRow = (rowIdx) => {
    setModel((prev) => {
      if (prev.rows.length <= 1) return prev;
      const next = cloneModel(prev);
      next.rows.splice(rowIdx, 1);
      return next;
    });
  };

  const moveRow = (rowIdx, delta) => {
    const j = rowIdx + delta;
    setModel((prev) => {
      if (j < 0 || j >= prev.rows.length) return prev;
      const next = cloneModel(prev);
      const t = next.rows[rowIdx];
      next.rows[rowIdx] = next.rows[j];
      next.rows[j] = t;
      return next;
    });
  };

  const sortByColumn = (colIdx, direction) => {
    setModel((prev) => {
      const next = cloneModel(prev);
      const mult = direction === 'desc' ? -1 : 1;
      next.rows.sort((ra, rb) => {
        const a = String(ra[colIdx] ?? '');
        const b = String(rb[colIdx] ?? '');
        const na = parseFloat(a.replace(/,/g, ''));
        const nb = parseFloat(b.replace(/,/g, ''));
        if (!Number.isNaN(na) && !Number.isNaN(nb) && a.trim() !== '' && b.trim() !== '') {
          return (na - nb) * mult;
        }
        return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }) * mult;
      });
      return next;
    });
    setSortAnchor(null);
  };

  const handleApply = () => {
    const m = ensureWidths(model);
    const md = serializeMarkdownTable(m);
    onApply(md);
    onClose();
  };

  return (
    <ResizableRightDrawer
      open={open}
      onClose={onClose}
      defaultWidth={520}
      minWidth={320}
      storageKey="markdownEditor_tableDrawerWidth"
      ModalProps={{ keepMounted: true }}
    >
      <Box sx={{ p: 2, boxSizing: 'border-box' }} role="presentation">
        <Typography variant="h6" sx={{ mb: 0.5 }}>
          Table editor
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
          Edit the table graphically. Apply writes GitHub-flavored Markdown pipe syntax into the document.
        </Typography>

        <Stack direction="row" spacing={1} sx={{ mb: 2, flexWrap: 'wrap', gap: 1 }}>
          <Button size="small" variant="outlined" startIcon={<Add />} onClick={addColumn}>
            Add column
          </Button>
          <Button
            size="small"
            variant="outlined"
            startIcon={<Sort />}
            onClick={(e) => setSortAnchor(e.currentTarget)}
          >
            Sort
          </Button>
          <Menu anchorEl={sortAnchor} open={Boolean(sortAnchor)} onClose={() => setSortAnchor(null)}>
            {model.headers.flatMap((h, idx) => [
              <ListSubheader key={`sub-${idx}`} disableSticky sx={{ lineHeight: 1.5, py: 0.5 }}>
                {h || `Column ${idx + 1}`}
              </ListSubheader>,
              <MenuItem key={`asc-${idx}`} dense onClick={() => sortByColumn(idx, 'asc')}>
                Sort ascending
              </MenuItem>,
              <MenuItem key={`desc-${idx}`} dense onClick={() => sortByColumn(idx, 'desc')}>
                Sort descending
              </MenuItem>,
            ])}
          </Menu>
        </Stack>

        <Box sx={{ overflowX: 'auto', mb: 1 }}>
          <Stack spacing={1.5}>
            <Box>
              <Stack direction="row" spacing={1.25} sx={{ alignItems: 'flex-start', flexWrap: 'nowrap' }}>
                {model.headers.map((h, colIdx) => (
                  <Box
                    key={colIdx}
                    sx={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 0.5,
                      minWidth: 120,
                      flex: '1 1 120px',
                    }}
                  >
                    <TextField
                      size="small"
                      value={h}
                      onChange={(e) => setHeader(colIdx, e.target.value)}
                      fullWidth
                      placeholder="Header"
                      inputProps={{ 'aria-label': `Column ${colIdx + 1} header` }}
                    />
                    <Stack direction="row" spacing={0}>
                      <Tooltip title="Alignment (left / center / right)">
                        <IconButton size="small" onClick={() => toggleAlign(colIdx)} aria-label="column alignment">
                          {alignIcon(model.alignments[colIdx] || 'left')}
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Move column left">
                        <span>
                          <IconButton
                            size="small"
                            disabled={colIdx === 0}
                            onClick={() => moveColumn(colIdx, -1)}
                            aria-label="move column left"
                          >
                            <ArrowLeft fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                      <Tooltip title="Move column right">
                        <span>
                          <IconButton
                            size="small"
                            disabled={colIdx >= colCount - 1}
                            onClick={() => moveColumn(colIdx, 1)}
                            aria-label="move column right"
                          >
                            <ArrowRight fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                      <Tooltip title="Remove column">
                        <span>
                          <IconButton
                            size="small"
                            disabled={colCount <= 1}
                            onClick={() => removeColumn(colIdx)}
                            aria-label="remove column"
                          >
                            <Delete fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                    </Stack>
                  </Box>
                ))}
              </Stack>
            </Box>

            <Box>
              <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                Rows
              </Typography>
              {model.rows.map((row, rowIdx) => (
                <Stack
                  key={rowIdx}
                  direction="row"
                  spacing={1.25}
                  sx={{ alignItems: 'center', mb: 1.5, flexWrap: 'nowrap' }}
                >
                  {model.headers.map((_, colIdx) => (
                    <TextField
                      key={colIdx}
                      size="small"
                      value={row[colIdx] ?? ''}
                      onChange={(e) => setCell(rowIdx, colIdx, e.target.value)}
                      sx={{ minWidth: 120, flex: '1 1 120px' }}
                    />
                  ))}
                  <Stack direction="row" sx={{ flexShrink: 0 }}>
                    <Tooltip title="Move row up">
                      <span>
                        <IconButton size="small" disabled={rowIdx === 0} onClick={() => moveRow(rowIdx, -1)}>
                          <ArrowUpward fontSize="small" />
                        </IconButton>
                      </span>
                    </Tooltip>
                    <Tooltip title="Move row down">
                      <span>
                        <IconButton
                          size="small"
                          disabled={rowIdx >= model.rows.length - 1}
                          onClick={() => moveRow(rowIdx, 1)}
                        >
                          <ArrowDownward fontSize="small" />
                        </IconButton>
                      </span>
                    </Tooltip>
                    <Tooltip title="Remove row">
                      <span>
                        <IconButton size="small" disabled={model.rows.length <= 1} onClick={() => removeRow(rowIdx)}>
                          <Delete fontSize="small" />
                        </IconButton>
                      </span>
                    </Tooltip>
                  </Stack>
                </Stack>
              ))}
              <Button size="small" startIcon={<Add />} onClick={addRow} sx={{ mt: 0.5 }}>
                Add row
              </Button>
            </Box>
          </Stack>
        </Box>

        <Box sx={{ mt: 2, display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
          <Button variant="outlined" size="small" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="contained" size="small" onClick={handleApply}>
            {isEditing ? 'Apply' : 'Insert'}
          </Button>
        </Box>
      </Box>
    </ResizableRightDrawer>
  );
}
