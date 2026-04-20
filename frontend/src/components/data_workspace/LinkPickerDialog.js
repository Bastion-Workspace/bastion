import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  List,
  ListItemButton,
  ListItemText,
  CircularProgress,
  Box,
  Typography
} from '@mui/material';
import dataWorkspaceService from '../../services/dataWorkspaceService';
import { buildRefCell } from './referenceLinkUtils';

export default function LinkPickerDialog({
  open,
  onClose,
  databaseId,
  targetTableId,
  targetTableName,
  labelField,
  onSelect,
  container
}) {
  const [search, setSearch] = useState('');
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!open || !targetTableId) return;
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const acc = [];
        let offset = 0;
        const limit = 200;
        let total = Infinity;
        while (offset < total && acc.length < 2000) {
          const res = await dataWorkspaceService.getTableData(targetTableId, offset, limit, databaseId);
          const chunk = res.rows || [];
          total = res.total_rows ?? chunk.length;
          acc.push(...chunk);
          offset += limit;
          if (chunk.length < limit) break;
        }
        if (!cancelled) setRows(acc);
      } catch (e) {
        if (!cancelled) setError(e.message || 'Failed to load rows');
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open, targetTableId, databaseId]);

  const lf = labelField || 'name';
  const filtered = rows
    .filter((r) => {
      const rd = r.row_data || {};
      if (!search.trim()) return true;
      const q = search.toLowerCase();
      const label = rd[lf] != null ? String(rd[lf]).toLowerCase() : '';
      const id = String(r.row_id || '').toLowerCase();
      return label.includes(q) || id.includes(q);
    })
    .slice(0, 500);

  const pick = (row) => {
    const rd = row.row_data || {};
    let label = rd[lf];
    if (label == null || String(label).trim() === '') {
      label = row.row_id;
      for (const v of Object.values(rd)) {
        if (v != null && typeof v !== 'object' && String(v).trim()) {
          label = String(v);
          break;
        }
      }
    } else {
      label = String(label);
    }
    const preview = {};
    let n = 0;
    for (const [k, val] of Object.entries(rd)) {
      if (k === lf || n >= 4) continue;
      if (val != null && typeof val !== 'object') {
        preview[k] = val;
        n += 1;
      }
    }
    const inner = {
      v: 1,
      table_id: targetTableId,
      row_id: String(row.row_id),
      label
    };
    if (Object.keys(preview).length) inner.preview = preview;
    onSelect(buildRefCell(inner));
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth container={container}>
      <DialogTitle>Link to row{targetTableName ? `: ${targetTableName}` : ''}</DialogTitle>
      <DialogContent>
        <TextField
          fullWidth
          size="small"
          placeholder="Search…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          sx={{ mb: 1, mt: 0.5 }}
        />
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
            <CircularProgress size={28} />
          </Box>
        )}
        {error && (
          <Typography color="error" variant="body2">
            {error}
          </Typography>
        )}
        {!loading && !error && (
          <List dense sx={{ maxHeight: 360, overflow: 'auto' }}>
            {filtered.map((row) => {
              const rd = row.row_data || {};
              const title =
                rd[lf] != null && String(rd[lf]).trim() !== '' ? String(rd[lf]) : row.row_id;
              return (
                <ListItemButton key={row.row_id} onClick={() => pick(row)}>
                  <ListItemText primary={title} secondary={row.row_id} />
                </ListItemButton>
              );
            })}
          </List>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
      </DialogActions>
    </Dialog>
  );
}
