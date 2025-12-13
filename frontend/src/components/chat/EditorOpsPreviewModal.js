import React, { useEffect, useMemo, useState, useRef } from 'react';
import { Box, Dialog, DialogTitle, DialogContent, DialogActions, Button, Checkbox, FormControlLabel, Typography, Divider, Alert } from '@mui/material';

function computeSimpleHash(text) {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) >>> 0;
  return h.toString(16);
}

function getOriginalSlice(currentText, start, end) {
  const s = Math.max(0, Math.min(currentText.length, Number(start || 0)));
  const e = Math.max(s, Math.min(currentText.length, Number(end || s)));
  return currentText.slice(s, e);
}

function DiffBlock({ original, proposed }) {
  // Minimal unified diff-like presentation without heavy deps
  // Show original and proposed in two blocks for clarity
  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, mt: 1 }}>
      <Box sx={{ p: 1, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
        <Typography variant="caption" color="text.secondary">Original</Typography>
        <Box component="pre" sx={{ whiteSpace: 'pre-wrap', m: 0 }}>{original || ''}</Box>
      </Box>
      <Box sx={{ p: 1, border: '1px solid', borderColor: 'divider', borderRadius: 1, backgroundColor: 'rgba(76,175,80,0.06)' }}>
        <Typography variant="caption" color="text.secondary">Proposed</Typography>
        <Box component="pre" sx={{ whiteSpace: 'pre-wrap', m: 0 }}>{proposed || ''}</Box>
      </Box>
    </Box>
  );
}

export default function EditorOpsPreviewModal({ open, onClose, operations, manuscriptEdit, requestEditorContent, onApplySelected }) {
  const [currentText, setCurrentText] = useState('');
  const [selected, setSelected] = useState({});
  const [error, setError] = useState(null);
  const contentRef = useRef(null);
  const scrollPositionRef = useRef(0);

  useEffect(() => {
    if (!open) return;
    setError(null);
    setSelected({});
    try {
      // Ask the editor to provide current content
      const handler = (e) => {
        try {
          const detail = e.detail || {};
          if (typeof detail.content === 'string') {
            setCurrentText(detail.content.replace(/\r\n/g, '\n'));
          }
        } catch {}
      };
      window.addEventListener('codexProvideEditorContent', handler);
      window.dispatchEvent(new CustomEvent('codexRequestEditorContent'));
      return () => window.removeEventListener('codexProvideEditorContent', handler);
    } catch (e) {
      setError('Failed to request editor content');
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    // Default-select all operations on first open
    const defaults = {};
    (operations || []).forEach((op, idx) => { defaults[idx] = true; });
    setSelected(defaults);
  }, [open, operations]);

  const rows = useMemo(() => {
    const list = Array.isArray(operations) ? operations : [];
    return list.map((op, idx) => {
      // For replace_range operations, prefer original_text if available (more reliable)
      // Fallback to slicing currentText if original_text not available
      let original = '';
      if (op.op_type === 'replace_range' && op.original_text) {
        original = op.original_text;
      } else if (op.op_type === 'delete_range' && op.original_text) {
        original = op.original_text;
      } else {
        // For insert operations (insert_after_heading, insert_after) or when original_text not available, use slice
        original = getOriginalSlice(currentText, op.start, op.end);
      }
      const preOk = op.pre_hash ? computeSimpleHash(original) === op.pre_hash : true;
      return { idx, op, original, preOk };
    });
  }, [operations, currentText]);

  const handleToggle = (idx) => {
    // Preserve scroll position before state update
    if (contentRef.current) {
      scrollPositionRef.current = contentRef.current.scrollTop;
    }
    setSelected(prev => ({ ...prev, [idx]: !prev[idx] }));
  };

  // Restore scroll position after re-render
  useEffect(() => {
    if (contentRef.current && scrollPositionRef.current > 0) {
      // Use requestAnimationFrame to ensure DOM has updated
      requestAnimationFrame(() => {
        if (contentRef.current) {
          contentRef.current.scrollTop = scrollPositionRef.current;
        }
      });
    }
  }, [selected]);

  const selectedOps = useMemo(() => {
    const list = [];
    rows.forEach(r => {
      if (selected[r.idx]) list.push(r.op);
    });
    return list;
  }, [rows, selected]);

  const applySelected = () => {
    try {
      if (!selectedOps.length) return onClose && onClose();
      onApplySelected && onApplySelected(selectedOps, manuscriptEdit || null);
      onClose && onClose();
    } catch (e) {
      setError('Failed to apply selected operations');
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>Review Proposed Edits</DialogTitle>
      <DialogContent dividers ref={contentRef}>
        {error && <Alert severity="error" sx={{ mb: 1 }}>{String(error)}</Alert>}
        {(rows || []).map(({ idx, op, original, preOk }) => (
          <Box key={idx} sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <FormControlLabel
                control={<Checkbox checked={!!selected[idx]} onChange={() => handleToggle(idx)} />}
                label={<Typography variant="body2">Edit #{idx + 1} — {op.op_type || 'replace_range'} [{op.start}:{op.end}] {preOk ? '' : ' (⚠️ stale)'}
                </Typography>}
              />
            </Box>
            <DiffBlock original={original} proposed={op.op_type === 'delete_range' ? '' : (op.text || '')} />
            {!preOk && (
              <Typography variant="caption" color="warning.main" sx={{ mt: 0.5, display: 'block' }}>
                ⚠️ Pre-hash mismatch: underlying text changed. You may still apply, but consider re-running the agent.
              </Typography>
            )}
            <Divider sx={{ mt: 1 }} />
          </Box>
        ))}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="outlined" onClick={() => {
          // Apply none (explicit skip)
          onClose && onClose();
        }}>Skip All</Button>
        <Button variant="contained" onClick={applySelected} disabled={!selectedOps.length}>Apply Selected</Button>
      </DialogActions>
    </Dialog>
  );
}
