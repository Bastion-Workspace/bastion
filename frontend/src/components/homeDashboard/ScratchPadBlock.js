import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import {
  Box,
  CircularProgress,
  Stack,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import apiService from '../../services/apiService';

const MAX_BODY = 10000;
const MAX_LABEL = 30;
const PAD_COUNT = 4;

function normalizePads(raw) {
  if (!raw || !Array.isArray(raw.pads) || raw.pads.length !== PAD_COUNT) {
    return {
      pads: Array.from({ length: PAD_COUNT }, (_, i) => ({
        label: `Pad ${i + 1}`,
        body: '',
      })),
      active_index: 0,
    };
  }
  return {
    pads: raw.pads.map((p, i) => ({
      label: String(p?.label ?? `Pad ${i + 1}`).slice(0, MAX_LABEL),
      body: String(p?.body ?? '').slice(0, MAX_BODY),
    })),
    active_index:
      typeof raw.active_index === 'number' && raw.active_index >= 0 && raw.active_index < PAD_COUNT
        ? raw.active_index
        : 0,
  };
}

/**
 * Shared user scratch pad (four pads). Editable in dashboard view mode.
 * Fills the widget card; textarea scrolls — widget height does not grow with content.
 */
export default function ScratchPadBlock({ showLabels = true }) {
  const theme = useTheme();
  const queryClient = useQueryClient();
  const skipDebouncedSaveRef = useRef(true);
  const debounceRef = useRef(null);
  const saveHintTimeoutRef = useRef(null);
  const [pads, setPads] = useState(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [editingLabelIdx, setEditingLabelIdx] = useState(null);
  const [labelDraft, setLabelDraft] = useState('');
  const [saveHint, setSaveHint] = useState('');

  const { data, isLoading, error } = useQuery(['scratchpad'], () => apiService.getScratchpad(), {
    staleTime: 30 * 1000,
    refetchOnWindowFocus: false,
  });

  useEffect(() => {
    if (!data || pads !== null) return;
    const n = normalizePads(data);
    setPads(n.pads);
    setActiveIndex(n.active_index);
    skipDebouncedSaveRef.current = true;
  }, [data, pads]);

  const mutation = useMutation((body) => apiService.putScratchpad(body), {
    onSuccess: (saved) => {
      queryClient.setQueryData(['scratchpad'], saved);
      clearTimeout(saveHintTimeoutRef.current);
      setSaveHint('Saved');
      saveHintTimeoutRef.current = setTimeout(() => setSaveHint(''), 2000);
    },
  });

  const { mutate, isLoading: mutationLoading } = mutation;

  const flushSave = useCallback(() => {
    if (!pads) return;
    mutate({ pads, active_index: activeIndex });
  }, [pads, activeIndex, mutate]);

  useEffect(
    () => () => {
      clearTimeout(debounceRef.current);
      clearTimeout(saveHintTimeoutRef.current);
    },
    []
  );

  useEffect(() => {
    if (!pads) return;
    if (skipDebouncedSaveRef.current) {
      skipDebouncedSaveRef.current = false;
      return;
    }
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      mutate({ pads, active_index: activeIndex });
    }, 1500);
  }, [pads, activeIndex, mutate]);

  const updateBody = (text) => {
    if (!pads) return;
    const next = text.slice(0, MAX_BODY);
    setPads((prev) => {
      if (!prev) return prev;
      const copy = prev.map((p, i) =>
        i === activeIndex ? { ...p, body: next } : p
      );
      return copy;
    });
  };

  const handleTabChange = (_e, value) => {
    if (value === null) return;
    setEditingLabelIdx(null);
    setActiveIndex(value);
  };

  const startRename = (idx) => {
    if (!pads) return;
    setEditingLabelIdx(idx);
    setLabelDraft(pads[idx]?.label ?? `Pad ${idx + 1}`);
  };

  const commitRename = () => {
    if (editingLabelIdx === null || !pads) return;
    const trimmed = labelDraft.slice(0, MAX_LABEL);
    setPads((prev) => {
      if (!prev) return prev;
      return prev.map((p, i) =>
        i === editingLabelIdx ? { ...p, label: trimmed || `Pad ${i + 1}` } : p
      );
    });
    setEditingLabelIdx(null);
  };

  const tabDisplay = (p, i) => {
    if (!showLabels) return String(i + 1);
    return (p?.label || `Pad ${i + 1}`).slice(0, MAX_LABEL);
  };

  if (isLoading && !pads) {
    return (
      <Box display="flex" justifyContent="center" py={2}>
        <CircularProgress size={28} />
      </Box>
    );
  }

  if (error) {
    return (
      <Typography color="error" variant="body2">
        Could not load scratch pad.
      </Typography>
    );
  }

  if (!pads) return null;

  const bodyLen = (pads[activeIndex]?.body || '').length;
  const showCounter = bodyLen > 8000;

  return (
    <Stack
      spacing={1}
      sx={{
        height: '100%',
        minHeight: 0,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Stack direction="row" alignItems="center" justifyContent="space-between" flexWrap="wrap" gap={1}>
        <ToggleButtonGroup
          exclusive
          size="small"
          value={activeIndex}
          onChange={handleTabChange}
          sx={{ flexWrap: 'wrap' }}
        >
          {pads.map((p, i) => (
            <ToggleButton
              key={i}
              value={i}
              sx={{ textTransform: 'none', px: 1.5 }}
              onDoubleClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                startRename(i);
              }}
            >
              {editingLabelIdx === i ? (
                <TextField
                  size="small"
                  value={labelDraft}
                  autoFocus
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => setLabelDraft(e.target.value)}
                  onBlur={commitRename}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      commitRename();
                    }
                    if (e.key === 'Escape') {
                      e.preventDefault();
                      setEditingLabelIdx(null);
                    }
                  }}
                  inputProps={{ maxLength: MAX_LABEL }}
                  sx={{ minWidth: 80, '& .MuiInputBase-input': { py: 0.5 } }}
                />
              ) : (
                tabDisplay(p, i)
              )}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
        <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: 'nowrap' }}>
          {mutationLoading ? 'Saving…' : saveHint}
        </Typography>
      </Stack>
      <Typography variant="caption" color="text.secondary" component="div">
        Double-click a tab to rename. Content syncs across all dashboards.
      </Typography>
      <Box
        sx={{
          flex: 1,
          minHeight: 0,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <textarea
          value={pads[activeIndex]?.body ?? ''}
          onChange={(e) => updateBody(e.target.value)}
          onBlur={() => {
            clearTimeout(debounceRef.current);
            flushSave();
          }}
          spellCheck
          style={{
            width: '100%',
            flex: 1,
            minHeight: 120,
            boxSizing: 'border-box',
            resize: 'none',
            overflow: 'auto',
            fontFamily: 'inherit',
            fontSize: '0.875rem',
            lineHeight: 1.5,
            padding: '8px',
            borderRadius: 4,
            border: `1px solid ${theme.palette.divider}`,
            background: 'transparent',
            color: 'inherit',
          }}
        />
      </Box>
      {showCounter ? (
        <Typography variant="caption" color="text.secondary" align="right">
          {bodyLen} / {MAX_BODY}
        </Typography>
      ) : null}
    </Stack>
  );
}
