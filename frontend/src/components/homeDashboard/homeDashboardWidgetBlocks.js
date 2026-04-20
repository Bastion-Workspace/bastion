import React, { useCallback, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import {
  Box,
  Button,
  Chip,
  CircularProgress,
  Divider,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import apiService from '../../services/apiService';
import orgService from '../../services/org/OrgService';
import { flattenFolderTree } from './homeDashboardUtils';

function agendaTypeColor(t) {
  if (t === 'DEADLINE') return 'error';
  if (t === 'SCHEDULED') return 'primary';
  return 'default';
}

export function OrgAgendaBlock({ config, navigate }) {
  const daysAhead = config?.days_ahead ?? 7;
  const includeScheduled = config?.include_scheduled !== false;
  const includeDeadlines = config?.include_deadlines !== false;
  const includeAppointments = config?.include_appointments !== false;

  const { data, isLoading, error } = useQuery(
    [
      'homeDashboardOrgAgenda',
      daysAhead,
      includeScheduled,
      includeDeadlines,
      includeAppointments,
    ],
    () =>
      orgService.getAgenda({
        daysAhead,
        includeScheduled,
        includeDeadlines,
        includeAppointments,
      }),
    { staleTime: 60 * 1000 }
  );

  const openItem = useCallback(
    async (item) => {
      let docId = item.document_id;
      const title = item.heading || item.filename || 'Org';
      if (!docId && item.filename) {
        try {
          const res = await orgService.lookupDocument(item.filename);
          docId = res?.document?.document_id || res?.document_id;
        } catch (_) {}
      }
      if (docId) {
        navigate(
          `/documents?document=${encodeURIComponent(docId)}&doc_title=${encodeURIComponent(title)}`
        );
      }
    },
    [navigate]
  );

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={2}>
        <CircularProgress size={28} />
      </Box>
    );
  }
  if (error || !data?.success) {
    return (
      <Typography color="error" variant="body2">
        Could not load org agenda.
      </Typography>
    );
  }

  const grouped = data.grouped_by_date || {};
  const dates = Object.keys(grouped).sort();
  if (!dates.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        No scheduled items in this range.
      </Typography>
    );
  }

  return (
    <Stack spacing={2} divider={<Divider flexItem />}>
      {dates.map((d) => (
        <Box key={d}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            {d}
          </Typography>
          <Stack spacing={0.5}>
            {(grouped[d] || []).map((item, idx) => (
              <Box
                key={`${d}-${idx}-${item.line_number}-${item.filename}`}
                sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, alignItems: 'center' }}
              >
                <Chip size="small" label={item.agenda_type || 'ITEM'} color={agendaTypeColor(item.agenda_type)} />
                {item.todo_state ? (
                  <Chip size="small" variant="outlined" label={item.todo_state} />
                ) : null}
                <Button
                  size="small"
                  variant="text"
                  sx={{ textAlign: 'left', justifyContent: 'flex-start', flex: '1 1 auto', minWidth: 0 }}
                  onClick={() => openItem(item)}
                >
                  {item.heading || item.preview || item.filename || 'Item'}
                </Button>
                {item.time ? (
                  <Typography variant="caption" color="text.secondary">
                    {item.time}
                  </Typography>
                ) : null}
              </Box>
            ))}
          </Stack>
        </Box>
      ))}
    </Stack>
  );
}

export function FolderShortcutsView({ config, navigate }) {
  const items = config?.items || [];
  const { data: treeRes, isLoading, error } = useQuery(
    ['homeDashboardFolderTree'],
    () => apiService.getFolderTree('user', false),
    { staleTime: 5 * 60 * 1000 }
  );

  const flat = useMemo(
    () => flattenFolderTree(treeRes?.folders || []),
    [treeRes?.folders]
  );
  const byId = useMemo(() => new Map(flat.map((f) => [f.folder_id, f])), [flat]);

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
        Could not load folders.
      </Typography>
    );
  }

  if (!items.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        No folder shortcuts. Edit layout to add folders.
      </Typography>
    );
  }

  return (
    <Stack direction="row" flexWrap="wrap" gap={1}>
      {items.map((it, idx) => {
        const meta = byId.get(it.folder_id);
        const label = (it.label && it.label.trim()) || meta?.label || it.folder_id;
        return (
          <Button
            key={`${it.folder_id}-${idx}`}
            variant="outlined"
            size="small"
            onClick={() =>
              navigate(`/documents?folder=${encodeURIComponent(it.folder_id)}`)
            }
          >
            {label}
          </Button>
        );
      })}
    </Stack>
  );
}

export function PinnedDocumentsBlock({ config }) {
  const navigate = useNavigate();
  const limit = config?.limit ?? 10;
  const showPreview = Boolean(config?.show_preview);
  const queryClient = useQueryClient();

  const { data, isLoading, error } = useQuery(
    ['homeDashboardDocumentPins', showPreview],
    () => apiService.getDocumentPins(showPreview),
    { staleTime: 30 * 1000 }
  );

  const deleteMutation = useMutation(
    (pinId) => apiService.deleteDocumentPin(pinId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['homeDashboardDocumentPins']);
      },
    }
  );

  const pins = (data?.pins || []).slice(0, limit);

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
        Could not load pinned documents.
      </Typography>
    );
  }

  if (!pins.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        No pinned documents. Use Add below or pin from Documents later.
      </Typography>
    );
  }

  return (
    <Stack spacing={1} divider={<Divider flexItem />}>
      {pins.map((p) => (
        <Box key={p.pin_id}>
          <Stack direction="row" alignItems="flex-start" justifyContent="space-between" gap={1}>
            <Button
              size="small"
              sx={{ textAlign: 'left', justifyContent: 'flex-start', flex: 1, minWidth: 0 }}
              onClick={() =>
                navigate(
                  `/documents?document=${encodeURIComponent(p.document_id)}&doc_title=${encodeURIComponent(p.label || p.title || p.filename || 'Document')}`
                )
              }
            >
              {p.label || p.title || p.filename || p.document_id}
            </Button>
            <Button
              size="small"
              color="error"
              onClick={() => deleteMutation.mutate(p.pin_id)}
              disabled={deleteMutation.isLoading}
            >
              Remove
            </Button>
          </Stack>
          {showPreview && p.content_preview ? (
            <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
              {p.content_preview}
            </Typography>
          ) : null}
        </Box>
      ))}
    </Stack>
  );
}

export function PinnedDocumentsBlockWithAdd({ config }) {
  const [docId, setDocId] = useState('');
  const queryClient = useQueryClient();
  const addMutation = useMutation(
    (payload) => apiService.postDocumentPin(payload),
    {
      onSuccess: () => {
        setDocId('');
        queryClient.invalidateQueries(['homeDashboardDocumentPins']);
      },
    }
  );

  return (
    <Stack spacing={2}>
      <PinnedDocumentsBlock config={config} />
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems="flex-start">
        <TextField
          size="small"
          label="Document ID"
          value={docId}
          onChange={(e) => setDocId(e.target.value)}
          placeholder="Paste document_id"
          sx={{ flex: 1, minWidth: 200 }}
        />
        <Button
          variant="outlined"
          size="small"
          disabled={!docId.trim() || addMutation.isLoading}
          onClick={() => addMutation.mutate({ document_id: docId.trim() })}
        >
          Add pin
        </Button>
      </Stack>
      {addMutation.isError ? (
        <Typography variant="caption" color="error">
          {addMutation.error?.response?.data?.detail || addMutation.error?.message || 'Add failed'}
        </Typography>
      ) : null}
    </Stack>
  );
}
