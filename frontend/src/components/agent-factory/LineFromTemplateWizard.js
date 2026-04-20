/**
 * Create an agent line from a built-in template (goals, workspace seed, heartbeat defaults).
 */

import React, { useMemo, useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Checkbox,
  CircularProgress,
} from '@mui/material';
import { useQuery } from 'react-query';
import apiService from '../../services/apiService';

export default function LineFromTemplateWizard({ open, onClose, onSuccess }) {
  const [templateId, setTemplateId] = useState('');
  const [name, setName] = useState('');
  const [handle, setHandle] = useState('');
  const [ceoId, setCeoId] = useState('');
  const [memberIds, setMemberIds] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  const { data: templates = [], isLoading: templatesLoading } = useQuery(
    ['agentFactoryLineTemplates'],
    () => apiService.agentFactory.listLineTemplates(),
    { enabled: open }
  );

  const { data: profilesRaw } = useQuery(
    ['agentFactoryProfiles'],
    () => apiService.agentFactory.listProfiles(),
    { enabled: open }
  );

  const profiles = useMemo(() => {
    const raw = profilesRaw ?? [];
    return Array.isArray(raw) ? raw : (raw?.data ?? []);
  }, [profilesRaw]);

  React.useEffect(() => {
    if (!open) return;
    setError('');
    if (templates.length && !templateId) {
      setTemplateId(String(templates[0].id || ''));
    }
  }, [open, templates, templateId]);

  const toggleMember = (id) => {
    const sid = String(id);
    if (sid === String(ceoId)) return;
    setMemberIds((prev) => {
      const s = new Set(prev.map(String));
      if (s.has(sid)) return prev.filter((x) => String(x) !== sid);
      return [...prev, sid];
    });
  };

  const handleCreate = async () => {
    setError('');
    if (!templateId || !name.trim() || !ceoId) {
      setError('Template, line name, and CEO are required.');
      return;
    }
    setSubmitting(true);
    try {
      const body = {
        template_id: templateId,
        name: name.trim(),
        ceo_agent_profile_id: String(ceoId),
        handle: handle.trim() || undefined,
        member_agent_profile_ids: memberIds.length ? memberIds.map(String) : undefined,
      };
      const line = await apiService.agentFactory.createLineFromTemplate(body);
      onSuccess?.(line);
      onClose?.();
      setName('');
      setHandle('');
      setCeoId('');
      setMemberIds([]);
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || 'Failed to create line');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onClose={() => !submitting && onClose?.()} maxWidth="sm" fullWidth>
      <DialogTitle>Line from template</DialogTitle>
      <DialogContent dividers>
        {templatesLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 3 }}>
            <CircularProgress />
          </Box>
        ) : (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 0.5 }}>
            <FormControl fullWidth size="small">
              <InputLabel id="tpl-label">Template</InputLabel>
              <Select
                labelId="tpl-label"
                label="Template"
                value={templateId}
                onChange={(e) => setTemplateId(e.target.value)}
              >
                {templates.map((t) => (
                  <MenuItem key={t.id} value={t.id}>
                    {t.title || t.id}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            {templateId && (
              <Typography variant="body2" color="text.secondary">
                {(templates.find((x) => x.id === templateId) || {}).description || ''}
              </Typography>
            )}
            <TextField
              label="Line name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              fullWidth
              size="small"
            />
            <TextField
              label="Handle (optional)"
              value={handle}
              onChange={(e) => setHandle(e.target.value)}
              fullWidth
              size="small"
              placeholder="e.g. markets-brief"
            />
            <FormControl fullWidth size="small">
              <InputLabel id="ceo-label">CEO agent</InputLabel>
              <Select
                labelId="ceo-label"
                label="CEO agent"
                value={ceoId}
                onChange={(e) => setCeoId(e.target.value)}
                required
              >
                {profiles.map((p) => (
                  <MenuItem key={p.id} value={String(p.id)}>
                    {p.name || p.handle || p.id}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <Typography variant="subtitle2" color="text.secondary">
              Optional workers (report to CEO)
            </Typography>
            <Box sx={{ maxHeight: 200, overflowY: 'auto', border: 1, borderColor: 'divider', borderRadius: 1, p: 1 }}>
              {profiles.map((p) => {
                const sid = String(p.id);
                if (sid === String(ceoId)) return null;
                return (
                  <FormControlLabel
                    key={p.id}
                    control={
                      <Checkbox
                        size="small"
                        checked={memberIds.map(String).includes(sid)}
                        onChange={() => toggleMember(p.id)}
                      />
                    }
                    label={p.name || p.handle || sid}
                  />
                );
              })}
            </Box>
            {error && (
              <Typography color="error" variant="body2">
                {error}
              </Typography>
            )}
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={() => onClose?.()} disabled={submitting}>
          Cancel
        </Button>
        <Button variant="contained" onClick={handleCreate} disabled={submitting || templatesLoading}>
          {submitting ? 'Creating…' : 'Create line'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
