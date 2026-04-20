import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  CircularProgress,
  Alert,
} from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import apiService from '../../services/apiService';
import {
  defaultGridForWidgetType,
  newWidgetId,
} from '../homeDashboard/homeDashboardUtils';

/**
 * Append an artifact_embed widget to a dashboard layout.
 */
export default function PinToDashboardDialog({ open, onClose, artifactId }) {
  const [dashboards, setDashboards] = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [doneDashboardId, setDoneDashboardId] = useState(null);

  useEffect(() => {
    if (!open) {
      setDoneDashboardId(null);
      setError('');
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError('');
    apiService
      .listHomeDashboards()
      .then((res) => {
        if (cancelled) return;
        const list = res?.dashboards || [];
        setDashboards(list);
        const def = list.find((d) => d.is_default) || list[0];
        setSelectedId(def?.id || '');
      })
      .catch((e) => {
        if (!cancelled) setError(e?.message || 'Failed to load dashboards');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  const handlePin = async () => {
    if (!artifactId || !selectedId) return;
    setSaving(true);
    setError('');
    try {
      const layout = await apiService.getHomeDashboardLayout(selectedId);
      const grid = defaultGridForWidgetType('artifact_embed');
      const widget = {
        type: 'artifact_embed',
        id: newWidgetId(),
        config: { artifact_id: artifactId },
        grid,
      };
      const next = {
        schema_version: layout.schema_version ?? 1,
        layout_mode: layout.layout_mode || 'stack',
        widgets: [...(layout.widgets || []), widget],
      };
      await apiService.putHomeDashboardLayout(selectedId, next);
      setDoneDashboardId(selectedId);
    } catch (e) {
      const d = e?.response?.data?.detail;
      setError(typeof d === 'string' ? d : e?.message || 'Save failed');
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Pin to dashboard</DialogTitle>
      <DialogContent>
        {loading ? (
          <CircularProgress size={28} sx={{ display: 'block', mx: 'auto', my: 2 }} />
        ) : null}
        {error ? (
          <Alert severity="error" sx={{ mb: 1 }}>
            {error}
          </Alert>
        ) : null}
        {doneDashboardId ? (
          <Typography variant="body2" sx={{ mb: 1 }}>
            Added to your dashboard.{' '}
            <Button component={RouterLink} to={`/home/${doneDashboardId}`} size="small">
              Open dashboard
            </Button>
          </Typography>
        ) : (
          <>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Choose where to add this saved artifact as a widget.
            </Typography>
            <FormControl size="small" fullWidth sx={{ mt: 1 }}>
              <InputLabel>Dashboard</InputLabel>
              <Select
                label="Dashboard"
                value={selectedId}
                onChange={(e) => setSelectedId(e.target.value)}
              >
                {dashboards.map((d) => (
                  <MenuItem key={d.id} value={d.id}>
                    {d.name}
                    {d.is_default ? ' (default)' : ''}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
        {!doneDashboardId ? (
          <Button
            variant="contained"
            onClick={() => void handlePin()}
            disabled={saving || !selectedId || !artifactId}
          >
            {saving ? 'Saving…' : 'Add widget'}
          </Button>
        ) : null}
      </DialogActions>
    </Dialog>
  );
}
