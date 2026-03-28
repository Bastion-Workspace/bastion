import React, { useEffect, useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  Button,
  IconButton,
  CircularProgress,
  Snackbar,
  Alert,
  Collapse,
} from '@mui/material';
import { PlayArrow, Pause, Stop, SkipNext, SkipPrevious, Close, VolumeUp, VolumeOff, Refresh, PowerSettingsNew, Lightbulb, Thermostat, ExpandMore, ExpandLess } from '@mui/icons-material';
import { useControlPanes } from '../../contexts/ControlPaneContext';
import apiService from '../../services/apiService';

function getByPath(obj, path) {
  if (!obj || !path) return undefined;
  const parts = String(path).split('.');
  let current = obj;
  for (const key of parts) {
    if (current == null) return undefined;
    current = current[key];
  }
  return current;
}

function buildParams(control, baseParams, state) {
  const extra = {};
  for (const src of control.param_source || []) {
    if (src.param && src.from_control_id) {
      const v = state[src.from_control_id];
      if (v !== undefined && v !== null && v !== '') {
        extra[src.param] = v;
      }
    }
  }
  return { ...extra, ...baseParams };
}

const ICON_MAP = {
  PlayArrow,
  Pause,
  Stop,
  SkipNext,
  SkipPrevious,
  Close,
  VolumeUp,
  VolumeOff,
  Refresh,
  PowerSettingsNew,
  Lightbulb,
  Thermostat,
};

function getIconComponent(iconName) {
  if (!iconName) return PlayArrow;
  return ICON_MAP[iconName] || PlayArrow;
}

const ControlPanePopover = ({ pane, onClose }) => {
  const { paneStates, executeAction, refreshPaneState, setPaneControlValue } = useControlPanes();
  const [loading, setLoading] = useState({});
  const [optionsCache, setOptionsCache] = useState({});
  const [error, setError] = useState(null);
  const [lastResult, setLastResult] = useState(null);
  const [resultPreviewOpen, setResultPreviewOpen] = useState(true);
  const paneId = pane?.id;
  const controls = pane?.controls || [];
  const refreshInterval = Math.max(0, Number(pane?.refresh_interval) || 0);
  const state = paneId ? (paneStates[paneId] || {}) : {};

  useEffect(() => {
    if (paneId && controls.some((c) => c.refresh_endpoint_id)) {
      refreshPaneState(paneId);
    }
  }, [paneId, refreshPaneState]);

  useEffect(() => {
    if (!paneId || refreshInterval <= 0) return;
    const intervalId = setInterval(() => {
      refreshPaneState(paneId);
    }, refreshInterval * 1000);
    return () => clearInterval(intervalId);
  }, [paneId, refreshInterval, refreshPaneState]);

  useEffect(() => {
    if (!error) return;
    const t = setTimeout(() => setError(null), 4000);
    return () => clearTimeout(t);
  }, [error]);

  useEffect(() => {
    if (!lastResult) return;
    const t = setTimeout(() => setLastResult(null), 5000);
    return () => clearTimeout(t);
  }, [lastResult]);

  const setLoadingFor = useCallback((controlId, value) => {
    setLoading((prev) => ({ ...prev, [controlId]: value }));
  }, []);

  const handleSliderDrag = useCallback(
    (control, _e, value) => {
      setPaneControlValue(paneId, control.id, value);
    },
    [paneId, setPaneControlValue]
  );

  const handleSliderCommit = useCallback(
    async (control, value) => {
      setLoadingFor(control.id, true);
      try {
        const params = buildParams(control, { [control.param_key]: value }, state);
        const result = await executeAction(paneId, control.endpoint_id, params);
        setError(null);
        setLastResult({ controlId: control.id, data: result, timestamp: Date.now() });
        await refreshPaneState(paneId);
      } catch (e) {
        setError(`Failed to execute ${control.label}`);
      } finally {
        setLoadingFor(control.id, false);
      }
    },
    [paneId, state, executeAction, refreshPaneState, setLoadingFor]
  );

  const handleToggleChange = useCallback(
    async (control, e) => {
      const value = e.target.checked;
      setPaneControlValue(paneId, control.id, value);
      setLoadingFor(control.id, true);
      try {
        const params = buildParams(control, { [control.param_key]: value }, state);
        const result = await executeAction(paneId, control.endpoint_id, params);
        setError(null);
        setLastResult({ controlId: control.id, data: result, timestamp: Date.now() });
        await refreshPaneState(paneId);
      } catch (e) {
        setError(`Failed to execute ${control.label}`);
      } finally {
        setLoadingFor(control.id, false);
      }
    },
    [paneId, state, executeAction, setPaneControlValue, refreshPaneState, setLoadingFor]
  );

  const handleDropdownChange = useCallback(
    async (control, value) => {
      setPaneControlValue(paneId, control.id, value);
      setLoadingFor(control.id, true);
      try {
        const params = buildParams(control, { [control.param_key]: value }, state);
        const result = await executeAction(paneId, control.endpoint_id, params);
        setError(null);
        setLastResult({ controlId: control.id, data: result, timestamp: Date.now() });
        await refreshPaneState(paneId);
      } catch (e) {
        setError(`Failed to execute ${control.label}`);
      } finally {
        setLoadingFor(control.id, false);
      }
    },
    [paneId, state, executeAction, setPaneControlValue, refreshPaneState, setLoadingFor]
  );

  const handleButtonClick = useCallback(
    async (control) => {
      setLoadingFor(control.id, true);
      try {
        const params = buildParams(control, {}, state);
        const result = await executeAction(paneId, control.endpoint_id, params);
        setError(null);
        setLastResult({ controlId: control.id, data: result, timestamp: Date.now() });
        await refreshPaneState(paneId);
      } catch (e) {
        setError(`Failed to execute ${control.label}`);
      } finally {
        setLoadingFor(control.id, false);
      }
    },
    [paneId, state, executeAction, refreshPaneState, setLoadingFor]
  );

  const loadOptions = useCallback(
    async (control) => {
      const key = `${paneId}-${control.id}`;
      if (optionsCache[key]) return optionsCache[key];
      try {
        const res = await apiService.controlPanes.executeAction(
          paneId,
          control.options_endpoint_id,
          {}
        );
        const raw = res?.raw_response ?? res?.records ?? res;
        const list = Array.isArray(raw) ? raw : raw?.result != null ? (Array.isArray(raw.result) ? raw.result : [raw.result]) : [];
        const labelPath = control.options_label_path || 'name';
        const valuePath = control.options_value_path || 'id';
        const options = list.map((item) => ({
          label: getByPath(item, labelPath) ?? String(item),
          value: getByPath(item, valuePath) ?? item,
        }));
        setOptionsCache((prev) => ({ ...prev, [key]: options }));
        return options;
      } catch (e) {
        return [];
      }
    },
    [paneId, optionsCache]
  );

  if (!pane) return null;

  return (
    <Box sx={{ minWidth: 220, maxWidth: 320, p: 1.5 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="subtitle2">{pane.name}</Typography>
        <IconButton size="small" onClick={onClose} aria-label="Close">
          <Close fontSize="small" />
        </IconButton>
      </Box>
      {controls.map((control) => {
        const isLoading = loading[control.id];
        const value = state[control.id];

        if (control.type === 'slider') {
          const min = control.min ?? 0;
          const max = control.max ?? 100;
          const step = control.step ?? 1;
          const numValue = typeof value === 'number' ? value : Number(value);
          const safeValue = Number.isFinite(numValue) ? Math.min(max, Math.max(min, numValue)) : min;
          return (
            <Box key={control.id} sx={{ mb: 1.5 }}>
              <Typography variant="caption" color="text.secondary">
                {control.label}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Slider
                  size="small"
                  value={safeValue}
                  min={min}
                  max={max}
                  step={step}
                  onChange={(e, v) => handleSliderDrag(control, e, v)}
                  onChangeCommitted={(e, v) => handleSliderCommit(control, v)}
                  disabled={isLoading}
                  sx={{ flex: 1 }}
                />
                {isLoading && <CircularProgress size={16} />}
              </Box>
            </Box>
          );
        }

        if (control.type === 'toggle') {
          const checked = value === true || value === 'true' || value === 1;
          return (
            <Box key={control.id} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="body2">{control.label}</Typography>
              <Switch
                size="small"
                checked={checked}
                onChange={(e) => handleToggleChange(control, e)}
                disabled={isLoading}
              />
              {isLoading && <CircularProgress size={14} sx={{ ml: 0.5 }} />}
            </Box>
          );
        }

        if (control.type === 'dropdown') {
          return (
            <DropdownControl
              key={control.id}
              control={control}
              value={value}
              loading={isLoading}
              paneId={paneId}
              loadOptions={loadOptions}
              onChange={handleDropdownChange}
            />
          );
        }

        if (control.type === 'button') {
          const IconComponent = getIconComponent(control.icon);
          return (
            <Box key={control.id} sx={{ mb: 1 }}>
              <Button
                size="small"
                variant="outlined"
                fullWidth
                startIcon={isLoading ? <CircularProgress size={14} /> : <IconComponent />}
                onClick={() => handleButtonClick(control)}
                disabled={isLoading}
              >
                {control.label}
              </Button>
            </Box>
          );
        }

        if (control.type === 'text_display') {
          return (
            <Box key={control.id} sx={{ mb: 1 }}>
              <Typography variant="caption" color="text.secondary">
                {control.label}
              </Typography>
              <Typography variant="body2" sx={{ wordBreak: 'break-word' }}>
                {value != null && value !== '' ? String(value) : '—'}
              </Typography>
            </Box>
          );
        }

        return null;
      })}
      {controls.length === 0 && (
        <Typography variant="body2" color="text.secondary">
          No controls configured.
        </Typography>
      )}
      {lastResult && lastResult.data && (
        <Box sx={{ mt: 1.5 }}>
          <Button
            size="small"
            fullWidth
            onClick={() => setResultPreviewOpen((o) => !o)}
            endIcon={resultPreviewOpen ? <ExpandLess /> : <ExpandMore />}
            sx={{ justifyContent: 'space-between', textTransform: 'none' }}
          >
            <Typography variant="caption" color="text.secondary">
              Last response
            </Typography>
          </Button>
          <Collapse in={resultPreviewOpen}>
            <Box sx={{ bgcolor: 'action.hover', borderRadius: 1, p: 1, mt: 0.5 }}>
              {typeof lastResult.data.formatted === 'string' ? (
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                  {lastResult.data.formatted}
                </Typography>
              ) : typeof lastResult.data === 'object' ? (
                <Typography variant="body2" component="pre" sx={{ fontSize: '0.75rem', overflow: 'auto', maxHeight: 80, m: 0 }}>
                  {JSON.stringify(lastResult.data, null, 2).slice(0, 500)}
                  {JSON.stringify(lastResult.data).length > 500 ? '…' : ''}
                </Typography>
              ) : (
                <Typography variant="body2">{String(lastResult.data)}</Typography>
              )}
            </Box>
          </Collapse>
        </Box>
      )}
      <Snackbar
        open={!!error}
        autoHideDuration={4000}
        onClose={() => setError(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      </Snackbar>
    </Box>
  );
};

function DropdownControl({ control, value, loading, paneId, loadOptions, onChange }) {
  const [options, setOptions] = useState([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (control.options && Array.isArray(control.options) && control.options.length > 0) {
      const opts = control.options.map((o) =>
        typeof o === 'string' ? { label: o, value: o } : o
      );
      setOptions(opts);
      setLoaded(true);
      return;
    }
    if (!control.options_endpoint_id) {
      setLoaded(true);
      return;
    }
    let cancelled = false;
    loadOptions(control).then((opts) => {
      if (!cancelled) {
        setOptions(opts || []);
        setLoaded(true);
      }
    });
    return () => { cancelled = true; };
  }, [control.options, control.options_endpoint_id, loadOptions]);

  return (
    <Box sx={{ mb: 1.5 }}>
      <FormControl size="small" fullWidth disabled={loading}>
        <InputLabel>{control.label}</InputLabel>
        <Select
          value={value ?? ''}
          label={control.label}
          onChange={(e) => onChange(control, e.target.value)}
          disabled={!loaded || loading}
        >
          {options.map((opt) => (
            <MenuItem key={String(opt.value)} value={opt.value}>
              {opt.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    </Box>
  );
}

export default ControlPanePopover;
