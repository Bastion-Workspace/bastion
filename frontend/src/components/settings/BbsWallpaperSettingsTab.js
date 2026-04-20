import React, { useState, useCallback, useEffect } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Divider,
  FormControl,
  FormControlLabel,
  FormLabel,
  InputLabel,
  MenuItem,
  Radio,
  RadioGroup,
  Select,
  Slider,
  Snackbar,
  Switch,
  TextField,
  Typography,
} from '@mui/material';
import { Settings, Add, Delete, CloudUpload } from '@mui/icons-material';
import { useMutation, useQuery } from 'react-query';
import apiService from '../../services/apiService';
import { useAuth } from '../../contexts/AuthContext';

/** Must match `backend/utils/bbs_builtin_wallpaper_animations` sentinels. */
const BUILTIN_ANIM_MATRIX_RAIN = '__bastion_builtin_matrix_rain__';
const BUILTIN_ANIM_SNOWMAN = '__bastion_builtin_snowman__';

const SAMPLE_WALLPAPER = [
  '   ____',
  '  /    \\',
  ' |  o o |',
  ' |   ^  |',
  '  \\____/',
  '   Bastion BBS',
].join('\n');

const emptyConfig = () => ({
  version: 1,
  display_mode: 'static',
  active_id: '',
  cycle_interval_seconds: 30,
  items: [],
  animation_document_id: '',
  animation_fps: 12,
  animation_loop: true,
});

function normalizeConfig(raw) {
  if (!raw || typeof raw !== 'object') return emptyConfig();
  const items = Array.isArray(raw.items) ? raw.items : [];
  let active_id = raw.active_id || '';
  if (items.length && !items.some((it) => it.id === active_id)) {
    active_id = items[0].id || '';
  }
  const dm =
    raw.display_mode === 'cycle'
      ? 'cycle'
      : raw.display_mode === 'animated'
        ? 'animated'
        : 'static';
  return {
    version: raw.version ?? 1,
    display_mode: dm,
    active_id,
    cycle_interval_seconds: Math.min(
      86400,
      Math.max(5, Number(raw.cycle_interval_seconds) || 30)
    ),
    items: items.map((it) => ({
      id: it.id || crypto.randomUUID(),
      name: it.name ?? 'Untitled',
      content: it.content ?? '',
      enabled: it.enabled !== false,
    })),
    animation_document_id: raw.animation_document_id || '',
    animation_fps: Math.min(30, Math.max(1, Number(raw.animation_fps) || 12)),
    animation_loop: raw.animation_loop !== false,
  };
}

const BbsWallpaperSettingsTab = () => {
  const { user } = useAuth();
  const fileInputRef = React.useRef(null);
  const [config, setConfig] = useState(emptyConfig);
  const [selectedEditId, setSelectedEditId] = useState('');
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [uploadingAnim, setUploadingAnim] = useState(false);

  const { data, isLoading, refetch } = useQuery(
    'userBbsWallpaper',
    () => apiService.settings.getUserBbsWallpaper(),
    {
      refetchOnWindowFocus: false,
      staleTime: 60_000,
      onError: () => {
        setSnackbar({ open: true, message: 'Failed to load wallpaper settings', severity: 'error' });
      },
    }
  );

  useEffect(() => {
    if (data && typeof data.config === 'object') {
      setConfig(normalizeConfig(data.config));
    }
  }, [data]);

  useEffect(() => {
    setSelectedEditId((prev) => {
      const ids = config.items.map((i) => i.id);
      if (ids.length === 0) return '';
      if (prev && ids.includes(prev)) return prev;
      return config.active_id && ids.includes(config.active_id)
        ? config.active_id
        : ids[0];
    });
  }, [config.items, config.active_id]);

  const saveMutation = useMutation(
    (cfg) => apiService.settings.setUserBbsWallpaper({ config: cfg }),
    {
      onSuccess: () => {
        setSnackbar({ open: true, message: 'Wallpaper saved', severity: 'success' });
        refetch();
      },
      onError: (error) => {
        const msg =
          error.response?.data?.detail ||
          error.message ||
          'Failed to save wallpaper';
        setSnackbar({ open: true, message: typeof msg === 'string' ? msg : 'Save failed', severity: 'error' });
      },
    }
  );

  const selectedItem = config.items.find((it) => it.id === selectedEditId) || null;

  const updateItem = useCallback((id, patch) => {
    setConfig((c) => ({
      ...c,
      items: c.items.map((it) => (it.id === id ? { ...it, ...patch } : it)),
    }));
  }, []);

  const addItem = () => {
    const id = crypto.randomUUID();
    setConfig((c) => ({
      ...c,
      items: [
        ...c.items,
        { id, name: 'Untitled', content: '', enabled: true },
      ],
      active_id: c.items.length === 0 ? id : c.active_id,
    }));
    setSelectedEditId(id);
  };

  const removeSelectedItem = () => {
    if (!selectedEditId) return;
    setConfig((c) => {
      const items = c.items.filter((it) => it.id !== selectedEditId);
      let active_id = c.active_id;
      if (active_id === selectedEditId) {
        active_id = items[0]?.id || '';
      }
      return { ...c, items, active_id };
    });
  };

  const loadSample = () => {
    const id = crypto.randomUUID();
    setConfig((c) => ({
      ...c,
      display_mode: 'static',
      active_id: id,
      items: [
        {
          id,
          name: 'Sample',
          content: SAMPLE_WALLPAPER,
          enabled: true,
        },
      ],
    }));
    setSelectedEditId(id);
  };

  const handleSave = () => {
    saveMutation.mutate(config);
  };

  const handleAnimationUpload = async (event) => {
    const file = event.target.files && event.target.files[0];
    if (event.target) {
      event.target.value = '';
    }
    if (!file) return;
    const uid = user?.user_id ?? user?.id ?? null;
    if (!uid) {
      setSnackbar({ open: true, message: 'Not signed in', severity: 'error' });
      return;
    }
    setUploadingAnim(true);
    try {
      const res = await apiService.uploadUserDocument(file, uid);
      const docId = res?.document_id;
      if (!docId) {
        throw new Error('Upload did not return document_id');
      }
      setConfig((c) => ({
        ...c,
        display_mode: 'animated',
        animation_document_id: docId,
      }));
      setSnackbar({
        open: true,
        message: 'Animation file uploaded (Animated mode on). Adjust FPS/loop, then Save.',
        severity: 'success',
      });
    } catch (e) {
      setSnackbar({
        open: true,
        message: e instanceof Error ? e.message : 'Upload failed',
        severity: 'error',
      });
    } finally {
      setUploadingAnim(false);
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Wallpaper
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          ASCII art for the BBS full-screen Wallpaper menu (W). Static uses one active piece; Cycle
          rotates enabled pieces on the server; Animated plays an uploaded file (ASCII Motion JSON
          export, or plain text with frames separated by a line containing only a comma). The BBS
          polls for static/cycle about every 5 seconds; animated mode plays frames locally at your
          FPS setting.
        </Typography>

        {isLoading && (
          <Box display="flex" alignItems="center" gap={2} sx={{ mb: 2 }}>
            <CircularProgress size={24} />
            <Typography variant="body2">Loading…</Typography>
          </Box>
        )}

        <FormControl component="fieldset" sx={{ mb: 2 }}>
          <FormLabel component="legend">Display mode</FormLabel>
          <RadioGroup
            row
            value={config.display_mode}
            onChange={(e) =>
              setConfig((c) => ({ ...c, display_mode: e.target.value }))
            }
          >
            <FormControlLabel value="static" control={<Radio />} label="Static (active item)" />
            <FormControlLabel value="cycle" control={<Radio />} label="Cycle (enabled items)" />
            <FormControlLabel value="animated" control={<Radio />} label="Animated (upload)" />
          </RadioGroup>
        </FormControl>

        {config.display_mode === 'animated' && (
          <Box
            sx={{
              mb: 2,
              p: 2,
              border: 1,
              borderColor: 'divider',
              borderRadius: 1,
            }}
          >
            <Typography variant="subtitle2" gutterBottom>
              Animated wallpaper file
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
              Upload <code>.json</code> (ASCII Motion export) or <code>.txt</code> (frames split by
              a line that contains only a comma). Document must stay text-backed in Bastion. Or pick
              a built-in screensaver below (no upload).
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={1} sx={{ mb: 2 }}>
              <Button
                variant="outlined"
                color="secondary"
                onClick={() =>
                  setConfig((c) => ({
                    ...c,
                    display_mode: 'animated',
                    animation_document_id: BUILTIN_ANIM_MATRIX_RAIN,
                  }))
                }
              >
                Matrix rain
              </Button>
              <Button
                variant="outlined"
                color="secondary"
                onClick={() =>
                  setConfig((c) => ({
                    ...c,
                    display_mode: 'animated',
                    animation_document_id: BUILTIN_ANIM_SNOWMAN,
                  }))
                }
              >
                Winter snowman
              </Button>
            </Box>
            <input
              type="file"
              ref={fileInputRef}
              hidden
              accept=".json,.txt,.text"
              onChange={handleAnimationUpload}
            />
            <Box display="flex" flexWrap="wrap" gap={1} alignItems="center" sx={{ mb: 2 }}>
              <Button
                variant="outlined"
                startIcon={uploadingAnim ? <CircularProgress size={18} color="inherit" /> : <CloudUpload />}
                disabled={uploadingAnim}
                onClick={() => fileInputRef.current && fileInputRef.current.click()}
              >
                {uploadingAnim ? 'Uploading…' : 'Upload animation'}
              </Button>
              <Button
                variant="text"
                color="warning"
                size="small"
                disabled={!config.animation_document_id}
                onClick={() =>
                  setConfig((c) => ({
                    ...c,
                    animation_document_id: '',
                  }))
                }
              >
                Clear document
              </Button>
            </Box>
            {config.animation_document_id ? (
              <Typography variant="body2" sx={{ fontFamily: 'monospace', mb: 2 }} noWrap title={config.animation_document_id}>
                Animation: {config.animation_document_id}
              </Typography>
            ) : (
              <Alert severity="warning" sx={{ mb: 2 }}>
                Select Animated mode, then upload a file or a built-in screensaver, then Save. The
                server needs an animation document id or built-in id.
              </Alert>
            )}
            <Typography gutterBottom>FPS (BBS playback)</Typography>
            <Slider
              value={config.animation_fps}
              onChange={(_, v) =>
                setConfig((c) => ({
                  ...c,
                  animation_fps: Array.isArray(v) ? v[0] : v,
                }))
              }
              min={1}
              max={30}
              step={1}
              marks
              valueLabelDisplay="auto"
              sx={{ maxWidth: 360, mb: 2 }}
            />
            <FormControlLabel
              control={
                <Switch
                  checked={config.animation_loop}
                  onChange={(e) =>
                    setConfig((c) => ({ ...c, animation_loop: e.target.checked }))
                  }
                />
              }
              label="Loop animation"
            />
          </Box>
        )}

        <TextField
          label="Cycle interval (seconds)"
          type="number"
          size="small"
          value={config.cycle_interval_seconds}
          onChange={(e) =>
            setConfig((c) => ({
              ...c,
              cycle_interval_seconds: Math.min(
                86400,
                Math.max(5, parseInt(e.target.value, 10) || 5)
              ),
            }))
          }
          inputProps={{ min: 5, max: 86400 }}
          sx={{ mb: 2, maxWidth: 280 }}
          helperText="Used when Cycle mode is on (5–86400)."
          disabled={config.display_mode === 'animated'}
        />

        <Box display="flex" flexWrap="wrap" gap={1} alignItems="center" sx={{ mb: 2 }}>
          <Button variant="outlined" startIcon={<Add />} onClick={addItem}>
            New wallpaper
          </Button>
          <Button
            variant="outlined"
            color="error"
            startIcon={<Delete />}
            onClick={removeSelectedItem}
            disabled={!selectedItem}
          >
            Delete selected
          </Button>
          <Button variant="outlined" onClick={loadSample}>
            Load sample
          </Button>
          <Button
            variant="contained"
            onClick={handleSave}
            disabled={saveMutation.isLoading}
            startIcon={saveMutation.isLoading ? <CircularProgress size={20} /> : <Settings />}
          >
            {saveMutation.isLoading ? 'Saving…' : 'Save'}
          </Button>
        </Box>

        {config.items.length === 0 ? (
          <Alert severity="info" sx={{ mb: 2 }}>
            No wallpapers yet. Use &quot;New wallpaper&quot; or &quot;Load sample&quot;, then Save.
          </Alert>
        ) : (
          <>
            <FormControl fullWidth sx={{ mb: 2 }} size="small">
              <InputLabel id="bbs-wallpaper-select-label">Wallpaper to edit</InputLabel>
              <Select
                labelId="bbs-wallpaper-select-label"
                id="bbs-wallpaper-select"
                label="Wallpaper to edit"
                value={selectedEditId}
                onChange={(e) => setSelectedEditId(e.target.value)}
              >
                {config.items.map((it) => (
                  <MenuItem key={it.id} value={it.id}>
                    {it.name?.trim() || 'Untitled'}
                    {config.active_id === it.id && config.display_mode === 'static' ? ' (active for BBS)' : ''}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {selectedItem && (
              <Box
                sx={{
                  border: 1,
                  borderColor: 'divider',
                  borderRadius: 1,
                  p: 2,
                }}
              >
                <Box display="flex" flexWrap="wrap" alignItems="center" gap={2} sx={{ mb: 2 }}>
                  <TextField
                    label="Name"
                    size="small"
                    value={selectedItem.name}
                    onChange={(e) => updateItem(selectedItem.id, { name: e.target.value })}
                    sx={{ minWidth: 220, flex: '1 1 200px' }}
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={selectedItem.enabled}
                        onChange={(e) =>
                          updateItem(selectedItem.id, { enabled: e.target.checked })
                        }
                      />
                    }
                    label="Enabled for cycle"
                  />
                  {config.display_mode === 'static' && (
                    <Button
                      size="small"
                      variant={config.active_id === selectedItem.id ? 'contained' : 'outlined'}
                      onClick={() => setConfig((c) => ({ ...c, active_id: selectedItem.id }))}
                    >
                      {config.active_id === selectedItem.id
                        ? 'Active for BBS'
                        : 'Set as active for BBS'}
                    </Button>
                  )}
                </Box>
                <TextField
                  fullWidth
                  multiline
                  rows={16}
                  label="ASCII art"
                  value={selectedItem.content}
                  onChange={(e) =>
                    updateItem(selectedItem.id, { content: e.target.value })
                  }
                  inputProps={{ maxLength: 16384, spellCheck: false }}
                  InputProps={{
                    sx: {
                      fontFamily: 'ui-monospace, monospace',
                      '& textarea': { whiteSpace: 'pre' },
                    },
                  }}
                />
              </Box>
            )}
          </>
        )}

        <Divider sx={{ my: 2 }} />

        <Snackbar
          open={snackbar.open}
          autoHideDuration={6000}
          onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert
            onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
            severity={snackbar.severity}
            sx={{ width: '100%' }}
          >
            {snackbar.message}
          </Alert>
        </Snackbar>
      </CardContent>
    </Card>
  );
};

export default BbsWallpaperSettingsTab;
