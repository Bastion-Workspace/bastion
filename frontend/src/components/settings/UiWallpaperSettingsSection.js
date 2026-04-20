import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  FormControl,
  FormControlLabel,
  FormLabel,
  Grid,
  InputLabel,
  MenuItem,
  Select,
  Slider,
  Snackbar,
  Switch,
  Typography,
} from '@mui/material';
import { Image as ImageIcon, Save, CloudUpload } from '@mui/icons-material';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';
import { useAuth } from '../../contexts/AuthContext';
import {
  UI_WALLPAPER_BUILTINS,
  UI_WALLPAPER_QUERY_KEY,
  getUiWallpaperBuiltinByKey,
  wallpaperPublicUrl,
} from '../../config/uiWallpaperBuiltins';

function emptyConfig() {
  return {
    version: 1,
    enabled: false,
    source: 'none',
    builtin_key: null,
    document_id: null,
    opacity: 0.62,
    scrim_opacity: 0.22,
    blur_px: 0,
    size: 'cover',
    repeat: 'no-repeat',
  };
}

function normalizeConfig(raw) {
  if (!raw || typeof raw !== 'object') return emptyConfig();
  const size = raw.size === 'contain' || raw.size === 'auto' ? raw.size : 'cover';
  const repeat = raw.repeat === 'repeat' ? 'repeat' : 'no-repeat';
  const src = raw.source === 'builtin' || raw.source === 'document' ? raw.source : 'none';
  return {
    version: raw.version ?? 1,
    enabled: raw.enabled === true,
    source: src,
    builtin_key: raw.builtin_key || null,
    document_id: raw.document_id || null,
    opacity: Math.min(1, Math.max(0, Number(raw.opacity) || 0.62)),
    scrim_opacity: Math.min(1, Math.max(0, Number(raw.scrim_opacity) || 0.22)),
    blur_px: Math.min(20, Math.max(0, Number(raw.blur_px) || 0)),
    size,
    repeat,
  };
}

const UiWallpaperSettingsSection = () => {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const fileInputRef = useRef(null);
  const [config, setConfig] = useState(emptyConfig);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [uploading, setUploading] = useState(false);

  const { data, isLoading, refetch } = useQuery(
    [UI_WALLPAPER_QUERY_KEY],
    () => apiService.settings.getUserUiWallpaper(),
    {
      refetchOnWindowFocus: false,
      staleTime: 30_000,
      onError: () => {
        setSnackbar({ open: true, message: 'Failed to load UI wallpaper settings', severity: 'error' });
      },
    }
  );

  useEffect(() => {
    if (data && data.config && typeof data.config === 'object') {
      setConfig(normalizeConfig(data.config));
    }
  }, [data]);

  const saveMutation = useMutation(
    (cfg) => apiService.settings.setUserUiWallpaper({ config: cfg }),
    {
      onSuccess: () => {
        setSnackbar({ open: true, message: 'Wallpaper saved', severity: 'success' });
        queryClient.invalidateQueries([UI_WALLPAPER_QUERY_KEY]);
        refetch();
      },
      onError: (error) => {
        const msg =
          error.response?.data?.detail ||
          error.message ||
          'Failed to save wallpaper';
        setSnackbar({
          open: true,
          message: typeof msg === 'string' ? msg : 'Save failed',
          severity: 'error',
        });
      },
    }
  );

  const handleSave = useCallback(() => {
    const payload = { ...config };
    if (!payload.enabled) {
      payload.source = 'none';
      payload.builtin_key = null;
      payload.document_id = null;
    } else if (payload.source === 'builtin') {
      payload.document_id = null;
      if (!payload.builtin_key && UI_WALLPAPER_BUILTINS[0]) {
        payload.builtin_key = UI_WALLPAPER_BUILTINS[0].key;
      }
    } else if (payload.source === 'document') {
      payload.builtin_key = null;
      if (!payload.document_id) {
        setSnackbar({
          open: true,
          message: 'Upload an image or choose a built-in wallpaper before saving.',
          severity: 'warning',
        });
        return;
      }
    }
    saveMutation.mutate(payload);
  }, [config, saveMutation]);

  const applyBuiltinDefaults = useCallback((key) => {
    const b = getUiWallpaperBuiltinByKey(key);
    setConfig((prev) => ({
      ...prev,
      enabled: true,
      source: 'builtin',
      builtin_key: key,
      document_id: null,
      repeat: b?.defaultRepeat || 'no-repeat',
      size: b?.defaultSize || 'cover',
    }));
  }, []);

  const onPickFile = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file || !file.type.startsWith('image/')) {
      setSnackbar({ open: true, message: 'Choose an image file', severity: 'warning' });
      return;
    }
    setUploading(true);
    try {
      const uid = user?.user_id ?? user?.id ?? null;
      const res = await apiService.uploadUserDocument(file, uid);
      const docId = res?.document_id;
      if (!docId) {
        throw new Error('Upload did not return document_id');
      }
      setConfig((prev) => ({
        ...prev,
        enabled: true,
        source: 'document',
        document_id: docId,
        builtin_key: null,
      }));
      setSnackbar({ open: true, message: 'Image uploaded. Click Save to apply.', severity: 'success' });
    } catch (err) {
      setSnackbar({
        open: true,
        message: err?.message || 'Upload failed',
        severity: 'error',
      });
    } finally {
      setUploading(false);
    }
  };

  const clearCustom = () => {
    setConfig((prev) => ({
      ...prev,
      document_id: null,
      source: prev.source === 'document' ? 'none' : prev.source,
      enabled: prev.source === 'document' ? false : prev.enabled,
    }));
  };

  if (isLoading && !data) {
    return (
      <Card>
        <CardContent sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" mb={2}>
          <ImageIcon sx={{ mr: 2, color: 'primary.main' }} />
          <Typography variant="h6">Web app wallpaper</Typography>
        </Box>
        <Typography variant="body2" color="text.secondary" paragraph>
          Background for the workspace (behind panes). Built-in tiles ship with the app; you can also upload your own
          image. Adjust opacity and scrim so content stays readable. BBS terminal ASCII wallpapers are configured below.
        </Typography>

        <FormControlLabel
          control={
            <Switch
              checked={config.enabled}
              onChange={(_, v) =>
                setConfig((c) => ({
                  ...c,
                  enabled: v,
                  source: v && c.source === 'none' ? 'builtin' : v ? c.source : 'none',
                  builtin_key: v && c.source === 'none' ? UI_WALLPAPER_BUILTINS[0]?.key || null : c.builtin_key,
                }))
              }
            />
          }
          label="Enable wallpaper"
        />

        {config.enabled && (
          <Box sx={{ mt: 2 }}>
            <FormLabel component="legend" sx={{ mb: 1, display: 'block' }}>
              Source
            </FormLabel>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Built-in
                </Typography>
                <Grid container spacing={1}>
                  {UI_WALLPAPER_BUILTINS.map((b) => (
                    <Grid item key={b.key}>
                      <Button
                        size="small"
                        variant={config.source === 'builtin' && config.builtin_key === b.key ? 'contained' : 'outlined'}
                        onClick={() => applyBuiltinDefaults(b.key)}
                        sx={{
                          width: 120,
                          height: 72,
                          backgroundImage: `url("${wallpaperPublicUrl(b.path)}")`,
                          backgroundSize: b.defaultSize || 'cover',
                          backgroundRepeat: b.defaultRepeat || 'no-repeat',
                          backgroundPosition: 'center',
                          border: 1,
                          borderColor: 'divider',
                        }}
                      >
                        <Typography
                          variant="caption"
                          sx={{
                            bgcolor: 'rgba(0,0,0,0.55)',
                            color: 'common.white',
                            px: 0.5,
                            borderRadius: 0.5,
                          }}
                        >
                          {b.label}
                        </Typography>
                      </Button>
                    </Grid>
                  ))}
                </Grid>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Custom image
                </Typography>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={onPickFile}
                />
                <Button
                  startIcon={uploading ? <CircularProgress size={18} color="inherit" /> : <CloudUpload />}
                  variant="outlined"
                  disabled={uploading}
                  onClick={() => fileInputRef.current?.click()}
                >
                  Upload image
                </Button>
                {config.source === 'document' && config.document_id && (
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="caption" color="text.secondary" display="block">
                      Document: {config.document_id}
                    </Typography>
                    <Button size="small" onClick={clearCustom}>
                      Clear custom image
                    </Button>
                  </Box>
                )}
              </Grid>
            </Grid>

            <Box sx={{ mt: 3 }}>
              <Typography gutterBottom>Wallpaper opacity</Typography>
              <Slider
                value={config.opacity}
                min={0.05}
                max={1}
                step={0.05}
                valueLabelDisplay="auto"
                onChange={(_, v) => setConfig((c) => ({ ...c, opacity: v }))}
              />
            </Box>
            <Box sx={{ mt: 2 }}>
              <Typography gutterBottom>Scrim (readability overlay)</Typography>
              <Slider
                value={config.scrim_opacity}
                min={0}
                max={1}
                step={0.05}
                valueLabelDisplay="auto"
                onChange={(_, v) => setConfig((c) => ({ ...c, scrim_opacity: v }))}
              />
            </Box>
            <Box sx={{ mt: 2 }}>
              <Typography gutterBottom>Blur (px)</Typography>
              <Slider
                value={config.blur_px}
                min={0}
                max={16}
                step={1}
                valueLabelDisplay="auto"
                onChange={(_, v) => setConfig((c) => ({ ...c, blur_px: v }))}
              />
            </Box>
            <Grid container spacing={2} sx={{ mt: 1 }}>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth size="small" margin="normal">
                  <InputLabel id="ui-wallpaper-size">Background size</InputLabel>
                  <Select
                    labelId="ui-wallpaper-size"
                    label="Background size"
                    value={config.size}
                    onChange={(e) => setConfig((c) => ({ ...c, size: e.target.value }))}
                  >
                    <MenuItem value="cover">Cover</MenuItem>
                    <MenuItem value="contain">Contain</MenuItem>
                    <MenuItem value="auto">Auto (good for tiles)</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth size="small" margin="normal">
                  <InputLabel id="ui-wallpaper-repeat">Repeat</InputLabel>
                  <Select
                    labelId="ui-wallpaper-repeat"
                    label="Repeat"
                    value={config.repeat}
                    onChange={(e) => setConfig((c) => ({ ...c, repeat: e.target.value }))}
                  >
                    <MenuItem value="no-repeat">No repeat</MenuItem>
                    <MenuItem value="repeat">Repeat (tile)</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Box>
        )}

        <Box sx={{ mt: 3, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            startIcon={saveMutation.isLoading ? <CircularProgress size={20} color="inherit" /> : <Save />}
            disabled={saveMutation.isLoading}
            onClick={handleSave}
          >
            Save
          </Button>
        </Box>

      </CardContent>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={5000}
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
    </Card>
  );
};

export default UiWallpaperSettingsSection;
