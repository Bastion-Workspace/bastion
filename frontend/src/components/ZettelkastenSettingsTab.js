/**
 * Zettelkasten / PKM settings: daily markdown notes, wikilinks, discovery toggles.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Switch,
  FormControlLabel,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack,
} from '@mui/material';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import apiService from '../services/apiService';
import JournalLocationPicker from './JournalLocationPicker';

const ZettelkastenSettingsTab = () => {
  const navigate = useNavigate();
  const [settings, setSettings] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [dailyLoading, setDailyLoading] = useState(false);

  const loadSettings = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.get('/api/zettelkasten/settings');
      if (response?.success && response.settings) {
        setSettings(response.settings);
      } else {
        setError('Failed to load Zettelkasten settings');
      }
    } catch (err) {
      console.error(err);
      setError(err.message || 'Failed to load settings');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  const savePartial = async (partial) => {
    try {
      setSaving(true);
      setError(null);
      setSuccess(null);
      const response = await apiService.put('/api/zettelkasten/settings', partial);
      if (response?.success && response.settings) {
        setSettings(response.settings);
        setSuccess('Saved');
        setTimeout(() => setSuccess(null), 2500);
      } else {
        setError('Save failed');
      }
    } catch (err) {
      console.error(err);
      setError(err.message || 'Save failed');
    } finally {
      setSaving(false);
    }
  };

  const openDailyNote = async () => {
    try {
      setDailyLoading(true);
      const r = await apiService.get('/api/zettelkasten/daily-note');
      if (r?.success && r.document_id) {
        navigate(`/documents?document=${encodeURIComponent(r.document_id)}`);
      } else {
        setError(r?.error || 'Could not open daily note');
      }
    } catch (err) {
      setError(err.message || 'Daily note failed');
    } finally {
      setDailyLoading(false);
    }
  };

  if (loading && !settings) {
    return (
      <Box display="flex" justifyContent="center" p={4}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Zettelkasten
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Markdown daily notes, [[wikilinks]], and backlinks (independent of Org-mode journal).
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  General
                </Typography>
                <FormControlLabel
                  control={(
                    <Switch
                      checked={!!settings?.enabled}
                      onChange={(e) => savePartial({ enabled: e.target.checked })}
                      disabled={saving}
                    />
                  )}
                  label="Enable Zettelkasten features"
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12}>
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.03 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Daily note (markdown)
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  New files are created under the folder you pick. This is separate from the Org-mode journal.
                </Typography>
                <JournalLocationPicker
                  locationsUrl="/api/zettelkasten/settings/locations"
                  value={settings?.daily_note_location || ''}
                  onChange={(v) => savePartial({ daily_note_location: v || null })}
                  disabled={saving}
                  error={null}
                />
                <FormControl fullWidth margin="normal" sx={{ mt: 2 }}>
                  <InputLabel>Daily note filename</InputLabel>
                  <Select
                    label="Daily note filename"
                    value={settings?.daily_note_format || 'YYYY-MM-DD'}
                    onChange={(e) => savePartial({ daily_note_format: e.target.value })}
                    disabled={saving}
                  >
                    <MenuItem value="YYYY-MM-DD">YYYY-MM-DD.md</MenuItem>
                    <MenuItem value="YYYY-MM-DD-dddd">YYYY-MM-DD-Weekday.md</MenuItem>
                    <MenuItem value="YYYYMMDD">YYYYMMDD.md</MenuItem>
                  </Select>
                </FormControl>
                <TextField
                  fullWidth
                  multiline
                  minRows={3}
                  label="Template for new daily notes (optional)"
                  value={settings?.daily_note_template || ''}
                  onChange={(e) => setSettings((s) => ({ ...s, daily_note_template: e.target.value }))}
                  onBlur={(e) => savePartial({ daily_note_template: e.target.value })}
                  margin="normal"
                  helperText="Markdown body only; a default heading is used if empty."
                  disabled={saving}
                />
                <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
                  <Button
                    variant="contained"
                    onClick={openDailyNote}
                    disabled={dailyLoading}
                  >
                    {dailyLoading ? <CircularProgress size={22} /> : "Open today's daily note"}
                  </Button>
                </Stack>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12}>
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.06 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Authoring
                </Typography>
                <FormControlLabel
                  control={(
                    <Switch
                      checked={settings?.wikilink_autocomplete !== false}
                      onChange={(e) => savePartial({ wikilink_autocomplete: e.target.checked })}
                      disabled={saving}
                    />
                  )}
                  label="Wikilink autocomplete after [["
                />
                <FormControlLabel
                  control={(
                    <Switch
                      checked={!!settings?.note_id_prefix}
                      onChange={(e) => savePartial({ note_id_prefix: e.target.checked })}
                      disabled={saving}
                    />
                  )}
                  label="Timestamp prefix for new notes created from wikilinks"
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12}>
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.09 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Discovery
                </Typography>
                <FormControlLabel
                  control={(
                    <Switch
                      checked={settings?.backlinks_enabled !== false}
                      onChange={(e) => savePartial({ backlinks_enabled: e.target.checked })}
                      disabled={saving}
                    />
                  )}
                  label="Show backlinks / unlinked mentions in document viewer"
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12}>
          <Button variant="outlined" color="warning" onClick={async () => {
            await apiService.delete('/api/zettelkasten/settings');
            await loadSettings();
          }}
          >
            Reset to defaults
          </Button>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ZettelkastenSettingsTab;
