/**
 * Org-Mode Settings Tab
 * 
 * **BULLY!** Configure your org-mode experience!
 * 
 * Provides UI for:
 * - TODO state sequences
 * - Tag definitions
 * - Agenda preferences
 * - Display preferences
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Chip,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Paper,
  Stack
} from '@mui/material';
import {
  Add,
  Delete,
  Edit,
  ExpandMore,
  Save,
  Refresh,
  CheckCircle,
  Cancel
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import apiService from '../services/apiService';
import JournalLocationPicker from './JournalLocationPicker';

const OrgModeSettingsTab = () => {
  // State
  const [settings, setSettings] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  
  // Dialog states
  const [todoDialogOpen, setTodoDialogOpen] = useState(false);
  const [tagDialogOpen, setTagDialogOpen] = useState(false);
  const [editingTodoSequence, setEditingTodoSequence] = useState(null);
  const [editingTag, setEditingTag] = useState(null);

  // Load settings on mount
  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.get('/api/org/settings');
      
      if (response.success && response.settings) {
        setSettings(response.settings);
      } else {
        setError('Failed to load settings');
      }
    } catch (err) {
      console.error('Failed to load org settings:', err);
      setError(err.message || 'Failed to load settings');
    } finally {
      setLoading(false);
    }
  };

  const saveSettings = async (updatedSettings) => {
    try {
      setSaving(true);
      setError(null);
      setSuccess(null);

      const response = await apiService.put('/api/org/settings', updatedSettings);
      
      if (response.success && response.settings) {
        setSettings(response.settings);
        setSuccess('Settings saved successfully!');
        setTimeout(() => setSuccess(null), 3000);
      } else {
        setError('Failed to save settings');
      }
    } catch (err) {
      console.error('Failed to save org settings:', err);
      setError(err.message || 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const resetSettings = async () => {
    if (!window.confirm('Reset all org-mode settings to defaults?')) return;
    
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.delete('/api/org/settings');
      
      if (response.success && response.settings) {
        setSettings(response.settings);
        setSuccess('Settings reset to defaults!');
        setTimeout(() => setSuccess(null), 3000);
      } else {
        setError('Failed to reset settings');
      }
    } catch (err) {
      console.error('Failed to reset org settings:', err);
      setError(err.message || 'Failed to reset settings');
    } finally {
      setLoading(false);
    }
  };

  // TODO Sequence Handlers
  const handleAddTodoSequence = () => {
    setEditingTodoSequence({
      name: '',
      active_states: ['TODO'],
      done_states: ['DONE'],
      is_default: false
    });
    setTodoDialogOpen(true);
  };

  const handleEditTodoSequence = (sequence) => {
    setEditingTodoSequence({ ...sequence });
    setTodoDialogOpen(true);
  };

  const handleSaveTodoSequence = () => {
    const updatedSequences = editingTodoSequence.is_default
      ? [editingTodoSequence, ...settings.todo_sequences.filter(s => s.name !== editingTodoSequence.name && !s.is_default)]
      : settings.todo_sequences.some(s => s.name === editingTodoSequence.name)
        ? settings.todo_sequences.map(s => s.name === editingTodoSequence.name ? editingTodoSequence : s)
        : [...settings.todo_sequences, editingTodoSequence];
    
    saveSettings({ todo_sequences: updatedSequences });
    setTodoDialogOpen(false);
    setEditingTodoSequence(null);
  };

  const handleDeleteTodoSequence = (sequenceName) => {
    if (!window.confirm(`Delete TODO sequence "${sequenceName}"?`)) return;
    
    const updatedSequences = settings.todo_sequences.filter(s => s.name !== sequenceName);
    saveSettings({ todo_sequences: updatedSequences });
  };

  // Tag Handlers
  const handleAddTag = () => {
    setEditingTag({
      name: '',
      category: '',
      color: '#1976d2',
      icon: '',
      description: ''
    });
    setTagDialogOpen(true);
  };

  const handleEditTag = (tag) => {
    setEditingTag({ ...tag });
    setTagDialogOpen(true);
  };

  const handleSaveTag = () => {
    const updatedTags = settings.tags.some(t => t.name === editingTag.name)
      ? settings.tags.map(t => t.name === editingTag.name ? editingTag : t)
      : [...settings.tags, editingTag];
    
    saveSettings({ tags: updatedTags });
    setTagDialogOpen(false);
    setEditingTag(null);
  };

  const handleDeleteTag = (tagName) => {
    if (!window.confirm(`Delete tag "${tagName}"?`)) return;
    
    const updatedTags = settings.tags.filter(t => t.name !== tagName);
    saveSettings({ tags: updatedTags });
  };

  // Agenda Preferences Handler
  const handleAgendaPreferenceChange = (field, value) => {
    const updatedPrefs = {
      ...settings.agenda_preferences,
      [field]: value
    };
    saveSettings({ agenda_preferences: updatedPrefs });
  };

  // Display Preferences Handler
  const handleDisplayPreferenceChange = (field, value) => {
    const updatedPrefs = {
      ...settings.display_preferences,
      [field]: value
    };
    saveSettings({ display_preferences: updatedPrefs });
  };

  // Journal Preferences Handlers
  const handleJournalEnabledChange = (e) => {
    const updatedPrefs = {
      ...settings.journal_preferences,
      enabled: e.target.checked
    };
    saveSettings({ journal_preferences: updatedPrefs });
  };

  const handleOrganizationModeChange = (e) => {
    const updatedPrefs = {
      ...settings.journal_preferences,
      organization_mode: e.target.value
    };
    saveSettings({ journal_preferences: updatedPrefs });
  };

  const handleTimestampsChange = (e) => {
    const updatedPrefs = {
      ...settings.journal_preferences,
      include_timestamps: e.target.checked
    };
    saveSettings({ journal_preferences: updatedPrefs });
  };

  const handleDefaultTagsChange = (e) => {
    const tags = e.target.value
      ? e.target.value.split(',').map(t => t.trim()).filter(t => t)
      : ['journal'];
    const updatedPrefs = {
      ...settings.journal_preferences,
      default_tags: tags
    };
    saveSettings({ journal_preferences: updatedPrefs });
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!settings) {
    return (
      <Alert severity="error">
        Failed to load org-mode settings. Please refresh the page.
      </Alert>
    );
  }

  return (
    <Box>
      {/* Success/Error Messages */}
      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Header Actions */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5">
          Org-Mode Configuration
        </Typography>
        <Stack direction="row" spacing={1}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={loadSettings}
            disabled={loading}
          >
            Reload
          </Button>
          <Button
            variant="outlined"
            color="warning"
            startIcon={<Delete />}
            onClick={resetSettings}
            disabled={loading}
          >
            Reset to Defaults
          </Button>
        </Stack>
      </Box>

      <Grid container spacing={3}>
        {/* Inbox Configuration */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Quick Capture Inbox
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Configure where quick captures (Ctrl+Shift+C) are saved
                </Typography>
                
                <TextField
                  fullWidth
                  label="Inbox File Location"
                  value={settings?.inbox_file || ''}
                  onChange={(e) => saveSettings({ inbox_file: e.target.value })}
                  helperText="Path relative to your user directory (e.g., 'inbox.org' or 'OrgMode/inbox.org'). Leave blank for auto-discovery."
                  sx={{ mb: 2 }}
                />
                
                {settings?.inbox_file && (
                  <Alert severity="info" sx={{ mb: 2 }}>
                    Captures will be saved to: <strong>Users/[username]/{settings.inbox_file}</strong>
                  </Alert>
                )}
                
                <TextField
                  fullWidth
                  type="number"
                  label="Refile Max Level"
                  value={settings?.refile_max_level || 2}
                  onChange={(e) => saveSettings({ refile_max_level: parseInt(e.target.value) || 2 })}
                  helperText="Maximum heading level to show in refile targets (1 = only *, 2 = * and **, etc.)"
                  inputProps={{ min: 1, max: 6 }}
                  sx={{ mb: 2 }}
                />
                
                {!settings?.inbox_file && (
                  <Alert severity="info">
                    Auto-discovery enabled: The system will search for any existing inbox.org in your directory
                  </Alert>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Archive Preferences */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.02 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Archive Preferences
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Configure default archive location for archived entries (Ctrl+Shift+A). This reduces clutter in the archive dialog.
                </Typography>
                
                <TextField
                  fullWidth
                  label="Default Archive Location"
                  value={settings?.archive_preferences?.default_archive_location || ''}
                  onChange={(e) => {
                    const updatedPrefs = {
                      ...settings.archive_preferences,
                      default_archive_location: e.target.value || null
                    };
                    saveSettings({ archive_preferences: updatedPrefs });
                  }}
                  helperText="Use %s for source filename. Examples: '%s_archive' (same directory), 'OrgMode/archives/%s_archive' (central folder), 'OrgMode/archive.org' (single file)"
                  placeholder="%s_archive"
                  sx={{ mb: 2 }}
                />
                
                {settings?.archive_preferences?.default_archive_location && (
                  <Alert severity="info" sx={{ mb: 2 }}>
                    <Typography variant="body2" gutterBottom>
                      <strong>Archive location:</strong> {settings.archive_preferences.default_archive_location}
                    </Typography>
                    <Typography variant="caption" component="div">
                      <strong>Examples:</strong>
                      <ul style={{ marginTop: 4, marginBottom: 0, paddingLeft: 20 }}>
                        <li><code>%s_archive</code> → Archives to <code>filename_archive.org</code> in same directory</li>
                        <li><code>OrgMode/archives/%s_archive</code> → Archives to central <code>archives/</code> folder</li>
                        <li><code>OrgMode/archive.org</code> → All archives go to single file</li>
                      </ul>
                    </Typography>
                  </Alert>
                )}
                
                {!settings?.archive_preferences?.default_archive_location && (
                  <Alert severity="info">
                    Default behavior: Archives to <code>{'{filename}'}_archive.org</code> in the same directory as the source file
                  </Alert>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Journal Settings */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Journal Settings
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Configure journal entry organization and capture preferences
                </Typography>
                
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.journal_preferences?.enabled || false}
                      onChange={handleJournalEnabledChange}
                    />
                  }
                  label="Enable Journal"
                  sx={{ mb: 2 }}
                />
                
                {settings.journal_preferences?.enabled && (
                  <>
                    <Box sx={{ mb: 2, mt: 2 }}>
                      <JournalLocationPicker
                        value={settings.journal_preferences?.journal_location || ''}
                        onChange={(newValue) => {
                          const updatedPrefs = {
                            ...settings.journal_preferences,
                            journal_location: newValue || null
                          };
                          saveSettings({ journal_preferences: updatedPrefs });
                        }}
                        disabled={saving}
                        error={error && error.includes('Journal location')}
                      />
                    </Box>
                    
                    <FormControl fullWidth margin="normal" sx={{ mb: 2 }}>
                      <InputLabel>Organization Mode</InputLabel>
                      <Select
                        value={settings.journal_preferences?.organization_mode || 'monolithic'}
                        onChange={handleOrganizationModeChange}
                        label="Organization Mode"
                      >
                        <MenuItem value="monolithic">Monolithic (Single journal.org file)</MenuItem>
                        <MenuItem value="yearly">Yearly (YYYY - Journal.org)</MenuItem>
                        <MenuItem value="monthly">Monthly (YYYY-MM - Journal.org)</MenuItem>
                        <MenuItem value="daily">Daily (YYYY-MM-DD - Journal.org)</MenuItem>
                      </Select>
                    </FormControl>
                    
                    <FormControlLabel
                      control={
                        <Switch
                          checked={settings.journal_preferences?.include_timestamps !== false}
                          onChange={handleTimestampsChange}
                        />
                      }
                      label="Include timestamps in entries"
                      sx={{ mb: 2, display: 'block' }}
                    />
                    
                    <TextField
                      fullWidth
                      label="Default Tags"
                      value={(settings.journal_preferences?.default_tags || ['journal']).join(', ')}
                      onChange={handleDefaultTagsChange}
                      helperText="Comma-separated tags (e.g., journal, personal)"
                      margin="normal"
                    />
                    
                    <Alert severity="info" sx={{ mt: 2 }}>
                      <Typography variant="body2">
                        <strong>Note:</strong> Changing organization mode does not migrate existing entries.
                        You'll need to manually reorganize if you switch modes.
                      </Typography>
                    </Alert>
                  </>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* TODO State Sequences */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">
                    TODO State Sequences
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<Add />}
                    onClick={handleAddTodoSequence}
                    size="small"
                  >
                    Add Sequence
                  </Button>
                </Box>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Define TODO state workflows (e.g., TODO → NEXT → DONE)
                </Typography>

                <List>
                  {settings.todo_sequences.map((sequence, idx) => (
                    <React.Fragment key={sequence.name}>
                      {idx > 0 && <Divider />}
                      <ListItem
                        secondaryAction={
                          <Stack direction="row" spacing={1}>
                            <IconButton onClick={() => handleEditTodoSequence(sequence)} size="small">
                              <Edit />
                            </IconButton>
                            <IconButton
                              onClick={() => handleDeleteTodoSequence(sequence.name)}
                              size="small"
                              disabled={sequence.is_default}
                            >
                              <Delete />
                            </IconButton>
                          </Stack>
                        }
                      >
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              {sequence.name}
                              {sequence.is_default && (
                                <Chip label="Default" size="small" color="primary" />
                              )}
                            </Box>
                          }
                          secondary={
                            <Box sx={{ mt: 1 }}>
                              <Typography variant="caption" display="block">
                                Active: {sequence.active_states.map(s => (
                                  <Chip key={s} label={s} size="small" color="error" sx={{ mr: 0.5, height: 20, fontSize: '0.7rem' }} />
                                ))}
                              </Typography>
                              <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                                Done: {sequence.done_states.map(s => (
                                  <Chip key={s} label={s} size="small" color="success" sx={{ mr: 0.5, height: 20, fontSize: '0.7rem' }} />
                                ))}
                              </Typography>
                            </Box>
                          }
                        />
                      </ListItem>
                    </React.Fragment>
                  ))}
                </List>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Tag Definitions - Will implement in next message */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">
                    Tag Definitions
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<Add />}
                    onClick={handleAddTag}
                    size="small"
                  >
                    Add Tag
                  </Button>
                </Box>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Pre-define tags for auto-complete and consistent styling
                </Typography>

                {settings.tags.length === 0 ? (
                  <Alert severity="info">No tags defined yet. Click "Add Tag" to create your first tag.</Alert>
                ) : (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {settings.tags.map(tag => (
                      <Chip
                        key={tag.name}
                        label={`${tag.icon || ''} ${tag.name}`.trim()}
                        onDelete={() => handleDeleteTag(tag.name)}
                        onClick={() => handleEditTag(tag)}
                        sx={{ backgroundColor: tag.color || '#1976d2', color: '#fff' }}
                      />
                    ))}
                  </Box>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Agenda Preferences */}
        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Agenda Preferences
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Configure how your agenda view behaves
                </Typography>

                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Default View</InputLabel>
                    <Select
                      value={settings.agenda_preferences?.default_view || 'week'}
                      label="Default View"
                      onChange={(e) => handleAgendaPreferenceChange('default_view', e.target.value)}
                    >
                      <MenuItem value="day">Day</MenuItem>
                      <MenuItem value="week">Week</MenuItem>
                      <MenuItem value="month">Month</MenuItem>
                    </Select>
                  </FormControl>

                  <TextField
                    fullWidth
                    size="small"
                    label="Default Days Ahead"
                    type="number"
                    value={settings.agenda_preferences?.default_days_ahead || 7}
                    onChange={(e) => handleAgendaPreferenceChange('default_days_ahead', parseInt(e.target.value))}
                    inputProps={{ min: 1, max: 90 }}
                  />

                  <TextField
                    fullWidth
                    size="small"
                    label="Deadline Warning Days"
                    type="number"
                    value={settings.agenda_preferences?.deadline_warning_days || 3}
                    onChange={(e) => handleAgendaPreferenceChange('deadline_warning_days', parseInt(e.target.value))}
                    inputProps={{ min: 0, max: 30 }}
                  />

                  <FormControl fullWidth size="small">
                    <InputLabel>Week Start Day</InputLabel>
                    <Select
                      value={settings.agenda_preferences?.week_start_day || 1}
                      label="Week Start Day"
                      onChange={(e) => handleAgendaPreferenceChange('week_start_day', parseInt(e.target.value))}
                    >
                      <MenuItem value={0}>Sunday</MenuItem>
                      <MenuItem value={1}>Monday</MenuItem>
                    </Select>
                  </FormControl>

                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.agenda_preferences?.show_scheduled ?? true}
                        onChange={(e) => handleAgendaPreferenceChange('show_scheduled', e.target.checked)}
                      />
                    }
                    label="Show Scheduled Items"
                  />

                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.agenda_preferences?.show_deadlines ?? true}
                        onChange={(e) => handleAgendaPreferenceChange('show_deadlines', e.target.checked)}
                      />
                    }
                    label="Show Deadline Items"
                  />

                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.agenda_preferences?.group_by_date ?? true}
                        onChange={(e) => handleAgendaPreferenceChange('group_by_date', e.target.checked)}
                      />
                    }
                    label="Group by Date"
                  />
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Display Preferences */}
        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Display Preferences
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Customize how org-mode content is displayed
                </Typography>

                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.display_preferences?.default_collapsed ?? false}
                        onChange={(e) => handleDisplayPreferenceChange('default_collapsed', e.target.checked)}
                      />
                    }
                    label="Default Collapsed Headings"
                  />

                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.display_preferences?.show_properties ?? true}
                        onChange={(e) => handleDisplayPreferenceChange('show_properties', e.target.checked)}
                      />
                    }
                    label="Show Property Drawers"
                  />

                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.display_preferences?.show_tags_inline ?? true}
                        onChange={(e) => handleDisplayPreferenceChange('show_tags_inline', e.target.checked)}
                      />
                    }
                    label="Show Tags Inline"
                  />

                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.display_preferences?.highlight_current_line ?? true}
                        onChange={(e) => handleDisplayPreferenceChange('highlight_current_line', e.target.checked)}
                      />
                    }
                    label="Highlight Current Line"
                  />

                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.display_preferences?.indent_subheadings ?? true}
                        onChange={(e) => handleDisplayPreferenceChange('indent_subheadings', e.target.checked)}
                      />
                    }
                    label="Indent Subheadings"
                  />

                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.display_preferences?.indent_content_to_heading ?? false}
                        onChange={(e) => handleDisplayPreferenceChange('indent_content_to_heading', e.target.checked)}
                      />
                    }
                    label="Indent Content to Heading Level"
                  />
                  <Typography variant="caption" color="text.secondary" sx={{ ml: 4, display: 'block', mb: 1 }}>
                    Visually indent body text under headings (does not modify file content)
                  </Typography>

                  <Divider sx={{ my: 1 }} />

                  <Typography variant="subtitle2" gutterBottom>
                    TODO State Colors
                  </Typography>

                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    {['TODO', 'NEXT', 'WAITING', 'DONE', 'CANCELED'].map(state => (
                      <Box key={state} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="body2" sx={{ minWidth: 80 }}>
                          {state}:
                        </Typography>
                        <TextField
                          size="small"
                          type="color"
                          value={settings.display_preferences?.todo_state_colors?.[state] || '#888888'}
                          onChange={(e) => {
                            const newColors = {
                              ...settings.display_preferences?.todo_state_colors,
                              [state]: e.target.value
                            };
                            handleDisplayPreferenceChange('todo_state_colors', newColors);
                          }}
                          sx={{ width: 80 }}
                        />
                        <Chip
                          label={state}
                          size="small"
                          sx={{
                            backgroundColor: settings.display_preferences?.todo_state_colors?.[state] || '#888888',
                            color: '#fff'
                          }}
                        />
                      </Box>
                    ))}
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* TODO Sequence Dialog - Simplified for now */}
      <Dialog open={todoDialogOpen} onClose={() => setTodoDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          {editingTodoSequence?.name ? 'Edit' : 'Add'} TODO Sequence
        </DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField
              fullWidth
              label="Sequence Name"
              value={editingTodoSequence?.name || ''}
              onChange={(e) => setEditingTodoSequence({ ...editingTodoSequence, name: e.target.value })}
              sx={{ mb: 2 }}
            />
            <FormControlLabel
              control={
                <Switch
                  checked={editingTodoSequence?.is_default || false}
                  onChange={(e) => setEditingTodoSequence({ ...editingTodoSequence, is_default: e.target.checked })}
                />
              }
              label="Set as default sequence"
            />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              Active States (comma-separated): TODO, NEXT, WAITING
            </Typography>
            <TextField
              fullWidth
              value={editingTodoSequence?.active_states?.join(', ') || ''}
              onChange={(e) => setEditingTodoSequence({
                ...editingTodoSequence,
                active_states: e.target.value.split(',').map(s => s.trim()).filter(Boolean)
              })}
              sx={{ mb: 2 }}
            />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              Done States (comma-separated): DONE, CANCELED
            </Typography>
            <TextField
              fullWidth
              value={editingTodoSequence?.done_states?.join(', ') || ''}
              onChange={(e) => setEditingTodoSequence({
                ...editingTodoSequence,
                done_states: e.target.value.split(',').map(s => s.trim()).filter(Boolean)
              })}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTodoDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleSaveTodoSequence}
            disabled={!editingTodoSequence?.name || editingTodoSequence?.active_states?.length === 0}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>

      {/* Tag Dialog - Simplified for now */}
      <Dialog open={tagDialogOpen} onClose={() => setTagDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          {editingTag?.name ? 'Edit' : 'Add'} Tag
        </DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField
              fullWidth
              label="Tag Name"
              value={editingTag?.name || ''}
              onChange={(e) => setEditingTag({ ...editingTag, name: e.target.value })}
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              label="Category (optional)"
              value={editingTag?.category || ''}
              onChange={(e) => setEditingTag({ ...editingTag, category: e.target.value })}
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              label="Color (hex)"
              type="color"
              value={editingTag?.color || '#1976d2'}
              onChange={(e) => setEditingTag({ ...editingTag, color: e.target.value })}
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              label="Icon (emoji)"
              value={editingTag?.icon || ''}
              onChange={(e) => setEditingTag({ ...editingTag, icon: e.target.value })}
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              label="Description (optional)"
              multiline
              rows={2}
              value={editingTag?.description || ''}
              onChange={(e) => setEditingTag({ ...editingTag, description: e.target.value })}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTagDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleSaveTag}
            disabled={!editingTag?.name}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default OrgModeSettingsTab;

