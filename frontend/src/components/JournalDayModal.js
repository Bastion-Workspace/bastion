/**
 * Journal for the day modal (Ctrl+Shift+J).
 * Loads and edits the current day's journal entry; supports date picker to view/edit other days.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Box,
  Typography,
  Alert,
  IconButton,
  Link,
} from '@mui/material';
import { MenuBook, Close } from '@mui/icons-material';
import { Link as RouterLink } from 'react-router-dom';
import apiService from '../services/apiService';

const todayStr = () => new Date().toISOString().split('T')[0];

const JournalDayModal = ({ open, onClose }) => {
  const [selectedDate, setSelectedDate] = useState(todayStr());
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [journalDisabled, setJournalDisabled] = useState(false);
  const contentRef = useRef(null);

  const fetchEntry = useCallback(async (date) => {
    const param = date === todayStr() ? 'today' : date;
    setLoading(true);
    setError(null);
    setJournalDisabled(false);
    try {
      const response = await apiService.get(`/api/org/journal/entry?date=${encodeURIComponent(param)}`);
      if (response.error && !response.success) {
        const disabled = (response.error || '').toLowerCase().includes('disabled');
        setJournalDisabled(disabled);
        setError(disabled ? null : response.error);
        setContent(disabled ? '' : (response.content ?? ''));
      } else {
        setContent(response.content ?? '');
        setError(null);
        setJournalDisabled(false);
      }
    } catch (err) {
      setError(err.message || 'Failed to load journal entry');
      setContent('');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) {
      setSelectedDate(todayStr());
      fetchEntry(todayStr());
    }
  }, [open, fetchEntry]);

  useEffect(() => {
    if (open && contentRef.current && !loading) {
      setTimeout(() => contentRef.current?.focus(), 100);
    }
  }, [open, loading]);

  useEffect(() => {
    if (!open) {
      setTimeout(() => {
        setContent('');
        setError(null);
        setSuccess(null);
        setJournalDisabled(false);
      }, 300);
    }
  }, [open]);

  const handleDateChange = (e) => {
    const next = e.target.value;
    setSelectedDate(next);
    fetchEntry(next);
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      const dateParam = selectedDate === todayStr() ? 'today' : selectedDate;
      const response = await apiService.put('/api/org/journal/entry', {
        date: dateParam,
        content,
      });
      if (response.success) {
        setSuccess('Saved');
        window.dispatchEvent(new CustomEvent('journalDocumentUpdated'));
        setTimeout(() => onClose(), 1000);
      } else {
        setError(response.error || 'Failed to save');
      }
    } catch (err) {
      setError(err.message || 'Failed to save journal entry');
    } finally {
      setSaving(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      if (!journalDisabled && !saving) handleSave();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{ sx: { minHeight: '400px' } }}
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <MenuBook color="primary" />
          <Typography variant="h6">Journal for the day</Typography>
          <Typography variant="caption" color="text.secondary">
            (Ctrl+Shift+J)
          </Typography>
        </Box>
        <IconButton size="small" onClick={onClose}>
          <Close />
        </IconButton>
      </DialogTitle>
      <DialogContent>
        {journalDisabled && (
          <Alert severity="info" sx={{ mb: 2 }}>
            Journal is disabled. Enable it in{' '}
            <Link component={RouterLink} to="/settings" onClick={onClose}>
              Settings → Org-Mode
            </Link>
            .
          </Alert>
        )}
        {error && !journalDisabled && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
        {success && (
          <Alert severity="success" sx={{ mb: 2 }}>
            {success}
          </Alert>
        )}
        {!journalDisabled && (
          <>
            <TextField
              fullWidth
              type="date"
              label="Date"
              value={selectedDate}
              onChange={handleDateChange}
              InputLabelProps={{ shrink: true }}
              margin="normal"
              sx={{ mb: 2 }}
              disabled={loading || saving}
            />
            <TextField
              inputRef={contentRef}
              fullWidth
              multiline
              minRows={8}
              maxRows={20}
              label="Entry"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="What happened today? Use %T (date+time), %t (date), %<%I:%M %p> (time)"
              helperText="Ctrl+Enter to save, Esc to close"
              disabled={loading || saving}
              sx={{ mb: 2 }}
            />
          </>
        )}
        {loading && (
          <Typography color="text.secondary" sx={{ py: 2 }}>
            Loading…
          </Typography>
        )}
      </DialogContent>
      {!journalDisabled && (
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={onClose} disabled={saving}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleSave}
            disabled={saving || loading}
          >
            {saving ? 'Saving…' : 'Save'}
          </Button>
        </DialogActions>
      )}
    </Dialog>
  );
};

export default JournalDayModal;
