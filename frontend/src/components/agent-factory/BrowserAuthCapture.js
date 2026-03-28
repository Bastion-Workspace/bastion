/**
 * Interactive browser login capture for playbook authentication.
 * Shows screenshot, supports click (on image), fill (selector + value), keypress (Enter/Tab),
 * then "I'm Logged In" to capture session state and resume the playbook.
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Button,
  TextField,
  Paper,
  CircularProgress,
  Alert,
} from '@mui/material';
import { KeyboardReturn, Tab, Check, Cancel } from '@mui/icons-material';
import apiService from '../../services/apiService';

const POLL_INTERVAL_MS = 2500;

export default function BrowserAuthCapture({
  pendingAuth,
  onComplete,
  onCancel,
}) {
  const {
    site_domain = '',
    session_id = '',
    screenshot: initialScreenshot = null,
    prompt = '',
  } = pendingAuth || {};

  const [screenshot, setScreenshot] = useState(initialScreenshot);
  const [fillSelector, setFillSelector] = useState('');
  const [fillValue, setFillValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [capturing, setCapturing] = useState(false);
  const imgRef = useRef(null);

  const pollScreenshot = useCallback(async () => {
    if (!session_id) return;
    try {
      const res = await apiService.agentFactory.browserAuthScreenshot(session_id);
      if (res?.success && res.screenshot_b64) {
        setScreenshot(`data:image/png;base64,${res.screenshot_b64}`);
      }
    } catch (err) {
      setError(err?.message || 'Screenshot failed');
    }
  }, [session_id]);

  useEffect(() => {
    if (initialScreenshot) setScreenshot(initialScreenshot);
  }, [initialScreenshot]);

  useEffect(() => {
    if (!session_id) return;
    pollScreenshot();
    const id = setInterval(pollScreenshot, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [session_id, pollScreenshot]);

  const handleImageClick = async (e) => {
    const img = imgRef.current;
    if (!img || !session_id) return;
    const rect = img.getBoundingClientRect();
    const scaleX = (img.naturalWidth || img.width) / rect.width;
    const scaleY = (img.naturalHeight || img.height) / rect.height;
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.agentFactory.browserAuthInteract({
        session_id,
        action: 'click',
        click_x: x,
        click_y: y,
      });
      if (res?.success && res.screenshot_b64) {
        setScreenshot(`data:image/png;base64,${res.screenshot_b64}`);
      }
    } catch (err) {
      setError(err?.message || 'Click failed');
    } finally {
      setLoading(false);
    }
  };

  const handleFill = async () => {
    if (!session_id || !fillSelector.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.agentFactory.browserAuthInteract({
        session_id,
        action: 'fill',
        selector: fillSelector.trim(),
        value: fillValue ?? '',
      });
      if (res?.success && res.screenshot_b64) {
        setScreenshot(`data:image/png;base64,${res.screenshot_b64}`);
      }
    } catch (err) {
      setError(err?.message || 'Fill failed');
    } finally {
      setLoading(false);
    }
  };

  const handleKeypress = async (key) => {
    if (!session_id) return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.agentFactory.browserAuthInteract({
        session_id,
        action: 'keypress',
        key,
      });
      if (res?.success && res.screenshot_b64) {
        setScreenshot(`data:image/png;base64,${res.screenshot_b64}`);
      }
    } catch (err) {
      setError(err?.message || 'Keypress failed');
    } finally {
      setLoading(false);
    }
  };

  const handleCaptureAndComplete = async () => {
    if (!session_id || !site_domain) return;
    setCapturing(true);
    setError(null);
    try {
      await apiService.agentFactory.browserAuthCapture(session_id, {
        site_domain,
      });
      onComplete?.();
    } catch (err) {
      setError(err?.message || 'Capture failed');
    } finally {
      setCapturing(false);
    }
  };

  if (!pendingAuth) return null;

  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <Typography variant="h6" sx={{ mb: 1 }}>
        Authentication required for {site_domain}
      </Typography>
      {prompt && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {prompt}
        </Typography>
      )}
      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      <Box
        sx={{
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 1,
          overflow: 'hidden',
          maxWidth: 900,
          mb: 2,
        }}
      >
        {screenshot ? (
          <img
            ref={imgRef}
            src={screenshot}
            alt="Browser"
            style={{
              width: '100%',
              height: 'auto',
              display: 'block',
              cursor: loading ? 'wait' : 'crosshair',
            }}
            onClick={handleImageClick}
          />
        ) : (
          <Box sx={{ p: 4, textAlign: 'center' }}>
            {loading ? (
              <CircularProgress size={32} />
            ) : (
              <Typography color="text.secondary">Loading screenshot…</Typography>
            )}
          </Box>
        )}
      </Box>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, alignItems: 'center', mb: 2 }}>
        <TextField
          size="small"
          placeholder="Selector (e.g. #email)"
          value={fillSelector}
          onChange={(e) => setFillSelector(e.target.value)}
          sx={{ minWidth: 160 }}
        />
        <TextField
          size="small"
          placeholder="Value to type"
          value={fillValue}
          onChange={(e) => setFillValue(e.target.value)}
          sx={{ minWidth: 140 }}
        />
        <Button size="small" variant="outlined" onClick={handleFill} disabled={loading || !fillSelector.trim()}>
          Fill
        </Button>
        <Button
          size="small"
          variant="outlined"
          startIcon={<KeyboardReturn />}
          onClick={() => handleKeypress('Enter')}
          disabled={loading}
        >
          Enter
        </Button>
        <Button
          size="small"
          variant="outlined"
          startIcon={<Tab />}
          onClick={() => handleKeypress('Tab')}
          disabled={loading}
        >
          Tab
        </Button>
      </Box>
      <Box sx={{ display: 'flex', gap: 1 }}>
        <Button
          variant="contained"
          startIcon={capturing ? <CircularProgress size={18} /> : <Check />}
          onClick={handleCaptureAndComplete}
          disabled={capturing}
        >
          I'm Logged In
        </Button>
        <Button variant="outlined" startIcon={<Cancel />} onClick={onCancel} disabled={capturing}>
          Cancel
        </Button>
      </Box>
    </Paper>
  );
}
