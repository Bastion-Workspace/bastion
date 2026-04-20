import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  CircularProgress,
  Alert,
  TextField,
  FormControlLabel,
  Checkbox
} from '@mui/material';
import { Archive, CheckCircle } from '@mui/icons-material';
import apiService from '../services/apiService';

/**
 * Single-entry org archive dialog.
 * - If the source file defines #+ARCHIVE:, archives immediately (spinner only, no confirm step).
 * - Otherwise shows destination and optional override before archiving.
 */
const OrgArchiveDialog = ({
  open,
  onClose,
  sourceFile,
  sourceLine,
  sourceHeading,
  onArchiveComplete
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [customLocation, setCustomLocation] = useState('');
  const [useCustomLocation, setUseCustomLocation] = useState(false);
  const [archiveLocationInfo, setArchiveLocationInfo] = useState(null);
  const [locationLoading, setLocationLoading] = useState(false);
  /** When false, user only sees a spinner (location resolve or one-click archive). */
  const [confirmUI, setConfirmUI] = useState(false);

  const onCloseRef = useRef(onClose);
  const onArchiveCompleteRef = useRef(onArchiveComplete);
  const openRef = useRef(open);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    onArchiveCompleteRef.current = onArchiveComplete;
  }, [onArchiveComplete]);

  useEffect(() => {
    openRef.current = open;
  }, [open]);

  const resetLocalState = useCallback(() => {
    setError(null);
    setSuccess(false);
    setCustomLocation('');
    setUseCustomLocation(false);
    setArchiveLocationInfo(null);
    setLocationLoading(false);
    setLoading(false);
    setConfirmUI(false);
  }, []);

  useEffect(() => {
    if (!open || !sourceFile) {
      resetLocalState();
      return;
    }

    setLocationLoading(true);
    setConfirmUI(false);
    setArchiveLocationInfo(null);
    setError(null);
    setSuccess(false);
    setCustomLocation('');
    setUseCustomLocation(false);
    setLoading(false);

    let cancelled = false;

    (async () => {
      let locationInfo = null;
      try {
        const response = await apiService.get(
          `/api/org/archive-location?source_file=${encodeURIComponent(sourceFile)}`
        );
        if (cancelled || !openRef.current) return;
        if (response.success) {
          locationInfo = response;
          setArchiveLocationInfo(response);
        } else {
          setArchiveLocationInfo(null);
        }
      } catch (err) {
        console.warn('Failed to load archive location:', err);
        if (!cancelled && openRef.current) {
          setArchiveLocationInfo(null);
        }
      } finally {
        if (!cancelled && openRef.current) {
          setLocationLoading(false);
        }
      }

      if (cancelled || !openRef.current) return;

      if (sourceLine == null) {
        setConfirmUI(true);
        return;
      }

      if (!locationInfo || locationInfo.source_type !== 'file') {
        setConfirmUI(true);
        return;
      }

      setLoading(true);
      setError(null);
      try {
        const archiveResponse = await apiService.post('/api/org/archive', {
          source_file: sourceFile,
          line_number: sourceLine,
          archive_location: null
        });
        if (cancelled || !openRef.current) return;
        if (archiveResponse.success) {
          setLoading(false);
          onArchiveCompleteRef.current?.(archiveResponse);
          onCloseRef.current();
          return;
        } else {
          setError(archiveResponse.error || 'Archive failed');
          setConfirmUI(true);
        }
      } catch (err) {
        if (cancelled || !openRef.current) return;
        setError(err.message || 'Archive operation failed');
        setConfirmUI(true);
      } finally {
        if (!cancelled && openRef.current) {
          setLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [open, sourceFile, sourceLine, resetLocalState]);

  const getArchiveLocationDisplay = () => {
    if (!archiveLocationInfo) {
      return sourceFile ? sourceFile.replace('.org', '_archive.org') : 'default location';
    }

    return archiveLocationInfo.preview || archiveLocationInfo.archive_location || 'default location';
  };

  const getArchiveSourceDisplay = () => {
    if (!archiveLocationInfo) return null;

    const sourceType = archiveLocationInfo.source_type;
    if (sourceType === 'file') {
      return {
        type: 'file',
        label: 'File-level #+ARCHIVE: directive',
        value: archiveLocationInfo.file_archive
      };
    } else if (sourceType === 'settings') {
      return {
        type: 'settings',
        label: 'Settings default',
        value: archiveLocationInfo.settings_archive
      };
    }
    return {
      type: 'default',
      label: 'Default behavior',
      value: null
    };
  };

  const handleArchive = async () => {
    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      const response = await apiService.post('/api/org/archive', {
        source_file: sourceFile,
        line_number: sourceLine,
        archive_location: useCustomLocation && customLocation ? customLocation : null
      });

      if (response.success) {
        setSuccess(true);
        setTimeout(() => {
          onArchiveCompleteRef.current?.(response);
          resetLocalState();
          onCloseRef.current();
        }, 1500);
      } else {
        setError(response.error || 'Archive failed');
      }
    } catch (err) {
      setError(err.message || 'Archive operation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    resetLocalState();
    onClose();
  };

  useEffect(() => {
    if (!open) return;

    const handleKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (confirmUI && !loading && !success && !(useCustomLocation && !customLocation)) {
          handleArchive();
        }
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        handleClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, confirmUI, loading, success, useCustomLocation, customLocation]);

  const showSpinner = Boolean(open && (locationLoading || (loading && !success)));
  const showForm = Boolean(
    open && confirmUI && !locationLoading && !loading && !success
  );
  const showManualArchiving = Boolean(open && confirmUI && loading && !success);
  const showSuccess = Boolean(open && success);

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Archive />
        {locationLoading ? 'Archive entry' : loading && !confirmUI ? 'Archiving…' : 'Archive entry'}
      </DialogTitle>

      <DialogContent>
        {showSpinner && !showManualArchiving && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 3 }}>
            <CircularProgress size={40} sx={{ mb: 2 }} />
            <Typography variant="body2" color="text.secondary">
              {locationLoading ? 'Loading archive location…' : 'Archiving entry…'}
            </Typography>
          </Box>
        )}

        {showManualArchiving && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 3 }}>
            <CircularProgress size={40} sx={{ mb: 2 }} />
            <Typography variant="body2" color="text.secondary">
              Archiving entry…
            </Typography>
          </Box>
        )}

        {showForm && (
          <Box>
            <Typography variant="body2" color="text.secondary" paragraph>
              {archiveLocationInfo
                ? 'Confirm archive destination (set #+ARCHIVE: in the org file to skip this step).'
                : 'Could not read archive settings. Choose a destination or try again.'}
            </Typography>

            <Box sx={{ bgcolor: 'background.default', p: 2, borderRadius: 1, mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Entry to archive
              </Typography>
              <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                {sourceHeading}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Line {sourceLine} in {sourceFile}
              </Typography>
            </Box>

            {archiveLocationInfo ? (
              (() => {
                const sourceInfo = getArchiveSourceDisplay();
                return (
                  <Alert
                    severity={sourceInfo.type === 'file' ? 'success' : 'info'}
                    sx={{ mb: 2 }}
                  >
                    <Typography variant="body2" gutterBottom>
                      <strong>
                        {sourceInfo.type === 'file' && 'File-level #+ARCHIVE: directive'}
                        {sourceInfo.type === 'settings' && 'Default from settings'}
                        {sourceInfo.type === 'default' && 'Default archive file'}
                      </strong>
                    </Typography>
                    {sourceInfo.value && (
                      <Typography variant="caption" component="div" sx={{ fontFamily: 'monospace', mb: 1 }}>
                        {sourceInfo.value}
                      </Typography>
                    )}
                    <Typography variant="caption" component="div">
                      Will archive to: <code>{getArchiveLocationDisplay()}</code>
                    </Typography>
                  </Alert>
                );
              })()
            ) : (
              <Alert severity="info" sx={{ mb: 2 }}>
                <Typography variant="body2">
                  Will archive to: <code>{getArchiveLocationDisplay()}</code>
                </Typography>
              </Alert>
            )}

            <FormControlLabel
              control={
                <Checkbox
                  checked={useCustomLocation}
                  onChange={(e) => {
                    setUseCustomLocation(e.target.checked);
                    if (e.target.checked && !customLocation) {
                      const defaultArchiveLocation =
                        archiveLocationInfo?.file_archive ||
                        archiveLocationInfo?.settings_archive ||
                        archiveLocationInfo?.archive_location ||
                        '';
                      if (defaultArchiveLocation) {
                        setCustomLocation(defaultArchiveLocation);
                      }
                    }
                  }}
                />
              }
              label="Use a different archive file for this action"
            />

            {useCustomLocation && (
              <TextField
                fullWidth
                size="small"
                placeholder="e.g., OrgMode/archive/2025.org or %s_archive"
                value={customLocation}
                onChange={(e) => setCustomLocation(e.target.value)}
                helperText="Use %s for the source file stem (without .org)."
                sx={{ mt: 1 }}
              />
            )}

            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
          </Box>
        )}

        {showSuccess && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 3 }}>
            <CheckCircle color="success" sx={{ fontSize: 48, mb: 2 }} />
            <Typography variant="body1" color="success.main">
              Entry archived successfully!
            </Typography>
          </Box>
        )}
      </DialogContent>

      <DialogActions>
        {showSpinner && !showManualArchiving && (
          <Button onClick={handleClose}>
            Cancel
          </Button>
        )}
        {showManualArchiving && (
          <Button onClick={handleClose} disabled>
            Cancel
          </Button>
        )}
        {showForm && (
          <>
            <Button onClick={handleClose} disabled={loading}>
              Cancel
            </Button>
            <Button
              onClick={handleArchive}
              variant="contained"
              color="primary"
              disabled={loading || (useCustomLocation && !customLocation)}
              startIcon={loading ? <CircularProgress size={16} /> : <Archive />}
            >
              {loading ? 'Archiving…' : 'Archive (Ctrl+Enter)'}
            </Button>
          </>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default OrgArchiveDialog;
