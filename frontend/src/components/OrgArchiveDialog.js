import React, { useState, useEffect } from 'react';
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
 * OrgArchiveDialog - Roosevelt's "Clean Desk Policy"
 * 
 * **BULLY!** Archive completed tasks to keep files lean and mean!
 * 
 * Features:
 * - Archive single entry at cursor position
 * - Custom archive location (optional)
 * - Uses default from Settings if configured
 * - Success/error feedback
 * - Confirmation for safety
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

  // Fetch archive location info (file-level > settings > default)
  useEffect(() => {
    if (open && sourceFile) {
      const fetchArchiveLocation = async () => {
        try {
          setLocationLoading(true);
          const response = await apiService.get(`/api/org/archive-location?source_file=${encodeURIComponent(sourceFile)}`);
          if (response.success) {
            setArchiveLocationInfo(response);
          } else {
            setArchiveLocationInfo(null);
          }
        } catch (err) {
          console.warn('Failed to load archive location:', err);
          setArchiveLocationInfo(null);
        } finally {
          setLocationLoading(false);
        }
      };
      fetchArchiveLocation();
    } else {
      // Reset when dialog closes
      setArchiveLocationInfo(null);
      setCustomLocation('');
      setUseCustomLocation(false);
    }
  }, [open, sourceFile]);

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
      console.log('📦 ROOSEVELT: Archiving entry:', {
        sourceFile,
        sourceLine,
        sourceHeading,
        customLocation: useCustomLocation ? customLocation : 'default'
      });

      const response = await apiService.post('/api/org/archive', {
        source_file: sourceFile,
        line_number: sourceLine,
        archive_location: useCustomLocation && customLocation ? customLocation : null
      });

      if (response.success) {
        console.log('✅ Archive successful:', response);
        setSuccess(true);
        
        // Close after brief success message
        setTimeout(() => {
          if (onArchiveComplete) {
            onArchiveComplete(response);
          }
          handleClose();
        }, 1500);
      } else {
        setError(response.error || 'Archive failed');
      }
    } catch (err) {
      console.error('❌ Archive failed:', err);
      setError(err.message || 'Archive operation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    if (!loading) {
      setError(null);
      setSuccess(false);
      setCustomLocation('');
      setUseCustomLocation(false);
      onClose();
    }
  };
  
  // Handle keyboard shortcuts
  useEffect(() => {
    if (!open) return;
    
    const handleKeyDown = (e) => {
      // Ctrl+Enter or Cmd+Enter to archive
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        // Check if we can archive (same conditions as button)
        if (!loading && !success && !(useCustomLocation && !customLocation)) {
          handleArchive();
        }
      }
      // Escape to close
      if (e.key === 'Escape' && !loading) {
        e.preventDefault();
        handleClose();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, loading, success, useCustomLocation, customLocation]);

  return (
    <Dialog 
      open={open} 
      onClose={handleClose}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Archive />
        Archive Entry
      </DialogTitle>

      <DialogContent>
        {!success && !loading && (
          <Box>
            <Typography variant="body2" color="text.secondary" paragraph>
              Archive this entry to keep your active files clean.
            </Typography>

            <Box sx={{ bgcolor: 'background.default', p: 2, borderRadius: 1, mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Entry to Archive:
              </Typography>
              <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                {sourceHeading}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Line {sourceLine} in {sourceFile}
              </Typography>
            </Box>

            {locationLoading ? (
              <Alert severity="info" sx={{ mb: 2 }}>
                <Typography variant="body2">Loading archive location...</Typography>
              </Alert>
            ) : archiveLocationInfo ? (
              (() => {
                const sourceInfo = getArchiveSourceDisplay();
                return (
                  <Alert 
                    severity={sourceInfo.type === 'file' ? 'success' : 'info'} 
                    sx={{ mb: 2 }}
                  >
                    <Typography variant="body2" gutterBottom>
                      <strong>
                        {sourceInfo.type === 'file' && '📄 Using file-level #+ARCHIVE: directive'}
                        {sourceInfo.type === 'settings' && '⚙️ Using default from Settings'}
                        {sourceInfo.type === 'default' && '📦 Using default behavior'}
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
                    {sourceInfo.type === 'file' && (
                      <Typography variant="caption" component="div" sx={{ mt: 0.5, fontStyle: 'italic' }}>
                        File-level directive takes priority over Settings. You can override below if needed.
                      </Typography>
                    )}
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
                      // Pre-populate with default if available
                      const defaultArchiveLocation = archiveLocationInfo?.file_archive || archiveLocationInfo?.settings_archive || archiveLocationInfo?.archive_location || '';
                      if (defaultArchiveLocation) {
                        setCustomLocation(defaultArchiveLocation);
                      }
                    }
                  }}
                />
              }
              label="Override with custom archive location"
            />

            {useCustomLocation && (
              <TextField
                fullWidth
                size="small"
                placeholder="e.g., OrgMode/archive/2025.org or %s_archive"
                value={customLocation}
                onChange={(e) => setCustomLocation(e.target.value)}
                helperText={
                  archiveLocationInfo?.source_type === 'file' 
                    ? "Use %s for source filename. Leave empty to use file-level #+ARCHIVE: directive."
                    : "Use %s for source filename. Leave empty to use default (file-level directive, Settings, or default behavior)."
                }
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

        {loading && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 3 }}>
            <CircularProgress size={40} sx={{ mb: 2 }} />
            <Typography variant="body2" color="text.secondary">
              Archiving entry...
            </Typography>
          </Box>
        )}

        {success && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 3 }}>
            <CheckCircle color="success" sx={{ fontSize: 48, mb: 2 }} />
            <Typography variant="body1" color="success.main">
              Entry archived successfully!
            </Typography>
          </Box>
        )}
      </DialogContent>

      <DialogActions>
        {!success && (
          <>
            <Button onClick={handleClose} disabled={loading}>
              Cancel (Esc)
            </Button>
            <Button
              onClick={handleArchive}
              variant="contained"
              color="primary"
              disabled={loading || (useCustomLocation && !customLocation)}
              startIcon={loading ? <CircularProgress size={16} /> : <Archive />}
            >
              {loading ? 'Archiving...' : 'Archive (Ctrl+Enter)'}
            </Button>
          </>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default OrgArchiveDialog;



