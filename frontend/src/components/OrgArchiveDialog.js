import React, { useState } from 'react';
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
              Archive this entry to keep your active files clean. Archived entries are moved to <code>{sourceFile.replace('.org', '_archive.org')}</code>
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

            <FormControlLabel
              control={
                <Checkbox
                  checked={useCustomLocation}
                  onChange={(e) => setUseCustomLocation(e.target.checked)}
                />
              }
              label="Use custom archive location"
            />

            {useCustomLocation && (
              <TextField
                fullWidth
                size="small"
                placeholder="e.g., OrgMode/archive/2025.org"
                value={customLocation}
                onChange={(e) => setCustomLocation(e.target.value)}
                helperText="Leave empty for default ({filename}_archive.org)"
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
              Cancel
            </Button>
            <Button
              onClick={handleArchive}
              variant="contained"
              color="primary"
              disabled={loading || (useCustomLocation && !customLocation)}
              startIcon={loading ? <CircularProgress size={16} /> : <Archive />}
            >
              {loading ? 'Archiving...' : 'Archive'}
            </Button>
          </>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default OrgArchiveDialog;



