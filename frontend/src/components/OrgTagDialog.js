/**
 * Org Tag Dialog Component
 * Add tags to org-mode headings with Ctrl+Shift+E
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Box,
  Typography,
  Chip,
  Stack,
  Alert,
  IconButton,
  Tooltip,
  FormControlLabel,
  Checkbox
} from '@mui/material';
import { Close, LocalOffer } from '@mui/icons-material';
import apiService from '../services/apiService';

const OrgTagDialog = ({ open, onClose, document, lineNumber, currentHeading }) => {
  const [tags, setTags] = useState([]);
  const [tagInput, setTagInput] = useState('');
  const [replaceExisting, setReplaceExisting] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  
  const inputRef = useRef(null);
  
  // Auto-focus input when dialog opens
  useEffect(() => {
    if (open && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [open]);
  
  // Reset state when dialog opens
  useEffect(() => {
    if (open) {
      setTags([]);
      setTagInput('');
      setReplaceExisting(false);
      setError(null);
      setSuccess(null);
    }
  }, [open]);
  
  const handleAddTag = () => {
    const tag = tagInput.trim();
    if (tag && !tags.includes(tag)) {
      setTags([...tags, tag]);
      setTagInput('');
    }
  };
  
  const handleRemoveTag = (tagToRemove) => {
    setTags(tags.filter(t => t !== tagToRemove));
  };
  
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (tagInput.trim()) {
        handleAddTag();
      } else if (tags.length > 0) {
        handleSubmit();
      }
    } else if (e.key === 'Escape') {
      onClose();
    }
  };
  
  const handleSubmit = async () => {
    if (tags.length === 0) {
      setError('Please add at least one tag');
      return;
    }
    
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      // Determine file path
      let filePath = document.filename;
      if (document.folder_id && document.folder_name) {
        filePath = `${document.folder_name}/${document.filename}`;
      } else {
        filePath = `OrgMode/${document.filename}`;
      }
      
      const response = await apiService.post('/api/org/tag', {
        file_path: filePath,
        line_number: lineNumber,
        tags: tags,
        replace_existing: replaceExisting
      });
      
      if (response.success) {
        setSuccess(`Tags added successfully: ${response.tags_applied?.join(', ')}`);
        
        // Close dialog after short delay
        setTimeout(() => {
          onClose();
          // Force refresh of the document viewer
          window.location.reload();
        }, 1000);
      } else {
        setError(response.message || 'Failed to add tags');
      }
    } catch (err) {
      console.error('Tag error:', err);
      setError(err.message || 'Failed to add tags');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: {
          bgcolor: 'background.paper',
          backgroundImage: 'none'
        }
      }}
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <LocalOffer color="primary" />
          <Typography variant="h6">Add Tags</Typography>
        </Box>
        <IconButton onClick={onClose} size="small">
          <Close />
        </IconButton>
      </DialogTitle>
      
      <DialogContent dividers>
        {/* Heading info */}
        <Box sx={{ mb: 2, p: 2, bgcolor: 'action.hover', borderRadius: 1 }}>
          <Typography variant="body2" color="text.secondary">
            Line {lineNumber}
          </Typography>
          <Typography variant="body1" sx={{ mt: 0.5, fontFamily: 'monospace' }}>
            {currentHeading || 'Current heading'}
          </Typography>
        </Box>
        
        {/* Error/Success messages */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
        
        {success && (
          <Alert severity="success" sx={{ mb: 2 }}>
            {success}
          </Alert>
        )}
        
        {/* Tag input */}
        <TextField
          inputRef={inputRef}
          fullWidth
          label="Add Tag"
          value={tagInput}
          onChange={(e) => setTagInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="e.g., @outside, urgent, work"
          helperText="Press Enter to add tag, Escape to cancel"
          disabled={loading}
          sx={{ mb: 2 }}
        />
        
        {/* Tag chips */}
        {tags.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Tags to add:
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {tags.map((tag, idx) => (
                <Chip
                  key={idx}
                  label={tag}
                  onDelete={() => handleRemoveTag(tag)}
                  color="primary"
                  variant="outlined"
                  size="small"
                />
              ))}
            </Stack>
          </Box>
        )}
        
        {/* Replace existing option */}
        <FormControlLabel
          control={
            <Checkbox
              checked={replaceExisting}
              onChange={(e) => setReplaceExisting(e.target.checked)}
              disabled={loading}
            />
          }
          label="Replace existing tags (instead of merging)"
        />
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <Button
          onClick={handleSubmit}
          variant="contained"
          disabled={loading || tags.length === 0}
          startIcon={<LocalOffer />}
        >
          {loading ? 'Adding Tags...' : 'Add Tags'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default OrgTagDialog;

