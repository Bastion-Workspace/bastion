/**
 * DiffMergeDialog - Compare and merge current content with new LLM description.
 * Two text areas: left = current (editable), right = new (read-only). User can copy/paste between them.
 */

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Box,
  Typography
} from '@mui/material';
import { Close, Check, SwapHoriz } from '@mui/icons-material';

const DiffMergeDialog = ({
  open,
  onClose,
  currentContent = '',
  newContent = '',
  onKeepOld,
  onReplaceWithNew
}) => {
  const [leftValue, setLeftValue] = useState(currentContent);
  const [rightValue] = useState(newContent);

  useEffect(() => {
    if (open) {
      setLeftValue(currentContent);
    }
  }, [open, currentContent]);

  const handleKeepOld = () => {
    if (onKeepOld) onKeepOld();
    onClose();
  };

  const handleReplaceWithNew = () => {
    if (onReplaceWithNew) onReplaceWithNew(rightValue);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth PaperProps={{ sx: { minHeight: '400px' } }}>
      <DialogTitle>
        <Typography variant="h6">Merge with LLM description</Typography>
        <Typography variant="body2" color="text.secondary">
          Compare current content with the new description. Copy/paste between boxes to merge, or choose an action below.
        </Typography>
      </DialogTitle>
      <DialogContent>
        <Box sx={{ display: 'flex', gap: 2, flexDirection: { xs: 'column', md: 'row' }, mt: 1 }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Current content (editable)
            </Typography>
            <TextField
              fullWidth
              multiline
              minRows={10}
              maxRows={20}
              value={leftValue}
              onChange={(e) => setLeftValue(e.target.value)}
              variant="outlined"
              size="small"
              placeholder="No content yet"
              inputProps={{ 'aria-label': 'Current content' }}
            />
          </Box>
          <Box sx={{ flex: 1 }}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              New LLM description (read-only)
            </Typography>
            <TextField
              fullWidth
              multiline
              minRows={10}
              maxRows={20}
              value={rightValue}
              variant="outlined"
              size="small"
              placeholder="No description"
              InputProps={{ readOnly: true }}
              inputProps={{ 'aria-label': 'New LLM description' }}
            />
          </Box>
        </Box>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button startIcon={<Close />} onClick={handleKeepOld}>
          Keep Old
        </Button>
        <Button startIcon={<SwapHoriz />} variant="contained" onClick={handleReplaceWithNew}>
          Replace with New
        </Button>
        <Button onClick={onClose}>Cancel</Button>
      </DialogActions>
    </Dialog>
  );
};

export default DiffMergeDialog;
