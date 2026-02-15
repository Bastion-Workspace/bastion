/**
 * Org Quick Capture Component
 * Emacs-style quick capture to inbox.org via global hotkey
 * 
 * Emacs-style capture: Ctrl+Shift+C to capture from anywhere.
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  ToggleButtonGroup,
  ToggleButton,
  Box,
  Typography,
  Chip,
  Stack,
  Alert,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  CheckBox,
  Note,
  MenuBook,
  Group,
  Close,
  Send,
  CalendarToday,
  Flag,
  Person
} from '@mui/icons-material';
import apiService from '../services/apiService';
import OrgContactCapture from './OrgContactCapture';

const OrgQuickCapture = ({ open, onClose }) => {
  // State
  const [content, setContent] = useState('');
  const [templateType, setTemplateType] = useState('note');
  const [tags, setTags] = useState([]);
  const [tagInput, setTagInput] = useState('');
  const [priority, setPriority] = useState(null);
  const [scheduled, setScheduled] = useState('');
  const [deadline, setDeadline] = useState('');
  const [entryDate, setEntryDate] = useState(new Date().toISOString().split('T')[0]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [inboxWarning, setInboxWarning] = useState(null);
  
  // Ref for auto-focus
  const contentRef = useRef(null);
  
  // Template definitions
  const templates = [
    { type: 'note', label: 'Note', icon: <Note />, description: 'Quick note' },
    { type: 'todo', label: 'TODO', icon: <CheckBox />, description: 'Task item' },
    { type: 'contact', label: 'Contact', icon: <Person />, description: 'Contact info' },
    { type: 'journal', label: 'Journal', icon: <MenuBook />, description: 'Journal entry' },
    { type: 'meeting', label: 'Meeting', icon: <Group />, description: 'Meeting notes' }
  ];
  
  // Auto-focus content field when dialog opens
  useEffect(() => {
    if (open && contentRef.current) {
      setTimeout(() => contentRef.current?.focus(), 100);
    }
  }, [open]);
  
  // When dialog opens: refresh journal entry date to today and check inbox
  useEffect(() => {
    if (open) {
      setEntryDate(new Date().toISOString().split('T')[0]);
      checkInboxStatus();
    }
  }, [open]);
  
  // Check inbox status
  const checkInboxStatus = async () => {
    try {
      const response = await apiService.get('/api/org/check-inbox');
      
      if (response.multiple_inboxes) {
        setInboxWarning(response.warning);
        console.warn('⚠️ Multiple inbox.org files detected:', response.all_locations);
      } else {
        setInboxWarning(null);
      }
    } catch (err) {
      console.error('❌ Failed to check inbox status:', err);
      // Don't block capture on check failure
    }
  };
  
  // Reset state when dialog closes
  useEffect(() => {
    if (!open) {
      setTimeout(() => {
        setContent('');
        setTemplateType('note');
        setTags([]);
        setTagInput('');
        setPriority(null);
        setScheduled('');
        setDeadline('');
        setEntryDate(new Date().toISOString().split('T')[0]);
        setShowAdvanced(false);
        setError(null);
        setSuccess(null);
      }, 300); // Delay to allow dialog close animation
    }
  }, [open]);
  
  // Handle template change
  const handleTemplateChange = (event, newTemplate) => {
    if (newTemplate !== null) {
      setTemplateType(newTemplate);
      // Show advanced options for TODO
      if (newTemplate === 'todo') {
        setShowAdvanced(true);
      }
    }
  };
  
  // Handle tag input
  const handleTagKeyPress = (e) => {
    if (e.key === 'Enter' && tagInput.trim()) {
      e.preventDefault();
      if (!tags.includes(tagInput.trim())) {
        setTags([...tags, tagInput.trim()]);
      }
      setTagInput('');
    }
  };
  
  // Handle tag input key down (for Ctrl+Enter)
  const handleTagKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleCapture();
    }
  };
  
  // Remove tag
  const handleRemoveTag = (tagToRemove) => {
    setTags(tags.filter(tag => tag !== tagToRemove));
  };
  
  // Handle capture (Enter key in content field)
  const handleContentKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleCapture();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    }
  };
  
  // Perform capture
  const handleCapture = async () => {
    if (!content.trim()) {
      setError('Content cannot be empty');
      return;
    }
    
    try {
      setCapturing(true);
      setError(null);
      
      const captureRequest = {
        content: content.trim(),
        template_type: templateType,
        tags: tags.length > 0 ? tags : null,
        priority: priority || null,
        scheduled: scheduled || null,
        deadline: deadline || null,
        entry_date: templateType === 'journal' ? entryDate : null
      };
      
      const response = await apiService.post('/api/org/capture', captureRequest);
      
      if (response.success) {
        // Use backend message, but make it cleaner for journal entries
        let successMessage = response.message || 'Captured successfully';
        
        // For journal entries, show "Captured to Journal" instead of filename
        if (templateType === 'journal' && successMessage.includes('journal')) {
          successMessage = '✅ Captured to Journal';
        } else if (successMessage.includes('Successfully captured to')) {
          // Extract just the filename for cleaner display
          const filename = successMessage.replace('Successfully captured to ', '');
          successMessage = `✅ Captured to ${filename}`;
        } else {
          successMessage = `✅ ${successMessage}`;
        }
        
        setSuccess(successMessage);

        if (templateType === 'journal') {
          window.dispatchEvent(new CustomEvent('journalDocumentUpdated'));
        }

        setTimeout(() => {
          onClose();
        }, 1000);
      } else {
        setError(response.message || 'Failed to capture');
      }
    } catch (err) {
      console.error('❌ Capture error:', err);
      setError(err.message || 'Failed to capture to inbox.org');
    } finally {
      setCapturing(false);
    }
  };
  
  // Route to specialized contact form if contact template selected
  if (templateType === 'contact') {
    return <OrgContactCapture open={open} onClose={onClose} />;
  }

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          minHeight: '400px'
        }
      }}
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Note color="primary" />
          <Typography variant="h6">Quick Capture</Typography>
          <Typography variant="caption" color="text.secondary">
            (Ctrl+Shift+C)
          </Typography>
        </Box>
        <IconButton size="small" onClick={onClose}>
          <Close />
        </IconButton>
      </DialogTitle>
      
      <DialogContent>
        {/* Template Selection */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Template
          </Typography>
          <ToggleButtonGroup
            value={templateType}
            exclusive
            onChange={handleTemplateChange}
            fullWidth
            size="small"
          >
            {templates.map((template) => (
              <ToggleButton 
                key={template.type} 
                value={template.type}
                sx={{ py: 1.5 }}
              >
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.5 }}>
                  {template.icon}
                  <Typography variant="caption">{template.label}</Typography>
                </Box>
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Box>
        
        {/* Content Input */}
        <TextField
          inputRef={contentRef}
          fullWidth
          multiline
          rows={4}
          label={templateType === 'todo' ? 'Task description' : 'Content'}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          onKeyDown={handleContentKeyDown}
          placeholder={
            templateType === 'todo' ? 'What needs to be done?' :
            templateType === 'journal' ? 'What happened today? Use %T (date+time), %t (date), %<%I:%M %p> (time e.g. 02:30 PM)' :
            templateType === 'meeting' ? 'Meeting topic or title' :
            'Quick note...'
          }
          helperText="Ctrl+Enter to capture, Esc to cancel"
          sx={{ mb: 2 }}
          disabled={capturing}
        />
        
        {/* Tags */}
        <Box sx={{ mb: 2 }}>
          <TextField
            fullWidth
            size="small"
            label="Tags (press Enter to add, Ctrl+Enter to capture)"
            value={tagInput}
            onChange={(e) => setTagInput(e.target.value)}
            onKeyPress={handleTagKeyPress}
            onKeyDown={handleTagKeyDown}
            placeholder="work, urgent, review..."
            disabled={capturing}
          />
          {tags.length > 0 && (
            <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
              {tags.map((tag) => (
                <Chip
                  key={tag}
                  label={tag}
                  size="small"
                  onDelete={() => handleRemoveTag(tag)}
                  disabled={capturing}
                />
              ))}
            </Box>
          )}
        </Box>
        
        {/* Date Picker (for Journal) */}
        {templateType === 'journal' && (
          <TextField
            fullWidth
            type="date"
            label="Entry Date"
            value={entryDate}
            onChange={(e) => setEntryDate(e.target.value)}
            InputLabelProps={{ shrink: true }}
            helperText="Select date for this journal entry (defaults to today)"
            margin="normal"
            sx={{ mb: 2 }}
            disabled={capturing}
          />
        )}
        
        {/* Advanced Options (for TODO) */}
        {templateType === 'todo' && (
          <Box sx={{ mb: 2 }}>
            {!showAdvanced ? (
              <Button
                size="small"
                onClick={() => setShowAdvanced(true)}
                disabled={capturing}
              >
                Show Advanced Options
              </Button>
            ) : (
              <Stack spacing={2}>
                {/* Priority */}
                <Box>
                  <Typography variant="subtitle2" sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Flag fontSize="small" />
                    Priority
                  </Typography>
                  <ToggleButtonGroup
                    value={priority}
                    exclusive
                    onChange={(e, val) => setPriority(val)}
                    size="small"
                    disabled={capturing}
                  >
                    <ToggleButton value="A">A</ToggleButton>
                    <ToggleButton value="B">B</ToggleButton>
                    <ToggleButton value="C">C</ToggleButton>
                    <ToggleButton value={null}>None</ToggleButton>
                  </ToggleButtonGroup>
                </Box>
                
                {/* Scheduled & Deadline */}
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <TextField
                    fullWidth
                    size="small"
                    type="date"
                    label="Scheduled"
                    value={scheduled}
                    onChange={(e) => setScheduled(e.target.value)}
                    InputLabelProps={{ shrink: true }}
                    disabled={capturing}
                  />
                  <TextField
                    fullWidth
                    size="small"
                    type="date"
                    label="Deadline"
                    value={deadline}
                    onChange={(e) => setDeadline(e.target.value)}
                    InputLabelProps={{ shrink: true }}
                    disabled={capturing}
                  />
                </Box>
              </Stack>
            )}
          </Box>
        )}
        
        {/* Inbox Warning */}
        {inboxWarning && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            {inboxWarning}
            <br />
            <small>Configure inbox location in Settings → Org-Mode Settings</small>
          </Alert>
        )}
        
        {/* Error/Success Messages */}
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
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose} disabled={capturing}>
          Cancel (Esc)
        </Button>
        <Button
          variant="contained"
          onClick={handleCapture}
          disabled={!content.trim() || capturing}
          startIcon={<Send />}
        >
          {capturing ? 'Capturing...' : 'Capture (Ctrl+Enter)'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default OrgQuickCapture;

