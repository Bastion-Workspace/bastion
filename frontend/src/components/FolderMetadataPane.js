import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Chip,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  IconButton,
  Divider,
  Alert,
  CircularProgress,
  Autocomplete,
  Stack,
  Paper,
  FormControlLabel,
  Checkbox
} from '@mui/material';
import {
  Close,
  Save,
  Add,
  Tag,
  Category,
  Folder
} from '@mui/icons-material';
import { useMutation, useQueryClient } from 'react-query';
import apiService from '../services/apiService';

const FolderMetadataPane = ({ 
  folder, 
  open, 
  onClose, 
  position = { x: 0, y: 0 }
}) => {
  const queryClient = useQueryClient();
  
  // Local state for form fields
  const [category, setCategory] = useState('');
  const [tags, setTags] = useState([]);
  const [newTag, setNewTag] = useState('');
  const [inheritTags, setInheritTags] = useState(true);
  
  // Update mutation
  const updateMetadataMutation = useMutation(
    (data) => apiService.put(`/api/folders/${folder?.folder_id}/metadata`, data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['folders', 'tree']);
        queryClient.invalidateQueries(['folders', 'contents']);
        onClose();
      },
    }
  );
  
  // Initialize form when folder changes
  useEffect(() => {
    if (folder) {
      setCategory(folder.category || '');
      setTags(folder.tags || []);
      setInheritTags(folder.inherit_tags !== false); // Default to true
    }
  }, [folder]);
  
  const handleAddTag = () => {
    if (newTag.trim() && !tags.includes(newTag.trim())) {
      setTags([...tags, newTag.trim()]);
      setNewTag('');
    }
  };
  
  const handleRemoveTag = (tagToRemove) => {
    setTags(tags.filter(tag => tag !== tagToRemove));
  };
  
  const handleSave = () => {
    if (!folder) return;
    
    const updateData = {
      category: category || null,
      tags: tags,
      inherit_tags: inheritTags
    };
    
    updateMetadataMutation.mutate(updateData);
  };
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddTag();
    }
  };
  
  if (!open || !folder) return null;
  
  // **ROOSEVELT FIX**: Calculate position to keep pane within viewport
  const calculatePosition = () => {
    const paneWidth = 400;
    const paneMaxHeight = window.innerHeight * 0.8; // 80vh
    
    let top = position.y;
    let left = position.x;
    
    // Adjust horizontal position if pane would go off right edge
    if (left + paneWidth > window.innerWidth) {
      left = Math.max(10, window.innerWidth - paneWidth - 10);
    }
    
    // Adjust vertical position if pane would go off bottom
    if (top + paneMaxHeight > window.innerHeight) {
      top = Math.max(10, window.innerHeight - paneMaxHeight - 10);
    }
    
    return { top, left };
  };
  
  const adjustedPosition = calculatePosition();
  
  return (
    <Paper
      elevation={8}
      sx={{
        position: 'fixed',
        top: adjustedPosition.top,
        left: adjustedPosition.left,
        width: 400,
        maxHeight: '80vh',
        overflow: 'auto',
        zIndex: 1300,
        p: 3,
        display: 'flex',
        flexDirection: 'column',
        gap: 2
      }}
    >
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Folder color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Folder Metadata
          </Typography>
        </Box>
        <IconButton size="small" onClick={onClose}>
          <Close />
        </IconButton>
      </Box>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        {folder.name}
      </Typography>
      
      <Alert severity="info" sx={{ mb: 2 }}>
        📋 Documents uploaded to this folder will automatically inherit these tags!
      </Alert>
      
      <Divider sx={{ mb: 2 }} />
      
      {/* Form Fields */}
      <Stack spacing={2}>
        {/* Category */}
        <FormControl fullWidth size="small">
          <InputLabel>Category</InputLabel>
          <Select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            label="Category"
            startAdornment={<Category fontSize="small" sx={{ mr: 1, color: 'action.active' }} />}
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            <MenuItem value="technical">Technical</MenuItem>
            <MenuItem value="academic">Academic</MenuItem>
            <MenuItem value="business">Business</MenuItem>
            <MenuItem value="legal">Legal</MenuItem>
            <MenuItem value="medical">Medical</MenuItem>
            <MenuItem value="literature">Literature</MenuItem>
            <MenuItem value="manual">Manual</MenuItem>
            <MenuItem value="reference">Reference</MenuItem>
            <MenuItem value="research">Research</MenuItem>
            <MenuItem value="personal">Personal</MenuItem>
            <MenuItem value="news">News</MenuItem>
            <MenuItem value="education">Education</MenuItem>
            <MenuItem value="other">Other</MenuItem>
          </Select>
        </FormControl>
        
        {/* Tags */}
        <Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Tag fontSize="small" color="action" />
            <Typography variant="body2" fontWeight={500}>
              Tags
            </Typography>
          </Box>
          
          {/* Existing Tags */}
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1 }}>
            {tags.map((tag) => (
              <Chip
                key={tag}
                label={tag}
                size="small"
                onDelete={() => handleRemoveTag(tag)}
                color="primary"
                variant="outlined"
              />
            ))}
          </Box>
          
          {/* Add New Tag */}
          <Box sx={{ display: 'flex', gap: 1 }}>
            <TextField
              placeholder="Add tag..."
              size="small"
              value={newTag}
              onChange={(e) => setNewTag(e.target.value)}
              onKeyPress={handleKeyPress}
              sx={{ flexGrow: 1 }}
            />
            <Button
              variant="outlined"
              size="small"
              onClick={handleAddTag}
              disabled={!newTag.trim()}
              startIcon={<Add />}
            >
              Add
            </Button>
          </Box>
        </Box>
        
        {/* Inherit Tags Toggle */}
        <FormControlLabel
          control={
            <Checkbox
              checked={inheritTags}
              onChange={(e) => setInheritTags(e.target.checked)}
              color="primary"
            />
          }
          label={
            <Typography variant="body2">
              Automatically apply these tags to uploaded documents
            </Typography>
          }
        />
      </Stack>
      
      <Divider sx={{ my: 2 }} />
      
      {/* Actions */}
      <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
        <Button onClick={onClose} variant="outlined">
          Cancel
        </Button>
        <Button
          onClick={handleSave}
          variant="contained"
          disabled={updateMetadataMutation.isLoading}
          startIcon={updateMetadataMutation.isLoading ? <CircularProgress size={16} /> : <Save />}
        >
          {updateMetadataMutation.isLoading ? 'Saving...' : 'Save Changes'}
        </Button>
      </Box>
      
      {/* Error Display */}
      {updateMetadataMutation.error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          Failed to update folder metadata: {updateMetadataMutation.error.message}
        </Alert>
      )}
    </Paper>
  );
};

export default FolderMetadataPane;


