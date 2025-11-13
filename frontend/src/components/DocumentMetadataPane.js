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
  Paper
} from '@mui/material';
import {
  Close,
  Save,
  Add,
  Tag,
  Title,
  Description,
  Person,
  Category,
  CalendarToday,
  Edit
} from '@mui/icons-material';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import apiService from '../services/apiService';

const DocumentMetadataPane = ({ 
  document, 
  open, 
  onClose, 
  anchorEl = null,
  position = { x: 0, y: 0 }
}) => {
  const queryClient = useQueryClient();
  
  // Local state for form fields
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [author, setAuthor] = useState('');
  const [category, setCategory] = useState('');
  const [tags, setTags] = useState([]);
  const [newTag, setNewTag] = useState('');
  const [publicationDate, setPublicationDate] = useState(null);
  
  // Available categories and tags for autocomplete
  const [availableCategories, setAvailableCategories] = useState([]);
  const [availableTags, setAvailableTags] = useState([]);
  
  // Update mutation
  const updateMetadataMutation = useMutation(
    (data) => apiService.put(`/api/documents/${document?.document_id}/metadata`, data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['folders', 'tree']);
        queryClient.invalidateQueries(['documents', 'root']);
        onClose();
      },
    }
  );
  
  // Load available categories and tags
  const { data: categoriesData } = useQuery(
    ['document-categories'],
    () => apiService.get('/api/documents/categories'),
    {
      enabled: open,
      staleTime: 300000, // 5 minutes
    }
  );
  
  // Initialize form when document changes
  useEffect(() => {
    if (document) {
      setTitle(document.title || '');
      setDescription(document.description || '');
      setAuthor(document.author || '');
      setCategory(document.category || '');
      setTags(document.tags || []);
      setPublicationDate(document.publication_date ? new Date(document.publication_date) : null);
    }
  }, [document]);
  
  // Update available categories and tags
  useEffect(() => {
    if (categoriesData) {
      setAvailableCategories(categoriesData.categories?.map(cat => cat.category) || []);
      setAvailableTags(categoriesData.tags?.map(tag => tag.tag) || []);
    }
  }, [categoriesData]);
  
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
    if (!document) return;
    
    const updateData = {
      title: title.trim() || null,
      description: description.trim() || null,
      author: author.trim() || null,
      category: category || null,
      tags: tags,
      publication_date: publicationDate ? publicationDate.toISOString().split('T')[0] : null
    };
    
    updateMetadataMutation.mutate(updateData);
  };
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddTag();
    }
  };
  
  if (!open || !document) return null;
  
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
        top: anchorEl ? undefined : adjustedPosition.top,
        left: anchorEl ? undefined : adjustedPosition.left,
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
          <Edit color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Document Metadata
          </Typography>
        </Box>
        <IconButton size="small" onClick={onClose}>
          <Close />
        </IconButton>
      </Box>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        {document.filename}
      </Typography>
      
      <Divider sx={{ mb: 2 }} />
      
      {/* Form Fields */}
      <Stack spacing={2}>
        {/* Title */}
        <TextField
          label="Title"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          fullWidth
          size="small"
          InputProps={{
            startAdornment: <Title fontSize="small" sx={{ mr: 1, color: 'action.active' }} />
          }}
        />
        
        {/* Description */}
        <TextField
          label="Description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          fullWidth
          multiline
          rows={3}
          size="small"
          InputProps={{
            startAdornment: <Description fontSize="small" sx={{ mr: 1, color: 'action.active' }} />
          }}
        />
        
        {/* Author */}
        <TextField
          label="Author"
          value={author}
          onChange={(e) => setAuthor(e.target.value)}
          fullWidth
          size="small"
          InputProps={{
            startAdornment: <Person fontSize="small" sx={{ mr: 1, color: 'action.active' }} />
          }}
        />
        
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
            {availableCategories.map((cat) => (
              <MenuItem key={cat} value={cat}>
                {cat}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        
        {/* Publication Date */}
        <LocalizationProvider dateAdapter={AdapterDateFns}>
          <DatePicker
            label="Publication Date"
            value={publicationDate}
            onChange={setPublicationDate}
            renderInput={(params) => (
              <TextField
                {...params}
                fullWidth
                size="small"
                InputProps={{
                  ...params.InputProps,
                  startAdornment: <CalendarToday fontSize="small" sx={{ mr: 1, color: 'action.active' }} />
                }}
              />
            )}
          />
        </LocalizationProvider>
        
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
            <Autocomplete
              freeSolo
              options={availableTags.filter(tag => !tags.includes(tag))}
              value={newTag}
              onChange={(_, value) => setNewTag(value || '')}
              onKeyPress={handleKeyPress}
              renderInput={(params) => (
                <TextField
                  {...params}
                  placeholder="Add tag..."
                  size="small"
                  sx={{ flexGrow: 1 }}
                />
              )}
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
          Failed to update metadata: {updateMetadataMutation.error.message}
        </Alert>
      )}
    </Paper>
  );
};

export default DocumentMetadataPane; 