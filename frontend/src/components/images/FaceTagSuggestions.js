/**
 * Face Tag Suggestions Component
 * Shows suggestions when metadata tags match known identities and there are untagged faces
 */

import React, { useState, useEffect } from 'react';
import {
  Alert,
  Button,
  Box,
  Typography
} from '@mui/material';
import {
  Face,
  Tag
} from '@mui/icons-material';
import apiService from '../../services/apiService';

const FaceTagSuggestions = ({ documentId, onOpenFaceTagger }) => {
  const [suggestions, setSuggestions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (documentId) {
      loadSuggestions();
    } else {
      setSuggestions(null);
    }
  }, [documentId]);

  const loadSuggestions = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiService.get(
        `/api/documents/${documentId}/suggest-face-tags`
      );
      
      if (response.success && response.has_suggestions) {
        setSuggestions(response);
      } else {
        setSuggestions(null);
      }
    } catch (err) {
      console.error('Failed to load face tag suggestions:', err);
      setError(err.response?.data?.detail || 'Failed to load suggestions');
      setSuggestions(null);
    } finally {
      setLoading(false);
    }
  };

  if (loading || !suggestions || !suggestions.has_suggestions) {
    return null;
  }

  const tagNames = suggestions.suggestions.map(s => s.tag).join(', ');

  return (
    <Alert 
      severity="info" 
      icon={<Face />}
      sx={{ mb: 2 }}
      action={
        <Button
          size="small"
          color="primary"
          startIcon={<Tag />}
          onClick={() => onOpenFaceTagger && onOpenFaceTagger()}
        >
          Tag Faces
        </Button>
      }
    >
      <Box>
        <Typography variant="body2" component="div">
          We found {suggestions.untagged_faces_count} untagged face{suggestions.untagged_faces_count !== 1 ? 's' : ''}. 
          You tagged this image with {tagNames}. Would you like to identify them?
        </Typography>
      </Box>
    </Alert>
  );
};

export default FaceTagSuggestions;
