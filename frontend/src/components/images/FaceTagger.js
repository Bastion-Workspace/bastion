/**
 * Face Tagger Component
 * Dialog for detecting and tagging faces in images with bounding boxes
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
  Typography,
  Alert,
  CircularProgress,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton
} from '@mui/material';
import {
  Face,
  Close,
  PersonAdd,
  CheckCircle
} from '@mui/icons-material';
import apiService from '../../services/apiService';

const FaceTagger = ({ open, onClose, documentId, imageUrl }) => {
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [detectedFaces, setDetectedFaces] = useState([]);
  const [selectedFace, setSelectedFace] = useState(null);
  const [identityName, setIdentityName] = useState('');
  const [tagging, setTagging] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [showTagSuggestion, setShowTagSuggestion] = useState(null);
  const [addingToTags, setAddingToTags] = useState(false);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const [displayDimensions, setDisplayDimensions] = useState({ width: 0, height: 0 });
  const imageRef = React.useRef(null);

  // Load existing faces when dialog opens
  useEffect(() => {
    if (open && documentId) {
      loadDetectedFaces();
    } else {
      resetState();
    }
  }, [open, documentId]);

  // Calculate display dimensions when image loads
  useEffect(() => {
    if (imageRef.current && imageDimensions.width > 0) {
      const img = imageRef.current;
      const displayWidth = img.clientWidth;
      const displayHeight = img.clientHeight;
      setDisplayDimensions({ width: displayWidth, height: displayHeight });
    }
  }, [imageDimensions, detectedFaces]);

  const loadDetectedFaces = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiService.get(`/api/documents/${documentId}/faces`);
      
      if (response.success && response.faces) {
        setDetectedFaces(response.faces);
      }
    } catch (err) {
      console.error('Failed to load detected faces:', err);
      // Not an error - just no faces detected yet
      setDetectedFaces([]);
    } finally {
      setLoading(false);
    }
  };

  const resetState = () => {
    setDetectedFaces([]);
    setSelectedFace(null);
    setIdentityName('');
    setError(null);
    setSuggestions([]);
    setShowTagSuggestion(null);
    setImageDimensions({ width: 0, height: 0 });
    setDisplayDimensions({ width: 0, height: 0 });
  };

  const handleAddToTags = async () => {
    if (!showTagSuggestion) return;
    
    setAddingToTags(true);
    setError(null);
    
    try {
      const response = await apiService.post(
        `/api/documents/${showTagSuggestion.documentId}/add-identity-tag`,
        { identity_name: showTagSuggestion.identityName }
      );
      
      if (response.success) {
        setShowTagSuggestion(null);
        // Optionally reload document metadata to show updated tags
      } else {
        setError(response.error || 'Failed to add identity to tags');
      }
    } catch (err) {
      console.error('Failed to add identity to tags:', err);
      setError(err.response?.data?.detail || 'Failed to add identity to tags');
    } finally {
      setAddingToTags(false);
    }
  };

  const analyzeFaces = async () => {
    setAnalyzing(true);
    setError(null);
    
    try {
      const response = await apiService.post(`/api/documents/${documentId}/analyze-faces`);
      
      if (response.success) {
        // Update faces list with IDs from database
        setDetectedFaces(response.faces || []);
        setImageDimensions({
          width: response.image_width,
          height: response.image_height
        });
      } else {
        setError(response.error || 'Face detection failed');
      }
    } catch (err) {
      console.error('Failed to analyze faces:', err);
      if (err.response?.status === 403) {
        setError('Vision features are disabled. Enable them in settings to use face detection.');
      } else if (err.response?.status === 500 && err.response?.data?.detail?.includes('unavailable')) {
        setError('Vision service is unavailable. The service may not be running.');
      } else {
        setError(err.response?.data?.detail || 'Failed to analyze faces');
      }
    } finally {
      setAnalyzing(false);
    }
  };

  const handleFaceClick = (face) => {
    setSelectedFace(face);
    // Pre-fill with suggested identity if available, otherwise use existing identity or empty
    setIdentityName(face.identity_name || face.suggested_identity || '');
    setSuggestions([]);
  };

  const tagFace = async () => {
    if (!selectedFace || !identityName.trim()) {
      return;
    }

    setTagging(true);
    setError(null);

    try {
      const faceId = selectedFace.id;
      
      if (!faceId || typeof faceId !== 'number') {
        setError('Invalid face ID. Please analyze faces again.');
        setTagging(false);
        return;
      }

      // Fetch face encoding from backend
      // The backend will get it from the database
      // For now, we'll pass an empty encoding and let backend fetch it
      const response = await apiService.put(
        `/api/documents/${documentId}/faces/${faceId}/tag`,
        {
          identity_name: identityName.trim(),
          face_encoding: [] // Backend will fetch from database
        }
      );

      if (response.success) {
        // Reload faces to get updated data
        await loadDetectedFaces();
        
        if (response.suggestions && response.suggestions.length > 0) {
          setSuggestions(response.suggestions);
        }
        
        // Show suggestion to add identity to metadata.json tags
        setShowTagSuggestion({
          identityName: identityName.trim(),
          documentId: documentId
        });
        
        setSelectedFace(null);
        setIdentityName('');
      } else {
        setError('Failed to tag face');
      }
    } catch (err) {
      console.error('Failed to tag face:', err);
      setError(err.response?.data?.detail || 'Failed to tag face');
    } finally {
      setTagging(false);
    }
  };

  // Calculate bounding box position relative to displayed image
  const getBoundingBoxStyle = (face) => {
    if (imageDimensions.width === 0 || displayDimensions.width === 0) {
      return { display: 'none' };
    }

    const scaleX = displayDimensions.width / imageDimensions.width;
    const scaleY = displayDimensions.height / imageDimensions.height;

    // Color coding: green for confirmed, orange for suggested, blue for untagged
    let borderColor = '#1976d2'; // Blue for untagged
    let bgColor = 'transparent';
    
    if (face.identity_confirmed) {
      borderColor = '#2e7d32'; // Green for confirmed
      bgColor = selectedFace?.id === face.id ? 'rgba(46, 125, 50, 0.1)' : 'transparent';
    } else if (face.suggested_identity) {
      borderColor = '#ed6c02'; // Orange for suggested
      bgColor = selectedFace?.id === face.id ? 'rgba(237, 108, 2, 0.1)' : 'transparent';
    } else {
      bgColor = selectedFace?.id === face.id ? 'rgba(25, 118, 210, 0.1)' : 'transparent';
    }

    return {
      position: 'absolute',
      left: `${face.bbox_x * scaleX}px`,
      top: `${face.bbox_y * scaleY}px`,
      width: `${face.bbox_width * scaleX}px`,
      height: `${face.bbox_height * scaleY}px`,
      border: selectedFace?.id === face.id ? `3px solid ${borderColor}` : `2px solid ${borderColor}`,
      backgroundColor: bgColor,
      cursor: 'pointer',
      zIndex: 10
    };
  };

  const getBoundingBoxLabel = (face, index) => {
    if (face.identity_name) {
      return face.identity_name;
    }
    if (face.suggested_identity) {
      return `${face.suggested_identity}? (${face.suggested_confidence}%)`;
    }
    return `Person ${index + 1}`;
  };

  const confirmSuggestedIdentity = async (face) => {
    if (!face.suggested_identity) return;
    
    setSelectedFace(face);
    setIdentityName(face.suggested_identity);
    
    // Auto-tag with suggested identity
    setTagging(true);
    setError(null);
    
    try {
      const response = await apiService.put(
        `/api/documents/${documentId}/faces/${face.id}/tag`,
        {
          identity_name: face.suggested_identity,
          face_encoding: []
        }
      );
      
      if (response.success) {
        await loadDetectedFaces();
        
        if (response.suggestions && response.suggestions.length > 0) {
          setSuggestions(response.suggestions);
        }
        
        // Only show tag suggestion dialog if tags were NOT auto-synced
        // Backend automatically adds identity to tags when face is confirmed
        if (!response.tags_auto_synced) {
          setShowTagSuggestion({
            identityName: face.suggested_identity,
            documentId: documentId
          });
        }
        
        setSelectedFace(null);
        setIdentityName('');
      } else {
        setError('Failed to confirm identity');
      }
    } catch (err) {
      console.error('Failed to confirm identity:', err);
      setError(err.response?.data?.detail || 'Failed to confirm identity');
    } finally {
      setTagging(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box display="flex" alignItems="center" gap={1}>
          <Face />
          <Typography variant="h6">Tag Faces</Typography>
        </Box>
      </DialogTitle>
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {showTagSuggestion && (
          <Alert 
            severity="info" 
            sx={{ mb: 2 }}
            action={
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  size="small"
                  color="primary"
                  onClick={handleAddToTags}
                  disabled={addingToTags}
                >
                  {addingToTags ? 'Adding...' : 'Add to Tags'}
                </Button>
                <Button
                  size="small"
                  onClick={() => setShowTagSuggestion(null)}
                  disabled={addingToTags}
                >
                  Skip
                </Button>
              </Box>
            }
          >
            Added '{showTagSuggestion.identityName}' to face detection. Add to image tags?
          </Alert>
        )}

        <Box sx={{ mb: 2 }}>
          <Button
            variant="contained"
            startIcon={<Face />}
            onClick={analyzeFaces}
            disabled={analyzing || loading}
            sx={{ mb: 2 }}
          >
            {analyzing ? 'Analyzing Faces...' : 'Analyze Faces'}
          </Button>
        </Box>

        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
          <Box sx={{ position: 'relative', display: 'inline-block' }}>
            <Box
              ref={imageRef}
              component="img"
              src={imageUrl}
              alt="Image for face tagging"
              onLoad={(e) => {
                const img = e.target;
                setDisplayDimensions({ width: img.clientWidth, height: img.clientHeight });
              }}
              sx={{
                maxWidth: '100%',
                maxHeight: '400px',
                display: 'block'
              }}
            />
            
            {/* Bounding boxes overlay */}
            {detectedFaces.map((face, idx) => (
              <Box
                key={face.id || idx}
                onClick={() => handleFaceClick(face)}
                sx={getBoundingBoxStyle(face)}
              >
              <Box
                sx={{
                  position: 'absolute',
                  top: -25,
                  left: 0,
                  background: face.identity_confirmed ? '#2e7d32' : face.suggested_identity ? '#ed6c02' : '#1976d2',
                  color: 'white',
                  padding: '2px 6px',
                  fontSize: '12px',
                  borderRadius: '4px',
                  whiteSpace: 'nowrap'
                }}
              >
                {getBoundingBoxLabel(face, idx)}
              </Box>
              </Box>
            ))}
          </Box>
        </Box>

        {selectedFace && (
          <Paper sx={{ p: 2, mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Tag Face: {getBoundingBoxLabel(selectedFace, detectedFaces.indexOf(selectedFace))}
            </Typography>
            <TextField
              fullWidth
              label="Person Name"
              value={identityName}
              onChange={(e) => setIdentityName(e.target.value)}
              placeholder="e.g., Steve McQueen"
              sx={{ mb: 2 }}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && identityName.trim()) {
                  tagFace();
                }
              }}
            />
            <Button
              variant="contained"
              startIcon={<PersonAdd />}
              onClick={tagFace}
              disabled={!identityName.trim() || tagging}
              fullWidth
            >
              {tagging ? 'Tagging...' : 'Tag Face'}
            </Button>

            {suggestions.length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="caption" color="text.secondary">
                  Found {suggestions.length} similar face(s) in other images:
                </Typography>
                <List dense>
                  {suggestions.map((suggestion, idx) => (
                    <ListItem key={idx}>
                      <ListItemText
                        primary={suggestion.title || suggestion.filename}
                        secondary={`Confidence: ${suggestion.confidence}%`}
                      />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
          </Paper>
        )}

        {detectedFaces.length > 0 && !selectedFace && (
          <Paper sx={{ p: 2, mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Detected Faces ({detectedFaces.length})
            </Typography>
            <List dense>
              {detectedFaces.map((face, idx) => (
                <ListItem
                  key={face.id || idx}
                  button
                  onClick={() => handleFaceClick(face)}
                  selected={selectedFace?.id === face.id}
                >
                  <ListItemText
                    primary={getBoundingBoxLabel(face, idx)}
                    secondary={
                      face.identity_confirmed ? 'Tagged' : 
                      face.suggested_identity ? `Suggested: ${face.suggested_identity} (${face.suggested_confidence}% match)` :
                      'Not tagged'
                    }
                  />
                  <ListItemSecondaryAction>
                    {face.identity_confirmed ? (
                      <CheckCircle color="success" fontSize="small" />
                    ) : face.suggested_identity ? (
                      <Button
                        size="small"
                        variant="contained"
                        onClick={(e) => {
                          e.stopPropagation();
                          confirmSuggestedIdentity(face);
                        }}
                        disabled={tagging}
                      >
                        Confirm
                      </Button>
                    ) : null}
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          </Paper>
        )}

        {loading && (
          <Box display="flex" justifyContent="center" p={2}>
            <CircularProgress size={24} />
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} startIcon={<Close />}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default FaceTagger;
