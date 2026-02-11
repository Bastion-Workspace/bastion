/**
 * Image Metadata Modal
 * Dialog for editing searchable metadata for images
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  Typography,
  Chip,
  Stack,
  Alert,
  CircularProgress,
  FormControlLabel,
  Switch,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Tabs,
  Tab,
  Slider,
  IconButton
} from '@mui/material';
import {
  ImageSearch,
  Close,
  Save,
  Face,
  AutoAwesome,
  Category,
  PersonAdd,
  CheckCircle,
  CenterFocusStrong,
  AddBox,
  Delete as DeleteIcon,
  Block as BlockIcon,
  Edit as EditIcon
} from '@mui/icons-material';
import apiService from '../../services/apiService';
import { useCapabilities } from '../../contexts/CapabilitiesContext';
import DiffMergeDialog from './DiffMergeDialog';
import ObjectAnnotationCanvas from './ObjectAnnotationCanvas';

const IMAGE_TYPES = [
  { value: 'comic', label: 'Comic' },
  { value: 'artwork', label: 'Artwork' },
  { value: 'meme', label: 'Meme' },
  { value: 'screenshot', label: 'Screenshot' },
  { value: 'medical', label: 'Medical' },
  { value: 'documentation', label: 'Documentation' },
  { value: 'maps', label: 'Maps' },
  { value: 'photo', label: 'Photo' },
  { value: 'other', label: 'Other' }
];

const ImageMetadataModal = ({ open, onClose, documentId, filename, imageUrl, onOpenFaceTagger }) => {
  const { has: hasCapability } = useCapabilities();
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [describeLoading, setDescribeLoading] = useState(false);
  const [diffDialogOpen, setDiffDialogOpen] = useState(false);
  const [pendingNewContent, setPendingNewContent] = useState('');
  const [pendingLLMMetadata, setPendingLLMMetadata] = useState(null);
  
  const [formData, setFormData] = useState({
    type: 'other',
    title: '',
    content: '',
    author: '',
    date: '',
    series: '',
    tags: [],
    // Type-specific fields
    location: '',
    event: '',
    medium: '',
    dimensions: '',
    body_part: '',
    modality: '',
    map_type: '',
    coordinates: '',
    application: '',
    platform: '',
    llm_metadata: null,
    faces: null,
    objects: null
  });
  
  const [tagInput, setTagInput] = useState('');
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [imageDisplaySize, setImageDisplaySize] = useState({ w: 0, h: 0 });
  const [editingObjectTagId, setEditingObjectTagId] = useState(null);
  const [editingObjectTagValue, setEditingObjectTagValue] = useState('');
  const imageRef = useRef(null);
  const [fetchedImageUrl, setFetchedImageUrl] = useState(null);
  const blobUrlRef = useRef(null);
  // Face tagging (unified with metadata)
  const [detectedFaces, setDetectedFaces] = useState([]);
  const [analyzingFaces, setAnalyzingFaces] = useState(false);
  const [selectedFace, setSelectedFace] = useState(null);
  const [identityName, setIdentityName] = useState('');
  const [tagging, setTagging] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [showTagSuggestion, setShowTagSuggestion] = useState(null);
  const [addingToTags, setAddingToTags] = useState(false);
  const [metadataExists, setMetadataExists] = useState(false);
  const [detectingObjects, setDetectingObjects] = useState(false);
  const [detectedObjects, setDetectedObjects] = useState([]);
  const [showObjectCanvas, setShowObjectCanvas] = useState(false);
  const [userAnnotations, setUserAnnotations] = useState([]);
  const [activeTab, setActiveTab] = useState(0);
  const [objectConfidence, setObjectConfidence] = useState(0.5);

  // Load existing metadata and detected faces when modal opens
  useEffect(() => {
    if (open && documentId) {
      loadMetadata();
      loadDetectedFaces();
      loadDetectedObjects();
      loadUserAnnotations();
    } else {
      resetForm();
      setDetectedFaces([]);
      setSelectedFace(null);
      setIdentityName('');
      setSuggestions([]);
      setShowTagSuggestion(null);
    }
  }, [open, documentId]);

  // Re-measure image when faces/objects change so overlay can draw (e.g. after Analyze Faces or new annotation)
  useEffect(() => {
    if ((detectedFaces.length > 0 || detectedObjects.length > 0 || userAnnotations.length > 0) && imageRef.current) {
      const el = imageRef.current;
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setImageDisplaySize({ w: rect.width, h: rect.height });
      }
    }
  }, [detectedFaces.length, detectedObjects.length, userAnnotations.length]);

  // When opened without imageUrl (e.g. from file tree), fetch image with auth so preview works (img src does not send JWT)
  useEffect(() => {
    if (!open || !filename || imageUrl) {
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
      setFetchedImageUrl(null);
      return;
    }
    let cancelled = false;
    const token = apiService.getToken();
    if (!token) {
      setFetchedImageUrl(null);
      return;
    }
    const apiUrl = `/api/images/${encodeURIComponent(filename)}`;
    fetch(apiUrl, { headers: { Authorization: `Bearer ${token}` } })
      .then((res) => {
        if (!res.ok) throw new Error(res.statusText);
        return res.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
        const blobUrl = URL.createObjectURL(blob);
        blobUrlRef.current = blobUrl;
        setFetchedImageUrl(blobUrl);
      })
      .catch(() => {
        if (!cancelled) setFetchedImageUrl(null);
      });
    return () => {
      cancelled = true;
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, [open, filename, imageUrl]);

  const loadMetadata = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiService.get(`/api/documents/${documentId}/image-metadata`);
      
      if (response.exists && response.metadata) {
        setMetadataExists(true);
        // Load existing metadata (including type-specific fields)
        setFormData({
          type: response.metadata.type || 'other',
          title: response.metadata.title || '',
          content: response.metadata.content || '',
          author: response.metadata.author || '',
          date: response.metadata.date || '',
          series: response.metadata.series || '',
          tags: response.metadata.tags || [],
          // Type-specific fields
          location: response.metadata.location || '',
          event: response.metadata.event || '',
          medium: response.metadata.medium || '',
          dimensions: response.metadata.dimensions || '',
          body_part: response.metadata.body_part || '',
          modality: response.metadata.modality || '',
          map_type: response.metadata.map_type || '',
          coordinates: response.metadata.coordinates || '',
          application: response.metadata.application || '',
          platform: response.metadata.platform || '',
          llm_metadata: response.metadata.llm_metadata || null,
          faces: response.metadata.faces || null,
          objects: response.metadata.objects || null
        });
      } else {
        // New metadata - use filename as default title
        setFormData({
          type: 'other',
          title: filename || '',
          content: '',
          author: '',
          date: '',
          series: '',
          tags: [],
          location: '',
          event: '',
          medium: '',
          dimensions: '',
          body_part: '',
          modality: '',
          map_type: '',
          coordinates: '',
        application: '',
        platform: '',
        llm_metadata: null,
        faces: null,
        objects: null
        });
      }
    } catch (err) {
      console.error('Failed to load image metadata:', err);
      setError('Failed to load metadata. You can still create new metadata.');
      setMetadataExists(false);
      // Set default values
      setFormData({
        type: 'other',
        title: filename || '',
        content: '',
        author: '',
        date: '',
        series: '',
        tags: [],
        location: '',
        event: '',
        medium: '',
        dimensions: '',
        body_part: '',
        modality: '',
        map_type: '',
        coordinates: '',
        application: '',
        platform: '',
        llm_metadata: null,
        faces: null,
        objects: null
      });
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      type: 'other',
      title: '',
      content: '',
      author: '',
      date: '',
      series: '',
      tags: [],
      location: '',
      event: '',
      medium: '',
      dimensions: '',
      body_part: '',
      modality: '',
      map_type: '',
      coordinates: '',
      application: '',
      platform: '',
      llm_metadata: null,
      faces: null,
      objects: null
    });
    setTagInput('');
    setError(null);
    setSuccess(null);
    setMetadataExists(false);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleTagInputKeyPress = (e) => {
    if (e.key === 'Enter' && tagInput.trim()) {
      e.preventDefault();
      addTag(tagInput.trim());
    }
  };

  const addTag = (tag) => {
    if (tag && !formData.tags.includes(tag)) {
      setFormData(prev => ({
        ...prev,
        tags: [...prev.tags, tag]
      }));
      setTagInput('');
    }
  };

  const removeTag = (tagToRemove) => {
    setFormData(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove)
    }));
  };

  const handleDescribeLLM = async () => {
    if (!documentId) return;
    setDescribeLoading(true);
    setError(null);
    try {
      const response = await apiService.post(`/api/documents/${documentId}/describe-image`);
      const description = response.description || '';
      const hasExisting = (formData.content || '').trim().length > 0;
      if (hasExisting) {
        setPendingNewContent(description);
        setPendingLLMMetadata({
          model: response.model_used,
          timestamp: new Date().toISOString(),
          confidence: response.confidence
        });
        setDiffDialogOpen(true);
      } else {
        setFormData(prev => ({
          ...prev,
          content: description
        }));
        if (response.model_used) {
          setFormData(prev => ({
            ...prev,
            llm_metadata: {
              model: response.model_used,
              timestamp: new Date().toISOString(),
              confidence: response.confidence
            }
          }));
        }
      }
    } catch (err) {
      console.error('Describe image failed:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to describe image');
    } finally {
      setDescribeLoading(false);
    }
  };

  const handleDiffReplaceWithNew = (newContent) => {
    setFormData(prev => ({
      ...prev,
      content: newContent,
      ...(pendingLLMMetadata && { llm_metadata: pendingLLMMetadata })
    }));
    setDiffDialogOpen(false);
    setPendingNewContent('');
    setPendingLLMMetadata(null);
  };

  const handleDiffKeepOld = () => {
    setDiffDialogOpen(false);
    setPendingNewContent('');
    setPendingLLMMetadata(null);
  };

  const loadDetectedFaces = async () => {
    if (!documentId) return;
    try {
      const response = await apiService.get(`/api/documents/${documentId}/faces`);
      if (response.success && response.faces) {
        setDetectedFaces(response.faces);
      } else {
        setDetectedFaces([]);
      }
    } catch (err) {
      setDetectedFaces([]);
    }
  };

  const loadDetectedObjects = async () => {
    if (!documentId) return;
    try {
      const response = await apiService.getDetectedObjects(documentId);
      if (response.success && response.objects) {
        setDetectedObjects(response.objects);
      } else {
        setDetectedObjects([]);
      }
    } catch (err) {
      setDetectedObjects([]);
    }
  };

  const handleConfirmObject = async (obj) => {
    if (!obj?.id) return;
    try {
      await apiService.confirmObjectDetection(obj.id);
      await loadDetectedObjects();
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || 'Failed to confirm object');
    }
  };

  const handleRejectObject = async (obj) => {
    if (!obj?.id) return;
    try {
      await apiService.updateDetectedObject(obj.id, { rejected: true });
      await loadDetectedObjects();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to reject object');
    }
  };

  const handleSaveObjectTag = async () => {
    if (editingObjectTagId == null) return;
    try {
      await apiService.updateDetectedObject(editingObjectTagId, {
        user_tag: editingObjectTagValue.trim() || null
      });
      await loadDetectedObjects();
      setEditingObjectTagId(null);
      setEditingObjectTagValue('');
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to update tag');
    }
  };

  const loadUserAnnotations = async () => {
    if (!documentId) return;
    try {
      const response = await apiService.getObjectAnnotations(documentId);
      if (response.success && response.annotations) {
        setUserAnnotations(response.annotations);
      } else {
        setUserAnnotations([]);
      }
    } catch (err) {
      setUserAnnotations([]);
    }
  };

  const handleDetectObjects = async () => {
    if (!documentId) return;
    setDetectingObjects(true);
    setError(null);
    try {
      const options = {
        confidence_threshold: objectConfidence,
      };
      const response = await apiService.detectObjects(documentId, options);
      if (response.success && response.objects) {
        setDetectedObjects(response.objects);
      } else {
        setError(response?.error || 'Object detection failed');
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to detect objects');
    } finally {
      setDetectingObjects(false);
    }
  };

  const handleAnnotationSaved = (annotation) => {
    setUserAnnotations((prev) => [annotation, ...prev]);
    setShowObjectCanvas(false);
  };

  const handleDeleteAnnotation = async (annotationId) => {
    try {
      await apiService.deleteObjectAnnotation(annotationId);
      setUserAnnotations((prev) => prev.filter((a) => a.id !== annotationId));
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to delete annotation');
    }
  };

  const analyzeFaces = async () => {
    if (!documentId) return;
    setAnalyzingFaces(true);
    setError(null);
    try {
      const response = await apiService.post(`/api/documents/${documentId}/analyze-faces`);
      if (response.success && response.faces) {
        setDetectedFaces(response.faces || []);
      } else {
        setError(response?.error || 'Face detection failed');
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to analyze faces');
    } finally {
      setAnalyzingFaces(false);
    }
  };

  const getFaceLabel = (face, index) => {
    if (face.identity_name) return face.identity_name;
    if (face.suggested_identity) return `${face.suggested_identity}? (${face.suggested_confidence || 0}%)`;
    return `Person ${index + 1}`;
  };

  const handleFaceClick = (face) => {
    setSelectedFace(face);
    setIdentityName(face.identity_name || face.suggested_identity || '');
    setSuggestions([]);
  };

  const tagFace = async () => {
    if (!selectedFace || !identityName.trim() || !documentId) return;
    setTagging(true);
    setError(null);
    try {
      const response = await apiService.put(
        `/api/documents/${documentId}/faces/${selectedFace.id}/tag`,
        { identity_name: identityName.trim(), face_encoding: [] }
      );
      if (response.success) {
        await loadDetectedFaces();
        if (response.suggestions?.length > 0) setSuggestions(response.suggestions);
        if (!response.tags_auto_synced) {
          setShowTagSuggestion({ identityName: identityName.trim(), documentId });
        }
        setSelectedFace(null);
        setIdentityName('');
      } else {
        setError(response?.error || 'Failed to tag face');
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to tag face');
    } finally {
      setTagging(false);
    }
  };

  const confirmSuggestedIdentity = async (face) => {
    if (!face.suggested_identity) return;
    setSelectedFace(face);
    setIdentityName(face.suggested_identity);
    setTagging(true);
    setError(null);
    try {
      const response = await apiService.put(
        `/api/documents/${documentId}/faces/${face.id}/tag`,
        { identity_name: face.suggested_identity, face_encoding: [] }
      );
      if (response.success) {
        await loadDetectedFaces();
        if (!response.tags_auto_synced) setShowTagSuggestion({ identityName: face.suggested_identity, documentId });
        setSelectedFace(null);
        setIdentityName('');
      } else setError('Failed to confirm identity');
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to confirm identity');
    } finally {
      setTagging(false);
    }
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
      if (response.success) setShowTagSuggestion(null);
      else setError(response?.error || 'Failed to add identity to tags');
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to add identity to tags');
    } finally {
      setAddingToTags(false);
    }
  };

  const handleAnalyzeFaces = () => {
    analyzeFaces();
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      let objectsToSave = detectedObjects?.length > 0 ? detectedObjects : (Array.isArray(formData?.objects) ? formData.objects : []);
      try {
        const objRes = await apiService.getDetectedObjects(documentId);
        if (objRes?.success && Array.isArray(objRes.objects) && objRes.objects.length > 0) objectsToSave = objRes.objects;
      } catch (_) {}
      // Single canonical label: class_name = user_tag when user_tag is set (so metadata.json is consistent)
      objectsToSave = (Array.isArray(objectsToSave) ? objectsToSave : []).map((obj) => {
        const o = { ...obj };
        const tag = (o.user_tag || '').trim();
        if (tag) o.class_name = tag;
        return o;
      });
      const metadataPayload = {
        type: formData.type,
        title: (formData.title || '').trim() || null,
        content: (formData.content || '').trim() || null,
        author: formData.author.trim() || null,
        date: formData.date.trim() || null,
        series: formData.series.trim() || null,
        tags: formData.tags,
        llm_metadata: formData.llm_metadata || null,
        faces: detectedFaces.length > 0 ? detectedFaces : (formData.faces || null),
        objects: objectsToSave,
        // Type-specific fields
        location: formData.location?.trim() || null,
        event: formData.event?.trim() || null,
        medium: formData.medium?.trim() || null,
        dimensions: formData.dimensions?.trim() || null,
        body_part: formData.body_part?.trim() || null,
        modality: formData.modality?.trim() || null,
        map_type: formData.map_type?.trim() || null,
        coordinates: formData.coordinates?.trim() || null,
        application: formData.application?.trim() || null,
        platform: formData.platform?.trim() || null
      };

      // Remove null/empty fields; never remove objects or faces (keep arrays for metadata.json)
      Object.keys(metadataPayload).forEach(key => {
        if (key === 'objects' || key === 'faces') return;
        if (metadataPayload[key] === null || metadataPayload[key] === '') {
          delete metadataPayload[key];
        }
      });

      await apiService.post(`/api/documents/${documentId}/image-metadata`, metadataPayload);
      
      setSuccess('Image metadata saved successfully!');
      setMetadataExists(true);
      // Close modal after a brief delay to show success message
      setTimeout(() => {
        onClose(true); // Pass true to indicate success
      }, 1000);
    } catch (err) {
      console.error('Failed to save image metadata:', err);
      setError(err.message || 'Failed to save metadata. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!window.confirm('Are you sure you want to delete the metadata for this image?')) {
      return;
    }

    setSaving(true);
    setError(null);

    try {
      await apiService.delete(`/api/documents/${documentId}/image-metadata`);
      setSuccess('Metadata deleted successfully!');
      setMetadataExists(false);
      setTimeout(() => {
        onClose(true);
      }, 1000);
    } catch (err) {
      console.error('Failed to delete image metadata:', err);
      setError(err.message || 'Failed to delete metadata. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  const effectiveImageUrl = imageUrl || fetchedImageUrl;
  const facesForOverlay = detectedFaces.length > 0 ? detectedFaces : (Array.isArray(formData.faces) ? formData.faces : []);
  const hasFaces = facesForOverlay.length > 0;
  // Merge detected objects with user annotations so custom objects show in bounding box overlay (including newly added before refetch)
  const objectsForOverlay = [
    ...detectedObjects,
    ...userAnnotations
      .filter((a) => !detectedObjects.some((o) => o.id === 'ann-' + a.id))
      .map((a) => ({
        id: 'ann-' + a.id,
        bbox_x: a.bbox_x,
        bbox_y: a.bbox_y,
        bbox_width: a.bbox_width,
        bbox_height: a.bbox_height,
        user_tag: a.object_name,
        class_name: a.object_name,
        detection_method: 'user_defined'
      }))
  ];
  const hasObjectsForOverlay = objectsForOverlay.length > 0;

  return (
    <Dialog
      open={open}
      onClose={() => onClose(false)}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: {
          minHeight: '560px',
          maxHeight: '90vh'
        }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ImageSearch color="primary" />
          <Typography variant="h6">Edit Image Metadata & Tag Faces</Typography>
        </Box>
        {filename && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
            {filename}
          </Typography>
        )}
      </DialogTitle>

      <DialogContent sx={{ p: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 300 }}>
            <CircularProgress />
          </Box>
        ) : (
          <Box sx={{ display: 'flex', flex: 1, minHeight: 0 }}>
            {/* Left: tabs + form / object detection */}
            <Box sx={{ flex: 1, minWidth: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column', borderRight: 1, borderColor: 'divider' }}>
              <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ borderBottom: 1, borderColor: 'divider', px: 1 }}>
                <Tab label="Metadata & Faces" icon={<Face />} iconPosition="start" />
                <Tab label="Object Detection" icon={<CenterFocusStrong />} iconPosition="start" />
              </Tabs>
              <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
                {error && (
                  <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 1 }}>
                    {error}
                  </Alert>
                )}
                {success && (
                  <Alert severity="success" sx={{ mb: 1 }}>
                    {success}
                  </Alert>
                )}

                {activeTab === 0 && (
              <Stack spacing={3} sx={{ mt: 0 }}>
                {/* Add-to-tags suggestion */}
                {showTagSuggestion && (
                  <Alert
                    severity="info"
                    action={
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Button size="small" color="primary" onClick={handleAddToTags} disabled={addingToTags}>
                          {addingToTags ? 'Adding...' : 'Add to Tags'}
                        </Button>
                        <Button size="small" onClick={() => setShowTagSuggestion(null)} disabled={addingToTags}>
                          Skip
                        </Button>
                      </Box>
                    }
                  >
                    Added &apos;{showTagSuggestion.identityName}&apos; to face detection. Add to image tags?
                  </Alert>
                )}

                {/* Action bar */}
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={analyzingFaces ? <CircularProgress size={16} /> : <Face />}
                    onClick={handleAnalyzeFaces}
                    disabled={loading || analyzingFaces || !documentId}
                  >
                    {analyzingFaces ? 'Analyzing...' : 'Analyze Faces'}
                  </Button>
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={detectingObjects ? <CircularProgress size={16} /> : <CenterFocusStrong />}
                    onClick={handleDetectObjects}
                    disabled={loading || detectingObjects || !documentId}
                  >
                    {detectingObjects ? 'Detecting...' : 'Detect Objects'}
                  </Button>
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={describeLoading ? <CircularProgress size={16} /> : <AutoAwesome />}
                    onClick={handleDescribeLLM}
                    disabled={loading || describeLoading || !documentId || !hasCapability('feature.image.llm_description')}
                  >
                    {describeLoading ? 'Describing...' : 'Describe via LLM'}
                  </Button>
                </Box>

                {/* Face tagging: selected face + name + Tag button */}
                {selectedFace && (
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Tag face: {getFaceLabel(selectedFace, detectedFaces.indexOf(selectedFace))}
                    </Typography>
                    <TextField
                      fullWidth
                      size="small"
                      label="Person name"
                      value={identityName}
                      onChange={(e) => setIdentityName(e.target.value)}
                      placeholder="e.g., Steve McQueen"
                      sx={{ mb: 1 }}
                      onKeyPress={(e) => {
                        if (e.key === 'Enter' && identityName.trim()) tagFace();
                      }}
                    />
                    <Button
                      variant="contained"
                      size="small"
                      startIcon={<PersonAdd />}
                      onClick={tagFace}
                      disabled={!identityName.trim() || tagging}
                      fullWidth
                    >
                      {tagging ? 'Tagging...' : 'Tag face'}
                    </Button>
                  </Paper>
                )}

                {/* Face list */}
                {detectedFaces.length > 0 && !selectedFace && (
                  <Paper variant="outlined" sx={{ p: 1 }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Detected faces ({detectedFaces.length})
                    </Typography>
                    <List dense sx={{ maxHeight: 180, overflow: 'auto' }}>
                      {detectedFaces.map((face, idx) => (
                        <ListItem
                          key={face.id ?? idx}
                          button
                          onClick={() => handleFaceClick(face)}
                          selected={selectedFace?.id === face.id}
                        >
                          <ListItemText
                            primary={getFaceLabel(face, idx)}
                            secondary={
                              face.identity_confirmed ? 'Tagged' :
                              face.suggested_identity ? `Suggested: ${face.suggested_identity}` : 'Not tagged'
                            }
                          />
                          <ListItemSecondaryAction>
                            {face.identity_confirmed ? (
                              <CheckCircle color="success" fontSize="small" />
                            ) : face.suggested_identity ? (
                              <Button
                                size="small"
                                variant="contained"
                                onClick={(e) => { e.stopPropagation(); confirmSuggestedIdentity(face); }}
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

                {/* Image Type */}
            <FormControl fullWidth>
              <InputLabel>Image Type</InputLabel>
              <Select
                name="type"
                value={formData.type}
                onChange={handleInputChange}
                label="Image Type"
              >
                {IMAGE_TYPES.map(type => (
                  <MenuItem key={type.value} value={type.value}>
                    {type.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Title */}
            <TextField
              name="title"
              label="Title"
              value={formData.title}
              onChange={handleInputChange}
              fullWidth
              helperText="Optional: A descriptive title for the image"
            />

            {/* Content/Description */}
            <TextField
              name="content"
              label="Content / Description"
              value={formData.content}
              onChange={handleInputChange}
              fullWidth
              multiline
              rows={4}
              helperText="Optional: Describe what's in the image, transcript, or key information"
            />

            {/* Author */}
            <TextField
              name="author"
              label="Author / Creator"
              value={formData.author}
              onChange={handleInputChange}
              fullWidth
              helperText="Optional: Name of the creator or author"
            />

            {/* Date */}
            <TextField
              name="date"
              label="Date"
              type="date"
              value={formData.date}
              onChange={handleInputChange}
              fullWidth
              InputLabelProps={{
                shrink: true,
              }}
              helperText="Optional: Date associated with the image (YYYY-MM-DD)"
            />

            {/* Series */}
            <TextField
              name="series"
              label="Series / Collection"
              value={formData.series}
              onChange={handleInputChange}
              fullWidth
              helperText="Optional: Series name or collection (e.g., 'Calvin and Hobbes', 'Documentation v2.0')"
            />

            {/* Tags */}
            <Box>
              <TextField
                label="Tags"
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyPress={handleTagInputKeyPress}
                fullWidth
                helperText="Press Enter to add a tag"
                InputProps={{
                  endAdornment: tagInput && (
                    <Button
                      size="small"
                      onClick={() => addTag(tagInput.trim())}
                      sx={{ mr: -1 }}
                    >
                      Add
                    </Button>
                  )
                }}
              />
              {formData.tags.length > 0 && (
                <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {formData.tags.map((tag, index) => (
                    <Chip
                      key={index}
                      label={tag}
                      onDelete={() => removeTag(tag)}
                      size="small"
                    />
                  ))}
                </Box>
              )}
            </Box>

            {/* Type-specific fields */}
            {(formData.type === 'photo' || formData.type === 'maps') && (
              <>
                <Typography variant="subtitle2" color="text.secondary" sx={{ mt: 1 }}>
                  {formData.type === 'photo' ? 'Photo-specific Fields' : 'Map-specific Fields'}
                </Typography>
                
                <TextField
                  name="location"
                  label="Location"
                  value={formData.location}
                  onChange={handleInputChange}
                  fullWidth
                  helperText="Geographic location or venue"
                />
                
                {formData.type === 'photo' && (
                  <TextField
                    name="event"
                    label="Event"
                    value={formData.event}
                    onChange={handleInputChange}
                    fullWidth
                    helperText="Event name (e.g., 'Birthday Party', 'Conference 2024')"
                  />
                )}
                
                {formData.type === 'maps' && (
                  <>
                    <TextField
                      name="map_type"
                      label="Map Type"
                      value={formData.map_type}
                      onChange={handleInputChange}
                      fullWidth
                      helperText="Type of map (e.g., 'topographic', 'political', 'street')"
                    />
                    <TextField
                      name="coordinates"
                      label="Coordinates"
                      value={formData.coordinates}
                      onChange={handleInputChange}
                      fullWidth
                      helperText="Geographic coordinates (e.g., '40.7128,-74.0060')"
                    />
                  </>
                )}
              </>
            )}

            {formData.type === 'artwork' && (
              <>
                <Typography variant="subtitle2" color="text.secondary" sx={{ mt: 1 }}>
                  Artwork-specific Fields
                </Typography>
                
                <TextField
                  name="medium"
                  label="Medium"
                  value={formData.medium}
                  onChange={handleInputChange}
                  fullWidth
                  helperText="Art medium (e.g., 'Oil on canvas', 'Watercolor', 'Digital')"
                />
                
                <TextField
                  name="dimensions"
                  label="Dimensions"
                  value={formData.dimensions}
                  onChange={handleInputChange}
                  fullWidth
                  helperText="Physical dimensions (e.g., '24x36 inches', '60x90 cm')"
                />
              </>
            )}

            {formData.type === 'medical' && (
              <>
                <Typography variant="subtitle2" color="text.secondary" sx={{ mt: 1 }}>
                  Medical Image Fields
                </Typography>
                
                <TextField
                  name="body_part"
                  label="Body Part"
                  value={formData.body_part}
                  onChange={handleInputChange}
                  fullWidth
                  helperText="Body part imaged (e.g., 'chest', 'skull', 'knee')"
                />
                
                <TextField
                  name="modality"
                  label="Imaging Modality"
                  value={formData.modality}
                  onChange={handleInputChange}
                  fullWidth
                  helperText="Imaging type (e.g., 'X-ray', 'MRI', 'CT', 'Ultrasound')"
                />
              </>
            )}

            {formData.type === 'screenshot' && (
              <>
                <Typography variant="subtitle2" color="text.secondary" sx={{ mt: 1 }}>
                  Screenshot-specific Fields
                </Typography>
                
                <TextField
                  name="application"
                  label="Application"
                  value={formData.application}
                  onChange={handleInputChange}
                  fullWidth
                  helperText="Application name (e.g., 'VS Code', 'Chrome', 'Photoshop')"
                />
                
                <TextField
                  name="platform"
                  label="Platform / OS"
                  value={formData.platform}
                  onChange={handleInputChange}
                  fullWidth
                  helperText="Operating system (e.g., 'Windows 11', 'macOS Sonoma', 'Linux')"
                />
              </>
            )}
              </Stack>
                )}

                {activeTab === 1 && (
                  <Stack spacing={2}>
                    <Button
                      variant="contained"
                      size="small"
                      startIcon={detectingObjects ? <CircularProgress size={16} /> : <CenterFocusStrong />}
                      onClick={handleDetectObjects}
                      disabled={loading || detectingObjects || !documentId}
                    >
                      {detectingObjects ? 'Detecting...' : 'Detect Objects'}
                    </Button>
                    <Typography variant="subtitle2" color="text.secondary">Confidence threshold</Typography>
                    <Slider
                      value={objectConfidence}
                      onChange={(_, v) => setObjectConfidence(v)}
                      min={0.3}
                      max={0.9}
                      step={0.1}
                      valueLabelDisplay="auto"
                      marks={[{ value: 0.5, label: '0.5' }]}
                    />
                    {detectedObjects.length > 0 && (
                      <Paper variant="outlined" sx={{ p: 1 }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                          Detected objects ({detectedObjects.length})
                        </Typography>
                        <List dense sx={{ maxHeight: 200, overflow: 'auto' }}>
                          {detectedObjects.map((obj, idx) => (
                            <ListItem key={obj.id ?? idx}>
                              <ListItemText
                                primary={obj.user_tag || obj.class_name || obj.matched_description || 'Object'}
                                secondary={obj.confidence != null ? `${(obj.confidence * 100).toFixed(0)}% · ${obj.detection_method || 'yolo'}` : (obj.detection_method || 'yolo')}
                              />
                              {obj.id && (
                                <ListItemSecondaryAction>
                                  <Button size="small" onClick={() => handleConfirmObject(obj)}>
                                    Confirm
                                  </Button>
                                  <IconButton size="small" onClick={() => { setEditingObjectTagId(obj.id); setEditingObjectTagValue(obj.user_tag || obj.class_name || ''); }} aria-label="Edit tag" title="Refined label">
                                    <EditIcon fontSize="small" />
                                  </IconButton>
                                  <IconButton size="small" onClick={() => handleRejectObject(obj)} aria-label="Reject" title="Hide this detection">
                                    <BlockIcon fontSize="small" color="error" />
                                  </IconButton>
                                </ListItemSecondaryAction>
                              )}
                            </ListItem>
                          ))}
                        </List>
                      </Paper>
                    )}
                    <Button
                      variant="outlined"
                      size="small"
                      startIcon={<AddBox />}
                      onClick={() => setShowObjectCanvas(true)}
                      disabled={!effectiveImageUrl || !documentId}
                    >
                      Annotate Custom Object
                    </Button>
                    {userAnnotations.length > 0 && (
                      <Paper variant="outlined" sx={{ p: 1 }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                          User annotations
                        </Typography>
                        <List dense sx={{ maxHeight: 140, overflow: 'auto' }}>
                          {userAnnotations.map((ann) => (
                            <ListItem key={ann.id}>
                              <ListItemText primary={ann.object_name} secondary={ann.description || null} />
                              <ListItemSecondaryAction>
                                <IconButton size="small" onClick={() => handleDeleteAnnotation(ann.id)} aria-label="Delete">
                                  <DeleteIcon fontSize="small" />
                                </IconButton>
                              </ListItemSecondaryAction>
                            </ListItem>
                          ))}
                        </List>
                      </Paper>
                    )}
                  </Stack>
                )}
              </Box>
            </Box>

            {/* Right: image with bounding box overlay (or loading when fetching with auth) */}
            {(effectiveImageUrl || (filename && !imageUrl)) && (
              <Box
                sx={{
                  width: { xs: '100%', sm: 380 },
                  minWidth: 320,
                  flexShrink: 0,
                  p: 2,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'flex-start',
                  bgcolor: 'grey.100',
                  maxHeight: '70vh'
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', mb: 1 }}>
                  <Typography variant="subtitle2" color="text.secondary">Preview</Typography>
                  {(hasFaces || hasObjectsForOverlay) && effectiveImageUrl && (
                    <FormControlLabel
                      control={
                        <Switch
                          size="small"
                          checked={showBoundingBoxes}
                          onChange={(e) => setShowBoundingBoxes(e.target.checked)}
                        />
                      }
                      label="Show bounding boxes"
                    />
                  )}
                </Box>
                {!effectiveImageUrl ? (
                  <Box sx={{ py: 4, px: 2, textAlign: 'center' }}>
                    <CircularProgress size={32} />
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>Loading image...</Typography>
                  </Box>
                ) : (
                <Box
                  sx={{
                    position: 'relative',
                    display: 'inline-block',
                    maxWidth: '100%',
                    maxHeight: '65vh',
                    borderRadius: 1,
                    overflow: 'hidden',
                    bgcolor: 'grey.200'
                  }}
                >
                  <img
                    ref={imageRef}
                    src={effectiveImageUrl}
                    alt=""
                    style={{ display: 'block', maxWidth: 340, maxHeight: '65vh', objectFit: 'contain', verticalAlign: 'top' }}
                    onLoad={() => {
                      const el = imageRef.current;
                      if (el) {
                        const rect = el.getBoundingClientRect();
                        setImageDisplaySize({ w: rect.width, h: rect.height });
                      }
                    }}
                  />
                  {showBoundingBoxes && hasFaces && imageRef.current && imageDisplaySize.w > 0 && (
                    <svg
                      style={{
                        position: 'absolute',
                        left: 0,
                        top: 0,
                        width: imageDisplaySize.w,
                        height: imageDisplaySize.h,
                        pointerEvents: 'none'
                      }}
                      width={imageDisplaySize.w}
                      height={imageDisplaySize.h}
                      viewBox={`0 0 ${imageDisplaySize.w} ${imageDisplaySize.h}`}
                    >
                      {facesForOverlay.map((face, idx) => {
                        const nw = imageRef.current?.naturalWidth || 1;
                        const nh = imageRef.current?.naturalHeight || 1;
                        const sx = imageDisplaySize.w / nw;
                        const sy = imageDisplaySize.h / nh;
                        const x = (face.bbox_x ?? face.left ?? 0) * sx;
                        const y = (face.bbox_y ?? face.top ?? 0) * sy;
                        const w = (face.bbox_width ?? (face.right - face.left) ?? 0) * sx;
                        const h = (face.bbox_height ?? (face.bottom - face.top) ?? 0) * sy;
                        return (
                          <g key={face.id ?? idx}>
                            <rect
                              x={x}
                              y={y}
                              width={w}
                              height={h}
                              fill="none"
                              stroke="lime"
                              strokeWidth={2}
                            />
                            {(face.identity_name || face.suggested_identity) && (
                              <text
                                x={x}
                                y={Math.max(y - 4, 12)}
                                fill="white"
                                fontSize={12}
                                fontWeight="bold"
                                style={{ paintOrder: 'stroke', stroke: 'black', strokeWidth: 2 }}
                              >
                                {face.identity_name || `${face.suggested_identity}?`}
                              </text>
                            )}
                          </g>
                        );
                      })}
                    </svg>
                  )}
                  {showBoundingBoxes && hasObjectsForOverlay && imageRef.current && imageDisplaySize.w > 0 && (
                    <svg
                      style={{
                        position: 'absolute',
                        left: 0,
                        top: 0,
                        width: imageDisplaySize.w,
                        height: imageDisplaySize.h,
                        pointerEvents: 'none'
                      }}
                      width={imageDisplaySize.w}
                      height={imageDisplaySize.h}
                      viewBox={`0 0 ${imageDisplaySize.w} ${imageDisplaySize.h}`}
                    >
                      {(() => {
                        const nw = imageRef.current?.naturalWidth || 1;
                        const nh = imageRef.current?.naturalHeight || 1;
                        const sx = imageDisplaySize.w / nw;
                        const sy = imageDisplaySize.h / nh;
                        return objectsForOverlay.map((obj, idx) => {
                          const x = (obj.bbox_x ?? obj.x ?? 0) * sx;
                          const y = (obj.bbox_y ?? obj.y ?? 0) * sy;
                          const w = (obj.bbox_width ?? obj.width ?? 0) * sx;
                          const h = (obj.bbox_height ?? obj.height ?? 0) * sy;
                          const label = obj.user_tag || obj.class_name || obj.matched_description || 'Object';
                          const isUserDefined = obj.detection_method === 'user_defined';
                          return (
                            <g key={obj.id ?? idx}>
                              <rect
                                x={x}
                                y={y}
                                width={w}
                                height={h}
                                fill="none"
                                stroke={isUserDefined ? '#9333ea' : 'deepskyblue'}
                                strokeWidth={2}
                              />
                              <text
                                x={x}
                                y={Math.max(y - 4, 12)}
                                fill="white"
                                fontSize={12}
                                fontWeight="bold"
                                style={{ paintOrder: 'stroke', stroke: 'black', strokeWidth: 2 }}
                              >
                                {label}
                              </text>
                            </g>
                          );
                        });
                      })()}
                    </svg>
                  )}
                </Box>
                )}
              </Box>
            )}
          </Box>
        )}
      </DialogContent>

      <DialogActions>
        <Button
          onClick={() => onClose(false)}
          disabled={saving}
          startIcon={<Close />}
        >
          Cancel
        </Button>
        {(formData.title || formData.content || (Array.isArray(formData.tags) && formData.tags.length > 0)) && (
          <Button
            onClick={handleDelete}
            disabled={saving || loading}
            color="error"
          >
            Delete Metadata
          </Button>
        )}
        <Button
          onClick={handleSave}
          disabled={saving || loading}
          variant="contained"
          startIcon={saving ? <CircularProgress size={16} /> : <Save />}
        >
          {saving ? 'Saving...' : 'Save'}
        </Button>
      </DialogActions>
      <Dialog
        open={editingObjectTagId != null}
        onClose={() => { setEditingObjectTagId(null); setEditingObjectTagValue(''); }}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>Refined label</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            fullWidth
            label="Tag (e.g. BMW i3)"
            value={editingObjectTagValue}
            onChange={(e) => setEditingObjectTagValue(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleSaveObjectTag(); }}
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setEditingObjectTagId(null); setEditingObjectTagValue(''); }}>Cancel</Button>
          <Button variant="contained" onClick={handleSaveObjectTag}>Save</Button>
        </DialogActions>
      </Dialog>
      <Dialog open={showObjectCanvas} onClose={() => setShowObjectCanvas(false)} maxWidth="md" fullWidth>
        <DialogTitle>Annotate custom object</DialogTitle>
        <DialogContent>
          {effectiveImageUrl && documentId && (
            <ObjectAnnotationCanvas
              documentId={documentId}
              imageUrl={effectiveImageUrl}
              imageWidth={imageRef.current?.naturalWidth || 0}
              imageHeight={imageRef.current?.naturalHeight || 0}
              detectedObjects={detectedObjects}
              userAnnotations={userAnnotations}
              onAnnotationSaved={handleAnnotationSaved}
              onClose={() => setShowObjectCanvas(false)}
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowObjectCanvas(false)}>Close</Button>
        </DialogActions>
      </Dialog>
      <DiffMergeDialog
        open={diffDialogOpen}
        onClose={() => { setDiffDialogOpen(false); setPendingNewContent(''); setPendingLLMMetadata(null); }}
        currentContent={formData.content || ''}
        newContent={pendingNewContent}
        onKeepOld={handleDiffKeepOld}
        onReplaceWithNew={handleDiffReplaceWithNew}
      />
    </Dialog>
  );
};

export default ImageMetadataModal;
