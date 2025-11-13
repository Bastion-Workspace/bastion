import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Switch,
  FormControlLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Tooltip,
  Alert
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Archive as ArchiveIcon,
  InsertDriveFile as FileIcon,
  Folder as FolderIcon,
  Link as LinkIcon,
  LinkOff as LinkOffIcon
} from '@mui/icons-material';
import apiService from '../services/apiService';

const ZipHierarchyView = ({ documents, onDocumentUpdate, onDocumentDelete }) => {
  const [expandedZips, setExpandedZips] = useState(new Set());
  const [editingDoc, setEditingDoc] = useState(null);
  const [deleteDialog, setDeleteDialog] = useState({ open: false, doc: null });
  const [editForm, setEditForm] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const toggleZipExpansion = (zipId) => {
    const newExpanded = new Set(expandedZips);
    if (newExpanded.has(zipId)) {
      newExpanded.delete(zipId);
    } else {
      newExpanded.add(zipId);
    }
    setExpandedZips(newExpanded);
  };

  const handleEditMetadata = (doc) => {
    setEditingDoc(doc);
    setEditForm({
      title: doc.title || '',
      category: doc.category || '',
      tags: doc.tags ? doc.tags.join(', ') : '',
      description: doc.description || '',
      author: doc.author || '',
      publication_date: doc.publication_date || ''
    });
  };

  const handleSaveMetadata = async () => {
    if (!editingDoc) return;

    setLoading(true);
    setError(null);

    try {
      const metadata = {
        title: editForm.title.trim() || null,
        category: editForm.category || null,
        tags: editForm.tags ? editForm.tags.split(',').map(tag => tag.trim()).filter(tag => tag) : [],
        description: editForm.description.trim() || null,
        author: editForm.author.trim() || null,
        publication_date: editForm.publication_date || null
      };

      // Filter out null/empty values
      const cleanMetadata = Object.fromEntries(
        Object.entries(metadata).filter(([_, v]) => v !== null && v !== '')
      );

      if (editingDoc.is_zip_container) {
        // Update ZIP and propagate to children
        await apiService.updateZipMetadata(editingDoc.document_id, cleanMetadata);
      } else {
        // Update individual document
        await apiService.updateDocumentMetadata(editingDoc.document_id, cleanMetadata);
      }

      setEditingDoc(null);
      onDocumentUpdate();
    } catch (error) {
      setError(`Failed to update metadata: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleInheritance = async (childDoc) => {
    try {
      const newInheritance = !childDoc.inherit_parent_metadata;
      await apiService.toggleMetadataInheritance(childDoc.document_id, newInheritance);
      onDocumentUpdate();
    } catch (error) {
      setError(`Failed to toggle inheritance: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handleDeleteDocument = async (deleteChildren = true) => {
    const { doc } = deleteDialog;
    if (!doc) return;

    setLoading(true);
    setError(null);

    try {
      if (doc.is_zip_container) {
        await apiService.deleteZipHierarchy(doc.document_id, deleteChildren);
      } else {
        await apiService.deleteDocument(doc.document_id);
      }

      setDeleteDialog({ open: false, doc: null });
      onDocumentDelete(doc.document_id);
    } catch (error) {
      setError(`Failed to delete: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'success';
      case 'processing': return 'warning';
      case 'embedding': return 'info';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const renderDocument = (doc, isChild = false, parentDoc = null) => (
    <ListItem
      key={doc.document_id}
      sx={{
        pl: isChild ? 4 : 1,
        borderLeft: isChild ? '2px solid #e0e0e0' : 'none',
        mb: 0.5
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mr: 1 }}>
        {doc.is_zip_container ? <ArchiveIcon color="primary" /> : <FileIcon />}
      </Box>
      
      <ListItemText
        primary={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
            <Typography variant="body2" sx={{ fontWeight: doc.is_zip_container ? 'bold' : 'normal' }}>
              {doc.filename}
            </Typography>
            
            {doc.title && (
              <Typography variant="caption" color="text.secondary">
                ({doc.title})
              </Typography>
            )}
            
            <Chip 
              label={doc.processing_status} 
              size="small" 
              color={getStatusColor(doc.processing_status)}
              variant="outlined"
            />
            
            {doc.category && (
              <Chip label={doc.category} size="small" color="secondary" variant="outlined" />
            )}

            {isChild && (
              <Tooltip title={doc.inherit_parent_metadata ? "Inherits parent metadata" : "Independent metadata"}>
                <IconButton
                  size="small"
                  onClick={() => handleToggleInheritance(doc)}
                  color={doc.inherit_parent_metadata ? "primary" : "default"}
                >
                  {doc.inherit_parent_metadata ? <LinkIcon /> : <LinkOffIcon />}
                </IconButton>
              </Tooltip>
            )}
          </Box>
        }
        secondary={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mt: 0.5 }}>
            <Typography variant="caption">
              {formatFileSize(doc.file_size)}
            </Typography>
            <Typography variant="caption">
              {new Date(doc.upload_date).toLocaleDateString()}
            </Typography>
            {doc.original_zip_path && (
              <Typography variant="caption" color="text.secondary">
                Path: {doc.original_zip_path}
              </Typography>
            )}
            {doc.tags && doc.tags.length > 0 && (
              <Box sx={{ display: 'flex', gap: 0.5 }}>
                {doc.tags.slice(0, 3).map(tag => (
                  <Chip key={tag} label={tag} size="small" variant="outlined" />
                ))}
                {doc.tags.length > 3 && (
                  <Typography variant="caption">+{doc.tags.length - 3} more</Typography>
                )}
              </Box>
            )}
          </Box>
        }
      />
      
      <ListItemSecondaryAction>
        <IconButton size="small" onClick={() => handleEditMetadata(doc)}>
          <EditIcon />
        </IconButton>
        <IconButton 
          size="small" 
          onClick={() => setDeleteDialog({ open: true, doc })}
          color="error"
        >
          <DeleteIcon />
        </IconButton>
      </ListItemSecondaryAction>
    </ListItem>
  );

  return (
    <Box>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <List>
        {documents.map(group => {
          const parentDoc = group.parent;
          const children = group.children || [];
          const isZipContainer = parentDoc.is_zip_container;

          if (isZipContainer && children.length > 0) {
            // Render as expandable ZIP container
            return (
              <Box key={parentDoc.document_id}>
                <Accordion
                  expanded={expandedZips.has(parentDoc.document_id)}
                  onChange={() => toggleZipExpansion(parentDoc.document_id)}
                  sx={{ mb: 1 }}
                >
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                      <FolderIcon color="primary" />
                      <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                        {parentDoc.filename}
                      </Typography>
                      {parentDoc.title && (
                        <Typography variant="caption" color="text.secondary">
                          ({parentDoc.title})
                        </Typography>
                      )}
                      <Chip 
                        label={`${children.length} files`} 
                        size="small" 
                        color="primary" 
                        variant="outlined"
                      />
                      <Chip 
                        label={parentDoc.processing_status} 
                        size="small" 
                        color={getStatusColor(parentDoc.processing_status)}
                        variant="outlined"
                      />
                    </Box>
                  </AccordionSummary>
                  
                  <AccordionDetails sx={{ p: 0 }}>
                    <List sx={{ pl: 2 }}>
                      {children.map(child => renderDocument(child, true, parentDoc))}
                    </List>
                  </AccordionDetails>
                </Accordion>
              </Box>
            );
          } else {
            // Render as regular document
            return renderDocument(parentDoc);
          }
        })}
      </List>

      {/* Edit Metadata Dialog */}
      <Dialog open={!!editingDoc} onClose={() => setEditingDoc(null)} maxWidth="md" fullWidth>
        <DialogTitle>
          Edit Metadata - {editingDoc?.filename}
          {editingDoc?.is_zip_container && (
            <Typography variant="caption" display="block" color="text.secondary">
              Changes will be inherited by child files that have inheritance enabled
            </Typography>
          )}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              label="Title"
              value={editForm.title || ''}
              onChange={(e) => setEditForm({ ...editForm, title: e.target.value })}
              fullWidth
            />
            
            <FormControl fullWidth>
              <InputLabel>Category</InputLabel>
              <Select
                value={editForm.category || ''}
                onChange={(e) => setEditForm({ ...editForm, category: e.target.value })}
                label="Category"
              >
                <MenuItem value="">None</MenuItem>
                <MenuItem value="technical">Technical</MenuItem>
                <MenuItem value="academic">Academic</MenuItem>
                <MenuItem value="business">Business</MenuItem>
                <MenuItem value="legal">Legal</MenuItem>
                <MenuItem value="medical">Medical</MenuItem>
                <MenuItem value="manual">Manual</MenuItem>
                <MenuItem value="reference">Reference</MenuItem>
                <MenuItem value="research">Research</MenuItem>
                <MenuItem value="personal">Personal</MenuItem>
                <MenuItem value="other">Other</MenuItem>
              </Select>
            </FormControl>
            
            <TextField
              label="Tags (comma-separated)"
              value={editForm.tags || ''}
              onChange={(e) => setEditForm({ ...editForm, tags: e.target.value })}
              fullWidth
              helperText="Enter tags separated by commas"
            />
            
            <TextField
              label="Description"
              value={editForm.description || ''}
              onChange={(e) => setEditForm({ ...editForm, description: e.target.value })}
              fullWidth
              multiline
              rows={3}
            />
            
            <TextField
              label="Author"
              value={editForm.author || ''}
              onChange={(e) => setEditForm({ ...editForm, author: e.target.value })}
              fullWidth
            />
            
            <TextField
              label="Publication Date"
              type="date"
              value={editForm.publication_date || ''}
              onChange={(e) => setEditForm({ ...editForm, publication_date: e.target.value })}
              fullWidth
              InputLabelProps={{ shrink: true }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditingDoc(null)}>Cancel</Button>
          <Button onClick={handleSaveMetadata} variant="contained" disabled={loading}>
            Save
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialog.open} onClose={() => setDeleteDialog({ open: false, doc: null })}>
        <DialogTitle>
          Delete {deleteDialog.doc?.is_zip_container ? 'ZIP Archive' : 'Document'}
        </DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete "{deleteDialog.doc?.filename}"?
          </Typography>
          {deleteDialog.doc?.is_zip_container && (
            <Box sx={{ mt: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    defaultChecked
                    onChange={(e) => setDeleteDialog({ 
                      ...deleteDialog, 
                      deleteChildren: e.target.checked 
                    })}
                  />
                }
                label="Also delete all extracted files from this ZIP"
              />
              <Typography variant="caption" display="block" color="text.secondary">
                If unchecked, the extracted files will be preserved as independent documents
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog({ open: false, doc: null })}>Cancel</Button>
          <Button 
            onClick={() => handleDeleteDocument(deleteDialog.deleteChildren !== false)}
            color="error" 
            variant="contained"
            disabled={loading}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ZipHierarchyView; 