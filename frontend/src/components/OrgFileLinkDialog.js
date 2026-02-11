import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Box,
  Typography,
  CircularProgress,
  Alert
} from '@mui/material';
import { Description, Folder } from '@mui/icons-material';
import apiService from '../services/apiService';
import { useAuth } from '../contexts/AuthContext';

/**
 * File Link Dialog
 *
 * Allows user to select a document from their document tree to insert as a file link.
 * Supports Org format [[file:path][description]] or Markdown format [description](path).
 */
const OrgFileLinkDialog = ({ open, onClose, onSelect, currentDocumentPath, linkFormat = 'org' }) => {
  const { user } = useAuth();
  const [documents, setDocuments] = useState([]);
  const [filteredDocuments, setFilteredDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDocument, setSelectedDocument] = useState(null);

  // Load documents when dialog opens
  useEffect(() => {
    if (open) {
      loadDocuments();
    } else {
      // Reset on close
      setSearchQuery('');
      setSelectedDocument(null);
      setError(null);
    }
  }, [open]);

  // Filter documents based on search query
  useEffect(() => {
    if (!searchQuery) {
      setFilteredDocuments(documents);
    } else {
      const query = searchQuery.toLowerCase();
      const filtered = documents.filter(doc => 
        doc.filename?.toLowerCase().includes(query) ||
        doc.title?.toLowerCase().includes(query) ||
        (doc.path && doc.path.toLowerCase().includes(query))
      );
      setFilteredDocuments(filtered);
    }
  }, [searchQuery, documents]);

  const loadDocuments = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await apiService.getUserDocuments(0, 1000);
      
          if (response && response.documents) {
        // Build document list with paths
        const docsWithPaths = response.documents.map(doc => {
          // Build relative path if we have current document path
          let relativePath = doc.filename || doc.title || '';
          
          if (currentDocumentPath && doc.canonical_path) {
            try {
              // Calculate relative path from current document to selected document
              const normalizePath = (path) => {
                // Normalize path separators and remove leading/trailing slashes
                return path.replace(/\\/g, '/').replace(/^\/+|\/+$/g, '');
              };
              
              const currentPath = normalizePath(currentDocumentPath);
              const targetPath = normalizePath(doc.canonical_path);
              
              // Get directory of current document
              const currentDir = currentPath.substring(0, currentPath.lastIndexOf('/') || 0);
              const targetDir = targetPath.substring(0, targetPath.lastIndexOf('/') || 0);
              const targetFile = targetPath.substring(targetPath.lastIndexOf('/') + 1);
              
              if (currentDir === targetDir) {
                // Same directory - use just filename
                relativePath = `./${targetFile}`;
              } else if (targetDir.startsWith(currentDir + '/')) {
                // Target is in subdirectory
                const subPath = targetDir.substring(currentDir.length + 1);
                relativePath = `./${subPath}/${targetFile}`;
              } else {
                // Different directory - calculate relative path
                const currentParts = currentDir ? currentDir.split('/') : [];
                const targetParts = targetDir ? targetDir.split('/') : [];
                
                // Find common prefix
                let commonLength = 0;
                while (commonLength < currentParts.length && 
                       commonLength < targetParts.length &&
                       currentParts[commonLength] === targetParts[commonLength]) {
                  commonLength++;
                }
                
                // Calculate relative path
                const upLevels = currentParts.length - commonLength;
                const downPath = targetParts.slice(commonLength).join('/');
                
                if (upLevels === 0 && downPath === '') {
                  // Same directory
                  relativePath = `./${targetFile}`;
                } else {
                  const upPath = '../'.repeat(upLevels);
                  const fullPath = downPath ? `${upPath}${downPath}/${targetFile}` : `${upPath}${targetFile}`;
                  relativePath = fullPath.startsWith('./') ? fullPath : `./${fullPath}`;
                }
              }
            } catch (err) {
              console.warn('Failed to calculate relative path:', err);
              relativePath = `./${doc.filename || doc.title || ''}`;
            }
          } else {
            // No current path - use filename
            relativePath = `./${doc.filename || doc.title || ''}`;
          }
          
          return {
            ...doc,
            relativePath,
            displayPath: doc.folder_path ? `${doc.folder_path}/${doc.filename || doc.title}` : (doc.filename || doc.title)
          };
        });
        
        setDocuments(docsWithPaths);
        setFilteredDocuments(docsWithPaths);
      } else {
        setError('Failed to load documents');
      }
    } catch (err) {
      console.error('Failed to load documents:', err);
      setError(err.message || 'Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  const handleSelect = () => {
    if (selectedDocument && onSelect) {
      const linkPath = selectedDocument.relativePath || `./${selectedDocument.filename || selectedDocument.title || ''}`;
      const description = selectedDocument.title || selectedDocument.filename || 'Link';
      const linkText = linkFormat === 'markdown'
        ? `[${description}](${linkPath})`
        : `[[file:${linkPath}][${description}]]`;
      onSelect(linkText);
    }
    onClose();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && selectedDocument && !e.shiftKey) {
      e.preventDefault();
      handleSelect();
    } else if (e.key === 'Escape') {
      onClose();
    }
  };

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="md"
      fullWidth
      onKeyDown={handleKeyDown}
    >
      <DialogTitle>
        Insert File Link
      </DialogTitle>
      
      <DialogContent>
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            {linkFormat === 'markdown'
              ? 'Select a document to insert as a Markdown link [description](path)'
              : 'Select a document to insert as an Org Mode file link'}
          </Typography>
          
          <TextField
            fullWidth
            size="small"
            placeholder="Search documents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            autoFocus
            sx={{ mt: 1 }}
          />
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress />
          </Box>
        ) : (
          <List 
            sx={{ 
              maxHeight: 400, 
              overflow: 'auto',
              border: '1px solid',
              borderColor: 'divider',
              borderRadius: 1
            }}
          >
            {filteredDocuments.length === 0 ? (
              <Box sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  {searchQuery ? 'No matching documents found' : 'No documents available'}
                </Typography>
              </Box>
            ) : (
              filteredDocuments.map((doc) => (
                <ListItem key={doc.document_id} disablePadding>
                  <ListItemButton
                    selected={selectedDocument?.document_id === doc.document_id}
                    onClick={() => setSelectedDocument(doc)}
                    sx={{
                      '&.Mui-selected': {
                        backgroundColor: 'primary.light',
                        '&:hover': {
                          backgroundColor: 'primary.light',
                        }
                      }
                    }}
                  >
                    <Box sx={{ mr: 1, display: 'flex', alignItems: 'center' }}>
                      <Description fontSize="small" />
                    </Box>
                    <ListItemText
                      primary={
                        <Typography variant="body2" component="span">
                          {doc.title || doc.filename || 'Untitled'}
                        </Typography>
                      }
                      secondary={
                        <Typography variant="caption" color="text.secondary">
                          {doc.displayPath}
                          {doc.relativePath && (
                            <Box component="span" sx={{ ml: 1, fontFamily: 'monospace', color: 'primary.main' }}>
                              → {doc.relativePath}
                            </Box>
                          )}
                        </Typography>
                      }
                    />
                  </ListItemButton>
                </ListItem>
              ))
            )}
          </List>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>
          Cancel
        </Button>
        <Button
          onClick={handleSelect}
          disabled={!selectedDocument}
          variant="contained"
        >
          Insert Link
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default OrgFileLinkDialog;
