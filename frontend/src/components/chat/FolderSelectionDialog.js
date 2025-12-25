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
import { Folder, FolderOpen } from '@mui/icons-material';
import apiService from '../../services/apiService';

/**
 * Folder Selection Dialog
 * 
 * Allows user to select a folder from their folder tree
 */
const FolderSelectionDialog = ({ open, onClose, onSelect }) => {
  const [folders, setFolders] = useState([]);
  const [filteredFolders, setFilteredFolders] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFolder, setSelectedFolder] = useState(null);
  const [expandedFolders, setExpandedFolders] = useState(new Set());

  // Load folder tree when dialog opens
  useEffect(() => {
    if (open) {
      loadFolders();
    } else {
      // Reset on close
      setSearchQuery('');
      setSelectedFolder(null);
      setError(null);
      setExpandedFolders(new Set());
    }
  }, [open]);

  // Filter folders based on search query
  useEffect(() => {
    if (!searchQuery) {
      setFilteredFolders(folders);
    } else {
      const query = searchQuery.toLowerCase();
      const filtered = folders.filter(f => 
        f.name.toLowerCase().includes(query) ||
        (f.path && f.path.toLowerCase().includes(query))
      );
      setFilteredFolders(filtered);
    }
  }, [searchQuery, folders]);

  const loadFolders = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await apiService.getFolderTree('user');
      
      if (response && response.folders) {
        // Flatten folder tree for display
        const flattenFolders = (folderList, parentPath = '', level = 0) => {
          const result = [];
          folderList.forEach(folder => {
            const path = parentPath ? `${parentPath} / ${folder.name}` : folder.name;
            result.push({
              ...folder,
              path,
              level
            });
            if (folder.children && folder.children.length > 0) {
              result.push(...flattenFolders(folder.children, path, level + 1));
            }
          });
          return result;
        };
        
        const flattened = flattenFolders(response.folders);
        setFolders(flattened);
        setFilteredFolders(flattened);
      } else {
        setError('Failed to load folders');
      }
    } catch (err) {
      console.error('Failed to load folders:', err);
      setError(err.message || 'Failed to load folders');
    } finally {
      setLoading(false);
    }
  };

  const handleSelect = () => {
    if (selectedFolder && onSelect) {
      onSelect(selectedFolder);
    }
    onClose();
  };

  const toggleFolder = (folderId) => {
    const newExpanded = new Set(expandedFolders);
    if (newExpanded.has(folderId)) {
      newExpanded.delete(folderId);
    } else {
      newExpanded.add(folderId);
    }
    setExpandedFolders(newExpanded);
  };

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        Select Folder
      </DialogTitle>
      
      <DialogContent>
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Choose a folder to import the image into
          </Typography>
          
          <TextField
            fullWidth
            size="small"
            placeholder="Search folders..."
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
            {filteredFolders.length === 0 ? (
              <Box sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  {searchQuery ? 'No matching folders found' : 'No folders available'}
                </Typography>
              </Box>
            ) : (
              filteredFolders.map((folder) => (
                <ListItem key={folder.folder_id} disablePadding>
                  <ListItemButton
                    selected={selectedFolder?.folder_id === folder.folder_id}
                    onClick={() => setSelectedFolder(folder)}
                    sx={{
                      pl: 2 + (folder.level * 2),
                      '&.Mui-selected': {
                        backgroundColor: 'primary.light',
                        '&:hover': {
                          backgroundColor: 'primary.light',
                        }
                      }
                    }}
                  >
                    <Box sx={{ mr: 1, display: 'flex', alignItems: 'center' }}>
                      {selectedFolder?.folder_id === folder.folder_id ? (
                        <FolderOpen fontSize="small" color="primary" />
                      ) : (
                        <Folder fontSize="small" />
                      )}
                    </Box>
                    <ListItemText
                      primary={
                        <Typography 
                          variant="body2" 
                          component="span"
                        >
                          {folder.name}
                        </Typography>
                      }
                      secondary={
                        folder.path && folder.path !== folder.name && (
                          <Typography variant="caption" color="text.secondary">
                            {folder.path}
                          </Typography>
                        )
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
        <Button 
          onClick={onClose}
        >
          Cancel
        </Button>
        <Button
          onClick={handleSelect}
          disabled={!selectedFolder}
          variant="contained"
        >
          Select
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default FolderSelectionDialog;

