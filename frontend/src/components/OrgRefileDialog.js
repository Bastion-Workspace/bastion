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
import { Article, Folder } from '@mui/icons-material';
import apiService from '../services/apiService';

/**
 * Org Refile Dialog - Roosevelt's Moving Day!
 * 
 * Allows user to select a target location for refiling an org entry
 */
const OrgRefileDialog = ({ open, onClose, sourceFile, sourceLine, sourceHeading }) => {
  const [targets, setTargets] = useState([]);
  const [filteredTargets, setFilteredTargets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [refiling, setRefiling] = useState(false);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTarget, setSelectedTarget] = useState(null);

  // Load targets when dialog opens
  useEffect(() => {
    if (open) {
      loadTargets();
    } else {
      // Reset on close
      setSearchQuery('');
      setSelectedTarget(null);
      setError(null);
    }
  }, [open]);

  // Filter targets based on search query
  useEffect(() => {
    if (!searchQuery) {
      setFilteredTargets(targets);
    } else {
      const query = searchQuery.toLowerCase();
      const filtered = targets.filter(t => 
        t.display_name.toLowerCase().includes(query) ||
        t.filename.toLowerCase().includes(query)
      );
      setFilteredTargets(filtered);
    }
  }, [searchQuery, targets]);

  const loadTargets = async () => {
    try {
      setLoading(true);
      setError(null);
      
      console.log('🎯 ROOSEVELT: Discovering refile targets...');
      const response = await apiService.get('/api/org/discover-targets');
      
      if (response.success) {
        setTargets(response.targets);
        setFilteredTargets(response.targets);
        console.log(`✅ ROOSEVELT: Found ${response.count} refile targets`);
      } else {
        setError('Failed to load refile targets');
      }
    } catch (err) {
      console.error('❌ Failed to load targets:', err);
      setError(err.message || 'Failed to load refile targets');
    } finally {
      setLoading(false);
    }
  };

  const handleRefile = async () => {
    if (!selectedTarget) return;
    
    try {
      setRefiling(true);
      setError(null);
      
      console.log('📦 ROOSEVELT: Refiling to:', selectedTarget);
      
      const response = await apiService.post('/api/org/refile', {
        source_file: sourceFile,
        source_line: sourceLine,
        target_file: selectedTarget.file,
        target_heading_line: selectedTarget.heading_line || null
      });
      
      if (response.success) {
        console.log('✅ ROOSEVELT: Refile successful!');
        onClose({ success: true, target: selectedTarget });
      } else {
        setError(response.error || 'Refile failed');
      }
    } catch (err) {
      console.error('❌ Refile failed:', err);
      setError(err.message || 'Refile operation failed');
    } finally {
      setRefiling(false);
    }
  };

  const getTargetIcon = (target) => {
    if (target.level === 0) {
      return <Folder fontSize="small" color="primary" />;
    }
    return <Article fontSize="small" />;
  };

  return (
    <Dialog 
      open={open} 
      onClose={() => !refiling && onClose()}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        📦 Refile: {sourceHeading}
      </DialogTitle>
      
      <DialogContent>
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Select a destination for this TODO
          </Typography>
          
          <TextField
            fullWidth
            size="small"
            placeholder="Search targets..."
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
            {filteredTargets.length === 0 ? (
              <Box sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  {searchQuery ? 'No matching targets found' : 'No refile targets available'}
                </Typography>
              </Box>
            ) : (
              filteredTargets.map((target, index) => (
                <ListItem key={index} disablePadding>
                  <ListItemButton
                    selected={selectedTarget === target}
                    onClick={() => setSelectedTarget(target)}
                    sx={{
                      pl: 2 + (target.level * 2),
                      '&.Mui-selected': {
                        backgroundColor: 'primary.light',
                        '&:hover': {
                          backgroundColor: 'primary.light',
                        }
                      }
                    }}
                  >
                    <Box sx={{ mr: 1, display: 'flex', alignItems: 'center' }}>
                      {getTargetIcon(target)}
                    </Box>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography 
                            variant="body2" 
                            component="span"
                            sx={{ 
                              fontFamily: 'monospace',
                              fontSize: '0.9rem'
                            }}
                          >
                            {target.display_name}
                          </Typography>
                        </Box>
                      }
                      secondary={
                        target.level > 0 && (
                          <Typography variant="caption" color="text.secondary">
                            {target.filename}
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
          onClick={() => onClose()}
          disabled={refiling}
        >
          Cancel
        </Button>
        <Button
          onClick={handleRefile}
          disabled={!selectedTarget || refiling}
          variant="contained"
          startIcon={refiling && <CircularProgress size={16} />}
        >
          {refiling ? 'Refiling...' : 'Refile'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default OrgRefileDialog;



