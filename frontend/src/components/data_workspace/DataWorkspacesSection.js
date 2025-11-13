import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Tooltip,
  Collapse
} from '@mui/material';
import {
  Storage as StorageIcon,
  Add as AddIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';

import dataWorkspaceService from '../../services/dataWorkspaceService';

const WORKSPACE_ICONS = ['📊', '💼', '🔬', '📈', '💾', '🗃️', '📁', '🎯'];
const WORKSPACE_COLORS = ['#1976d2', '#388e3c', '#d32f2f', '#f57c00', '#7b1fa2', '#0288d1', '#c62828'];

const DataWorkspacesSection = ({ onWorkspaceClick }) => {
  const [workspaces, setWorkspaces] = useState([]);
  const [expanded, setExpanded] = useState(true);
  const [loading, setLoading] = useState(false);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newWorkspace, setNewWorkspace] = useState({
    name: '',
    description: '',
    icon: '📊',
    color: '#1976d2'
  });

  useEffect(() => {
    loadWorkspaces();
  }, []);

  const loadWorkspaces = async () => {
    try {
      setLoading(true);
      const data = await dataWorkspaceService.listWorkspaces();
      setWorkspaces(data);
    } catch (error) {
      console.error('Failed to load workspaces:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateWorkspace = async () => {
    try {
      if (!newWorkspace.name.trim()) {
        return;
      }

      await dataWorkspaceService.createWorkspace(newWorkspace);
      await loadWorkspaces();
      
      setCreateDialogOpen(false);
      setNewWorkspace({
        name: '',
        description: '',
        icon: '📊',
        color: '#1976d2'
      });
    } catch (error) {
      console.error('Failed to create workspace:', error);
    }
  };

  const handleDeleteWorkspace = async (workspaceId, event) => {
    event.stopPropagation();
    
    if (window.confirm('Are you sure you want to delete this workspace? All data will be lost.')) {
      try {
        await dataWorkspaceService.deleteWorkspace(workspaceId);
        await loadWorkspaces();
      } catch (error) {
        console.error('Failed to delete workspace:', error);
      }
    }
  };

  const handleWorkspaceClick = (workspace) => {
    if (onWorkspaceClick) {
      onWorkspaceClick(workspace);
    }
  };

  return (
    <Box sx={{ mt: 2 }}>
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 2,
          py: 1,
          cursor: 'pointer'
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <StorageIcon sx={{ fontSize: 20, color: 'text.secondary' }} />
          <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
            Data Workspaces
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Tooltip title="Create Workspace">
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation();
                setCreateDialogOpen(true);
              }}
            >
              <AddIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <IconButton size="small">
            {expanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
          </IconButton>
        </Box>
      </Box>

      {/* Workspaces List */}
      <Collapse in={expanded}>
        <List dense sx={{ py: 0 }}>
          {loading ? (
            <ListItem>
              <ListItemText 
                primary={<Typography variant="caption" color="text.secondary">Loading...</Typography>}
              />
            </ListItem>
          ) : workspaces.length === 0 ? (
            <ListItem>
              <ListItemText 
                primary={<Typography variant="caption" color="text.secondary">No workspaces yet</Typography>}
              />
            </ListItem>
          ) : (
            workspaces.map((workspace) => (
              <ListItem
                key={workspace.workspace_id}
                disablePadding
                secondaryAction={
                  <IconButton
                    edge="end"
                    size="small"
                    onClick={(e) => handleDeleteWorkspace(workspace.workspace_id, e)}
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                }
              >
                <ListItemButton
                  onClick={() => handleWorkspaceClick(workspace)}
                  sx={{ pl: 4 }}
                >
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <Box
                      sx={{
                        fontSize: 18,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}
                    >
                      {workspace.icon || '📊'}
                    </Box>
                  </ListItemIcon>
                  <ListItemText
                    primary={workspace.name}
                    primaryTypographyProps={{
                      variant: 'body2',
                      noWrap: true
                    }}
                  />
                </ListItemButton>
              </ListItem>
            ))
          )}
        </List>
      </Collapse>

      {/* Create Workspace Dialog */}
      <Dialog
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Create Data Workspace</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Workspace Name"
            fullWidth
            value={newWorkspace.name}
            onChange={(e) => setNewWorkspace({ ...newWorkspace, name: e.target.value })}
            required
          />
          <TextField
            margin="dense"
            label="Description"
            fullWidth
            multiline
            rows={2}
            value={newWorkspace.description}
            onChange={(e) => setNewWorkspace({ ...newWorkspace, description: e.target.value })}
          />
          
          <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
            Icon
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {WORKSPACE_ICONS.map((icon) => (
              <Box
                key={icon}
                sx={{
                  fontSize: 24,
                  cursor: 'pointer',
                  padding: 1,
                  borderRadius: 1,
                  border: '2px solid',
                  borderColor: newWorkspace.icon === icon ? 'primary.main' : 'transparent',
                  bgcolor: newWorkspace.icon === icon ? 'action.selected' : 'transparent',
                  '&:hover': {
                    bgcolor: 'action.hover'
                  }
                }}
                onClick={() => setNewWorkspace({ ...newWorkspace, icon })}
              >
                {icon}
              </Box>
            ))}
          </Box>

          <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
            Color
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {WORKSPACE_COLORS.map((color) => (
              <Box
                key={color}
                sx={{
                  width: 32,
                  height: 32,
                  borderRadius: '50%',
                  bgcolor: color,
                  cursor: 'pointer',
                  border: '3px solid',
                  borderColor: newWorkspace.color === color ? 'background.paper' : 'transparent',
                  boxShadow: newWorkspace.color === color ? 2 : 0,
                  '&:hover': {
                    boxShadow: 2
                  }
                }}
                onClick={() => setNewWorkspace({ ...newWorkspace, color })}
              />
            ))}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleCreateWorkspace} variant="contained" disabled={!newWorkspace.name.trim()}>
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DataWorkspacesSection;





