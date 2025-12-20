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
  Collapse,
  Divider
} from '@mui/material';
import {
  Add as AddIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Delete as DeleteIcon,
  Share as ShareIcon
} from '@mui/icons-material';

import dataWorkspaceService from '../../services/dataWorkspaceService';
import WorkspaceShareDialog from './WorkspaceShareDialog';

const WORKSPACE_ICONS = ['📊', '💼', '🔬', '📈', '💾', '🗃️', '📁', '🎯'];
const WORKSPACE_COLORS = ['#1976d2', '#388e3c', '#d32f2f', '#f57c00', '#7b1fa2', '#0288d1', '#c62828'];

const DataWorkspacesSection = ({ onWorkspaceClick }) => {
  const [workspaces, setWorkspaces] = useState([]);
  const [expanded, setExpanded] = useState(() => {
    const saved = localStorage.getItem('dataWorkspacesExpanded');
    return saved !== null ? JSON.parse(saved) : true;
  });
  const [loading, setLoading] = useState(false);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  const [selectedWorkspace, setSelectedWorkspace] = useState(null);
  const [filter, setFilter] = useState('all'); // 'all', 'owned', 'shared'
  const [newWorkspace, setNewWorkspace] = useState({
    name: '',
    description: '',
    icon: '📊',
    color: '#1976d2'
  });

  useEffect(() => {
    loadWorkspaces();
  }, []);

  // Persist expanded state to localStorage
  useEffect(() => {
    localStorage.setItem('dataWorkspacesExpanded', JSON.stringify(expanded));
  }, [expanded]);

  const loadWorkspaces = async () => {
    try {
      setLoading(true);
      const data = await dataWorkspaceService.listWorkspaces(true); // Include shared
      setWorkspaces(data);
    } catch (error) {
      console.error('Failed to load workspaces:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleShareWorkspace = (workspace, event) => {
    event.stopPropagation();
    setSelectedWorkspace(workspace);
    setShareDialogOpen(true);
  };

  const handleShareDialogClose = () => {
    setShareDialogOpen(false);
    setSelectedWorkspace(null);
    loadWorkspaces(); // Reload to refresh share status
  };

  const getFilteredWorkspaces = () => {
    if (filter === 'owned') {
      return workspaces.filter(w => !w.is_shared);
    } else if (filter === 'shared') {
      return workspaces.filter(w => w.is_shared);
    }
    return workspaces;
  };

  const isWorkspaceOwner = (workspace) => {
    return !workspace.is_shared || workspace.permission_level === 'admin';
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
    <>
      <Divider sx={{ my: 0.5 }} />
      <Box>
        {/* Header */}
        <Box sx={{ px: 2, pb: 0.5 }}>
        <Box 
          sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            mb: 1, 
            cursor: 'pointer',
            '&:hover': { backgroundColor: 'action.hover' },
            borderRadius: 1,
            p: 0.5
          }}
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
          <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'text.secondary', fontSize: '0.75rem', textTransform: 'uppercase', ml: 0.5 }}>
            💾 Data Workspaces
          </Typography>
        </Box>
        
        {/* Filter Tabs */}
        {expanded && (
          <Box sx={{ display: 'flex', gap: 0.5, mb: 1 }}>
            <Button
              size="small"
              variant={filter === 'all' ? 'contained' : 'text'}
              onClick={() => setFilter('all')}
              sx={{ minWidth: 'auto', px: 1, fontSize: '0.7rem' }}
            >
              All
            </Button>
            <Button
              size="small"
              variant={filter === 'owned' ? 'contained' : 'text'}
              onClick={() => setFilter('owned')}
              sx={{ minWidth: 'auto', px: 1, fontSize: '0.7rem' }}
            >
              My Workspaces
            </Button>
            <Button
              size="small"
              variant={filter === 'shared' ? 'contained' : 'text'}
              onClick={() => setFilter('shared')}
              sx={{ minWidth: 'auto', px: 1, fontSize: '0.7rem' }}
            >
              Shared with Me
            </Button>
          </Box>
        )}
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
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="caption" color="text.secondary">No workspaces yet</Typography>
                    <Tooltip title="Create Workspace">
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          setCreateDialogOpen(true);
                        }}
                        sx={{ 
                          width: 20, 
                          height: 20,
                          '& .MuiSvgIcon-root': { fontSize: '0.875rem' }
                        }}
                      >
                        <AddIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                }
              />
            </ListItem>
          ) : (
            getFilteredWorkspaces().map((workspace) => (
              <ListItem
                key={workspace.workspace_id}
                disablePadding
                secondaryAction={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    {isWorkspaceOwner(workspace) && (
                      <Tooltip title="Share Workspace">
                        <IconButton
                          edge="end"
                          size="small"
                          onClick={(e) => handleShareWorkspace(workspace, e)}
                        >
                          <ShareIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    )}
                    {isWorkspaceOwner(workspace) && (
                      <IconButton
                        edge="end"
                        size="small"
                        onClick={(e) => handleDeleteWorkspace(workspace.workspace_id, e)}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    )}
                  </Box>
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
                        justifyContent: 'center',
                        position: 'relative'
                      }}
                    >
                      {workspace.icon || '📊'}
                      {workspace.is_shared && (
                        <Box
                          sx={{
                            position: 'absolute',
                            top: -4,
                            right: -4,
                            width: 8,
                            height: 8,
                            borderRadius: '50%',
                            bgcolor: 'primary.main',
                            border: '1px solid',
                            borderColor: 'background.paper'
                          }}
                        />
                      )}
                    </Box>
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Typography
                          variant="body2"
                          noWrap
                        >
                          {workspace.name}
                        </Typography>
                        {workspace.is_shared && (
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            sx={{ fontSize: '0.65rem' }}
                          >
                            ({workspace.share_type || 'shared'})
                          </Typography>
                        )}
                      </Box>
                    }
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

      {/* Share Workspace Dialog */}
      {selectedWorkspace && (
        <WorkspaceShareDialog
          open={shareDialogOpen}
          onClose={handleShareDialogClose}
          workspaceId={selectedWorkspace.workspace_id}
          workspaceName={selectedWorkspace.name}
        />
      )}
      </Box>
    </>
  );
};

export default DataWorkspacesSection;





