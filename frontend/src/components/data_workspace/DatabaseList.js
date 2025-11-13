import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Menu,
  MenuItem,
  Chip
} from '@mui/material';
import {
  Add as AddIcon,
  MoreVert as MoreVertIcon,
  Storage as StorageIcon,
  Upload as UploadIcon,
  Delete as DeleteIcon,
  ViewInAr as ViewInArIcon
} from '@mui/icons-material';

import dataWorkspaceService from '../../services/dataWorkspaceService';
import DataImportWizard from './DataImportWizard';

const DatabaseList = ({ workspaceId, databases, loading, onRefresh, onViewTables }) => {
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newDatabase, setNewDatabase] = useState({ name: '', description: '' });
  const [anchorEl, setAnchorEl] = useState(null);
  const [selectedDatabase, setSelectedDatabase] = useState(null);
  const [importWizardOpen, setImportWizardOpen] = useState(false);
  const [importTargetDatabase, setImportTargetDatabase] = useState(null);

  const handleCreateDatabase = async () => {
    try {
      if (!newDatabase.name.trim()) return;

      await dataWorkspaceService.createDatabase({
        workspace_id: workspaceId,
        name: newDatabase.name,
        description: newDatabase.description,
        source_type: 'imported'
      });

      setCreateDialogOpen(false);
      setNewDatabase({ name: '', description: '' });
      onRefresh();
    } catch (error) {
      console.error('Failed to create database:', error);
    }
  };

  const handleDeleteDatabase = async (databaseId) => {
    if (window.confirm('Are you sure you want to delete this database? All data will be lost.')) {
      try {
        await dataWorkspaceService.deleteDatabase(databaseId);
        onRefresh();
      } catch (error) {
        console.error('Failed to delete database:', error);
      }
    }
    setAnchorEl(null);
  };

  const handleMenuOpen = (event, database) => {
    setAnchorEl(event.currentTarget);
    setSelectedDatabase(database);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setSelectedDatabase(null);
  };

  const handleImportData = (database) => {
    setImportTargetDatabase(database);
    setImportWizardOpen(true);
    handleMenuClose();
  };

  const handleImportComplete = () => {
    onRefresh(); // Refresh database stats after import
  };

  const handleView3D = (database) => {
    // TODO: Launch 3D navigator for this database
    console.log('View 3D for database:', database.database_id);
    handleMenuClose();
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
        <Typography color="text.secondary">Loading databases...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">
          Databases ({databases.length})
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setCreateDialogOpen(true)}
        >
          Create Database
        </Button>
      </Box>

      {databases.length === 0 ? (
        <Card sx={{ textAlign: 'center', py: 8 }}>
          <StorageIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            No databases yet
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Create a database to start importing and organizing your data
          </Typography>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setCreateDialogOpen(true)}
          >
            Create Your First Database
          </Button>
        </Card>
      ) : (
        <Grid container spacing={3}>
          {databases.map((database) => (
            <Grid item xs={12} sm={6} md={4} key={database.database_id}>
              <Card 
                sx={{ 
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 4
                  }
                }}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <StorageIcon color="primary" />
                      <Typography variant="h6" component="div" noWrap>
                        {database.name}
                      </Typography>
                    </Box>
                    <IconButton 
                      size="small"
                      onClick={(e) => handleMenuOpen(e, database)}
                    >
                      <MoreVertIcon />
                    </IconButton>
                  </Box>

                  {database.description && (
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {database.description}
                    </Typography>
                  )}

                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip 
                      label={`${database.table_count} ${database.table_count === 1 ? 'table' : 'tables'}`}
                      size="small"
                      variant="outlined"
                    />
                    <Chip 
                      label={`${database.total_rows.toLocaleString()} rows`}
                      size="small"
                      variant="outlined"
                    />
                    <Chip 
                      label={database.source_type}
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                  </Box>
                </CardContent>

                <CardActions sx={{ justifyContent: 'space-between', px: 2, pb: 2 }}>
                  <Button 
                    size="small"
                    onClick={() => onViewTables && onViewTables(database)}
                    variant="contained"
                  >
                    View Tables
                  </Button>
                  <Button 
                    size="small"
                    startIcon={<UploadIcon />}
                    onClick={() => handleImportData(database)}
                  >
                    Import
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Context Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => handleImportData(selectedDatabase)}>
          <UploadIcon sx={{ mr: 1 }} fontSize="small" />
          Import Data
        </MenuItem>
        <MenuItem onClick={() => handleView3D(selectedDatabase)}>
          <ViewInArIcon sx={{ mr: 1 }} fontSize="small" />
          View in 3D
        </MenuItem>
        <MenuItem 
          onClick={() => selectedDatabase && handleDeleteDatabase(selectedDatabase.database_id)}
          sx={{ color: 'error.main' }}
        >
          <DeleteIcon sx={{ mr: 1 }} fontSize="small" />
          Delete
        </MenuItem>
      </Menu>

      {/* Create Database Dialog */}
      <Dialog
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Create Database</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Database Name"
            fullWidth
            value={newDatabase.name}
            onChange={(e) => setNewDatabase({ ...newDatabase, name: e.target.value })}
            required
          />
          <TextField
            margin="dense"
            label="Description"
            fullWidth
            multiline
            rows={3}
            value={newDatabase.description}
            onChange={(e) => setNewDatabase({ ...newDatabase, description: e.target.value })}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleCreateDatabase} 
            variant="contained" 
            disabled={!newDatabase.name.trim()}
          >
            Create
          </Button>
        </DialogActions>
      </Dialog>

      {/* Data Import Wizard */}
      {importTargetDatabase && (
        <DataImportWizard
          open={importWizardOpen}
          onClose={() => {
            setImportWizardOpen(false);
            setImportTargetDatabase(null);
          }}
          workspaceId={workspaceId}
          databaseId={importTargetDatabase.database_id}
          existingTables={[]}
          onImportComplete={handleImportComplete}
        />
      )}
    </Box>
  );
};

export default DatabaseList;

