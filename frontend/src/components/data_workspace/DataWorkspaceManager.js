import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  IconButton,
  Tabs,
  Tab,
  Tooltip,
  CircularProgress,
  Grid,
  Card,
  CardContent,
  Menu,
  MenuItem
} from '@mui/material';
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  ViewInAr as ViewInArIcon,
  Dashboard as DashboardIcon,
  Storage as StorageIcon,
  MoreVert as MoreVertIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';

import dataWorkspaceService from '../../services/dataWorkspaceService';
import DatabaseList from './DatabaseList';
import TableCreationWizard from './TableCreationWizard';
import DataTableView from './DataTableView';

const DataWorkspaceManager = ({ workspaceId, fullScreen, onToggleFullScreen }) => {
  const [workspace, setWorkspace] = useState(null);
  const [databases, setDatabases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(0);
  const [showCreateDatabase, setShowCreateDatabase] = useState(false);
  
  // Table viewing state
  const [selectedDatabase, setSelectedDatabase] = useState(null);
  const [tables, setTables] = useState([]);
  const [selectedTable, setSelectedTable] = useState(null);
  const [tableSchema, setTableSchema] = useState(null);
  const [showTableWizard, setShowTableWizard] = useState(false);
  
  // Table menu state
  const [tableMenuAnchor, setTableMenuAnchor] = useState(null);
  const [tableForMenu, setTableForMenu] = useState(null);

  useEffect(() => {
    if (workspaceId) {
      loadWorkspace();
      loadDatabases();
    }
  }, [workspaceId]);

  const loadWorkspace = async () => {
    try {
      const data = await dataWorkspaceService.getWorkspace(workspaceId);
      setWorkspace(data);
    } catch (error) {
      console.error('Failed to load workspace:', error);
    }
  };

  const loadDatabases = async () => {
    try {
      setLoading(true);
      const data = await dataWorkspaceService.listDatabases(workspaceId);
      setDatabases(data);
    } catch (error) {
      console.error('Failed to load databases:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    loadDatabases();
  };

  const handleLaunch3DNavigator = () => {
    // TODO: Launch 3D navigator in new tab or modal
    console.log('Launch 3D Navigator for workspace:', workspaceId);
  };

  const handleViewTables = async (database) => {
    setSelectedDatabase(database);
    setActiveTab(2); // Switch to tables tab
    
    try {
      const tablesData = await dataWorkspaceService.listTables(database.database_id);
      setTables(tablesData);
    } catch (error) {
      console.error('Failed to load tables:', error);
      setTables([]);
    }
  };

  const handleTableCreated = async (newTable) => {
    // Reload tables from backend to ensure we have the latest data
    if (selectedDatabase) {
      try {
        const tablesData = await dataWorkspaceService.listTables(selectedDatabase.database_id);
        setTables(tablesData);
      } catch (error) {
        console.error('Failed to reload tables:', error);
      }
    }
    loadDatabases(); // Refresh database stats
    setShowTableWizard(false); // Close wizard
  };

  const handleSelectTable = async (table) => {
    setSelectedTable(table);
    try {
      // Parse schema
      const schema = typeof table.table_schema_json === 'string' 
        ? JSON.parse(table.table_schema_json) 
        : table.table_schema_json;
      setTableSchema(schema);
    } catch (error) {
      console.error('Failed to parse table schema:', error);
    }
  };

  const handleBackToDatabases = () => {
    setSelectedDatabase(null);
    setSelectedTable(null);
    setTableSchema(null);
    setActiveTab(0);
  };

  const handleTableMenuOpen = (event, table) => {
    event.stopPropagation(); // Prevent card click
    setTableMenuAnchor(event.currentTarget);
    setTableForMenu(table);
  };

  const handleTableMenuClose = () => {
    setTableMenuAnchor(null);
    setTableForMenu(null);
  };

  const handleDeleteTable = async () => {
    if (!tableForMenu) return;
    
    if (window.confirm(`Are you sure you want to delete the table "${tableForMenu.name}"? All data will be lost.`)) {
      try {
        await dataWorkspaceService.deleteTable(tableForMenu.table_id);
        // Reload tables
        const tablesData = await dataWorkspaceService.listTables(selectedDatabase.database_id);
        setTables(tablesData);
        // Reload database stats
        loadDatabases();
      } catch (error) {
        console.error('Failed to delete table:', error);
        alert('Failed to delete table. Please try again.');
      }
    }
    handleTableMenuClose();
  };

  if (!workspace) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', bgcolor: 'background.default' }}>
      {/* Header */}
      <Paper 
        elevation={1} 
        sx={{ 
          p: 2, 
          borderRadius: 0,
          borderBottom: 1,
          borderColor: 'divider'
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Box sx={{ fontSize: 32 }}>
              {workspace.icon || '📊'}
            </Box>
            <Box>
              <Typography variant="h5" sx={{ fontWeight: 600 }}>
                {workspace.name}
              </Typography>
              {workspace.description && (
                <Typography variant="body2" color="text.secondary">
                  {workspace.description}
                </Typography>
              )}
            </Box>
          </Box>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Tooltip title="3D Navigator (FSN Style)">
              <IconButton onClick={handleLaunch3DNavigator} color="primary">
                <ViewInArIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Refresh">
              <IconButton onClick={handleRefresh}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </Paper>

      {/* Tabs */}
      <Paper elevation={0} sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)}>
          <Tab icon={<StorageIcon />} label="Databases" iconPosition="start" />
          <Tab icon={<DashboardIcon />} label="Overview" iconPosition="start" />
          {selectedDatabase && (
            <Tab 
              icon={<StorageIcon />} 
              label={`Tables - ${selectedDatabase.name}`} 
              iconPosition="start" 
            />
          )}
        </Tabs>
      </Paper>

      {/* Content Area */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', p: 3 }}>
        {activeTab === 0 && (
          <DatabaseList
            workspaceId={workspaceId}
            databases={databases}
            loading={loading}
            onRefresh={loadDatabases}
            onViewTables={handleViewTables}
          />
        )}
        
        {activeTab === 1 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Workspace Overview
            </Typography>
            <Paper sx={{ p: 3, mt: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Statistics and visualizations coming soon...
              </Typography>
            </Paper>
          </Box>
        )}

        {activeTab === 2 && selectedDatabase && (
          <Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Box>
                <Button onClick={handleBackToDatabases} sx={{ mb: 1 }}>
                  ← Back to Databases
                </Button>
                <Typography variant="h6">
                  Tables in {selectedDatabase.name}
                </Typography>
              </Box>
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={() => setShowTableWizard(true)}
              >
                Create Table
              </Button>
            </Box>

            {selectedTable && tableSchema ? (
              <Box>
                <Button onClick={() => setSelectedTable(null)} sx={{ mb: 2 }}>
                  ← Back to Tables
                </Button>
                <Paper sx={{ height: 'calc(100vh - 300px)' }}>
                  <DataTableView
                    tableId={selectedTable.table_id}
                    schema={tableSchema}
                    onDataChange={() => {}}
                  />
                </Paper>
              </Box>
            ) : (
              <Box>
                {tables.length === 0 ? (
                  <Paper sx={{ textAlign: 'center', py: 8 }}>
                    <StorageIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="h6" gutterBottom>
                      No tables yet
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      Create your first table to start organizing data
                    </Typography>
                    <Button
                      variant="contained"
                      startIcon={<AddIcon />}
                      onClick={() => setShowTableWizard(true)}
                    >
                      Create Your First Table
                    </Button>
                  </Paper>
                ) : (
                  <Grid container spacing={2}>
                    {tables.map((table) => (
                      <Grid item xs={12} sm={6} md={4} key={table.table_id}>
                        <Card 
                          sx={{ 
                            cursor: 'pointer',
                            '&:hover': { boxShadow: 4 },
                            position: 'relative'
                          }}
                          onClick={() => handleSelectTable(table)}
                        >
                          <IconButton
                            size="small"
                            onClick={(e) => handleTableMenuOpen(e, table)}
                            sx={{
                              position: 'absolute',
                              top: 8,
                              right: 8
                            }}
                          >
                            <MoreVertIcon />
                          </IconButton>
                          <CardContent>
                            <Typography variant="h6" gutterBottom sx={{ pr: 4 }}>
                              {table.name}
                            </Typography>
                            {table.description && (
                              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                {table.description}
                              </Typography>
                            )}
                            <Typography variant="body2" color="text.secondary">
                              {table.row_count.toLocaleString()} rows
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                )}
              </Box>
            )}
          </Box>
        )}
      </Box>

      {/* Table Menu */}
      <Menu
        anchorEl={tableMenuAnchor}
        open={Boolean(tableMenuAnchor)}
        onClose={handleTableMenuClose}
      >
        <MenuItem 
          onClick={handleDeleteTable}
          sx={{ color: 'error.main' }}
        >
          <DeleteIcon sx={{ mr: 1 }} fontSize="small" />
          Delete Table
        </MenuItem>
      </Menu>

      {/* Table Creation Wizard */}
      <TableCreationWizard
        open={showTableWizard}
        onClose={() => setShowTableWizard(false)}
        databaseId={selectedDatabase?.database_id}
        onTableCreated={handleTableCreated}
      />
    </Box>
  );
};

export default DataWorkspaceManager;

