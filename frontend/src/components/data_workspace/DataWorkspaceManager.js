import React, { useState, useEffect, useRef } from 'react';
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
  MenuItem,
  ListItemText,
  Breadcrumbs,
  Link
} from '@mui/material';
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  Dashboard as DashboardIcon,
  Storage as StorageIcon,
  MoreVert as MoreVertIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Code as CodeIcon,
  Fullscreen,
  FullscreenExit
} from '@mui/icons-material';

import dataWorkspaceService from '../../services/dataWorkspaceService';
import DatabaseList from './DatabaseList';
import TableCreationWizard from './TableCreationWizard';
import TableSchemaEditDialog from './TableSchemaEditDialog';
import DataTableView from './DataTableView';
import RunSqlDialog from './RunSqlDialog';

const DataWorkspaceManager = ({ workspaceId, onFullscreenChange }) => {
  const fullscreenContainerRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const modalContainer =
    isFullscreen && fullscreenContainerRef.current ? fullscreenContainerRef.current : undefined;

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
  const [showEditSchemaDialog, setShowEditSchemaDialog] = useState(false);
  const [tableForEdit, setTableForEdit] = useState(null);
  const [showRunSqlDialog, setShowRunSqlDialog] = useState(false);
  const [currentRows, setCurrentRows] = useState([]);
  const [currentTotalRows, setCurrentTotalRows] = useState(0);

  useEffect(() => {
    if (workspaceId) {
      loadWorkspace();
      loadDatabases();
    }
  }, [workspaceId]);

  useEffect(() => {
    return () => {
      try {
        localStorage.removeItem('data_workspace_ctx_cache');
      } catch (e) {
        // ignore
      }
    };
  }, []);

  useEffect(() => {
    if (!selectedTable || !tableSchema) {
      try {
        localStorage.removeItem('data_workspace_ctx_cache');
      } catch (e) {
        // ignore
      }
      return;
    }
    if (!workspace || !selectedDatabase) return;
    const columns = tableSchema.columns || (Array.isArray(tableSchema) ? tableSchema : []);
    if (columns.length === 0) return;
    try {
      const cache = {
        workspace_id: workspaceId,
        workspace_name: workspace.name || '',
        database_id: selectedDatabase.database_id,
        database_name: selectedDatabase.name || '',
        table_id: selectedTable.table_id,
        table_name: selectedTable.name || '',
        row_count: selectedTable.row_count ?? currentTotalRows,
        schema: columns.map((col) => ({
          name: col.name,
          type: col.type || 'TEXT',
          description: col.description || ''
        })),
        visible_rows: currentRows,
        visible_row_count: currentRows.length
      };
      localStorage.setItem('data_workspace_ctx_cache', JSON.stringify(cache));
    } catch (e) {
      // ignore
    }
  }, [workspaceId, workspace, selectedDatabase, selectedTable, tableSchema, currentRows, currentTotalRows]);

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

  const handleToggleFullscreen = async () => {
    if (!fullscreenContainerRef.current || typeof document === 'undefined' || !document) return;

    const element = fullscreenContainerRef.current;
    const isCurrentlyFullscreen = !!(
      document.fullscreenElement ||
      document.webkitFullscreenElement ||
      document.mozFullScreenElement ||
      document.msFullscreenElement
    );

    try {
      if (!isCurrentlyFullscreen) {
        if (element.requestFullscreen) await element.requestFullscreen();
        else if (element.webkitRequestFullscreen) await element.webkitRequestFullscreen();
        else if (element.mozRequestFullScreen) await element.mozRequestFullScreen();
        else if (element.msRequestFullscreen) await element.msRequestFullscreen();
        else {
          console.warn('Fullscreen API not supported');
          return;
        }
        setIsFullscreen(true);
      } else {
        if (document.exitFullscreen) await document.exitFullscreen();
        else if (document.webkitExitFullscreen) await document.webkitExitFullscreen();
        else if (document.mozCancelFullScreen) await document.mozCancelFullScreen();
        else if (document.msExitFullscreen) await document.msExitFullscreen();
        setIsFullscreen(false);
      }
    } catch (err) {
      console.error('Fullscreen error:', err);
      setIsFullscreen(false);
    }
  };

  useEffect(() => {
    if (typeof document === 'undefined' || !document) return;

    const handleFullscreenChange = () => {
      const next = !!(
        document.fullscreenElement ||
        document.webkitFullscreenElement ||
        document.mozFullScreenElement ||
        document.msFullscreenElement
      );
      setIsFullscreen(next);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('mozfullscreenchange', handleFullscreenChange);
    document.addEventListener('MSFullscreenChange', handleFullscreenChange);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
      document.removeEventListener('mozfullscreenchange', handleFullscreenChange);
      document.removeEventListener('MSFullscreenChange', handleFullscreenChange);
    };
  }, []);

  useEffect(() => {
    if (typeof onFullscreenChange === 'function') {
      onFullscreenChange(isFullscreen);
    }
  }, [isFullscreen, onFullscreenChange]);

  const handleViewTables = async (database) => {
    setSelectedDatabase(database);
    setActiveTab(1); // Switch to tables tab
    
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

  const handleOpenLinkedRow = async ({ targetTableId }) => {
    if (!targetTableId || !selectedDatabase) return;
    let t = tables.find((x) => x.table_id === targetTableId);
    if (!t) {
      try {
        const fresh = await dataWorkspaceService.listTables(selectedDatabase.database_id);
        setTables(fresh);
        t = fresh.find((x) => x.table_id === targetTableId);
      } catch (e) {
        console.error('Failed to load tables for linked row:', e);
        return;
      }
    }
    if (!t) return;
    await handleSelectTable(t);
    setActiveTab(1);
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

  const handleEditTableSchema = () => {
    if (!tableForMenu) return;
    setTableForEdit(tableForMenu);
    setShowEditSchemaDialog(true);
    handleTableMenuClose();
  };

  const handleEditSchemaDialogClose = () => {
    setShowEditSchemaDialog(false);
    setTableForEdit(null);
  };

  const handleEditSchemaSaved = async () => {
    if (selectedDatabase) {
      try {
        const tablesData = await dataWorkspaceService.listTables(selectedDatabase.database_id);
        setTables(tablesData);
        if (tableForEdit && selectedTable?.table_id === tableForEdit.table_id) {
          const updated = tablesData.find((t) => t.table_id === tableForEdit.table_id);
          if (updated) {
            setSelectedTable(updated);
            const schema = typeof updated.table_schema_json === 'string'
              ? JSON.parse(updated.table_schema_json)
              : updated.table_schema_json;
            setTableSchema(schema || {});
          }
        }
      } catch (err) {
        console.error('Failed to reload tables after edit:', err);
      }
    }
    loadDatabases();
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
    <Box
      ref={fullscreenContainerRef}
      sx={{ height: '100%', display: 'flex', flexDirection: 'column', bgcolor: 'background.default', overflow: 'hidden' }}
    >
      {/* Header */}
      <Paper 
        elevation={1} 
        sx={{ 
          px: 2,
          py: 1.25,
          borderRadius: 0,
          borderBottom: 1,
          borderColor: 'divider'
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Box sx={{ fontSize: 24, lineHeight: 1 }}>
              {workspace.icon || '📊'}
            </Box>
            <Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 700, lineHeight: 1.1 }}>
                {workspace.name}
              </Typography>
              {workspace.description && (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', lineHeight: 1.2 }}>
                  {workspace.description}
                </Typography>
              )}
            </Box>
          </Box>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Tooltip title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}>
              <IconButton onClick={handleToggleFullscreen} color="primary" aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}>
                {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
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
          {selectedDatabase && (
            <Tab icon={<StorageIcon />} label="Tables" iconPosition="start" />
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
            modalContainer={modalContainer}
          />
        )}
        
        {activeTab === 1 && selectedDatabase && (
          <Box>
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: 2,
                mb: 2
              }}
            >
              <Breadcrumbs aria-label="Data workspace navigation" sx={{ minWidth: 0 }}>
                <Link
                  component="button"
                  variant="body2"
                  underline="hover"
                  color="inherit"
                  onClick={handleBackToDatabases}
                  sx={{
                    cursor: 'pointer',
                    border: 0,
                    background: 'none',
                    font: 'inherit',
                    p: 0,
                    textAlign: 'left',
                    maxWidth: { xs: '40vw', sm: 'none' }
                  }}
                >
                  Databases
                </Link>
                {selectedTable && tableSchema ? (
                  <>
                    <Link
                      component="button"
                      variant="body2"
                      underline="hover"
                      color="inherit"
                      onClick={() => setSelectedTable(null)}
                      sx={{
                        cursor: 'pointer',
                        border: 0,
                        background: 'none',
                        font: 'inherit',
                        p: 0,
                        textAlign: 'left',
                        maxWidth: { xs: '35vw', sm: 'none' }
                      }}
                    >
                      {selectedDatabase.name}
                    </Link>
                    <Typography color="text.primary" variant="body2" noWrap sx={{ maxWidth: { xs: '35vw', sm: 240 } }}>
                      {selectedTable.name}
                    </Typography>
                  </>
                ) : (
                  <Typography color="text.primary" variant="body2" noWrap sx={{ maxWidth: { xs: '50vw', sm: 320 } }}>
                    {selectedDatabase.name}
                  </Typography>
                )}
              </Breadcrumbs>
              <Box sx={{ display: 'flex', gap: 1, flexShrink: 0 }}>
                <Button
                  variant="outlined"
                  startIcon={<CodeIcon />}
                  onClick={() => setShowRunSqlDialog(true)}
                >
                  Run SQL
                </Button>
                <Button
                  variant="contained"
                  startIcon={<AddIcon />}
                  onClick={() => setShowTableWizard(true)}
                >
                  Create Table
                </Button>
              </Box>
            </Box>

            {selectedTable && tableSchema ? (
              <Box>
                <Paper sx={{ height: 'calc(100vh - 300px)', mt: 1 }}>
                  <DataTableView
                    tableId={selectedTable.table_id}
                    databaseId={selectedTable.database_id}
                    schema={tableSchema}
                    modalContainer={modalContainer}
                    onDataChange={() => {}}
                    onRowsLoaded={(rows, total) => {
                      setCurrentRows(rows || []);
                      setCurrentTotalRows(total ?? 0);
                    }}
                    onEditTableSchema={() => {
                      setTableForEdit(selectedTable);
                      setShowEditSchemaDialog(true);
                    }}
                    onOpenLinkedRow={handleOpenLinkedRow}
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
        container={modalContainer}
      >
        <MenuItem
          onClick={handleEditTableSchema}
          sx={{ alignItems: 'flex-start', py: 1, maxWidth: 280 }}
        >
          <EditIcon sx={{ mr: 1, mt: 0.25 }} fontSize="small" />
          <ListItemText
            primary="Table schema"
            secondary="Add, remove, or reorder columns"
            primaryTypographyProps={{ variant: 'body2' }}
            secondaryTypographyProps={{ variant: 'caption' }}
          />
        </MenuItem>
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
        container={modalContainer}
      />

      {/* Table schema edit dialog */}
      <TableSchemaEditDialog
        open={showEditSchemaDialog}
        onClose={handleEditSchemaDialogClose}
        table={tableForEdit}
        onSaved={handleEditSchemaSaved}
        container={modalContainer}
      />

      {/* Run SQL dialog */}
      <RunSqlDialog
        open={showRunSqlDialog}
        onClose={() => setShowRunSqlDialog(false)}
        workspaceId={workspaceId}
        container={modalContainer}
        onSuccess={() => {
          if (selectedDatabase) {
            dataWorkspaceService.listTables(selectedDatabase.database_id).then(setTables).catch(() => {});
          }
          loadDatabases();
        }}
      />
    </Box>
  );
};

export default DataWorkspaceManager;

