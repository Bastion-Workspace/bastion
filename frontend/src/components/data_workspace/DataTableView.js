import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  TextField,
  Button,
  Tooltip,
  Typography,
  CircularProgress,
  Pagination,
  Alert,
  Checkbox,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  Delete as DeleteIcon,
  Edit as EditIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  Add as AddIcon,
  MoreVert as MoreVertIcon,
  FilterList as FilterIcon,
  ArrowUpward as ArrowUpwardIcon,
  ArrowDownward as ArrowDownwardIcon,
  Functions as FunctionsIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

import dataWorkspaceService from '../../services/dataWorkspaceService';
import FormulaBar from './FormulaBar';

const DataTableView = ({ 
  tableId, 
  schema, 
  onDataChange,
  readOnly = false 
}) => {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const [totalRows, setTotalRows] = useState(0);
  const [rowsPerPage] = useState(100);
  const [editingRow, setEditingRow] = useState(null);
  const [editingData, setEditingData] = useState({});
  const [editingCell, setEditingCell] = useState(null); // { rowId, columnName }
  const [selectedRows, setSelectedRows] = useState(new Set());
  const [contextMenu, setContextMenu] = useState(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [rowToDelete, setRowToDelete] = useState(null);
  const [sortColumn, setSortColumn] = useState(null);
  const [sortDirection, setSortDirection] = useState('asc'); // 'asc' or 'desc'
  const [selectedCell, setSelectedCell] = useState(null); // { rowId, columnName, rowIndex, columnIndex }
  const [recalculating, setRecalculating] = useState(false);

  useEffect(() => {
    if (tableId) {
      loadData();
    }
  }, [tableId, page]);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const offset = (page - 1) * rowsPerPage;
      const response = await dataWorkspaceService.getTableData(tableId, offset, rowsPerPage);
      
      let loadedRows = response.rows || [];
      
      // Ensure formula_data exists for each row
      loadedRows = loadedRows.map(row => ({
        ...row,
        formula_data: row.formula_data || {}
      }));
      
      // Apply client-side sorting if a sort column is set
      if (sortColumn) {
        loadedRows = sortRows(loadedRows, sortColumn, sortDirection);
      }
      
      setRows(loadedRows);
      setTotalRows(response.total_rows || 0);
    } catch (err) {
      console.error('Failed to load table data:', err);
      setError(err.message || 'Failed to load data');
      // Show empty table on error
      setRows([]);
      setTotalRows(0);
    } finally {
      setLoading(false);
    }
  };

  const sortRows = (rowsToSort, columnName, direction) => {
    return [...rowsToSort].sort((a, b) => {
      const aVal = a.row_data[columnName];
      const bVal = b.row_data[columnName];
      
      // Handle null/undefined values
      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;
      
      // Compare values
      let comparison = 0;
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        comparison = aVal - bVal;
      } else if (typeof aVal === 'boolean' && typeof bVal === 'boolean') {
        comparison = aVal === bVal ? 0 : aVal ? -1 : 1;
      } else {
        comparison = String(aVal).localeCompare(String(bVal));
      }
      
      return direction === 'asc' ? comparison : -comparison;
    });
  };

  const handleColumnSort = (columnName) => {
    if (sortColumn === columnName) {
      // Toggle direction
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      // New column, default to ascending
      setSortColumn(columnName);
      setSortDirection('asc');
    }
  };

  // Re-sort when sort column or direction changes
  useEffect(() => {
    if (sortColumn && rows.length > 0) {
      const sorted = sortRows(rows, sortColumn, sortDirection);
      setRows(sorted);
    }
  }, [sortColumn, sortDirection]);

  const handleStartEdit = (row) => {
    setEditingRow(row.row_id);
    setEditingData({ ...row.row_data });
    setEditingCell(null); // Clear any cell editing
  };

  const handleCancelEdit = () => {
    setEditingRow(null);
    setEditingData({});
    setEditingCell(null);
  };

  const handleCellDoubleClick = (row, column) => {
    if (readOnly || editingRow) return; // Don't allow cell edit if already editing row
    
    const columnIndex = schema.columns.findIndex(c => c.name === column.name);
    setSelectedCell({
      rowId: row.row_id,
      columnName: column.name,
      rowIndex: row.row_index,
      columnIndex: columnIndex
    });
    
    setEditingCell({ rowId: row.row_id, columnName: column.name });
    const cellValue = row.row_data[column.name];
    const formula = row.formula_data && row.formula_data[column.name] ? row.formula_data[column.name] : null;
    setEditingData({ [column.name]: formula || cellValue });
  };
  
  const handleCellClick = (row, column) => {
    if (readOnly) return;
    
    const columnIndex = schema.columns.findIndex(c => c.name === column.name);
    setSelectedCell({
      rowId: row.row_id,
      columnName: column.name,
      rowIndex: row.row_index,
      columnIndex: columnIndex
    });
  };
  
  const isFormula = (value) => {
    return typeof value === 'string' && value.trim().startsWith('=');
  };
  
  const getCellFormula = (row, columnName) => {
    return row.formula_data && row.formula_data[columnName] ? row.formula_data[columnName] : null;
  };
  
  const handleFormulaSave = async (formula) => {
    if (!selectedCell) return;
    
    try {
      const isFormulaValue = isFormula(formula);
      await dataWorkspaceService.updateTableCell(
        tableId,
        selectedCell.rowId,
        selectedCell.columnName,
        isFormulaValue ? null : formula,
        isFormulaValue ? formula : null
      );
      
      await loadData();
      if (onDataChange) onDataChange();
    } catch (err) {
      console.error('Failed to save formula:', err);
      setError(err.message || 'Failed to save formula');
    }
  };
  
  const handleRecalculate = async () => {
    try {
      setRecalculating(true);
      setError(null);
      await dataWorkspaceService.recalculateTable(tableId);
      await loadData();
      if (onDataChange) onDataChange();
    } catch (err) {
      console.error('Failed to recalculate:', err);
      setError(err.message || 'Failed to recalculate formulas');
    } finally {
      setRecalculating(false);
    }
  };

  const handleSaveCellEdit = async (rowId, columnName) => {
    try {
      const value = editingData[columnName];
      const formulaValue = isFormula(value) ? value : null;
      const actualValue = isFormula(value) ? null : value;
      
      await dataWorkspaceService.updateTableCell(
        tableId,
        rowId,
        columnName,
        actualValue,
        formulaValue
      );
      
      await loadData();
      setEditingCell(null);
      setEditingData({});
      
      if (onDataChange) onDataChange();
    } catch (err) {
      console.error('Failed to save cell:', err);
      setError(err.message || 'Failed to save cell');
    }
  };

  const handleCancelCellEdit = () => {
    setEditingCell(null);
    setEditingData({});
  };

  const handleSaveEdit = async (rowId) => {
    try {
      await dataWorkspaceService.updateTableRow(tableId, rowId, editingData);
      
      // Update local state
      const updatedRows = rows.map(row => 
        row.row_id === rowId ? { ...row, row_data: editingData } : row
      );
      setRows(updatedRows);
      setEditingRow(null);
      setEditingData({});
      
      if (onDataChange) onDataChange();
    } catch (err) {
      console.error('Failed to save row:', err);
      setError(err.message || 'Failed to save changes');
    }
  };

  const handleDeleteRow = async (rowId) => {
    try {
      await dataWorkspaceService.deleteTableRow(tableId, rowId);
      
      // Update local state
      const updatedRows = rows.filter(row => row.row_id !== rowId);
      setRows(updatedRows);
      setTotalRows(totalRows - 1);
      setDeleteDialogOpen(false);
      setRowToDelete(null);
      
      if (onDataChange) onDataChange();
    } catch (err) {
      console.error('Failed to delete row:', err);
      setError(err.message || 'Failed to delete row');
    }
  };

  const handleAddRow = async () => {
    try {
      const newRowData = schema.columns.reduce((acc, col) => {
        if (col.defaultValue) acc[col.name] = col.defaultValue;
        else if (col.type === 'INTEGER' || col.type === 'REAL') acc[col.name] = 0;
        else if (col.type === 'BOOLEAN') acc[col.name] = false;
        else if (col.type === 'TIMESTAMP') acc[col.name] = new Date().toISOString();
        else acc[col.name] = '';
        return acc;
      }, {});

      const response = await dataWorkspaceService.insertTableRow(tableId, newRowData);
      
      // Use response from server or create a temporary row
      const newRow = response || {
        row_id: `row_new_${Date.now()}`,
        row_data: newRowData
      };
      
      setRows([...rows, newRow]);
      setTotalRows(totalRows + 1);
      setEditingRow(newRow.row_id);
      setEditingData(newRowData);
      
      if (onDataChange) onDataChange();
    } catch (err) {
      console.error('Failed to add row:', err);
      setError(err.message || 'Failed to add row');
    }
  };

  const handleCellChange = (columnName, value) => {
    setEditingData({
      ...editingData,
      [columnName]: value
    });
  };

  const handleSelectRow = (rowId) => {
    const newSelected = new Set(selectedRows);
    if (newSelected.has(rowId)) {
      newSelected.delete(rowId);
    } else {
      newSelected.add(rowId);
    }
    setSelectedRows(newSelected);
  };

  const handleSelectAll = (event) => {
    if (event.target.checked) {
      setSelectedRows(new Set(rows.map(row => row.row_id)));
    } else {
      setSelectedRows(new Set());
    }
  };

  const renderCell = (row, column) => {
    const isEditingRow = editingRow === row.row_id;
    const isEditingCell = editingCell?.rowId === row.row_id && editingCell?.columnName === column.name;
    const isEditing = isEditingRow || isEditingCell;
    const cellFormula = getCellFormula(row, column.name);
    const value = isEditing ? editingData[column.name] : row.row_data[column.name];
    const hasFormula = cellFormula !== null;
    
    if (!isEditing) {
      let displayValue = value;
      if (column.type === 'BOOLEAN') {
        displayValue = value ? '✓' : '✗';
      } else if (column.type === 'TIMESTAMP' && value) {
        displayValue = new Date(value).toLocaleString();
      } else if (value === null || value === undefined) {
        displayValue = '-';
      }
      
      return (
        <Box
          onClick={() => handleCellClick(row, column)}
          onDoubleClick={() => handleCellDoubleClick(row, column)}
          sx={{
            cursor: readOnly ? 'default' : 'pointer',
            minHeight: 24,
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
            '&:hover': readOnly ? {} : {
              backgroundColor: 'action.hover',
              borderRadius: 0.5
            }
          }}
        >
          {hasFormula && (
            <FunctionsIcon fontSize="small" sx={{ color: 'primary.main', fontSize: 16 }} />
          )}
          <Typography 
            variant="body2" 
            sx={{ 
              color: column.color || 'inherit',
              fontWeight: column.color ? 600 : 400
            }}
          >
            {String(displayValue)}
          </Typography>
        </Box>
      );
    }

    // Editing mode (row or cell)
    if (column.type === 'BOOLEAN') {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Checkbox
            checked={Boolean(value)}
            onChange={(e) => handleCellChange(column.name, e.target.checked)}
            size="small"
            autoFocus={isEditingCell}
          />
          {isEditingCell && (
            <Box sx={{ display: 'flex', gap: 0.5 }}>
              <IconButton
                size="small"
                color="primary"
                onClick={() => handleSaveCellEdit(row.row_id, column.name)}
              >
                <SaveIcon fontSize="small" />
              </IconButton>
              <IconButton
                size="small"
                onClick={handleCancelCellEdit}
              >
                <CancelIcon fontSize="small" />
              </IconButton>
            </Box>
          )}
        </Box>
      );
    } else if (column.type === 'INTEGER' || column.type === 'REAL') {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <TextField
            value={value || ''}
            onChange={(e) => handleCellChange(column.name, e.target.value)}
            onKeyDown={(e) => {
              if (isEditingCell) {
                if (e.key === 'Enter') {
                  handleSaveCellEdit(row.row_id, column.name);
                } else if (e.key === 'Escape') {
                  handleCancelCellEdit();
                }
              }
            }}
            type="number"
            size="small"
            fullWidth
            autoFocus={isEditingCell}
            inputProps={{ step: column.type === 'REAL' ? 0.01 : 1 }}
          />
          {isEditingCell && (
            <Box sx={{ display: 'flex', gap: 0.5 }}>
              <IconButton
                size="small"
                color="primary"
                onClick={() => handleSaveCellEdit(row.row_id, column.name)}
              >
                <SaveIcon fontSize="small" />
              </IconButton>
              <IconButton
                size="small"
                onClick={handleCancelCellEdit}
              >
                <CancelIcon fontSize="small" />
              </IconButton>
            </Box>
          )}
        </Box>
      );
    } else {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <TextField
            value={value || ''}
            onChange={(e) => handleCellChange(column.name, e.target.value)}
            onKeyDown={(e) => {
              if (isEditingCell) {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSaveCellEdit(row.row_id, column.name);
                } else if (e.key === 'Escape') {
                  handleCancelCellEdit();
                }
              }
            }}
            size="small"
            fullWidth
            autoFocus={isEditingCell}
            multiline={column.type === 'TEXT'}
            maxRows={3}
          />
          {isEditingCell && (
            <Box sx={{ display: 'flex', gap: 0.5 }}>
              <IconButton
                size="small"
                color="primary"
                onClick={() => handleSaveCellEdit(row.row_id, column.name)}
              >
                <SaveIcon fontSize="small" />
              </IconButton>
              <IconButton
                size="small"
                onClick={handleCancelCellEdit}
              >
                <CancelIcon fontSize="small" />
              </IconButton>
            </Box>
          )}
        </Box>
      );
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {error}
      </Alert>
    );
  }

  const getSelectedCellFormula = () => {
    if (!selectedCell) return null;
    const row = rows.find(r => r.row_id === selectedCell.rowId);
    if (!row || !row.formula_data) return null;
    return row.formula_data[selectedCell.columnName] || null;
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Formula Bar */}
      {!readOnly && (
        <FormulaBar
          selectedCell={selectedCell}
          formula={getSelectedCellFormula()}
          onSave={handleFormulaSave}
          onCancel={() => setSelectedCell(null)}
        />
      )}
      
      {/* Toolbar */}
      <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            {totalRows} rows total
          </Typography>
          {selectedRows.size > 0 && (
            <Typography variant="body2" color="primary">
              ({selectedRows.size} selected)
            </Typography>
          )}
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          {!readOnly && (
            <>
              <Button
                variant="outlined"
                startIcon={<RefreshIcon />}
                onClick={handleRecalculate}
                size="small"
                disabled={recalculating}
              >
                {recalculating ? 'Recalculating...' : 'Recalculate'}
              </Button>
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={handleAddRow}
                size="small"
              >
                Add Row
              </Button>
            </>
          )}
        </Box>
      </Box>

      {/* Data Table */}
      <TableContainer component={Paper} sx={{ flexGrow: 1, overflow: 'auto' }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              {!readOnly && (
                <TableCell padding="checkbox">
                  <Checkbox
                    indeterminate={selectedRows.size > 0 && selectedRows.size < rows.length}
                    checked={rows.length > 0 && selectedRows.size === rows.length}
                    onChange={handleSelectAll}
                  />
                </TableCell>
              )}
              {schema.columns.map((column) => (
                <TableCell 
                  key={column.name}
                  sx={{ 
                    fontWeight: 600,
                    backgroundColor: column.color ? `${column.color}20` : 'inherit',
                    borderBottom: column.color ? `2px solid ${column.color}` : undefined,
                    cursor: 'pointer',
                    userSelect: 'none',
                    '&:hover': {
                      backgroundColor: column.color ? `${column.color}30` : 'action.hover'
                    }
                  }}
                  onClick={() => handleColumnSort(column.name)}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <span>
                      {column.name}
                      {!column.nullable && <span style={{ color: 'red' }}> *</span>}
                    </span>
                    {sortColumn === column.name && (
                      sortDirection === 'asc' 
                        ? <ArrowUpwardIcon fontSize="small" /> 
                        : <ArrowDownwardIcon fontSize="small" />
                    )}
                  </Box>
                </TableCell>
              ))}
              {!readOnly && <TableCell align="center">Actions</TableCell>}
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((row) => (
              <TableRow 
                key={row.row_id} 
                hover={!editingRow}
                selected={selectedRows.has(row.row_id)}
              >
                {!readOnly && (
                  <TableCell padding="checkbox">
                    <Checkbox
                      checked={selectedRows.has(row.row_id)}
                      onChange={() => handleSelectRow(row.row_id)}
                    />
                  </TableCell>
                )}
                {schema.columns.map((column) => (
                  <TableCell 
                    key={column.name}
                    sx={{ 
                      backgroundColor: column.color ? `${column.color}10` : 'inherit'
                    }}
                  >
                    {renderCell(row, column)}
                  </TableCell>
                ))}
                {!readOnly && (
                  <TableCell align="center">
                    {editingRow === row.row_id ? (
                      <Box sx={{ display: 'flex', gap: 0.5, justifyContent: 'center' }}>
                        <Tooltip title="Save">
                          <IconButton
                            size="small"
                            color="primary"
                            onClick={() => handleSaveEdit(row.row_id)}
                          >
                            <SaveIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Cancel">
                          <IconButton
                            size="small"
                            onClick={handleCancelEdit}
                          >
                            <CancelIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    ) : (
                      <Box sx={{ display: 'flex', gap: 0.5, justifyContent: 'center' }}>
                        <Tooltip title="Edit">
                          <IconButton
                            size="small"
                            onClick={() => handleStartEdit(row)}
                          >
                            <EditIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete">
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => {
                              setRowToDelete(row.row_id);
                              setDeleteDialogOpen(true);
                            }}
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    )}
                  </TableCell>
                )}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Pagination */}
      {totalRows > rowsPerPage && (
        <Box sx={{ p: 2, display: 'flex', justifyContent: 'center', borderTop: 1, borderColor: 'divider' }}>
          <Pagination
            count={Math.ceil(totalRows / rowsPerPage)}
            page={page}
            onChange={(e, value) => setPage(value)}
            color="primary"
          />
        </Box>
      )}

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Row?</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this row? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={() => handleDeleteRow(rowToDelete)} 
            color="error"
            variant="contained"
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DataTableView;


