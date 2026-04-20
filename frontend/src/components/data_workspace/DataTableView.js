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
  InputBase,
  Button,
  Chip,
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
  Refresh as RefreshIcon,
  TableRows as TableRowsIcon,
  GridOn as GridOnIcon,
  OpenInNew as OpenInNewIcon
} from '@mui/icons-material';

import dataWorkspaceService from '../../services/dataWorkspaceService';
import FormulaBar from './FormulaBar';
import LinkPickerDialog from './LinkPickerDialog';
import { parseRefFromCell, formatRefLabel } from './referenceLinkUtils';

const DataTableView = ({
  tableId,
  databaseId = null,
  schema,
  onDataChange,
  onRowsLoaded,
  onEditTableSchema,
  onOpenLinkedRow,
  modalContainer,
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
  /** Edit every cell on the current page only (matches loaded `rows`; tab order follows the grid). */
  const [quickPageEdit, setQuickPageEdit] = useState(false);
  /** Local overrides while quick-editing: { [rowId]: { [columnName]: value } } */
  const [pageEditDrafts, setPageEditDrafts] = useState({});
  const [linkPicker, setLinkPicker] = useState(null);

  useEffect(() => {
    if (tableId) {
      loadData();
    }
  }, [tableId, databaseId, page]);

  useEffect(() => {
    setQuickPageEdit(false);
    setPageEditDrafts({});
  }, [page, sortColumn, sortDirection]);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const offset = (page - 1) * rowsPerPage;
      const response = await dataWorkspaceService.getTableData(tableId, offset, rowsPerPage, databaseId);
      
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
      if (typeof onRowsLoaded === 'function') {
        onRowsLoaded(loadedRows, response.total_rows || 0);
      }
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
    const colMeta = schema.columns.find((c) => c.name === columnName);
    return [...rowsToSort].sort((a, b) => {
      let aVal = a.row_data[columnName];
      let bVal = b.row_data[columnName];

      if (colMeta?.type === 'REFERENCE') {
        aVal = formatRefLabel(aVal);
        bVal = formatRefLabel(bVal);
      }

      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;

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
    if (quickPageEdit) return;
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
    if (readOnly || editingRow || quickPageEdit) return;

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
        const def = col.default_value ?? col.defaultValue;
        if (def !== undefined && def !== null && String(def).trim() !== '') {
          acc[col.name] = def;
        } else if (col.type === 'INTEGER' || col.type === 'REAL') acc[col.name] = 0;
        else if (col.type === 'BOOLEAN') acc[col.name] = false;
        else if (col.type === 'TIMESTAMP') acc[col.name] = new Date().toISOString();
        else if (col.type === 'REFERENCE') {
          /* no default link */
        } else acc[col.name] = '';
        return acc;
      }, {});

      const response = await dataWorkspaceService.insertTableRow(tableId, newRowData);

      const newRow = response || {
        row_id: `row_new_${Date.now()}`,
        row_data: newRowData
      };

      setRows([...rows, { ...newRow, formula_data: newRow.formula_data || {} }]);
      setTotalRows((n) => n + 1);

      if (quickPageEdit) {
        setEditingRow(null);
        setEditingData({});
        setEditingCell(null);
      } else {
        setEditingRow(newRow.row_id);
        setEditingData(newRow.row_data || newRowData);
      }

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

  /** Flat, grid-integrated editor — avoids outlined “bubble” inputs in cells */
  const gridCellInputSx = {
    width: '100%',
    fontSize: '0.8125rem',
    lineHeight: 1.4,
    px: 0.75,
    py: 0.5,
    minHeight: 30,
    boxSizing: 'border-box',
    borderRadius: 0,
    border: '1px solid',
    borderColor: 'divider',
    bgcolor: (theme) =>
      theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.02)',
    transition: 'border-color 0.12s ease, box-shadow 0.12s ease, background-color 0.12s ease',
    '&:hover': {
      borderColor: 'action.active',
      bgcolor: 'action.hover'
    },
    '&.Mui-focused': {
      borderColor: 'primary.main',
      boxShadow: (theme) => `inset 0 0 0 1px ${theme.palette.primary.main}`,
      bgcolor: 'background.paper'
    }
  };

  const handleBooleanEditChange = async (row, columnName, checked) => {
    if (quickPageEdit) {
      await commitQuickCell(row, columnName, checked);
      return;
    }
    if (editingCell?.rowId === row.row_id && editingCell?.columnName === columnName) {
      try {
        setError(null);
        await dataWorkspaceService.updateTableCell(tableId, row.row_id, columnName, checked, null);
        await loadData();
        setEditingCell(null);
        setEditingData({});
        if (onDataChange) onDataChange();
      } catch (err) {
        console.error('Failed to save cell:', err);
        setError(err.message || 'Failed to save cell');
      }
      return;
    }
    handleCellChange(columnName, checked);
  };

  const clearDraftCell = (rowId, columnName) => {
    setPageEditDrafts((prev) => {
      const rowDraft = prev[rowId] ? { ...prev[rowId] } : null;
      if (!rowDraft) return prev;
      delete rowDraft[columnName];
      const next = { ...prev };
      if (Object.keys(rowDraft).length === 0) {
        delete next[rowId];
      } else {
        next[rowId] = rowDraft;
      }
      return next;
    });
  };

  const handleQuickDraftChange = (rowId, columnName, value) => {
    setPageEditDrafts((prev) => ({
      ...prev,
      [rowId]: { ...prev[rowId], [columnName]: value }
    }));
  };

  const valuesMatchPersisted = (row, columnName, raw) => {
    if (typeof raw === 'boolean') {
      return Boolean(row.row_data[columnName]) === raw;
    }
    const str = raw == null ? '' : String(raw);
    const trimmed = str.trim();
    const prevFormula = getCellFormula(row, columnName);
    if (trimmed.startsWith('=')) {
      return String(prevFormula || '').trim() === trimmed;
    }
    if (prevFormula != null && String(prevFormula).length > 0) {
      return false;
    }
    const prev = row.row_data[columnName];
    if (prev === null || prev === undefined) {
      return trimmed === '';
    }
    const col = schema.columns.find((c) => c.name === columnName);
    if (col?.type === 'REFERENCE') {
      const enc = (x) => JSON.stringify(parseRefFromCell(x));
      return enc(prev) === enc(raw);
    }
    if (col && (col.type === 'INTEGER' || col.type === 'REAL')) {
      const n = Number(str);
      const p = Number(prev);
      return !Number.isNaN(n) && n === p;
    }
    if (typeof prev === 'boolean') {
      return (trimmed === 'true' || trimmed === '1') === prev;
    }
    return String(prev) === str;
  };

  const commitQuickCell = async (row, columnName, rawDisplay) => {
    try {
      setError(null);
      if (valuesMatchPersisted(row, columnName, rawDisplay)) {
        clearDraftCell(row.row_id, columnName);
        return;
      }
      const formulaValue =
        typeof rawDisplay === 'string' && isFormula(rawDisplay) ? rawDisplay : null;
      const actualValue =
        formulaValue != null
          ? null
          : typeof rawDisplay === 'boolean'
            ? rawDisplay
            : rawDisplay;

      await dataWorkspaceService.updateTableCell(
        tableId,
        row.row_id,
        columnName,
        actualValue,
        formulaValue
      );
      await loadData();
      clearDraftCell(row.row_id, columnName);
      if (onDataChange) onDataChange();
    } catch (err) {
      console.error('Failed to save cell:', err);
      setError(err.message || 'Failed to save cell');
    }
  };

  const focusQuickInput = (rowId, columnName) => {
    const id = `qe-${rowId}-${columnName}`.replace(/[^a-zA-Z0-9_-]/g, '_');
    requestAnimationFrame(() => {
      const el = document.getElementById(id);
      if (el && typeof el.focus === 'function') {
        el.focus();
        if (typeof el.select === 'function') {
          el.select();
        }
      }
    });
  };

  const toggleQuickPageEdit = () => {
    if (quickPageEdit) {
      const hasDrafts = Object.values(pageEditDrafts).some(
        (r) => r && Object.keys(r).length > 0
      );
      if (
        hasDrafts &&
        !window.confirm(
          'Some cells still have local edits that were not saved (tab away or press Enter to save each cell). Exit quick edit anyway?'
        )
      ) {
        return;
      }
      setQuickPageEdit(false);
      setPageEditDrafts({});
      return;
    }
    handleCancelEdit();
    handleCancelCellEdit();
    setPageEditDrafts({});
    setQuickPageEdit(true);
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
    const isEditingRowOnly = editingRow === row.row_id;
    const isEditingCellMode =
      editingCell?.rowId === row.row_id && editingCell?.columnName === column.name;
    const isLegacyEditing = isEditingRowOnly || isEditingCellMode;
    const isGridQuick = quickPageEdit && !readOnly;
    const isEditing = isLegacyEditing || isGridQuick;

    const cellFormula = getCellFormula(row, column.name);
    const hasFormula = cellFormula !== null;

    let value;
    if (isLegacyEditing) {
      value = editingData[column.name];
    } else if (isGridQuick) {
      const d = pageEditDrafts[row.row_id]?.[column.name];
      value = d !== undefined ? d : (cellFormula ?? row.row_data[column.name]);
    } else {
      value = row.row_data[column.name];
    }

    const quickInputId = `qe-${row.row_id}-${column.name}`.replace(/[^a-zA-Z0-9_-]/g, '_');

    if (column.type === 'REFERENCE') {
      const refValue = isLegacyEditing
        ? editingData[column.name]
        : isGridQuick
          ? pageEditDrafts[row.row_id]?.[column.name] !== undefined
            ? pageEditDrafts[row.row_id][column.name]
            : (cellFormula ?? row.row_data[column.name])
          : (cellFormula ?? row.row_data[column.name]);
      const label = formatRefLabel(refValue);
      const inner = parseRefFromCell(refValue);

      if (!isEditing) {
        return (
          <Box
            onClick={() => handleCellClick(row, column)}
            onDoubleClick={() => handleCellDoubleClick(row, column)}
            sx={{
              cursor: readOnly ? 'default' : 'pointer',
              minHeight: 28,
              display: 'flex',
              alignItems: 'center',
              gap: 0.25,
              px: 0.25,
              overflow: 'hidden',
              '&:hover': readOnly ? {} : { backgroundColor: 'action.hover' }
            }}
          >
            <Chip
              size="small"
              label={label || '—'}
              variant={inner ? 'filled' : 'outlined'}
              sx={{
                maxWidth: '72%',
                height: 22,
                '& .MuiChip-label': { px: 0.75, fontSize: '0.75rem' }
              }}
            />
            {!readOnly && column.ref?.target_table_id && (
              <Tooltip title="Change link">
                <IconButton
                  size="small"
                  sx={{ p: 0.25 }}
                  aria-label="Choose linked row"
                  onClick={(e) => {
                    e.stopPropagation();
                    setLinkPicker({ row, column });
                  }}
                >
                  <EditIcon sx={{ fontSize: 16 }} />
                </IconButton>
              </Tooltip>
            )}
            {!readOnly && inner?.table_id && typeof onOpenLinkedRow === 'function' && (
              <Tooltip title="Open linked table">
                <IconButton
                  size="small"
                  sx={{ p: 0.25 }}
                  aria-label="Open linked table"
                  onClick={(e) => {
                    e.stopPropagation();
                    onOpenLinkedRow({
                      targetTableId: inner.table_id,
                      targetRowId: inner.row_id
                    });
                  }}
                >
                  <OpenInNewIcon sx={{ fontSize: 16 }} />
                </IconButton>
              </Tooltip>
            )}
          </Box>
        );
      }

      const openPicker = () => {
        if (!column.ref?.target_table_id) {
          setError('Reference column has no target table. Edit table schema.');
          return;
        }
        setLinkPicker({ row, column });
      };

      const clearLink = async () => {
        try {
          setError(null);
          if (isGridQuick) {
            await commitQuickCell(row, column.name, null);
          } else {
            await dataWorkspaceService.updateTableCell(tableId, row.row_id, column.name, null, null);
            await loadData();
            if (editingCell?.rowId === row.row_id && editingCell?.columnName === column.name) {
              setEditingCell(null);
              setEditingData({});
            }
            if (editingRow === row.row_id) {
              setEditingData((prev) => ({ ...prev, [column.name]: null }));
            }
            if (onDataChange) onDataChange();
          }
        } catch (err) {
          setError(err.message || 'Failed to clear link');
        }
      };

      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', minHeight: 30 }}>
          <Chip size="small" label={label || 'None'} sx={{ height: 22 }} />
          <Button size="small" variant="outlined" onClick={openPicker} disabled={readOnly || !column.ref?.target_table_id}>
            Choose…
          </Button>
          {inner && !readOnly && (
            <Button size="small" color="warning" onClick={() => void clearLink()}>
              Clear
            </Button>
          )}
        </Box>
      );
    }

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
            minHeight: 28,
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
            px: 0.25,
            overflow: 'hidden',
            '&:hover': readOnly ? {} : { backgroundColor: 'action.hover' }
          }}
        >
          {hasFormula && (
            <FunctionsIcon fontSize="small" sx={{ color: 'primary.main', fontSize: 16 }} />
          )}
          <Typography
            variant="body2"
            noWrap
            title={String(displayValue)}
            sx={{
              color: column.color || 'inherit',
              fontWeight: column.color ? 600 : 400,
              fontSize: '0.8125rem'
            }}
          >
            {String(displayValue)}
          </Typography>
        </Box>
      );
    }

    const syncSelectedCell = () => {
      const columnIndex = schema.columns.findIndex((c) => c.name === column.name);
      setSelectedCell({
        rowId: row.row_id,
        columnName: column.name,
        rowIndex: row.row_index,
        columnIndex
      });
    };

    if (column.type === 'BOOLEAN') {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', minHeight: 30 }}>
          <Checkbox
            checked={Boolean(value)}
            onChange={(e) => handleBooleanEditChange(row, column.name, e.target.checked)}
            onFocus={() => isGridQuick && syncSelectedCell()}
            size="small"
            autoFocus={isEditingCellMode && !isGridQuick}
            tabIndex={isGridQuick ? 0 : undefined}
          />
        </Box>
      );
    }

    if (column.type === 'INTEGER' || column.type === 'REAL') {
      const isFormulaValue = isFormula(value);
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', minWidth: 0 }}>
          {isFormulaValue ? (
            <FunctionsIcon fontSize="inherit" sx={{ color: 'primary.main', mr: 0.5, flexShrink: 0, fontSize: 16 }} />
          ) : null}
          <InputBase
            value={value ?? ''}
            onChange={(e) =>
              isGridQuick
                ? handleQuickDraftChange(row.row_id, column.name, e.target.value)
                : handleCellChange(column.name, e.target.value)
            }
            onFocus={() => isGridQuick && syncSelectedCell()}
            onBlur={
              isGridQuick
                ? (e) => {
                    void commitQuickCell(row, column.name, e.target.value);
                  }
                : undefined
            }
            onKeyDown={(e) => {
              if (isGridQuick) {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  void (async () => {
                    await commitQuickCell(row, column.name, e.currentTarget.value);
                    const rIdx = rows.findIndex((r) => r.row_id === row.row_id);
                    if (rIdx >= 0 && rIdx < rows.length - 1) {
                      focusQuickInput(rows[rIdx + 1].row_id, column.name);
                    }
                  })();
                } else if (e.key === 'Escape') {
                  e.preventDefault();
                  clearDraftCell(row.row_id, column.name);
                }
                return;
              }
              if (isEditingRowOnly) {
                if (e.key === 'Enter') e.stopPropagation();
                return;
              }
              if (e.key === 'Enter') {
                handleSaveCellEdit(row.row_id, column.name);
              } else if (e.key === 'Escape') {
                handleCancelCellEdit();
              }
            }}
            fullWidth
            autoFocus={isEditingCellMode && !isGridQuick}
            placeholder={isFormulaValue ? '=formula' : 'Number or =formula'}
            inputProps={{
              id: isGridQuick ? quickInputId : undefined,
              'aria-label': column.name,
              spellCheck: false
            }}
            sx={gridCellInputSx}
          />
        </Box>
      );
    }

    return (
      <Box sx={{ display: 'flex', alignItems: 'center', minWidth: 0, minHeight: 30 }}>
        <InputBase
          value={value ?? ''}
          onChange={(e) =>
            isGridQuick
              ? handleQuickDraftChange(row.row_id, column.name, e.target.value)
              : handleCellChange(column.name, e.target.value)
          }
          onFocus={() => isGridQuick && syncSelectedCell()}
          onBlur={
            isGridQuick
              ? (e) => {
                  void commitQuickCell(row, column.name, e.target.value);
                }
              : undefined
          }
          onKeyDown={(e) => {
            if (isGridQuick) {
              if (e.key === 'Enter') {
                e.preventDefault();
                void (async () => {
                  await commitQuickCell(row, column.name, e.currentTarget.value);
                  const rIdx = rows.findIndex((r) => r.row_id === row.row_id);
                  if (rIdx >= 0 && rIdx < rows.length - 1) {
                    focusQuickInput(rows[rIdx + 1].row_id, column.name);
                  }
                })();
              } else if (e.key === 'Escape') {
                e.preventDefault();
                clearDraftCell(row.row_id, column.name);
              }
              return;
            }
            if (isEditingRowOnly) {
              if (e.key === 'Enter') e.stopPropagation();
              return;
            }
            if (e.key === 'Enter') {
              e.preventDefault();
              handleSaveCellEdit(row.row_id, column.name);
            } else if (e.key === 'Escape') {
              handleCancelCellEdit();
            }
          }}
          fullWidth
          multiline={false}
          autoFocus={isEditingCellMode && !isGridQuick}
          placeholder={column.name}
          inputProps={{
            id: isGridQuick ? quickInputId : undefined,
            'aria-label': column.name,
            maxLength: 8192,
            spellCheck: false
          }}
          sx={{
            ...gridCellInputSx,
            '& input': {
              textOverflow: 'ellipsis',
              overflow: 'hidden',
              whiteSpace: 'nowrap'
            },
            '&:focus-within input': {
              textOverflow: 'clip',
              overflowX: 'auto',
              whiteSpace: 'nowrap'
            }
          }}
        />
      </Box>
    );
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
      <LinkPickerDialog
        open={!!linkPicker}
        onClose={() => setLinkPicker(null)}
        databaseId={databaseId}
        targetTableId={linkPicker?.column?.ref?.target_table_id}
        targetTableName=""
        labelField={linkPicker?.column?.ref?.label_field || 'name'}
        container={modalContainer}
        onSelect={async (cellObj) => {
          const ctx = linkPicker;
          setLinkPicker(null);
          if (!ctx) return;
          const { row: r, column: col } = ctx;
          try {
            setError(null);
            if (quickPageEdit) {
              await commitQuickCell(r, col.name, cellObj);
              return;
            }
            if (editingCell?.rowId === r.row_id && editingCell?.columnName === col.name) {
              await dataWorkspaceService.updateTableCell(tableId, r.row_id, col.name, cellObj, null);
              await loadData();
              setEditingCell(null);
              setEditingData({});
              if (onDataChange) onDataChange();
              return;
            }
            if (editingRow === r.row_id) {
              handleCellChange(col.name, cellObj);
              return;
            }
            await dataWorkspaceService.updateTableCell(tableId, r.row_id, col.name, cellObj, null);
            await loadData();
            if (onDataChange) onDataChange();
          } catch (err) {
            setError(err.message || 'Failed to save link');
          }
        }}
      />
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
      <Box sx={{ px: 2, py: 1.25, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 1, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
          <Typography variant="body2" color="text.secondary">
            {totalRows === 1 ? '1 row total' : `${totalRows} rows total`}
          </Typography>
          {quickPageEdit && (
            <Typography variant="caption" color="primary">
              Quick edit: this page only ({rows.length} rows). Tab between cells; blur or Enter saves; Esc reverts drafts. Add Row appends here and stays in quick edit. Change page or sort exits.
            </Typography>
          )}
          {editingCell && !quickPageEdit && (
            <Typography variant="caption" color="primary">
              Cell edit: Enter save · Esc cancel
            </Typography>
          )}
          {selectedRows.size > 0 && (
            <Typography variant="body2" color="primary">
              ({selectedRows.size} selected)
            </Typography>
          )}
        </Box>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          {!readOnly && (
            <>
              {typeof onEditTableSchema === 'function' && (
                <Tooltip title="Add, remove, or reorder columns">
                  <Button
                    variant="outlined"
                    startIcon={<TableRowsIcon />}
                    onClick={onEditTableSchema}
                    size="small"
                  >
                    Table schema
                  </Button>
                </Tooltip>
              )}
              <Tooltip
                title={
                  quickPageEdit
                    ? 'Finish quick edit'
                    : `Edit every cell on this page (${rows.length} rows). Tab between cells; blur or Enter saves.`
                }
              >
                <Button
                  variant={quickPageEdit ? 'contained' : 'outlined'}
                  color={quickPageEdit ? 'secondary' : 'primary'}
                  startIcon={<GridOnIcon />}
                  onClick={toggleQuickPageEdit}
                  size="small"
                >
                  {quickPageEdit ? 'Done' : 'Quick edit page'}
                </Button>
              </Tooltip>
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
      <TableContainer component={Paper} variant="outlined" sx={{ flexGrow: 1, overflow: 'auto', borderRadius: 0 }}>
        <Table
          stickyHeader
          size="small"
          sx={{
            borderCollapse: 'separate',
            '& .MuiTableCell-root': {
              borderRight: 1,
              borderColor: 'divider',
              verticalAlign: 'middle'
            },
            '& .MuiTableCell-root:last-of-type': {
              borderRight: 0
            }
          }}
        >
          <TableHead>
            <TableRow>
              {!readOnly && (
                <TableCell padding="checkbox" sx={{ bgcolor: 'background.paper' }}>
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
                    fontSize: '0.75rem',
                    bgcolor: column.color ? `${column.color}18` : 'background.paper',
                    borderBottom: (theme) =>
                      column.color ? `2px solid ${column.color}` : `2px solid ${theme.palette.divider}`,
                    cursor: quickPageEdit ? 'not-allowed' : 'pointer',
                    userSelect: 'none',
                    opacity: quickPageEdit ? 0.65 : 1,
                    '&:hover': {
                      bgcolor: quickPageEdit
                        ? undefined
                        : column.color
                          ? `${column.color}28`
                          : 'action.hover'
                    }
                  }}
                  onClick={() => {
                    if (!quickPageEdit) handleColumnSort(column.name);
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <span>
                      {column.name}
                      {!column.nullable && <span style={{ color: 'red' }}> *</span>}
                      {(column.default_value ?? column.defaultValue) && (
                        <Typography component="span" variant="caption" display="block" color="text.secondary" sx={{ fontWeight: 400 }}>
                          default: {String(column.default_value ?? column.defaultValue)}
                        </Typography>
                      )}
                    </span>
                    {sortColumn === column.name && (
                      sortDirection === 'asc' 
                        ? <ArrowUpwardIcon fontSize="small" /> 
                        : <ArrowDownwardIcon fontSize="small" />
                    )}
                  </Box>
                </TableCell>
              ))}
              {!readOnly && (
                <TableCell
                  align="center"
                  sx={{
                    fontWeight: 600,
                    fontSize: '0.75rem',
                    width: 108,
                    bgcolor: 'background.paper',
                    borderBottom: (theme) => `2px solid ${theme.palette.divider}`
                  }}
                >
                  Actions
                </TableCell>
              )}
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((row) => (
              <TableRow
                key={row.row_id}
                hover={!editingRow && !quickPageEdit}
                selected={selectedRows.has(row.row_id)}
                sx={{
                  ...(editingRow === row.row_id
                    ? { bgcolor: 'action.selected' }
                    : null),
                  ...(quickPageEdit ? { bgcolor: 'action.hover' } : null)
                }}
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
                      py: 0.5,
                      px: 1,
                      maxWidth: 280,
                      bgcolor: column.color ? `${column.color}0d` : 'transparent'
                    }}
                  >
                    {renderCell(row, column)}
                  </TableCell>
                ))}
                {!readOnly && (
                  <TableCell align="center">
                    {quickPageEdit ? (
                      <Box sx={{ display: 'flex', gap: 0.5, justifyContent: 'center' }}>
                        <Tooltip title="Delete row">
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
                    ) : editingRow === row.row_id ? (
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
                        <Tooltip title="Edit row">
                          <IconButton
                            size="small"
                            onClick={() => handleStartEdit(row)}
                          >
                            <EditIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete row">
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
          {quickPageEdit ? (
            <Tooltip title="Click Done to exit quick edit, then change page">
              <span>
                <Pagination
                  count={Math.ceil(totalRows / rowsPerPage)}
                  page={page}
                  onChange={(e, value) => setPage(value)}
                  color="primary"
                  disabled
                />
              </span>
            </Tooltip>
          ) : (
            <Pagination
              count={Math.ceil(totalRows / rowsPerPage)}
              page={page}
              onChange={(e, value) => setPage(value)}
              color="primary"
            />
          )}
        </Box>
      )}

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)} container={modalContainer}>
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


