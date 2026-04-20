import React, { useState } from 'react';
import dataWorkspaceService from '../../services/dataWorkspaceService';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Select,
  MenuItem,
  FormControl,
  Button,
  IconButton,
  Checkbox,
  FormControlLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tooltip
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  ArrowUpward as ArrowUpIcon,
  ArrowDownward as ArrowDownIcon,
  Palette as PaletteIcon
} from '@mui/icons-material';

const COLUMN_TYPES = [
  { value: 'TEXT', label: 'Text', icon: '📝' },
  { value: 'INTEGER', label: 'Number (Integer)', icon: '🔢' },
  { value: 'REAL', label: 'Number (Decimal)', icon: '💯' },
  { value: 'BOOLEAN', label: 'True/False', icon: '✓' },
  { value: 'TIMESTAMP', label: 'Date & Time', icon: '📅' },
  { value: 'DATE', label: 'Date', icon: '📆' },
  { value: 'JSON', label: 'JSON Data', icon: '📋' },
  { value: 'REFERENCE', label: 'Link to row (reference)', icon: '🔗' }
];

const PRESET_COLORS = [
  '#F44336', '#E91E63', '#9C27B0', '#673AB7',
  '#3F51B5', '#2196F3', '#03A9F4', '#00BCD4',
  '#009688', '#4CAF50', '#8BC34A', '#CDDC39',
  '#FFEB3B', '#FFC107', '#FF9800', '#FF5722'
];

const ColumnSchemaEditor = ({
  initialColumns = [],
  onChange,
  readOnly = false,
  dialogContainer,
  databaseId = null,
  peerTables = [],
  currentTableId = null
}) => {
  const [columns, setColumns] = useState(initialColumns.length > 0 ? initialColumns : [
    { name: 'id', type: 'INTEGER', nullable: false, isPrimaryKey: true, defaultValue: '', color: '', description: '' }
  ]);
  const [refColumnNamesByTable, setRefColumnNamesByTable] = useState({});
  const [editingColumn, setEditingColumn] = useState(null);
  const [colorPickerOpen, setColorPickerOpen] = useState(false);
  const [selectedColumnIndex, setSelectedColumnIndex] = useState(null);

  const handleAddColumn = () => {
    const newColumn = {
      name: `column_${columns.length + 1}`,
      type: 'TEXT',
      nullable: true,
      isPrimaryKey: false,
      defaultValue: '',
      color: '',
      description: ''
    };
    const updatedColumns = [...columns, newColumn];
    setColumns(updatedColumns);
    if (onChange) onChange(updatedColumns);
  };

  const handleDeleteColumn = (index) => {
    const updatedColumns = columns.filter((_, i) => i !== index);
    setColumns(updatedColumns);
    if (onChange) onChange(updatedColumns);
  };

  const handleMoveUp = (index) => {
    if (index === 0) return;
    const updatedColumns = [...columns];
    [updatedColumns[index - 1], updatedColumns[index]] = [updatedColumns[index], updatedColumns[index - 1]];
    setColumns(updatedColumns);
    if (onChange) onChange(updatedColumns);
  };

  const handleMoveDown = (index) => {
    if (index === columns.length - 1) return;
    const updatedColumns = [...columns];
    [updatedColumns[index], updatedColumns[index + 1]] = [updatedColumns[index + 1], updatedColumns[index]];
    setColumns(updatedColumns);
    if (onChange) onChange(updatedColumns);
  };

  const handleUpdateColumn = (index, field, value) => {
    const updatedColumns = [...columns];
    updatedColumns[index] = { ...updatedColumns[index], [field]: value };
    setColumns(updatedColumns);
    if (onChange) onChange(updatedColumns);
  };

  const ensureRefColumnNames = async (tableId) => {
    if (!tableId || refColumnNamesByTable[tableId]) return;
    try {
      const t = await dataWorkspaceService.getTable(tableId);
      const raw = t.table_schema_json;
      const sch = typeof raw === 'string' ? JSON.parse(raw || '{}') : raw || {};
      const names = (sch.columns || []).map((c) => c.name).filter(Boolean);
      setRefColumnNamesByTable((prev) => ({ ...prev, [tableId]: names }));
    } catch {
      setRefColumnNamesByTable((prev) => ({ ...prev, [tableId]: [] }));
    }
  };

  const handleColumnTypeChange = (index, newType) => {
    const updatedColumns = [...columns];
    const cur = { ...updatedColumns[index], type: newType };
    if (newType === 'REFERENCE') {
      cur.ref = {
        target_table_id: cur.ref?.target_table_id || '',
        target_key: cur.ref?.target_key || 'row_id',
        label_field: cur.ref?.label_field || 'name'
      };
    } else {
      delete cur.ref;
    }
    updatedColumns[index] = cur;
    setColumns(updatedColumns);
    if (onChange) onChange(updatedColumns);
    if (newType === 'REFERENCE' && cur.ref?.target_table_id) {
      void ensureRefColumnNames(cur.ref.target_table_id);
    }
  };

  const handleRefTargetChange = async (index, tableId) => {
    const updatedColumns = [...columns];
    const prev = updatedColumns[index].ref || {};
    updatedColumns[index] = {
      ...updatedColumns[index],
      ref: {
        ...prev,
        target_table_id: tableId,
        target_key: 'row_id',
        label_field: prev.label_field || 'name'
      }
    };
    setColumns(updatedColumns);
    if (onChange) onChange(updatedColumns);
    if (tableId) await ensureRefColumnNames(tableId);
  };

  const handleRefLabelFieldChange = (index, labelField) => {
    const updatedColumns = [...columns];
    const prev = updatedColumns[index].ref || {};
    updatedColumns[index] = {
      ...updatedColumns[index],
      ref: { ...prev, label_field: labelField }
    };
    setColumns(updatedColumns);
    if (onChange) onChange(updatedColumns);
  };

  const handleOpenColorPicker = (index) => {
    setSelectedColumnIndex(index);
    setColorPickerOpen(true);
  };

  const handleSelectColor = (color) => {
    if (selectedColumnIndex !== null) {
      handleUpdateColumn(selectedColumnIndex, 'color', color);
    }
    setColorPickerOpen(false);
    setSelectedColumnIndex(null);
  };

  const getTypeIcon = (type) => {
    const typeInfo = COLUMN_TYPES.find(t => t.value === type);
    return typeInfo ? typeInfo.icon : '📝';
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1, gap: 1 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'text.secondary', fontSize: '0.75rem' }}>
          Columns
        </Typography>
        {!readOnly && (
          <Button
            variant="outlined"
            startIcon={<AddIcon sx={{ fontSize: 16 }} />}
            onClick={handleAddColumn}
            size="small"
            sx={{ fontSize: '0.75rem', py: 0.25, minHeight: 28 }}
          >
            Add column
          </Button>
        )}
      </Box>

      <TableContainer
        component={Paper}
        variant="outlined"
        sx={{ maxHeight: 320, '& .MuiTableCell-root': { py: 0.35, px: 0.75 } }}
      >
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell width={52} sx={{ fontSize: '0.65rem', fontWeight: 600, py: 0.5 }}>
                Order
              </TableCell>
              <TableCell sx={{ fontSize: '0.65rem', fontWeight: 600, minWidth: 100 }}>Name</TableCell>
              <TableCell sx={{ fontSize: '0.65rem', fontWeight: 600, minWidth: 120 }}>Type</TableCell>
              <TableCell sx={{ fontSize: '0.65rem', fontWeight: 600, minWidth: 140 }}>Description</TableCell>
              <TableCell align="center" sx={{ fontSize: '0.65rem', fontWeight: 600, width: 56, px: 0.5 }}>
                Null
              </TableCell>
              <TableCell align="center" sx={{ fontSize: '0.65rem', fontWeight: 600, width: 44, px: 0.5 }}>
                PK
              </TableCell>
              <TableCell sx={{ fontSize: '0.65rem', fontWeight: 600, minWidth: 88 }}>Default</TableCell>
              <TableCell align="center" sx={{ fontSize: '0.65rem', fontWeight: 600, width: 44, px: 0.5 }}>
                Color
              </TableCell>
              {!readOnly && (
                <TableCell align="center" sx={{ fontSize: '0.65rem', fontWeight: 600, width: 44, px: 0.5 }}>
                  Del
                </TableCell>
              )}
            </TableRow>
          </TableHead>
          <TableBody>
            {columns.map((column, index) => (
              <TableRow key={index} hover>
                <TableCell sx={{ verticalAlign: 'middle' }}>
                  {!readOnly && (
                    <Box sx={{ display: 'flex', flexDirection: 'row', alignItems: 'center', gap: 0 }}>
                      <IconButton
                        size="small"
                        sx={{ p: 0.25 }}
                        onClick={() => handleMoveUp(index)}
                        disabled={index === 0}
                        aria-label="Move column up"
                      >
                        <ArrowUpIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                      <IconButton
                        size="small"
                        sx={{ p: 0.25 }}
                        onClick={() => handleMoveDown(index)}
                        disabled={index === columns.length - 1}
                        aria-label="Move column down"
                      >
                        <ArrowDownIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Box>
                  )}
                </TableCell>
                <TableCell>
                  {readOnly ? (
                    <Typography variant="caption" sx={{ fontSize: '0.75rem' }}>{column.name}</Typography>
                  ) : (
                    <TextField
                      value={column.name}
                      onChange={(e) => handleUpdateColumn(index, 'name', e.target.value)}
                      size="small"
                      fullWidth
                      placeholder="column_name"
                      sx={{ '& .MuiInputBase-input': { fontSize: '0.75rem', py: 0.5 } }}
                    />
                  )}
                </TableCell>
                <TableCell>
                  {readOnly ? (
                    <Chip
                      label={`${getTypeIcon(column.type)} ${column.type}`}
                      size="small"
                      sx={{ height: 22, '& .MuiChip-label': { fontSize: '0.7rem', px: 0.75 } }}
                    />
                  ) : (
                    <FormControl size="small" fullWidth>
                      <Select
                        value={column.type}
                        onChange={(e) => handleColumnTypeChange(index, e.target.value)}
                        sx={{ fontSize: '0.75rem', '& .MuiSelect-select': { py: 0.5 } }}
                      >
                        {COLUMN_TYPES.map((type) => (
                          <MenuItem key={type.value} value={type.value} sx={{ fontSize: '0.8rem', py: 0.5 }}>
                            {type.icon} {type.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  )}
                </TableCell>
                <TableCell>
                  {readOnly ? (
                    column.type === 'REFERENCE' && column.ref?.target_table_id ? (
                      <Typography variant="caption" sx={{ fontSize: '0.7rem' }} noWrap>
                        → {column.ref.target_table_id.slice(0, 8)}… ({column.ref.label_field || 'name'})
                      </Typography>
                    ) : (
                      <Typography variant="caption" sx={{ maxWidth: 200, fontSize: '0.75rem' }} noWrap title={column.description || ''}>
                        {column.description || '—'}
                      </Typography>
                    )
                  ) : column.type === 'REFERENCE' && databaseId ? (
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, minWidth: 140 }}>
                      <FormControl size="small" fullWidth>
                        <Select
                          displayEmpty
                          value={column.ref?.target_table_id || ''}
                          onOpen={() => {
                            if (column.ref?.target_table_id) void ensureRefColumnNames(column.ref.target_table_id);
                          }}
                          onChange={(e) => handleRefTargetChange(index, e.target.value)}
                          sx={{ fontSize: '0.7rem', '& .MuiSelect-select': { py: 0.35 } }}
                        >
                          <MenuItem value="">
                            <em>Target table</em>
                          </MenuItem>
                          {(peerTables || [])
                            .filter((t) => !currentTableId || t.table_id !== currentTableId)
                            .map((t) => (
                              <MenuItem key={t.table_id} value={t.table_id} sx={{ fontSize: '0.75rem' }}>
                                {t.name || t.table_id}
                              </MenuItem>
                            ))}
                        </Select>
                      </FormControl>
                      <FormControl size="small" fullWidth disabled={!column.ref?.target_table_id}>
                        <Select
                          displayEmpty
                          value={column.ref?.label_field || 'name'}
                          onChange={(e) => handleRefLabelFieldChange(index, e.target.value)}
                          sx={{ fontSize: '0.7rem', '& .MuiSelect-select': { py: 0.35 } }}
                        >
                          {(refColumnNamesByTable[column.ref?.target_table_id] || ['name']).map((n) => (
                            <MenuItem key={n} value={n} sx={{ fontSize: '0.75rem' }}>
                              {n}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      <TextField
                        value={column.description || ''}
                        onChange={(e) => handleUpdateColumn(index, 'description', e.target.value)}
                        size="small"
                        fullWidth
                        placeholder="Description"
                        sx={{ '& .MuiInputBase-input': { fontSize: '0.7rem', py: 0.35 } }}
                      />
                    </Box>
                  ) : (
                    <TextField
                      value={column.description || ''}
                      onChange={(e) => handleUpdateColumn(index, 'description', e.target.value)}
                      size="small"
                      fullWidth
                      placeholder="Optional"
                      sx={{ '& .MuiInputBase-input': { fontSize: '0.75rem', py: 0.5 } }}
                    />
                  )}
                </TableCell>
                <TableCell align="center" sx={{ px: 0.25 }}>
                  {readOnly ? (
                    <Typography variant="caption" sx={{ fontSize: '0.75rem' }}>{column.nullable ? 'Y' : '—'}</Typography>
                  ) : (
                    <Checkbox
                      checked={column.nullable}
                      onChange={(e) => handleUpdateColumn(index, 'nullable', e.target.checked)}
                      size="small"
                      sx={{ p: 0.25 }}
                    />
                  )}
                </TableCell>
                <TableCell align="center" sx={{ px: 0.25 }}>
                  {readOnly ? (
                    <Typography variant="caption" sx={{ fontSize: '0.7rem', fontWeight: column.isPrimaryKey ? 600 : 400, color: column.isPrimaryKey ? 'primary.main' : 'text.disabled' }}>
                      {column.isPrimaryKey ? 'Y' : '—'}
                    </Typography>
                  ) : (
                    <Checkbox
                      checked={column.isPrimaryKey}
                      onChange={(e) => handleUpdateColumn(index, 'isPrimaryKey', e.target.checked)}
                      size="small"
                      sx={{ p: 0.25 }}
                    />
                  )}
                </TableCell>
                <TableCell>
                  {readOnly ? (
                    <Typography variant="caption" sx={{ fontSize: '0.75rem' }}>{column.defaultValue || '—'}</Typography>
                  ) : (
                    <TextField
                      value={column.defaultValue}
                      onChange={(e) => handleUpdateColumn(index, 'defaultValue', e.target.value)}
                      size="small"
                      fullWidth
                      placeholder="—"
                      sx={{ '& .MuiInputBase-input': { fontSize: '0.75rem', py: 0.5 } }}
                    />
                  )}
                </TableCell>
                <TableCell align="center" sx={{ px: 0.25 }}>
                  {readOnly ? (
                    column.color && (
                      <Box
                        sx={{
                          width: 18,
                          height: 18,
                          backgroundColor: column.color,
                          borderRadius: 0.5,
                          border: '1px solid',
                          borderColor: 'divider',
                          margin: '0 auto'
                        }}
                      />
                    )
                  ) : (
                    <Tooltip title="Column color">
                      <IconButton
                        size="small"
                        sx={{ p: 0.25 }}
                        onClick={() => handleOpenColorPicker(index)}
                      >
                        {column.color ? (
                          <Box
                            sx={{
                              width: 18,
                              height: 18,
                              backgroundColor: column.color,
                              borderRadius: 0.5,
                              border: '1px solid',
                              borderColor: 'divider'
                            }}
                          />
                        ) : (
                          <PaletteIcon sx={{ fontSize: 18 }} />
                        )}
                      </IconButton>
                    </Tooltip>
                  )}
                </TableCell>
                {!readOnly && (
                  <TableCell align="center" sx={{ px: 0.25 }}>
                    <Tooltip title="Remove column">
                      <IconButton
                        size="small"
                        sx={{ p: 0.25 }}
                        onClick={() => handleDeleteColumn(index)}
                        color="error"
                        disabled={column.isPrimaryKey}
                        aria-label="Remove column"
                      >
                        <DeleteIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                )}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {columns.length === 0 && (
        <Box sx={{ py: 1, textAlign: 'center' }}>
          <Typography variant="caption" color="text.secondary">
            No columns — use Add.
          </Typography>
        </Box>
      )}

      {/* Color Picker Dialog */}
      <Dialog open={colorPickerOpen} onClose={() => setColorPickerOpen(false)} maxWidth="xs" fullWidth container={dialogContainer}>
        <DialogTitle sx={{ py: 1, px: 2, fontSize: '0.9rem', fontWeight: 600 }}>Column color</DialogTitle>
        <DialogContent sx={{ pt: 0, px: 2, pb: 1 }}>
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 0.75, mb: 1 }}>
            {PRESET_COLORS.map((color) => (
              <Box
                key={color}
                onClick={() => handleSelectColor(color)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleSelectColor(color); }}
                sx={{
                  width: '100%',
                  paddingTop: '100%',
                  backgroundColor: color,
                  borderRadius: 0.75,
                  cursor: 'pointer',
                  border: '2px solid transparent',
                  position: 'relative',
                  '&:hover': {
                    borderColor: 'background.paper',
                    boxShadow: 1
                  }
                }}
              />
            ))}
          </Box>
          <Button
            fullWidth
            size="small"
            variant="outlined"
            onClick={() => handleSelectColor('')}
          >
            Clear
          </Button>
        </DialogContent>
        <DialogActions sx={{ px: 2, py: 0.5 }}>
          <Button size="small" onClick={() => setColorPickerOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ColumnSchemaEditor;









