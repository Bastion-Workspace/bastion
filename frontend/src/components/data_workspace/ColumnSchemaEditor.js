import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
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
  { value: 'JSON', label: 'JSON Data', icon: '📋' }
];

const PRESET_COLORS = [
  '#F44336', '#E91E63', '#9C27B0', '#673AB7',
  '#3F51B5', '#2196F3', '#03A9F4', '#00BCD4',
  '#009688', '#4CAF50', '#8BC34A', '#CDDC39',
  '#FFEB3B', '#FFC107', '#FF9800', '#FF5722'
];

const ColumnSchemaEditor = ({ initialColumns = [], onChange, readOnly = false }) => {
  const [columns, setColumns] = useState(initialColumns.length > 0 ? initialColumns : [
    { name: 'id', type: 'INTEGER', nullable: false, isPrimaryKey: true, defaultValue: '', color: '', description: '' }
  ]);
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
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Column Schema</Typography>
        {!readOnly && (
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleAddColumn}
            size="small"
          >
            Add Column
          </Button>
        )}
      </Box>

      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell width="40px">Order</TableCell>
              <TableCell>Column Name</TableCell>
              <TableCell>Type</TableCell>
              <TableCell align="center">Nullable</TableCell>
              <TableCell align="center">Primary Key</TableCell>
              <TableCell>Default Value</TableCell>
              <TableCell align="center">Color</TableCell>
              {!readOnly && <TableCell align="center">Actions</TableCell>}
            </TableRow>
          </TableHead>
          <TableBody>
            {columns.map((column, index) => (
              <TableRow key={index} hover>
                <TableCell>
                  {!readOnly && (
                    <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                      <IconButton
                        size="small"
                        onClick={() => handleMoveUp(index)}
                        disabled={index === 0}
                      >
                        <ArrowUpIcon fontSize="small" />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={() => handleMoveDown(index)}
                        disabled={index === columns.length - 1}
                      >
                        <ArrowDownIcon fontSize="small" />
                      </IconButton>
                    </Box>
                  )}
                </TableCell>
                <TableCell>
                  {readOnly ? (
                    <Typography variant="body2">{column.name}</Typography>
                  ) : (
                    <TextField
                      value={column.name}
                      onChange={(e) => handleUpdateColumn(index, 'name', e.target.value)}
                      size="small"
                      fullWidth
                      placeholder="column_name"
                    />
                  )}
                </TableCell>
                <TableCell>
                  {readOnly ? (
                    <Chip
                      label={`${getTypeIcon(column.type)} ${column.type}`}
                      size="small"
                    />
                  ) : (
                    <FormControl size="small" fullWidth>
                      <Select
                        value={column.type}
                        onChange={(e) => handleUpdateColumn(index, 'type', e.target.value)}
                      >
                        {COLUMN_TYPES.map((type) => (
                          <MenuItem key={type.value} value={type.value}>
                            {type.icon} {type.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  )}
                </TableCell>
                <TableCell align="center">
                  {readOnly ? (
                    <Typography variant="body2">{column.nullable ? '✓' : '✗'}</Typography>
                  ) : (
                    <Checkbox
                      checked={column.nullable}
                      onChange={(e) => handleUpdateColumn(index, 'nullable', e.target.checked)}
                      size="small"
                    />
                  )}
                </TableCell>
                <TableCell align="center">
                  {readOnly ? (
                    <Typography variant="body2">{column.isPrimaryKey ? '🔑' : ''}</Typography>
                  ) : (
                    <Checkbox
                      checked={column.isPrimaryKey}
                      onChange={(e) => handleUpdateColumn(index, 'isPrimaryKey', e.target.checked)}
                      size="small"
                    />
                  )}
                </TableCell>
                <TableCell>
                  {readOnly ? (
                    <Typography variant="body2">{column.defaultValue || '-'}</Typography>
                  ) : (
                    <TextField
                      value={column.defaultValue}
                      onChange={(e) => handleUpdateColumn(index, 'defaultValue', e.target.value)}
                      size="small"
                      fullWidth
                      placeholder="default value"
                    />
                  )}
                </TableCell>
                <TableCell align="center">
                  {readOnly ? (
                    column.color && (
                      <Box
                        sx={{
                          width: 24,
                          height: 24,
                          backgroundColor: column.color,
                          borderRadius: 1,
                          border: '1px solid rgba(0,0,0,0.2)',
                          margin: '0 auto'
                        }}
                      />
                    )
                  ) : (
                    <Tooltip title="Set column color">
                      <IconButton
                        size="small"
                        onClick={() => handleOpenColorPicker(index)}
                      >
                        {column.color ? (
                          <Box
                            sx={{
                              width: 24,
                              height: 24,
                              backgroundColor: column.color,
                              borderRadius: 1,
                              border: '1px solid rgba(0,0,0,0.2)'
                            }}
                          />
                        ) : (
                          <PaletteIcon fontSize="small" />
                        )}
                      </IconButton>
                    </Tooltip>
                  )}
                </TableCell>
                {!readOnly && (
                  <TableCell align="center">
                    <Tooltip title="Delete column">
                      <IconButton
                        size="small"
                        onClick={() => handleDeleteColumn(index)}
                        color="error"
                        disabled={column.isPrimaryKey}
                      >
                        <DeleteIcon fontSize="small" />
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
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            No columns defined. Click "Add Column" to get started.
          </Typography>
        </Box>
      )}

      {/* Color Picker Dialog */}
      <Dialog open={colorPickerOpen} onClose={() => setColorPickerOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>Select Column Color</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 1, mb: 2 }}>
            {PRESET_COLORS.map((color) => (
              <Box
                key={color}
                onClick={() => handleSelectColor(color)}
                sx={{
                  width: '100%',
                  paddingTop: '100%',
                  backgroundColor: color,
                  borderRadius: 1,
                  cursor: 'pointer',
                  border: '2px solid transparent',
                  position: 'relative',
                  '&:hover': {
                    border: '2px solid white',
                    boxShadow: 2
                  }
                }}
              />
            ))}
          </Box>
          <Button
            fullWidth
            variant="outlined"
            onClick={() => handleSelectColor('')}
          >
            Clear Color
          </Button>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setColorPickerOpen(false)}>Cancel</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ColumnSchemaEditor;





