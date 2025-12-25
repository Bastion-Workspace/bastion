import React, { useState, useEffect } from 'react';
import {
  Box,
  TextField,
  IconButton,
  Typography,
  Paper
} from '@mui/material';
import {
  Check as CheckIcon,
  Close as CloseIcon,
  Functions as FunctionsIcon
} from '@mui/icons-material';

const FormulaBar = ({
  selectedCell,
  formula,
  onFormulaChange,
  onCancel,
  onSave
}) => {
  const [editFormula, setEditFormula] = useState(formula || '');
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    setEditFormula(formula || '');
    setIsEditing(false);
  }, [formula, selectedCell]);

  const handleFormulaInput = (e) => {
    const value = e.target.value;
    setEditFormula(value);
    setIsEditing(true);
  };

  const handleSave = () => {
    if (onSave) {
      onSave(editFormula);
    }
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditFormula(formula || '');
    setIsEditing(false);
    if (onCancel) {
      onCancel();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSave();
    } else if (e.key === 'Escape') {
      handleCancel();
    }
  };

  if (!selectedCell) {
    return (
      <Box sx={{ p: 1, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="body2" color="text.secondary">
          Select a cell to edit
        </Typography>
      </Box>
    );
  }

  const cellRef = `${String.fromCharCode(65 + selectedCell.columnIndex)}${selectedCell.rowIndex + 1}`;

  return (
    <Paper
      elevation={0}
      sx={{
        p: 1,
        borderBottom: 1,
        borderColor: 'divider',
        display: 'flex',
        alignItems: 'center',
        gap: 1
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 60 }}>
        <FunctionsIcon fontSize="small" color="action" />
        <Typography variant="body2" sx={{ fontWeight: 500, color: 'text.secondary' }}>
          {cellRef}
        </Typography>
      </Box>
      
      <TextField
        fullWidth
        size="small"
        value={editFormula}
        onChange={handleFormulaInput}
        onKeyDown={handleKeyDown}
        placeholder={formula ? formula : "Enter formula (e.g., =A1+B1)"}
        InputProps={{
          startAdornment: formula && !isEditing ? (
            <Typography variant="body2" sx={{ mr: 1, color: 'text.secondary' }}>
              =
            </Typography>
          ) : null
        }}
        autoFocus={isEditing}
      />
      
      {isEditing && (
        <Box sx={{ display: 'flex', gap: 0.5 }}>
          <IconButton
            size="small"
            color="primary"
            onClick={handleSave}
          >
            <CheckIcon fontSize="small" />
          </IconButton>
          <IconButton
            size="small"
            onClick={handleCancel}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
      )}
    </Paper>
  );
};

export default FormulaBar;

