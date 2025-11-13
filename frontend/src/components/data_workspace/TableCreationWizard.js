import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Stepper,
  Step,
  StepLabel,
  Box,
  Typography,
  Alert,
  CircularProgress
} from '@mui/material';
import ColumnSchemaEditor from './ColumnSchemaEditor';
import dataWorkspaceService from '../../services/dataWorkspaceService';

const TableCreationWizard = ({ open, onClose, databaseId, onTableCreated }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [tableName, setTableName] = useState('');
  const [tableDescription, setTableDescription] = useState('');
  const [columns, setColumns] = useState([
    { name: 'id', type: 'INTEGER', nullable: false, isPrimaryKey: true, defaultValue: '', color: '', description: '' }
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const steps = ['Table Details', 'Define Columns', 'Review & Create'];

  const handleNext = () => {
    if (activeStep === 0 && !tableName.trim()) {
      setError('Table name is required');
      return;
    }
    if (activeStep === 1 && columns.length === 0) {
      setError('At least one column is required');
      return;
    }
    setError(null);
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setError(null);
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleCreate = async () => {
    try {
      setLoading(true);
      setError(null);

      // Format schema for API
      const schema = {
        columns: columns.map(col => ({
          name: col.name,
          type: col.type,
          nullable: col.nullable,
          is_primary_key: col.isPrimaryKey,
          default_value: col.defaultValue || null,
          color: col.color || null,
          format: null
        }))
      };

      // Create table via API
      const createdTable = await dataWorkspaceService.createTable({
        database_id: databaseId,
        name: tableName,
        description: tableDescription,
        table_schema: schema
      });

      if (onTableCreated) {
        onTableCreated(createdTable);
      }

      handleClose();
    } catch (err) {
      setError(err.message || 'Failed to create table');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setActiveStep(0);
    setTableName('');
    setTableDescription('');
    setColumns([
      { name: 'id', type: 'INTEGER', nullable: false, isPrimaryKey: true, defaultValue: '', color: '', description: '' }
    ]);
    setError(null);
    onClose();
  };

  const renderStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <Box sx={{ mt: 2 }}>
            <TextField
              autoFocus
              fullWidth
              label="Table Name"
              value={tableName}
              onChange={(e) => setTableName(e.target.value)}
              placeholder="e.g., books, customers, inventory"
              sx={{ mb: 3 }}
              required
              helperText="Use lowercase with underscores (e.g., my_table)"
            />
            <TextField
              fullWidth
              label="Description (Optional)"
              value={tableDescription}
              onChange={(e) => setTableDescription(e.target.value)}
              placeholder="Describe what this table stores"
              multiline
              rows={3}
            />
          </Box>
        );

      case 1:
        return (
          <Box sx={{ mt: 2 }}>
            <Alert severity="info" sx={{ mb: 2 }}>
              Define the columns for your table. An 'id' column is included by default.
            </Alert>
            <ColumnSchemaEditor
              initialColumns={columns}
              onChange={setColumns}
            />
          </Box>
        );

      case 2:
        return (
          <Box sx={{ mt: 2 }}>
            <Typography variant="h6" gutterBottom>
              Review Your Table
            </Typography>
            
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" color="text.secondary">
                Table Name:
              </Typography>
              <Typography variant="body1" gutterBottom>
                {tableName}
              </Typography>
              
              {tableDescription && (
                <>
                  <Typography variant="subtitle2" color="text.secondary" sx={{ mt: 2 }}>
                    Description:
                  </Typography>
                  <Typography variant="body1" gutterBottom>
                    {tableDescription}
                  </Typography>
                </>
              )}
              
              <Typography variant="subtitle2" color="text.secondary" sx={{ mt: 2 }}>
                Columns: {columns.length}
              </Typography>
              <Box sx={{ mt: 1 }}>
                {columns.map((col, idx) => (
                  <Box
                    key={idx}
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 1,
                      mb: 0.5,
                      p: 1,
                      backgroundColor: 'background.paper',
                      borderRadius: 1,
                      border: '1px solid',
                      borderColor: 'divider'
                    }}
                  >
                    {col.color && (
                      <Box
                        sx={{
                          width: 16,
                          height: 16,
                          backgroundColor: col.color,
                          borderRadius: 1,
                          flexShrink: 0
                        }}
                      />
                    )}
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {col.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      ({col.type})
                    </Typography>
                    {col.isPrimaryKey && (
                      <Typography variant="caption" sx={{ ml: 'auto', color: 'primary.main' }}>
                        🔑 Primary Key
                      </Typography>
                    )}
                    {!col.nullable && (
                      <Typography variant="caption" sx={{ ml: col.isPrimaryKey ? 0 : 'auto', color: 'error.main' }}>
                        * Required
                      </Typography>
                    )}
                  </Box>
                ))}
              </Box>
            </Box>

            <Alert severity="success">
              Ready to create! Click "Create Table" to proceed.
            </Alert>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Dialog 
      open={open} 
      onClose={handleClose} 
      maxWidth="md" 
      fullWidth
      PaperProps={{
        sx: { height: '80vh' }
      }}
    >
      <DialogTitle>
        Create New Table
      </DialogTitle>
      <DialogContent dividers>
        <Stepper activeStep={activeStep} sx={{ mb: 3 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {renderStepContent(activeStep)}
      </DialogContent>
      <DialogActions sx={{ p: 2, justifyContent: 'space-between' }}>
        <Button onClick={handleClose} disabled={loading}>
          Cancel
        </Button>
        <Box>
          <Button
            disabled={activeStep === 0 || loading}
            onClick={handleBack}
            sx={{ mr: 1 }}
          >
            Back
          </Button>
          {activeStep === steps.length - 1 ? (
            <Button
              variant="contained"
              onClick={handleCreate}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={16} /> : null}
            >
              {loading ? 'Creating...' : 'Create Table'}
            </Button>
          ) : (
            <Button variant="contained" onClick={handleNext}>
              Next
            </Button>
          )}
        </Box>
      </DialogActions>
    </Dialog>
  );
};

export default TableCreationWizard;


