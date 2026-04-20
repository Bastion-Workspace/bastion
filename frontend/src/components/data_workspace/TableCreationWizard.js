import React, { useState, useEffect } from 'react';
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

const TableCreationWizard = ({ open, onClose, databaseId, onTableCreated, container }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [tableName, setTableName] = useState('');
  const [tableDescription, setTableDescription] = useState('');
  const [columns, setColumns] = useState([
    { name: 'id', type: 'INTEGER', nullable: false, isPrimaryKey: true, defaultValue: '', color: '', description: '' }
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [peerTables, setPeerTables] = useState([]);

  useEffect(() => {
    if (!open || !databaseId) return;
    let cancelled = false;
    (async () => {
      try {
        const list = await dataWorkspaceService.listTables(databaseId);
        if (!cancelled) setPeerTables(Array.isArray(list) ? list : []);
      } catch {
        if (!cancelled) setPeerTables([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open, databaseId]);

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
        columns: columns.map((col) => {
          const base = {
            name: col.name,
            type: col.type,
            nullable: col.nullable,
            is_primary_key: col.isPrimaryKey,
            default_value: col.defaultValue || null,
            color: col.color || null,
            description: col.description || null,
            format: null
          };
          if (col.type === 'REFERENCE' && col.ref && col.ref.target_table_id) {
            base.ref = {
              target_table_id: col.ref.target_table_id,
              target_key: col.ref.target_key || 'row_id',
              label_field: col.ref.label_field || 'name'
            };
          }
          return base;
        })
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
          <Box sx={{ mt: 0.5 }}>
            <TextField
              autoFocus
              fullWidth
              size="small"
              margin="dense"
              label="Table Name"
              value={tableName}
              onChange={(e) => setTableName(e.target.value)}
              placeholder="e.g., books, customers, inventory"
              sx={{ mb: 1.25 }}
              required
              helperText="Lowercase with underscores (e.g. my_table)"
              FormHelperTextProps={{ sx: { fontSize: '0.7rem', m: 0, mt: 0.25 } }}
            />
            <TextField
              fullWidth
              size="small"
              margin="dense"
              label="Description (optional)"
              value={tableDescription}
              onChange={(e) => setTableDescription(e.target.value)}
              placeholder="What this table stores"
              multiline
              rows={2}
            />
          </Box>
        );

      case 1:
        return (
          <Box sx={{ mt: 0.5 }}>
            <Alert severity="info" sx={{ py: 0.25, px: 1, mb: 1, '& .MuiAlert-message': { fontSize: '0.75rem', py: 0.25 } }}>
              Define columns below. An <code style={{ fontSize: '0.7rem' }}>id</code> column is included by default.
            </Alert>
            <ColumnSchemaEditor
              initialColumns={columns}
              onChange={setColumns}
              dialogContainer={container}
              databaseId={databaseId}
              peerTables={peerTables}
              currentTableId={null}
            />
          </Box>
        );

      case 2:
        return (
          <Box sx={{ mt: 0.5 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.75 }}>
              Review
            </Typography>

            <Box sx={{ mb: 1 }}>
              <Typography variant="caption" color="text.secondary" display="block" sx={{ lineHeight: 1.2 }}>
                Name
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {tableName}
              </Typography>

              {tableDescription && (
                <>
                  <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.75, lineHeight: 1.2 }}>
                    Description
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {tableDescription}
                  </Typography>
                </>
              )}

              <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.75, lineHeight: 1.2 }}>
                Columns ({columns.length})
              </Typography>
              <Box sx={{ mt: 0.5, display: 'flex', flexDirection: 'column', gap: 0.25 }}>
                {columns.map((col, idx) => (
                  <Box
                    key={idx}
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 0.75,
                      py: 0.35,
                      px: 0.75,
                      backgroundColor: 'action.hover',
                      borderRadius: 0.75,
                      border: '1px solid',
                      borderColor: 'divider'
                    }}
                  >
                    {col.color && (
                      <Box
                        sx={{
                          width: 12,
                          height: 12,
                          backgroundColor: col.color,
                          borderRadius: 0.5,
                          flexShrink: 0
                        }}
                      />
                    )}
                    <Typography variant="caption" sx={{ fontWeight: 600, fontSize: '0.75rem' }}>
                      {col.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                      {col.type}
                    </Typography>
                    <Box sx={{ ml: 'auto', display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      {col.isPrimaryKey && (
                        <Typography component="span" variant="caption" sx={{ color: 'primary.main', fontSize: '0.65rem', fontWeight: 600 }}>
                          PK
                        </Typography>
                      )}
                      {!col.nullable && (
                        <Typography component="span" variant="caption" sx={{ color: 'error.main', fontSize: '0.65rem' }}>
                          req
                        </Typography>
                      )}
                    </Box>
                  </Box>
                ))}
              </Box>
            </Box>

            <Alert severity="success" sx={{ py: 0.25, px: 1, '& .MuiAlert-message': { fontSize: '0.75rem', py: 0.25 } }}>
              Click <strong>Create Table</strong> to finish.
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
      container={container}
      PaperProps={{
        sx: { maxHeight: 'min(88vh, 900px)' }
      }}
    >
      <DialogTitle sx={{ py: 1, px: 2, fontSize: '0.95rem', fontWeight: 600 }}>
        Create table
      </DialogTitle>
      <DialogContent
        dividers
        sx={{
          pt: 1.5,
          px: 2,
          pb: 1,
          fontSize: '0.8125rem'
        }}
      >
        <Stepper
          activeStep={activeStep}
          sx={{
            mb: 1.5,
            '& .MuiStepLabel-label': { fontSize: '0.7rem', mt: 0.25 },
            '& .MuiStepIcon-root': { fontSize: '1.15rem' },
            '& .MuiStepConnector-line': { minHeight: 6 }
          }}
        >
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {error && (
          <Alert
            severity="error"
            sx={{ mb: 1, py: 0.25, '& .MuiAlert-message': { fontSize: '0.75rem' } }}
            onClose={() => setError(null)}
          >
            {error}
          </Alert>
        )}

        {renderStepContent(activeStep)}
      </DialogContent>
      <DialogActions sx={{ px: 2, py: 1, justifyContent: 'space-between' }}>
        <Button size="small" onClick={handleClose} disabled={loading}>
          Cancel
        </Button>
        <Box sx={{ display: 'flex', gap: 0.5 }}>
          <Button
            size="small"
            disabled={activeStep === 0 || loading}
            onClick={handleBack}
          >
            Back
          </Button>
          {activeStep === steps.length - 1 ? (
            <Button
              size="small"
              variant="contained"
              onClick={handleCreate}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={14} /> : null}
            >
              {loading ? 'Creating…' : 'Create Table'}
            </Button>
          ) : (
            <Button size="small" variant="contained" onClick={handleNext}>
              Next
            </Button>
          )}
        </Box>
      </DialogActions>
    </Dialog>
  );
};

export default TableCreationWizard;


