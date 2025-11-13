import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Stepper,
  Step,
  StepLabel,
  Box,
  Typography,
  Alert,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  LinearProgress,
  IconButton,
  Tooltip,
  Chip,
  Grid,
  Card,
  CardContent,
  CircularProgress,
  TextField
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  SwapHoriz as MappingIcon,
  Check as CheckIcon,
  Close as CloseIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';
import dataWorkspaceService from '../../services/dataWorkspaceService';

const SUPPORTED_FORMATS = [
  { value: 'csv', label: 'CSV', mimeTypes: ['text/csv', 'application/csv'] },
  { value: 'json', label: 'JSON', mimeTypes: ['application/json'] },
  { value: 'jsonl', label: 'JSON Lines (JSONL)', mimeTypes: ['application/x-ndjson', 'application/jsonl'] }
];

const SQL_TYPES = ['TEXT', 'INTEGER', 'REAL', 'BOOLEAN', 'TIMESTAMP'];

const DataImportWizard = ({ 
  open, 
  onClose, 
  databaseId,
  workspaceId: propWorkspaceId,
  existingTables = [],
  onImportComplete 
}) => {
  const [activeStep, setActiveStep] = useState(0);
  const [file, setFile] = useState(null);
  const [fileFormat, setFileFormat] = useState('csv');
  const [dragActive, setDragActive] = useState(false);
  
  // Preview data
  const [previewData, setPreviewData] = useState(null);
  const [inferredSchema, setInferredSchema] = useState([]);
  const [estimatedRows, setEstimatedRows] = useState(0);
  
  // Mapping
  const [targetTable, setTargetTable] = useState('new');
  const [newTableName, setNewTableName] = useState('');
  const [fieldMapping, setFieldMapping] = useState({});
  const [typeOverrides, setTypeOverrides] = useState({});
  
  // Import status
  const [importing, setImporting] = useState(false);
  const [importProgress, setImportProgress] = useState(0);
  const [importJobId, setImportJobId] = useState(null);
  const [importError, setImportError] = useState(null);
  const [uploadedFilePath, setUploadedFilePath] = useState(null);
  const [workspaceId, setWorkspaceId] = useState(null);

  const steps = ['Upload File', 'Preview & Map Fields', 'Import'];

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    
    // Auto-detect format from extension
    const extension = selectedFile.name.split('.').pop().toLowerCase();
    if (extension === 'jsonl' || extension === 'ndjson') {
      setFileFormat('jsonl');
    } else if (extension === 'json') {
      setFileFormat('json');
    } else if (extension === 'csv') {
      setFileFormat('csv');
    }
  };

  const handlePreviewFile = async () => {
    if (!file) return;

    try {
      setImporting(true);
      setImportError(null);
      
      // Step 1: Upload file
      const uploadResponse = await dataWorkspaceService.uploadFile(propWorkspaceId, file);
      setUploadedFilePath(uploadResponse.file_path);
      
      // Step 2: Get preview
      const previewResponse = await dataWorkspaceService.previewImport({
        workspace_id: propWorkspaceId,
        file_path: uploadResponse.file_path,
        file_type: fileFormat,
        preview_rows: 10
      });
      
      setPreviewData(previewResponse.preview_data);
      setInferredSchema(previewResponse.inferred_types);
      setEstimatedRows(previewResponse.estimated_rows);
      
      // Initialize field mapping (source -> target)
      const initialMapping = {};
      const initialTypes = {};
      previewResponse.inferred_types.forEach(col => {
        initialMapping[col.name] = col.name;
        initialTypes[col.name] = col.type;
      });
      
      setFieldMapping(initialMapping);
      setTypeOverrides(initialTypes);
      setActiveStep(1);
      
    } catch (error) {
      console.error('Failed to preview file:', error);
      setImportError(`Failed to preview file: ${error.response?.data?.detail || error.message}`);
    } finally {
      setImporting(false);
    }
  };

  const handleExecuteImport = async () => {
    try {
      setImporting(true);
      setImportProgress(0);
      setImportError(null);
      
      // Execute import
      const tableName = targetTable === 'new' ? newTableName : targetTable;
      const response = await dataWorkspaceService.executeImport({
        workspace_id: propWorkspaceId,
        database_id: databaseId,
        table_name: tableName,
        file_path: uploadedFilePath,
        field_mapping: fieldMapping
      });
      
      setImportJobId(response.job_id);
      
      // Poll for import progress
      const pollInterval = setInterval(async () => {
        try {
          const status = await dataWorkspaceService.getImportStatus(response.job_id);
          setImportProgress(status.progress_percent);
          
          if (status.status === 'completed') {
            clearInterval(pollInterval);
            setActiveStep(2);
            setTimeout(() => {
              if (onImportComplete) onImportComplete();
              handleClose();
            }, 2000);
          } else if (status.status === 'failed') {
            clearInterval(pollInterval);
            setImportError(`Import failed: ${status.error_log}`);
            setImporting(false);
          }
        } catch (pollError) {
          console.error('Failed to poll import status:', pollError);
        }
      }, 1000);
      
    } catch (error) {
      console.error('Failed to import data:', error);
      setImportError(`Failed to import data: ${error.response?.data?.detail || error.message}`);
      setImporting(false);
    }
  };

  const handleClose = () => {
    setActiveStep(0);
    setFile(null);
    setFileFormat('csv');
    setPreviewData(null);
    setInferredSchema([]);
    setTargetTable('new');
    setNewTableName('');
    setFieldMapping({});
    setTypeOverrides({});
    setImporting(false);
    setImportProgress(0);
    setImportError(null);
    onClose();
  };

  const handleNext = () => {
    if (activeStep === 0) {
      handlePreviewFile();
    } else if (activeStep === 1) {
      handleExecuteImport();
    }
  };

  const handleBack = () => {
    setActiveStep((prev) => prev - 1);
  };

  const canProceed = () => {
    if (activeStep === 0) {
      return file && fileFormat;
    } else if (activeStep === 1) {
      const hasTableName = targetTable !== 'new' || newTableName.trim().length > 0;
      const hasValidMapping = Object.keys(fieldMapping).length > 0;
      return hasTableName && hasValidMapping;
    }
    return false;
  };

  // Step 1: File Upload
  const renderFileUpload = () => (
    <Box sx={{ py: 3 }}>
      <Box
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        sx={{
          border: '2px dashed',
          borderColor: dragActive ? 'primary.main' : 'divider',
          borderRadius: 2,
          p: 6,
          textAlign: 'center',
          backgroundColor: dragActive ? 'action.hover' : 'background.paper',
          cursor: 'pointer',
          transition: 'all 0.2s',
          '&:hover': {
            borderColor: 'primary.main',
            backgroundColor: 'action.hover'
          }
        }}
        onClick={() => document.getElementById('file-input').click()}
      >
        <input
          id="file-input"
          type="file"
          accept=".csv,.json,.jsonl,.ndjson"
          onChange={handleFileInput}
          style={{ display: 'none' }}
        />
        
        <UploadIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
        
        {file ? (
          <>
            <Typography variant="h6" gutterBottom>
              {file.name}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {(file.size / 1024).toFixed(2)} KB
            </Typography>
            <Button
              size="small"
              startIcon={<DeleteIcon />}
              onClick={(e) => {
                e.stopPropagation();
                setFile(null);
              }}
              sx={{ mt: 2 }}
            >
              Remove File
            </Button>
          </>
        ) : (
          <>
            <Typography variant="h6" gutterBottom>
              Drag & drop your file here
            </Typography>
            <Typography variant="body2" color="text.secondary">
              or click to browse
            </Typography>
          </>
        )}
      </Box>

      {file && (
        <Box sx={{ mt: 3 }}>
          <FormControl fullWidth>
            <InputLabel>File Format</InputLabel>
            <Select
              value={fileFormat}
              onChange={(e) => setFileFormat(e.target.value)}
              label="File Format"
            >
              {SUPPORTED_FORMATS.map(format => (
                <MenuItem key={format.value} value={format.value}>
                  {format.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
      )}

      <Alert severity="info" sx={{ mt: 3 }}>
        <Typography variant="body2" fontWeight={600} gutterBottom>
          Supported Formats:
        </Typography>
        <Typography variant="body2" component="div">
          • <strong>CSV</strong> - Comma-separated values<br />
          • <strong>JSON</strong> - Array of objects or single object<br />
          • <strong>JSONL</strong> - One JSON object per line (also called NDJSON)
        </Typography>
      </Alert>
    </Box>
  );

  // Step 2: Preview & Mapping
  const renderPreviewMapping = () => (
    <Box sx={{ py: 3 }}>
      {/* Table selection */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Import Destination
          </Typography>
          
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Target Table</InputLabel>
            <Select
              value={targetTable}
              onChange={(e) => setTargetTable(e.target.value)}
              label="Target Table"
            >
              <MenuItem value="new">
                <em>Create New Table</em>
              </MenuItem>
              {existingTables.map(table => (
                <MenuItem key={table.table_id} value={table.name}>
                  {table.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {targetTable === 'new' && (
            <TextField
              fullWidth
              label="New Table Name"
              value={newTableName}
              onChange={(e) => setNewTableName(e.target.value)}
              placeholder="Enter table name"
              required
            />
          )}
        </CardContent>
      </Card>

      {/* Preview data */}
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <InfoIcon color="primary" />
        Data Preview ({estimatedRows.toLocaleString()} rows estimated)
      </Typography>

      <TableContainer component={Paper} sx={{ mb: 3, maxHeight: 300 }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              {inferredSchema.map((col) => (
                <TableCell key={col.name}>
                  <Typography variant="subtitle2" fontWeight={600}>
                    {col.name}
                  </Typography>
                  <Chip 
                    label={typeOverrides[col.name] || col.type} 
                    size="small" 
                    sx={{ mt: 0.5 }}
                  />
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {previewData && previewData.map((row, idx) => (
              <TableRow key={idx}>
                {inferredSchema.map((col) => (
                  <TableCell key={col.name}>
                    {String(row[col.name] ?? '-')}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Field mapping */}
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <MappingIcon color="primary" />
        Field Mapping
      </Typography>

      <Grid container spacing={2}>
        {inferredSchema.map((col) => (
          <Grid item xs={12} md={6} key={col.name}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                {col.name}
              </Typography>
              
              <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                <InputLabel>Target Column</InputLabel>
                <Select
                  value={fieldMapping[col.name] || col.name}
                  onChange={(e) => setFieldMapping({ ...fieldMapping, [col.name]: e.target.value })}
                  label="Target Column"
                >
                  <MenuItem value={col.name}>{col.name}</MenuItem>
                  <MenuItem value="_skip">
                    <em>Skip this field</em>
                  </MenuItem>
                </Select>
              </FormControl>

              <FormControl fullWidth size="small">
                <InputLabel>Data Type</InputLabel>
                <Select
                  value={typeOverrides[col.name] || col.type}
                  onChange={(e) => setTypeOverrides({ ...typeOverrides, [col.name]: e.target.value })}
                  label="Data Type"
                >
                  {SQL_TYPES.map(type => (
                    <MenuItem key={type} value={type}>{type}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Paper>
          </Grid>
        ))}
      </Grid>

      {importError && (
        <Alert severity="error" sx={{ mt: 3 }}>
          {importError}
        </Alert>
      )}
    </Box>
  );

  // Step 3: Import Progress
  const renderImportProgress = () => (
    <Box sx={{ py: 3, textAlign: 'center' }}>
      {importing ? (
        <>
          <CircularProgress size={64} sx={{ mb: 3 }} />
          <Typography variant="h6" gutterBottom>
            Importing Data...
          </Typography>
          <LinearProgress 
            variant="determinate" 
            value={importProgress} 
            sx={{ my: 3, height: 8, borderRadius: 4 }}
          />
          <Typography variant="body2" color="text.secondary">
            {importProgress}% complete
          </Typography>
        </>
      ) : (
        <>
          <CheckIcon sx={{ fontSize: 64, color: 'success.main', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Import Complete!
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Successfully imported {estimatedRows.toLocaleString()} rows
          </Typography>
        </>
      )}
    </Box>
  );

  return (
    <Dialog 
      open={open} 
      onClose={importing ? undefined : handleClose}
      maxWidth="md" 
      fullWidth
    >
      <DialogTitle>
        Import Data
        {!importing && (
          <IconButton
            onClick={handleClose}
            sx={{ position: 'absolute', right: 8, top: 8 }}
          >
            <CloseIcon />
          </IconButton>
        )}
      </DialogTitle>

      <DialogContent>
        <Stepper activeStep={activeStep} sx={{ mb: 3 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {activeStep === 0 && renderFileUpload()}
        {activeStep === 1 && renderPreviewMapping()}
        {activeStep === 2 && renderImportProgress()}
      </DialogContent>

      {activeStep < 2 && (
        <DialogActions>
          <Button onClick={handleClose} disabled={importing}>
            Cancel
          </Button>
          {activeStep > 0 && (
            <Button onClick={handleBack} disabled={importing}>
              Back
            </Button>
          )}
          <Button
            onClick={handleNext}
            variant="contained"
            disabled={!canProceed() || importing}
          >
            {activeStep === 1 ? 'Import' : 'Next'}
          </Button>
        </DialogActions>
      )}
    </Dialog>
  );
};

export default DataImportWizard;

