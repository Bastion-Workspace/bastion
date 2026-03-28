import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Box,
  Alert,
  CircularProgress,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import { Add as AddIcon, Delete as DeleteIcon, ExpandMore as ExpandMoreIcon } from '@mui/icons-material';
import ColumnSchemaEditor from './ColumnSchemaEditor';
import dataWorkspaceService from '../../services/dataWorkspaceService';

const UPDATE_FREQUENCY_OPTIONS = [
  { value: '', label: 'Not set' },
  { value: 'realtime', label: 'Real-time' },
  { value: 'daily', label: 'Daily' },
  { value: 'weekly', label: 'Weekly' },
  { value: 'monthly', label: 'Monthly' },
  { value: 'manual', label: 'Manual' }
];

function schemaToEditorColumns(schema) {
  if (!schema || !schema.columns || !Array.isArray(schema.columns)) {
    return [{ name: 'id', type: 'INTEGER', nullable: false, isPrimaryKey: true, defaultValue: '', color: '', description: '' }];
  }
  return schema.columns.map((col) => ({
    name: col.name,
    type: col.type || 'TEXT',
    nullable: col.nullable !== false,
    isPrimaryKey: col.is_primary_key === true,
    defaultValue: col.default_value != null ? String(col.default_value) : '',
    color: col.color || '',
    description: col.description || ''
  }));
}

function editorColumnsToSchema(columns) {
  return {
    columns: columns.map((col) => ({
      name: col.name,
      type: col.type,
      nullable: col.nullable,
      is_primary_key: col.isPrimaryKey,
      default_value: col.defaultValue || null,
      color: col.color || null,
      description: col.description || null,
      format: null
    }))
  };
}

const TableSchemaEditDialog = ({ open, onClose, table, onSaved }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [columns, setColumns] = useState([]);
  const [businessContext, setBusinessContext] = useState('');
  const [metadataSource, setMetadataSource] = useState('');
  const [updateFrequency, setUpdateFrequency] = useState('');
  const [glossary, setGlossary] = useState([]);
  const [relationships, setRelationships] = useState([]);
  const [loaded, setLoaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (open && table) {
      setName(table.name || '');
      setDescription(table.description || '');
      const schema = typeof table.table_schema_json === 'string'
        ? JSON.parse(table.table_schema_json || '{}')
        : table.table_schema_json || {};
      setColumns(schemaToEditorColumns(schema));
      let meta = {};
      try {
        const raw = table.metadata_json;
        meta = (typeof raw === 'string' && raw) ? JSON.parse(raw) : (raw || {});
      } catch (_) {
        meta = {};
      }
      setBusinessContext(meta.business_context || '');
      setMetadataSource(meta.source || '');
      setUpdateFrequency(meta.update_frequency || '');
      setGlossary(
        meta.glossary && typeof meta.glossary === 'object'
          ? Object.entries(meta.glossary).map(([k, v]) => ({ key: k, value: v }))
          : []
      );
      setRelationships(
        Array.isArray(meta.relationships)
          ? meta.relationships.map((r) => ({
              column: r.column || '',
              references_table: r.references_table || r.referencesTable || '',
              references_column: r.references_column || r.referencesColumn || 'id'
            }))
          : []
      );
      setLoaded(true);
      setError(null);
    } else {
      setLoaded(false);
    }
  }, [open, table]);

  const handleGlossaryAdd = () => setGlossary((prev) => [...prev, { key: '', value: '' }]);
  const handleGlossaryRemove = (index) => setGlossary((prev) => prev.filter((_, i) => i !== index));
  const handleGlossaryChange = (index, field, value) => {
    setGlossary((prev) => prev.map((item, i) => (i === index ? { ...item, [field]: value } : item)));
  };
  const handleRelationshipAdd = () =>
    setRelationships((prev) => [...prev, { column: '', references_table: '', references_column: 'id' }]);
  const handleRelationshipRemove = (index) => setRelationships((prev) => prev.filter((_, i) => i !== index));
  const handleRelationshipChange = (index, field, value) => {
    setRelationships((prev) => prev.map((item, i) => (i === index ? { ...item, [field]: value } : item)));
  };

  const handleSave = async () => {
    if (!table) return;
    if (!name.trim()) {
      setError('Table name is required');
      return;
    }
    if (columns.length === 0) {
      setError('At least one column is required');
      return;
    }
    try {
      setLoading(true);
      setError(null);
      const tableSchema = editorColumnsToSchema(columns);
      const glossaryObj = {};
      glossary.forEach(({ key, value }) => {
        if (key && key.trim()) glossaryObj[key.trim()] = value || '';
      });
      const metadata = {};
      if (businessContext.trim()) metadata.business_context = businessContext.trim();
      if (metadataSource.trim()) metadata.source = metadataSource.trim();
      if (updateFrequency) metadata.update_frequency = updateFrequency;
      if (Object.keys(glossaryObj).length) metadata.glossary = glossaryObj;
      if (relationships.some((r) => r.column && r.references_table)) {
        metadata.relationships = relationships
          .filter((r) => r.column && r.references_table)
          .map((r) => ({
            column: r.column,
            references_table: r.references_table,
            references_column: r.references_column || 'id'
          }));
      }
      await dataWorkspaceService.updateTable(table.table_id, {
        name: name.trim(),
        description: description.trim() || null,
        table_schema: tableSchema,
        metadata: Object.keys(metadata).length ? metadata : null
      });
      if (onSaved) onSaved();
      onClose();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to update table');
    } finally {
      setLoading(false);
    }
  };

  if (!table) return null;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth PaperProps={{ sx: { height: '85vh' } }}>
      <DialogTitle>Edit table: {table.name}</DialogTitle>
      <DialogContent dividers>
        <Box sx={{ mt: 1 }}>
          <TextField
            fullWidth
            label="Table name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. my_table"
            sx={{ mb: 2 }}
          />
          <TextField
            fullWidth
            label="Description (optional)"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            multiline
            rows={2}
            sx={{ mb: 2 }}
          />
          {loaded && (
            <ColumnSchemaEditor
              key={table.table_id}
              initialColumns={columns}
              onChange={setColumns}
            />
          )}
          {loaded && (
            <Accordion defaultExpanded={false} sx={{ mt: 2 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">Table context (for agents)</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Optional business context and definitions so agents can interpret this table correctly.
                </Typography>
                <TextField
                  fullWidth
                  label="Business context"
                  value={businessContext}
                  onChange={(e) => setBusinessContext(e.target.value)}
                  placeholder="e.g. Monthly net revenue by product line, post-refund"
                  multiline
                  rows={2}
                  sx={{ mb: 2 }}
                />
                <TextField
                  fullWidth
                  label="Source"
                  value={metadataSource}
                  onChange={(e) => setMetadataSource(e.target.value)}
                  placeholder="e.g. QuickBooks CSV export"
                  sx={{ mb: 2 }}
                />
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Update frequency</InputLabel>
                  <Select
                    value={updateFrequency}
                    onChange={(e) => setUpdateFrequency(e.target.value)}
                    label="Update frequency"
                  >
                    {UPDATE_FREQUENCY_OPTIONS.map((opt) => (
                      <MenuItem key={opt.value || 'none'} value={opt.value}>
                        {opt.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Typography variant="caption" sx={{ display: 'block', mb: 1 }}>Glossary (column/term definitions)</Typography>
                {glossary.map((item, index) => (
                  <Box key={index} sx={{ display: 'flex', gap: 1, mb: 1, alignItems: 'center' }}>
                    <TextField
                      size="small"
                      placeholder="Term or column"
                      value={item.key}
                      onChange={(e) => handleGlossaryChange(index, 'key', e.target.value)}
                      sx={{ flex: 1 }}
                    />
                    <TextField
                      size="small"
                      placeholder="Definition"
                      value={item.value}
                      onChange={(e) => handleGlossaryChange(index, 'value', e.target.value)}
                      sx={{ flex: 2 }}
                    />
                    <IconButton size="small" onClick={() => handleGlossaryRemove(index)} color="error">
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Box>
                ))}
                <Button size="small" startIcon={<AddIcon />} onClick={handleGlossaryAdd} sx={{ mb: 2 }}>
                  Add glossary entry
                </Button>
                <Typography variant="caption" sx={{ display: 'block', mb: 1 }}>Relationships (column references another table)</Typography>
                {relationships.map((item, index) => (
                  <Box key={index} sx={{ display: 'flex', gap: 1, mb: 1, alignItems: 'center', flexWrap: 'wrap' }}>
                    <FormControl size="small" sx={{ minWidth: 120 }}>
                      <InputLabel>Column</InputLabel>
                      <Select
                        value={item.column}
                        onChange={(e) => handleRelationshipChange(index, 'column', e.target.value)}
                        label="Column"
                      >
                        {columns.map((col) => (
                          <MenuItem key={col.name} value={col.name}>{col.name}</MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <TextField
                      size="small"
                      placeholder="References table"
                      value={item.references_table}
                      onChange={(e) => handleRelationshipChange(index, 'references_table', e.target.value)}
                      sx={{ minWidth: 140 }}
                    />
                    <TextField
                      size="small"
                      placeholder="References column"
                      value={item.references_column}
                      onChange={(e) => handleRelationshipChange(index, 'references_column', e.target.value)}
                      sx={{ width: 120 }}
                    />
                    <IconButton size="small" onClick={() => handleRelationshipRemove(index)} color="error">
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Box>
                ))}
                <Button size="small" startIcon={<AddIcon />} onClick={handleRelationshipAdd}>
                  Add relationship
                </Button>
              </AccordionDetails>
            </Accordion>
          )}
        </Box>
        {error && (
          <Alert severity="error" sx={{ mt: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
      </DialogContent>
      <DialogActions sx={{ p: 2 }}>
        <Button onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <Button variant="contained" onClick={handleSave} disabled={loading} startIcon={loading ? <CircularProgress size={16} /> : null}>
          {loading ? 'Saving...' : 'Save changes'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default TableSchemaEditDialog;
