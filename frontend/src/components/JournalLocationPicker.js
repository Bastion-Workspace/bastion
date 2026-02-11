import React, { useState, useEffect } from 'react';
import {
  Autocomplete,
  TextField,
  Box,
  Typography,
  CircularProgress,
  Alert
} from '@mui/material';
import { Folder } from '@mui/icons-material';
import apiService from '../services/apiService';

/**
 * Journal Location Picker
 * 
 * Dropdown that displays filesystem directory tree for journal location selection
 */
const JournalLocationPicker = ({ value, onChange, disabled, error }) => {
  const [directories, setDirectories] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorState, setErrorState] = useState(null);
  const [options, setOptions] = useState([]);

  // Load directory tree
  useEffect(() => {
    loadDirectories();
  }, []);

  // Build flat list of options from tree
  useEffect(() => {
    const flattenTree = (tree, parentPath = '') => {
      const result = [];
      tree.forEach(dir => {
        const fullPath = parentPath ? `${parentPath}/${dir.name}` : dir.name;
        result.push({
          label: fullPath,
          value: fullPath,
          name: dir.name
        });
        if (dir.children && dir.children.length > 0) {
          result.push(...flattenTree(dir.children, fullPath));
        }
      });
      return result;
    };

    const flatOptions = flattenTree(directories);
    // Add root option (empty)
    flatOptions.unshift({
      label: '(Root - Users/[username])',
      value: '',
      name: ''
    });
    setOptions(flatOptions);
  }, [directories]);

  const loadDirectories = async () => {
    try {
      setLoading(true);
      setErrorState(null);
      
      const response = await apiService.get('/api/org/settings/journal-locations');
      
      if (response && response.success && response.directories) {
        setDirectories(response.directories);
      } else {
        setErrorState('Failed to load directories');
      }
    } catch (err) {
      console.error('Failed to load journal locations:', err);
      setErrorState(err.message || 'Failed to load directories');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Autocomplete
        value={options.find(opt => opt.value === value) || null}
        onChange={(event, newValue) => {
          onChange(newValue ? newValue.value : '');
        }}
        options={options}
        getOptionLabel={(option) => option.label || ''}
        isOptionEqualToValue={(option, value) => option.value === value.value}
        disabled={disabled || loading}
        loading={loading}
        renderInput={(params) => (
          <TextField
            {...params}
            label="Journal Location"
            helperText="Select folder for journal files (relative to your user directory). Leave empty for root."
            error={!!error}
            InputProps={{
              ...params.InputProps,
              startAdornment: (
                <>
                  <Folder sx={{ mr: 1, color: 'action.active' }} />
                  {params.InputProps.startAdornment}
                </>
              ),
              endAdornment: (
                <>
                  {loading ? <CircularProgress color="inherit" size={20} /> : null}
                  {params.InputProps.endAdornment}
                </>
              )
            }}
          />
        )}
        renderOption={(props, option) => (
          <Box component="li" {...props} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Folder fontSize="small" sx={{ color: 'text.secondary' }} />
            <Typography>{option.label}</Typography>
          </Box>
        )}
        noOptionsText={loading ? "Loading directories..." : "No directories found"}
      />
      {errorState && (
        <Alert severity="error" sx={{ mt: 1 }}>
          {errorState}
        </Alert>
      )}
    </Box>
  );
};

export default JournalLocationPicker;
