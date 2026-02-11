import React, { useState, useEffect, useCallback } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Box,
  Alert,
  FormControlLabel,
  Checkbox,
  CircularProgress,
} from '@mui/material';
import { MyLocation } from '@mui/icons-material';
import { useAuth } from '../../contexts/AuthContext';
import apiService from '../../services/apiService';

const LocationDialog = ({ open, onClose, onSave, location }) => {
  const { user } = useAuth();
  const isAdmin = user?.role === 'admin';

  const [formData, setFormData] = useState({
    name: '',
    address: '',
    latitude: '',
    longitude: '',
    notes: '',
    is_global: false,
  });
  const [geocoding, setGeocoding] = useState(false);
  const [error, setError] = useState('');

  const handleReverseGeocode = useCallback(async (lat, lng) => {
    // Reverse geocoding: convert coordinates to address
    setGeocoding(true);
    try {
      // Use OpenStreetMap Nominatim API for reverse geocoding (free, no API key needed)
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&zoom=18&addressdetails=1`
      );
      const data = await response.json();
      
      if (data && data.address) {
        // Build address string from components
        const addr = data.address;
        const parts = [];
        if (addr.house_number) parts.push(addr.house_number);
        if (addr.road) parts.push(addr.road);
        if (addr.city || addr.town || addr.village) parts.push(addr.city || addr.town || addr.village);
        if (addr.state) parts.push(addr.state);
        if (addr.postcode) parts.push(addr.postcode);
        if (addr.country) parts.push(addr.country);
        
        const addressString = parts.join(', ');
        if (addressString) {
          setFormData(prev => ({ ...prev, address: addressString }));
        }
      }
    } catch (err) {
      // Silently fail - user can enter address manually
      console.log('Reverse geocoding failed, user can enter address manually');
    } finally {
      setGeocoding(false);
    }
  }, []);

  useEffect(() => {
    if (location) {
      // Handle both full location objects (edit) and partial objects (double-click from map)
      setFormData({
        name: location.name || '',
        address: location.address || '',
        latitude: location.latitude ? location.latitude.toString() : '',
        longitude: location.longitude ? location.longitude.toString() : '',
        notes: location.notes || '',
        is_global: location.is_global || false,
      });
      
      // If we have coordinates but no address, try reverse geocoding
      if (location.latitude && location.longitude && !location.address && !location.location_id) {
        handleReverseGeocode(location.latitude, location.longitude);
      }
    } else {
      setFormData({
        name: '',
        address: '',
        latitude: '',
        longitude: '',
        notes: '',
        is_global: false,
      });
    }
    setError('');
  }, [location, open, handleReverseGeocode]);

  const handleGeocodeAddress = async () => {
    if (!formData.address) {
      setError('Please enter an address first');
      return;
    }

    setGeocoding(true);
    setError('');

    try {
      // Create a temporary location to trigger backend geocoding
      const response = await apiService.post('/api/locations', {
        name: 'temp_geocode',
        address: formData.address,
        latitude: null,
        longitude: null,
      });

      if (response && response.latitude && response.longitude) {
        setFormData(prev => ({
          ...prev,
          latitude: response.latitude.toString(),
          longitude: response.longitude.toString(),
        }));
        // Delete the temp location
        if (response.location_id) {
          await apiService.delete(`/api/locations/${response.location_id}`).catch(() => {});
        }
      }
    } catch (err) {
      setError('Failed to geocode address. Coordinates will be filled automatically when you save, or you can enter them manually.');
    } finally {
      setGeocoding(false);
    }
  };

  const handleSave = () => {
    if (!formData.name) {
      setError('Location name is required');
      return;
    }

    // Address is optional if we have coordinates (from double-click)
    // If we have coordinates but no address, use a placeholder
    const address = formData.address || 
      (formData.latitude && formData.longitude 
        ? `${formData.latitude}, ${formData.longitude}` 
        : '');

    if (!address && !formData.latitude && !formData.longitude) {
      setError('Either address or coordinates are required');
      return;
    }

    onSave({
      name: formData.name,
      address: address,
      latitude: formData.latitude ? parseFloat(formData.latitude) : null,
      longitude: formData.longitude ? parseFloat(formData.longitude) : null,
      notes: formData.notes || null,
      is_global: formData.is_global,
    });
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        {location?.location_id ? 'Edit Location' : 'Add Location'}
      </DialogTitle>
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <TextField
          label="Location Name"
          fullWidth
          value={formData.name}
          onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
          placeholder="Home, Work, Office, etc."
          sx={{ mt: 2, mb: 2 }}
        />

        <TextField
          label="Address"
          fullWidth
          value={formData.address}
          onChange={(e) => setFormData(prev => ({ ...prev, address: e.target.value }))}
          placeholder="123 Main St, Los Angeles, CA (or double-click map to set coordinates)"
          sx={{ mb: 1 }}
          helperText={formData.latitude && formData.longitude && !formData.address ? "Address will be reverse-geocoded from coordinates" : ""}
        />

        <Button
          variant="outlined"
          startIcon={geocoding ? <CircularProgress size={16} /> : <MyLocation />}
          onClick={handleGeocodeAddress}
          disabled={geocoding || !formData.address}
          sx={{ mb: 2 }}
        >
          {geocoding ? 'Geocoding...' : 'Find Coordinates (Optional)'}
        </Button>

        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <TextField
            label="Latitude"
            value={formData.latitude}
            onChange={(e) => setFormData(prev => ({ ...prev, latitude: e.target.value }))}
            type="number"
            inputProps={{ step: 0.00000001 }}
            fullWidth
            helperText="Will be auto-filled when saving if left empty"
          />
          <TextField
            label="Longitude"
            value={formData.longitude}
            onChange={(e) => setFormData(prev => ({ ...prev, longitude: e.target.value }))}
            type="number"
            inputProps={{ step: 0.00000001 }}
            fullWidth
            helperText="Will be auto-filled when saving if left empty"
          />
        </Box>

        <TextField
          label="Notes (Optional)"
          fullWidth
          multiline
          rows={3}
          value={formData.notes}
          onChange={(e) => setFormData(prev => ({ ...prev, notes: e.target.value }))}
          placeholder="Additional information about this location"
          sx={{ mb: 2 }}
        />

        {isAdmin && (
          <FormControlLabel
            control={
              <Checkbox
                checked={formData.is_global}
                onChange={(e) => setFormData(prev => ({ ...prev, is_global: e.target.checked }))}
              />
            }
            label="Global Location (visible to all users with map access)"
          />
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSave} variant="contained">
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default LocationDialog;
