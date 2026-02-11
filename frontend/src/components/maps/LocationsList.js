import React from 'react';
import {
  List,
  ListItem,
  ListItemText,
  IconButton,
  Box,
  Typography,
  Chip,
  CircularProgress,
} from '@mui/material';
import { Edit, Delete, Public, Lock } from '@mui/icons-material';

const LocationsList = ({ locations, onEdit, onDelete, isLoading, readOnly }) => {
  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (locations.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', p: 4 }}>
        <Typography color="text.secondary">
          No locations yet. Click "Add Location" to get started.
        </Typography>
      </Box>
    );
  }

  return (
    <List>
      {locations.map((location) => (
        <ListItem
          key={location.location_id}
          secondaryAction={
            !readOnly && (
              <Box>
                <IconButton edge="end" onClick={() => onEdit(location)}>
                  <Edit />
                </IconButton>
                <IconButton edge="end" onClick={() => onDelete(location.location_id)}>
                  <Delete />
                </IconButton>
              </Box>
            )
          }
        >
          <ListItemText
            primary={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {location.name}
                {location.is_global ? (
                  <Chip icon={<Public />} label="Global" size="small" />
                ) : (
                  <Chip icon={<Lock />} label="Private" size="small" />
                )}
              </Box>
            }
            secondary={
              <Box>
                {location.address && (
                  <Typography variant="body2" color="text.secondary">
                    {location.address}
                  </Typography>
                )}
                <Typography variant="caption" color="text.secondary">
                  {location.latitude}, {location.longitude}
                </Typography>
                {location.notes && (
                  <Typography variant="body2" sx={{ mt: 0.5 }}>
                    {location.notes}
                  </Typography>
                )}
              </Box>
            }
          />
        </ListItem>
      ))}
    </List>
  );
};

export default LocationsList;
