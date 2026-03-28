import React from 'react';
import { Box, Typography } from '@mui/material';
import ControlPanesSettingsTab from './settings/ControlPanesSettingsTab';

const ControlPanesPage = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Control Panes
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Configure status bar controls and pane behavior.
      </Typography>
      <ControlPanesSettingsTab />
    </Box>
  );
};

export default ControlPanesPage;
