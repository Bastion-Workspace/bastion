import React from 'react';
import { Box, IconButton, Typography, ToggleButton, ToggleButtonGroup } from '@mui/material';
import { ArrowBack } from '@mui/icons-material';

const OregonUiToolbar = ({ onBack, title, uiVariant, onUiVariantChange, styles }) => (
  <Box
    sx={{
      display: 'flex',
      alignItems: 'center',
      gap: 1,
      mb: 2,
      flexWrap: 'wrap',
      justifyContent: 'space-between',
    }}
  >
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0 }}>
      <IconButton onClick={onBack} size="small" color="inherit" aria-label="Back">
        <ArrowBack />
      </IconButton>
      <Typography
        variant="h6"
        sx={{
          fontFamily: styles.isBbs ? styles.mono : undefined,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
      >
        {title}
      </Typography>
    </Box>
    <ToggleButtonGroup
      size="small"
      value={uiVariant}
      exclusive
      onChange={(_, v) => {
        if (v) onUiVariantChange(v);
      }}
      aria-label="Oregon Trail display style"
    >
      <ToggleButton value="app">App</ToggleButton>
      <ToggleButton value="bbs" sx={styles.isBbs ? { fontFamily: styles.mono } : undefined}>
        Terminal
      </ToggleButton>
    </ToggleButtonGroup>
  </Box>
);

export default OregonUiToolbar;
