import React from 'react';
import {
  Box,
  Typography,
  IconButton,
  Tooltip,
  Button,
} from '@mui/material';
import {
  CloudUpload,
  Fullscreen,
} from '@mui/icons-material';

/**
 * Safely renders standalone Plotly HTML in an iframe.
 * Supports fullscreen and Import to Library (static SVG/PNG).
 */
const ChartRenderer = ({ html, staticData, staticFormat, onImport, onFullScreen }) => {
  return (
    <Box
      sx={{
        my: 2,
        width: '100%',
        overflow: 'hidden',
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
        backgroundColor: '#fff',
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      {/* Action header */}
      <Box sx={{
        p: 1,
        borderBottom: '1px solid',
        borderColor: 'divider',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        bgcolor: 'rgba(0, 0, 0, 0.02)'
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
            Interactive Visualization
          </Typography>
          <Tooltip title="View Full Screen">
            <IconButton size="small" onClick={() => onFullScreen(html)}>
              <Fullscreen fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
        {staticData && (
          <Button
            size="small"
            variant="text"
            startIcon={<CloudUpload fontSize="small" />}
            onClick={() => onImport(staticData, staticFormat)}
            sx={{ textTransform: 'none', py: 0 }}
          >
            Import to Library
          </Button>
        )}
      </Box>

      {/* Interactive Iframe */}
      <Box sx={{ height: '500px', width: '100%' }}>
        <iframe
          srcDoc={html}
          title="Visualization"
          width="100%"
          height="100%"
          style={{ border: 'none' }}
          sandbox="allow-scripts allow-same-origin"
        />
      </Box>
    </Box>
  );
};

export default ChartRenderer;
