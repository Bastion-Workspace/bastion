import React from 'react';
import { Box } from '@mui/material';

/**
 * Component for rendering embedded content (YouTube videos, Rumble videos, etc.)
 */
const EmbeddedContent = ({ embedType, embedData }) => {
  if (embedType === 'youtube' && embedData?.videoId) {
    return (
      <Box
        sx={{
          position: 'relative',
          width: '100%',
          maxWidth: '100%',
          mb: 2,
          mt: 2,
          '&::before': {
            content: '""',
            display: 'block',
            paddingTop: '56.25%' // 16:9 aspect ratio
          }
        }}
      >
        <Box
          component="iframe"
          src={`https://www.youtube.com/embed/${embedData.videoId}`}
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            border: 'none',
            borderRadius: 1
          }}
        />
      </Box>
    );
  }
  
  if (embedType === 'rumble' && embedData?.videoId) {
    return (
      <Box
        sx={{
          position: 'relative',
          width: '100%',
          maxWidth: '100%',
          mb: 2,
          mt: 2,
          '&::before': {
            content: '""',
            display: 'block',
            paddingTop: '56.25%' // 16:9 aspect ratio
          }
        }}
      >
        <Box
          component="iframe"
          src={`https://rumble.com/embed/v${embedData.videoId}/`}
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            border: 'none',
            borderRadius: 1
          }}
        />
      </Box>
    );
  }
  
  // Fallback for unsupported embed types
  return null;
};

export default EmbeddedContent;

