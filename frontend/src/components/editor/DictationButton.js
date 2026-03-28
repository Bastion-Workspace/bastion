import React from 'react';
import { IconButton, Tooltip, Box, Typography } from '@mui/material';
import { Mic } from '@mui/icons-material';
import { useDictation } from '../../hooks/useDictation';

const DictationButton = ({ insertText, disabled = false, size = 'small' }) => {
  const {
    isDictating,
    startDictation,
    stopDictation,
    liveTranscript,
    segmentCount,
  } = useDictation();

  const handleClick = () => {
    if (isDictating) {
      stopDictation();
      return;
    }
    if (typeof insertText === 'function') {
      startDictation(insertText);
    }
  };

  const tooltipTitle = isDictating
    ? liveTranscript
      ? liveTranscript
      : 'Dictating... (click to stop)'
    : 'Start dictation';

  return (
    <Tooltip title={tooltipTitle} placement="bottom">
      <Box sx={{ position: 'relative', display: 'inline-flex' }}>
        <IconButton
          size={size}
          onClick={handleClick}
          disabled={disabled}
          color={isDictating ? 'error' : 'default'}
          sx={{
            backgroundColor: isDictating ? 'error.main' : 'action.hover',
            color: isDictating ? 'white' : 'inherit',
            '&:hover': {
              backgroundColor: isDictating ? 'error.dark' : 'action.selected',
            },
            ...(isDictating && {
              animation: 'pulse 1.5s ease-in-out infinite',
              '@keyframes pulse': {
                '0%, 100%': { opacity: 1 },
                '50%': { opacity: 0.7 },
              },
            }),
          }}
        >
          <Mic fontSize={size} />
        </IconButton>
        {segmentCount > 0 && isDictating && (
          <Typography
            component="span"
            variant="caption"
            sx={{
              position: 'absolute',
              top: -4,
              right: -4,
              minWidth: 18,
              height: 18,
              borderRadius: '50%',
              bgcolor: 'error.main',
              color: 'white',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '0.7rem',
            }}
          >
            {segmentCount}
          </Typography>
        )}
      </Box>
    </Tooltip>
  );
};

export default DictationButton;
