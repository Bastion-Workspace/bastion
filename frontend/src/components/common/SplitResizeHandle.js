import React from 'react';
import { Box } from '@mui/material';

/**
 * Vertical split drag handle for resizable side panels (document library, AI chat, etc.).
 * Use edge="trailing" on the right edge of a left column; edge="leading" on the left edge of a right column.
 */
const SplitResizeHandle = ({
  onMouseDown,
  isResizing = false,
  edge = 'trailing',
  sx = {},
}) => {
  const isTrailing = edge === 'trailing';

  return (
    <Box
      onMouseDown={onMouseDown}
      aria-hidden
      sx={{
        position: 'absolute',
        top: 0,
        bottom: 0,
        width: 6,
        cursor: 'ew-resize',
        zIndex: 2,
        userSelect: 'none',
        touchAction: 'none',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        ...(isTrailing
          ? { right: 0 }
          : { left: 0 }),
        backgroundColor: isResizing ? 'action.selected' : 'transparent',
        transition: 'background-color 0.15s ease',
        '&:hover': {
          backgroundColor: 'action.hover',
        },
        '&:active': {
          backgroundColor: 'action.selected',
        },
        '&::after': {
          content: '""',
          width: 3,
          height: 56,
          borderRadius: '2px',
          backgroundColor: 'primary.main',
          opacity: isResizing ? 0.55 : 0.3,
          transition: 'opacity 0.15s ease',
        },
        '&:hover::after': {
          opacity: 0.75,
        },
        ...sx,
      }}
    />
  );
};

export default SplitResizeHandle;
