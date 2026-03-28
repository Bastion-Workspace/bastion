import React, { useState, useCallback, useEffect } from 'react';
import { Box, Drawer } from '@mui/material';

function readStoredWidth(storageKey, minWidth, maxWidth, fallback) {
  if (typeof window === 'undefined') return fallback;
  if (!storageKey) return fallback;
  try {
    const v = parseInt(localStorage.getItem(storageKey), 10);
    if (!Number.isNaN(v) && v >= minWidth) return Math.min(v, maxWidth);
  } catch {
    /* ignore */
  }
  return fallback;
}

/**
 * Right-anchored Drawer with a draggable left edge to change width.
 * Persists width to localStorage when storageKey is set.
 */
export default function ResizableRightDrawer({
  open,
  onClose,
  children,
  defaultWidth = 360,
  minWidth = 260,
  maxWidthFraction = 0.92,
  storageKey = null,
  ModalProps,
  PaperProps,
  ...drawerProps
}) {
  const maxW = () => (typeof window !== 'undefined' ? window.innerWidth * maxWidthFraction : defaultWidth * 2);

  const [width, setWidth] = useState(() =>
    readStoredWidth(storageKey, minWidth, maxW(), defaultWidth)
  );

  useEffect(() => {
    if (!open) return;
    const onResize = () => {
      setWidth((w) => Math.min(w, maxW()));
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [open]);

  const startResize = useCallback(
    (event) => {
      event.preventDefault();
      const getX = (e) => (e.touches && e.touches[0] ? e.touches[0].clientX : e.clientX);
      const startX = getX(event);
      const startW = width;
      let latest = startW;

      const onMove = (e) => {
        if (e.type === 'touchmove') {
          e.preventDefault();
        }
        const x = getX(e);
        const cap = maxW();
        latest = Math.max(minWidth, Math.min(cap, startW + (startX - x)));
        setWidth(latest);
      };

      const onUp = () => {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        document.removeEventListener('touchmove', onMove);
        document.removeEventListener('touchend', onUp);
        document.removeEventListener('touchcancel', onUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        if (storageKey) {
          try {
            localStorage.setItem(storageKey, String(latest));
          } catch {
            /* ignore */
          }
        }
      };

      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
      document.addEventListener('touchmove', onMove, { passive: false });
      document.addEventListener('touchcancel', onUp);
      document.addEventListener('touchend', onUp);
    },
    [width, minWidth, maxWidthFraction, storageKey]
  );

  const paperSx = {
    width,
    maxWidth: '100vw',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    ...(PaperProps && typeof PaperProps.sx === 'object' && !Array.isArray(PaperProps.sx) ? PaperProps.sx : {}),
  };

  const { sx: _omit, ...paperRest } = PaperProps || {};

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      ModalProps={ModalProps}
      PaperProps={{
        ...paperRest,
        sx: paperSx,
      }}
      {...drawerProps}
    >
      <Box
        sx={{
          position: 'relative',
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
          minWidth: 0,
        }}
      >
        <Box
          onMouseDown={startResize}
          onTouchStart={startResize}
          aria-label="Resize panel"
          title="Drag to resize"
          sx={{
            position: 'absolute',
            left: 0,
            top: 0,
            bottom: 0,
            width: 10,
            marginLeft: '-5px',
            cursor: 'col-resize',
            zIndex: 10,
            touchAction: 'none',
            '&:hover': {
              backgroundColor: 'action.hover',
            },
          }}
        />
        <Box sx={{ flex: 1, overflow: 'auto', minHeight: 0 }}>{children}</Box>
      </Box>
    </Drawer>
  );
}
