/**
 * Right-anchored MUI Drawer with a draggable left edge to resize width.
 * Width is persisted in localStorage under the given storageKey.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Drawer, Box } from '@mui/material';

function getStoredWidth(storageKey, minWidth, maxWidth) {
  try {
    const stored = localStorage.getItem(storageKey);
    if (stored == null) return null;
    const n = parseInt(stored, 10);
    if (!Number.isFinite(n)) return null;
    return Math.min(maxWidth, Math.max(minWidth, n));
  } catch {
    return null;
  }
}

function setStoredWidth(storageKey, width) {
  try {
    localStorage.setItem(storageKey, String(width));
  } catch (_) {}
}

export default function ResizableDrawer({
  open,
  onClose,
  storageKey,
  defaultWidth = 400,
  minWidth = 320,
  maxWidth = 900,
  children,
  zIndex = 1400,
  bottomOffset = 32,
}) {
  const [drawerWidth, setDrawerWidth] = useState(() => getStoredWidth(storageKey, minWidth, maxWidth) ?? defaultWidth);
  const [isResizing, setIsResizing] = useState(false);
  const drawerWidthRef = useRef(drawerWidth);

  useEffect(() => {
    drawerWidthRef.current = drawerWidth;
  }, [drawerWidth]);

  const handleResizeStart = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsResizing(true);
      const onMove = (ev) => {
        const x = ev.clientX != null ? ev.clientX : ev.touches?.[0]?.clientX;
        if (x == null) return;
        const cap = Math.min(maxWidth, (window.innerWidth || 1024) * 0.95);
        setDrawerWidth(Math.round(Math.min(cap, Math.max(minWidth, window.innerWidth - x))));
      };
      const onEnd = () => {
        setIsResizing(false);
        setStoredWidth(storageKey, drawerWidthRef.current);
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onEnd);
        window.removeEventListener('touchmove', onMove, { passive: true });
        window.removeEventListener('touchend', onEnd);
      };
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onEnd);
      window.addEventListener('touchmove', onMove, { passive: true });
      window.addEventListener('touchend', onEnd);
    },
    [storageKey, minWidth, maxWidth]
  );

  useEffect(() => {
    if (!isResizing) return;
    const prevUserSelect = document.body.style.userSelect;
    const prevCursor = document.body.style.cursor;
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'col-resize';
    return () => {
      document.body.style.userSelect = prevUserSelect;
      document.body.style.cursor = prevCursor;
    };
  }, [isResizing]);

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      disablePortal={false}
      ModalProps={{
        container: typeof document !== 'undefined' ? document.body : undefined,
        style: { zIndex },
      }}
      PaperProps={{
        sx: {
          width: drawerWidth,
          minWidth: minWidth,
          maxWidth: '95vw',
          position: 'fixed',
          top: 0,
          right: 0,
          left: 'auto',
          bottom: bottomOffset,
          height: bottomOffset ? `calc(100vh - ${bottomOffset}px)` : '100vh',
          maxHeight: 'none',
          display: 'flex',
          flexDirection: 'column',
        },
      }}
    >
      <Box
        role="separator"
        aria-valuenow={drawerWidth}
        aria-valuemin={minWidth}
        aria-valuemax={maxWidth}
        aria-label="Resize drawer"
        onMouseDown={handleResizeStart}
        onTouchStart={handleResizeStart}
        sx={{
          position: 'absolute',
          left: 0,
          top: 0,
          bottom: 0,
          width: 10,
          cursor: 'col-resize',
          zIndex: 10,
          touchAction: 'none',
          '&:hover': { bgcolor: 'action.hover' },
          '&::after': {
            content: '""',
            position: 'absolute',
            left: '50%',
            top: '50%',
            transform: 'translate(-50%, -50%)',
            width: 2,
            height: 40,
            borderRadius: 1,
            bgcolor: 'divider',
            opacity: 0.6,
          },
        }}
      />
      {children}
    </Drawer>
  );
}
