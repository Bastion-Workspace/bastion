import React, { useEffect, useState } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';

/**
 * Renders a document /file URL with JWT (img src cannot send Authorization).
 */
export function AuthDocumentFileImage({ documentId, alt, sx }) {
  const [blobUrl, setBlobUrl] = useState(null);
  const [error, setError] = useState(false);

  const url = documentId
    ? `/api/documents/${encodeURIComponent(documentId)}/file`
    : null;

  useEffect(() => {
    if (!url) {
      setBlobUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return null;
      });
      return undefined;
    }
    const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
    let objectUrl = null;
    let cancelled = false;
    setError(false);
    setBlobUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });

    fetch(url, { headers: token ? { Authorization: `Bearer ${token}` } : {} })
      .then((res) => {
        if (!res.ok) throw new Error(String(res.status));
        return res.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setBlobUrl(objectUrl);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [url]);

  if (!documentId) return null;
  if (error) {
    return (
      <Box sx={{ py: 2, px: 1, bgcolor: 'action.hover', borderRadius: 1, ...sx }}>
        <Typography variant="caption" color="text.secondary">
          Could not load image
        </Typography>
      </Box>
    );
  }
  if (!blobUrl) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: 160,
          bgcolor: 'action.hover',
          borderRadius: 1,
          ...sx,
        }}
      >
        <CircularProgress size={32} />
      </Box>
    );
  }
  return (
    <Box
      component="img"
      src={blobUrl}
      alt={alt || 'Document'}
      sx={{
        maxWidth: '100%',
        maxHeight: 360,
        width: 'auto',
        height: 'auto',
        objectFit: 'contain',
        display: 'block',
        mx: 'auto',
        borderRadius: 1,
        ...sx,
      }}
    />
  );
}
