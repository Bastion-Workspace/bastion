/**
 * OPDS / in-memory PDF viewer: renders bytes via react-pdf (no document API route).
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import { Box, IconButton, Typography, CircularProgress, Stack } from '@mui/material';
import { ChevronLeft, ChevronRight } from '@mui/icons-material';

pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

function toStandaloneArrayBuffer(data) {
  if (!data) return null;
  if (data instanceof ArrayBuffer) {
    return data.byteLength === 0 ? data : data.slice(0);
  }
  if (ArrayBuffer.isView(data)) {
    const u = new Uint8Array(data.byteLength);
    u.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    return u.buffer;
  }
  return null;
}

export default function PdfBytesViewer({ rawBytes, onPageLabelChange, onReadyChange }) {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [scale, setScale] = useState(1.0);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [containerWidth, setContainerWidth] = useState(null);
  const containerRef = useRef(null);
  const blobUrlRef = useRef(null);

  const file = useMemo(() => {
    const buf = toStandaloneArrayBuffer(rawBytes);
    if (!buf || buf.byteLength === 0) return null;
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }
    const blob = new Blob([buf], { type: 'application/pdf' });
    const url = URL.createObjectURL(blob);
    blobUrlRef.current = url;
    return url;
  }, [rawBytes]);

  useEffect(() => {
    return () => {
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    const updateWidth = () => {
      if (containerRef.current) {
        setContainerWidth(Math.max(200, containerRef.current.offsetWidth - 24));
      }
    };
    updateWidth();
    window.addEventListener('resize', updateWidth);
    let ro;
    if (el && typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(() => updateWidth());
      ro.observe(el);
    }
    return () => {
      window.removeEventListener('resize', updateWidth);
      if (ro) ro.disconnect();
    };
  }, []);

  useEffect(() => {
    if (typeof onPageLabelChange === 'function') {
      if (numPages && pageNumber) {
        onPageLabelChange(`Page ${pageNumber} of ${numPages}`);
      } else {
        onPageLabelChange('—');
      }
    }
  }, [numPages, pageNumber, onPageLabelChange]);

  const onDocumentLoadSuccess = useCallback(
    ({ numPages: n }) => {
      setNumPages(n);
      setPageNumber(1);
      setLoading(false);
      setError(null);
      if (typeof onReadyChange === 'function') onReadyChange(true);
    },
    [onReadyChange]
  );

  const onDocumentLoadError = useCallback(
    (err) => {
      setError(err?.message || 'Failed to load PDF');
      setLoading(false);
      if (typeof onReadyChange === 'function') onReadyChange(false);
    },
    [onReadyChange]
  );

  useEffect(() => {
    const onKey = (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setPageNumber((p) => Math.max(1, p - 1));
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        setPageNumber((p) => (numPages ? Math.min(numPages, p + 1) : p));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [numPages]);

  const goPrev = useCallback(() => {
    setPageNumber((p) => Math.max(1, p - 1));
  }, []);

  const goNext = useCallback(() => {
    setPageNumber((p) => (numPages ? Math.min(numPages, p + 1) : p));
  }, [numPages]);

  if (!file) {
    return (
      <Box p={2}>
        <Typography color="error">No PDF data</Typography>
      </Box>
    );
  }

  return (
    <Box
      ref={containerRef}
      sx={{
        flex: 1,
        minHeight: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        overflow: 'auto',
        bgcolor: 'grey.100',
        position: 'relative',
      }}
    >
      {error && (
        <Typography color="error" sx={{ p: 2 }}>
          {error}
        </Typography>
      )}
      {loading && !error && (
        <Box sx={{ py: 4 }}>
          <CircularProgress size={36} />
        </Box>
      )}
      <Document file={file} onLoadSuccess={onDocumentLoadSuccess} onLoadError={onDocumentLoadError} loading="">
        {numPages ? (
          <Page
            pageNumber={pageNumber}
            width={containerWidth || 600}
            scale={scale}
            renderTextLayer
            renderAnnotationLayer
          />
        ) : null}
      </Document>

      {numPages ? (
        <Stack
          direction="row"
          alignItems="center"
          spacing={2}
          sx={{
            position: 'sticky',
            bottom: 0,
            py: 1,
            px: 2,
            bgcolor: 'rgba(0,0,0,0.65)',
            width: '100%',
            justifyContent: 'center',
          }}
        >
          <IconButton size="small" sx={{ color: '#fff' }} onClick={goPrev} disabled={pageNumber <= 1} aria-label="Previous page">
            <ChevronLeft />
          </IconButton>
          <Typography variant="body2" sx={{ color: '#fff', minWidth: 120, textAlign: 'center' }}>
            {pageNumber} / {numPages}
          </Typography>
          <IconButton
            size="small"
            sx={{ color: '#fff' }}
            onClick={goNext}
            disabled={pageNumber >= numPages}
            aria-label="Next page"
          >
            <ChevronRight />
          </IconButton>
          <IconButton size="small" sx={{ color: '#fff' }} onClick={() => setScale((s) => Math.min(2.5, s + 0.15))} aria-label="Zoom in">
            +
          </IconButton>
          <IconButton size="small" sx={{ color: '#fff' }} onClick={() => setScale((s) => Math.max(0.5, s - 0.15))} aria-label="Zoom out">
            −
          </IconButton>
        </Stack>
      ) : null}

    </Box>
  );
}
