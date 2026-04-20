/**
 * PDF document viewer: zoom, continuous scroll, download, in-document find.
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import {
  Box,
  Paper,
  IconButton,
  Typography,
  Tooltip,
  CircularProgress,
  Alert,
  Stack,
  Divider,
  ButtonGroup
} from '@mui/material';
import {
  ZoomIn,
  ZoomOut,
  ZoomOutMap,
  FitScreen,
  Download,
  Description,
  Search as SearchIcon,
} from '@mui/icons-material';
import FindInDocumentBar from './FindInDocumentBar';

// Configure PDF.js worker from CDN
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

const PDFDocumentViewer = ({ documentId, filename, onTextExtracted }) => {
  const [numPages, setNumPages] = useState(null);
  const [scale, setScale] = useState(1.0);
  const [loading, setLoading] = useState(false);  // Start false to let Document component render and load
  const [error, setError] = useState(null);
  const [containerWidth, setContainerWidth] = useState(null);
  const containerRef = useRef(null);
  const pdfFindRootRef = useRef(null);
  const pdfViewerRootRef = useRef(null);
  const [findOpen, setFindOpen] = useState(false);

  const [pdfData, setPdfData] = React.useState(null);

  // Fetch PDF with authentication
  React.useEffect(() => {
    const fetchPdfWithAuth = async () => {
      try {
        const token = localStorage.getItem('auth_token');
        if (!token) {
          console.error('No auth token available');
          setError('Authentication token not found');
          return;
        }
        
        const response = await fetch(`/api/documents/${documentId}/pdf`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setPdfData(url);
      } catch (error) {
        console.error('Failed to fetch PDF with auth:', error);
        setError('Failed to load PDF document. Authentication may have failed.');
        setLoading(false);
      }
    };
    
    fetchPdfWithAuth();
  }, [documentId, filename]);
  
  // Cleanup: revoke the blob URL when component unmounts or pdfData changes
  React.useEffect(() => {
    return () => {
      if (pdfData) {
        URL.revokeObjectURL(pdfData);
      }
    };
  }, [pdfData]);

  // Update container width on mount and resize
  React.useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.offsetWidth - 40); // Subtract padding
      }
    };

    updateWidth();
    window.addEventListener('resize', updateWidth);
    return () => window.removeEventListener('resize', updateWidth);
  }, []);

  // Extract plain text for agent / {editor} read-only context (client-side pdf.js)
  React.useEffect(() => {
    let cancelled = false;
    const run = async () => {
      if (!pdfData || !numPages || typeof onTextExtracted !== 'function') return;
      try {
        const loadingTask = pdfjs.getDocument(pdfData);
        const pdf = await loadingTask.promise;
        const parts = [];
        for (let i = 1; i <= numPages; i++) {
          const page = await pdf.getPage(i);
          const textContent = await page.getTextContent();
          const line = textContent.items
            .map((item) => (item && typeof item.str === 'string' ? item.str : ''))
            .join(' ');
          parts.push(line);
        }
        const fullText = parts.filter(Boolean).join('\n\n');
        if (!cancelled && fullText.trim()) {
          onTextExtracted(fullText);
        }
      } catch (e) {
        console.warn('PDF text extraction failed:', e);
      }
    };
    run();
    return () => {
      cancelled = true;
    };
  }, [pdfData, numPages, onTextExtracted]);

  // PDF document load success handler
  const onDocumentLoadSuccess = useCallback(({ numPages }) => {
    setNumPages(numPages);
    setLoading(false);
  }, []);

  // PDF document load error handler
  const onDocumentLoadError = useCallback((error) => {
    console.error('Failed to load PDF:', error);
    setError('Failed to load PDF document. The file may be corrupted or unavailable.');
    setLoading(false);
  }, []);

  // Zoom controls
  const handleZoomIn = useCallback(() => {
    setScale(prev => Math.min(prev + 0.25, 3.0));
  }, []);

  const handleZoomOut = useCallback(() => {
    setScale(prev => Math.max(prev - 0.25, 0.5));
  }, []);

  const handleFitWidth = useCallback(() => {
    if (containerWidth) {
      // Scale to fit container width (accounting for padding)
      setScale(containerWidth / 612); // 612 is default PDF page width in points
    }
  }, [containerWidth]);

  const handleFitPage = useCallback(() => {
    setScale(1.0);
  }, []);

  const handleResetZoom = useCallback(() => {
    setScale(1.0);
  }, []);

  // Download PDF
  const handleDownload = useCallback(() => {
    const link = document.createElement('a');
    link.href = `/api/documents/${documentId}/pdf`;
    link.download = filename || 'document.pdf';
    link.click();
  }, [documentId, filename]);

  useEffect(() => {
    const onKeyDown = (e) => {
      const root = pdfViewerRootRef.current;
      const t = e.target;
      if (!root || !(t instanceof Node) || !root.contains(t)) return;
      if ((e.ctrlKey || e.metaKey) && (e.key === 'f' || e.key === 'F')) {
        e.preventDefault();
        setFindOpen(true);
      }
    };
    document.addEventListener('keydown', onKeyDown, true);
    return () => document.removeEventListener('keydown', onKeyDown, true);
  }, []);

  return (
    <Box
      ref={pdfViewerRootRef}
      tabIndex={-1}
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#f5f5f5',
        outline: 'none',
        minHeight: 0,
      }}
    >
      {/* Toolbar */}
      <Paper
        elevation={2}
        sx={{
          p: 1.5,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderRadius: 0,
          borderBottom: '1px solid #e0e0e0'
        }}
      >
        <Stack direction="row" spacing={1} alignItems="center">
          <Description color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            {filename || 'PDF Document'}
          </Typography>
        </Stack>

        <Stack direction="row" spacing={1} alignItems="center">
          {/* Zoom Controls */}
          <ButtonGroup variant="outlined" size="small">
            <Tooltip title="Zoom Out">
              <span>
                <IconButton 
                  onClick={handleZoomOut} 
                  disabled={scale <= 0.5}
                  size="small"
                >
                  <ZoomOut fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>
            <Tooltip title="Reset Zoom (100%)">
              <IconButton onClick={handleResetZoom} size="small">
                <Typography variant="body2" sx={{ minWidth: 45, fontWeight: 600 }}>
                  {Math.round(scale * 100)}%
                </Typography>
              </IconButton>
            </Tooltip>
            <Tooltip title="Zoom In">
              <span>
                <IconButton 
                  onClick={handleZoomIn} 
                  disabled={scale >= 3.0}
                  size="small"
                >
                  <ZoomIn fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>
          </ButtonGroup>

          <Divider orientation="vertical" flexItem />

          {/* Fit Controls */}
          <Tooltip title="Fit Width">
            <IconButton onClick={handleFitWidth} size="small">
              <ZoomOutMap fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Fit Page (100%)">
            <IconButton onClick={handleFitPage} size="small">
              <FitScreen fontSize="small" />
            </IconButton>
          </Tooltip>

          <Divider orientation="vertical" flexItem />

          {/* Page Count Display */}
          {numPages && (
            <Typography variant="body2" sx={{ minWidth: 80, textAlign: 'center', color: 'text.secondary' }}>
              {numPages} {numPages === 1 ? 'page' : 'pages'}
            </Typography>
          )}

          <Divider orientation="vertical" flexItem />

          <Tooltip title="Find in document (Ctrl+F)">
            <IconButton
              onClick={() => setFindOpen((o) => !o)}
              size="small"
              color={findOpen ? 'primary' : 'default'}
              aria-label="Find in document"
            >
              <SearchIcon />
            </IconButton>
          </Tooltip>

          {/* Download */}
          <Tooltip title="Download PDF">
            <IconButton onClick={handleDownload} size="small" color="primary">
              <Download />
            </IconButton>
          </Tooltip>
        </Stack>
      </Paper>

      {findOpen && (
        <FindInDocumentBar
          containerRef={pdfFindRootRef}
          open={findOpen}
          onClose={() => setFindOpen(false)}
        />
      )}

      {/* PDF Viewer */}
      <Box
        ref={containerRef}
        sx={{
          flex: 1,
          overflow: 'auto',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'flex-start',
          p: 2,
          backgroundColor: '#e0e0e0'
        }}
      >
        {error ? (
          <Alert severity="error" sx={{ maxWidth: 600 }}>
            {error}
          </Alert>
        ) : !pdfData ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, mt: 4 }}>
            <CircularProgress size={60} />
            <Typography variant="body1" color="text.secondary">
              Loading PDF document...
            </Typography>
          </Box>
        ) : (
          <Document
            file={pdfData}
            onLoadSuccess={onDocumentLoadSuccess}
            onLoadError={onDocumentLoadError}
            loading={
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, mt: 4 }}>
                <CircularProgress size={60} />
                <Typography variant="body1" color="text.secondary">
                  Loading PDF document...
                </Typography>
              </Box>
            }
            error={
              <Alert severity="error" sx={{ maxWidth: 600 }}>
                Failed to load PDF document. The file may be corrupted or unavailable.
              </Alert>
            }
          >
            <Box
              ref={pdfFindRootRef}
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 2,
                '& .react-pdf__Page': {
                  display: 'flex',
                  justifyContent: 'center',
                  backgroundColor: 'white',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
                  marginBottom: 2
                },
                '& .react-pdf__Page__canvas': {
                  maxWidth: '100%',
                  height: 'auto !important'
                }
              }}
            >
              {Array.from(new Array(numPages), (el, index) => (
                <Page
                  key={`page_${index + 1}`}
                  pageNumber={index + 1}
                  scale={scale}
                  renderTextLayer={true}
                  renderAnnotationLayer={true}
                  width={containerWidth || undefined}
                />
              ))}
            </Box>
          </Document>
        )}
      </Box>
    </Box>
  );
};

export default PDFDocumentViewer;


