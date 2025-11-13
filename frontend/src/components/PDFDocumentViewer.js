/**
 * PDF Document Viewer - Roosevelt's Cavalry Charge for PDF Display!
 * 
 * Features:
 * - Zoom controls (zoom in, out, fit width, fit page)
 * - Pan/scroll for large PDFs
 * - Multi-page navigation
 * - Download PDF
 * - Responsive layout
 */

import React, { useState, useCallback, useRef } from 'react';
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
  NavigateBefore,
  NavigateNext,
  Download,
  Description
} from '@mui/icons-material';

// Configure PDF.js worker from CDN
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

const PDFDocumentViewer = ({ documentId, filename }) => {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [scale, setScale] = useState(1.0);
  const [loading, setLoading] = useState(false);  // Start false to let Document component render and load
  const [error, setError] = useState(null);
  const [containerWidth, setContainerWidth] = useState(null);
  const containerRef = useRef(null);

  const [pdfData, setPdfData] = React.useState(null);

  // Fetch PDF with authentication
  React.useEffect(() => {
    console.log('PDFDocumentViewer mounted:', { documentId, filename });
    console.log('PDF URL will be:', `/api/documents/${documentId}/pdf`);
    
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
        console.log('PDF fetched successfully with auth');
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

  // PDF document load success handler
  const onDocumentLoadSuccess = useCallback(({ numPages }) => {
    setNumPages(numPages);
    setLoading(false);
    console.log(`📄 PDF loaded successfully - ${numPages} pages`);
  }, []);

  // PDF document load error handler
  const onDocumentLoadError = useCallback((error) => {
    console.error('❌ Failed to load PDF:', error);
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

  // Page navigation
  const handlePreviousPage = useCallback(() => {
    setPageNumber(prev => Math.max(prev - 1, 1));
  }, []);

  const handleNextPage = useCallback(() => {
    setPageNumber(prev => Math.min(prev + 1, numPages || prev));
  }, [numPages]);

  // Download PDF
  const handleDownload = useCallback(() => {
    const link = document.createElement('a');
    link.href = `/api/documents/${documentId}/pdf`;
    link.download = filename || 'document.pdf';
    link.click();
  }, [documentId, filename]);

  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#f5f5f5'
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

          {/* Page Navigation */}
          {numPages && numPages > 1 && (
            <>
              <Tooltip title="Previous Page">
                <span>
                  <IconButton 
                    onClick={handlePreviousPage} 
                    disabled={pageNumber <= 1}
                    size="small"
                  >
                    <NavigateBefore />
                  </IconButton>
                </span>
              </Tooltip>
              <Typography variant="body2" sx={{ minWidth: 80, textAlign: 'center' }}>
                Page {pageNumber} of {numPages}
              </Typography>
              <Tooltip title="Next Page">
                <span>
                  <IconButton 
                    onClick={handleNextPage} 
                    disabled={pageNumber >= numPages}
                    size="small"
                  >
                    <NavigateNext />
                  </IconButton>
                </span>
              </Tooltip>

              <Divider orientation="vertical" flexItem />
            </>
          )}

          {/* Download */}
          <Tooltip title="Download PDF">
            <IconButton onClick={handleDownload} size="small" color="primary">
              <Download />
            </IconButton>
          </Tooltip>
        </Stack>
      </Paper>

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
              sx={{
                backgroundColor: 'white',
                boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
                '& .react-pdf__Page': {
                  display: 'flex',
                  justifyContent: 'center'
                },
                '& .react-pdf__Page__canvas': {
                  maxWidth: '100%',
                  height: 'auto !important'
                }
              }}
            >
              <Page
                pageNumber={pageNumber}
                scale={scale}
                renderTextLayer={true}
                renderAnnotationLayer={true}
              />
            </Box>
          </Document>
        )}
      </Box>
    </Box>
  );
};

export default PDFDocumentViewer;


