/**
 * DocX Document Viewer
 * Displays Microsoft Word (.docx) files with formatting preserved
 * 
 * Features:
 * - Converts DocX to HTML using mammoth.js
 * - Preserves formatting (bold, italic, headings, lists, tables)
 * - Theme-aware styling (light/dark mode)
 * - Download functionality
 * - Responsive layout
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Paper,
  IconButton,
  Typography,
  Tooltip,
  CircularProgress,
  Alert,
  Stack,
  Divider
} from '@mui/material';
import {
  Description,
  Download
} from '@mui/icons-material';
import mammoth from 'mammoth';
import DOMPurify from 'dompurify';
import { useTheme } from '../contexts/ThemeContext';
import { useImageLightbox } from './common/ImageLightbox';

const DocxViewer = ({ documentId, filename }) => {
  const [htmlContent, setHtmlContent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { darkMode } = useTheme();
  const contentRef = useRef(null);
  const { openLightbox } = useImageLightbox();

  // Fetch and convert DocX file
  useEffect(() => {
    const fetchAndConvertDocx = async () => {
      try {
        setLoading(true);
        setError(null);

        const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
        if (!token) {
          throw new Error('Authentication token not found');
        }

        // Fetch the DocX file as array buffer
        const response = await fetch(`/api/documents/${documentId}/file`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const arrayBuffer = await response.arrayBuffer();

        // Convert DocX to HTML using mammoth
        const result = await mammoth.convertToHtml(
          { arrayBuffer: arrayBuffer },
          {
            styleMap: [
              "p[style-name='Heading 1'] => h1:fresh",
              "p[style-name='Heading 2'] => h2:fresh",
              "p[style-name='Heading 3'] => h3:fresh",
              "p[style-name='Heading 4'] => h4:fresh",
              "p[style-name='Heading 5'] => h5:fresh",
              "p[style-name='Heading 6'] => h6:fresh"
            ]
          }
        );

        // Sanitize the HTML content
        const sanitizedHtml = DOMPurify.sanitize(result.value, {
          ALLOWED_TAGS: [
            'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'strong', 'em', 'u', 's', 'b', 'i',
            'ul', 'ol', 'li',
            'table', 'thead', 'tbody', 'tr', 'th', 'td',
            'br', 'hr',
            'a', 'img',
            'span', 'div', 'blockquote', 'pre', 'code'
          ],
          ALLOWED_ATTR: ['href', 'src', 'alt', 'title', 'class', 'style']
        });

        setHtmlContent(sanitizedHtml);

        // Log warnings if any
        if (result.messages && result.messages.length > 0) {
          console.warn('DocX conversion warnings:', result.messages);
        }
      } catch (err) {
        console.error('Failed to load or convert DocX file:', err);
        setError(`Failed to load DocX document: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };

    if (documentId) {
      fetchAndConvertDocx();
    }
  }, [documentId]);

  // Add click handlers to embedded images
  useEffect(() => {
    if (htmlContent && contentRef.current) {
      const handleImageClick = (e) => {
        if (e.target.tagName === 'IMG') {
          e.preventDefault();
          openLightbox(e.target.src, { alt: e.target.alt || 'Document image' });
        }
      };
      
      const container = contentRef.current;
      container.addEventListener('click', handleImageClick);
      
      return () => {
        container.removeEventListener('click', handleImageClick);
      };
    }
  }, [htmlContent, openLightbox]);

  // Download handler
  const handleDownload = useCallback(() => {
    const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
    
    // For authenticated downloads, fetch and create blob
    if (token) {
      fetch(`/api/documents/${documentId}/file`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
        .then(response => response.blob())
        .then(blob => {
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = filename || 'document.docx';
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);
        })
        .catch(err => {
          console.error('Failed to download file:', err);
        });
    } else {
      // Fallback for unauthenticated downloads
      const link = document.createElement('a');
      link.href = `/api/documents/${documentId}/file`;
      link.download = filename || 'document.docx';
      link.click();
    }
  }, [documentId, filename]);

  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: darkMode ? '#121212' : '#f5f5f5'
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
          borderBottom: '1px solid',
          borderColor: 'divider'
        }}
      >
        <Stack direction="row" spacing={1} alignItems="center">
          <Description color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            {filename || 'Word Document'}
          </Typography>
        </Stack>

        <Stack direction="row" spacing={1} alignItems="center">
          {/* Download */}
          <Tooltip title="Download DocX">
            <IconButton onClick={handleDownload} size="small" color="primary">
              <Download />
            </IconButton>
          </Tooltip>
        </Stack>
      </Paper>

      {/* Content Viewer */}
      <Box
        ref={contentRef}
        sx={{
          flex: 1,
          overflow: 'auto',
          display: 'flex',
          justifyContent: 'center',
          p: 3,
          backgroundColor: darkMode ? '#121212' : '#f5f5f5'
        }}
      >
        {error ? (
          <Alert severity="error" sx={{ maxWidth: 600, mt: 2 }}>
            {error}
          </Alert>
        ) : loading ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, mt: 4 }}>
            <CircularProgress size={60} />
            <Typography variant="body1" color="text.secondary">
              Loading Word document...
            </Typography>
          </Box>
        ) : htmlContent ? (
          <Paper
            elevation={3}
            sx={{
              maxWidth: '8.5in',
              width: '100%',
              minHeight: '11in',
              p: 4,
              backgroundColor: darkMode ? '#1e1e1e' : 'white',
              color: darkMode ? '#e0e0e0' : 'text.primary',
              '& h1': {
                fontSize: '2em',
                fontWeight: 'bold',
                marginTop: '1em',
                marginBottom: '0.5em',
                color: darkMode ? '#ffffff' : 'text.primary'
              },
              '& h2': {
                fontSize: '1.5em',
                fontWeight: 'bold',
                marginTop: '0.8em',
                marginBottom: '0.4em',
                color: darkMode ? '#ffffff' : 'text.primary'
              },
              '& h3': {
                fontSize: '1.25em',
                fontWeight: 'bold',
                marginTop: '0.6em',
                marginBottom: '0.3em',
                color: darkMode ? '#ffffff' : 'text.primary'
              },
              '& h4, & h5, & h6': {
                fontWeight: 'bold',
                marginTop: '0.5em',
                marginBottom: '0.25em',
                color: darkMode ? '#ffffff' : 'text.primary'
              },
              '& p': {
                marginTop: '0.5em',
                marginBottom: '0.5em',
                lineHeight: 1.6,
                color: darkMode ? '#e0e0e0' : 'text.primary'
              },
              '& ul, & ol': {
                marginLeft: '1.5em',
                marginTop: '0.5em',
                marginBottom: '0.5em',
                paddingLeft: '1em'
              },
              '& li': {
                marginTop: '0.25em',
                marginBottom: '0.25em',
                lineHeight: 1.6
              },
              '& table': {
                width: '100%',
                borderCollapse: 'collapse',
                marginTop: '1em',
                marginBottom: '1em',
                border: `1px solid ${darkMode ? '#444' : '#ddd'}`
              },
              '& th, & td': {
                border: `1px solid ${darkMode ? '#444' : '#ddd'}`,
                padding: '0.5em',
                textAlign: 'left'
              },
              '& th': {
                backgroundColor: darkMode ? '#333' : '#f5f5f5',
                fontWeight: 'bold'
              },
              '& strong, & b': {
                fontWeight: 'bold'
              },
              '& em, & i': {
                fontStyle: 'italic'
              },
              '& u': {
                textDecoration: 'underline'
              },
              '& blockquote': {
                borderLeft: `3px solid ${darkMode ? '#666' : '#ccc'}`,
                paddingLeft: '1em',
                marginLeft: '1em',
                marginTop: '0.5em',
                marginBottom: '0.5em',
                fontStyle: 'italic',
                color: darkMode ? '#aaa' : '#666'
              },
              '& a': {
                color: darkMode ? '#90caf9' : 'primary.main',
                textDecoration: 'none',
                '&:hover': {
                  textDecoration: 'underline'
                }
              },
              '& img': {
                maxWidth: '100%',
                height: 'auto',
                marginTop: '0.5em',
                marginBottom: '0.5em',
                cursor: 'pointer'
              }
            }}
          >
            <div
              dangerouslySetInnerHTML={{ __html: htmlContent }}
            />
          </Paper>
        ) : null}
      </Box>
    </Box>
  );
};

export default DocxViewer;
