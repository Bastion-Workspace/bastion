/**
 * PPTX Document Viewer
 * Displays PowerPoint (.pptx) files as HTML slides
 *
 * Features:
 * - Converts PPTX to HTML using @jvmr/pptx-to-html
 * - Slide-by-slide view with prev/next navigation
 * - Theme-aware styling (light/dark mode)
 * - Download functionality
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
  Stack
} from '@mui/material';
import {
  Slideshow,
  Download,
  NavigateBefore,
  NavigateNext
} from '@mui/icons-material';
import DOMPurify from 'dompurify';
import { useTheme } from '../contexts/ThemeContext';
import { useImageLightbox } from './common/ImageLightbox';

const PptxViewer = ({ documentId, filename }) => {
  const [slidesHtml, setSlidesHtml] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentSlide, setCurrentSlide] = useState(0);
  const { darkMode } = useTheme();
  const contentRef = useRef(null);
  const { openLightbox } = useImageLightbox();

  useEffect(() => {
    const fetchAndConvertPptx = async () => {
      try {
        setLoading(true);
        setError(null);

        const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
        if (!token) {
          throw new Error('Authentication token not found');
        }

        const response = await fetch(`/api/documents/${documentId}/file`, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const arrayBuffer = await response.arrayBuffer();

        const { pptxToHtml } = await import('@jvmr/pptx-to-html');
        const rawSlides = await pptxToHtml(arrayBuffer, {
          width: 960,
          height: 540,
          scaleToFit: true,
          letterbox: true
        });

        const sanitized = (rawSlides || []).map((slideHtml) =>
          DOMPurify.sanitize(slideHtml, {
            ALLOWED_TAGS: [
              'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
              'strong', 'em', 'u', 's', 'b', 'i',
              'ul', 'ol', 'li',
              'table', 'thead', 'tbody', 'tr', 'th', 'td',
              'br', 'hr',
              'a', 'img', 'svg', 'path', 'g',
              'span', 'div', 'blockquote'
            ],
            ALLOWED_ATTR: ['href', 'src', 'alt', 'title', 'class', 'style', 'viewBox', 'd', 'fill', 'stroke', 'transform']
          })
        );

        setSlidesHtml(sanitized);
        setCurrentSlide(0);
      } catch (err) {
        console.error('Failed to load or convert PPTX file:', err);
        setError(`Failed to load PowerPoint document: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };

    if (documentId) {
      fetchAndConvertPptx();
    }
  }, [documentId]);

  useEffect(() => {
    if (slidesHtml.length && contentRef.current) {
      const handleImageClick = (e) => {
        if (e.target.tagName === 'IMG') {
          e.preventDefault();
          openLightbox(e.target.src, { alt: e.target.alt || 'Slide image' });
        }
      };
      const container = contentRef.current;
      container.addEventListener('click', handleImageClick);
      return () => container.removeEventListener('click', handleImageClick);
    }
  }, [slidesHtml, openLightbox]);

  const handleDownload = useCallback(() => {
    const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
    const downloadFilename = filename || 'presentation.pptx';
    if (token) {
      fetch(`/api/documents/${documentId}/file`, {
        headers: { Authorization: `Bearer ${token}` }
      })
        .then((res) => res.blob())
        .then((blob) => {
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = downloadFilename;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);
        })
        .catch((err) => console.error('Failed to download file:', err));
    } else {
      const link = document.createElement('a');
      link.href = `/api/documents/${documentId}/file`;
      link.download = downloadFilename;
      link.click();
    }
  }, [documentId, filename]);

  const totalSlides = slidesHtml.length;
  const canPrev = totalSlides > 0 && currentSlide > 0;
  const canNext = totalSlides > 0 && currentSlide < totalSlides - 1;

  let content;
  if (error) {
    content = (
      <Alert severity="error" sx={{ maxWidth: 600, mt: 2 }}>
        {error}
      </Alert>
    );
  } else if (loading) {
    content = (
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, mt: 4 }}>
        <CircularProgress size={60} />
        <Typography variant="body1" color="text.secondary">
          Loading PowerPoint...
        </Typography>
      </Box>
    );
  } else if (totalSlides > 0 && slidesHtml[currentSlide]) {
    const slideHtml = slidesHtml[currentSlide];
    content = (
      <Paper
        elevation={3}
        sx={{
          width: '100%',
          maxWidth: 960,
          aspectRatio: '16/10',
          overflow: 'hidden',
          backgroundColor: darkMode ? '#1e1e1e' : 'white',
          color: darkMode ? '#e0e0e0' : 'text.primary',
          '& img': {
            maxWidth: '100%',
            height: 'auto',
            cursor: 'pointer'
          }
        }}
      >
        <Box
          sx={{
            width: '100%',
            height: '100%',
            overflow: 'auto',
            p: 2,
            '& .slide': {
              minHeight: '100%'
            }
          }}
          dangerouslySetInnerHTML={{ __html: slideHtml }}
        />
      </Paper>
    );
  } else if (totalSlides === 0 && !loading) {
    content = (
      <Typography color="text.secondary">No slides in this presentation.</Typography>
    );
  } else {
    content = null;
  }

  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: darkMode ? '#121212' : '#f5f5f5'
      }}
    >
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
          <Slideshow color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            {filename || 'PowerPoint'}
          </Typography>
        </Stack>

        <Stack direction="row" spacing={1} alignItems="center">
          {totalSlides > 0 && (
            <>
              <Tooltip title="Previous slide">
                <span>
                  <IconButton
                    onClick={() => setCurrentSlide((i) => Math.max(0, i - 1))}
                    disabled={!canPrev}
                    size="small"
                    color="primary"
                  >
                    <NavigateBefore />
                  </IconButton>
                </span>
              </Tooltip>
              <Typography variant="body2" color="text.secondary" sx={{ minWidth: 64, textAlign: 'center' }}>
                {currentSlide + 1} / {totalSlides}
              </Typography>
              <Tooltip title="Next slide">
                <span>
                  <IconButton
                    onClick={() => setCurrentSlide((i) => Math.min(totalSlides - 1, i + 1))}
                    disabled={!canNext}
                    size="small"
                    color="primary"
                  >
                    <NavigateNext />
                  </IconButton>
                </span>
              </Tooltip>
            </>
          )}
          <Tooltip title="Download PPTX">
            <IconButton onClick={handleDownload} size="small" color="primary">
              <Download />
            </IconButton>
          </Tooltip>
        </Stack>
      </Paper>

      <Box
        ref={contentRef}
        sx={{
          flex: 1,
          overflow: 'auto',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          p: 3,
          backgroundColor: darkMode ? '#121212' : '#f5f5f5'
        }}
      >
        {content}
      </Box>
    </Box>
  );
};

export default PptxViewer;
