import React, { useState, useRef, useCallback, isValidElement, Children } from 'react';
import { Box, IconButton, Tooltip, Alert, Typography, CircularProgress } from '@mui/material';
import { ContentCopy, Check } from '@mui/icons-material';
import DOMPurify from 'dompurify';
import { useTheme } from '../../contexts/ThemeContext';
import { useMermaidSvg } from '../../hooks/useMermaidSvg';

/**
 * MUI sx fragment for markdown preview containers (DocumentViewer).
 * Uses theme.palette.mode for code/pre/blockquote surfaces.
 */
export function markdownPreviewContainerSx(theme) {
  const isDark = theme.palette.mode === 'dark';
  return {
    '& h1, & h2, & h3, & h4, & h5, & h6': { mt: 2, mb: 1, fontWeight: 'bold' },
    '& p': { mb: 1.5, lineHeight: 1.6 },
    '& img': { maxWidth: '100%', height: 'auto', borderRadius: 1, my: 2 },
    '& a': { color: 'primary.main', textDecoration: 'none', '&:hover': { textDecoration: 'underline' } },
    '& blockquote': {
      borderLeft: 3,
      borderColor: 'primary.main',
      pl: 2,
      ml: 0,
      my: 2,
      backgroundColor: isDark ? 'rgba(255, 255, 255, 0.06)' : theme.palette.grey[100],
      py: 1,
      pr: 2,
    },
    '& code': {
      backgroundColor: isDark ? 'rgba(255, 255, 255, 0.12)' : theme.palette.grey[200],
      px: 0.5,
      py: 0.25,
      borderRadius: 0.5,
      fontFamily: 'monospace',
      fontSize: '0.875em',
    },
    '& pre': {
      position: 'relative',
      backgroundColor: isDark ? 'rgba(255, 255, 255, 0.08)' : theme.palette.grey[200],
      p: 2,
      pt: 5,
      borderRadius: 1,
      overflow: 'auto',
      border: isDark ? `1px solid ${theme.palette.divider}` : 'none',
      '& code': {
        backgroundColor: 'transparent',
        p: 0,
        fontSize: '0.875rem',
      },
    },
    '& ul, & ol': { pl: 3, mb: 1.5 },
    '& li': { mb: 0.5 },
    '& strong': { fontWeight: 'bold' },
    '& em': { fontStyle: 'italic' },
    '& details': { mb: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1, p: 1 },
    '& summary': { cursor: 'pointer', fontWeight: 'medium', py: 1, '&:hover': { opacity: 0.8 } },
    '& table': {
      borderCollapse: 'collapse',
      width: '100%',
      maxWidth: '100%',
      my: 1.5,
    },
    '& th, & td': {
      border: '1px solid',
      borderColor: theme.palette.divider,
      padding: '10px 14px',
      verticalAlign: 'middle',
      lineHeight: 1.5,
    },
    '& thead th': {
      fontWeight: 600,
      backgroundColor: isDark ? 'rgba(255, 255, 255, 0.06)' : theme.palette.grey[100],
      borderBottomWidth: 2,
    },
    '& .markdown-mermaid-block': { my: 2 },
  };
}

function collectCodeText(node) {
  if (node == null) return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(collectCodeText).join('');
  if (isValidElement(node)) return collectCodeText(node.props?.children);
  return '';
}

/** @returns {{ source: string } | null} */
function tryParseMermaidFence(children) {
  const arr = Children.toArray(children);
  if (arr.length !== 1 || !isValidElement(arr[0])) return null;
  const codeEl = arr[0];
  if (codeEl.type !== 'code') return null;
  const className = codeEl.props.className || '';
  if (!/\blanguage-mermaid\b/.test(className)) return null;
  const source = collectCodeText(codeEl.props.children);
  return { source };
}

function MarkdownMermaidFenceBlock({ source }) {
  const { darkMode } = useTheme();
  const { svg, error, loading } = useMermaidSvg(source, { darkMode, enabled: true });
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(source);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard may be denied; ignore
    }
  }, [source]);

  const sanitized = svg
    ? DOMPurify.sanitize(svg, { USE_PROFILES: { svg: true } })
    : '';

  return (
    <Box
      className="markdown-mermaid-block"
      sx={{
        position: 'relative',
        display: 'block',
        width: '100%',
        borderRadius: 1,
        border: 1,
        borderColor: 'divider',
        p: 2,
        pt: 5,
        bgcolor: 'action.hover',
        overflow: 'auto',
      }}
    >
      <Tooltip title={copied ? 'Copied' : 'Copy Mermaid source'}>
        <span>
          <IconButton
            type="button"
            size="small"
            onClick={handleCopy}
            aria-label={copied ? 'Copied' : 'Copy Mermaid source'}
            sx={{
              position: 'absolute',
              top: 4,
              right: 4,
              zIndex: 1,
              color: 'text.secondary',
              bgcolor: 'action.hover',
              '&:hover': { bgcolor: 'action.selected' },
            }}
          >
            {copied ? <Check fontSize="small" /> : <ContentCopy fontSize="small" />}
          </IconButton>
        </span>
      </Tooltip>
      {error && (
        <Alert severity="warning" sx={{ mb: 1 }}>
          {error}
        </Alert>
      )}
      {loading && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 1 }}>
          <CircularProgress size={18} />
          <Typography variant="body2" color="text.secondary">
            Rendering diagram…
          </Typography>
        </Box>
      )}
      {sanitized ? (
        <Box sx={{ '& svg': { maxWidth: '100%', height: 'auto', display: 'block' } }} dangerouslySetInnerHTML={{ __html: sanitized }} />
      ) : null}
    </Box>
  );
}

/** react-markdown `pre` with copy-to-clipboard control */
export function MarkdownPreWithCopy({ children, node: _node, ...props }) {
  const [copied, setCopied] = useState(false);
  const preRef = useRef(null);

  const handleCopy = useCallback(async () => {
    const el = preRef.current;
    if (!el) return;
    const text = el.innerText ?? '';
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard may be denied; ignore
    }
  }, []);

  return (
    <Box sx={{ position: 'relative', display: 'block', width: '100%' }}>
      <Tooltip title={copied ? 'Copied' : 'Copy code'}>
        <span>
          <IconButton
            type="button"
            size="small"
            onClick={handleCopy}
            aria-label={copied ? 'Copied' : 'Copy code'}
            sx={{
              position: 'absolute',
              top: 4,
              right: 4,
              zIndex: 1,
              color: 'text.secondary',
              bgcolor: 'action.hover',
              '&:hover': { bgcolor: 'action.selected' },
            }}
          >
            {copied ? <Check fontSize="small" /> : <ContentCopy fontSize="small" />}
          </IconButton>
        </span>
      </Tooltip>
      <pre ref={preRef} {...props}>
        {children}
      </pre>
    </Box>
  );
}

/** react-markdown `pre`: Mermaid fenced blocks render as diagrams; other fences use MarkdownPreWithCopy */
export function MarkdownPreWithMermaid({ children, node: _node, ...props }) {
  const info = tryParseMermaidFence(children);
  if (info) {
    return <MarkdownMermaidFenceBlock source={info.source} />;
  }
  return <MarkdownPreWithCopy {...props}>{children}</MarkdownPreWithCopy>;
}
