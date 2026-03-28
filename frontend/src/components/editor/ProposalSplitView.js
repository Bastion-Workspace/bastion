import React, { useMemo, useRef, useEffect } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { EditorView } from '@codemirror/view';
import { markdown, markdownLanguage } from '@codemirror/lang-markdown';
import { Box, Button, Typography } from '@mui/material';
import { useTheme } from '../../contexts/ThemeContext';
import { ACCENT_PALETTES } from '../../theme/themeConfig';
import { applyProposalsToContent } from '../../services/proposalPreviewService';

const createSplitViewTheme = (darkMode, accentId = 'blue') => {
  const palette = ACCENT_PALETTES[accentId] || ACCENT_PALETTES.blue;
  const accent = darkMode ? palette.dark : palette.light;
  const primaryMain = accent?.primary?.main ?? (darkMode ? '#90caf9' : '#1976d2');
  return EditorView.baseTheme({
    '&': {
      backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
      color: darkMode ? '#d4d4d4' : '#212121',
    },
    '.cm-editor': {
      backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
    },
    '.cm-scroller': {
      backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
    },
    '.cm-content': {
      fontFamily: 'monospace',
      fontSize: '14px',
      lineHeight: '1.5',
      wordBreak: 'break-word',
      overflowWrap: 'anywhere',
      backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
      color: darkMode ? '#d4d4d4' : '#212121',
    },
    '.cm-gutters': {
      backgroundColor: darkMode ? '#1e1e1e' : '#f5f5f5',
      color: darkMode ? '#858585' : '#999999',
      border: 'none',
    },
  });
};

export default function ProposalSplitView({
  content,
  operations = [],
  onAcceptAll,
  onRejectAll,
  onClose,
  height = '100%',
}) {
  const { darkMode, accentId } = useTheme();
  const leftRef = useRef(null);
  const rightRef = useRef(null);
  const scrollSyncRef = useRef(false);

  const previewContent = useMemo(
    () => applyProposalsToContent(content || '', operations),
    [content, operations]
  );

  const mdTheme = useMemo(
    () => createSplitViewTheme(darkMode, accentId),
    [darkMode, accentId]
  );

  const extensions = useMemo(
    () => [
      markdown({ base: markdownLanguage }),
      EditorView.lineWrapping,
      mdTheme,
      EditorView.editable.of(false),
    ],
    [mdTheme]
  );

  useEffect(() => {
    let cleanup = null;
    const id = setTimeout(() => {
      const leftScroll = leftRef.current?.view?.scrollDOM;
      const rightScroll = rightRef.current?.view?.scrollDOM;
      if (!leftScroll || !rightScroll) return;

      const syncLeftToRight = () => {
        if (scrollSyncRef.current) return;
        scrollSyncRef.current = true;
        rightScroll.scrollTop = leftScroll.scrollTop;
        rightScroll.scrollLeft = leftScroll.scrollLeft;
        requestAnimationFrame(() => { scrollSyncRef.current = false; });
      };
      const syncRightToLeft = () => {
        if (scrollSyncRef.current) return;
        scrollSyncRef.current = true;
        leftScroll.scrollTop = rightScroll.scrollTop;
        leftScroll.scrollLeft = rightScroll.scrollLeft;
        requestAnimationFrame(() => { scrollSyncRef.current = false; });
      };

      leftScroll.addEventListener('scroll', syncLeftToRight, { passive: true });
      rightScroll.addEventListener('scroll', syncRightToLeft, { passive: true });
      cleanup = () => {
        leftScroll.removeEventListener('scroll', syncLeftToRight);
        rightScroll.removeEventListener('scroll', syncRightToLeft);
      };
    }, 150);
    return () => {
      clearTimeout(id);
      if (cleanup) cleanup();
    };
  }, [content, operations]);

  const count = Array.isArray(operations) ? operations.length : 0;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height, minHeight: 0 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', px: 1, py: 0.5, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="caption" color="text.secondary">
          Compare: current (left) vs preview with {count} proposal{count !== 1 ? 's' : ''} applied (right)
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          {onClose && (
            <Button size="small" variant="outlined" onClick={onClose}>
              Close compare
            </Button>
          )}
          <Button size="small" variant="outlined" color="success" onClick={onAcceptAll}>
            Accept All
          </Button>
          <Button size="small" variant="outlined" color="error" onClick={onRejectAll}>
            Reject All
          </Button>
        </Box>
      </Box>
      <Box sx={{ flex: 1, minHeight: 0, display: 'flex' }}>
        <Box sx={{ flex: 1, minWidth: 0, borderRight: 1, borderColor: 'divider', display: 'flex', flexDirection: 'column' }}>
          <Typography variant="caption" sx={{ px: 1, py: 0.5, bgcolor: 'action.hover' }}>Current</Typography>
          <Box sx={{ flex: 1, minHeight: 0 }}>
            <CodeMirror
              ref={leftRef}
              value={content || ''}
              height="100%"
              basicSetup={false}
              readOnly
              extensions={extensions}
              style={{ height: '100%' }}
            />
          </Box>
        </Box>
        <Box sx={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
          <Typography variant="caption" sx={{ px: 1, py: 0.5, bgcolor: 'action.hover' }}>Preview</Typography>
          <Box sx={{ flex: 1, minHeight: 0 }}>
            <CodeMirror
              ref={rightRef}
              value={previewContent}
              height="100%"
              basicSetup={false}
              readOnly
              extensions={extensions}
              style={{ height: '100%' }}
            />
          </Box>
        </Box>
      </Box>
    </Box>
  );
}
