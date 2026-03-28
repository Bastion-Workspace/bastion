import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  TextField,
  InputAdornment,
  List,
  ListItemButton,
  Typography,
  Chip,
  CircularProgress,
  Box,
  useTheme,
  useMediaQuery
} from '@mui/material';
import { Search, Description, Close } from '@mui/icons-material';
import documentService from '../services/document/DocumentService';

const DEBOUNCE_MS = 300;

function extractHeadingFromContent(text) {
  if (!text || typeof text !== 'string') return null;
  const match = text.match(/^#{1,6}\s+(.+)$/m);
  return match ? match[1].trim() : null;
}

const DocumentSearchOverlay = ({ open, onClose }) => {
  const theme = useTheme();
  const fullScreen = useMediaQuery(theme.breakpoints.down('sm'));
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const resultsListRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => setDebouncedQuery(query.trim()), DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [query, open]);

  useEffect(() => {
    if (!open) return;
    inputRef.current?.focus();
  }, [open]);

  const runSearch = useCallback(async (q) => {
    if (!q) {
      setResults(null);
      setError(null);
      return;
    }
    setSearching(true);
    setError(null);
    try {
      const response = await documentService.searchDocuments(q, {
        searchMode: 'hybrid',
        limit: 20
      });
      if (response && Array.isArray(response.results)) {
        setResults(response.results);
        setSelectedIndex(-1);
      } else {
        setResults([]);
      }
    } catch (err) {
      setError(err.message || 'Search failed');
      setResults([]);
    } finally {
      setSearching(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    if (debouncedQuery) {
      runSearch(debouncedQuery);
    } else {
      setResults(null);
      setError(null);
    }
  }, [debouncedQuery, open, runSearch]);

  const handleResultClick = (result) => {
    const tcm = window.tabbedContentManagerRef;
    if (!tcm?.openDocument) return;
    const doc = result.document || {};
    const documentName = doc.title || doc.filename || result.document_id || 'Document';
    const scrollToHeading = extractHeadingFromContent(result.text || result.content);
    tcm.openDocument(result.document_id, documentName, scrollToHeading ? { scrollToHeading } : {});
    onClose();
  };

  const handleKeyDown = (e) => {
    const list = results && results.length ? results : [];
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((i) => (i < list.length - 1 ? i + 1 : i));
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((i) => (i > 0 ? i - 1 : -1));
      return;
    }
    if (e.key === 'Enter' && selectedIndex >= 0 && list[selectedIndex]) {
      e.preventDefault();
      handleResultClick(list[selectedIndex]);
      return;
    }
    if (e.key === 'Escape') {
      onClose();
    }
  };

  useEffect(() => {
    if (selectedIndex >= 0 && resultsListRef.current) {
      const el = resultsListRef.current.querySelector(`[data-result-index="${selectedIndex}"]`);
      el?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }, [selectedIndex]);

  const snippetHtml = (result) => {
    const raw = result.highlighted_snippet || result.highlighted_text || result.text || result.content || '';
    if (typeof raw !== 'string') return '';
    return raw;
  };

  const hasSnippetMarkup = (result) => {
    const s = result.highlighted_snippet || '';
    return typeof s === 'string' && s.includes('<mark>');
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      fullScreen={fullScreen}
      maxWidth="md"
      fullWidth
      onKeyDown={handleKeyDown}
      PaperProps={{
        sx: { minHeight: fullScreen ? '100%' : 360 }
      }}
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', pb: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Search color="primary" />
          <Typography variant="h6">Search Documents</Typography>
        </Box>
        <Chip
          icon={<Close />}
          label="Close"
          onClick={onClose}
          size="small"
          variant="outlined"
          sx={{ cursor: 'pointer' }}
        />
      </DialogTitle>
      <DialogContent>
        <TextField
          inputRef={inputRef}
          fullWidth
          placeholder="Search document content..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search />
              </InputAdornment>
            ),
            endAdornment: searching ? (
              <InputAdornment position="end">
                <CircularProgress size={20} />
              </InputAdornment>
            ) : null
          }}
          sx={{ mb: 2 }}
        />

        {error && (
          <Typography color="error" variant="body2" sx={{ mb: 2 }}>
            {error}
          </Typography>
        )}

        {results && results.length === 0 && debouncedQuery && !searching && (
          <Typography color="text.secondary" sx={{ py: 3 }}>
            No documents match &quot;{debouncedQuery}&quot;
          </Typography>
        )}

        {results && results.length > 0 && (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {results.length} result{results.length !== 1 ? 's' : ''}
          </Typography>
        )}

        <List
          ref={resultsListRef}
          disablePadding
          sx={{ overflow: 'auto', maxHeight: fullScreen ? 'calc(100vh - 220px)' : 320 }}
        >
          {results && results.map((result, idx) => {
            const doc = result.document || {};
            const title = doc.title || doc.filename || result.document_id || 'Document';
            const docType = doc.doc_type || (doc.filename && doc.filename.split('.').pop()) || '';
            const snippet = snippetHtml(result);
            const useHtml = hasSnippetMarkup(result);
            const isSelected = selectedIndex === idx;

            return (
              <ListItemButton
                key={result.chunk_id || `${result.document_id}-${idx}`}
                data-result-index={idx}
                selected={isSelected}
                onClick={() => handleResultClick(result)}
                sx={{
                  py: 1.5,
                  px: 2,
                  borderLeft: '3px solid transparent',
                  ...(isSelected ? { borderLeftColor: 'primary.main', bgcolor: 'action.selected' } : {})
                }}
              >
                <Box sx={{ width: '100%' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                    <Description fontSize="small" color="action" />
                    <Typography variant="subtitle1" fontWeight={600}>
                      {title}
                    </Typography>
                    {docType && (
                      <Chip label={docType} size="small" variant="outlined" sx={{ fontSize: '0.7rem' }} />
                    )}
                  </Box>
                  {snippet && (
                    useHtml ? (
                      <Typography
                        variant="body2"
                        color="text.secondary"
                        component="div"
                        sx={{
                          fontSize: '0.875rem',
                          '& mark': {
                            backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255,235,59,0.4)' : 'rgba(255,235,59,0.6)',
                            padding: '0 2px',
                            borderRadius: 1
                          }
                        }}
                        dangerouslySetInnerHTML={{ __html: snippet }}
                      />
                    ) : (
                      <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.875rem' }}>
                        {snippet.slice(0, 200)}{snippet.length > 200 ? '…' : ''}
                      </Typography>
                    )
                  )}
                </Box>
              </ListItemButton>
            );
          })}
        </List>
      </DialogContent>
    </Dialog>
  );
};

export default DocumentSearchOverlay;
