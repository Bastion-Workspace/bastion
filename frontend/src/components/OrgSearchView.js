import React, { useState, useCallback } from 'react';
import {
  Box,
  TextField,
  InputAdornment,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Typography,
  Chip,
  CircularProgress,
  Alert,
  Divider,
  Paper,
  Checkbox,
  FormControlLabel
} from '@mui/material';
import {
  Search,
  Description,
  LocalOffer,
  CheckCircle,
  Schedule,
  Error as ErrorIcon,
  Archive
} from '@mui/icons-material';
import apiService from '../services/apiService';

/**
 * ROOSEVELT'S ORG SEARCH VIEW
 * Full-text search across org files with filtering
 */
const OrgSearchView = ({ onOpenDocument }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [includeArchives, setIncludeArchives] = useState(false);

  // Execute search
  const handleSearch = useCallback(async (query, includeArchiveFiles = false) => {
    if (!query || query.trim().length === 0) {
      setResults(null);
      return;
    }

    try {
      setSearching(true);
      setError(null);

      const response = await apiService.get(
        `/api/org/search?query=${encodeURIComponent(query)}&include_archives=${includeArchiveFiles}`
      );

      if (response.success) {
        setResults(response);
      } else {
        setError(response.error || 'Search failed');
      }
    } catch (err) {
      console.error('❌ Search error:', err);
      setError(err.message || 'Failed to search org files');
    } finally {
      setSearching(false);
    }
  }, []);

  // Handle search input change
  const handleSearchChange = (e) => {
    const value = e.target.value;
    setSearchQuery(value);

    // Debounced search (simple version - search on Enter or after pause)
    if (value.length === 0) {
      setResults(null);
    }
  };

  // Handle search on Enter key
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch(searchQuery, includeArchives);
    }
  };

  // Re-search when includeArchives checkbox changes
  React.useEffect(() => {
    if (searchQuery && results) {
      handleSearch(searchQuery, includeArchives);
    }
  }, [includeArchives]); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle clicking a search result
  const handleResultClick = (result) => {
    if (!onOpenDocument) return;

    // document_id is already in the search results!
    if (!result.document_id) {
      console.error('❌ Search result missing document_id:', result);
      alert(`❌ Could not find document ID for: ${result.filename}`);
      return;
    }

    console.log('✅ ROOSEVELT: Opening org file:', result.document_id);
    
    // Open document with scroll parameters
    onOpenDocument({
      documentId: result.document_id,
      documentName: result.filename,
      scrollToLine: result.line_number,
      scrollToHeading: result.heading
    });
  };

  // Get badge color for TODO state
  const getTodoStateColor = (state) => {
    const doneStates = ['DONE', 'CANCELED', 'CANCELLED', 'WONTFIX', 'FIXED'];
    return doneStates.includes(state) ? 'success' : 'error';
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Search Header */}
      <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider', backgroundColor: 'background.paper' }}>
        <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <Search /> Search Org Files
        </Typography>

        <TextField
          fullWidth
          placeholder="Search across all your org files..."
          value={searchQuery}
          onChange={handleSearchChange}
          onKeyPress={handleKeyPress}
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
          sx={{ mb: 1 }}
        />

        {/* **ROOSEVELT ARCHIVE SEARCH!** Option to include archived items */}
        <FormControlLabel
          control={
            <Checkbox
              checked={includeArchives}
              onChange={(e) => setIncludeArchives(e.target.checked)}
              size="small"
            />
          }
          label={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Archive fontSize="small" sx={{ color: 'text.secondary' }} />
              <Typography variant="body2" color="text.secondary">
                Include archived items (_archive.org files)
              </Typography>
            </Box>
          }
          sx={{ mb: 1 }}
        />

        <Typography variant="caption" color="text.secondary">
          Press Enter to search • Searches headings, content, tags, and TODO states
        </Typography>
      </Box>

      {/* Results Area */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} icon={<ErrorIcon />}>
            {error}
          </Alert>
        )}

        {results && (
          <>
            {/* Results Summary */}
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Found <strong>{results.count}</strong> results
                {results.total_matches > results.count && ` (showing top ${results.count} of ${results.total_matches})`}
                {' in '}<strong>{results.files_searched}</strong> files
              </Typography>
            </Box>

            {results.count === 0 ? (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="body1" color="text.secondary">
                  No results found for "{results.query}"
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Try different keywords or check your org files
                </Typography>
              </Box>
            ) : (
              <List disablePadding>
                {results.results.map((result, idx) => (
                  <React.Fragment key={idx}>
                    {idx > 0 && <Divider />}
                    <ListItem disablePadding>
                      <ListItemButton onClick={() => handleResultClick(result)} sx={{ py: 1.5 }}>
                        <Box sx={{ width: '100%' }}>
                          {/* Heading and Badges */}
                          <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, mb: 0.5 }}>
                            <Typography
                              variant="body1"
                              sx={{
                                fontWeight: result.heading_match ? 600 : 500,
                                flex: 1,
                                color: result.heading_match ? 'primary.main' : 'text.primary'
                              }}
                            >
                              {'•'.repeat(result.level)} {result.heading}
                            </Typography>

                            {result.todo_state && (
                              <Chip
                                label={result.todo_state}
                                size="small"
                                color={getTodoStateColor(result.todo_state)}
                                sx={{ fontWeight: 600, fontSize: '0.7rem' }}
                              />
                            )}
                          </Box>

                          {/* Preview */}
                          {result.preview && result.preview !== result.heading && (
                            <Typography
                              variant="body2"
                              color="text.secondary"
                              sx={{
                                mb: 0.5,
                                fontSize: '0.875rem',
                                fontStyle: result.content_match ? 'italic' : 'normal'
                              }}
                            >
                              {result.preview}
                            </Typography>
                          )}

                          {/* Metadata Row */}
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                            <Chip
                              icon={<Description fontSize="small" />}
                              label={result.filename}
                              size="small"
                              variant="outlined"
                              sx={{ fontSize: '0.7rem' }}
                            />

                            {result.tags && result.tags.length > 0 && (
                              <Box sx={{ display: 'flex', gap: 0.5 }}>
                                {result.tags.map(tag => (
                                  <Chip
                                    key={tag}
                                    icon={<LocalOffer sx={{ fontSize: 12 }} />}
                                    label={tag}
                                    size="small"
                                    color="primary"
                                    variant="outlined"
                                    sx={{ fontSize: '0.7rem', height: 20 }}
                                  />
                                ))}
                              </Box>
                            )}

                            {result.scheduled && (
                              <Chip
                                icon={<Schedule sx={{ fontSize: 12 }} />}
                                label={`SCHED: ${result.scheduled.split(' ')[0]}`}
                                size="small"
                                color="info"
                                variant="outlined"
                                sx={{ fontSize: '0.7rem', height: 20 }}
                              />
                            )}

                            {result.deadline && (
                              <Chip
                                icon={<ErrorIcon sx={{ fontSize: 12 }} />}
                                label={`DUE: ${result.deadline.split(' ')[0]}`}
                                size="small"
                                color="warning"
                                variant="outlined"
                                sx={{ fontSize: '0.7rem', height: 20 }}
                              />
                            )}

                            <Typography variant="caption" color="text.secondary">
                              Line {result.line_number}
                            </Typography>
                          </Box>
                        </Box>
                      </ListItemButton>
                    </ListItem>
                  </React.Fragment>
                ))}
              </List>
            )}
          </>
        )}

        {!results && !searching && (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <Search sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              Search Your Org Files
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Enter a search query to find headings, content, tags, and TODOs
            </Typography>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default OrgSearchView;

