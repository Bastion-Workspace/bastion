import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Typography,
  TextField,
  InputAdornment,
  List,
  ListItem,
  ListItemButton,
  Chip,
  CircularProgress,
  Alert,
  Divider,
  Checkbox,
  FormControlLabel,
  ToggleButtonGroup,
  ToggleButton,
  Button
} from '@mui/material';
import {
  LocalOffer,
  Search,
  Description,
  Schedule,
  Error as ErrorIcon,
  Archive
} from '@mui/icons-material';
import orgService from '../services/org/OrgService';

const getTodoStateColor = (state) => {
  const doneStates = ['DONE', 'CANCELED', 'CANCELLED', 'WONTFIX', 'FIXED'];
  return doneStates.includes(state) ? 'success' : 'error';
};

/**
 * Browse org headline tags and open headings that match selected tags (any / all).
 */
const OrgTagsView = ({ onOpenDocument }) => {
  const [tagList, setTagList] = useState([]);
  const [tagsLoading, setTagsLoading] = useState(true);
  const [tagsError, setTagsError] = useState(null);
  const [listFilter, setListFilter] = useState('');
  const [selectedTags, setSelectedTags] = useState(() => new Set());
  const [tagsMatch, setTagsMatch] = useState('any');
  const [includeArchives, setIncludeArchives] = useState(false);
  const [results, setResults] = useState(null);
  const [resultsLoading, setResultsLoading] = useState(false);
  const [resultsError, setResultsError] = useState(null);

  const loadTagIndex = useCallback(async () => {
    try {
      setTagsLoading(true);
      setTagsError(null);
      const res = await orgService.listOrgTags({ includeArchives });
      if (res.success) {
        setTagList(res.tags || []);
      } else {
        setTagsError(res.error || 'Failed to load tags');
        setTagList([]);
      }
    } catch (err) {
      setTagsError(err.message || 'Failed to load tags');
      setTagList([]);
    } finally {
      setTagsLoading(false);
    }
  }, [includeArchives]);

  useEffect(() => {
    loadTagIndex();
  }, [loadTagIndex]);

  const selectedKey = useMemo(() => [...selectedTags].sort().join('\u0001'), [selectedTags]);

  const runTagSearch = useCallback(async () => {
    if (!selectedKey) {
      setResults(null);
      setResultsError(null);
      return;
    }
    const tagsArr = selectedKey.split('\u0001');
    try {
      setResultsLoading(true);
      setResultsError(null);
      const res = await orgService.searchOrgFiles('', {
        tags: tagsArr,
        tagsMatch,
        includeArchives,
        includeContent: true,
        limit: 500
      });
      if (res.success) {
        setResults(res);
      } else {
        setResultsError(res.error || 'Search failed');
        setResults(null);
      }
    } catch (err) {
      setResultsError(err.message || 'Search failed');
      setResults(null);
    } finally {
      setResultsLoading(false);
    }
  }, [selectedKey, tagsMatch, includeArchives]);

  useEffect(() => {
    runTagSearch();
  }, [runTagSearch]);

  const toggleTag = (tag) => {
    setSelectedTags((prev) => {
      const next = new Set(prev);
      if (next.has(tag)) next.delete(tag);
      else next.add(tag);
      return next;
    });
  };

  const clearSelection = () => setSelectedTags(new Set());

  const filteredTagEntries = useMemo(() => {
    const q = listFilter.trim().toLowerCase();
    if (!q) return tagList;
    return tagList.filter((t) => (t.tag || '').toLowerCase().includes(q));
  }, [tagList, listFilter]);

  const handleResultClick = (result) => {
    if (!onOpenDocument) return;
    if (!result.document_id) {
      alert(`Could not find document ID for: ${result.filename}`);
      return;
    }
    onOpenDocument({
      documentId: result.document_id,
      documentName: result.filename,
      scrollToLine: result.line_number,
      scrollToHeading: result.heading
    });
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider', backgroundColor: 'background.paper' }}>
        <Typography variant="h6" sx={{ mb: 1.5, display: 'flex', alignItems: 'center', gap: 1 }}>
          <LocalOffer /> Tags Browser
        </Typography>

        <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 1, mb: 1.5 }}>
          <ToggleButtonGroup
            value={tagsMatch}
            exclusive
            size="small"
            onChange={(e, v) => v && setTagsMatch(v)}
          >
            <ToggleButton value="any">Match any</ToggleButton>
            <ToggleButton value="all">Match all</ToggleButton>
          </ToggleButtonGroup>
          <FormControlLabel
            control={
              <Checkbox
                size="small"
                checked={includeArchives}
                onChange={(e) => setIncludeArchives(e.target.checked)}
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Archive fontSize="small" sx={{ color: 'text.secondary' }} />
                <Typography variant="body2" color="text.secondary">
                  Include _archive.org
                </Typography>
              </Box>
            }
          />
        </Box>

        <TextField
          fullWidth
          size="small"
          placeholder="Filter tag list…"
          value={listFilter}
          onChange={(e) => setListFilter(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search fontSize="small" />
              </InputAdornment>
            )
          }}
          sx={{ mb: 1 }}
        />

        {selectedTags.size > 0 && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 0.5, mb: 1 }}>
            <Typography variant="caption" color="text.secondary" sx={{ mr: 0.5 }}>
              Selected:
            </Typography>
            {[...selectedTags].sort((a, b) => a.localeCompare(b)).map((t) => (
              <Chip
                key={t}
                label={t}
                size="small"
                color="primary"
                onDelete={() => toggleTag(t)}
              />
            ))}
            <Button size="small" onClick={clearSelection}>
              Clear
            </Button>
          </Box>
        )}

        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
          Click tags to select; headings must include {tagsMatch === 'all' ? 'every' : 'at least one'} selected tag.
        </Typography>

        <Box
          sx={{
            maxHeight: 160,
            overflow: 'auto',
            display: 'flex',
            flexWrap: 'wrap',
            gap: 0.5,
            alignContent: 'flex-start',
            p: 0.5,
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: 1,
            backgroundColor: 'action.hover'
          }}
        >
          {tagsLoading && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 1 }}>
              <CircularProgress size={20} />
              <Typography variant="body2" color="text.secondary">
                Loading tags…
              </Typography>
            </Box>
          )}
          {!tagsLoading && tagsError && (
            <Alert severity="error" sx={{ width: '100%' }}>
              {tagsError}
            </Alert>
          )}
          {!tagsLoading && !tagsError && filteredTagEntries.length === 0 && (
            <Typography variant="body2" color="text.secondary" sx={{ p: 1 }}>
              {tagList.length === 0
                ? 'No tags found in your org files.'
                : 'No tags match the filter.'}
            </Typography>
          )}
          {!tagsLoading &&
            !tagsError &&
            filteredTagEntries.map(({ tag, count }) => (
              <Chip
                key={tag}
                label={`${tag} (${count})`}
                size="small"
                variant={selectedTags.has(tag) ? 'filled' : 'outlined'}
                color={selectedTags.has(tag) ? 'primary' : 'default'}
                onClick={() => toggleTag(tag)}
                sx={{ cursor: 'pointer' }}
              />
            ))}
        </Box>
      </Box>

      <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
        {resultsError && (
          <Alert severity="error" sx={{ mb: 2 }} icon={<ErrorIcon />}>
            {resultsError}
          </Alert>
        )}

        {selectedTags.size === 0 && !resultsLoading && (
          <Box sx={{ textAlign: 'center', py: 6 }}>
            <LocalOffer sx={{ fontSize: 56, color: 'text.disabled', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              Select one or more tags
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Matching headings appear here. Use Match all to require every selected tag on the same headline.
            </Typography>
          </Box>
        )}

        {selectedTags.size > 0 && resultsLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
            <CircularProgress />
          </Box>
        )}

        {selectedTags.size > 0 && !resultsLoading && results && (
          <>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Found <strong>{results.count}</strong> results
                {results.total_matches > results.count &&
                  ` (showing top ${results.count} of ${results.total_matches})`}
                {' in '}
                <strong>{results.files_searched}</strong> files
              </Typography>
            </Box>

            {results.count === 0 ? (
              <Typography variant="body2" color="text.secondary">
                No headings match the selected tags with the current match mode.
              </Typography>
            ) : (
              <List disablePadding>
                {results.results.map((result, idx) => (
                  <React.Fragment key={`${result.document_id || result.filename}-${result.line_number}-${idx}`}>
                    {idx > 0 && <Divider />}
                    <ListItem disablePadding>
                      <ListItemButton onClick={() => handleResultClick(result)} sx={{ py: 1.5 }}>
                        <Box sx={{ width: '100%' }}>
                          <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, mb: 0.5 }}>
                            <Typography variant="body1" sx={{ fontWeight: 500, flex: 1 }}>
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

                          {result.preview && result.preview !== result.heading && (
                            <Typography
                              variant="body2"
                              color="text.secondary"
                              sx={{ mb: 0.5, fontSize: '0.875rem' }}
                            >
                              {result.preview}
                            </Typography>
                          )}

                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                            <Chip
                              icon={<Description fontSize="small" />}
                              label={result.filename}
                              size="small"
                              variant="outlined"
                              sx={{ fontSize: '0.7rem' }}
                            />
                            {result.tags && result.tags.length > 0 && (
                              <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                                {result.tags.map((tg) => (
                                  <Chip
                                    key={tg}
                                    icon={<LocalOffer sx={{ fontSize: 12 }} />}
                                    label={tg}
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
      </Box>
    </Box>
  );
};

export default OrgTagsView;
