/**
 * RSS Feed Manager Modal
 * Component for adding and managing RSS feeds
 */

import React, { useState, useEffect, useCallback, useMemo, useId } from 'react';
import { useQueryClient } from 'react-query';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Typography,
  Box,
  Alert,
  Stack,
  CircularProgress,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useTheme } from '@mui/material/styles';
import rssService from '../services/rssService';
import { useAuth } from '../contexts/AuthContext';

const STATIC_CATEGORIES = [
  'technology',
  'science',
  'news',
  'business',
  'politics',
  'entertainment',
  'sports',
  'health',
  'education',
  'other',
];

function titleCaseCategoryKey(key) {
  if (key === 'uncategorized') return 'Uncategorized';
  return key
    .split(/[\s_]+/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(' ');
}

const RSSFeedManager = ({ isOpen, onClose, onFeedAdded, feedContext = null }) => {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const theme = useTheme();
  const formDomId = `rss-feed-mgr-${useId().replace(/:/g, '')}`;

  const [feedData, setFeedData] = useState({
    feed_url: '',
    feed_name: '',
    category: 'technology',
    tags: [],
    check_interval: 3600,
  });

  const feedScope = feedContext?.isGlobal ? 'global' : 'user';
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
  const [validating, setValidating] = useState(false);

  const contextDefaultCategory = feedContext?.defaultCategory ?? '';
  const contextDefaultCategoryLabel = feedContext?.defaultCategoryLabel ?? '';

  const categoryOptions = useMemo(() => {
    const dc = contextDefaultCategory.trim().toLowerCase();
    const base = [...STATIC_CATEGORIES];
    if (dc && !base.includes(dc)) base.push(dc);
    return base;
  }, [contextDefaultCategory]);

  const categoryOptionLabel = useCallback(
    (value) => {
      const dc = contextDefaultCategory.trim().toLowerCase();
      if (value === dc && contextDefaultCategoryLabel) return contextDefaultCategoryLabel;
      if (STATIC_CATEGORIES.includes(value)) {
        return value.charAt(0).toUpperCase() + value.slice(1);
      }
      return titleCaseCategoryKey(value);
    },
    [contextDefaultCategory, contextDefaultCategoryLabel]
  );

  const checkIntervals = [
    { value: 900, label: '15 minutes' },
    { value: 1800, label: '30 minutes' },
    { value: 3600, label: '1 hour' },
    { value: 7200, label: '2 hours' },
    { value: 14400, label: '4 hours' },
    { value: 28800, label: '8 hours' },
    { value: 86400, label: '24 hours' },
  ];

  const canCreateGlobalFeeds = user?.role === 'admin';

  useEffect(() => {
    if (!isOpen) return;
    const dc = contextDefaultCategory.trim().toLowerCase();
    setFeedData({
      feed_url: '',
      feed_name: '',
      category: dc || 'technology',
      tags: [],
      check_interval: 3600,
    });
    setError(null);
    setPreview(null);
  }, [isOpen, contextDefaultCategory, contextDefaultCategoryLabel]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFeedData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleTagsChange = (e) => {
    const tags = e.target.value
      .split(',')
      .map((tag) => tag.trim())
      .filter((tag) => tag);
    setFeedData((prev) => ({
      ...prev,
      tags,
    }));
  };

  const validateFeedUrl = async () => {
    if (!feedData.feed_url) {
      setError('Please enter a feed URL');
      return false;
    }

    if (
      !feedData.feed_url.startsWith('http://') &&
      !feedData.feed_url.startsWith('https://')
    ) {
      setError('Feed URL must start with http:// or https://');
      return false;
    }

    return true;
  };

  const previewFeed = async () => {
    if (!(await validateFeedUrl())) return;

    setValidating(true);
    setError(null);

    try {
      const result = await rssService.validateFeedUrl(feedData.feed_url);

      if (result.status === 'success') {
        setPreview(result.data);
      } else {
        throw new Error('Invalid RSS feed URL');
      }
    } catch (err) {
      setError('Failed to preview feed. Please check the URL and try again.');
    } finally {
      setValidating(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!(await validateFeedUrl())) return;

    if (!feedData.feed_name.trim()) {
      setError('Please enter a feed name');
      return;
    }

    if (feedScope === 'global' && !canCreateGlobalFeeds) {
      setError('Only admin users can create global RSS feeds');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const isGlobal = feedScope === 'global';
      const newFeed = await rssService.createFeed(feedData, isGlobal);

      queryClient.invalidateQueries(['rss', 'feeds']);
      queryClient.invalidateQueries(['rss', 'unread-counts']);
      queryClient.invalidateQueries({ queryKey: ['folders', 'tree'], exact: false });

      const scopeText = isGlobal ? 'global' : 'personal';
      showToast(
        `RSS feed "${newFeed.feed_name}" added successfully as ${scopeText} feed!`,
        'success'
      );

      onClose();
      if (onFeedAdded) {
        onFeedAdded(newFeed);
      }
    } catch (err) {
      setError(err.message || 'Failed to add RSS feed');
    } finally {
      setLoading(false);
    }
  };

  const showToast = useCallback(
    (message, type = 'info') => {
      const toast = document.createElement('div');
      toast.className = `toast toast-${type}`;
      toast.textContent = message;
      const bg =
        type === 'success' ? theme.palette.success.main : theme.palette.info.main;
      toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${bg};
            color: ${theme.palette.getContrastText(bg)};
            padding: 12px 20px;
            border-radius: 4px;
            z-index: 10000;
            box-shadow: ${theme.shadows[8]};
        `;

      document.body.appendChild(toast);

      setTimeout(() => {
        document.body.removeChild(toast);
      }, 3000);
    },
    [theme]
  );

  return (
    <Dialog
      open={isOpen}
      onClose={loading ? undefined : onClose}
      maxWidth="sm"
      fullWidth
      scroll="paper"
      sx={{
        '& .MuiDialog-paper': {
          overflowX: 'hidden',
        },
      }}
      PaperProps={{
        sx: {
          maxHeight: '90vh',
          overflowX: 'hidden',
          // Avoid 100vw — it ignores scrollbar gutter and can force page-level horizontal scroll
          maxWidth: { xs: 'min(600px, calc(100% - 32px))', sm: 600 },
        },
      }}
    >
      <DialogTitle
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          pr: 1,
          gap: 1,
          overflow: 'hidden',
          boxSizing: 'border-box',
        }}
      >
        <Typography component="span" variant="h6" sx={{ minWidth: 0, flex: '1 1 auto' }}>
          Add RSS Feed
        </Typography>
        <IconButton
          aria-label="close"
          onClick={onClose}
          disabled={loading}
          edge="end"
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent
        dividers
        sx={{
          overflowX: 'hidden',
          overflowY: 'auto',
          boxSizing: 'border-box',
          minWidth: 0,
        }}
      >
        <Box
          component="form"
          id={formDomId}
          onSubmit={handleSubmit}
          noValidate
          sx={{ minWidth: 0, maxWidth: '100%' }}
        >
          <Stack spacing={2.5} sx={{ minWidth: 0, maxWidth: '100%' }}>
            <Box sx={{ minWidth: 0, maxWidth: '100%' }}>
              <Typography component="label" htmlFor="feed_url" variant="body2" sx={{ mb: 1, display: 'block', fontWeight: 500 }}>
                Feed URL *
              </Typography>
              <TextField
                id="feed_url"
                name="feed_url"
                type="url"
                value={feedData.feed_url}
                onChange={handleInputChange}
                placeholder="https://example.com/feed.xml"
                disabled={loading}
                required
                fullWidth
                size="small"
                sx={{
                  maxWidth: '100%',
                  display: 'block',
                  '& .MuiOutlinedInput-root': { minWidth: 0 },
                }}
              />
              <Button
                type="button"
                variant="contained"
                onClick={previewFeed}
                disabled={loading || validating || !feedData.feed_url}
                sx={{ mt: 1, alignSelf: 'flex-start' }}
              >
                {validating ? (
                  <CircularProgress size={22} color="inherit" />
                ) : (
                  'Preview'
                )}
              </Button>
            </Box>

            <TextField
              label="Feed Name"
              id="feed_name"
              name="feed_name"
              value={feedData.feed_name}
              onChange={handleInputChange}
              placeholder="Enter a name for this feed"
              disabled={loading}
              required
              fullWidth
              size="small"
              sx={{ minWidth: 0, '& .MuiOutlinedInput-root': { minWidth: 0 } }}
            />

            <Box sx={{ minWidth: 0, maxWidth: '100%' }}>
              <Typography variant="body2" sx={{ mb: 1, fontWeight: 500 }}>
                Feed Scope
              </Typography>
              <Box
                sx={{
                  px: 1.5,
                  py: 1,
                  border: 1,
                  borderColor: 'divider',
                  borderRadius: 1,
                  bgcolor: 'action.hover',
                  typography: 'body2',
                  overflowWrap: 'break-word',
                  wordBreak: 'break-word',
                }}
              >
                {feedScope === 'global'
                  ? 'Global Feed (Global Documents)'
                  : 'Personal Feed (My Documents)'}
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block', overflowWrap: 'break-word' }}>
                {feedScope === 'global'
                  ? 'Global feeds are visible to all users and appear in Global Documents'
                  : 'Personal feeds are only visible to you and appear in My Documents'}
              </Typography>
            </Box>

            <FormControl fullWidth size="small" sx={{ minWidth: 0, maxWidth: '100%' }}>
              <InputLabel id="rss-category-label">Category</InputLabel>
              <Select
                labelId="rss-category-label"
                id="category"
                value={feedData.category}
                label="Category"
                onChange={(e) =>
                  setFeedData((prev) => ({ ...prev, category: e.target.value }))
                }
                disabled={loading}
              >
                {categoryOptions.map((category) => (
                  <MenuItem key={category} value={category}>
                    {categoryOptionLabel(category)}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              label="Tags (comma-separated)"
              id="tags"
              name="tags"
              value={feedData.tags.join(', ')}
              onChange={handleTagsChange}
              placeholder="tech, news, ai"
              disabled={loading}
              fullWidth
              size="small"
              sx={{ minWidth: 0, '& .MuiOutlinedInput-root': { minWidth: 0 } }}
            />

            <FormControl fullWidth size="small" sx={{ minWidth: 0, maxWidth: '100%' }}>
              <InputLabel id="rss-interval-label">Check Interval</InputLabel>
              <Select
                labelId="rss-interval-label"
                id="check_interval"
                name="check_interval"
                value={feedData.check_interval}
                label="Check Interval"
                onChange={(e) =>
                  setFeedData((prev) => ({
                    ...prev,
                    check_interval: Number(e.target.value),
                  }))
                }
                disabled={loading}
              >
                {checkIntervals.map((interval) => (
                  <MenuItem key={interval.value} value={interval.value}>
                    {interval.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {error && (
              <Alert severity="error" sx={{ maxWidth: '100%', overflowWrap: 'break-word', wordBreak: 'break-word' }}>
                {error}
              </Alert>
            )}

            {preview && (
              <Box
                sx={{
                  p: 2,
                  borderRadius: 1,
                  border: 1,
                  borderColor: 'divider',
                  bgcolor: 'action.hover',
                  minWidth: 0,
                  maxWidth: '100%',
                  overflowWrap: 'break-word',
                  wordBreak: 'break-word',
                }}
              >
                <Typography variant="subtitle1" gutterBottom>
                  Feed Preview
                </Typography>
                <Typography variant="body2" paragraph sx={{ overflowWrap: 'break-word' }}>
                  <strong>Title:</strong> {preview.title}
                </Typography>
                <Typography variant="body2" paragraph sx={{ overflowWrap: 'break-word' }}>
                  <strong>Description:</strong> {preview.description}
                </Typography>
                <Typography variant="body2" fontWeight={600}>
                  Sample Articles:
                </Typography>
                <Box component="ul" sx={{ m: 0, pl: 2.5, mt: 0.5, maxWidth: '100%' }}>
                  {preview.articles.map((article, index) => (
                    <Typography key={index} component="li" variant="body2" sx={{ overflowWrap: 'break-word' }}>
                      {article.title}
                    </Typography>
                  ))}
                </Box>
              </Box>
            )}
          </Stack>
        </Box>
      </DialogContent>

      <DialogActions
        sx={{
          px: 3,
          py: 2,
          flexWrap: 'wrap',
          gap: 1,
          boxSizing: 'border-box',
          maxWidth: '100%',
          overflowX: 'hidden',
        }}
      >
        <Button onClick={onClose} disabled={loading} color="inherit">
          Cancel
        </Button>
        <Button
          type="submit"
          form={formDomId}
          variant="contained"
          color="success"
          disabled={loading || !feedData.feed_url || !feedData.feed_name}
        >
          {loading ? 'Adding Feed...' : 'Add RSS Feed'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default RSSFeedManager;
