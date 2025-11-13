import React from 'react';
import { useQuery } from 'react-query';
import { Box, Typography, Paper, Button, Chip, Tooltip, CircularProgress } from '@mui/material';
import apiService from '../services/apiService';

export default function NewsHeadlinesPane({ onOpenArticle }) {
  const { data, isLoading, isError } = useQuery('newsHeadlines', () => apiService.get('/api/news/headlines'));
  const headlines = data?.headlines || [];
  const [purging, setPurging] = React.useState(false);

  const breaking = headlines.filter(h => h.severity === 'breaking').length;
  const urgent = headlines.filter(h => h.severity === 'urgent').length;

  return (
    <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 2, height: '100%', overflow: 'auto' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h6" sx={{ fontWeight: 700 }}>News Headlines</Typography>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <Chip size="small" label={`breaking ${breaking}`} color="error" />
          <Chip size="small" label={`urgent ${urgent}`} color="warning" />
          <Chip size="small" label={`total ${headlines.length}`} />
          <Tooltip title="Delete all news articles (admin)">
            <span>
              <Button size="small" color="error" variant="outlined" disabled={purging} onClick={async () => {
                if (!window.confirm('Delete ALL news articles? This cannot be undone.')) return;
                try {
                  setPurging(true);
                  await apiService.delete('/api/news/purge');
                  // Refetch headlines after purge
                  await apiService.get('/api/news/headlines');
                  window.dispatchEvent(new CustomEvent('news-purge-complete'));
                } finally {
                  setPurging(false);
                }
              }}>
                {purging ? <CircularProgress size={14} sx={{ mr: 1 }} /> : null}
                Purge All
              </Button>
            </span>
          </Tooltip>
        </Box>
      </Box>

      {isLoading && <Typography variant="body2">Loading headlines...</Typography>}
      {isError && <Typography variant="body2" color="error">Failed to load headlines</Typography>}

      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 2 }}>
        {headlines.map(h => (
          <Paper key={h.id} sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 1 }} variant="outlined">
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, justifyContent: 'space-between' }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>{h.title}</Typography>
              <Chip size="small" label={(h.severity || 'news').toUpperCase()} color={h.severity === 'breaking' ? 'error' : h.severity === 'urgent' ? 'warning' : 'default'} />
            </Box>
            <Typography variant="body2" color="text.secondary">{h.summary}</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip size="small" label={`${h.sources_count || 0} sources`} />
              {typeof h.diversity_score === 'number' && <Chip size="small" label={`diversity ${Math.round((h.diversity_score||0)*100)}%`} />}
            </Box>
            <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
              <Button size="small" variant="contained" onClick={() => {
                if (typeof onOpenArticle === 'function') onOpenArticle(h.id);
                else {
                  try {
                    if (window?.history && typeof window.history.pushState === 'function') {
                      window.history.pushState({}, '', `/news/${h.id}`);
                      window.dispatchEvent(new PopStateEvent('popstate'));
                    } else {
                      window.location.href = `/news/${h.id}`;
                    }
                  } catch {
                    window.location.href = `/news/${h.id}`;
                  }
                }
              }}>Open</Button>
            </Box>
          </Paper>
        ))}
      </Box>
    </Box>
  );
}


