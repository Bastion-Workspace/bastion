import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from 'react-query';
import { Box, Typography, Paper, Button, Chip } from '@mui/material';
import apiService from '../services/apiService';

export default function NewsPage() {
  const navigate = useNavigate();
  const { data, isLoading, isError } = useQuery('newsHeadlines', () => apiService.get('/api/news/headlines'));
  const headlines = data?.headlines || [];

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 2 }}>News</Typography>
      {isLoading && <Typography variant="body2">Loading headlines...</Typography>}
      {isError && <Typography variant="body2" color="error">Failed to load headlines</Typography>}
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 2 }}>
        {headlines.map(h => (
          <Paper key={h.id} sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 1 }} variant="outlined">
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, justifyContent: 'space-between' }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>{h.title}</Typography>
              <Chip size="small" label={h.severity.toUpperCase()} color={h.severity === 'breaking' ? 'error' : h.severity === 'urgent' ? 'warning' : 'default'} />
            </Box>
            <Typography variant="body2" color="text.secondary">{h.summary}</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip size="small" label={`${h.sources_count} sources`} />
              <Chip size="small" label={`diversity ${(h.diversity_score*100).toFixed(0)}%`} />
            </Box>
            <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
              <Button size="small" variant="contained" onClick={() => navigate(`/news/${h.id}`)}>Open</Button>
            </Box>
          </Paper>
        ))}
      </Box>
    </Box>
  );
}


