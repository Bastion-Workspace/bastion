import React from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from 'react-query';
import { Box, Typography, Paper, Chip, Link as MuiLink } from '@mui/material';
import apiService from '../services/apiService';

export default function NewsDetailPage() {
  const { newsId } = useParams();
  const { data, isLoading, isError } = useQuery(['newsDetail', newsId], () => apiService.get(`/api/news/${newsId}`), { enabled: !!newsId });
  const article = data || null;

  return (
    <Box>
      {isLoading && <Typography variant="body2">Loading article...</Typography>}
      {isError && <Typography variant="body2" color="error">Failed to load article</Typography>}
      {article && (
        <Paper variant="outlined" sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, justifyContent: 'space-between' }}>
            <Typography variant="h5" sx={{ fontWeight: 700 }}>{article.title}</Typography>
            <Chip size="small" label={article.severity?.toUpperCase() || 'NEWS'} color={article.severity === 'breaking' ? 'error' : article.severity === 'urgent' ? 'warning' : 'default'} />
          </Box>
          <Typography variant="subtitle1" color="text.secondary">{article.lede}</Typography>
          <Box component="div" sx={{ whiteSpace: 'pre-wrap' }}>
            <Typography variant="body1">{article.balanced_body}</Typography>
          </Box>

          {Array.isArray(article.key_points) && article.key_points.length > 0 && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>Key Points</Typography>
              <ul style={{ marginTop: 0 }}>
                {article.key_points.map((kp, idx) => (
                  <li key={idx}><Typography variant="body2">{kp}</Typography></li>
                ))}
              </ul>
            </Box>
          )}

          {Array.isArray(article.citations) && article.citations.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>Citations</Typography>
              <ul style={{ marginTop: 0 }}>
                {article.citations.map((c, idx) => (
                  <li key={idx}>
                    <Typography variant="body2">
                      {(c.name || 'Source')} — {c.published_at ? new Date(c.published_at).toLocaleString() : ''}
                      {c.url && (
                        <MuiLink href={c.url} target="_blank" rel="noopener noreferrer" sx={{ ml: 1 }}>
                          Open original
                        </MuiLink>
                      )}
                    </Typography>
                  </li>
                ))}
              </ul>
            </Box>
          )}
        </Paper>
      )}
    </Box>
  );
}


