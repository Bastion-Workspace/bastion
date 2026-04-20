import React, { useEffect, useState } from 'react';
import { useParams, useSearchParams, Link as RouterLink } from 'react-router-dom';
import { Box, Typography, Button, CircularProgress, Chip } from '@mui/material';
import ArtifactRenderer from './chat/ArtifactRenderer';
import savedArtifactService from '../services/savedArtifactService';

/**
 * Standalone page for shared artifacts (no app chrome). Optional ?embed=1 for iframe embeds.
 */
export default function PublicArtifactPage() {
  const { shareToken } = useParams();
  const [searchParams] = useSearchParams();
  const embed = searchParams.get('embed') === '1';

  const [artifact, setArtifact] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError('');
    savedArtifactService
      .fetchPublic(shareToken)
      .then((data) => {
        if (!cancelled) {
          setArtifact({
            artifact_type: data.artifact_type,
            title: data.title,
            code: data.code,
            language: data.language || undefined,
          });
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e?.message || 'Could not load this shared artifact.');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [shareToken]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error || !artifact) {
    return (
      <Box sx={{ p: 3, maxWidth: 480, mx: 'auto', mt: 4 }}>
        <Typography color="error" variant="body1">
          {error || 'Not found'}
        </Typography>
      </Box>
    );
  }

  if (embed) {
    return (
      <Box sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
        <ArtifactRenderer artifact={artifact} height="100%" />
      </Box>
    );
  }

  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', bgcolor: 'background.default' }}>
      <Box
        sx={{
          flexShrink: 0,
          px: 2,
          py: 1,
          borderBottom: 1,
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          flexWrap: 'wrap',
        }}
      >
        <Typography variant="subtitle1" sx={{ flex: 1, minWidth: 0 }} noWrap>
          {artifact.title || 'Shared artifact'}
        </Typography>
        <Chip size="small" label={String(artifact.artifact_type || '').toUpperCase()} />
        <Button component={RouterLink} to="/login" size="small" variant="outlined">
          Open in Bastion
        </Button>
      </Box>
      <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', p: 1 }}>
        <ArtifactRenderer artifact={artifact} height="100%" />
      </Box>
    </Box>
  );
}
