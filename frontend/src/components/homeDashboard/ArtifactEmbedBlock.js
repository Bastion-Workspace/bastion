import React from 'react';
import { useQuery } from 'react-query';
import { Box, CircularProgress, Typography } from '@mui/material';
import ArtifactRenderer from '../chat/ArtifactRenderer';
import savedArtifactService from '../../services/savedArtifactService';

/**
 * Dashboard widget: loads a saved artifact by ID and renders with ArtifactRenderer.
 */
export default function ArtifactEmbedBlock({ config }) {
  const artifactId = config?.artifact_id || null;

  const { data, isLoading, error } = useQuery(
    ['savedArtifactFull', artifactId],
    () => savedArtifactService.get(artifactId),
    {
      enabled: Boolean(artifactId),
      staleTime: 2 * 60 * 1000,
    }
  );

  if (!artifactId) {
    return (
      <Typography variant="body2" color="text.secondary">
        Edit this widget and choose a saved artifact.
      </Typography>
    );
  }

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={2}>
        <CircularProgress size={28} />
      </Box>
    );
  }

  if (error || !data) {
    return (
      <Typography variant="body2" color="error">
        Could not load saved artifact. It may have been deleted.
      </Typography>
    );
  }

  const artifact = {
    artifact_type: data.artifact_type,
    title: data.title,
    code: data.code,
    language: data.language || undefined,
  };

  return (
    <Box sx={{ height: '100%', minHeight: 200, display: 'flex', flexDirection: 'column' }}>
      <ArtifactRenderer artifact={artifact} artifactId={artifactId} height="100%" />
    </Box>
  );
}
