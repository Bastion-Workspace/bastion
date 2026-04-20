import React from 'react';
import { Box, Paper, Typography, Button } from '@mui/material';
import { OpenInNew } from '@mui/icons-material';
import { artifactTypeIcon } from './artifactTypeIcons';

/**
 * Compact card in the message list; opens the artifact drawer when clicked.
 */
const ArtifactCard = ({
  artifact,
  onOpen,
  activeArtifact = null,
  artifactCollapsed = false,
}) => {
  if (!artifact?.artifact_type) return null;
  const title = artifact.title || 'Artifact';
  const isActive =
    Boolean(activeArtifact && activeArtifact.code === artifact.code);
  const expandCollapsed = isActive && artifactCollapsed;
  const buttonLabel = expandCollapsed ? 'Expand' : 'Open';

  return (
    <Paper
      variant="outlined"
      sx={{
        p: 1.5,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 1,
        borderRadius: 1,
        bgcolor: 'action.hover',
        ...(isActive
          ? {
              borderColor: 'primary.main',
              borderWidth: 2,
              boxShadow: (theme) =>
                theme.palette.mode === 'dark'
                  ? '0 0 0 1px rgba(144, 202, 249, 0.35)'
                  : '0 0 0 1px rgba(25, 118, 210, 0.25)',
            }
          : {}),
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0, flex: 1 }}>
        {artifactTypeIcon(artifact.artifact_type)}
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="subtitle2" noWrap title={title}>
            {title}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {String(artifact.artifact_type).toUpperCase()} artifact
          </Typography>
        </Box>
      </Box>
      <Button
        size="small"
        variant={expandCollapsed ? 'outlined' : 'contained'}
        color="primary"
        startIcon={<OpenInNew />}
        onClick={() => onOpen && onOpen(artifact)}
        sx={{ flexShrink: 0, textTransform: 'none' }}
      >
        {buttonLabel}
      </Button>
    </Paper>
  );
};

export default ArtifactCard;
