import React from 'react';
import { Box, Card, CardActionArea, Typography, Chip } from '@mui/material';
import { CheckCircle } from '@mui/icons-material';

export default function EmbyPosterCard({
  title,
  subtitle,
  imageUrl,
  played,
  progressRatio,
  onClick,
  onDoubleClick,
  height = 220,
}) {
  const pct =
    typeof progressRatio === 'number' && progressRatio > 0 && progressRatio < 1
      ? Math.round(progressRatio * 100)
      : null;

  return (
    <Card
      variant="outlined"
      sx={{
        height,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        position: 'relative',
        transition: 'transform 0.15s ease',
        '&:hover': { transform: 'scale(1.02)' },
      }}
    >
      <CardActionArea
        onClick={onClick}
        onDoubleClick={onDoubleClick}
        sx={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'stretch' }}
      >
        <Box
          sx={{
            flex: 1,
            minHeight: 140,
            background: 'linear-gradient(180deg, rgba(80,80,80,0.35), rgba(20,20,20,0.9))',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {imageUrl && (
            <Box
              component="img"
              src={imageUrl}
              alt=""
              loading="lazy"
              sx={{
                position: 'absolute',
                inset: 0,
                width: '100%',
                height: '100%',
                objectFit: 'cover',
              }}
            />
          )}
          {played && (
            <Chip
              icon={<CheckCircle sx={{ fontSize: 16 }} />}
              label="Watched"
              size="small"
              sx={{ position: 'absolute', top: 6, right: 6, bgcolor: 'rgba(0,0,0,0.55)', color: 'white' }}
            />
          )}
          {pct != null && (
            <LinearProgressBar pct={pct} />
          )}
        </Box>
        <Box sx={{ p: 1, pt: 0.5 }}>
          <Typography variant="body2" noWrap title={title}>
            {title || 'Untitled'}
          </Typography>
          {subtitle && (
            <Typography variant="caption" color="text.secondary" noWrap>
              {subtitle}
            </Typography>
          )}
        </Box>
      </CardActionArea>
    </Card>
  );
}

function LinearProgressBar({ pct }) {
  return (
    <Box
      sx={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        height: 4,
        bgcolor: 'rgba(0,0,0,0.4)',
      }}
    >
      <Box sx={{ height: '100%', width: `${pct}%`, bgcolor: 'primary.main' }} />
    </Box>
  );
}
