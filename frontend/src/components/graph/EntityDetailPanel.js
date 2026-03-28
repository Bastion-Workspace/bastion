/**
 * Entity detail side panel: shows document mentions, co-occurring entities, and "Ask AI" action.
 * Extracted from EntityRelationGraph for reuse in entity and unified graph views.
 */
import React from 'react';
import {
  Box,
  Button,
  Chip,
  IconButton,
  Typography,
  CircularProgress,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

export default function EntityDetailPanel({
  selectedEntity,
  entityDetail,
  detailLoading,
  onClose,
  onOpenDocument,
  onAskAI,
}) {
  if (!selectedEntity) return null;

  return (
    <Box
      sx={{
        width: 300,
        flexShrink: 0,
        borderLeft: '1px solid',
        borderColor: 'divider',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'background.paper',
        overflow: 'hidden',
      }}
    >
      <Box
        sx={{
          p: 1.5,
          borderBottom: '1px solid',
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 1,
        }}
      >
        <Box sx={{ minWidth: 0, flex: 1 }}>
          <Typography variant="subtitle2" noWrap>
            {selectedEntity.name}
          </Typography>
          <Chip
            size="small"
            label={(selectedEntity.entity_type || 'MISC').toUpperCase()}
            sx={{ mt: 0.5 }}
          />
        </Box>
        <IconButton size="small" onClick={onClose} aria-label="Close panel">
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>
      <Box sx={{ flex: 1, overflow: 'auto', p: 1.5 }}>
        {detailLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
            <CircularProgress size={24} />
          </Box>
        ) : entityDetail ? (
          <>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              Documents
            </Typography>
            {(entityDetail.document_mentions || []).map((m) => (
              <Box key={m.document_id} sx={{ mb: 1.5 }}>
                <Button
                  size="small"
                  variant="text"
                  sx={{ textAlign: 'left', textTransform: 'none', display: 'block', p: 0, minHeight: 0 }}
                  onClick={() => onOpenDocument?.(m.document_id, m.title || m.filename || m.document_id)}
                >
                  {m.title || m.filename || m.document_id}
                </Button>
                {m.context && (
                  <Typography
                    variant="caption"
                    sx={{
                      display: 'block',
                      fontStyle: 'italic',
                      color: 'text.secondary',
                      mt: 0.5,
                      pl: 0.5,
                      borderLeft: '2px solid',
                      borderColor: 'divider',
                    }}
                  >
                    &quot;{m.context.slice(0, 150)}{m.context.length > 150 ? '…' : ''}&quot;
                  </Typography>
                )}
              </Box>
            ))}
            {(entityDetail.co_occurring_entities || []).length > 0 && (
              <>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2, mb: 1 }}>
                  Co-occurring
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {(entityDetail.co_occurring_entities || []).map((co) => (
                    <Chip
                      key={co.name}
                      size="small"
                      label={co.name}
                      variant="outlined"
                      sx={{ fontSize: '0.7rem' }}
                    />
                  ))}
                </Box>
              </>
            )}
            {onAskAI && (
              <Button
                size="small"
                variant="contained"
                fullWidth
                sx={{ mt: 2 }}
                onClick={() => onAskAI(selectedEntity.name)}
              >
                Ask AI about this entity
              </Button>
            )}
          </>
        ) : null}
      </Box>
    </Box>
  );
}
