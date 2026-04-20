import React, { useState, useEffect } from 'react';
import { useQuery } from 'react-query';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Collapse,
  Divider,
} from '@mui/material';
import { ExpandMore as ExpandMoreIcon, ExpandLess as ExpandLessIcon } from '@mui/icons-material';
import ebooksService from '../services/ebooksService';

const STORAGE_KEY = 'ebooksOpdsSidebarExpanded';

/**
 * Documents sidebar segment for OPDS ebook catalogs (opens hub in main tabs).
 * Shown only when the user has saved at least one catalog in Settings.
 */
export default function EbooksOpdsSidebarSection({ onOpenHub }) {
  const { data, isLoading, isError } = useQuery(['ebooks-settings'], () => ebooksService.getSettings(), {
    staleTime: 30_000,
  });

  const [expanded, setExpanded] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved !== null ? JSON.parse(saved) : true;
    } catch {
      return true;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(expanded));
    } catch {
      /* ignore */
    }
  }, [expanded]);

  const hasCatalogs = Array.isArray(data?.catalogs) && data.catalogs.length > 0;
  if (isLoading || isError || !hasCatalogs) {
    return null;
  }

  return (
    <>
      <Divider sx={{ my: 0.5 }} />
      <Box sx={{ px: 2, pb: 0.5 }}>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            flex: 1,
            minWidth: 0,
            cursor: 'pointer',
            '&:hover': { backgroundColor: 'action.hover' },
            borderRadius: 1,
            p: 0.5,
            mb: 0.5,
          }}
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
          <Typography
            variant="subtitle2"
            sx={{
              fontWeight: 600,
              color: 'text.secondary',
              fontSize: '0.75rem',
              textTransform: 'uppercase',
              ml: 0.5,
            }}
          >
            Ebooks
          </Typography>
        </Box>

        <Collapse in={expanded} timeout="auto">
          <List dense sx={{ py: 0 }}>
            <ListItem disablePadding>
              <ListItemButton
                onClick={() => onOpenHub?.()}
                sx={{
                  borderRadius: 1,
                  minHeight: 36,
                  pl: 0.5,
                  '&:hover': { backgroundColor: 'action.hover' },
                }}
              >
                <ListItemText primary={<Typography variant="body2">OPDS catalogs</Typography>} />
              </ListItemButton>
            </ListItem>
          </List>
        </Collapse>
      </Box>
    </>
  );
}
