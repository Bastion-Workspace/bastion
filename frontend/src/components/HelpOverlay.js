import React, { useState, useEffect, useMemo } from 'react';
import {
  Dialog,
  DialogContent,
  Box,
  List,
  ListItemButton,
  ListItemText,
  ListSubheader,
  Typography,
  IconButton,
  Paper,
  useTheme,
  CircularProgress,
  Skeleton,
  TextField,
  InputAdornment,
} from '@mui/material';
import {
  Close,
  OpenInNew,
  PushPin,
  Minimize,
  CropSquare,
  DragIndicator,
  Search as SearchIcon,
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkBreaks from 'remark-breaks';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';
import { useQuery } from 'react-query';
import { motion, useDragControls, useMotionValue } from 'framer-motion';

import apiService from '../services/apiService';

const STORAGE_KEYS = {
  floating: 'help_panel_floating',
  x: 'help_panel_x',
  y: 'help_panel_y',
  minimized: 'help_panel_minimized',
};

const FLOATING_WIDTH = 580;
const FLOATING_HEIGHT = 520;
const FLOATING_HEADER_HEIGHT = 56;
const FLOATING_MARGIN = 16;

const readStoredBoolean = (key, fallback = false) => {
  if (typeof window === 'undefined') {
    return fallback;
  }

  const raw = window.localStorage.getItem(key);
  if (raw === null) {
    return fallback;
  }

  return raw === 'true';
};

const readStoredNumber = (key, fallback = null) => {
  if (typeof window === 'undefined') {
    return fallback;
  }

  const raw = window.localStorage.getItem(key);
  if (raw === null) {
    return fallback;
  }

  const parsed = Number.parseFloat(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

const HelpPanelContent = ({
  theme,
  topics,
  topicsLoading,
  selectedTopic,
  onTopicSelect,
  contentLoading,
  displayContent,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const searchLower = searchQuery.trim().toLowerCase();
  const filteredTopics = useMemo(() => {
    if (!searchLower) return topics;
    return topics.filter((t) => {
      const title = (t.title || '').toLowerCase();
      const category = (t.category || 'General').toLowerCase();
      return title.includes(searchLower) || category.includes(searchLower);
    });
  }, [topics, searchLower]);

  return (
  <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
    <Box
      sx={{
        width: 250,
        borderRight: '1px solid',
        borderColor: 'divider',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
      }}
    >
      <TextField
        size="small"
        placeholder="Search help…"
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon fontSize="small" color="action" />
            </InputAdornment>
          ),
        }}
        sx={{
          m: 1,
          flexShrink: 0,
          '& .MuiOutlinedInput-root': { backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)' },
        }}
      />
      <Box sx={{ overflowY: 'auto', flex: 1, minHeight: 0 }}>
      {topicsLoading ? (
        <Box sx={{ p: 2, display: 'flex', justifyContent: 'center' }}>
          <CircularProgress size={24} />
        </Box>
      ) : (
        <List sx={{ py: 0 }}>
          {(() => {
            const byCategory = filteredTopics.reduce((acc, topic) => {
              const category = topic.category || 'General';
              if (!acc[category]) acc[category] = [];
              acc[category].push(topic);
              return acc;
            }, {});

            const categories = Object.keys(byCategory).sort((a, b) => {
              if (a === 'General') return -1;
              if (b === 'General') return 1;
              return a.localeCompare(b);
            });

            return categories.map((category) => (
              <React.Fragment key={category}>
                <ListSubheader
                  disableSticky
                  sx={{
                    lineHeight: 2,
                    fontWeight: 600,
                    background: 'transparent',
                  }}
                >
                  {category}
                </ListSubheader>
                {byCategory[category].map((topic) => (
                  <ListItemButton
                    key={topic.id}
                    selected={selectedTopic === topic.id}
                    onClick={() => onTopicSelect(topic.id)}
                    sx={{
                      pl: 2,
                      '&.Mui-selected': {
                        backgroundColor:
                          theme.palette.mode === 'dark'
                            ? 'rgba(255, 255, 255, 0.08)'
                            : 'rgba(0, 0, 0, 0.04)',
                        '&:hover': {
                          backgroundColor:
                            theme.palette.mode === 'dark'
                              ? 'rgba(255, 255, 255, 0.12)'
                              : 'rgba(0, 0, 0, 0.08)',
                        },
                      },
                    }}
                  >
                    <ListItemText primary={topic.title} />
                  </ListItemButton>
                ))}
              </React.Fragment>
            ));
          })()}
        </List>
      )}
      </Box>
    </Box>

    <Box
      sx={{
        flex: 1,
        overflowY: 'auto',
        p: 3,
      }}
    >
      {!selectedTopic ? (
        <Typography variant="body2" color="text.secondary">
          Select a topic from the sidebar
        </Typography>
      ) : contentLoading ? (
        <Box>
          <Skeleton variant="text" width="60%" height={32} />
          <Skeleton variant="text" width="100%" />
          <Skeleton variant="text" width="100%" />
          <Skeleton variant="text" width="80%" />
        </Box>
      ) : displayContent ? (
        <ReactMarkdown
          remarkPlugins={[remarkBreaks, remarkGfm]}
          rehypePlugins={[rehypeRaw, rehypeSanitize]}
          components={{
            a: ({ node, ...props }) => (
              <a {...props} target="_blank" rel="noopener noreferrer" />
            ),
          }}
        >
          {displayContent}
        </ReactMarkdown>
      ) : (
        <Typography variant="body2" color="text.secondary">
          No content for this topic
        </Typography>
      )}
    </Box>
  </Box>
  );
};

const HelpOverlay = ({ open, onClose }) => {
  const theme = useTheme();
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [isFloating, setIsFloating] = useState(() => readStoredBoolean(STORAGE_KEYS.floating, false));
  const [isMinimized, setIsMinimized] = useState(() => readStoredBoolean(STORAGE_KEYS.minimized, false));
  const [savedX, setSavedX] = useState(() => readStoredNumber(STORAGE_KEYS.x, null));
  const [savedY, setSavedY] = useState(() => readStoredNumber(STORAGE_KEYS.y, null));

  const dragControls = useDragControls();
  const x = useMotionValue(0);
  const y = useMotionValue(0);

  const { data: versionData } = useQuery(
    'appVersion',
    () => apiService.get('/api/version'),
    { enabled: open, staleTime: 5 * 60 * 1000 }
  );
  const appVersion = versionData?.version ?? '…';

  const { data: topics = [], isLoading: topicsLoading } = useQuery(
    'helpTopics',
    () => apiService.get('/api/help/topics'),
    { enabled: open, staleTime: 5 * 60 * 1000 }
  );

  const { data: topicContent, isLoading: contentLoading } = useQuery(
    ['helpTopicContent', selectedTopic],
    () => apiService.get(`/api/help/topics/${selectedTopic}`),
    { enabled: open && !!selectedTopic, staleTime: 5 * 60 * 1000 }
  );

  useEffect(() => {
    if (open && topics.length > 0 && !selectedTopic) {
      setSelectedTopic(topics[0].id);
    }
  }, [open, topics, selectedTopic]);

  const dragBounds = useMemo(() => {
    if (typeof window === 'undefined') {
      return { top: 0, left: 0, right: 0, bottom: 0 };
    }

    const panelHeight = isMinimized ? FLOATING_HEADER_HEIGHT : FLOATING_HEIGHT;
    return {
      top: 0,
      left: 0,
      right: Math.max(0, window.innerWidth - FLOATING_WIDTH),
      bottom: Math.max(0, window.innerHeight - panelHeight),
    };
  }, [isMinimized]);

  useEffect(() => {
    if (!open || !isFloating) {
      return;
    }

    const panelHeight = isMinimized ? FLOATING_HEADER_HEIGHT : FLOATING_HEIGHT;
    const maxX = Math.max(0, window.innerWidth - FLOATING_WIDTH);
    const maxY = Math.max(0, window.innerHeight - panelHeight);
    const defaultX = Math.max(FLOATING_MARGIN, maxX - 60);
    const defaultY = Math.max(FLOATING_MARGIN, maxY - 60);
    const targetX = clamp(savedX ?? defaultX, 0, maxX);
    const targetY = clamp(savedY ?? defaultY, 0, maxY);

    x.set(targetX);
    y.set(targetY);

    setSavedX(targetX);
    setSavedY(targetY);
  }, [open, isFloating, isMinimized, savedX, savedY, x, y]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const handleResize = () => {
      const panelHeight = isMinimized ? FLOATING_HEADER_HEIGHT : FLOATING_HEIGHT;
      const maxX = Math.max(0, window.innerWidth - FLOATING_WIDTH);
      const maxY = Math.max(0, window.innerHeight - panelHeight);
      const currentX = clamp(x.get(), 0, maxX);
      const currentY = clamp(y.get(), 0, maxY);

      x.set(currentX);
      y.set(currentY);
      setSavedX(currentX);
      setSavedY(currentY);
      window.localStorage.setItem(STORAGE_KEYS.x, String(currentX));
      window.localStorage.setItem(STORAGE_KEYS.y, String(currentY));
    };

    if (open && isFloating) {
      window.addEventListener('resize', handleResize);
    }

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [open, isFloating, isMinimized, x, y]);

  const handleTopicSelect = (topicId) => {
    setSelectedTopic(topicId);
  };

  const saveFloatingState = (nextValue) => {
    setIsFloating(nextValue);
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_KEYS.floating, String(nextValue));
    }
  };

  const saveMinimizedState = (nextValue) => {
    setIsMinimized(nextValue);
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_KEYS.minimized, String(nextValue));
    }
  };

  const handleFloatingDragStart = (event) => {
    dragControls.start(event);
  };

  const handleFloatingDragEnd = () => {
    const nextX = x.get();
    const nextY = y.get();
    setSavedX(nextX);
    setSavedY(nextY);

    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_KEYS.x, String(nextX));
      window.localStorage.setItem(STORAGE_KEYS.y, String(nextY));
    }
  };

  const handlePopOut = () => {
    saveFloatingState(true);
  };

  const handleDockBack = () => {
    saveFloatingState(false);
    saveMinimizedState(false);
  };

  const toggleMinimized = () => {
    saveMinimizedState(!isMinimized);
  };

  const displayContent = topicContent?.content
    ? topicContent.content.replace(/\{\{VERSION\}\}/g, appVersion)
    : '';

  if (!open) {
    return null;
  }

  if (isFloating) {
    return (
      <motion.div
        drag
        dragControls={dragControls}
        dragListener={false}
        dragMomentum={false}
        dragElastic={0}
        dragConstraints={dragBounds}
        onDragEnd={handleFloatingDragEnd}
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          zIndex: 1300,
          width: FLOATING_WIDTH,
          height: isMinimized ? FLOATING_HEADER_HEIGHT : FLOATING_HEIGHT,
          x,
          y,
        }}
      >
        <Paper
          elevation={8}
          sx={{
            width: '100%',
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            border: '1px solid',
            borderColor: 'divider',
          }}
        >
          <Box
            onPointerDown={handleFloatingDragStart}
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              px: 1,
              py: 1,
              borderBottom: isMinimized ? 'none' : '1px solid',
              borderColor: 'divider',
              cursor: 'grab',
              userSelect: 'none',
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <DragIndicator fontSize="small" />
              <Typography variant="h6">Help</Typography>
            </Box>
            <Box
              onPointerDown={(event) => event.stopPropagation()}
              sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}
            >
              <IconButton size="small" onClick={toggleMinimized} aria-label={isMinimized ? 'Restore help panel' : 'Minimize help panel'}>
                {isMinimized ? <CropSquare fontSize="small" /> : <Minimize fontSize="small" />}
              </IconButton>
              <IconButton size="small" onClick={handleDockBack} aria-label="Dock help panel">
                <PushPin fontSize="small" />
              </IconButton>
              <IconButton size="small" onClick={onClose} aria-label="Close help panel">
                <Close fontSize="small" />
              </IconButton>
            </Box>
          </Box>

          {!isMinimized && (
            <HelpPanelContent
              theme={theme}
              topics={topics}
              topicsLoading={topicsLoading}
              selectedTopic={selectedTopic}
              onTopicSelect={handleTopicSelect}
              contentLoading={contentLoading}
              displayContent={displayContent}
            />
          )}
        </Paper>
      </motion.div>
    );
  }

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: {
          height: '80vh',
          maxHeight: '80vh',
        }
      }}
    >
      <DialogContent sx={{ p: 0, height: '100%', display: 'flex', flexDirection: 'column' }}>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            px: 2,
            py: 1.5,
            borderBottom: '1px solid',
            borderColor: 'divider',
          }}
        >
          <Typography variant="h6">Help</Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <IconButton onClick={handlePopOut} aria-label="Pop out help panel">
              <OpenInNew fontSize="small" />
            </IconButton>
            <IconButton onClick={onClose} aria-label="Close help panel">
              <Close />
            </IconButton>
          </Box>
        </Box>

        <HelpPanelContent
          theme={theme}
          topics={topics}
          topicsLoading={topicsLoading}
          selectedTopic={selectedTopic}
          onTopicSelect={handleTopicSelect}
          contentLoading={contentLoading}
          displayContent={displayContent}
        />
      </DialogContent>
    </Dialog>
  );
};

export default HelpOverlay;
