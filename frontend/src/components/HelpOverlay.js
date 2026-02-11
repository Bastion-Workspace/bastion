import React, { useState, useEffect, useMemo } from 'react';
import {
  Dialog,
  DialogContent,
  Box,
  List,
  ListItemButton,
  ListItemText,
  Typography,
  IconButton,
  Divider,
  useTheme,
} from '@mui/material';
import { Close } from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkBreaks from 'remark-breaks';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';

// Import help topics
import { helpTopics as importedHelpTopics } from '../help/index';

const HelpOverlay = ({ open, onClose }) => {
  const theme = useTheme();
  const [selectedTopic, setSelectedTopic] = useState(null);

  // Set first topic as selected when dialog opens
  useEffect(() => {
    if (open && importedHelpTopics.length > 0) {
      setSelectedTopic(importedHelpTopics[0].id);
    }
  }, [open]);

  const selectedContent = useMemo(() => {
    if (!selectedTopic) return null;
    return importedHelpTopics.find(topic => topic.id === selectedTopic);
  }, [selectedTopic]);

  const handleTopicSelect = (topicId) => {
    setSelectedTopic(topicId);
  };

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
        {/* Header with close button */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            p: 2,
            borderBottom: '1px solid',
            borderColor: 'divider',
            position: 'relative',
          }}
        >
          <Typography variant="h6">Help</Typography>
          <IconButton
            onClick={onClose}
            sx={{
              position: 'absolute',
              right: 8,
              top: 8,
            }}
          >
            <Close />
          </IconButton>
        </Box>

        {/* Main content area */}
        <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          {/* Sidebar */}
          <Box
            sx={{
              width: 250,
              borderRight: '1px solid',
              borderColor: 'divider',
              overflowY: 'auto',
              flexShrink: 0,
            }}
          >
            <List sx={{ py: 0 }}>
              {importedHelpTopics.map((topic) => (
                  <React.Fragment key={topic.id}>
                    <ListItemButton
                      selected={selectedTopic === topic.id}
                      onClick={() => handleTopicSelect(topic.id)}
                      sx={{
                        '&.Mui-selected': {
                          backgroundColor: theme.palette.mode === 'dark'
                            ? 'rgba(255, 255, 255, 0.08)'
                            : 'rgba(0, 0, 0, 0.04)',
                          '&:hover': {
                            backgroundColor: theme.palette.mode === 'dark'
                              ? 'rgba(255, 255, 255, 0.12)'
                              : 'rgba(0, 0, 0, 0.08)',
                          },
                        },
                      }}
                    >
                      <ListItemText primary={topic.title} />
                    </ListItemButton>
                    <Divider />
                  </React.Fragment>
                ))}
            </List>
          </Box>

          {/* Content area */}
          <Box
            sx={{
              flex: 1,
              overflowY: 'auto',
              p: 3,
            }}
          >
            {selectedContent ? (
              <ReactMarkdown
                remarkPlugins={[remarkBreaks, remarkGfm]}
                rehypePlugins={[rehypeRaw, rehypeSanitize]}
                components={{
                  a: ({ node, ...props }) => (
                    <a {...props} target="_blank" rel="noopener noreferrer" />
                  ),
                }}
              >
                {selectedContent.content}
              </ReactMarkdown>
            ) : (
              <Typography variant="body2" color="text.secondary">
                Select a topic from the sidebar
              </Typography>
            )}
          </Box>
        </Box>
      </DialogContent>
    </Dialog>
  );
};

export default HelpOverlay;
