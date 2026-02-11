import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  IconButton,
  TextField,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Button,
  Divider,
  Chip,
  CircularProgress,
  Alert,
  ClickAwayListener,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar,
} from '@mui/material';
import {
  Close,
  Search,
  Add,
  PushPin,
  Chat,
  MoreVert,
  Delete,
} from '@mui/icons-material';
import { useQuery, useQueryClient } from 'react-query';
import apiService from '../services/apiService';

const FloatingHistoryWindow = ({ 
  onClose, 
  onSelectConversation, 
  onNewChat,
  onClearCurrentConversation, // Add prop to handle clearing current conversation
  activeConversationId,
  anchorEl,
  anchorOrigin = { vertical: 'top', horizontal: 'left' },
  transformOrigin = { vertical: 'top', horizontal: 'right' }
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredConversations, setFilteredConversations] = useState([]);
  const [menuAnchorEl, setMenuAnchorEl] = useState(null);
  const [selectedConversation, setSelectedConversation] = useState(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteAllDialogOpen, setDeleteAllDialogOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isDeletingAll, setIsDeletingAll] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

  const queryClient = useQueryClient();

  // Fetch conversations
  const { data: conversationsData, isLoading, error } = useQuery(
    ['conversations'],
    () => apiService.listConversations(0, 50),
    {
      refetchOnWindowFocus: false,
      staleTime: 300000, // 5 minutes
    }
  );

  // Extract conversations from the response
  const conversations = conversationsData?.conversations || [];

  // Filter conversations based on search query
  useEffect(() => {
    if (!conversations) return;
    
    const filtered = conversations.filter(conversation => {
      const searchLower = searchQuery.toLowerCase();
      const title = conversation.title || 'Untitled Conversation';
      const preview = conversation.metadata_json?.last_message_preview?.content || '';
      
      return title.toLowerCase().includes(searchLower) || 
             preview.toLowerCase().includes(searchLower);
    });
    
    setFilteredConversations(filtered);
  }, [conversations, searchQuery]);

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    
    const date = new Date(timestamp);
    const now = new Date();
    const diffInHours = (now - date) / (1000 * 60 * 60);
    
    if (diffInHours < 1) {
      return 'Just now';
    } else if (diffInHours < 24) {
      return `${Math.floor(diffInHours)}h ago`;
    } else if (diffInHours < 168) { // 7 days
      return `${Math.floor(diffInHours / 24)}d ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  const formatLastActivity = (conversation) => {
    // ROOSEVELT'S TIMESTAMP FIX: Show last activity time instead of empty preview
    const lastActivity = conversation.updated_at || conversation.created_at;
    if (!lastActivity) return '';
    
    const date = new Date(lastActivity);
    const now = new Date();
    const diffInMinutes = Math.floor((now - date) / (1000 * 60));
    const diffInHours = Math.floor(diffInMinutes / 60);
    const diffInDays = Math.floor(diffInHours / 24);
    
    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
    if (diffInHours < 24) return `${diffInHours}h ago`;
    if (diffInDays < 7) return `${diffInDays}d ago`;
    
    return date.toLocaleDateString();
  };

  const handleClickAway = () => {
    // Don't close the window if we're in the middle of deleting a conversation
    if (!isDeleting && !deleteDialogOpen) {
      onClose();
    }
  };

  const handleMenuOpen = (event, conversation) => {
    event.stopPropagation();
    setMenuAnchorEl(event.currentTarget);
    setSelectedConversation(conversation);
  };

  const handleMenuClose = () => {
    setMenuAnchorEl(null);
    setSelectedConversation(null);
  };

  const handleDeleteClick = () => {
    // Close the menu but don't clear the selected conversation yet
    setMenuAnchorEl(null);
    
    // Open the delete dialog
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!selectedConversation) {
      console.error('❌ No selected conversation for deletion');
      alert('Error: No conversation selected for deletion');
      return;
    }
    
    setIsDeleting(true);
    try {
      const result = await apiService.deleteConversation(selectedConversation.conversation_id);
      
      // Invalidate and refetch conversations
      await queryClient.invalidateQueries(['conversations']);
      
      setSnackbar({
        open: true,
        message: 'Conversation deleted successfully',
        severity: 'success'
      });
      
      setDeleteDialogOpen(false);
      setSelectedConversation(null);
      
      // CRITICAL: Clear current conversation if it was the one being deleted
      if (onClearCurrentConversation) {
        onClearCurrentConversation();
      }
      
      // Add a small delay to prevent ClickAwayListener from firing during DOM updates
      setTimeout(() => {
        setIsDeleting(false);
      }, 100);
    } catch (error) {
      console.error('Failed to delete conversation:', error);
      console.error('Error details:', error.response?.data);
      setSnackbar({
        open: true,
        message: `Failed to delete conversation: ${error.response?.data?.detail || error.message}`,
        severity: 'error'
      });
    } finally {
      // Add a small delay to prevent ClickAwayListener from firing during DOM updates
      setTimeout(() => {
        setIsDeleting(false);
      }, 100);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteDialogOpen(false);
    setSelectedConversation(null);
  };

  const handleDeleteAllClick = () => {
    setDeleteAllDialogOpen(true);
  };

  const handleDeleteAllConfirm = async () => {
    setIsDeletingAll(true);
    try {
      const currentConv = conversations.find(c => c.conversation_id === activeConversationId);
      const isCurrentEmpty =
        activeConversationId &&
        currentConv &&
        (currentConv.message_count === 0 || currentConv.message_count == null);

      if (isCurrentEmpty) {
        const toDelete = conversations.filter(c => c.conversation_id !== activeConversationId);
        await Promise.all(toDelete.map(c => apiService.deleteConversation(c.conversation_id)));
      } else {
        await apiService.deleteAllConversations();
      }

      await queryClient.invalidateQueries(['conversations']);

      setSnackbar({
        open: true,
        message: 'All conversations deleted successfully',
        severity: 'success'
      });

      setDeleteAllDialogOpen(false);

      if (!isCurrentEmpty) {
        if (onClearCurrentConversation) onClearCurrentConversation();
        if (onNewChat) onNewChat();
      }

      onClose();
    } catch (error) {
      console.error('Failed to delete all conversations:', error);
      setSnackbar({
        open: true,
        message: `Failed to delete all conversations: ${error.response?.data?.detail || error.message}`,
        severity: 'error'
      });
    } finally {
      setIsDeletingAll(false);
    }
  };

  const handleDeleteAllCancel = () => {
    setDeleteAllDialogOpen(false);
  };

  const handleSnackbarClose = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  if (error) {
    return (
      <ClickAwayListener onClickAway={handleClickAway}>
        <Box
          sx={{
            position: 'fixed',
            top: '20px',
            right: '20px',
            width: '320px',
            maxHeight: '500px',
            backgroundColor: 'background.paper',
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: 2,
            boxShadow: 3,
            zIndex: 1000,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <Alert severity="error" sx={{ m: 2 }}>
            Failed to load conversations: {error.message}
          </Alert>
        </Box>
      </ClickAwayListener>
    );
  }

  return (
    <>
      <ClickAwayListener onClickAway={handleClickAway}>
        <Box
          sx={{
            position: 'fixed',
            top: '20px',
            right: '20px',
            width: '320px',
            maxHeight: '500px',
            backgroundColor: 'background.paper',
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: 2,
            boxShadow: 3,
            zIndex: 1000,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {/* Header */}
          <Box sx={{ 
            p: 2, 
            borderBottom: 1, 
            borderColor: 'divider',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <Typography variant="h6" sx={{ fontWeight: 500 }}>
              Chat History
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {filteredConversations.length > 0 && (
                <Button
                  variant="outlined"
                  size="small"
                  color="error"
                  startIcon={<Delete />}
                  onClick={handleDeleteAllClick}
                  sx={{ 
                    minWidth: 'auto',
                    px: 1,
                    py: 0.5,
                    fontSize: '0.75rem'
                  }}
                >
                  Clear All
                </Button>
              )}
              <IconButton onClick={onClose} size="small">
                <Close />
              </IconButton>
            </Box>
          </Box>

          {/* Search */}
          <Box sx={{ p: 2 }}>
            <TextField
              placeholder="Search conversations..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              size="small"
              fullWidth
              InputProps={{
                startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />,
              }}
            />
          </Box>

          {/* Conversations List */}
          <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
            {isLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress size={24} />
              </Box>
            ) : filteredConversations.length === 0 ? (
              <Box sx={{ p: 3, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  {searchQuery ? 'No conversations found' : 'No conversations yet'}
                </Typography>
              </Box>
            ) : (
              <List sx={{ p: 0 }}>
                {filteredConversations.map((conversation, index) => (
                  <React.Fragment key={conversation.conversation_id}>
                    <ListItem sx={{ p: 0 }}>
                      <ListItemButton
                        onClick={() => onSelectConversation(conversation.conversation_id)}
                        sx={{
                          py: 1.5,
                          px: 2,
                          '&:hover': { backgroundColor: 'action.hover' },
                        }}
                      >
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                {conversation.title || 'Untitled Conversation'}
                              </Typography>
                              {conversation.is_pinned && (
                                <PushPin fontSize="small" color="primary" />
                              )}
                            </Box>
                          }
                          secondary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                              <Typography variant="caption" color="text.secondary">
                                {formatLastActivity(conversation)}
                              </Typography>
                              {conversation.message_count > 0 && (
                                <Chip 
                                  label={conversation.message_count} 
                                  size="small" 
                                  variant="outlined"
                                  sx={{ height: '16px', fontSize: '0.7rem' }}
                                />
                              )}
                            </Box>
                          }
                        />
                        <IconButton
                          size="small"
                          onClick={(e) => handleMenuOpen(e, conversation)}
                          sx={{ 
                            opacity: 0.7,
                            '&:hover': { opacity: 1 }
                          }}
                        >
                          <MoreVert fontSize="small" />
                        </IconButton>
                      </ListItemButton>
                    </ListItem>
                    {index < filteredConversations.length - 1 && (
                      <Divider sx={{ mx: 2 }} />
                    )}
                  </React.Fragment>
                ))}
              </List>
            )}
          </Box>

          {/* New Chat Button */}
          <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
            <Button
              variant="contained"
              fullWidth
              startIcon={<Add />}
              onClick={onNewChat}
              sx={{ py: 1 }}
            >
              New Chat
            </Button>
          </Box>
        </Box>
      </ClickAwayListener>

      {/* Menu */}
      <Menu
        anchorEl={menuAnchorEl}
        open={Boolean(menuAnchorEl)}
        onClose={handleMenuClose}
        anchorOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
      >
        <MenuItem onClick={handleDeleteClick}>
          <Delete fontSize="small" sx={{ mr: 1 }} />
          Delete
        </MenuItem>
      </Menu>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={handleDeleteCancel}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Delete Conversation</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete "{selectedConversation?.title || 'Untitled Conversation'}"? 
            This action cannot be undone and will permanently remove all messages in this conversation.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteCancel} disabled={isDeleting}>
            Cancel
          </Button>
          <Button 
            onClick={handleDeleteConfirm}
            color="error" 
            variant="contained"
            disabled={isDeleting}
            startIcon={isDeleting ? <CircularProgress size={16} /> : <Delete />}
            sx={{ 
              backgroundColor: 'error.main',
              '&:hover': { backgroundColor: 'error.dark' },
              '&:active': { backgroundColor: 'error.dark' }
            }}
          >
            {isDeleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete All Conversations Confirmation Dialog */}
      <Dialog
        open={deleteAllDialogOpen}
        onClose={handleDeleteAllCancel}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Delete All Conversations</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete <strong>ALL {filteredConversations.length} conversations</strong>? This action cannot be undone.
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            This will permanently delete all your conversation history, including all messages and context.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteAllCancel} disabled={isDeletingAll}>
            Cancel
          </Button>
          <Button 
            onClick={handleDeleteAllConfirm}
            color="error" 
            variant="contained"
            disabled={isDeletingAll}
            startIcon={isDeletingAll ? <CircularProgress size={16} /> : <Delete />}
            sx={{ 
              backgroundColor: 'error.main',
              '&:hover': { backgroundColor: 'error.dark' },
              '&:active': { backgroundColor: 'error.dark' }
            }}
          >
            {isDeletingAll
              ? 'Deleting...'
              : `Delete All (${filteredConversations.length})`}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for feedback */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleSnackbarClose} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </>
  );
};

export default FloatingHistoryWindow; 