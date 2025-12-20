import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  IconButton,
  Tooltip,
  Collapse,
  Divider,
  Chip,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Alert,
  CircularProgress,
  Checkbox,
  FormControlLabel,
} from '@mui/material';
import {
  ChevronLeft,
  ChevronRight,
  Add,
  Chat,
  Delete,
  Edit,
  PushPin,
  Archive,
  MoreVert,
  Search,
  Folder,
  DragIndicator,
  Lock,
  LockOpen,
  Share,
  People,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import apiService from '../services/apiService';
import conversationService from '../services/conversation/ConversationService';
import ConversationShareDialog from './ConversationShareDialog';

const ConversationSidebar = ({ 
  currentConversationId, 
  onConversationSelect, 
  onNewConversation,
  onClearCurrentConversation, // Add prop to handle clearing current conversation
  isCollapsed,
  onToggleCollapse,
  isCreatingConversation = false // Add prop to track if conversation is being created
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedConversation, setSelectedConversation] = useState(null);
  const [contextMenu, setContextMenu] = useState(null);
  const [editDialog, setEditDialog] = useState({ open: false, conversation: null });
  const [deleteDialog, setDeleteDialog] = useState({ open: false, conversation: null });
  const [deleteAllDialog, setDeleteAllDialog] = useState({ open: false });
  const [shareDialog, setShareDialog] = useState({ open: false, conversation: null });
  const [orderLocked, setOrderLocked] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  
  const queryClient = useQueryClient();

  // Fetch conversations with search and filtering
  const { data: conversationsData, isLoading, error } = useQuery(
    ['conversations', searchQuery],
    async () => {
      const [ownedConversations, sharedConversations] = await Promise.all([
        apiService.listConversations(0, 100),
        conversationService.getSharedConversations(0, 100).catch(() => ({ conversations: [] }))
      ]);
      
      // Merge conversations, marking shared ones
      const allConversations = [
        ...(ownedConversations?.conversations || []).map(conv => ({ ...conv, is_shared: false })),
        ...(sharedConversations?.conversations || []).map(conv => ({ ...conv, is_shared: true }))
      ];
      
      // Remove duplicates (in case user owns a conversation they also have shared)
      const uniqueConversations = Array.from(
        new Map(allConversations.map(conv => [conv.conversation_id, conv])).values()
      );
      
      return {
        conversations: uniqueConversations,
        total_count: uniqueConversations.length,
        has_more: false
      };
    },
    {
      refetchOnWindowFocus: true,
      refetchOnMount: true,
      staleTime: 300000, // Consider data stale after 5 minutes
      // No refetchInterval - rely on WebSocket updates and manual triggers
    }
  );

  // Real-time conversation updates via WebSocket
  useEffect(() => {
    // Connect to conversation updates WebSocket (if available)
    const token = apiService.getToken();
    if (!token) {
      console.error('❌ No authentication token available for conversation WebSocket');
      return;
    }

    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws/conversations?token=${encodeURIComponent(token)}`;
    let ws = null;

    try {
      ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('📡 Connected to conversation updates WebSocket');
      };

      ws.onmessage = (event) => {
        try {
          const update = JSON.parse(event.data);
          
          if (update.type === 'conversation_created' || 
              update.type === 'conversation_updated' || 
              update.type === 'conversation_deleted') {
            console.log('🔄 Received conversation update, refreshing list');
            // Invalidate queries to trigger refresh
            queryClient.invalidateQueries(['conversations']);
          }
        } catch (error) {
          console.error('❌ Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log(`📡 Disconnected from conversation updates WebSocket. Code: ${event.code}, Reason: ${event.reason}`);
        
        // If closed due to auth error, log it clearly
        if (event.code === 1008) {
          console.error(`❌ WebSocket authentication failed: ${event.reason}`);
        }
      };

      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
      };

    } catch (error) {
      console.error('❌ Failed to connect to conversation updates WebSocket:', error);
    }

    // Cleanup on unmount
    return () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [queryClient]);

  // Update conversation mutation
  const updateConversationMutation = useMutation(
    ({ conversationId, updates }) => 
      apiService.updateConversation(conversationId, updates.title, updates),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['conversations']);
        setEditDialog({ open: false, conversation: null });
      },
    }
  );

  // Delete conversation mutation
  const deleteConversationMutation = useMutation(
    (conversationId) => {
      console.log('🔄 Attempting to delete conversation:', conversationId);
      console.log('🔄 Mutation function called with ID:', conversationId);
      console.log('🔄 API service available:', !!apiService);
      console.log('🔄 API service deleteConversation method:', !!apiService.deleteConversation);
      return apiService.deleteConversation(conversationId);
    },
    {
      onSuccess: (data, conversationId) => {
        console.log('✅ Conversation deleted successfully:', conversationId, data);
        queryClient.invalidateQueries(['conversations']);
        setDeleteDialog({ open: false, conversation: null });
        // If deleting current conversation, clear the current conversation but don't auto-create new one
        // Don't call onConversationSelect here as it would close the history window
        if (currentConversationId === deleteDialog.conversation?.conversation_id) {
          // Clear current conversation without triggering history window close
          if (onClearCurrentConversation) {
            onClearCurrentConversation();
          }
        }
      },
      onError: (error, conversationId) => {
        console.error('❌ Failed to delete conversation:', conversationId, error);
        console.error('❌ Error details:', error);
        // Show error message to user
        alert(`Failed to delete conversation: ${error.response?.data?.detail || error.message}`);
      },
      onMutate: (conversationId) => {
        console.log('🔄 Mutation starting for conversation:', conversationId);
      },
    }
  );

  // Delete all conversations mutation
  const deleteAllConversationsMutation = useMutation(
    () => {
      console.log('🔄 Attempting to delete ALL conversations');
      return apiService.deleteAllConversations();
    },
    {
      onSuccess: (data) => {
        console.log('✅ All conversations deleted successfully:', data);
        queryClient.invalidateQueries(['conversations']);
        setDeleteAllDialog({ open: false });
        // Clear current conversation if it exists
        if (onClearCurrentConversation) {
          onClearCurrentConversation();
        }
      },
      onError: (error) => {
        console.error('❌ Failed to delete all conversations:', error);
        alert(`Failed to delete all conversations: ${error.response?.data?.detail || error.message}`);
      },
    }
  );

  // Reorder conversations mutation
  const reorderConversationsMutation = useMutation(
    ({ conversationIds, orderLocked }) => apiService.reorderConversations(conversationIds, orderLocked),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['conversations']);
      },
    }
  );

  const conversations = conversationsData?.conversations || [];

  const handleConversationClick = (conversation) => {
    setSelectedConversation(conversation.conversation_id);
    onConversationSelect(conversation);
    setContextMenu(null);
  };

  const handleContextMenu = (event, conversation) => {
    console.log('🖱️ Context menu triggered for conversation:', conversation.conversation_id);
    console.log('🖱️ Event type:', event.type);
    console.log('🖱️ Mouse position:', { x: event.clientX, y: event.clientY });
    
    event.preventDefault();
    event.stopPropagation();
    
    setContextMenu({
      mouseX: event.clientX - 2,
      mouseY: event.clientY - 4,
      conversation,
    });
    
    console.log('🖱️ Context menu state set');
  };

  const handleCloseContextMenu = () => {
    setContextMenu(null);
  };

  const handleEditConversation = () => {
    setEditDialog({ open: true, conversation: contextMenu.conversation });
    handleCloseContextMenu();
  };

  const handleDeleteConversation = () => {
    console.log('🗑️ Delete conversation clicked:', contextMenu.conversation);
    console.log('🗑️ Conversation title:', contextMenu.conversation?.title);
    console.log('🗑️ Conversation ID:', contextMenu.conversation?.conversation_id);
    
    // Immediate feedback
    alert('Delete option selected from context menu!');
    
    setDeleteDialog({ open: true, conversation: contextMenu.conversation });
    handleCloseContextMenu();
  };

  const handleTogglePin = async () => {
    const conversation = contextMenu.conversation;
    await updateConversationMutation.mutateAsync({
      conversationId: conversation.conversation_id,
      updates: { is_pinned: !conversation.is_pinned }
    });
    handleCloseContextMenu();
  };

  const handleToggleArchive = async () => {
    const conversation = contextMenu.conversation;
    await updateConversationMutation.mutateAsync({
      conversationId: conversation.conversation_id,
      updates: { is_archived: !conversation.is_archived }
    });
    handleCloseContextMenu();
  };

  const handleShareConversation = () => {
    setShareDialog({ open: true, conversation: contextMenu.conversation });
    handleCloseContextMenu();
  };

  const handleDeleteAllConversations = () => {
    setDeleteAllDialog({ open: true });
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffInHours = (now - date) / (1000 * 60 * 60);
    
    if (diffInHours < 24) {
      // Show time with date for recent items
      return date.toLocaleString('en-US', { 
        month: 'short',
        day: 'numeric',
        hour: 'numeric', 
        minute: '2-digit' 
      });
    } else if (diffInHours < 168) { // Less than a week
      // Show weekday with date
      return date.toLocaleDateString('en-US', { 
        weekday: 'short',
        month: 'short',
        day: 'numeric'
      });
    } else {
      // Show full date
      return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric',
        year: 'numeric'
      });
    }
  };

  const getSidebarWidth = () => isCollapsed ? 60 : 320;

  const handleDragEnd = (result) => {
    setIsDragging(false);
    
    if (!result.destination) {
      return;
    }

    const items = Array.from(conversations);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    const conversationIds = items.map(conv => conv.conversation_id);
    
    // Update order in backend
    reorderConversationsMutation.mutate({
      conversationIds,
      orderLocked
    });
  };

  const handleDragStart = () => {
    setIsDragging(true);
  };

  const handleToggleOrderLock = () => {
    const newOrderLocked = !orderLocked;
    setOrderLocked(newOrderLocked);
    
    if (newOrderLocked) {
      // Lock current order
      const conversationIds = conversations.map(conv => conv.conversation_id);
      reorderConversationsMutation.mutate({
        conversationIds,
        orderLocked: true
      });
    } else {
      // Unlock order - conversations will auto-sort by last_message_at
      const conversationIds = conversations.map(conv => conv.conversation_id);
      reorderConversationsMutation.mutate({
        conversationIds,
        orderLocked: false
      });
    }
  };

  return (
    <>
      <Paper
        component={motion.div}
        initial={{ width: getSidebarWidth() }}
        animate={{ width: getSidebarWidth() }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        sx={{
          height: { xs: 'calc(var(--appvh, 100vh) - 59px)', md: 'calc(100dvh - 59px)' },
          display: 'flex',
          flexDirection: 'column',
          borderRadius: 0,
          borderRight: '1px solid #e0e0e0',
          overflow: 'hidden',
          position: 'fixed',
          left: 0,
          top: '59px',
          zIndex: 1200,
        }}
      >
        {/* Header */}
        <Box sx={{ 
          p: isCollapsed ? 1 : 1.5, 
          borderBottom: '1px solid #e0e0e0',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          minHeight: 44,
        }}>
          {!isCollapsed && (
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Conversations
            </Typography>
          )}
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {!isCollapsed && (
              <Tooltip title="New Conversation">
                <Button
                  variant="contained"
                  size="small"
                  startIcon={<Add />}
                  onClick={() => onNewConversation()}
                  disabled={isCreatingConversation}
                  sx={{ minWidth: 'auto' }}
                >
                  {isCreatingConversation ? 'Creating...' : 'New'}
                </Button>
              </Tooltip>
            )}
            
            <Tooltip title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}>
              <IconButton 
                onClick={onToggleCollapse}
                size="small"
              >
                {isCollapsed ? <ChevronRight /> : <ChevronLeft />}
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Collapsed State */}
        {isCollapsed && (
          <Box sx={{ p: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
            <Tooltip title="New Conversation" placement="right">
              <IconButton
                onClick={() => onNewConversation()}
                disabled={isCreatingConversation}
                sx={{ 
                  backgroundColor: isCreatingConversation ? 'action.disabled' : 'primary.main',
                  color: 'white',
                  '&:hover': { backgroundColor: isCreatingConversation ? 'action.disabled' : 'primary.dark' }
                }}
              >
                <Add />
              </IconButton>
            </Tooltip>
            
            {conversations.slice(0, 8).map((conversation) => (
              <Tooltip 
                key={conversation.conversation_id}
                title={conversation.title}
                placement="right"
              >
                <IconButton
                  onClick={() => handleConversationClick(conversation)}
                  sx={{
                    backgroundColor: currentConversationId === conversation.conversation_id 
                      ? 'primary.light' : 'transparent',
                    '&:hover': { backgroundColor: 'action.hover' }
                  }}
                >
                  <Chat fontSize="small" />
                </IconButton>
              </Tooltip>
            ))}
          </Box>
        )}

        {/* Expanded State */}
        {!isCollapsed && (
          <>
            {/* Search Bar and Controls */}
            <Box sx={{ p: 2, borderBottom: '1px solid #e0e0e0' }}>
              <TextField
                size="small"
                fullWidth
                placeholder="Search conversations..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                InputProps={{
                  startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />,
                }}
                sx={{ mb: 1 }}
              />
              
              {/* Controls Row */}
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                {/* Order Lock Control */}
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={orderLocked}
                      onChange={handleToggleOrderLock}
                      size="small"
                      icon={<LockOpen fontSize="small" />}
                      checkedIcon={<Lock fontSize="small" />}
                    />
                  }
                  label={
                    <Typography variant="caption" color="text.secondary">
                      Lock order
                    </Typography>
                  }
                  sx={{ m: 0 }}
                />
                
                {/* Delete All Conversations Button */}
                {conversations.length > 0 && (
                  <Tooltip title="Delete All Conversations">
                    <Button
                      variant="outlined"
                      size="small"
                      color="error"
                      startIcon={<Delete />}
                      onClick={handleDeleteAllConversations}
                      sx={{ 
                        minWidth: 'auto',
                        px: 1,
                        py: 0.5,
                        fontSize: '0.75rem'
                      }}
                    >
                      Clear All
                    </Button>
                  </Tooltip>
                )}
              </Box>
            </Box>

            {/* Conversations List */}
            <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
              {isLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress />
                </Box>
              ) : error ? (
                <Alert severity="error" sx={{ m: 2 }}>
                  Failed to load conversations
                </Alert>
              ) : conversations.length === 0 ? (
                <Box sx={{ p: 3, textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary">
                    No conversations yet
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    Click "New" above to start your first conversation
                  </Typography>
                </Box>
              ) : (
                <DragDropContext onDragEnd={handleDragEnd} onDragStart={handleDragStart}>
                  <Droppable droppableId="conversations" isDropDisabled={!orderLocked}>
                    {(provided, snapshot) => (
                      <List 
                        sx={{ p: 0 }}
                        ref={provided.innerRef}
                        {...provided.droppableProps}
                      >
                        {conversations.map((conversation, index) => (
                          <Draggable
                            key={conversation.conversation_id}
                            draggableId={conversation.conversation_id}
                            index={index}
                            isDragDisabled={!orderLocked}
                          >
                            {(provided, snapshot) => (
                              <ListItem
                                ref={provided.innerRef}
                                {...provided.draggableProps}
                                disablePadding
                                sx={{
                                  borderBottom: '1px solid #f0f0f0',
                                  backgroundColor: currentConversationId === conversation.conversation_id 
                                    ? 'primary.light' : 'transparent',
                                  opacity: snapshot.isDragging ? 0.8 : 1,
                                  transform: snapshot.isDragging ? 'rotate(5deg)' : 'none',
                                }}
                              >
                                <ListItemButton
                                  onClick={() => handleConversationClick(conversation)}
                                  onContextMenu={(e) => handleContextMenu(e, conversation)}
                                  sx={{ 
                                    py: 1.5,
                                    '&:hover': { backgroundColor: 'action.hover' }
                                  }}
                                >
                                  {/* Drag Handle */}
                                  {orderLocked && (
                                    <Box
                                      {...provided.dragHandleProps}
                                      sx={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        mr: 1,
                                        cursor: 'grab',
                                        '&:active': { cursor: 'grabbing' }
                                      }}
                                    >
                                      <DragIndicator sx={{ fontSize: 16, color: 'text.secondary' }} />
                                    </Box>
                                  )}

                                  <Box sx={{ flexGrow: 1, minWidth: 0 }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                                      <Chat sx={{ fontSize: 16, mr: 1, color: 'text.secondary' }} />
                                      <Typography 
                                        variant="subtitle2" 
                                        sx={{ 
                                          fontWeight: 500,
                                          overflow: 'hidden',
                                          textOverflow: 'ellipsis',
                                          whiteSpace: 'nowrap',
                                          flexGrow: 1
                                        }}
                                      >
                                        {conversation.title}
                                      </Typography>
                                      {conversation.is_pinned && (
                                        <PushPin sx={{ fontSize: 14, color: 'primary.main', ml: 0.5 }} />
                                      )}
                                      {conversation.is_shared && (
                                        <People sx={{ fontSize: 14, color: 'secondary.main', ml: 0.5 }} />
                                      )}
                                    </Box>
                                    
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                      <Typography variant="caption" color="text.secondary">
                                        {conversation.message_count} messages
                                      </Typography>
                                      <Typography variant="caption" color="text.secondary">
                                        {conversation.last_message_at 
                                          ? formatTimestamp(conversation.last_message_at)
                                          : formatTimestamp(conversation.created_at)
                                        }
                                      </Typography>
                                    </Box>
                                    
                                    {conversation.tags && conversation.tags.length > 0 && (
                                      <Box sx={{ mt: 0.5, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                                        {conversation.tags.slice(0, 2).map((tag, index) => (
                                          <Chip
                                            key={index}
                                            label={tag}
                                            size="small"
                                            variant="outlined"
                                            sx={{ fontSize: '0.7rem', height: 20 }}
                                          />
                                        ))}
                                        {conversation.tags.length > 2 && (
                                          <Chip
                                            label={`+${conversation.tags.length - 2}`}
                                            size="small"
                                            variant="outlined"
                                            sx={{ fontSize: '0.7rem', height: 20 }}
                                          />
                                        )}
                                      </Box>
                                    )}
                                  </Box>
                                  
                                  <IconButton
                                    size="small"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleContextMenu(e, conversation);
                                    }}
                                    sx={{ ml: 1 }}
                                  >
                                    <MoreVert fontSize="small" />
                                  </IconButton>
                                </ListItemButton>
                              </ListItem>
                            )}
                          </Draggable>
                        ))}
                        {provided.placeholder}
                      </List>
                    )}
                  </Droppable>
                </DragDropContext>
              )}
            </Box>
          </>
        )}
      </Paper>

      {/* Context Menu */}
      <Menu
        open={contextMenu !== null}
        onClose={handleCloseContextMenu}
        anchorReference="anchorPosition"
        anchorPosition={
          contextMenu !== null
            ? { top: contextMenu.mouseY, left: contextMenu.mouseX }
            : undefined
        }
      >
        <MenuItem onClick={handleEditConversation}>
          <Edit sx={{ mr: 1 }} />
          Edit
        </MenuItem>
        <MenuItem onClick={handleShareConversation}>
          <Share sx={{ mr: 1 }} />
          Share
        </MenuItem>
        <MenuItem onClick={handleTogglePin}>
          <PushPin sx={{ mr: 1 }} />
          {contextMenu?.conversation?.is_pinned ? 'Unpin' : 'Pin'}
        </MenuItem>
        <MenuItem onClick={handleToggleArchive}>
          <Archive sx={{ mr: 1 }} />
          {contextMenu?.conversation?.is_archived ? 'Unarchive' : 'Archive'}
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleDeleteConversation} sx={{ color: 'error.main' }}>
          <Delete sx={{ mr: 1 }} />
          Delete
        </MenuItem>
      </Menu>

      {/* Edit Dialog */}
      <Dialog open={editDialog.open} onClose={() => setEditDialog({ open: false, conversation: null })}>
        <DialogTitle>Edit Conversation</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Title"
            fullWidth
            variant="outlined"
            defaultValue={editDialog.conversation?.title || ''}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                // Handle save
              }
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialog({ open: false, conversation: null })}>
            Cancel
          </Button>
          <Button variant="contained">
            Save
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialog.open} onClose={() => setDeleteDialog({ open: false, conversation: null })}>
        <DialogTitle>Delete Conversation</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete "{deleteDialog.conversation?.title || 'Untitled Conversation'}"? This action cannot be undone.
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            Conversation ID: {deleteDialog.conversation?.conversation_id}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog({ open: false, conversation: null })}>
            Cancel
          </Button>
          <Button 
            variant="contained" 
            color="error"
            onClick={(e) => {
              console.log('🗑️ Delete button clicked! Event:', e);
              console.log('🗑️ Delete button clicked in dialog:', deleteDialog.conversation);
              console.log('🗑️ Conversation ID to delete:', deleteDialog.conversation?.conversation_id);
              console.log('🗑️ Delete mutation available:', !!deleteConversationMutation);
              console.log('🗑️ Delete mutation function:', deleteConversationMutation.mutate);
              
              // Immediate visual feedback
              alert('Delete button clicked!');
              
              if (deleteDialog.conversation?.conversation_id) {
                console.log('🗑️ Calling delete mutation with ID:', deleteDialog.conversation.conversation_id);
                deleteConversationMutation.mutate(deleteDialog.conversation.conversation_id);
              } else {
                console.error('❌ No conversation ID found for deletion');
                alert('Error: No conversation ID found for deletion');
              }
            }}
            sx={{ 
              backgroundColor: 'error.main',
              '&:hover': { backgroundColor: 'error.dark' },
              '&:active': { backgroundColor: 'error.dark' }
            }}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete All Conversations Confirmation Dialog */}
      <Dialog open={deleteAllDialog.open} onClose={() => setDeleteAllDialog({ open: false })}>
        <DialogTitle>Delete All Conversations</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete <strong>ALL {conversations.length} conversations</strong>? This action cannot be undone.
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            This will permanently delete all your conversation history, including all messages and context.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteAllDialog({ open: false })}>
            Cancel
          </Button>
          <Button 
            variant="contained" 
            color="error"
            onClick={() => {
              console.log('🗑️ Delete ALL conversations button clicked!');
              deleteAllConversationsMutation.mutate();
            }}
            sx={{ 
              backgroundColor: 'error.main',
              '&:hover': { backgroundColor: 'error.dark' },
              '&:active': { backgroundColor: 'error.dark' }
            }}
          >
            Delete All ({conversations.length})
          </Button>
        </DialogActions>
      </Dialog>

      {/* Share Dialog */}
      <ConversationShareDialog
        open={shareDialog.open}
        onClose={() => {
          setShareDialog({ open: false, conversation: null });
          // Refresh conversation list to show any new shares
          queryClient.invalidateQueries(['conversations']);
        }}
        conversationId={shareDialog.conversation?.conversation_id}
        conversationTitle={shareDialog.conversation?.title}
      />
    </>
  );
};

export default ConversationSidebar;
