import React from 'react';
import {
  Popover,
  Box,
  Typography,
  List,
  ListItemButton,
  ListItemText,
  Divider,
  Link,
  IconButton,
} from '@mui/material';
import { Close } from '@mui/icons-material';
import { formatDistanceToNow } from 'date-fns';
import { useNotifications } from '../contexts/NotificationContext';
import { useChatSidebar } from '../contexts/ChatSidebarContext';

const MAX_DISPLAY = 20;

function NotificationDropdown({ anchorEl, open, onClose }) {
  const { notifications, markAllRead, markOneRead, dismissNotification, clearNotifications } = useNotifications();
  const { selectConversation, isCollapsed, toggleSidebar } = useChatSidebar();

  const displayList = (notifications || []).slice(0, MAX_DISPLAY);

  const handleMarkAllRead = (e) => {
    e.stopPropagation();
    markAllRead();
  };

  const handleNotificationClick = (item) => {
    if (!item.read) markOneRead(item.id);
    if (item.conversation_id && selectConversation) {
      selectConversation(item.conversation_id);
      if (isCollapsed) toggleSidebar();
    }
    onClose();
  };

  const handleDismiss = (e, id) => {
    e.stopPropagation();
    dismissNotification(id);
  };

  const getBorderColor = (item) => {
    if (item.subtype === 'schedule_paused') return 'error.main';
    if (item.subtype === 'model_configuration') return 'warning.main';
    if (item.subtype === 'chat_completion') return 'primary.main';
    return 'success.main';
  };

  return (
    <Popover
      open={open}
      anchorEl={anchorEl}
      onClose={onClose}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      PaperProps={{
        sx: {
          width: 360,
          maxHeight: 420,
          mt: 1.5,
        },
      }}
    >
      <Box sx={{ p: 2, pb: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="subtitle1" fontWeight={600}>
          Notifications
        </Typography>
        {notifications?.length > 0 && (
          <Box sx={{ display: 'flex', gap: 1.5 }}>
            <Link
              component="button"
              variant="body2"
              onClick={handleMarkAllRead}
              sx={{ cursor: 'pointer' }}
            >
              Mark all read
            </Link>
            <Link
              component="button"
              variant="body2"
              onClick={() => clearNotifications()}
              sx={{ cursor: 'pointer' }}
            >
              Clear all
            </Link>
          </Box>
        )}
      </Box>
      <Divider />
      <List dense disablePadding sx={{ maxHeight: 360, overflow: 'auto' }}>
        {displayList.length === 0 ? (
          <Box sx={{ py: 3, px: 2, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              No notifications yet
            </Typography>
          </Box>
        ) : (
          displayList.map((item) => {
            const isUnread = item.read !== true;
            return (
              <ListItemButton
                key={item.id}
                onClick={() => handleNotificationClick(item)}
                sx={{
                  borderLeft: '3px solid',
                  borderColor: getBorderColor(item),
                  py: 1.5,
                  alignItems: 'flex-start',
                  position: 'relative',
                  bgcolor: isUnread ? 'action.hover' : undefined,
                }}
              >
                <ListItemText
                  primary={
                    <Typography
                      variant="body2"
                      fontWeight={isUnread ? 600 : 400}
                      component="span"
                    >
                      {item.agent_name || 'Agent'}
                    </Typography>
                  }
                secondary={
                  <>
                    <Typography variant="caption" display="block" color="text.secondary" noWrap sx={{ maxWidth: 320 }}>
                      {(item.title || item.preview || 'Notification').slice(0, 100)}
                      {(item.title || item.preview || '').length > 100 ? '…' : ''}
                    </Typography>
                    {item.timestamp && (
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 0.25, display: 'block' }}>
                        {formatDistanceToNow(new Date(item.timestamp), { addSuffix: true })}
                      </Typography>
                    )}
                  </>
                }
                  primaryTypographyProps={{ variant: 'body2' }}
                />
                <IconButton
                  size="small"
                  onClick={(e) => handleDismiss(e, item.id)}
                  sx={{ position: 'absolute', top: 4, right: 4 }}
                  aria-label="Dismiss notification"
                >
                  <Close fontSize="small" />
                </IconButton>
              </ListItemButton>
            );
          })
        )}
      </List>
    </Popover>
  );
}

export default NotificationDropdown;
