/**
 * Individual message bubble for room chat.
 * Handles reactions, reply preview, edit display, mention highlights, and markdown.
 */

import React, { useMemo } from 'react';
import {
  Box, Typography, Paper, Avatar, Chip, IconButton, Button, Tooltip,
} from '@mui/material';
import {
  AddReaction as AddReactionIcon,
  Reply,
  Edit,
  Download,
  SmartToy,
  Hub,
  Schedule,
  Check,
  WarningAmber,
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import AudioPlayer from '../AudioPlayer';
import messagingService from '../../services/messagingService';
import { formatTimestamp } from '../../utils/chatUtils';

const mentionRegex = /@([\w\s.-]+?)(?=\s|$|[.,!?;:])/g;

function highlightMentions(text) {
  if (!text) return text;
  const parts = [];
  let lastIndex = 0;
  let match;
  const regex = new RegExp(mentionRegex.source, 'g');
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) parts.push(text.slice(lastIndex, match.index));
    parts.push(
      <Typography
        key={match.index}
        component="span"
        sx={{ fontWeight: 600, color: 'info.main', bgcolor: 'action.selected', borderRadius: 0.5, px: 0.3 }}
      >
        {match[0]}
      </Typography>
    );
    lastIndex = regex.lastIndex;
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex));
  return parts.length > 0 ? parts : text;
}

const markdownComponents = {
  p: ({ children }) => (
    <Typography variant="body2" component="span" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', display: 'block', color: 'inherit' }}>
      {children}
    </Typography>
  ),
  code: ({ children }) => (
    <Box component="code" sx={{ bgcolor: 'action.hover', px: 0.5, py: 0.25, borderRadius: 0.5, fontSize: '0.85em', fontFamily: 'monospace' }}>
      {children}
    </Box>
  ),
  pre: ({ children }) => (
    <Box component="pre" sx={{ bgcolor: 'action.hover', p: 1, borderRadius: 1, overflow: 'auto', fontSize: '0.85em', my: 0.5 }}>
      {children}
    </Box>
  ),
  a: ({ href, children }) => (
    <Typography component="a" href={href} target="_blank" rel="noopener noreferrer" sx={{ color: 'primary.light', textDecoration: 'underline' }}>
      {children}
    </Typography>
  ),
};

const RoomMessageBubble = ({
  message,
  isOwn,
  currentUserId,
  roomIsFederated,
  onReactionClick,
  onReactionChipClick,
  onReply,
  onEdit,
  attachmentsList,
  imageBlobUrls,
  openLightbox,
}) => {
  const hasMentions = message.metadata?.mentions?.length > 0;
  const useMarkdown = /[*_`#\[\]>~|]/.test(message.content || '');
  const isBot = message.metadata?.from_agent_profile_id;

  const groupedReactions = useMemo(() => {
    if (!message.reactions?.length) return null;
    const grouped = {};
    message.reactions.forEach(r => {
      if (!grouped[r.emoji]) grouped[r.emoji] = [];
      grouped[r.emoji].push(r);
    });
    return grouped;
  }, [message.reactions]);

  const deliveryStatus = message.federation_delivery_status;
  const deliveryTooltip =
    deliveryStatus === 'failed'
      ? 'Delivery to peer failed'
      : deliveryStatus === 'peer_suspended'
        ? 'Peer suspended — not delivered'
        : deliveryStatus === 'delivered'
          ? 'Delivered to peer'
          : deliveryStatus === 'pending'
            ? 'Delivery pending'
            : '';

  return (
    <Box sx={{ maxWidth: '70%' }}>
      {/* Reply preview */}
      {message.reply_preview && (
        <Box sx={{
          borderLeft: 2, borderColor: 'primary.main', pl: 1, mb: 0.5, opacity: 0.75,
          ml: isOwn ? 'auto' : 0, maxWidth: '90%',
        }}>
          <Typography variant="caption" sx={{ fontWeight: 600 }}>
            {message.reply_preview.sender_name || 'Unknown'}
          </Typography>
          <Typography variant="caption" display="block" noWrap>
            {message.reply_preview.content}
          </Typography>
        </Box>
      )}

      <Paper
        elevation={1}
        sx={{
          p: 1.5,
          backgroundColor: isOwn ? 'primary.main' : 'background.paper',
          color: isOwn ? 'primary.contrastText' : 'text.primary',
          position: 'relative',
          '&:hover .msg-actions': { opacity: 1 },
        }}
      >
        {!isOwn && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.25, flexWrap: 'wrap' }}>
            {isBot && <SmartToy sx={{ fontSize: 14, opacity: 0.6 }} />}
            {message.is_federated && (
              <Chip
                size="small"
                icon={<Hub sx={{ fontSize: '0.85rem !important' }} />}
                label="Remote"
                sx={{ height: 18, fontSize: '0.65rem', '& .MuiChip-icon': { ml: 0.25 } }}
              />
            )}
            <Typography variant="caption" sx={{ fontWeight: 600, color: 'inherit' }}>
              {message.display_name || message.username}
            </Typography>
          </Box>
        )}

        {useMarkdown ? (
          <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]} components={markdownComponents}>
            {message.content}
          </ReactMarkdown>
        ) : hasMentions ? (
          <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: 'inherit' }}>
            {highlightMentions(message.content)}
          </Typography>
        ) : (
          <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: 'inherit' }}>
            {message.content}
          </Typography>
        )}

        {(attachmentsList || []).map((att) => (
          <Box key={att.attachment_id} mt={1}>
            {att.mime_type?.startsWith('audio/') ? (
              <AudioPlayer src={messagingService.getAttachmentUrl(att.attachment_id)} filename={att.filename} />
            ) : att.mime_type?.startsWith('image/') ? (
              <Box
                component="img"
                src={imageBlobUrls?.[att.attachment_id] || messagingService.getAttachmentUrl(att.attachment_id)}
                alt={att.filename}
                onClick={() => openLightbox?.(
                  imageBlobUrls?.[att.attachment_id] || messagingService.getAttachmentUrl(att.attachment_id),
                  { filename: att.filename }
                )}
                sx={{ maxWidth: '100%', maxHeight: 300, borderRadius: 1, cursor: 'pointer', display: 'block', objectFit: 'contain' }}
              />
            ) : (
              <Button variant="outlined" size="small" href={messagingService.getAttachmentUrl(att.attachment_id)} download={att.filename || 'attachment'} startIcon={<Download />}>
                {att.filename || 'Download'}
              </Button>
            )}
          </Box>
        ))}

        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mt: 0.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Typography variant="caption" sx={{ opacity: 0.7, fontSize: '0.7rem', color: 'inherit' }}>
              {formatTimestamp(message.created_at)}
            </Typography>
            {message.is_edited && (
              <Typography variant="caption" sx={{ opacity: 0.5, fontSize: '0.65rem', color: 'inherit', fontStyle: 'italic' }}>
                (edited)
              </Typography>
            )}
            {isOwn && roomIsFederated && (
              <Tooltip title={deliveryTooltip || deliveryStatus || 'Federation delivery'}>
                <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', ml: 0.25, opacity: 0.85 }}>
                  {(!deliveryStatus || deliveryStatus === 'pending') && (
                    <Schedule sx={{ fontSize: 14 }} aria-label="Pending" />
                  )}
                  {deliveryStatus === 'delivered' && (
                    <Check sx={{ fontSize: 14 }} aria-label="Delivered" />
                  )}
                  {(deliveryStatus === 'failed' || deliveryStatus === 'peer_suspended') && (
                    <WarningAmber sx={{ fontSize: 14 }} aria-label="Delivery issue" />
                  )}
                </Box>
              </Tooltip>
            )}
          </Box>
          <Box className="msg-actions" sx={{ opacity: 0, transition: 'opacity 0.15s', display: 'flex' }}>
            <Tooltip title="Reply">
              <IconButton size="small" onClick={() => onReply?.(message)} sx={{ p: 0.25, color: 'inherit' }}>
                <Reply sx={{ fontSize: 14 }} />
              </IconButton>
            </Tooltip>
            {isOwn && (
              <Tooltip title="Edit">
                <IconButton size="small" onClick={() => onEdit?.(message)} sx={{ p: 0.25, color: 'inherit' }}>
                  <Edit sx={{ fontSize: 14 }} />
                </IconButton>
              </Tooltip>
            )}
            <IconButton size="small" onClick={(e) => onReactionClick?.(e, message.message_id)} sx={{ p: 0.25, color: 'inherit' }}>
              <AddReactionIcon sx={{ fontSize: 14 }} />
            </IconButton>
          </Box>
        </Box>
      </Paper>

      {groupedReactions && (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5, justifyContent: isOwn ? 'flex-end' : 'flex-start' }}>
          {Object.entries(groupedReactions).map(([emoji, reactions]) => {
            const isMine = reactions.some(r => r.user_id === currentUserId);
            return (
              <Chip
                key={emoji}
                label={`${emoji} ${reactions.length}`}
                size="small"
                variant={isMine ? 'filled' : 'outlined'}
                color={isMine ? 'primary' : 'default'}
                onClick={() => onReactionChipClick?.(message.message_id, emoji, message.reactions)}
                sx={{ height: 24, fontSize: '0.75rem', cursor: 'pointer' }}
              />
            );
          })}
        </Box>
      )}
    </Box>
  );
};

export default React.memo(RoomMessageBubble);
