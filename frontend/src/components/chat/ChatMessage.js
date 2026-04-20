import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Typography,
  Paper,
  IconButton,
  Button,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  useTheme,
  TextField,
} from '@mui/material';
import {
  Person,
  Groups,
  ChevronLeft,
  ChevronRight,
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkBreaks from 'remark-breaks';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';
import rehypeHighlightChatSearch from '../../utils/rehypeHighlightChatSearch';
import { highlightPlainTextSearch } from '../../utils/highlightPlainTextSearch';
import { renderCitations } from '../../utils/chatUtils';
import { alpha } from '@mui/material/styles';
import ExportButton from './ExportButton';
import ReadAloudButton from './ReadAloudButton';
import LessonPreviewCard from '../LessonPreviewCard';
import ArtifactCard from './ArtifactCard';

/** Parse artifact payload from message metadata (object or JSON string). */
export const parseArtifactFromMessageMetadata = (metadata) => {
  if (!metadata?.artifact) return null;
  let a = metadata.artifact;
  if (typeof a === 'string') {
    try {
      a = JSON.parse(a);
    } catch {
      return null;
    }
  }
  if (!a || typeof a !== 'object' || !a.artifact_type) return null;
  return a;
};

/** All artifact payloads from message metadata (plural array or singular). */
export const parseArtifactsFromMessageMetadata = (metadata) => {
  if (!metadata) return [];
  let list = metadata.artifacts;
  if (typeof list === 'string') {
    try {
      list = JSON.parse(list);
    } catch {
      list = null;
    }
  }
  if (Array.isArray(list) && list.length > 0) {
    return list.filter((a) => a && typeof a === 'object' && a.artifact_type);
  }
  const single = parseArtifactFromMessageMetadata(metadata);
  return single ? [single] : [];
};

/** True if URL is an in-system document file (research result); needs auth and no Import. */
const isDocumentFileUrl = (url) =>
  typeof url === 'string' && url.includes('/api/documents/') && url.includes('/file');

const parseJsonStringArray = (value) => {
  if (!value) return [];
  if (Array.isArray(value)) return value.filter(Boolean);
  if (typeof value !== 'string') return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.filter(Boolean) : [];
  } catch {
    return [];
  }
};

const formatDurationMs = (durationMs) => {
  const ms = Number(durationMs);
  if (!Number.isFinite(ms) || ms <= 0) return '';

  const totalSeconds = Math.round(ms) / 1000;
  if (totalSeconds < 60) return `${totalSeconds.toFixed(1)}s`;

  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.round(totalSeconds % 60);
  return `${minutes}m ${seconds}s`;
};

const formatSkillLabel = (name) => {
  if (!name || typeof name !== 'string') return '';
  const cleaned = name.replace(/_(agent|fragment)$/i, '');
  return cleaned
    .split('_')
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
};

const TOOL_CATEGORY_LABELS = {
  search: 'Search',
  knowledge: 'Knowledge',
  email: 'Email',
  calendar: 'Calendar',
  org: 'Org Mode',
  image: 'Image',
  math: 'Math',
  navigation: 'Navigation',
  data_workspace: 'Data',
  utility: 'Utility',
  file: 'Documents',
  document: 'Documents',
  agent: 'Agent',
  teams: 'Teams',
  analysis: 'Analysis',
};

const formatCategoryLabel = (cat) => {
  if (!cat || typeof cat !== 'string') return '';
  return TOOL_CATEGORY_LABELS[cat] || cat.charAt(0).toUpperCase() + cat.slice(1);
};

/**
 * Loads /api/documents/{id}/file with Authorization and displays as image.
 * Other URLs are rendered as normal img src.
 * onClick(isDisplaySrc) is called with the effective src (blob URL for document files) for lightbox.
 */
const AuthDocumentImage = React.memo(({ url, alt, onClick, sx, ...rest }) => {
  const [blobUrl, setBlobUrl] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!url || !isDocumentFileUrl(url)) return undefined;
    const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
    let objectUrl = null;
    fetch(url, { headers: token ? { Authorization: `Bearer ${token}` } : {} })
      .then((res) => {
        if (!res.ok) throw new Error(res.status);
        return res.blob();
      })
      .then((blob) => {
        objectUrl = URL.createObjectURL(blob);
        setBlobUrl(objectUrl);
        setError(false);
      })
      .catch(() => {
        setError(true);
      });
    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [url]);

  const effectiveSrc = (isDocumentFileUrl(url) && blobUrl) ? blobUrl : url;
  const handleClick = (e) => {
    if (onClick) onClick(e, effectiveSrc);
  };

  if (!url) return null;
  if (isDocumentFileUrl(url)) {
    if (error) {
      return (
        <Box sx={{ py: 1, px: 1.5, bgcolor: 'action.hover', borderRadius: 1, ...sx }}>
          <Typography variant="caption" color="text.secondary">
            Could not load image (sign in may be required)
          </Typography>
        </Box>
      );
    }
    if (!blobUrl) {
      return (
        <Box sx={{ minHeight: 80, bgcolor: 'action.hover', borderRadius: 1, ...sx }} />
      );
    }
    return (
      <Box
        component="img"
        src={blobUrl}
        alt={alt}
        onClick={handleClick}
        sx={sx}
        {...rest}
      />
    );
  }
  return (
    <Box component="img" src={url} alt={alt} onClick={handleClick} sx={sx} {...rest} />
  );
});

const CHAT_MESSAGE_REHYPE_SANITIZE_OPTIONS = {
  tagNames: [
    'details', 'summary', 'div', 'span',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'p', 'br', 'strong', 'em', 'b', 'i', 'u', 's',
    'ul', 'ol', 'li',
    'blockquote', 'pre', 'code',
    'table', 'thead', 'tbody', 'tr', 'th', 'td',
    'a', 'img',
    'hr',
    'mark',
  ],
  attributes: {
    '*': ['class', 'id', 'align'],
    'a': ['href', 'title'],
    'img': ['src', 'alt', 'title', 'width', 'height'],
    'div': ['style'],
    'span': ['style'],
    'details': ['open'],
    'summary': [],
    mark: ['className', 'dataChatScrollTarget'],
  },
  protocols: {
    href: ['http', 'https', 'mailto'],
  },
};

/** Stable reference for ReactMarkdown (plugins are stateless). */
const CHAT_MESSAGE_REMARK_PLUGINS = [remarkBreaks, remarkGfm];

function chatMessageCitationsEqual(prevCitations, nextCitations) {
  if (prevCitations === nextCitations) return true;
  if (prevCitations == null && nextCitations == null) return true;
  if (!Array.isArray(prevCitations) || !Array.isArray(nextCitations)) return false;
  if (prevCitations.length !== nextCitations.length) return false;
  for (let i = 0; i < prevCitations.length; i += 1) {
    if (prevCitations[i] !== nextCitations[i]) return false;
  }
  return true;
}

/**
 * Fingerprint for memo equality; false negative = extra re-render only.
 * @param {unknown} artifact
 * @returns {string}
 */
function artifactMemoFingerprint(artifact) {
  if (artifact == null || artifact === '') return String(artifact);
  if (typeof artifact === 'string') {
    try {
      const parsed = JSON.parse(artifact);
      if (parsed && typeof parsed === 'object') {
        return artifactMemoFingerprint(parsed);
      }
    } catch {
      return `rawstr:${artifact.length}`;
    }
    return `str:${artifact.length}`;
  }
  if (typeof artifact !== 'object') return String(artifact);
  const code = artifact.code;
  const codeLen =
    typeof code === 'string' ? code.length : code != null ? String(code).length : 0;
  return `${artifact.artifact_type ?? ''}:${artifact.id ?? artifact.message_id ?? ''}:${(artifact.title || '').slice(0, 120)}:${codeLen}`;
}

function artifactPayloadsMemoEqual(prevMeta, nextMeta) {
  if (artifactMemoFingerprint(prevMeta?.artifact ?? null) !== artifactMemoFingerprint(nextMeta?.artifact ?? null)) {
    return false;
  }
  const pas = prevMeta?.artifacts ?? null;
  const nas = nextMeta?.artifacts ?? null;
  if (pas === nas) return true;
  if (pas == null && nas == null) return true;
  if (Array.isArray(pas) && Array.isArray(nas)) {
    if (pas.length !== nas.length) return false;
    for (let i = 0; i < pas.length; i += 1) {
      if (artifactMemoFingerprint(pas[i]) !== artifactMemoFingerprint(nas[i])) return false;
    }
    return true;
  }
  return artifactMemoFingerprint(pas) === artifactMemoFingerprint(nas);
}

/**
 * ChatMessage - Memoized individual message component
 * Extracted from ChatMessagesArea to prevent unnecessary re-renders
 */
const ChatMessage = React.memo(({
  message,
  index,
  messageListIndex,
  inThreadSearchQuery = '',
  inThreadSearchActive = false,
  isLoading,
  theme,
  aiName,
  markdownComponents,
  handleContextMenu,
  handleImportImage,
  formatTimestamp,
  handleCopyMessage,
  handleSaveAsMarkdown,
  isHITLPermissionRequest,
  handleHITLResponse,
  hasResearchPlan,
  executingPlans,
  extractImageUrls,
  getImageApiUrl,
  openLightbox,
  currentConversationId,
  copiedMessageId,
  savingNoteFor,
  isAdmin,
  has,
  onEditAndResend,
  onSwitchBranch,
  siblingInfo,
  anyMessageStreaming,
  setActiveArtifact,
  openArtifact,
  activeArtifact = null,
  artifactCollapsed = false,
}) => {
  const [imageDetailIndex, setImageDetailIndex] = useState(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editDraft, setEditDraft] = useState('');

  const rehypePluginsForMessage = useMemo(() => {
    const plugins = [rehypeRaw];
    if (inThreadSearchActive && inThreadSearchQuery.trim()) {
      plugins.push([
        rehypeHighlightChatSearch,
        { query: inThreadSearchQuery.trim(), markScrollTarget: true },
      ]);
    }
    plugins.push([rehypeSanitize, CHAT_MESSAGE_REHYPE_SANITIZE_OPTIONS]);
    return plugins;
  }, [inThreadSearchActive, inThreadSearchQuery]);

  // Signal Corps: Render notifications as centered, borderless pills
  if (message.type === 'notification') {
    const severityColor = {
      'info': 'info',
      'success': 'success',
      'warning': 'warning',
      'error': 'error'
    }[message.severity] || 'info';
    
    // Determine background color for dark mode
    let bgColor = 'transparent';
    if (theme.palette.mode === 'dark') {
      if (severityColor === 'info') {
        bgColor = 'rgba(33, 150, 243, 0.1)';
      } else if (severityColor === 'success') {
        bgColor = 'rgba(76, 175, 80, 0.1)';
      } else if (severityColor === 'warning') {
        bgColor = 'rgba(255, 152, 0, 0.1)';
      } else if (severityColor === 'error') {
        bgColor = 'rgba(244, 67, 54, 0.1)';
      }
    }
    
    return (
      <Box
        key={message.id || index}
        data-chat-message-index={messageListIndex ?? index}
        sx={{
          width: '100%',
          display: 'flex',
          justifyContent: 'center',
          my: 1
        }}
      >
        <Chip
          label={message.content}
          color={severityColor}
          variant="outlined"
          size="small"
          sx={{
            fontStyle: 'italic',
            opacity: 0.8,
            backgroundColor: bgColor,
            borderColor: (theme.palette[severityColor] && theme.palette[severityColor].main) || theme.palette.info.main,
            '& .MuiChip-label': {
              fontSize: '0.75rem',
              px: 1.5
            }
          }}
        />
      </Box>
    );
  }

  const serverMessageId = message.message_id;
  const canEditUser =
    message.role === 'user' &&
    onEditAndResend &&
    serverMessageId &&
    !isLoading &&
    !anyMessageStreaming;

  const startEditingUserMessage = () => {
    setEditDraft(message.content || '');
    setIsEditing(true);
  };

  const submitUserMessageEdit = async () => {
    const text = editDraft.trim();
    if (!text || !serverMessageId) return;
    setIsEditing(false);
    await onEditAndResend(serverMessageId, text);
  };

  // Regular message rendering
  return (
    <Box
      key={message.id || index}
      data-chat-message-index={messageListIndex ?? index}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: message.role === 'user' ? 'flex-end' : 'flex-start',
      }}
    >
      {siblingInfo &&
        siblingInfo.total > 1 &&
        onSwitchBranch && (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.25,
              mb: 0.5,
              opacity: 0.9,
              alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start',
            }}
          >
            <IconButton
              size="small"
              aria-label="Previous branch"
              onClick={() => onSwitchBranch(serverMessageId || message.id, -1)}
            >
              <ChevronLeft fontSize="small" />
            </IconButton>
            <Typography variant="caption" color="text.secondary" sx={{ minWidth: 36, textAlign: 'center' }}>
              {siblingInfo.index + 1} / {siblingInfo.total}
            </Typography>
            <IconButton
              size="small"
              aria-label="Next branch"
              onClick={() => onSwitchBranch(serverMessageId || message.id, 1)}
            >
              <ChevronRight fontSize="small" />
            </IconButton>
          </Box>
        )}
      <Paper
        elevation={1}
        onContextMenu={(e) => handleContextMenu(e, message)}
        sx={{
          p: 2,
          maxWidth: '85%',
          backgroundColor: (() => {
            const lineRole = (message.metadata?.line_role || '').toLowerCase();
            const isLine =
              Boolean(message.metadata?.line_id) ||
              Boolean(message.metadata?.line_dispatch_sub_agent);
            if (message.role !== 'user' && isLine) {
              if (lineRole === 'ceo' || lineRole === 'root')
                return theme.palette.mode === 'dark'
                  ? 'rgba(156, 39, 176, 0.12)'
                  : 'rgba(103, 58, 183, 0.06)';
              return theme.palette.mode === 'dark'
                ? 'rgba(33, 150, 243, 0.1)'
                : 'rgba(33, 150, 243, 0.04)';
            }
            return message.role === 'user'
              ? theme.palette.mode === 'dark'
                ? alpha(theme.palette.primary.main, 0.4)
                : 'primary.light'
              : message.isError
                ? 'error.light'
                : message.isToolStatus
                  ? 'action.hover'
                  : 'background.paper';
          })(),
          border: message.isError ? '1px solid' : message.isToolStatus ? '1px dashed' : 'none',
          borderColor: message.isError ? 'error.main' : message.isToolStatus ? 'primary.main' : 'transparent',
          cursor: 'context-menu',
        }}
      >
        {/* Message Header */}
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: 1, 
          mb: 1,
          justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
          userSelect: 'none'
        }}>
          {message.role !== 'user' && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
              {message.metadata?.line_id || message.metadata?.line_dispatch_sub_agent ? (
                <Groups sx={{ width: 18, height: 18, color: 'text.secondary' }} />
              ) : (
                <Box
                  component="img"
                  src="/images/favicon.ico"
                  sx={{ width: 18, height: 18, objectFit: 'contain' }}
                />
              )}
              <Typography 
                variant="caption" 
                color="text.secondary"
                sx={{ fontSize: '0.75rem', fontWeight: 600 }}
              >
                {message.metadata?.persona_ai_name || aiName}
              </Typography>
              {(message.metadata?.agent_display_name || message.metadata?.agent_type || message.agent_type) && (
                <Chip 
                  size="small" 
                  label={
                    message.metadata?.agent_display_name ||
                    (message.metadata?.agent_type || message.agent_type)
                      .split('_')
                      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                      .join(' ')
                  }
                  sx={{ 
                    height: 16, 
                    fontSize: '0.6rem', 
                    backgroundColor: 'rgba(0, 0, 0, 0.05)',
                    '& .MuiChip-label': { px: 1 }
                  }} 
                />
              )}
              {message.metadata?.line_role && (
                <Chip
                  size="small"
                  label={String(message.metadata.line_role)}
                  sx={{
                    height: 16,
                    fontSize: '0.6rem',
                    backgroundColor: 'rgba(0, 0, 0, 0.08)',
                    '& .MuiChip-label': { px: 1, textTransform: 'capitalize' },
                  }}
                />
              )}
            </Box>
          )}
          
          {message.role === 'user' && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography 
                variant="caption" 
                color="text.secondary"
                sx={{ fontSize: '0.75rem', fontWeight: 600 }}
              >
                You
              </Typography>
              <Person fontSize="small" sx={{ color: 'primary.main' }} />
            </Box>
          )}
        </Box>

        {/* Lesson Preview Card (if lesson_data present) */}
        {message.metadata?.lesson_data && message.role !== 'user' && (
          <Box sx={{ mb: 2 }}>
            <LessonPreviewCard lesson={message.metadata.lesson_data} />
          </Box>
        )}

        {(() => {
          const arts = parseArtifactsFromMessageMetadata(message.metadata);
          const open = openArtifact || setActiveArtifact;
          if (!arts.length || message.role === 'user' || !open) return null;
          return (
            <Box sx={{ mb: 2 }}>
              {arts.map((art, i) => (
                <Box
                  key={`${art.title || 'artifact'}-${art.artifact_type}-${i}-${String(art.code || '').slice(0, 40)}`}
                  sx={{ mb: 1 }}
                >
                  <ArtifactCard
                    artifact={art}
                    onOpen={(payload) => open(payload)}
                    activeArtifact={activeArtifact}
                    artifactCollapsed={artifactCollapsed}
                  />
                </Box>
              ))}
            </Box>
          );
        })()}

        {/* Message Content */}
        <Box 
          sx={{ 
            mb: 1, 
            userSelect: 'text',
            WebkitUserSelect: 'text',
            MozUserSelect: 'text',
            msUserSelect: 'text',
          }}
        >
          {message.role === 'user' ? (
            <>
              {isEditing ? (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, width: '100%' }}>
                  <TextField
                    multiline
                    minRows={2}
                    maxRows={16}
                    fullWidth
                    value={editDraft}
                    onChange={(e) => setEditDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key !== 'Enter') return;
                      if (e.ctrlKey || e.metaKey || e.shiftKey) return;
                      e.preventDefault();
                      void submitUserMessageEdit();
                    }}
                    size="small"
                    variant="outlined"
                    autoFocus
                  />
                  <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                    <Button size="small" onClick={() => setIsEditing(false)}>
                      Cancel
                    </Button>
                    <Button
                      size="small"
                      variant="contained"
                      disabled={!editDraft.trim()}
                      onClick={() => void submitUserMessageEdit()}
                    >
                      Submit
                    </Button>
                  </Box>
                </Box>
              ) : (
                <Typography
                  variant="body2"
                  component="div"
                  role={canEditUser ? 'button' : undefined}
                  tabIndex={canEditUser ? 0 : undefined}
                  title={canEditUser ? 'Click to edit' : undefined}
                  onClick={canEditUser ? () => startEditingUserMessage() : undefined}
                  onKeyDown={
                    canEditUser
                      ? (e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            startEditingUserMessage();
                          }
                        }
                      : undefined
                  }
                  sx={{
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    color: message.isError ? 'error.main' : 'text.primary',
                    ...(canEditUser && {
                      cursor: 'pointer',
                      borderRadius: 0.5,
                      outlineOffset: 2,
                      '&:hover': {
                        textDecoration: 'underline',
                        textDecorationColor: 'currentColor',
                        textUnderlineOffset: '0.2em',
                      },
                    }),
                    '& mark.chat-in-thread-search-mark': {
                      backgroundColor:
                        theme.palette.mode === 'dark'
                          ? 'rgba(255, 213, 79, 0.35)'
                          : 'rgba(255, 213, 79, 0.55)',
                      color: 'inherit',
                      padding: '0 2px',
                      borderRadius: '2px',
                    },
                  }}
                >
                  {inThreadSearchActive && inThreadSearchQuery.trim()
                    ? highlightPlainTextSearch(message.content, inThreadSearchQuery)
                    : message.content}
                </Typography>
              )}
            </>
          ) : (
            <Box sx={{ 
              color: message.isError ? 'error.main' : 'text.primary',
              '& .markdown-content': {
                whiteSpace: 'normal',  // ROOSEVELT'S NEWLINE FIX - Let remarkBreaks handle line breaks
                wordBreak: 'break-word',
                lineHeight: 1.6,
                // Ensure text selection works properly without jumping
                userSelect: 'text',
                WebkitUserSelect: 'text',
              },
              '& pre': { 
                margin: '8px 0',
                borderRadius: '4px',
                overflow: 'auto',
                backgroundColor: 'rgba(0, 0, 0, 0.05)',
                padding: '12px'
              },
              '& code': {
                fontFamily: 'monospace',
                backgroundColor: 'rgba(0, 0, 0, 0.1)',
                padding: '2px 4px',
                borderRadius: '3px',
                fontSize: '0.9em'
              },
              '& p': {
                marginBottom: '12px',
                whiteSpace: 'normal',  // ROOSEVELT'S FIX - Ensure paragraphs don't conflict
                userSelect: 'text',
                // Prevent double-click from selecting entire paragraph
                WebkitUserSelect: 'text',
              },
              '& h1, & h2, & h3, & h4, & h5, & h6': {
                marginTop: '16px',
                marginBottom: '8px'
              },
              '& ul, & ol': {
                marginBottom: '12px',
                paddingLeft: '20px'
              },
              '& li': {
                marginBottom: '4px'
              },
              '& blockquote': {
                margin: '16px 0',
                padding: '8px 16px'
              },
              '& mark.chat-in-thread-search-mark': {
                backgroundColor:
                  theme.palette.mode === 'dark'
                    ? 'rgba(255, 213, 79, 0.35)'
                    : 'rgba(255, 213, 79, 0.55)',
                color: 'inherit',
                padding: '0 2px',
                borderRadius: '2px',
              },
            }}>
              <ReactMarkdown
                className="markdown-content"
                components={markdownComponents}
                remarkPlugins={CHAT_MESSAGE_REMARK_PLUGINS}
                rehypePlugins={rehypePluginsForMessage}
              >
                {message.content || ''}
              </ReactMarkdown>
            </Box>
          )}

          {/* ROOSEVELT'S ENHANCED CITATION DISPLAY: Support new numbered format */}
          {(message.metadata?.citations || message.citations) && renderCitations(message.metadata?.citations || message.citations)}

          {/* Editor proposals are shown as inline diffs in DocumentViewer (DB-only path) */}

          {/* Message Attachments */}
          {message.metadata?.attachments && Array.isArray(message.metadata.attachments) && message.metadata.attachments.length > 0 && (
            <Box sx={{ mt: 1.5, display: 'flex', flexDirection: 'column', gap: 1 }}>
              {message.metadata.attachments.map((attachment, idx) => {
                const isImage = attachment.content_type?.startsWith('image/');
                const isAudio = attachment.content_type?.startsWith('audio/');
                const attachmentUrl = `/api/conversations/${currentConversationId}/messages/${message.id || message.message_id}/attachments/${attachment.attachment_id}`;
                
                if (isImage) {
                  return (
                    <Paper key={`attachment-${idx}`} variant="outlined" sx={{ p: 1 }}>
                      <Box
                        component="img"
                        src={attachmentUrl}
                        alt={attachment.filename || 'Attachment'}
                        onClick={() => openLightbox(attachmentUrl, { alt: attachment.filename })}
                        sx={{
                          maxWidth: '100%',
                          maxHeight: '300px',
                          height: 'auto',
                          borderRadius: 1,
                          display: 'block',
                          cursor: 'pointer',
                          objectFit: 'contain'
                        }}
                      />
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                        {attachment.filename} ({(attachment.size_bytes / 1024 / 1024).toFixed(2)}MB)
                      </Typography>
                    </Paper>
                  );
                } else if (isAudio) {
                  return (
                    <Paper key={`attachment-${idx}`} variant="outlined" sx={{ p: 1 }}>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        {attachment.filename} ({(attachment.size_bytes / 1024 / 1024).toFixed(2)}MB)
                      </Typography>
                      <audio controls style={{ width: '100%' }}>
                        <source src={attachmentUrl} type={attachment.content_type} />
                        Your browser does not support the audio element.
                      </audio>
                    </Paper>
                  );
                } else {
                  return (
                    <Paper key={`attachment-${idx}`} variant="outlined" sx={{ p: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body2" sx={{ flex: 1 }}>
                        {attachment.filename} ({(attachment.size_bytes / 1024 / 1024).toFixed(2)}MB)
                      </Typography>
                      <Button
                        size="small"
                        variant="outlined"
                        href={attachmentUrl}
                        download={attachment.filename}
                      >
                        Download
                      </Button>
                    </Paper>
                  );
                }
              })}
            </Box>
          )}

          {/* Image Previews: generated images (Import) vs in-system documents (no Import) */}
          {message.role === 'assistant' && (() => {
            // CRITICAL: Check metadata.images FIRST (structured data from AgentResponse)
            // Fallback to extracting from markdown if no structured images
            let structuredImages = [];
            if (message.metadata && message.metadata.images) {
              try {
                // Parse JSON if it's a string, or use directly if already parsed
                structuredImages = typeof message.metadata.images === 'string'
                  ? JSON.parse(message.metadata.images)
                  : message.metadata.images;
              } catch (e) {
                console.error('Failed to parse metadata.images:', e);
                structuredImages = [];
              }
            }
            
            // Fallback: extract from markdown if no structured images
            const imageUrls = structuredImages.length > 0 
              ? structuredImages.map(img => img.url) 
              : extractImageUrls(message.content);
            
            return imageUrls.length > 0 ? (
              <Box mt={1.5} display="flex" flexDirection="column" gap={1.5}>
                {imageUrls.map((url, idx) => {
                  const displayUrl = url.startsWith('data:') ? url : getImageApiUrl(url);
                  const fromCollection = isDocumentFileUrl(url);
                  
                  // Get metadata for this image if available
                  const imageMetadata = structuredImages[idx]?.metadata || {};
                  const imageTitle = imageMetadata.title || 'Image';
                  const imageSeries = imageMetadata.series;
                  const imageDate = imageMetadata.date;
                  
                  return (
                    <Paper key={`${url}-${idx}`} variant="outlined" sx={{ p: 1.5 }}>
                      <AuthDocumentImage
                        url={displayUrl}
                        alt={fromCollection ? imageTitle : 'Generated image'}
                        onClick={(e, srcForLightbox) => openLightbox(srcForLightbox || displayUrl, { alt: fromCollection ? imageTitle : 'Generated image' })}
                        sx={{
                          maxWidth: '100%',
                          height: 'auto',
                          borderRadius: 1,
                          display: 'block',
                          cursor: 'pointer'
                        }}
                      />
                      {!fromCollection && (
                        <Box mt={1} display="flex" gap={1}>
                          <Button
                            onClick={() => handleImportImage(displayUrl)}
                            size="small"
                            variant="outlined"
                          >
                            Import
                          </Button>
                        </Box>
                      )}
                      {fromCollection && (
                        <Box sx={{ mt: 0.5 }}>
                          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                            {imageTitle}
                            {imageSeries && ` · ${imageMetadata.author ? `${imageSeries} by ${imageMetadata.author}` : imageSeries}`}
                          </Typography>
                          {(imageMetadata.content || imageMetadata.match_reason) && (
                            <Button
                              size="small"
                              variant="text"
                              sx={{ mt: 0.25, p: 0, minWidth: 0, fontSize: '0.75rem', textTransform: 'none' }}
                              onClick={() => setImageDetailIndex(idx)}
                            >
                              Why it matches
                            </Button>
                          )}
                        </Box>
                      )}
                    </Paper>
                  );
                })}
                {/* Modal: full description and why it matches (collection search results) */}
                {imageDetailIndex !== null && structuredImages[imageDetailIndex] && (() => {
                  const meta = structuredImages[imageDetailIndex].metadata || {};
                  const title = meta.title || 'Image';
                  const hasContent = meta.content || meta.match_reason || (meta.tags && meta.tags.length);
                  if (!hasContent) return null;
                  return (
                    <Dialog
                      open={true}
                      onClose={() => setImageDetailIndex(null)}
                      maxWidth="sm"
                      fullWidth
                      PaperProps={{ sx: { borderRadius: 2 } }}
                    >
                      <DialogTitle sx={{ pb: 0 }}>
                        {title}
                        {meta.series && (
                          <Typography variant="body2" color="text.secondary" component="span" sx={{ display: 'block', mt: 0.25 }}>
                            {meta.series}{meta.author ? ` by ${meta.author}` : ''}{meta.date ? ` · ${meta.date}` : ''}
                          </Typography>
                        )}
                      </DialogTitle>
                      <DialogContent>
                        {meta.match_reason && (
                          <Typography variant="body2" sx={{ mb: 1.5, color: 'primary.main', fontWeight: 600 }}>
                            Matches your query: {meta.match_reason}
                          </Typography>
                        )}
                        {meta.content && (
                          <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'pre-wrap' }}>
                            {meta.content}
                          </Typography>
                        )}
                        {meta.tags && meta.tags.length > 0 && (
                          <Box sx={{ mt: 1.5, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                            {meta.tags.map((tag, i) => (
                              <Chip key={i} label={tag} size="small" variant="outlined" />
                            ))}
                          </Box>
                        )}
                      </DialogContent>
                      <DialogActions>
                        <Button size="small" onClick={() => setImageDetailIndex(null)}>Close</Button>
                      </DialogActions>
                    </Dialog>
                  );
                })()}
              </Box>
            ) : null;
          })()}
        </Box>

        {/* Async Task Progress */}
        {/* HITL Permission Request Actions */}
        {isHITLPermissionRequest(message) && (
          <Box mt={2}>
            <Box display="flex" gap={1} mb={1}>
              <Button
                variant="contained"
                color="success"
                size="small"
                onClick={() => handleHITLResponse('Yes')}
                disabled={isLoading}
                sx={{ minWidth: '80px' }}
              >
                {isLoading ? 'Sending...' : 'Yes'}
              </Button>
              <Button
                variant="outlined"
                color="error"
                size="small"
                onClick={() => handleHITLResponse('No')}
                disabled={isLoading}
                sx={{ minWidth: '80px' }}
              >
                {isLoading ? 'Sending...' : 'No'}
              </Button>
            </Box>
            <Typography variant="caption" color="text.secondary" display="block">
              🛡️ Click "Yes" to auto-approve web search or "No" to use local resources only. Response will be sent automatically.
            </Typography>
          </Box>
        )}

        {/* Research Plan Actions */}
        {hasResearchPlan(message) && !isHITLPermissionRequest(message) && (
          <Box mt={2}>
            {message.planApproved ? (
              <Box display="flex" alignItems="center" gap={1}>
                <Chip 
                  label="✅ Plan Approved & Executing" 
                  color="success" 
                  size="small"
                  variant="outlined"
                />
                <Typography variant="caption" color="text.secondary">
                  Research tools are running based on this plan
                </Typography>
              </Box>
            ) : (
              <Box>
                {executingPlans && executingPlans.has(message.jobId || message.metadata?.job_id) ? (
                  <Box display="flex" alignItems="center" gap={1}>
                    <Button
                      variant="outlined"
                      color="info"
                      size="small"
                      disabled={true}
                      sx={{ mr: 1 }}
                    >
                      In Progress
                    </Button>
                    <Typography variant="caption" color="text.secondary" display="block" mt={0.5}>
                      Research plan is currently being executed. Please wait for completion.
                    </Typography>
                  </Box>
                ) : null}
              </Box>
            )}
          </Box>
        )}

        {/* Reactions Display */}
        {(() => {
          const metadata = message.metadata || message.metadata_json || {};
          const reactions = metadata.reactions || {};
          const hasReactions = Object.keys(reactions).length > 0;
          
          if (!hasReactions) return null;
          
          return (
            <Box sx={{ 
              display: 'flex', 
              flexWrap: 'wrap', 
              gap: 0.5, 
              mt: 1,
              pt: 1,
              borderTop: '1px solid',
              borderColor: 'divider'
            }}>
              {Object.entries(reactions).map(([emoji, userIds]) => {
                if (!userIds || userIds.length === 0) return null;
                return (
                  <Chip
                    key={emoji}
                    label={`${emoji} ${userIds.length}`}
                    size="small"
                    sx={{ 
                      height: 24,
                      fontSize: '0.75rem',
                      '& .MuiChip-label': { px: 0.5 }
                    }}
                  />
                );
              })}
            </Box>
          );
        })()}

        {/* Message Footer */}
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          gap: 1,
          userSelect: 'none',
          mt: 1
        }}>
          {(() => {
            const durationLabel =
              message.role !== 'user' ? formatDurationMs(message.metadata?.duration_ms) : '';
            const skillsUsed =
              message.role !== 'user' ? parseJsonStringArray(message.metadata?.skills_used) : [];
            const toolCategories =
              message.role !== 'user' ? parseJsonStringArray(message.metadata?.tools_used_categories) : [];

            return (
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.75,
                  flexWrap: 'wrap',
                  minWidth: 0,
                }}
              >
                <Typography variant="caption" color="text.secondary">
                  {formatTimestamp(message.timestamp)}
                </Typography>

                {durationLabel && (
                  <Typography variant="caption" color="text.secondary">
                    · {durationLabel}
                  </Typography>
                )}

                {skillsUsed.length > 0 && (
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    {skillsUsed.map((skillName) => (
                      <Chip
                        key={skillName}
                        size="small"
                        label={formatSkillLabel(skillName)}
                        sx={{
                          height: 16,
                          fontSize: '0.6rem',
                          backgroundColor: 'rgba(0, 0, 0, 0.05)',
                          '& .MuiChip-label': { px: 1 },
                        }}
                      />
                    ))}
                  </Box>
                )}

                {toolCategories.length > 0 && (
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    {toolCategories.map((cat) => (
                      <Chip
                        key={cat}
                        size="small"
                        variant="outlined"
                        label={formatCategoryLabel(cat)}
                        sx={{
                          height: 16,
                          fontSize: '0.6rem',
                          '& .MuiChip-label': { px: 1 },
                        }}
                      />
                    ))}
                  </Box>
                )}
              </Box>
            );
          })()}
          
          <Box sx={{ display: 'flex', gap: 0.5 }}>
            {message.role !== 'user' && (
              <ReadAloudButton content={message.content} isUser={message.role === 'user'} />
            )}
            <ExportButton
              message={message}
              onCopyMessage={handleCopyMessage}
              onSaveAsNote={handleSaveAsMarkdown}
              copiedMessageId={copiedMessageId}
              savingNoteFor={savingNoteFor}
              currentConversationId={currentConversationId}
              isUser={message.role === 'user'}
            />
          </Box>
        </Box>
      </Paper>
    </Box>
  );
}, (prevProps, nextProps) => {
  // Custom comparison function for React.memo
  // Only re-render if message content or relevant props change
  return (
    prevProps.message.id === nextProps.message.id &&
    prevProps.message.content === nextProps.message.content &&
    prevProps.message.role === nextProps.message.role &&
    prevProps.message.isError === nextProps.message.isError &&
    prevProps.message.isStreaming === nextProps.message.isStreaming &&
    prevProps.message.isToolStatus === nextProps.message.isToolStatus &&
    chatMessageCitationsEqual(
      prevProps.message.metadata?.citations || prevProps.message.citations,
      nextProps.message.metadata?.citations || nextProps.message.citations,
    ) &&
    (prevProps.message.metadata?.duration_ms || null) === (nextProps.message.metadata?.duration_ms || null) &&
    (prevProps.message.metadata?.skills_used || null) === (nextProps.message.metadata?.skills_used || null) &&
    (prevProps.message.metadata?.tools_used_categories || null) === (nextProps.message.metadata?.tools_used_categories || null) &&
    (prevProps.message.metadata?.line_id || null) === (nextProps.message.metadata?.line_id || null) &&
    (prevProps.message.metadata?.line_role || null) === (nextProps.message.metadata?.line_role || null) &&
    (prevProps.message.metadata?.agent_display_name || null) === (nextProps.message.metadata?.agent_display_name || null) &&
    (prevProps.message.metadata?.persona_ai_name || null) === (nextProps.message.metadata?.persona_ai_name || null) &&
    artifactPayloadsMemoEqual(prevProps.message.metadata, nextProps.message.metadata) &&
    (prevProps.message.jobId || prevProps.message.metadata?.job_id || null) ===
      (nextProps.message.jobId || nextProps.message.metadata?.job_id || null) &&
    prevProps.message.isPermissionRequest === nextProps.message.isPermissionRequest &&
    prevProps.message.requiresApproval === nextProps.message.requiresApproval &&
    prevProps.message.planApproved === nextProps.message.planApproved &&
    prevProps.message.research_plan === nextProps.message.research_plan &&
    prevProps.setActiveArtifact === nextProps.setActiveArtifact &&
    prevProps.openArtifact === nextProps.openArtifact &&
    prevProps.artifactCollapsed === nextProps.artifactCollapsed &&
    prevProps.activeArtifact?.code === nextProps.activeArtifact?.code &&
    prevProps.isLoading === nextProps.isLoading &&
    prevProps.copiedMessageId === nextProps.copiedMessageId &&
    prevProps.savingNoteFor === nextProps.savingNoteFor &&
    prevProps.anyMessageStreaming === nextProps.anyMessageStreaming &&
    prevProps.onEditAndResend === nextProps.onEditAndResend &&
    prevProps.onSwitchBranch === nextProps.onSwitchBranch &&
    prevProps.siblingInfo?.index === nextProps.siblingInfo?.index &&
    prevProps.siblingInfo?.total === nextProps.siblingInfo?.total &&
    prevProps.inThreadSearchQuery === nextProps.inThreadSearchQuery &&
    prevProps.inThreadSearchActive === nextProps.inThreadSearchActive &&
    prevProps.index === nextProps.index &&
    prevProps.messageListIndex === nextProps.messageListIndex &&
    prevProps.theme?.palette?.mode === nextProps.theme?.palette?.mode &&
    prevProps.aiName === nextProps.aiName &&
    prevProps.markdownComponents === nextProps.markdownComponents &&
    prevProps.handleContextMenu === nextProps.handleContextMenu &&
    prevProps.handleImportImage === nextProps.handleImportImage &&
    prevProps.formatTimestamp === nextProps.formatTimestamp &&
    prevProps.handleCopyMessage === nextProps.handleCopyMessage &&
    prevProps.handleSaveAsMarkdown === nextProps.handleSaveAsMarkdown &&
    prevProps.isHITLPermissionRequest === nextProps.isHITLPermissionRequest &&
    prevProps.handleHITLResponse === nextProps.handleHITLResponse &&
    prevProps.hasResearchPlan === nextProps.hasResearchPlan &&
    prevProps.extractImageUrls === nextProps.extractImageUrls &&
    prevProps.getImageApiUrl === nextProps.getImageApiUrl &&
    prevProps.openLightbox === nextProps.openLightbox &&
    prevProps.currentConversationId === nextProps.currentConversationId &&
    prevProps.isAdmin === nextProps.isAdmin &&
    prevProps.has === nextProps.has &&
    prevProps.executingPlans === nextProps.executingPlans
  );
});

ChatMessage.displayName = 'ChatMessage';

export default ChatMessage;
