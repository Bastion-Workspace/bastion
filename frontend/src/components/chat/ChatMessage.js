import React, { useState, useEffect } from 'react';
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
} from '@mui/material';
import {
  Person,
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkBreaks from 'remark-breaks';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';
import { renderCitations } from '../../utils/chatUtils';
import ExportButton from './ExportButton';
import LessonPreviewCard from '../LessonPreviewCard';

/** True if URL is an in-system document file (research result); needs auth and no Import. */
const isDocumentFileUrl = (url) =>
  typeof url === 'string' && url.includes('/api/documents/') && url.includes('/file');

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

/**
 * ChatMessage - Memoized individual message component
 * Extracted from ChatMessagesArea to prevent unnecessary re-renders
 */
const ChatMessage = React.memo(({
  message,
  index,
  isLoading,
  theme,
  aiName,
  markdownComponents,
  handleContextMenu,
  handleImportImage,
  setFullScreenChart,
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
  setPreviewOpenFor,
  currentConversationId,
  copiedMessageId,
  savingNoteFor,
  isAdmin,
  has,
}) => {
  const [imageDetailIndex, setImageDetailIndex] = useState(null);

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

  // Regular message rendering
  return (
    <Box
      key={message.id || index}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: message.role === 'user' ? 'flex-end' : 'flex-start',
      }}
    >
      <Paper
        elevation={1}
        onContextMenu={(e) => handleContextMenu(e, message)}
        sx={{
          p: 2,
          maxWidth: '85%',
          backgroundColor: message.role === 'user' 
            ? (theme.palette.mode === 'dark' 
                ? 'rgba(25, 118, 210, 0.4)' 
                : 'primary.light')
            : message.isError 
              ? 'error.light'
              : message.isToolStatus
                ? 'action.hover'
                : 'background.paper',
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
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box 
                component="img" 
                src="/images/favicon.ico" 
                sx={{ width: 18, height: 18, objectFit: 'contain' }} 
              />
              <Typography 
                variant="caption" 
                color="text.secondary"
                sx={{ fontSize: '0.75rem', fontWeight: 600 }}
              >
                {aiName}
              </Typography>
              {(message.metadata?.agent_type || message.agent_type) && (
                <Chip 
                  size="small" 
                  label={
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
            <Typography 
              variant="body2" 
              sx={{ 
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                color: message.isError ? 'error.main' : 'text.primary',
              }}
            >
              {message.content}
            </Typography>
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
              }
            }}>
              {(() => {
                const displayContent = message.content || '';
                
                const handleChartImport = (data, format) => {
                  // For SVG, we convert to a Data URI
                  let dataUri = data;
                  if (format === 'svg' && !data.startsWith('data:')) {
                    dataUri = `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(data)))}`;
                  } else if (format === 'base64_png' && !data.startsWith('data:')) {
                    dataUri = `data:image/png;base64,${data}`;
                  }
                  handleImportImage(dataUri);
                };

                return (
                  <ReactMarkdown 
                    className="markdown-content"
                    components={{
                      ...markdownComponents,
                      code: (props) => markdownComponents.code({
                        ...props,
                        staticData: message.metadata?.static_visualization_data,
                        staticFormat: message.metadata?.static_format,
                        onImport: handleChartImport,
                        onFullScreen: (html) => setFullScreenChart(html)
                      })
                    }}
                    remarkPlugins={[remarkBreaks, remarkGfm]}
                    rehypePlugins={[
                      rehypeRaw,
                      [
                        rehypeSanitize,
                        {
                          tagNames: [
                            'details', 'summary', 'div', 'span',
                            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                            'p', 'br', 'strong', 'em', 'b', 'i', 'u', 's',
                            'ul', 'ol', 'li',
                            'blockquote', 'pre', 'code',
                            'table', 'thead', 'tbody', 'tr', 'th', 'td',
                            'a', 'img',
                            'hr'
                          ],
                          attributes: {
                            '*': ['class', 'id'],
                            'a': ['href', 'title'],
                            'img': ['src', 'alt', 'title', 'width', 'height'],
                            'div': ['style'],
                            'span': ['style'],
                            'details': ['open'],
                            'summary': []
                          },
                          protocols: {
                            href: ['http', 'https', 'mailto']
                            // NOTE: Not restricting 'src' protocols - allow data URIs, absolute URLs, and relative paths
                          }
                        }
                      ]
                    ]}
                  >
                    {displayContent}
                  </ReactMarkdown>
                );
              })()}
            </Box>
          )}

          {/* ROOSEVELT'S ENHANCED CITATION DISPLAY: Support new numbered format */}
          {(message.metadata?.citations || message.citations) && renderCitations(message.metadata?.citations || message.citations)}

          {/* Fiction editing HITL controls */}
          {message.role === 'assistant' && Array.isArray(message.editor_operations) && message.editor_operations.length > 0 && (
            <Box sx={{ mt: 2 }}>
              {/* ROOSEVELT'S BEFORE/AFTER EDIT PREVIEW */}
              <Box sx={{ mb: 2, display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                {message.editor_operations.slice(0, 3).map((op, idx) => {
                  const original = op.original_text || op.anchor_text || '';
                  const newText = op.text || '';
                  const opType = op.op_type || 'replace_range';
                  const isInsert = opType === 'insert_after_heading' || opType === 'insert_after';
                  
                  return (
                    <Paper key={idx} variant="outlined" sx={{ p: 1.5, bgcolor: 'background.default' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <Chip 
                          size="small" 
                          label={`Edit ${idx + 1}`} 
                          color="primary" 
                          variant="outlined"
                        />
                        <Chip 
                          size="small" 
                          label={isInsert ? 'insert' : 'replace'} 
                          color={isInsert ? 'success' : 'warning'}
                        />
                      </Box>
                      
                      {original && (
                        <Box sx={{ mb: 1 }}>
                          <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600, display: 'block', mb: 0.5 }}>
                            {isInsert ? 'Insert after:' : 'Replace:'}
                          </Typography>
                          <Box sx={{ 
                            p: 1, 
                            bgcolor: 'rgba(211, 47, 47, 0.08)', 
                            border: '1px solid rgba(211, 47, 47, 0.2)',
                            borderRadius: 1,
                            fontFamily: 'monospace',
                            fontSize: '0.875rem',
                            whiteSpace: 'pre-wrap',
                            maxHeight: '80px',
                            overflow: 'hidden',
                            position: 'relative'
                          }}>
                            {original.length > 150 ? original.substring(0, 150) + '...' : original}
                          </Box>
                        </Box>
                      )}
                      
                      <Box>
                        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600, display: 'block', mb: 0.5 }}>
                          {isInsert ? 'New text:' : 'With:'}
                        </Typography>
                        <Box sx={{ 
                          p: 1, 
                          bgcolor: 'rgba(46, 125, 50, 0.08)', 
                          border: '1px solid rgba(46, 125, 50, 0.2)',
                          borderRadius: 1,
                          fontFamily: 'monospace',
                          fontSize: '0.875rem',
                          whiteSpace: 'pre-wrap',
                          maxHeight: '80px',
                          overflow: 'hidden'
                        }}>
                          {newText.length > 150 ? newText.substring(0, 150) + '...' : newText}
                        </Box>
                      </Box>
                    </Paper>
                  );
                })}
                
                {message.editor_operations.length > 3 && (
                  <Typography variant="caption" color="text.secondary" sx={{ textAlign: 'center', fontStyle: 'italic' }}>
                    ... and {message.editor_operations.length - 3} more edit{message.editor_operations.length - 3 > 1 ? 's' : ''}
                  </Typography>
                )}
              </Box>
              
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                <Chip size="small" color="primary" label={`${message.editor_operations.length} edit${message.editor_operations.length > 1 ? 's' : ''} ready`} />
                <Button size="small" variant="outlined" onClick={() => setPreviewOpenFor(message.id || message.timestamp || Date.now())}>Review all edits</Button>
                <Button size="small" variant="contained" onClick={() => {
                  try {
                    const ops = Array.isArray(message.editor_operations) ? message.editor_operations : [];
                    const mEdit = message.manuscript_edit || null;
                    window.dispatchEvent(new CustomEvent('codexApplyEditorOps', { detail: { operations: ops, manuscript_edit: mEdit } }));
                  } catch (e) {
                    console.error('Failed to dispatch editor operations apply event:', e);
                  }
                }}>Apply all</Button>
              </Box>
            </Box>
          )}

          {/* News results rendering */}
          {message.role === 'assistant' && (isAdmin || has('feature.news.view')) && Array.isArray(message.news_results) && message.news_results.length > 0 && (
            <Box sx={{ mt: 1.5, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 1 }}>
              {message.news_results.map((h, idx) => (
                <Paper key={`${h.id}-${idx}`} variant="outlined" sx={{ p: 1.5, display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{h.title}</Typography>
                    <Chip size="small" label={h.severity?.toUpperCase() || 'NEWS'} color={h.severity === 'breaking' ? 'error' : h.severity === 'urgent' ? 'warning' : 'default'} />
                  </Box>
                  <Typography variant="body2" color="text.secondary">{h.summary}</Typography>
                  <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                    <Chip size="small" label={`${h.sources_count || 0} sources`} />
                    {typeof h.diversity_score === 'number' && <Chip size="small" label={`diversity ${Math.round((h.diversity_score||0)*100)}%`} />}
                  </Box>
                  <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                    <Button size="small" variant="contained" onClick={() => {
                      try {
                        // Prefer client-side navigation to preserve app state
                        if (window?.history && typeof window.history.pushState === 'function') {
                          window.history.pushState({}, '', `/news/${h.id}`);
                          // Dispatch a popstate event so routers listening can react
                          window.dispatchEvent(new PopStateEvent('popstate'));
                        } else {
                          window.location.href = `/news/${h.id}`;
                        }
                      } catch {
                        window.location.href = `/news/${h.id}`;
                      }
                    }}>Open</Button>
                  </Box>
                </Paper>
              ))}
            </Box>
          )}

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
          <Typography variant="caption" color="text.secondary">
            {formatTimestamp(message.timestamp)}
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 0.5 }}>
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
    JSON.stringify(prevProps.message.metadata?.citations || prevProps.message.citations) === 
    JSON.stringify(nextProps.message.metadata?.citations || nextProps.message.citations) &&
    JSON.stringify(prevProps.message.editor_operations) === 
    JSON.stringify(nextProps.message.editor_operations) &&
    prevProps.isLoading === nextProps.isLoading &&
    prevProps.copiedMessageId === nextProps.copiedMessageId &&
    prevProps.savingNoteFor === nextProps.savingNoteFor
  );
});

ChatMessage.displayName = 'ChatMessage';

export default ChatMessage;
