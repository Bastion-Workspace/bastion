import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import {
  Box,
  Typography,
  Paper,
  IconButton,
  Tooltip,
  CircularProgress,
  Alert,
  Button,
  Chip,
  useTheme,
  Dialog,
  DialogContent,
  Menu,
  MenuItem,
  Popover,
} from '@mui/material';
import {
  AutoAwesome,
  CloudUpload,
  Fullscreen,
  Close,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import ExportButton from './ExportButton';
import { useChatSidebar } from '../../contexts/ChatSidebarContext';
import apiService from '../../services/apiService';
import ReactMarkdown from 'react-markdown';
import remarkBreaks from 'remark-breaks';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { materialLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { markdownToPlainText, renderCitations, smartCopy } from '../../utils/chatUtils';
import { useCapabilities } from '../../contexts/CapabilitiesContext';
import EditorOpsPreviewModal from './EditorOpsPreviewModal';
import FolderSelectionDialog from './FolderSelectionDialog';
import { useImageLightbox } from '../common/ImageLightbox';
import ChatMessage from './ChatMessage';

/**
 * ROOSEVELT'S CHART RENDERER: Safely renders standalone Plotly HTML in an iframe
 * Now with support for Library Import!
 */
const ChartRenderer = ({ html, staticData, staticFormat, onImport, onFullScreen }) => {
  return (
    <Box 
      sx={{ 
        my: 2, 
        width: '100%', 
        overflow: 'hidden',
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
        backgroundColor: '#fff',
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      {/* Action header */}
      <Box sx={{ 
        p: 1, 
        borderBottom: '1px solid', 
        borderColor: 'divider', 
        display: 'flex', 
        justifyContent: 'space-between',
        alignItems: 'center',
        bgcolor: 'rgba(0, 0, 0, 0.02)'
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
            Interactive Visualization
          </Typography>
          <Tooltip title="View Full Screen">
            <IconButton size="small" onClick={() => onFullScreen(html)}>
              <Fullscreen fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
        {staticData && (
          <Button 
            size="small" 
            variant="text" 
            startIcon={<CloudUpload fontSize="small" />}
            onClick={() => onImport(staticData, staticFormat)}
            sx={{ textTransform: 'none', py: 0 }}
          >
            Import to Library
          </Button>
        )}
      </Box>

      {/* Interactive Iframe */}
      <Box sx={{ height: '500px', width: '100%' }}>
        <iframe
          srcDoc={html}
          title="Visualization"
          width="100%"
          height="100%"
          style={{ border: 'none' }}
          sandbox="allow-scripts allow-same-origin"
        />
      </Box>
    </Box>
  );
};

const ChatMessagesArea = ({ darkMode: darkModeProp }) => {
  const theme = useTheme();
  const { 
    messages, 
    setMessages,
    isLoading, 
    currentConversationId, 
    executingPlans,
    replyToMessage,
    setReplyToMessage,
    sendMessage,
    backgroundJobService
  } = useChatSidebar();
  const { isAdmin, has } = useCapabilities();
  const { openLightbox } = useImageLightbox();

  const { data: conversationData, isLoading: conversationLoading } = useQuery(
    ['conversation', currentConversationId],
    () => currentConversationId ? apiService.getConversation(currentConversationId) : null,
    {
      enabled: !!currentConversationId,
      refetchOnWindowFocus: false,
      staleTime: 300000, // 5 minutes
    }
  );

  const { data: messagesData, isLoading: messagesLoading } = useQuery(
    ['conversationMessages', currentConversationId],
    () => currentConversationId ? apiService.getConversationMessages(currentConversationId) : null,
    {
      enabled: !!currentConversationId,
      refetchOnWindowFocus: false,
      staleTime: 300000, // 5 minutes
    }
  );
  const messagesEndRef = useRef(null);

  // Handle HITL permission response - DIRECT API CALL VERSION
  const handleHITLResponse = async (response) => {
    console.log('🛡️ HITL Response - Direct submission:', response);
    
    try {
      // ROOSEVELT'S DIRECT CHARGE: Use sendMessage with override parameter
      // sendMessage will handle adding the user message, so we don't duplicate it here
      await sendMessage('auto', response);
      
      console.log('✅ HITL response sent directly via sendMessage override');
    } catch (error) {
      console.error('❌ Failed to send HITL response directly:', error);
      
      // Show user what happened
      setMessages?.(prev => [...prev, {
        id: Date.now(),
        role: 'system',
        type: 'system',
        content: `⚠️ Auto-submission failed. Please copy and resend: "${response}"`,
        timestamp: new Date().toISOString(),
        isError: true
      }]);
    }
  };

  // Fetch AI name from prompt settings
  const { data: promptSettings } = useQuery(
    'promptSettings',
    () => apiService.getPromptSettings(),
    {
      staleTime: 300000, // 5 minutes
      refetchOnWindowFocus: false,
    }
  );

  // Get AI name from settings, fallback to "Alex"
  const aiName = promptSettings?.ai_name || 'Alex';

  // ROOSEVELT'S INTELLIGENT AUTO-SCROLL: Only scroll when user is near bottom or new message arrives
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
  const [userHasScrolled, setUserHasScrolled] = useState(false);
  const messagesContainerRef = useRef(null);
  const scrollTimeoutRef = useRef(null);
  const lastScrollTopRef = useRef(0);
  const [hasTextSelection, setHasTextSelection] = useState(false);
  const [fullScreenChart, setFullScreenChart] = useState(null);
  const lastMessageCountRef = useRef(0);  // Track actual NEW messages vs updates
  const isScrollingRef = useRef(false);  // Track if user is actively scrolling

  // Track text selection anywhere in the document and scope it to this container
  useEffect(() => {
    const handleSelectionChange = () => {
      try {
        const selection = document.getSelection();
        const container = messagesContainerRef.current;
        if (!container || !selection) {
          setHasTextSelection((prev) => prev ? false : prev);
          return;
        }
        const hasRange = selection.rangeCount > 0 && !selection.isCollapsed;
        if (!hasRange) {
          setHasTextSelection((prev) => prev ? false : prev);
          return;
        }
        const anchorNode = selection.anchorNode;
        const focusNode = selection.focusNode;
        const within = (anchorNode && container.contains(anchorNode)) || (focusNode && container.contains(focusNode));
        setHasTextSelection((prev) => prev !== !!within ? !!within : prev);
        
        // Prevent selection from being cleared by scroll or other events
        // Only if selection is within our container
        if (within && hasRange) {
          // Don't interfere with the selection, just track it
        }
      } catch {
        setHasTextSelection((prev) => prev ? false : prev);
      }
    };

    document.addEventListener('selectionchange', handleSelectionChange);
    return () => {
      document.removeEventListener('selectionchange', handleSelectionChange);
    };
  }, []);

  // Check if user is near bottom of messages
  const isNearBottom = () => {
    if (!messagesContainerRef.current) return true;
    const container = messagesContainerRef.current;
    const threshold = 100; // pixels from bottom
    return container.scrollHeight - container.scrollTop - container.clientHeight < threshold;
  };

  // Debounced scroll handler to prevent excessive state updates
  const handleScroll = useCallback(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    // Mark that user is actively scrolling
    isScrollingRef.current = true;

    // Clear any pending timeout
    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current);
    }

    // Detect if user scrolled up manually
    const currentScrollTop = container.scrollTop;
    const hasScrolledUp = currentScrollTop < lastScrollTopRef.current;
    
    if (hasScrolledUp) {
      setUserHasScrolled(true);
    }
    
    lastScrollTopRef.current = currentScrollTop;

    // Debounce the auto-scroll decision
    scrollTimeoutRef.current = setTimeout(() => {
      const nearBottom = isNearBottom();
      setShouldAutoScroll(nearBottom);
      
      // Reset user scroll flag when back near bottom
      if (nearBottom) {
        setUserHasScrolled(false);
      }

      // Mark scrolling as finished after debounce
      isScrollingRef.current = false;
    }, 150); // 150ms debounce - slightly longer to be more forgiving
  }, []);

  // ROOSEVELT'S ENHANCED AUTO-SCROLL: Only scroll on NEW messages, not updates
  useEffect(() => {
    const currentMessageCount = messages.length;
    const hasNewMessages = currentMessageCount > lastMessageCountRef.current;
    
    // Update the ref for next comparison
    lastMessageCountRef.current = currentMessageCount;

    // ROOSEVELT'S STRICT NO-SCROLL CONDITIONS: Don't interrupt the user!
    // 1. User is selecting text
    // 2. User has manually scrolled up and is not near bottom
    // 3. User is actively scrolling right now
    // 4. No new messages (just an update to existing message content)
    if (
      hasTextSelection || 
      (userHasScrolled && !isNearBottom()) ||
      isScrollingRef.current ||
      !hasNewMessages
    ) {
      return;
    }

    // Only auto-scroll if we should and there are messages
    if (shouldAutoScroll && messages.length > 0) {
      // Use requestAnimationFrame for smoother timing
      requestAnimationFrame(() => {
        // Double-check user hasn't started scrolling or selecting text in the meantime
        const currentSelection = window.getSelection();
        const hasActiveSelection = currentSelection && currentSelection.rangeCount > 0 && !currentSelection.isCollapsed;
        
        if (!isScrollingRef.current && !hasActiveSelection) {
          messagesEndRef.current?.scrollIntoView({ 
            behavior: 'smooth',
            block: 'end',
            inline: 'nearest'
          });
        }
      });
    }
  }, [messages, shouldAutoScroll, userHasScrolled, hasTextSelection]);

  // Add scroll listener to messages container with cleanup
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (container) {
      // Use passive listener for better performance
      container.addEventListener('scroll', handleScroll, { passive: true });
      
      return () => {
        container.removeEventListener('scroll', handleScroll);
        if (scrollTimeoutRef.current) {
          clearTimeout(scrollTimeoutRef.current);
        }
      };
    }
  }, [handleScroll]);

  const [copiedMessageId, setCopiedMessageId] = useState(null);
  const [savingNoteFor, setSavingNoteFor] = useState(null);
  const [previewOpenFor, setPreviewOpenFor] = useState(null);
  const [folderDialogOpen, setFolderDialogOpen] = useState(false);
  const [imageToImport, setImageToImport] = useState(null);
  const [importing, setImporting] = useState(false);
  const [contextMenu, setContextMenu] = useState(null);
  const [contextMenuMessage, setContextMenuMessage] = useState(null);
  const [reactionAnchor, setReactionAnchor] = useState(null);
  const [reactionMessage, setReactionMessage] = useState(null);

  const handleCopyMessage = async (message) => {
    try {
      setCopiedMessageId(message.id);
      const content = String(message?.content || '');
      const doCopy = async () => {
        try {
          // For very large messages, avoid heavy markdown conversion on main thread
          if (content.length > 50000) {
            await navigator.clipboard.writeText(content);
          } else {
            // Use smartCopy to preserve markdown formatting as rich text (HTML)
            const success = await smartCopy(content);
            if (!success) {
              // Fallback to plain text if rich text copy fails
              await navigator.clipboard.writeText(content);
            }
          }
        } catch (copyErr) {
          console.error('Failed to copy message:', copyErr);
        } finally {
          setTimeout(() => { setCopiedMessageId(null); }, 1200);
        }
      };
      // Yield to the browser to paint the spinner before heavy work
      if (typeof requestIdleCallback === 'function') {
        requestIdleCallback(() => setTimeout(doCopy, 0), { timeout: 250 });
      } else {
        setTimeout(doCopy, 0);
      }
    } catch (err) {
      console.error('Failed to schedule copy:', err);
    }
  };

  const handleSaveAsMarkdown = async (message) => {
    if (!currentConversationId || !message.message_id) {
      console.error('Cannot save message: missing conversation ID or message ID');
      return;
    }

    try {
      setSavingNoteFor(message.id);
      
      // Get the conversation details to use as context
      const conversation = await apiService.getConversation(currentConversationId);
      const conversationTitle = conversation?.title || 'Chat Conversation';
      
      // Create a filename based on conversation title and message timestamp
      const timestamp = new Date().toISOString().split('T')[0];
      const sanitizedTitle = conversationTitle.replace(/[^a-zA-Z0-9\s-]/g, '').replace(/\s+/g, '-');
      const filename = `${sanitizedTitle}-${timestamp}.md`;
      
      // Create markdown content
      const markdownContent = `# ${conversationTitle}

**Date:** ${new Date().toLocaleDateString()}
**Time:** ${new Date().toLocaleTimeString()}
**Message Type:** ${message.role === 'user' ? 'User Question' : 'Assistant Response'}

## Message Content

${message.content}

---
*Saved from conversation: ${conversationTitle}*
`;

      // Create the markdown file using the existing note creation API
      const noteData = {
        title: filename.replace('.md', ''),
        content: markdownContent,
        category: 'chat-export',
        tags: ['chat', 'export', message.role]
      };
      
      const result = await apiService.createNote(noteData);
      console.log('Message saved as markdown:', result);
      alert('Message saved as markdown file successfully!');
    } catch (error) {
      console.error('Failed to save message as markdown:', error);
      alert('Failed to save message as markdown. Please try again.');
    } finally {
      setSavingNoteFor(null);
    }
  };

  const handleImportImage = (imageUrl) => {
    setImageToImport(imageUrl);
    setFolderDialogOpen(true);
  };

  const handleFolderSelect = async (folder) => {
    if (!imageToImport || !folder) return;
    
    try {
      setImporting(true);
      
      // Extract filename from URL if possible
      const urlParts = imageToImport.split('/');
      const filename = urlParts[urlParts.length - 1];
      
      const result = await apiService.importImage(
        imageToImport,
        filename,
        folder.folder_id
      );
      
      if (result && result.document_id) {
        alert(`Image imported successfully to folder: ${folder.name}`);
        setFolderDialogOpen(false);
        setImageToImport(null);
      } else {
        throw new Error('Import failed - no document ID returned');
      }
    } catch (error) {
      console.error('Failed to import image:', error);
      alert(`Failed to import image: ${error.message || 'Unknown error'}`);
    } finally {
      setImporting(false);
    }
  };

  const handleContextMenu = (event, message) => {
    event.preventDefault();
    setContextMenu(
      contextMenu === null
        ? {
            mouseX: event.clientX + 2,
            mouseY: event.clientY - 6,
          }
        : null,
    );
    setContextMenuMessage(message);
  };

  const handleCloseContextMenu = () => {
    setContextMenu(null);
    setContextMenuMessage(null);
  };

  const handleReply = () => {
    if (contextMenuMessage) {
      setReplyToMessage(contextMenuMessage);
      handleCloseContextMenu();
    }
  };

  const handleReact = () => {
    if (contextMenuMessage) {
      setReactionMessage(contextMenuMessage);
      setReactionAnchor(contextMenu);
      handleCloseContextMenu();
    }
  };

  const handleReactionSelect = async (emoji) => {
    if (!reactionMessage || !currentConversationId) {
      return;
    }

    try {
      const messageId = reactionMessage.message_id || reactionMessage.id;
      const response = await apiService.post(`/api/conversations/${currentConversationId}/messages/${messageId}/react`, {
        emoji: emoji
      });

      // Update local message state to reflect reaction from backend response
      if (response && response.reactions) {
        setMessages(prev => prev.map(msg => {
          if (msg.id === reactionMessage.id || msg.message_id === messageId) {
            const metadata = msg.metadata || msg.metadata_json || {};
            return {
              ...msg,
              metadata: { ...metadata, reactions: response.reactions },
              metadata_json: { ...metadata, reactions: response.reactions }
            };
          }
          return msg;
        }));
      }

      setReactionAnchor(null);
      setReactionMessage(null);
    } catch (error) {
      console.error('Failed to add reaction:', error);
      alert('Failed to add reaction. Please try again.');
    }
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Extract image URLs from assistant message content for preview rendering
  const extractImageUrls = (text) => {
    try {
      if (!text || typeof text !== 'string') return [];
      const urls = [];
      
      // Match data URIs for base64 images (e.g., from comic search)
      const dataUriRegex = /!\[([^\]]*)\]\((data:image\/[^)]+)\)/g;
      let dataMatch;
      while ((dataMatch = dataUriRegex.exec(text)) !== null) {
        urls.push(dataMatch[2]); // Extract the data URI
      }
      
      // Match markdown images with /api/images/ URLs (e.g., from image generation)
      const apiImagesRegex = /!\[([^\]]*)\]\((\/api\/images\/[^)]+)\)/g;
      let apiMatch;
      while ((apiMatch = apiImagesRegex.exec(text)) !== null) {
        urls.push(apiMatch[2]); // Extract the /api/images/ URL
      }

      // Match markdown images with /api/documents/ file URLs (e.g., from research fast path)
      const apiDocumentsRegex = /!\[([^\]]*)\]\((\/api\/documents\/[^)]+)\)/g;
      let apiDocMatch;
      while ((apiDocMatch = apiDocumentsRegex.exec(text)) !== null) {
        urls.push(apiDocMatch[2]);
      }
      
      // Match HTTP/HTTPS URLs, static images, comic API endpoints, and document file URLs
      const regex = /(https?:\/\/[^\s)]+|\/static\/images\/[^\s)]+|\/api\/comics\/[^\s)]+|\/api\/images\/[^\s)]+|\/api\/documents\/[^\s)]+\/file)/g;
      let match;
      while ((match = regex.exec(text)) !== null) {
        const url = match[1];
        if (typeof url === 'string') {
          const lower = url.toLowerCase();
          // Document file URLs (research fast path) - no extension, served as image
          if (lower.includes('/api/documents/') && lower.endsWith('/file')) {
            urls.push(url);
          } else if (lower.endsWith('.png') || lower.endsWith('.jpg') || lower.endsWith('.jpeg') ||
              lower.endsWith('.webp') || lower.endsWith('.gif')) {
            urls.push(url);
          }
        }
      }
      return Array.from(new Set(urls));
    } catch {
      return [];
    }
  };

  // Convert static image URL to API endpoint for proper content-type headers
  const getImageApiUrl = (url) => {
    if (url.startsWith('/static/images/')) {
      const filename = url.replace('/static/images/', '');
      return `/api/images/${filename}`;
    }
    // API endpoints are already correct - return as-is
    if (url.startsWith('/api/comics/') || url.startsWith('/api/images/') || url.startsWith('/api/documents/')) {
      return url;
    }
    return url; // Return as-is for external URLs
  };

  // Open image in lightbox
  const handleOpenImage = (url) => {
      const apiUrl = getImageApiUrl(url);
    // Extract filename from URL if possible
    const filename = url.split('/').pop().split('?')[0];
    openLightbox(apiUrl, { filename, alt: 'Generated image' });
  };

  // Check if a message contains a research plan
  const hasResearchPlan = (message) => {
    return (
      message.research_plan ||
      (message.content && (
        message.content.includes('## Research Plan') ||
        message.content.includes('**Research Plan**') ||
        message.content.includes('### Research Plan') ||
        message.content.includes('Research Plan:') ||
        (message.content.includes('Step') && message.content.includes('Research'))
      ))
    );
  };

  // Check if a message is a HITL permission request
  const isHITLPermissionRequest = (message) => {
    // ROOSEVELT'S ENHANCED HITL: Use new tagging system first, fallback to content detection
    if (message.isPermissionRequest && message.requiresApproval) {
      return true;
    }
    
    // Fallback to content-based detection for legacy messages
    return (
      message.role === 'assistant' && 
      message.content && (
        message.content.includes('🔍 Web Search Permission Request') ||
        message.content.includes('Permission Request') ||
        message.content.includes('May I proceed') ||
        message.content.includes('Do you approve') ||
        message.content.includes('web search permission') ||
        message.content.includes('search the web') ||
        message.content.includes('external search') ||
        message.content.includes('Would you like me to proceed') ||
        (message.content.includes('Yes') && message.content.includes('No') && message.content.includes('permission'))
      )
    );
  };

  // Custom markdown components for better styling (memoized to avoid re-rendering all ReactMarkdown children)
  const markdownComponents = useMemo(() => ({
    // Style code blocks - ROOSEVELT'S ENHANCED CODE BLOCK HANDLING
    code: ({ node, inline, className, children, staticData, staticFormat, onImport, onFullScreen, ...props }) => {
      const match = /language-([\w:]+)/.exec(className || '');
      
      // Handle HTML chart visualizations - render as HTML in ChartRenderer
      if (!inline && match && match[1] === 'html:chart') {
        return (
          <ChartRenderer 
            html={String(children).replace(/\n$/, '')} 
            staticData={staticData}
            staticFormat={staticFormat}
            onImport={onImport}
            onFullScreen={onFullScreen}
          />
        );
      }
      
      return !inline && match ? (
        <SyntaxHighlighter
          style={materialLight}
          language={match[1].split(':')[0]} // Handle potential colons in language name for highlighter
          PreTag="div"
          {...props}
        >
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      ) : (
        <code className={className} {...props} style={{ 
          backgroundColor: 'rgba(0, 0, 0, 0.1)', 
          padding: '2px 4px', 
          borderRadius: '3px',
          fontSize: '0.9em'
        }}>
          {children}
        </code>
      );
    },
    
    // ROOSEVELT'S ENHANCED PRE HANDLING
    pre: ({ children, ...props }) => {
      return <pre {...props}>{children}</pre>;
    },
    
    // Style paragraphs with proper spacing - ROOSEVELT'S BLOCK DISPLAY FIX
    p: ({ children, ...props }) => (
      <Typography 
        variant="body2" 
        component="p" 
        sx={{ 
          mb: 1.5, 
          lineHeight: 1.6,
          display: 'block',
          width: '100%'
        }} 
        {...props}
      >
        {children}
      </Typography>
    ),
    
    // Style headings with proper hierarchy and spacing - ROOSEVELT'S BLOCK DISPLAY FIX
    h1: ({ children, ...props }) => (
      <Typography 
        variant="h4" 
        component="h1" 
        sx={{ 
          mb: 2, 
          mt: 3, 
          fontWeight: 700, 
          lineHeight: 1.3,
          display: 'block',
          width: '100%'
        }} 
        {...props}
      >
        {children}
      </Typography>
    ),
    h2: ({ children, ...props }) => (
      <Typography 
        variant="h5" 
        component="h2" 
        sx={{ 
          mb: 1.5, 
          mt: 2.5, 
          fontWeight: 600, 
          lineHeight: 1.4,
          display: 'block',
          width: '100%'
        }} 
        {...props}
      >
        {children}
      </Typography>
    ),
    h3: ({ children, ...props }) => (
      <Typography 
        variant="h6" 
        component="h3" 
        sx={{ 
          mb: 1, 
          mt: 2, 
          fontWeight: 600, 
          lineHeight: 1.4,
          display: 'block',
          width: '100%'
        }} 
        {...props}
      >
        {children}
      </Typography>
    ),
    h4: ({ children, ...props }) => (
      <Typography 
        variant="subtitle1" 
        component="h4" 
        sx={{ 
          mb: 1, 
          mt: 1.5, 
          fontWeight: 600, 
          lineHeight: 1.5,
          display: 'block',
          width: '100%'
        }} 
        {...props}
      >
        {children}
      </Typography>
    ),
    h5: ({ children, ...props }) => (
      <Typography 
        variant="subtitle2" 
        component="h5" 
        sx={{ 
          mb: 0.5, 
          mt: 1, 
          fontWeight: 600, 
          lineHeight: 1.5,
          display: 'block',
          width: '100%'
        }} 
        {...props}
      >
        {children}
      </Typography>
    ),
    h6: ({ children, ...props }) => (
      <Typography 
        variant="body1" 
        component="h6" 
        sx={{ 
          mb: 0.5, 
          mt: 1, 
          fontWeight: 600, 
          lineHeight: 1.5,
          display: 'block',
          width: '100%'
        }} 
        {...props}
      >
        {children}
      </Typography>
    ),
    
    // Style lists with proper spacing
    ul: ({ children, ...props }) => (
      <Box component="ul" sx={{ mb: 1.5, pl: 2, '& li': { mb: 0.5 } }} {...props}>
        {children}
      </Box>
    ),
    ol: ({ children, ...props }) => (
      <Box component="ol" sx={{ mb: 1.5, pl: 2, '& li': { mb: 0.5 } }} {...props}>
        {children}
      </Box>
    ),
    li: ({ children, ...props }) => (
      <Typography variant="body2" component="li" sx={{ mb: 0.5, lineHeight: 1.6 }} {...props}>
        {children}
      </Typography>
    ),
    
    // Style blockquotes with enhanced styling
    blockquote: ({ children, ...props }) => (
      <Box
        component="blockquote"
        sx={{
          borderLeft: '4px solid',
          borderColor: 'primary.main',
          pl: 2,
          ml: 0,
          my: 2,
          py: 1,
          fontStyle: 'italic',
          color: 'text.secondary',
          backgroundColor: 'rgba(25, 118, 210, 0.04)',
          borderRadius: '0 4px 4px 0'
        }}
        {...props}
      >
        {children}
      </Box>
    ),
    
    // Style links with better hover effects
    a: ({ children, href, ...props }) => (
      <Typography
        component="a"
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        sx={{
          color: 'primary.main',
          textDecoration: 'none',
          '&:hover': {
            textDecoration: 'underline',
            color: 'primary.dark'
          }
        }}
        {...props}
      >
        {children}
      </Typography>
    ),
    
    // Style strong/bold text
    strong: ({ children, ...props }) => (
      <Typography component="span" sx={{ fontWeight: 600 }} {...props}>
        {children}
      </Typography>
    ),
    
    // Style emphasis/italic text
    em: ({ children, ...props }) => (
      <Typography component="span" sx={{ fontStyle: 'italic' }} {...props}>
        {children}
      </Typography>
    ),
    
    // Style strikethrough text
    del: ({ children, ...props }) => (
      <Typography component="span" sx={{ textDecoration: 'line-through', color: 'text.secondary' }} {...props}>
        {children}
      </Typography>
    ),
    
    // Style horizontal rules
    hr: ({ ...props }) => (
      <Box
        component="hr"
        sx={{
          border: 'none',
          borderTop: '1px solid',
          borderColor: 'divider',
          my: 2,
          mx: 0
        }}
        {...props}
      />
    ),
    
    // Style tables (if using remarkGfm)
    table: ({ children, ...props }) => (
      <Box
        component="table"
        sx={{
          borderCollapse: 'collapse',
          width: '100%',
          mb: 2,
          '& th, & td': {
            border: '1px solid',
            borderColor: 'divider',
            padding: '8px 12px',
            textAlign: 'left'
          },
          '& th': {
            backgroundColor: 'action.hover',
            fontWeight: 600
          }
        }}
        {...props}
      >
        {children}
      </Box>
    ),
    
    // Style table headers
    th: ({ children, ...props }) => (
      <Typography component="th" variant="body2" sx={{ fontWeight: 600 }} {...props}>
        {children}
      </Typography>
    ),
    
    // Style table cells
    td: ({ children, ...props }) => (
      <Typography component="td" variant="body2" {...props}>
        {children}
      </Typography>
    ),
    
    // Handle chart images (base64 data URIs) and regular images
    img: ({ node, src, alt, ...props }) => {
      // Check if this is a base64 data URI image (charts, generated images, embedded images)
      if (src?.startsWith('data:image/')) {
        return (
          <Box sx={{ my: 2, textAlign: 'center' }}>
            <img 
              src={src}
              alt={alt || 'Image'}
              {...props}
              onClick={() => handleOpenImage(src)}
              style={{
                maxWidth: '100%',
                height: 'auto',
                border: '1px solid',
                borderColor: theme.palette.divider,
                borderRadius: theme.shape.borderRadius,
                cursor: 'pointer',
                ...props.style
              }}
            />
          </Box>
        );
      }
      // Regular image - fix double "Comics" in path for /api/comics URLs
      let imageSrc = src;
      if (src && src.startsWith('/api/comics/')) {
        // Remove duplicate "Comics" segment if present
        // e.g., /api/comics/Comics/Dilbert/... -> /api/comics/Dilbert/...
        imageSrc = src.replace(/^\/api\/comics\/Comics\//, '/api/comics/');
      }
      
      return (
        <Box sx={{ my: 2, textAlign: 'center' }}>
          <img 
            src={imageSrc}
            alt={alt}
            {...props}
            onClick={() => handleOpenImage(imageSrc)}
            style={{
              maxWidth: '100%',
              height: 'auto',
              cursor: 'pointer',
              ...props.style
            }}
          />
        </Box>
      );
    },
  }), [handleOpenImage, theme]);

  if (!currentConversationId) {
    return (
      <Box sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        height: '100%',
        p: 3
      }}>
        <Typography variant="body2" color="text.secondary" textAlign="center">
          Type a message below to start a new conversation
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ 
      height: '100%', 
      display: 'flex', 
      flexDirection: 'column',
      overflow: 'hidden'
    }}>
      {/* Messages Container */}
      <Box 
        ref={messagesContainerRef}
        onMouseDown={(e) => {
          // Prevent scroll from interfering with text selection
          // If user is starting a text selection, don't let scroll reset it
          const selection = window.getSelection();
          if (selection && selection.rangeCount > 0 && !selection.isCollapsed) {
            // User has an active selection, be careful not to clear it
            // Only prevent default if clicking on empty space
            if (e.target === e.currentTarget) {
              // Clicking on container background, allow normal behavior
            }
          }
        }}
        sx={{ 
          flexGrow: 1, 
          overflow: 'auto',
          p: 2,
          display: 'flex',
          flexDirection: 'column',
          gap: 2
        }}
      >
        {/* ROOSEVELT'S IMPROVED EMPTY STATE: Better feedback during conversation restoration */}
        {messages.length === 0 && !isLoading && !conversationLoading && !messagesLoading && (
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            height: '100%'
          }}>
            <Typography variant="body2" color="text.secondary" textAlign="center">
              {currentConversationId ? 
                "This conversation appears to be empty. Start chatting below!" :
                "No messages yet. Start the conversation!"
              }
            </Typography>
          </Box>
        )}

        {/* ROOSEVELT'S CONVERSATION RESTORATION INDICATOR: Show loading when restoring conversation */}
        {currentConversationId && (conversationLoading || messagesLoading) && messages.length === 0 && (
          <Box sx={{ 
            display: 'flex', 
            flexDirection: 'column',
            alignItems: 'center', 
            justifyContent: 'center',
            height: '100%',
            gap: 2
          }}>
            <CircularProgress size={32} />
            <Typography variant="body2" color="text.secondary" textAlign="center">
              🔄 Restoring conversation...
            </Typography>
            <Typography variant="caption" color="text.secondary" textAlign="center">
              {conversationData?.conversation?.title || `Conversation ${currentConversationId.substring(0, 8)}...`}
            </Typography>
          </Box>
        )}

        {messages.map((message, index) => {
          // ROOSEVELT'S REDUNDANCY CHECK: Don't render streaming messages while loading 
          // (the loading indicator at the bottom handles this feedback with agent name and rotating icon)
          if (isLoading && 
              index === messages.length - 1 && 
              (message.role === 'assistant' || message.type === 'assistant') && 
              (message.isStreaming || !message.content || message.content.trim() === '')) {
            return null;
          }

          return (
            <ChatMessage
              key={message.id || index}
              message={message}
              index={index}
              isLoading={isLoading}
              theme={theme}
              aiName={aiName}
              markdownComponents={markdownComponents}
              handleContextMenu={handleContextMenu}
              handleImportImage={handleImportImage}
              setFullScreenChart={setFullScreenChart}
              formatTimestamp={formatTimestamp}
              handleCopyMessage={handleCopyMessage}
              handleSaveAsMarkdown={handleSaveAsMarkdown}
              isHITLPermissionRequest={isHITLPermissionRequest}
              handleHITLResponse={handleHITLResponse}
              hasResearchPlan={hasResearchPlan}
              executingPlans={executingPlans}
              extractImageUrls={extractImageUrls}
              getImageApiUrl={getImageApiUrl}
              openLightbox={openLightbox}
              setPreviewOpenFor={setPreviewOpenFor}
              currentConversationId={currentConversationId}
              copiedMessageId={copiedMessageId}
              savingNoteFor={savingNoteFor}
              isAdmin={isAdmin}
              has={has}
            />
          );
        })}

        {/* Loading indicator */}
        {isLoading && (
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'flex-start',
            gap: 1.5,
            py: 1.5,
            px: 2,
            ml: 1,
            mb: 2,
            backgroundColor: 'background.paper',
            borderRadius: 2,
            boxShadow: 1,
            width: 'fit-content',
            maxWidth: '85%',
            border: '1px solid',
            borderColor: 'divider',
          }}>
            <Box sx={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <CircularProgress size={28} thickness={4} sx={{ color: 'secondary.main' }} />
              <Box 
                component="img" 
                src="/images/favicon.ico" 
                sx={{ 
                  position: 'absolute', 
                  width: 16, 
                  height: 16, 
                  objectFit: 'contain' 
                }} 
              />
            </Box>
            <Box>
              <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.primary', display: 'block', lineHeight: 1.2 }}>
                {aiName}
              </Typography>
              {(() => {
                const lastMessage = messages[messages.length - 1];
                const agentType = lastMessage?.metadata?.agent_type || lastMessage?.agent_type;
                if (agentType) {
                  return (
                    <Typography variant="caption" sx={{ color: 'secondary.main', fontSize: '0.65rem', fontWeight: 600 }}>
                      {agentType.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')} is chewing on it...
                    </Typography>
                  );
                }
                return (
                  <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.65rem' }}>
                    Chewing on it...
                  </Typography>
                );
              })()}
            </Box>
          </Box>
        )}

        {/* Scroll anchor */}
        <div ref={messagesEndRef} />
      </Box>

      {/* Editor Operations Preview Modal */}
      {(() => {
        const msg = messages.find(m => (m.id || m.timestamp) === previewOpenFor);
        const open = !!previewOpenFor && msg && Array.isArray(msg.editor_operations) && msg.editor_operations.length > 0;
        const close = () => setPreviewOpenFor(null);
        const onApplySelected = (ops, manuscriptEdit) => {
          try {
            window.dispatchEvent(new CustomEvent('codexApplyEditorOps', { detail: { operations: ops, manuscript_edit: manuscriptEdit || (msg ? msg.manuscript_edit : null) } }));
          } catch (e) {
            console.error('Failed to apply selected operations:', e);
          }
        };
        return (
          <EditorOpsPreviewModal
            key={previewOpenFor || 'preview-modal'}
            open={open}
            onClose={close}
            operations={open ? msg.editor_operations : []}
            manuscriptEdit={open ? (msg.manuscript_edit || null) : null}
            requestEditorContent={() => window.dispatchEvent(new CustomEvent('codexRequestEditorContent'))}
            onApplySelected={onApplySelected}
          />
        );
      })()}

      {/* Folder Selection Dialog for Image Import */}
      <FolderSelectionDialog
        open={folderDialogOpen}
        onClose={() => {
          setFolderDialogOpen(false);
          setImageToImport(null);
        }}
        onSelect={handleFolderSelect}
      />

      {/* Full Screen Chart Dialog */}
      <Dialog
        fullScreen
        open={!!fullScreenChart}
        onClose={() => setFullScreenChart(null)}
      >
        <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
          <Box sx={{ 
            p: 1, 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            borderBottom: '1px solid',
            borderColor: 'divider',
            bgcolor: 'background.paper'
          }}>
            <Typography variant="h6" sx={{ ml: 2 }}>Visualization Full Screen</Typography>
            <IconButton onClick={() => setFullScreenChart(null)}>
              <Close />
            </IconButton>
          </Box>
          <Box sx={{ flexGrow: 1, bgcolor: '#fff' }}>
            {fullScreenChart && (
              <iframe
                srcDoc={fullScreenChart}
                title="Full Screen Visualization"
                width="100%"
                height="100%"
                style={{ border: 'none' }}
                sandbox="allow-scripts allow-same-origin"
              />
            )}
          </Box>
        </Box>
      </Dialog>

      {/* ROOSEVELT'S "RETURN TO BOTTOM" CAVALRY BUTTON - Show when user has scrolled up */}
      {userHasScrolled && !shouldAutoScroll && (
        <Box
          sx={{
            position: 'absolute',
            bottom: 16,
            right: 16,
            zIndex: 1000,
          }}
        >
          <Button
            variant="contained"
            size="small"
            onClick={() => {
              setUserHasScrolled(false);
              setShouldAutoScroll(true);
              messagesEndRef.current?.scrollIntoView({ 
                behavior: 'smooth',
                block: 'end'
              });
            }}
            sx={{
              borderRadius: '20px',
              px: 2,
              py: 1,
              backgroundColor: 'primary.main',
              '&:hover': {
                backgroundColor: 'primary.dark',
              },
              boxShadow: 2,
            }}
          >
            ↓ New Messages
          </Button>
        </Box>
      )}

      {/* Context Menu for Reply and React */}
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
        <MenuItem onClick={handleReply}>Reply</MenuItem>
        <MenuItem onClick={handleReact}>React</MenuItem>
      </Menu>

      {/* Reaction Selection Popover */}
      <Popover
        open={Boolean(reactionAnchor)}
        anchorReference="anchorPosition"
        anchorPosition={
          reactionAnchor !== null
            ? { top: reactionAnchor.mouseY, left: reactionAnchor.mouseX }
            : undefined
        }
        onClose={() => {
          setReactionAnchor(null);
          setReactionMessage(null);
        }}
      >
        <Box sx={{ p: 1, display: 'flex', gap: 1 }}>
          {['👍', '👎', '😂', '❤️', '😢'].map((emoji) => (
            <IconButton
              key={emoji}
              onClick={() => handleReactionSelect(emoji)}
              sx={{ fontSize: '1.5rem', p: 1 }}
            >
              {emoji}
            </IconButton>
          ))}
        </Box>
      </Popover>
    </Box>
  );
};

export default React.memo(ChatMessagesArea); 