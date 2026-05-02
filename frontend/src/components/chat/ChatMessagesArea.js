import React, { useRef, useEffect, useState, useCallback, useMemo, forwardRef, useImperativeHandle, memo } from 'react';
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
  Menu,
  MenuItem,
  Popover,
} from '@mui/material';
import { useQuery } from 'react-query';
import ExportButton from './ExportButton';
import { useChatSidebar } from '../../contexts/ChatSidebarContext';
import { getSiblings } from '../../utils/messageTreeUtils';
import apiService from '../../services/apiService';
import ReactMarkdown from 'react-markdown';
import remarkBreaks from 'remark-breaks';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { materialLight, materialDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { markdownToPlainText, renderCitations, smartCopy } from '../../utils/chatUtils';
import { useCapabilities } from '../../contexts/CapabilitiesContext';
import FolderSelectionDialog from './FolderSelectionDialog';
import { useImageLightbox } from '../common/ImageLightbox';
import ChatMessage from './ChatMessage';
import ChatInConversationSearchBar from './ChatInConversationSearchBar';

const BOTTOM_SCROLL_THRESHOLD_PX = 100;

const ChatMessagesArea = forwardRef(function ChatMessagesArea({ darkMode: darkModeProp }, ref) {
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
    backgroundJobService,
    activeLineRouting,
    editAndResendMessage,
    switchBranch,
    messageTree,
    currentNodeId,
    conversationMessagesLoading,
    setActiveArtifact,
    openArtifact,
    activeArtifact,
    artifactCollapsed,
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

  const messagesEndRef = useRef(null);

  // Handle HITL permission response - DIRECT API CALL VERSION
  const handleHITLResponse = useCallback(async (response) => {
    try {
      await sendMessage('auto', response);
    } catch (error) {
      console.error('Failed to send HITL response:', error);
      setMessages?.((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: 'system',
          type: 'system',
          content: `Auto-submission failed. Please copy and resend: "${response}"`,
          timestamp: new Date().toISOString(),
          isError: true,
        },
      ]);
    }
  }, [sendMessage, setMessages]);

  const handleShellApproval = useCallback(
    async (action, message) => {
      const id = message.shellApprovalId;
      if (!id) return;
      try {
        if (action === 'run') {
          await apiService.post(
            `/api/settings/shell-approvals/${encodeURIComponent(id)}/grant`
          );
          await sendMessage('auto', 'yes, run the approved shell command.');
        } else {
          await apiService.post(
            `/api/settings/shell-approvals/${encodeURIComponent(id)}/reject`
          );
          await sendMessage('auto', 'skip, do not run that shell command.');
        }
      } catch (error) {
        console.error('Shell approval action failed:', error);
        setMessages?.((prev) => [
          ...prev,
          {
            id: Date.now(),
            role: 'system',
            type: 'system',
            content:
              action === 'run'
                ? 'Could not record approval. Try again or approve from Settings.'
                : 'Could not record skip. Try again.',
            timestamp: new Date().toISOString(),
            isError: true,
          },
        ]);
      }
    },
    [sendMessage, setMessages]
  );

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

  // Recompute only when branch structure or path shape changes — not on every streaming token.
  // `messages` is read from the render closure when deps change; omitting `messages` avoids work on every streaming tick.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const siblingInfoByMessageId = useMemo(() => {
    const map = {};
    if (!messageTree?.byId?.size || !messages?.length) return map;
    messages.forEach((msg) => {
      const mid = msg.message_id || msg.id;
      if (!mid) return;
      const s = getSiblings(messageTree, mid);
      if (s && s.total > 1) {
        map[mid] = { index: s.index, total: s.total };
      }
    });
    return map;
  }, [messageTree, currentNodeId, messages.length]);

  const anyMessageStreaming = useMemo(
    () => (messages || []).some((m) => m.isStreaming),
    [messages]
  );

  // ROOSEVELT'S INTELLIGENT AUTO-SCROLL: Only scroll when user is near bottom or new message arrives
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
  const [userHasScrolled, setUserHasScrolled] = useState(false);
  const messagesContainerRef = useRef(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchHighlightQuery, setSearchHighlightQuery] = useState('');
  const [searchHighlightMessageIndex, setSearchHighlightMessageIndex] = useState(null);

  const handleSearchActiveMatchChange = useCallback(({ query: q, messageListIndex }) => {
    setSearchHighlightQuery(q ?? '');
    setSearchHighlightMessageIndex(messageListIndex ?? null);
  }, []);

  const handleSearchClose = useCallback(() => {
    setSearchOpen(false);
    setSearchHighlightQuery('');
    setSearchHighlightMessageIndex(null);
  }, []);

  useImperativeHandle(
    ref,
    () => ({
      openConversationSearch: () => setSearchOpen(true),
    }),
    []
  );
  const scrollTimeoutRef = useRef(null);
  const lastScrollTopRef = useRef(0);
  const [hasTextSelection, setHasTextSelection] = useState(false);
  const lastMessageCountRef = useRef(0);  // Track actual NEW messages vs updates
  const isScrollingRef = useRef(false);  // Track if user is actively scrolling
  const messageCountRef = useRef(0);  // Current message count for scroll-up snapshot
  const messageCountWhenScrolledUpRef = useRef(0);  // Count when user scrolled up (to detect new messages below)
  const [hasNewMessagesBelow, setHasNewMessagesBelow] = useState(false);
  /** Updated synchronously on scroll/resize so the "New messages" control hides as soon as the user reaches the bottom */
  const [isNearBottomState, setIsNearBottomState] = useState(true);

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
    return container.scrollHeight - container.scrollTop - container.clientHeight < BOTTOM_SCROLL_THRESHOLD_PX;
  };

  // Debounced scroll handler to prevent excessive state updates
  const handleScroll = useCallback(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const nearBottomNow =
      container.scrollHeight - container.scrollTop - container.clientHeight < BOTTOM_SCROLL_THRESHOLD_PX;
    setIsNearBottomState(nearBottomNow);

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
      messageCountWhenScrolledUpRef.current = messageCountRef.current;
    }

    lastScrollTopRef.current = currentScrollTop;

    // Debounce the auto-scroll decision
    scrollTimeoutRef.current = setTimeout(() => {
      const nearBottom = isNearBottom();
      setShouldAutoScroll(nearBottom);

      // Reset user scroll flag and "new messages below" when back near bottom
      if (nearBottom) {
        setUserHasScrolled(false);
        setHasNewMessagesBelow(false);
        messageCountWhenScrolledUpRef.current = messageCountRef.current;
      }

      // Mark scrolling as finished after debounce
      isScrollingRef.current = false;
    }, 150); // 150ms debounce - slightly longer to be more forgiving
  }, []);

  // Keep current message count in ref for scroll handler
  useEffect(() => {
    messageCountRef.current = messages.length;
  }, [messages]);

  // Reset "new messages below" state when switching conversations
  useEffect(() => {
    setHasNewMessagesBelow(false);
    messageCountWhenScrolledUpRef.current = 0;
    setIsNearBottomState(true);
  }, [currentConversationId]);

  // Keep bottom proximity in sync when message list height changes (streaming, images, etc.)
  useEffect(() => {
    const el = messagesContainerRef.current;
    if (!el) return;
    const sync = () => {
      const near =
        el.scrollHeight - el.scrollTop - el.clientHeight < BOTTOM_SCROLL_THRESHOLD_PX;
      setIsNearBottomState(near);
    };
    const ro = new ResizeObserver(() => {
      requestAnimationFrame(sync);
    });
    ro.observe(el);
    requestAnimationFrame(sync);
    return () => ro.disconnect();
  }, [currentConversationId]);

  // ROOSEVELT'S ENHANCED AUTO-SCROLL: Only scroll on NEW messages, not updates
  useEffect(() => {
    const currentMessageCount = messages.length;
    const hasNewMessages = currentMessageCount > lastMessageCountRef.current;

    // Update the ref for next comparison
    lastMessageCountRef.current = currentMessageCount;

    const lastMsg = messages[messages.length - 1];
    const isFreshUserMessage =
      hasNewMessages && (lastMsg?.role === 'user' || lastMsg?.type === 'user');

    if (isFreshUserMessage) {
      setUserHasScrolled(false);
      setShouldAutoScroll(true);
      setHasNewMessagesBelow(false);
      messageCountWhenScrolledUpRef.current = currentMessageCount;
      requestAnimationFrame(() => {
        messagesEndRef.current?.scrollIntoView({
          behavior: 'smooth',
          block: 'end',
          inline: 'nearest',
        });
      });
      return;
    }

    // If user is scrolled up and new messages arrived below, show "New Messages" button
    if (userHasScrolled && !isNearBottom() && currentMessageCount > messageCountWhenScrolledUpRef.current) {
      setHasNewMessagesBelow(true);
    }

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
  const [folderDialogOpen, setFolderDialogOpen] = useState(false);
  const [imageToImport, setImageToImport] = useState(null);
  const [importing, setImporting] = useState(false);
  const [contextMenu, setContextMenu] = useState(null);
  const [contextMenuMessage, setContextMenuMessage] = useState(null);
  const [reactionAnchor, setReactionAnchor] = useState(null);
  const [reactionMessage, setReactionMessage] = useState(null);

  const handleCopyMessage = useCallback(async (message) => {
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
          setTimeout(() => {
            setCopiedMessageId(null);
          }, 1200);
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
  }, []);

  const handleSaveAsMarkdown = useCallback(async (message) => {
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
  }, [currentConversationId]);

  const handleImportImage = useCallback((imageUrl) => {
    setImageToImport(imageUrl);
    setFolderDialogOpen(true);
  }, []);

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

  const handleContextMenu = useCallback(
    (event, message) => {
      event.preventDefault();
      setContextMenu((prev) =>
        prev === null
          ? {
              mouseX: event.clientX + 2,
              mouseY: event.clientY - 6,
            }
          : null,
      );
      setContextMenuMessage(message);
    },
    [],
  );

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

  const formatTimestamp = useCallback((timestamp) => {
    if (!timestamp) return '';

    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }, []);

  // Extract image URLs from assistant message content for preview rendering
  const extractImageUrls = useCallback((text) => {
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
  }, []);

  // All image URLs are relative API paths — return as-is (server handles redirects for legacy schemes)
  const getImageApiUrl = useCallback((url) => url, []);

  // Open image in lightbox
  const handleOpenImage = useCallback(
    (url) => {
      const apiUrl = getImageApiUrl(url);
      // Extract filename from URL if possible
      const filename = url.split('/').pop().split('?')[0];
      openLightbox(apiUrl, { filename, alt: 'Generated image' });
    },
    [getImageApiUrl, openLightbox],
  );

  // Check if a message contains a research plan
  const hasResearchPlan = useCallback((message) => {
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
  }, []);

  // Check if a message is a HITL permission request
  const isHITLPermissionRequest = useCallback((message) => {
    if (
      message.interactionType === 'shell_command_approval' &&
      message.shellApprovalId
    ) {
      return false;
    }
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
  }, []);

  // Custom markdown components for better styling (memoized to avoid re-rendering all ReactMarkdown children)
  const markdownComponents = useMemo(() => ({
    // Style code blocks (syntax highlighting for fenced blocks)
    code: ({ node, inline, className, children, ...props }) => {
      const match = /language-([\w:]+)/.exec(className || '');

      return !inline && match ? (
        <SyntaxHighlighter
          style={theme.palette.mode === 'dark' ? materialDark : materialLight}
          language={match[1].split(':')[0]} // Handle potential colons in language name for highlighter
          PreTag="div"
          {...props}
        >
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      ) : (
        <code className={className} {...props} style={{ 
          backgroundColor:
            theme.palette.mode === 'dark'
              ? 'rgba(255, 255, 255, 0.15)'
              : 'rgba(0, 0, 0, 0.1)',
          color: theme.palette.text.primary,
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
    
    // Headings: block display; h1/h2 trimmed vs Typography h4/h5 for sidebar density (~4pt under theme size)
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
          width: '100%',
          fontSize: (theme) => {
            const fs = theme.typography.h4.fontSize;
            if (typeof fs === 'string') return `calc(${fs} - 4pt)`;
            if (typeof fs === 'number') return `calc(${fs}px - 4pt)`;
            return undefined;
          },
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
          width: '100%',
          fontSize: (theme) => {
            const fs = theme.typography.h5.fontSize;
            if (typeof fs === 'string') return `calc(${fs} - 4pt)`;
            if (typeof fs === 'number') return `calc(${fs}px - 4pt)`;
            return undefined;
          },
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
            padding: '10px 14px',
            verticalAlign: 'middle',
            lineHeight: 1.5,
          },
          '& th': {
            backgroundColor: 'action.hover',
            fontWeight: 600,
          },
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
      let imageSrc = src;
      
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
      <Box
        sx={{
          flexGrow: 1,
          minHeight: 0,
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
      <ChatInConversationSearchBar
        messages={messages}
        scrollContainerRef={messagesContainerRef}
        open={searchOpen}
        onClose={handleSearchClose}
        onActiveMatchChange={handleSearchActiveMatchChange}
      />
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
          minHeight: 0,
          overflow: 'auto',
          p: 2,
          display: 'flex',
          flexDirection: 'column',
          gap: 2
        }}
      >
        {activeLineRouting?.id && (
          <Box sx={{ flexShrink: 0 }}>
            <Chip
              size="small"
              color="secondary"
              variant="outlined"
              label={`Agent line: ${activeLineRouting.name || activeLineRouting.id.slice(0, 8)}… — type @auto to clear`}
            />
          </Box>
        )}
        {/* ROOSEVELT'S IMPROVED EMPTY STATE: Better feedback during conversation restoration */}
        {messages.length === 0 && !isLoading && !conversationLoading && !conversationMessagesLoading && (
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
        {currentConversationId && (conversationLoading || conversationMessagesLoading) && messages.length === 0 && (
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
              messageListIndex={index}
              inThreadSearchQuery={searchHighlightQuery}
              inThreadSearchActive={
                searchOpen && searchHighlightMessageIndex === index
              }
              isLoading={isLoading}
              theme={theme}
              aiName={aiName}
              markdownComponents={markdownComponents}
              handleContextMenu={handleContextMenu}
              handleImportImage={handleImportImage}
              formatTimestamp={formatTimestamp}
              handleCopyMessage={handleCopyMessage}
              handleSaveAsMarkdown={handleSaveAsMarkdown}
              isHITLPermissionRequest={isHITLPermissionRequest}
              handleHITLResponse={handleHITLResponse}
              handleShellApproval={handleShellApproval}
              hasResearchPlan={hasResearchPlan}
              executingPlans={executingPlans}
              extractImageUrls={extractImageUrls}
              getImageApiUrl={getImageApiUrl}
              openLightbox={openLightbox}
              currentConversationId={currentConversationId}
              copiedMessageId={copiedMessageId}
              savingNoteFor={savingNoteFor}
              isAdmin={isAdmin}
              has={has}
              onEditAndResend={editAndResendMessage}
              onSwitchBranch={switchBranch}
              siblingInfo={siblingInfoByMessageId[message.message_id || message.id]}
              anyMessageStreaming={anyMessageStreaming}
              setActiveArtifact={setActiveArtifact}
              openArtifact={openArtifact}
              activeArtifact={activeArtifact}
              artifactCollapsed={artifactCollapsed}
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
                {messages[messages.length - 1]?.metadata?.persona_ai_name || aiName}
              </Typography>
              {(() => {
                const lastMessage = messages[messages.length - 1];
                const displayName = lastMessage?.metadata?.agent_display_name;
                const agentType = lastMessage?.metadata?.agent_type || lastMessage?.agent_type;
                const meta = lastMessage?.metadata || {};
                const activitySub = (meta.activity_detail || '').trim();
                const detailBlock = activitySub ? (
                  <Typography
                    variant="caption"
                    sx={{
                      color: 'text.secondary',
                      fontSize: '0.6rem',
                      display: 'block',
                      mt: 0.25,
                      whiteSpace: 'normal',
                      wordBreak: 'break-word',
                    }}
                  >
                    {activitySub}
                  </Typography>
                ) : null;
                const wrapSx = { maxWidth: 'min(420px, 78vw)' };
                if (activeLineRouting?.id) {
                  return (
                    <Box sx={wrapSx}>
                      <Typography variant="caption" sx={{ color: 'secondary.main', fontSize: '0.65rem', fontWeight: 600, display: 'block' }}>
                        Agent line
                        {activeLineRouting.name ? ` (${activeLineRouting.name})` : ''} is working...
                      </Typography>
                      {detailBlock}
                    </Box>
                  );
                }
                if (displayName) {
                  return (
                    <Box sx={wrapSx}>
                      <Typography variant="caption" sx={{ color: 'secondary.main', fontSize: '0.65rem', fontWeight: 600, display: 'block' }}>
                        {displayName} is chewing on it...
                      </Typography>
                      {detailBlock}
                    </Box>
                  );
                }
                if (agentType) {
                  return (
                    <Box sx={wrapSx}>
                      <Typography variant="caption" sx={{ color: 'secondary.main', fontSize: '0.65rem', fontWeight: 600, display: 'block' }}>
                        {agentType.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')} is chewing on it...
                      </Typography>
                      {detailBlock}
                    </Box>
                  );
                }
                return (
                  <Box sx={wrapSx}>
                    <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.65rem', display: 'block' }}>
                      Chewing on it...
                    </Typography>
                    {detailBlock}
                  </Box>
                );
              })()}
            </Box>
          </Box>
        )}

        {/* Scroll anchor */}
        <div ref={messagesEndRef} />
      </Box>

      {userHasScrolled && hasNewMessagesBelow && !isNearBottomState && (
        <Box
          sx={{
            position: 'absolute',
            left: 0,
            right: 0,
            bottom: 12,
            display: 'flex',
            justifyContent: 'center',
            pointerEvents: 'none',
            zIndex: 2,
          }}
        >
          <Button
            variant="contained"
            size="small"
            onClick={() => {
              setHasNewMessagesBelow(false);
              messageCountWhenScrolledUpRef.current = messageCountRef.current;
              setUserHasScrolled(false);
              setShouldAutoScroll(true);
              setIsNearBottomState(true);
              messagesEndRef.current?.scrollIntoView({
                behavior: 'smooth',
                block: 'end'
              });
            }}
            sx={{
              pointerEvents: 'auto',
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
      </Box>

      {/* Folder Selection Dialog for Image Import */}
      <FolderSelectionDialog
        open={folderDialogOpen}
        onClose={() => {
          setFolderDialogOpen(false);
          setImageToImport(null);
        }}
        onSelect={handleFolderSelect}
      />

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
});

export default memo(ChatMessagesArea); 