import React, { useRef, useEffect, useState, useCallback } from 'react';
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
} from '@mui/material';
import {
  Person,
  Error,
  AutoAwesome,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import ExportButton from './ExportButton';
import AsyncTaskProgress from './AsyncTaskProgress';
import { useChatSidebar } from '../../contexts/ChatSidebarContext';
import apiService from '../../services/apiService';
import ReactMarkdown from 'react-markdown';
import remarkBreaks from 'remark-breaks';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { materialLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { markdownToPlainText, renderCitations, smartCopy } from '../../utils/chatUtils';
import { useCapabilities } from '../../contexts/CapabilitiesContext';
import EditorOpsPreviewModal from './EditorOpsPreviewModal';

const ChatMessagesArea = () => {
  const theme = useTheme();
  const { 
    messages, 
    setMessages,
    isLoading, 
    currentConversationId, 
    executingPlans,

    cancelAsyncTask,
    sendMessage,
    backgroundJobService
  } = useChatSidebar();
  const { isAdmin, has } = useCapabilities();

  // ROOSEVELT'S CONVERSATION LOADING STATE: Show proper feedback during restoration
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

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getMessageIcon = (role) => {
    switch (role) {
      case 'user':
        return <Person fontSize="small" />;
      case 'assistant':
        return (
          <Box
            component="img"
            src="/images/favicon.ico"
            alt="Bastion"
            sx={{ 
              width: 20, 
              height: 20,
              objectFit: 'contain'
            }}
          />
        );
      case 'system':
        return <Error fontSize="small" />;
      default:
        return <Person fontSize="small" />;
    }
  };

  const getMessageColor = (role, isError) => {
    if (isError) return 'error.main';
    
    switch (role) {
      case 'user':
        return 'primary.main';
      case 'assistant':
        return 'secondary.main';
      case 'system':
        return 'error.main';
      default:
        return 'text.primary';
    }
  };

  // Extract image URLs from assistant message content for preview rendering
  const extractImageUrls = (text) => {
    try {
      if (!text || typeof text !== 'string') return [];
      const urls = [];
      const regex = /(https?:\/\/[^\s)]+|\/static\/images\/[^\s)]+)/g;
      let match;
      while ((match = regex.exec(text)) !== null) {
        const url = match[1];
        if (typeof url === 'string') {
          const lower = url.toLowerCase();
          if (lower.endsWith('.png') || lower.endsWith('.jpg') || lower.endsWith('.jpeg') || lower.endsWith('.webp')) {
            urls.push(url);
          }
        }
      }
      return Array.from(new Set(urls));
    } catch {
      return [];
    }
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

  // Check if a message is a research plan that needs approval
  const isResearchPlanPending = (message) => {
    return (
      hasResearchPlan(message) &&
      !message.planApproved &&
      !message.isExecuting &&
      !executingPlans.has(message.jobId || message.metadata?.job_id)
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

  // Custom markdown components for better styling
  const markdownComponents = {
    // Style code blocks - ROOSEVELT'S ENHANCED CODE BLOCK HANDLING
    code: ({ node, inline, className, children, ...props }) => {
      const match = /language-(\w+)/.exec(className || '');
      
      return !inline && match ? (
        <SyntaxHighlighter
          style={materialLight}
          language={match[1]}
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
  };

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

        {messages.map((message, index) => (
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
                  <Box sx={{ color: getMessageColor(message.role, message.isError) }}>
                    {getMessageIcon(message.role)}
                  </Box>
                )}
                
                <Typography 
                  variant="caption" 
                  color="text.secondary"
                  sx={{ fontSize: '0.7rem' }}
                >
                  {message.role === 'user' ? 'You' : 
                   message.role === 'assistant' ? aiName : 'System'}
                </Typography>
                
                {message.role === 'user' && (
                  <Box sx={{ color: getMessageColor(message.role, message.isError) }}>
                    {getMessageIcon(message.role)}
                  </Box>
                )}
              </Box>

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
                      // Format agent type to display name
                      const formatAgentName = (agentType) => {
                        if (!agentType) return 'AI';
                        return agentType
                          .split('_')
                          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                          .join(' ');
                      };
                      
                      // If content is empty/undefined and we have agent_type, show agent name
                      const displayContent = (!message.content || message.content.trim() === '') && message.metadata?.agent_type
                        ? formatAgentName(message.metadata.agent_type)
                        : (message.content || '');
                      
                      return (
                        <ReactMarkdown 
                          className="markdown-content"
                          components={markdownComponents}
                          remarkPlugins={[remarkBreaks, remarkGfm]}
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

                {/* Image Previews for generated images */}
                {message.role === 'assistant' && (() => {
                  const imageUrls = extractImageUrls(message.content);
                  return imageUrls.length > 0 ? (
                    <Box mt={1.5} display="flex" flexDirection="column" gap={1.5}>
                      {imageUrls.map((url, idx) => (
                        <Paper key={`${url}-${idx}`} variant="outlined" sx={{ p: 1.5 }}>
                          <Box
                            component="img"
                            src={url}
                            alt="Generated image"
                            sx={{
                              maxWidth: '100%',
                              height: 'auto',
                              borderRadius: 1,
                              display: 'block'
                            }}
                          />
                          <Box mt={1} display="flex" gap={1}>
                            <Button
                              component="a"
                              href={url}
                              target="_blank"
                              rel="noopener noreferrer"
                              size="small"
                              variant="contained"
                            >
                              Open
                            </Button>
                            <Button
                              component="a"
                              href={url}
                              download
                              size="small"
                              variant="outlined"
                            >
                              Download
                            </Button>
                          </Box>
                        </Paper>
                      ))}
                    </Box>
                  ) : null;
                })()}
              </Box>

              {/* Async Task Progress */}
              <AsyncTaskProgress 
                message={message} 
                onCancel={cancelAsyncTask}
              />

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
                      startIcon={isLoading ? <CircularProgress size={16} color="inherit" /> : null}
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
                      startIcon={isLoading ? <CircularProgress size={16} color="inherit" /> : null}
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
                            startIcon={<CircularProgress size={16} />}
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

              {/* Message Footer */}
              <Box sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'space-between',
                gap: 1,
                userSelect: 'none'
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
        ))}

        {/* Loading indicator */}
        {isLoading && (
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            py: 2
          }}>
            <CircularProgress size={20} />
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
    </Box>
  );
};

export default ChatMessagesArea; 