import React from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  Link,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  ExpandMore,
  Description,
} from '@mui/icons-material';

/**
 * Format timestamp for display
 */
export const formatTimestamp = (timestamp) => {
  const date = new Date(timestamp);
  return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
};

/**
 * Render citations component - Roosevelt's Numbered Citation System
 */
export const renderCitations = (citations) => {
  if (!citations || citations.length === 0) return null;

  // Check if these are numbered citations (new format) or legacy format
  const isNumberedFormat = citations.some(citation => citation.id !== undefined);

  if (isNumberedFormat) {
    // New numbered citation format
    return (
      <Box mt={2}>
        <Typography variant="h6" sx={{ mb: 1, fontWeight: 'bold', color: 'text.primary' }}>
          Citations
        </Typography>
        <List dense>
          {citations
            .sort((a, b) => (a.id || 0) - (b.id || 0)) // Sort by citation number
            .map((citation) => (
            <ListItem key={citation.id || citation.title} sx={{ py: 0.5, px: 0, alignItems: 'flex-start' }}>
              <Typography variant="body2" sx={{ mr: 1, fontWeight: 'bold', mt: 0.25 }}>
                ({citation.id}).
              </Typography>
              <ListItemText
                primary={
                  <Box>
                    {citation.url ? (
                      <Link
                        href={citation.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        sx={{ textDecoration: 'none', fontWeight: 'medium' }}
                      >
                        {citation.title}
                      </Link>
                    ) : (
                      <Typography variant="body2" sx={{ fontWeight: 'medium', display: 'inline' }}>
                        {citation.title}
                      </Typography>
                    )}
                    {citation.author && (
                      <Typography variant="body2" sx={{ ml: 1, fontStyle: 'italic', display: 'inline' }}>
                        by {citation.author}
                      </Typography>
                    )}
                    {citation.date && (
                      <Typography variant="body2" sx={{ ml: 1, color: 'text.secondary', display: 'inline' }}>
                        ({citation.date})
                      </Typography>
                    )}
                    <Chip 
                      label={citation.type || 'document'} 
                      size="small" 
                      sx={{ ml: 1, height: '20px', fontSize: '0.75rem' }}
                    />
                  </Box>
                }
                secondary={citation.excerpt ? `"${citation.excerpt.substring(0, 150)}..."` : null}
              />
            </ListItem>
          ))}
        </List>
      </Box>
    );
  } else {
    // Legacy citation format (backward compatibility)
    return (
      <Box mt={2}>
        <Accordion 
          elevation={0}
          sx={{ 
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: 1,
            '&:before': {
              display: 'none',
            },
          }}
        >
          <AccordionSummary
            expandIcon={<ExpandMore />}
            sx={{
              px: 2,
              py: 1,
              minHeight: 'auto',
              '& .MuiAccordionSummary-content': {
                margin: '8px 0',
              },
            }}
          >
            <Typography variant="subtitle2" sx={{ fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: 1 }}>
              <Description fontSize="small" />
              Sources ({citations.length})
            </Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ px: 2, py: 0 }}>
            <List dense>
              {citations.map((citation, index) => (
                <ListItem key={index} sx={{ py: 0, px: 0 }}>
                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center" gap={1}>
                        <Description fontSize="small" />
                        <Link
                          href={citation.segment_id && citation.segment_id.startsWith('http') ? citation.segment_id : "#"}
                          onClick={(e) => {
                            if (!citation.segment_id || !citation.segment_id.startsWith('http')) {
                              e.preventDefault();
                              // Could implement document viewing here for non-web sources
                            }
                          }}
                          target={citation.segment_id && citation.segment_id.startsWith('http') ? "_blank" : undefined}
                          rel={citation.segment_id && citation.segment_id.startsWith('http') ? "noopener noreferrer" : undefined}
                          sx={{ textDecoration: 'none' }}
                        >
                          {citation.document_title || citation.title || citation.filename || 'Document'}
                        </Link>
                        {citation.page && (
                          <Chip label={`Page ${citation.page}`} size="small" />
                        )}
                      </Box>
                    }
                    secondary={citation.content ? `"${citation.content.substring(0, 100)}..."` : null}
                  />
                </ListItem>
              ))}
            </List>
          </AccordionDetails>
        </Accordion>
      </Box>
    );
  }
};

/**
 * Convert markdown to plain text for office applications
 */
export const markdownToPlainText = (markdown) => {
  if (!markdown) return '';
  
  // Fast path for massive payloads: avoid heavy regex on hundreds of KB
  if (markdown.length > 120000) {
    return String(markdown).replace(/\r\n/g, '\n');
  }
  
  let plainText = markdown;
  
  // Remove markdown formatting while preserving structure
  plainText = plainText
    // Headers - convert to plain text with line breaks
    .replace(/^#{1,6}\s+(.+)$/gm, '$1\n')
    // Bold and italic - remove formatting but keep text
    .replace(/\*\*\*(.+?)\*\*\*/g, '$1')  // Bold italic
    .replace(/\*\*(.+?)\*\*/g, '$1')      // Bold
    .replace(/\*(.+?)\*/g, '$1')          // Italic
    .replace(/__(.+?)__/g, '$1')          // Bold (underscore)
    .replace(/_(.+?)_/g, '$1')            // Italic (underscore)
    // Strikethrough - remove formatting but keep text
    .replace(/~~(.+?)~~/g, '$1')          // Strikethrough
    // Code blocks - preserve content but remove backticks and language specifiers
    .replace(/```[\w]*\n([\s\S]*?)```/g, '$1')  // Code blocks with language
    .replace(/```([\s\S]*?)```/g, '$1')         // Code blocks without language
    .replace(/`(.+?)`/g, '$1')            // Inline code
    // Links - show both text and URL
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1 ($2)')
    // Task lists - convert to simple format
    .replace(/^\s*- \[[xX]\]\s+/gm, '☑ ')  // Completed tasks
    .replace(/^\s*- \[ \]\s+/gm, '☐ ')     // Incomplete tasks
    // Lists - convert to simple format
    .replace(/^\s*[-*+]\s+/gm, '• ')      // Unordered lists
    .replace(/^\s*\d+\.\s+/gm, '• ')      // Ordered lists (convert to bullets)
    // Blockquotes - remove > but keep indentation
    .replace(/^\s*>\s*/gm, '  ')
    // Horizontal rules
    .replace(/^[-*_]{3,}$/gm, '---')
    // Tables - convert to simple format
    .replace(/\|/g, ' | ')
    // Remove table separators
    .replace(/^\s*\|[\s\-|:]+\|\s*$/gm, '')
    // Remove extra whitespace but preserve paragraph breaks
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]+/g, ' ')
    .trim();
  
  return plainText;
};

/**
 * Copy text to clipboard with fallback
 */
export const copyToClipboard = async (text) => {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (error) {
    console.error('Failed to copy with clipboard API, trying fallback:', error);
    // Fallback for browsers that don't support clipboard API
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    try {
      const success = document.execCommand('copy');
      document.body.removeChild(textArea);
      return success;
    } catch (fallbackError) {
      console.error('Fallback copy failed:', fallbackError);
      document.body.removeChild(textArea);
      return false;
    }
  }
};

/**
 * Copy text as plain text (for office applications like Word, Teams)
 */
export const copyAsPlainText = async (text) => {
  const plainText = markdownToPlainText(text);
  return await copyToClipboard(plainText);
};

/**
 * Copy with both markdown and plain text options
 */
export const copyWithOptions = async (text, asPlainText = false) => {
  if (asPlainText) {
    return await copyAsPlainText(text);
  } else {
    return await copyToClipboard(text);
  }
};

/**
 * Copy as rich text (HTML) for Word and other rich editors
 */
export const copyAsRichText = async (text) => {
  const { copyAsRichText: copyRichText } = await import('./htmlCopyUtils');
  return await copyRichText(text);
};

/**
 * Smart copy that automatically chooses the best format
 */
export const smartCopy = async (text) => {
  try {
    // Try rich text copy first (HTML + plain text)
    const success = await copyAsRichText(text);
    if (success) {
      return true;
    }
    
    // Fallback to plain text copy
    return await copyAsPlainText(text);
  } catch (error) {
    console.error('Smart copy failed, falling back to plain text:', error);
    return await copyAsPlainText(text);
  }
};
