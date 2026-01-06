/**
 * EML Email Viewer
 * Displays .eml email files with formatted headers and body content
 * 
 * Features:
 * - Parses email headers (From, To, Cc, Bcc, Subject, Date, etc.)
 * - Displays email body (text/plain and text/html)
 * - Handles multipart messages
 * - Theme-aware styling (light/dark mode)
 * - Download functionality
 * - Responsive layout
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  IconButton,
  Typography,
  Tooltip,
  CircularProgress,
  Alert,
  Stack,
  Divider,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableRow
} from '@mui/material';
import {
  Email,
  Download,
  Person,
  Schedule,
  Subject as SubjectIcon,
  Send
} from '@mui/icons-material';
import DOMPurify from 'dompurify';
import { useTheme } from '../contexts/ThemeContext';

const EMLViewer = ({ documentId, filename }) => {
  const [emailData, setEmailData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { darkMode } = useTheme();

  // Parse email content
  const parseEmail = useCallback((text) => {
    try {
      const lines = text.split('\n');
      const headers = {};
      let headerEndIndex = -1;
      
      // Find where headers end (blank line)
      for (let i = 0; i < lines.length; i++) {
        if (lines[i].trim() === '') {
          headerEndIndex = i;
          break;
        }
      }
      
      // Parse headers
      let currentHeader = null;
      for (let i = 0; i < (headerEndIndex >= 0 ? headerEndIndex : lines.length); i++) {
        const line = lines[i];
        
        // Check if this is a new header (starts with a word followed by colon)
        const headerMatch = line.match(/^([A-Za-z-]+):\s*(.+)$/);
        if (headerMatch) {
          currentHeader = headerMatch[1].toLowerCase();
          headers[currentHeader] = headerMatch[2].trim();
        } else if (currentHeader && line.match(/^\s+/)) {
          // Continuation line (starts with whitespace)
          headers[currentHeader] += ' ' + line.trim();
        }
      }
      
      // Parse body (everything after headers)
      const bodyStart = headerEndIndex >= 0 ? headerEndIndex + 1 : 0;
      let bodyText = lines.slice(bodyStart).join('\n');
      
      // Handle multipart messages
      const contentType = headers['content-type'] || '';
      const isMultipart = contentType.toLowerCase().includes('multipart/');
      
      let bodyContent = bodyText;
      let bodyType = 'text/plain';
      
      if (isMultipart) {
        // Extract boundary (handle quoted and unquoted)
        const boundaryMatch = contentType.match(/boundary=(?:"([^"]+)"|([^";\s]+))/i);
        if (boundaryMatch) {
          const boundary = boundaryMatch[1] || boundaryMatch[2];
          const parts = bodyText.split(`--${boundary}`);
          
          // Find text/plain and text/html parts
          let plainText = '';
          let htmlText = '';
          
          for (const part of parts) {
            if (!part.trim() || part.trim() === '--') continue;
            
            const partLines = part.split('\n');
            let partHeaders = {};
            let partHeaderEnd = -1;
            
            // Find part headers
            for (let i = 0; i < partLines.length; i++) {
              if (partLines[i].trim() === '') {
                partHeaderEnd = i;
                break;
              }
              const headerMatch = partLines[i].match(/^([A-Za-z-]+):\s*(.+)$/i);
              if (headerMatch) {
                partHeaders[headerMatch[1].toLowerCase()] = headerMatch[2].trim();
              }
            }
            
            const partContentType = partHeaders['content-type'] || '';
            const partBody = partLines.slice(partHeaderEnd + 1).join('\n').trim();
            
            if (partContentType.toLowerCase().includes('text/plain')) {
              plainText = partBody;
            } else if (partContentType.toLowerCase().includes('text/html')) {
              htmlText = partBody;
            }
          }
          
          // Prefer HTML if available, otherwise use plain text
          if (htmlText) {
            bodyContent = htmlText;
            bodyType = 'text/html';
          } else if (plainText) {
            bodyContent = plainText;
            bodyType = 'text/plain';
          }
        }
      } else {
        // Single part message
        if (contentType.toLowerCase().includes('text/html')) {
          bodyType = 'text/html';
        }
      }
      
      // Decode quoted-printable and base64 if needed
      const contentTransferEncoding = headers['content-transfer-encoding'] || '';
      if (contentTransferEncoding.toLowerCase() === 'quoted-printable') {
        bodyContent = decodeQuotedPrintable(bodyContent);
      } else if (contentTransferEncoding.toLowerCase() === 'base64') {
        try {
          bodyContent = atob(bodyContent.replace(/\s/g, ''));
        } catch (e) {
          // If base64 decode fails, use as-is
        }
      }
      
      // Decode header values (handle =?encoding?B/Q?text?= format)
      const decodeHeader = (value) => {
        if (!value) return '';
        
        // Handle encoded words like =?UTF-8?B?base64?= or =?UTF-8?Q?text?=
        const encodedWordRegex = /=\?([^?]+)\?([BQ])\?([^?]+)\?=/g;
        return value.replace(encodedWordRegex, (match, charset, encoding, text) => {
          try {
            if (encoding === 'B') {
              // Base64
              return atob(text.replace(/\s/g, ''));
            } else if (encoding === 'Q') {
              // Quoted-printable
              return text.replace(/=([0-9A-F]{2})/gi, (m, hex) => {
                return String.fromCharCode(parseInt(hex, 16));
              }).replace(/_/g, ' ');
            }
          } catch (e) {
            return match;
          }
          return match;
        });
      };
      
      // Decode all header values
      const decodedHeaders = {};
      for (const [key, value] of Object.entries(headers)) {
        decodedHeaders[key] = decodeHeader(value);
      }
      
      return {
        headers: decodedHeaders,
        body: bodyContent,
        bodyType: bodyType
      };
    } catch (err) {
      console.error('Failed to parse email:', err);
      throw new Error(`Failed to parse email: ${err.message}`);
    }
  }, []);
  
  // Decode quoted-printable encoding
  const decodeQuotedPrintable = (text) => {
    return text
      .replace(/=\r?\n/g, '') // Remove soft line breaks
      .replace(/=([0-9A-F]{2})/gi, (match, hex) => {
        return String.fromCharCode(parseInt(hex, 16));
      });
  };
  
  // Format date
  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown';
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch (e) {
      return dateString;
    }
  };
  
  // Format email addresses
  const formatEmailAddresses = (addressString) => {
    if (!addressString) return [];
    
    // Handle multiple addresses separated by commas
    const addresses = addressString.split(',').map(addr => addr.trim());
    
    return addresses.map(addr => {
      // Check if it's in "Name <email@domain.com>" format
      const match = addr.match(/^(.+?)\s*<(.+?)>$/);
      if (match) {
        return { name: match[1].trim(), email: match[2].trim() };
      }
      return { name: '', email: addr.trim() };
    });
  };

  // Fetch and parse EML file
  useEffect(() => {
    const fetchAndParseEmail = async () => {
      try {
        setLoading(true);
        setError(null);

        const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
        if (!token) {
          throw new Error('Authentication token not found');
        }

        // Fetch the EML file as text
        const response = await fetch(`/api/documents/${documentId}/file`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const text = await response.text();
        const parsed = parseEmail(text);
        setEmailData(parsed);
      } catch (err) {
        console.error('Failed to load or parse EML file:', err);
        setError(`Failed to load email: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };

    if (documentId) {
      fetchAndParseEmail();
    }
  }, [documentId, parseEmail]);

  // Download handler
  const handleDownload = useCallback(() => {
    const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
    
    if (token) {
      fetch(`/api/documents/${documentId}/file`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
        .then(response => response.blob())
        .then(blob => {
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = filename || 'email.eml';
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);
        })
        .catch(err => {
          console.error('Failed to download file:', err);
        });
    } else {
      const link = document.createElement('a');
      link.href = `/api/documents/${documentId}/file`;
      link.download = filename || 'email.eml';
      link.click();
    }
  }, [documentId, filename]);

  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: darkMode ? '#121212' : '#f5f5f5'
      }}
    >
      {/* Toolbar */}
      <Paper
        elevation={2}
        sx={{
          p: 1.5,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderRadius: 0,
          borderBottom: '1px solid',
          borderColor: 'divider'
        }}
      >
        <Stack direction="row" spacing={1} alignItems="center">
          <Email color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            {filename || 'Email Message'}
          </Typography>
        </Stack>

        <Stack direction="row" spacing={1} alignItems="center">
          <Tooltip title="Download EML">
            <IconButton onClick={handleDownload} size="small" color="primary">
              <Download />
            </IconButton>
          </Tooltip>
        </Stack>
      </Paper>

      {/* Content Viewer */}
      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          p: 3,
          backgroundColor: darkMode ? '#121212' : '#f5f5f5'
        }}
      >
        {error ? (
          <Alert severity="error" sx={{ maxWidth: 800, mt: 2 }}>
            {error}
          </Alert>
        ) : loading ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, mt: 4 }}>
            <CircularProgress size={60} />
            <Typography variant="body1" color="text.secondary">
              Loading email message...
            </Typography>
          </Box>
        ) : emailData ? (
          <Paper
            elevation={3}
            sx={{
              maxWidth: '800px',
              width: '100%',
              mx: 'auto',
              p: 3,
              backgroundColor: darkMode ? '#1e1e1e' : 'white'
            }}
          >
            {/* Subject */}
            {emailData.headers.subject && (
              <Box sx={{ mb: 3 }}>
                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                  <SubjectIcon color="primary" fontSize="small" />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    {emailData.headers.subject}
                  </Typography>
                </Stack>
              </Box>
            )}

            <Divider sx={{ my: 2 }} />

            {/* Email Headers */}
            <Table size="small" sx={{ mb: 3 }}>
              <TableBody>
                {emailData.headers.from && (
                  <TableRow>
                    <TableCell sx={{ width: '120px', fontWeight: 'bold', border: 'none', py: 1 }}>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Person fontSize="small" color="action" />
                        <Typography variant="body2">From:</Typography>
                      </Stack>
                    </TableCell>
                    <TableCell sx={{ border: 'none', py: 1 }}>
                      {formatEmailAddresses(emailData.headers.from).map((addr, idx) => (
                        <Box key={idx} sx={{ mb: 0.5 }}>
                          {addr.name ? (
                            <>
                              <Typography variant="body2" component="span" sx={{ fontWeight: 500 }}>
                                {addr.name}
                              </Typography>
                              <Typography variant="body2" component="span" sx={{ color: 'text.secondary', ml: 1 }}>
                                &lt;{addr.email}&gt;
                              </Typography>
                            </>
                          ) : (
                            <Typography variant="body2">{addr.email}</Typography>
                          )}
                        </Box>
                      ))}
                    </TableCell>
                  </TableRow>
                )}

                {emailData.headers.to && (
                  <TableRow>
                    <TableCell sx={{ width: '120px', fontWeight: 'bold', border: 'none', py: 1 }}>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Send fontSize="small" color="action" />
                        <Typography variant="body2">To:</Typography>
                      </Stack>
                    </TableCell>
                    <TableCell sx={{ border: 'none', py: 1 }}>
                      {formatEmailAddresses(emailData.headers.to).map((addr, idx) => (
                        <Chip
                          key={idx}
                          label={addr.name || addr.email}
                          size="small"
                          sx={{ mr: 0.5, mb: 0.5 }}
                        />
                      ))}
                    </TableCell>
                  </TableRow>
                )}

                {emailData.headers.cc && (
                  <TableRow>
                    <TableCell sx={{ width: '120px', fontWeight: 'bold', border: 'none', py: 1 }}>
                      <Typography variant="body2">Cc:</Typography>
                    </TableCell>
                    <TableCell sx={{ border: 'none', py: 1 }}>
                      {formatEmailAddresses(emailData.headers.cc).map((addr, idx) => (
                        <Chip
                          key={idx}
                          label={addr.name || addr.email}
                          size="small"
                          variant="outlined"
                          sx={{ mr: 0.5, mb: 0.5 }}
                        />
                      ))}
                    </TableCell>
                  </TableRow>
                )}

                {emailData.headers.bcc && (
                  <TableRow>
                    <TableCell sx={{ width: '120px', fontWeight: 'bold', border: 'none', py: 1 }}>
                      <Typography variant="body2">Bcc:</Typography>
                    </TableCell>
                    <TableCell sx={{ border: 'none', py: 1 }}>
                      {formatEmailAddresses(emailData.headers.bcc).map((addr, idx) => (
                        <Chip
                          key={idx}
                          label={addr.name || addr.email}
                          size="small"
                          variant="outlined"
                          sx={{ mr: 0.5, mb: 0.5 }}
                        />
                      ))}
                    </TableCell>
                  </TableRow>
                )}

                {emailData.headers.date && (
                  <TableRow>
                    <TableCell sx={{ width: '120px', fontWeight: 'bold', border: 'none', py: 1 }}>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Schedule fontSize="small" color="action" />
                        <Typography variant="body2">Date:</Typography>
                      </Stack>
                    </TableCell>
                    <TableCell sx={{ border: 'none', py: 1 }}>
                      <Typography variant="body2">{formatDate(emailData.headers.date)}</Typography>
                    </TableCell>
                  </TableRow>
                )}

                {/* Additional headers */}
                {Object.entries(emailData.headers).map(([key, value]) => {
                  const standardHeaders = ['from', 'to', 'cc', 'bcc', 'subject', 'date', 'content-type', 'content-transfer-encoding'];
                  if (standardHeaders.includes(key.toLowerCase())) return null;
                  
                  return (
                    <TableRow key={key}>
                      <TableCell sx={{ width: '120px', fontWeight: 'bold', border: 'none', py: 1 }}>
                        <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                          {key.replace(/-/g, ' ')}:
                        </Typography>
                      </TableCell>
                      <TableCell sx={{ border: 'none', py: 1 }}>
                        <Typography variant="body2" sx={{ wordBreak: 'break-word' }}>
                          {value}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>

            <Divider sx={{ my: 2 }} />

            {/* Email Body */}
            <Box sx={{ mt: 3 }}>
              {emailData.bodyType === 'text/html' ? (
                <Box
                  sx={{
                    '& *': {
                      maxWidth: '100%'
                    },
                    '& img': {
                      maxWidth: '100%',
                      height: 'auto'
                    },
                    '& a': {
                      color: darkMode ? '#90caf9' : 'primary.main',
                      textDecoration: 'none',
                      '&:hover': {
                        textDecoration: 'underline'
                      }
                    }
                  }}
                  dangerouslySetInnerHTML={{
                    __html: DOMPurify.sanitize(emailData.body, {
                      ALLOWED_TAGS: [
                        'p', 'br', 'div', 'span', 'strong', 'em', 'b', 'i', 'u',
                        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                        'ul', 'ol', 'li',
                        'a', 'img', 'table', 'tr', 'td', 'th', 'thead', 'tbody',
                        'blockquote', 'pre', 'code'
                      ],
                      ALLOWED_ATTR: ['href', 'src', 'alt', 'title', 'class', 'style']
                    })
                  }}
                />
              ) : (
                <Typography
                  variant="body1"
                  component="pre"
                  sx={{
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    fontFamily: 'inherit',
                    lineHeight: 1.6,
                    color: darkMode ? '#e0e0e0' : 'text.primary'
                  }}
                >
                  {emailData.body}
                </Typography>
              )}
            </Box>
          </Paper>
        ) : null}
      </Box>
    </Box>
  );
};

export default EMLViewer;
