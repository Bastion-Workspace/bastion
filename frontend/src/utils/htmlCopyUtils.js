import ReactMarkdown from 'react-markdown';
import DOMPurify from 'dompurify';
import { renderToString } from 'react-dom/server';

/**
 * Convert markdown to clean HTML for rich text editors
 */
export const markdownToHtml = (markdown) => {
  if (!markdown) return '';
  
  try {
    // Convert markdown to HTML using ReactMarkdown
    const htmlString = renderToString(
      <ReactMarkdown
        components={{
          // Customize code blocks for better Word compatibility
          code: ({ node, inline, className, children, ...props }) => {
            const match = /language-(\w+)/.exec(className || '');
            return !inline ? (
              <pre style={{
                backgroundColor: '#f6f8fa',
                padding: '16px',
                borderRadius: '6px',
                overflow: 'auto',
                fontSize: '14px',
                lineHeight: '1.45',
                fontFamily: 'Consolas, Monaco, "Andale Mono", monospace'
              }}>
                <code {...props}>{children}</code>
              </pre>
            ) : (
              <code style={{
                backgroundColor: '#f6f8fa',
                padding: '2px 4px',
                borderRadius: '3px',
                fontSize: '85%',
                fontFamily: 'Consolas, Monaco, "Andale Mono", monospace'
              }} {...props}>
                {children}
              </code>
            );
          },
          // Customize blockquotes for better formatting
          blockquote: ({ children, ...props }) => (
            <blockquote style={{
              borderLeft: '4px solid #dfe2e5',
              paddingLeft: '16px',
              margin: '16px 0',
              color: '#6a737d'
            }} {...props}>
              {children}
            </blockquote>
          ),
          // Customize tables for better Word compatibility
          table: ({ children, ...props }) => (
            <table style={{
              borderCollapse: 'collapse',
              width: '100%',
              margin: '16px 0'
            }} {...props}>
              {children}
            </table>
          ),
          th: ({ children, ...props }) => (
            <th style={{
              border: '1px solid #dfe2e5',
              padding: '8px 12px',
              backgroundColor: '#f6f8fa',
              fontWeight: 'bold',
              textAlign: 'left'
            }} {...props}>
              {children}
            </th>
          ),
          td: ({ children, ...props }) => (
            <td style={{
              border: '1px solid #dfe2e5',
              padding: '8px 12px'
            }} {...props}>
              {children}
            </td>
          ),
          // Customize lists for better formatting
          ul: ({ children, ...props }) => (
            <ul style={{
              paddingLeft: '24px',
              margin: '16px 0'
            }} {...props}>
              {children}
            </ul>
          ),
          ol: ({ children, ...props }) => (
            <ol style={{
              paddingLeft: '24px',
              margin: '16px 0'
            }} {...props}>
              {children}
            </ol>
          ),
          // Customize headers
          h1: ({ children, ...props }) => (
            <h1 style={{
              fontSize: '2em',
              fontWeight: 'bold',
              margin: '24px 0 16px 0',
              borderBottom: '1px solid #eaecef',
              paddingBottom: '8px'
            }} {...props}>
              {children}
            </h1>
          ),
          h2: ({ children, ...props }) => (
            <h2 style={{
              fontSize: '1.5em',
              fontWeight: 'bold',
              margin: '20px 0 12px 0',
              borderBottom: '1px solid #eaecef',
              paddingBottom: '6px'
            }} {...props}>
              {children}
            </h2>
          ),
          h3: ({ children, ...props }) => (
            <h3 style={{
              fontSize: '1.25em',
              fontWeight: 'bold',
              margin: '16px 0 8px 0'
            }} {...props}>
              {children}
            </h3>
          ),
          // Customize paragraphs
          p: ({ children, ...props }) => (
            <p style={{
              margin: '16px 0',
              lineHeight: '1.6'
            }} {...props}>
              {children}
            </p>
          ),
          // Customize links
          a: ({ children, href, ...props }) => (
            <a href={href} style={{
              color: '#0366d6',
              textDecoration: 'none'
            }} {...props}>
              {children}
            </a>
          ),
          // Customize emphasis - use <b> tag for better Word compatibility
          strong: ({ children, ...props }) => (
            <b {...props}>
              {children}
            </b>
          ),
          em: ({ children, ...props }) => (
            <i {...props}>
              {children}
            </i>
          )
        }}
      >
        {markdown}
      </ReactMarkdown>
    );
    
    // Sanitize the HTML to prevent XSS
    const sanitizedHtml = DOMPurify.sanitize(htmlString, {
      ALLOWED_TAGS: [
        'p', 'br', 'strong', 'b', 'em', 'i', 'u', 's', 'code', 'pre',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li',
        'blockquote',
        'table', 'thead', 'tbody', 'tr', 'th', 'td',
        'a'
      ],
      ALLOWED_ATTR: ['href', 'style'],
      ALLOW_DATA_ATTR: false,
      KEEP_CONTENT: true,
      RETURN_DOM: false,
      RETURN_DOM_FRAGMENT: false,
      RETURN_TRUSTED_TYPE: false
    });
    
    // Wrap in a container with Word-friendly styling
    const wordCompatibleHtml = `
      <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 14px; line-height: 1.6; color: #333;">
        ${sanitizedHtml}
      </div>
    `;
    
    // Debug logging to see what HTML is being generated
    console.log('Generated HTML for rich text copy:', wordCompatibleHtml);
    
    return wordCompatibleHtml;
  } catch (error) {
    console.error('Error converting markdown to HTML:', error);
    // Fallback to plain text if conversion fails
    return markdown;
  }
};

/**
 * Copy content as both HTML and plain text for maximum compatibility
 * Plain text preserves raw markdown syntax for markdown editors
 */
export const copyAsRichText = async (markdown) => {
  try {
    // Keep raw markdown for plain text (for markdown editors)
    const plainText = markdown;
    const htmlContent = markdownToHtml(markdown);
    
    // Use modern Clipboard API with multiple formats
    if (navigator.clipboard && navigator.clipboard.write) {
      const clipboardItem = new ClipboardItem({
        'text/plain': new Blob([plainText], { type: 'text/plain' }),
        'text/html': new Blob([htmlContent], { type: 'text/html' })
      });
      
      await navigator.clipboard.write([clipboardItem]);
      return true;
    } else {
      // Fallback for older browsers - copy as HTML
      const textArea = document.createElement('textarea');
      textArea.value = htmlContent;
      textArea.style.position = 'fixed';
      textArea.style.left = '-999999px';
      textArea.style.top = '-999999px';
      document.body.appendChild(textArea);
      textArea.focus();
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
  } catch (error) {
    console.error('Error copying as rich text:', error);
    return false;
  }
};

/**
 * Convert markdown to plain text (helper function)
 */
const markdownToPlainText = (markdown) => {
  if (!markdown) return '';
  
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
