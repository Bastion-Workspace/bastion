import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';
import html2pdf from 'html2pdf.js';
import { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType } from 'docx';
import { markdownToPlainText } from '../utils/chatUtils';
import { parseFrontmatter } from '../utils/frontmatterUtils';

class ExportService {
  // PDF dimensions and layout constants (US Letter)
  static PAGE_WIDTH_MM = 215.9;   // US Letter width in mm
  static PAGE_HEIGHT_MM = 279.4;  // US Letter height in mm
  static PAGE_WIDTH_PX = 816;     // US Letter width at 96 DPI
  static PAGE_HEIGHT_PX = 1056;   // US Letter height at 96 DPI
  static CONTENT_WIDTH_PX = 656;  // 175mm printable area (816 - 2*20mm margins converted to pixels)
  static MARGIN_MM = 20;          // 20mm margins

  /**
   * Export message content as PDF with enhanced HTML-based rendering
   */
  exportAsPDF = async (message) => {
    try {
      // Try enhanced HTML-based rendering first
      console.log('🖨️ PDF Export: Starting enhanced export...');
      try {
        const result = await this._exportAsPDFEnhanced(message);
        console.log('✅ PDF Export: Enhanced export successful');
        return result;
      } catch (enhancedError) {
        console.warn('❌ PDF Export: Enhanced export failed, falling back to basic method:', enhancedError);
        console.warn('Detailed error:', enhancedError.stack);
        return await this._exportAsPDFBasic(message);
      }
    } catch (error) {
      console.error('💥 PDF Export: Complete failure:', error);
      throw new Error('Failed to export PDF. Please try again.');
    }
  };

  /**
   * Enhanced PDF export with selectable text using markdown parsing
   */
  _exportAsPDFEnhanced = async (message) => {
    console.log('🔧 Enhanced PDF: Using text-based PDF generation for selectable content...');

    try {
      return await this._exportAsPDFTextBased(message);
    } catch (error) {
      console.warn('❌ Text-based PDF failed, falling back to html2pdf.js:', error);
      try {
        return await this._exportAsPDFHtml2Pdf(message);
      } catch (html2pdfError) {
        console.warn('❌ html2pdf.js also failed, falling back to canvas method:', html2pdfError);
        return await this._exportAsPDFCanvasFallback(message);
      }
    }
  };

  /**
   * Text-based PDF generation using jsPDF text methods for selectable content
   */
  _exportAsPDFTextBased = async (message) => {
    console.log('📝 Text PDF: Creating selectable text PDF...');

    // Create PDF with US Letter dimensions
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: [ExportService.PAGE_WIDTH_MM, ExportService.PAGE_HEIGHT_MM]
    });

    // Set up fonts and margins
    const margin = 20; // 20mm margin
    const pageWidth = ExportService.PAGE_WIDTH_MM - (2 * margin);
    const pageHeight = ExportService.PAGE_HEIGHT_MM - (2 * margin);
    let yPosition = margin;

    // Add title
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(18);
    const title = `Chat Message - ${new Date(message.timestamp).toLocaleString()}`;
    const titleLines = pdf.splitTextToSize(title, pageWidth);
    pdf.text(titleLines, margin, yPosition);
        yPosition += titleLines.length * 7 + 3; // Line height + spacing

    // Add underline for title
    pdf.setLineWidth(0.5);
    pdf.line(margin, yPosition, margin + pageWidth, yPosition);
    yPosition += 6;

    // Process markdown content
    const textElements = this._parseMarkdownForTextPDF(message.content);

    // Add content with proper formatting
    pdf.setFont('helvetica', 'normal');
    pdf.setFontSize(12);

    for (const element of textElements) {
      // Check if we need a new page
      if (yPosition > ExportService.PAGE_HEIGHT_MM - margin - 20) {
        pdf.addPage();
        yPosition = margin;
      }

      // Handle different element types
      if (element.type === 'spacer') {
        yPosition += 8; // Add some space
        continue;
      }

      if (element.type === 'heading') {
        pdf.setFont('helvetica', 'bold');
        const fontSize = element.level === 1 ? 16 : element.level === 2 ? 14 : 12;
        pdf.setFontSize(fontSize);

        // Render heading with inline formatting (minimal space after; user can add blank line in markdown)
        yPosition = this._renderTextWithFormatting(pdf, element.text, margin, yPosition, pageWidth, 'heading');
        yPosition += 2;
      }

      else if (element.type === 'codeblock') {
        // Draw code block background
        const codeLines = pdf.splitTextToSize(element.text, pageWidth - 10);
        const codeHeight = codeLines.length * 5;

        pdf.setFillColor(240, 248, 250); // Light gray background
        pdf.rect(margin - 2, yPosition - 3, pageWidth + 4, codeHeight + 6, 'F');

        pdf.setFont('courier', 'normal');
        pdf.setFontSize(10);
        pdf.text(codeLines, margin + 3, yPosition + 2);

        yPosition += codeHeight + 6;
      }

      else if (element.type === 'list') {
        const indent = element.indent * 10; // 10mm per indent level
        const bulletText = element.isOrdered ? element.marker : '•';

        pdf.setFont('helvetica', 'normal');
        pdf.setFontSize(12);

        // Draw bullet/number
        pdf.text(bulletText, margin + indent, yPosition);

        // Render list item text with inline formatting
        const listX = margin + indent + 8; // Space for bullet
        yPosition = this._renderTextWithFormatting(pdf, element.text, listX, yPosition, pageWidth - indent - 8, 'text');
      }

      else if (element.type === 'blockquote') {
        // Draw left border for blockquote
        pdf.setFillColor(200, 200, 200);
        pdf.rect(margin, yPosition - 2, 2, 12, 'F');

        pdf.setFont('helvetica', 'italic');
        pdf.setFontSize(12);

        // Render blockquote text with inline formatting
        yPosition = this._renderTextWithFormatting(pdf, element.text, margin + 8, yPosition, pageWidth - 8, 'text');
        yPosition += 2; // Extra space after blockquotes
      }

      else if (element.type === 'hr') {
        // Draw horizontal line
        pdf.setLineWidth(0.5);
        pdf.line(margin, yPosition, margin + pageWidth, yPosition);
        yPosition += 6;
      }

      else if (element.type === 'table-row') {
        // Simple table row rendering
        const cellWidth = pageWidth / element.cells.length;
        let cellX = margin;

        element.cells.forEach((cell, index) => {
          const cellText = this._flattenFormattedText(cell);
          const cellLines = pdf.splitTextToSize(cellText, cellWidth - 4);
          pdf.text(cellLines, cellX + 2, yPosition);

          // Draw cell border
          pdf.setLineWidth(0.2);
          pdf.rect(cellX, yPosition - 3, cellWidth, cellLines.length * 5 + 6);

          cellX += cellWidth;
        });

        yPosition += (Math.max(...element.cells.map(cell =>
          pdf.splitTextToSize(this._flattenFormattedText(cell), cellWidth - 4).length
        )) * 5) + 4;
      }

      else {
        // Regular paragraph with inline formatting
        yPosition = this._renderTextWithFormatting(pdf, element.text, margin, yPosition, pageWidth, 'text');
      }
    }

    // Add metadata if available
    if (message.queryTime || message.iterations || message.executionMode) {
      if (yPosition > ExportService.PAGE_HEIGHT_MM - margin - 25) {
        pdf.addPage();
        yPosition = margin;
      }

      yPosition += 6;
      pdf.setLineWidth(0.2);
      pdf.line(margin, yPosition, margin + pageWidth, yPosition);
      yPosition += 5;

      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(10);
      pdf.setTextColor(100, 100, 100);

      const metadata = [];
      if (message.queryTime) metadata.push(`Query Time: ${message.queryTime.toFixed(2)}s`);
      if (message.iterations) metadata.push(`Iterations: ${message.iterations}`);
      if (message.executionMode) metadata.push(`Mode: ${message.executionMode.toUpperCase()}`);

      pdf.text(metadata.join(' • '), margin, yPosition);
    }

    // Generate filename and save
    const timestamp = new Date(message.timestamp).toISOString().split('T')[0];
    const filename = `chat-message-${timestamp}.pdf`;

    console.log('💾 Text PDF: Saving selectable text PDF...');
    pdf.save(filename);

    console.log('✅ Text PDF: Export completed successfully!');
    return { success: true, filename };
  };

  /**
   * Parse markdown content into formatted text elements for PDF generation
   */
  _parseMarkdownForTextPDF = (markdown) => {
    const elements = [];
    const lines = markdown.split('\n');

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();

      if (!trimmed) {
        // Skip spacer when the previous element was a heading (no extra line after headers)
        const prev = elements[elements.length - 1];
        if (prev && prev.type === 'heading') {
          continue;
        }
        elements.push({ type: 'spacer', text: '' });
        continue;
      }

      // Headers
      const headerMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
      if (headerMatch) {
        elements.push({
          type: 'heading',
          level: headerMatch[1].length,
          text: this._parseInlineFormatting(headerMatch[2])
        });
        continue;
      }

      // Code blocks
      if (trimmed.startsWith('```')) {
        const language = trimmed.replace(/```/, '').trim();
        const codeLines = [];
        i++; // Skip the opening ```

        while (i < lines.length && !lines[i].trim().startsWith('```')) {
          codeLines.push(lines[i]);
          i++;
        }

        elements.push({
          type: 'codeblock',
          language: language,
          text: codeLines.join('\n')
        });
        continue;
      }

      // Lists
      const listMatch = trimmed.match(/^(\s*)([-*+]|\d+\.)\s+(.+)$/);
      if (listMatch) {
        const indent = Math.floor(listMatch[1].length / 2);
        const marker = listMatch[2];
        const rawText = listMatch[3];

        // Parse inline formatting in list items
        const parsedText = this._parseInlineFormatting(rawText);

        elements.push({
          type: 'list',
          indent: indent,
          marker: marker,
          text: parsedText,
          isOrdered: /^\d+\./.test(marker)
        });
        continue;
      }

      // Blockquotes
      const quoteMatch = trimmed.match(/^>\s*(.+)$/);
      if (quoteMatch) {
        elements.push({
          type: 'blockquote',
          text: this._parseInlineFormatting(quoteMatch[1])
        });
        continue;
      }

      // Horizontal rules
      if (trimmed.match(/^[-*_]{3,}$/)) {
        elements.push({
          type: 'hr',
          text: ''
        });
        continue;
      }

      // Table rows (basic support)
      if (trimmed.includes('|') && !trimmed.startsWith('|') && trimmed.split('|').length > 2) {
        const cells = trimmed.split('|').map(cell => cell.trim()).filter(cell => cell);
        elements.push({
          type: 'table-row',
          cells: cells.map(cell => this._parseInlineFormatting(cell))
        });
        continue;
      }

      // Regular paragraph with inline formatting
      elements.push({
        type: 'paragraph',
        text: this._parseInlineFormatting(trimmed)
      });
    }

    return elements;
  };

  /**
   * Parse inline markdown formatting (bold, italic, code, links)
   * Maintains proper text ordering by processing sequentially
   */
  _parseInlineFormatting = (text) => {
    // First handle emojis in the entire text
    const processedText = this._containsEmoji(text) ? this._replaceEmojis(text) : text;

    const segments = [];

    // Process inline formatting in order of specificity
    // Links first (most specific)
    const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
    let linkMatch;
    let lastIndex = 0;
    let hasLinks = false;

    while ((linkMatch = linkRegex.exec(processedText)) !== null) {
      hasLinks = true;
      
      // Add text before the link (with other inline formatting)
      if (linkMatch.index > lastIndex) {
        const beforeText = processedText.substring(lastIndex, linkMatch.index);
        if (beforeText) {
          segments.push(...this._parseSimpleFormatting(beforeText));
        }
      }

      // Add the link
      segments.push({
        text: linkMatch[1], // link text
        url: linkMatch[2],  // link URL
        type: 'link'
      });

      lastIndex = linkMatch.index + linkMatch[0].length;
    }

    // Add remaining text after last link (or entire text if no links)
    if (hasLinks && lastIndex < processedText.length) {
      const remainingText = processedText.substring(lastIndex);
      if (remainingText) {
        segments.push(...this._parseSimpleFormatting(remainingText));
      }
    } else if (!hasLinks) {
      // No links found, parse the entire text for other formatting
      segments.push(...this._parseSimpleFormatting(processedText));
    }

    return segments.length > 0 ? segments : [{ text: processedText, type: 'text' }];
  };

  /**
   * Parse simple inline formatting (bold, italic, code)
   * FIXED: Maintains proper text ordering by processing sequentially
   * Supports both single (`) and triple (```) backticks for inline code
   */
  _parseSimpleFormatting = (text) => {
    const segments = [];
    
    // Combined regex to match all inline formatting in order
    // Added support for triple backticks (```) as inline code (common non-standard usage)
    const combinedRegex = /(\*\*\*(.+?)\*\*\*|___(.+?)___|~~(.+?)~~|\*\*(.+?)\*\*|__(.+?)__|```([^`]+?)```|`([^`]+)`|\*([^*]+?)\*|_([^_]+?)_)/g;
    
    let lastIndex = 0;
    let match;
    
    while ((match = combinedRegex.exec(text)) !== null) {
      // Add any plain text before this match
      if (match.index > lastIndex) {
        const plainText = text.substring(lastIndex, match.index);
        if (plainText) {
          segments.push({ text: plainText, type: 'text' });
        }
      }
      
      // Determine the type and content of this match
      let content, type;
      
      if (match[1] && (match[2] || match[3])) {
        // Bold-italic (*** or ___)
        content = match[2] || match[3];
        type = 'bolditalic';
      } else if (match[4]) {
        // Strikethrough (~~)
        content = match[4];
        type = 'strikethrough';
      } else if (match[5] || match[6]) {
        // Bold (** or __)
        content = match[5] || match[6];
        type = 'bold';
      } else if (match[7]) {
        // Inline code with triple backticks (```)
        content = match[7];
        type = 'inline-code';
      } else if (match[8]) {
        // Inline code with single backtick (`)
        content = match[8];
        type = 'inline-code';
      } else if (match[9] || match[10]) {
        // Italic (* or _)
        content = match[9] || match[10];
        type = 'italic';
      }
      
      if (content) {
        segments.push({ text: content, type });
      }
      
      lastIndex = match.index + match[0].length;
    }
    
    // Add any remaining plain text
    if (lastIndex < text.length) {
      const remainingText = text.substring(lastIndex);
      if (remainingText) {
        segments.push({ text: remainingText, type: 'text' });
      }
    }
    
    return segments.length > 0 ? segments : [{ text, type: 'text' }];
  };

  /**
   * Check if text contains emojis
   */
  _containsEmoji = (text) => {
    const emojiRegex = /[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/u;
    return emojiRegex.test(text);
  };

  /**
   * Replace emojis with text descriptions for PDF compatibility
   */
  _replaceEmojis = (text) => {
    // Simple emoji replacements - can be expanded
    const emojiMap = {
      '😀': '[smile]',
      '😂': '[laugh]',
      '❤️': '[heart]',
      '👍': '[thumbs up]',
      '👎': '[thumbs down]',
      '✅': '[check]',
      '❌': '[cross]',
      '⚠️': '[warning]',
      'ℹ️': '[info]',
      '🔥': '[fire]',
      '💡': '[idea]',
      '⭐': '[star]',
      '🎯': '[target]',
      '🚀': '[rocket]',
      '💻': '[computer]',
      '📱': '[phone]',
      '🌟': '[sparkle]',
      '✨': '[sparkles]',
      '🎉': '[party]',
      '🎊': '[confetti]'
    };

    let result = text;
    for (const [emoji, replacement] of Object.entries(emojiMap)) {
      result = result.replace(new RegExp(emoji, 'g'), replacement);
    }

    // Replace any remaining emoji-like characters with [emoji]
    result = result.replace(/[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/gu, '[emoji]');

    return result;
  };

  /**
   * Render text with inline formatting segments
   * FIXED: Keeps inline code and formatted text together, moves to next line if needed
   */
  _renderTextWithFormatting = (pdf, segments, x, y, maxWidth, defaultType = 'text') => {
    let currentX = x;
    let currentY = y;
    const lineHeight = 5; // Base line height

    // Ensure segments is an array
    if (!Array.isArray(segments)) {
      segments = [{ text: segments, type: defaultType }];
    }

    // Process segments for emoji replacement
    const processedSegments = segments.map(segment => ({
      ...segment,
      text: this._containsEmoji(segment.text) ? this._replaceEmojis(segment.text) : segment.text
    }));

    let lastSegmentText = '';
    for (const segment of processedSegments) {
      let text = segment.text || '';
      const type = segment.type || defaultType;

      // Ensure space before inline code when previous segment didn't end with whitespace
      if (type === 'inline-code' && currentX > x && lastSegmentText.length > 0 && !/\s$/.test(lastSegmentText)) {
        text = ' ' + text;
      }
      lastSegmentText = segment.text || '';

      // Set font based on formatting type
      switch (type) {
        case 'bold':
          pdf.setFont('helvetica', 'bold');
          break;
        case 'italic':
          pdf.setFont('helvetica', 'italic');
          break;
        case 'bolditalic':
          pdf.setFont('helvetica', 'bolditalic');
          break;
        case 'inline-code':
          pdf.setFont('courier', 'normal');
          break;
        case 'link':
          pdf.setTextColor(0, 0, 255); // Blue for links
          pdf.setFont('helvetica', 'normal');
          break;
        case 'strikethrough':
          pdf.setFont('helvetica', 'normal');
          // Note: jsPDF doesn't support strikethrough directly
          break;
        default:
          pdf.setFont('helvetica', 'normal');
          pdf.setTextColor(0, 0, 0); // Black for normal text
      }

      // For inline code and links, try to keep them together on one line
      const availableWidth = maxWidth - (currentX - x);
      const textWidth = pdf.getTextWidth(text);
      
      if ((type === 'inline-code' || type === 'link') && textWidth > availableWidth && currentX > x) {
        // Move to next line to keep inline code/link together
        currentX = x;
        currentY += lineHeight;
      }

      // Now split text to fit (use full maxWidth if we just moved to new line)
      const widthToUse = (currentX === x) ? maxWidth : availableWidth;
      const lines = pdf.splitTextToSize(text, widthToUse);

      // Render the text
      for (let i = 0; i < lines.length; i++) {
        pdf.text(lines[i], currentX, currentY);
        
        if (i < lines.length - 1) {
          // Move to next line
          currentX = x;
          currentY += lineHeight;
        } else {
          // Last line - update X position
          currentX += pdf.getTextWidth(lines[i]);
          // jsPDF can under-measure trailing spaces; ensure we advance past them so next segment doesn't overwrite
          const trailingSpaces = (lines[i].match(/\s+$/) || [''])[0].length;
          if (trailingSpaces > 0) {
            currentX += pdf.getTextWidth(' ') * trailingSpaces;
          }
        }
      }

      // Reset text color
      if (type === 'link') {
        pdf.setTextColor(0, 0, 0);
      }
    }

    return currentY + lineHeight; // Return new Y position
  };

  /**
   * Flatten formatted text segments into plain text for width calculations
   */
  _flattenFormattedText = (segments) => {
    if (!Array.isArray(segments)) {
      return segments || '';
    }
    return segments.map(segment => segment.text || '').join('');
  };

  /**
   * Fallback method using html2pdf.js
   */
  _exportAsPDFHtml2Pdf = async (message) => {
    console.log('🔧 html2pdf.js Fallback: Using html2pdf for image-based PDF...');

    // Create HTML content
    const htmlContent = this._createPDFHtmlContent(message);

    // Configure html2pdf options for US Letter with proper margins
    const options = {
      margin: [20, 20, 20, 20], // [top, right, bottom, left] in mm
      filename: `chat-message-${new Date(message.timestamp).toISOString().split('T')[0]}.pdf`,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: {
        scale: 1.0,
        useCORS: true,
        allowTaint: false,
        backgroundColor: '#ffffff',
        logging: false
      },
      jsPDF: {
        unit: 'mm',
        format: [ExportService.PAGE_WIDTH_MM, ExportService.PAGE_HEIGHT_MM], // US Letter
        orientation: 'portrait'
      },
      pagebreak: { mode: ['avoid-all', 'css', 'legacy'] }
    };

    console.log('🔧 html2pdf.js: Generating PDF...');

    // Generate PDF using html2pdf.js
    await html2pdf().set(options).from(htmlContent).save();

    console.log('✅ html2pdf.js: Export completed successfully!');
    return {
      success: true,
      filename: `chat-message-${new Date(message.timestamp).toISOString().split('T')[0]}.pdf`
    };
  };

  /**
   * Create HTML content for PDF generation
   */
  _createPDFHtmlContent = (message) => {
    const htmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <style>
          body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            width: 100%;
            box-sizing: border-box;
          }
          .container {
            max-width: 100%;
            width: 100%;
          }
          h1 {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 20px;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
            page-break-after: avoid;
          }
          .content {
            margin-bottom: 30px;
            width: 100%;
          }
          .metadata {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ccc;
            font-size: 12px;
            color: #666;
            page-break-inside: avoid;
          }
          /* Ensure content flows properly */
          * {
            box-sizing: border-box;
          }
          pre {
            white-space: pre-wrap;
            word-wrap: break-word;
          }
          code {
            word-wrap: break-word;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Chat Message - ${new Date(message.timestamp).toLocaleString()}</h1>

          <div class="content">
            ${this._markdownToStyledHtml(message.content)}
          </div>

          ${(message.queryTime || message.iterations || message.executionMode) ? `
            <div class="metadata">
              ${[
                message.queryTime ? `Query Time: ${message.queryTime.toFixed(2)}s` : '',
                message.iterations ? `Iterations: ${message.iterations}` : '',
                message.executionMode ? `Mode: ${message.executionMode.toUpperCase()}` : ''
              ].filter(Boolean).join(' • ')}
            </div>
          ` : ''}
        </div>
      </body>
      </html>
    `;

    return htmlContent;
  };

  /**
   * Extract links from HTML content for PDF annotations
   */
  _extractLinksFromHtml = (htmlContent) => {
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = htmlContent;
    const links = [];
    const linkElements = tempDiv.querySelectorAll('a[href]');

    linkElements.forEach((link) => {
      const href = link.getAttribute('href');
      if (href && href.startsWith('http')) {
        // For jsPDF html() method, we need to calculate approximate positions
        // This is a simplified version - could be enhanced with more precise positioning
        links.push({
          url: href,
          x: 20, // margin
          y: 20, // approximate
          width: 100, // approximate
          height: 10, // approximate
          text: link.textContent
        });
      }
    });

    return links;
  };

  /**
   * Canvas-based fallback method for when jsPDF html() fails
   */
  _exportAsPDFCanvasFallback = async (message) => {
    console.log('🔧 Canvas Fallback: Using html2canvas method...');

    // Create temporary HTML element for rendering
    const tempDiv = document.createElement('div');
    tempDiv.style.position = 'absolute';
    tempDiv.style.left = '-9999px';
    tempDiv.style.top = '-9999px';
    tempDiv.style.width = `${ExportService.CONTENT_WIDTH_PX}px`;
    tempDiv.style.maxWidth = `${ExportService.CONTENT_WIDTH_PX}px`;
    tempDiv.style.minHeight = '100px';
    tempDiv.style.backgroundColor = 'white';
    tempDiv.style.fontFamily = '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif';
    tempDiv.style.fontSize = '14px';
    tempDiv.style.lineHeight = '1.6';
    tempDiv.style.color = '#333';
    tempDiv.style.padding = `${ExportService.MARGIN_MM}mm`;
    tempDiv.style.boxSizing = 'border-box';

    // Add title
    const titleDiv = document.createElement('h1');
    titleDiv.textContent = `Chat Message - ${new Date(message.timestamp).toLocaleString()}`;
    titleDiv.style.fontSize = '18px';
    titleDiv.style.fontWeight = 'bold';
    titleDiv.style.marginBottom = '20px';
    titleDiv.style.borderBottom = '2px solid #333';
    titleDiv.style.paddingBottom = '10px';
    tempDiv.appendChild(titleDiv);

    // Create content container
    const contentDiv = document.createElement('div');
    contentDiv.style.width = '100%';
    contentDiv.innerHTML = this._markdownToStyledHtml(message.content);
    tempDiv.appendChild(contentDiv);

    // Add metadata if available
    if (message.queryTime || message.iterations || message.executionMode) {
      const metadataDiv = document.createElement('div');
      metadataDiv.style.marginTop = '30px';
      metadataDiv.style.paddingTop = '20px';
      metadataDiv.style.borderTop = '1px solid #ccc';
      metadataDiv.style.fontSize = '12px';
      metadataDiv.style.color = '#666';

      const metadata = [];
      if (message.queryTime) metadata.push(`Query Time: ${message.queryTime.toFixed(2)}s`);
      if (message.iterations) metadata.push(`Iterations: ${message.iterations}`);
      if (message.executionMode) metadata.push(`Mode: ${message.executionMode.toUpperCase()}`);

      metadataDiv.textContent = metadata.join(' • ');
      tempDiv.appendChild(metadataDiv);
    }

    // Add to DOM temporarily
    console.log('🔧 Enhanced PDF: Adding to DOM...');
    document.body.appendChild(tempDiv);
    console.log('🔧 Enhanced PDF: DOM element added, processing links...');

    try {
      // Process links for PDF annotations
      const links = this._processLinksForPDF(tempDiv);
      console.log('🔧 Enhanced PDF: Links processed, calculating dimensions...');

      // Calculate how many pages we need
      const contentHeightPx = tempDiv.offsetHeight;
      const pagesNeeded = Math.ceil(contentHeightPx / ExportService.PAGE_HEIGHT_PX);
      const totalCanvasHeight = Math.min(contentHeightPx, ExportService.PAGE_HEIGHT_PX * pagesNeeded);
      console.log(`🔧 Enhanced PDF: Content height: ${contentHeightPx}px, Pages needed: ${pagesNeeded}, Canvas height: ${totalCanvasHeight}px`);

      // Use html2canvas to render the HTML as canvas with proper scaling
      console.log('🔧 Canvas PDF: Starting html2canvas...');

      const canvas = await html2canvas(tempDiv, {
        scale: 1.0, // Use 1:1 scaling to prevent corruption
        useCORS: true,
        allowTaint: false,
        backgroundColor: '#ffffff',
        width: ExportService.CONTENT_WIDTH_PX,
        height: totalCanvasHeight,
        scrollX: 0,
        scrollY: 0,
        ignoreElements: (element) => {
          // Skip elements that might cause issues
          return element.tagName === 'SCRIPT' || element.tagName === 'STYLE';
        },
        allowTaint: true,
        foreignObjectRendering: false,
        logging: false // Reduce console noise
      });

      console.log('✅ Enhanced PDF: html2canvas completed successfully');
      console.log(`📐 Canvas dimensions: ${canvas.width}x${canvas.height}`);

      // Remove temp element
      document.body.removeChild(tempDiv);
      console.log('🗑️ Canvas Fallback: Temp element removed, creating PDF...');

      // Create PDF from canvas with proper page handling (US Letter)
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: [ExportService.PAGE_WIDTH_MM, ExportService.PAGE_HEIGHT_MM]
      });

      // Canvas dimensions in pixels
      const canvasWidth = canvas.width;
      const canvasHeight = canvas.height;

      // PDF dimensions in mm
      const pdfPageWidth = ExportService.PAGE_WIDTH_MM;
      const pdfPageHeight = ExportService.PAGE_HEIGHT_MM;
      const pdfContentWidth = ExportService.PAGE_WIDTH_MM - (2 * ExportService.MARGIN_MM);
      const pdfContentHeight = ExportService.PAGE_HEIGHT_MM - (2 * ExportService.MARGIN_MM);

      // Split canvas into pages
      const pixelsPerPage = ExportService.PAGE_HEIGHT_PX;
      const pages = Math.ceil(canvasHeight / pixelsPerPage);

      for (let page = 0; page < pages; page++) {
        if (page > 0) {
          pdf.addPage();
        }

        // Calculate the portion of canvas for this page
        const startY = page * pixelsPerPage;
        const pageHeightPx = Math.min(pixelsPerPage, canvasHeight - startY);

        // Create a temporary canvas for this page
        const pageCanvas = document.createElement('canvas');
        const pageCtx = pageCanvas.getContext('2d');
        pageCanvas.width = canvasWidth;
        pageCanvas.height = pageHeightPx;

        // Draw the portion of the original canvas
        pageCtx.drawImage(
          canvas,
          0, startY, canvasWidth, pageHeightPx, // Source
          0, 0, canvasWidth, pageHeightPx     // Destination
        );

        // Convert to image data
        const pageImgData = pageCanvas.toDataURL('image/png');

        // Add to PDF with proper positioning (centered with margins)
        const marginLeft = 20; // 20mm left margin
        const marginTop = 20;  // 20mm top margin

        pdf.addImage(
          pageImgData,
          'PNG',
          marginLeft, marginTop, // x, y position
          pdfContentWidth, // width
          (pageHeightPx / pixelsPerPage) * pdfContentHeight // height scaled proportionally
        );
      }

      // Add clickable links to PDF
      if (links.length > 0) {
        console.log(`🔗 Enhanced PDF: Adding ${links.length} clickable links...`);
        this._addLinksToPDF(pdf, links, 2); // scaleFactor matches canvas scale
      }

      // Generate filename
      const timestamp = new Date(message.timestamp).toISOString().split('T')[0];
      const filename = `chat-message-${timestamp}.pdf`;

      console.log(`💾 Enhanced PDF: Saving PDF as ${filename}...`);
      // Save the PDF
      pdf.save(filename);

      console.log('🎉 Enhanced PDF: Export completed successfully!');
      return { success: true, filename };

    } catch (canvasError) {
      // Clean up and fall back
      document.body.removeChild(tempDiv);
      throw canvasError;
    }
  };

  /**
   * Process links and store them for PDF annotations
   */
  _processLinksForPDF = (tempDiv) => {
    const links = [];
    const linkElements = tempDiv.querySelectorAll('a[href]');

    linkElements.forEach((link, index) => {
      const rect = link.getBoundingClientRect();
      const href = link.getAttribute('href');

      if (href && href.startsWith('http')) {
        links.push({
          url: href,
          x: rect.left,
          y: rect.top,
          width: rect.width,
          height: rect.height,
          text: link.textContent
        });

        // Style the link for visual indication
        link.style.color = '#0366d6';
        link.style.textDecoration = 'underline';
      }
    });

    return links;
  };

  /**
   * Add clickable links to PDF using jsPDF annotations
   */
  _addLinksToPDF = (pdf, links, scaleFactor = 1) => {
    links.forEach(link => {
      // Convert pixel coordinates to PDF coordinates (points)
      const pdfX = (link.x / scaleFactor) * 0.75; // Convert to points (72 DPI)
      const pdfY = (link.y / scaleFactor) * 0.75;
      const pdfWidth = (link.width / scaleFactor) * 0.75;
      const pdfHeight = (link.height / scaleFactor) * 0.75;

      // Add link annotation
      pdf.link(pdfX, pdfY, pdfWidth, pdfHeight, { url: link.url });
    });
  };

  /**
   * Convert markdown to styled HTML for better PDF rendering
   */
  _markdownToStyledHtml = (markdown) => {
    if (!markdown) return '';

    // First convert to HTML using existing function
    let html = this._markdownToBasicHtml(markdown);

    // Add enhanced styling for PDF
    html = html
      // Code blocks - enhance styling
      .replace(/<pre><code>([\s\S]*?)<\/code><\/pre>/g, (match, code) => {
        return `<div style="background-color: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; margin: 16px 0; font-family: 'Courier New', monospace; font-size: 11px; line-height: 1.4; white-space: pre-wrap; word-wrap: break-word;">${code.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>`;
      })
      // Inline code - enhance styling
      .replace(/<code>(.*?)<\/code>/g, (match, code) => {
        return `<code style="background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px; font-family: 'Courier New', monospace; font-size: 11px;">${code.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code>`;
      })
      // Tables - enhance styling with better borders and spacing
      .replace(/<table([^>]*)>([\s\S]*?)<\/table>/g, (match, attrs, content) => {
        return `<table style="border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 12px; border: 1px solid #dfe2e5;">${content.replace(/<thead([^>]*)>([\s\S]*?)<\/thead>/g, '<thead style="background-color: #f6f8fa;">$2</thead>').replace(/<th([^>]*)>(.*?)<\/th>/g, '<th style="border: 1px solid #dfe2e5; padding: 10px 12px; font-weight: bold; text-align: left; background-color: #f6f8fa;">$2</th>').replace(/<td([^>]*)>(.*?)<\/td>/g, '<td style="border: 1px solid #dfe2e5; padding: 8px 12px; vertical-align: top;">$2</td>')}</table>`;
      })
      // Blockquotes - enhance styling
      .replace(/<blockquote([^>]*)>([\s\S]*?)<\/blockquote>/g, (match, attrs, content) => {
        return `<blockquote style="border-left: 4px solid #dfe2e5; padding-left: 16px; margin: 16px 0; color: #6a737d; font-style: italic;">${content}</blockquote>`;
      })
      // Headers - enhance styling
      .replace(/<h1([^>]*)>(.*?)<\/h1>/g, '<h1 style="font-size: 2em; font-weight: bold; margin: 24px 0 16px 0; border-bottom: 1px solid #eaecef; padding-bottom: 8px;">$2</h1>')
      .replace(/<h2([^>]*)>(.*?)<\/h2>/g, '<h2 style="font-size: 1.5em; font-weight: bold; margin: 20px 0 12px 0; border-bottom: 1px solid #eaecef; padding-bottom: 6px;">$2</h2>')
      .replace(/<h3([^>]*)>(.*?)<\/h3>/g, '<h3 style="font-size: 1.25em; font-weight: bold; margin: 16px 0 8px 0;">$2</h3>')
      // Lists - enhance styling
      .replace(/<ul([^>]*)>([\s\S]*?)<\/ul>/g, '<ul style="padding-left: 24px; margin: 16px 0;">$2</ul>')
      .replace(/<ol([^>]*)>([\s\S]*?)<\/ol>/g, '<ol style="padding-left: 24px; margin: 16px 0;">$2</ol>')
      .replace(/<li([^>]*)>(.*?)<\/li>/g, '<li style="margin: 4px 0;">$2</li>')
      // Links - enhance styling
      .replace(/<a([^>]*href="([^"]*)"[^>]*)>(.*?)<\/a>/g, '<a style="color: #0366d6; text-decoration: none;" href="$2">$3</a>')
      // Paragraphs - enhance styling
      .replace(/<p([^>]*)>(.*?)<\/p>/g, '<p style="margin: 16px 0; line-height: 1.6;">$2</p>');

    return html;
  };

  /**
   * Convert markdown to basic HTML for PDF rendering (no React components)
   * Enhanced to handle more markdown features
   */
  _markdownToBasicHtml = (markdown) => {
    if (!markdown) return '';

    try {
      let html = markdown;
      
      // Code blocks (must come before other replacements to avoid conflicts)
      html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
      
      // Headers (process from h6 to h1 to avoid conflicts)
      html = html.replace(/^###### (.*$)/gim, '<h6>$1</h6>');
      html = html.replace(/^##### (.*$)/gim, '<h5>$1</h5>');
      html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
      html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
      html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
      html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
      
      // Horizontal rules
      html = html.replace(/^[-*_]{3,}$/gim, '<hr>');
      
      // Blockquotes (process before paragraphs)
      html = html.replace(/^>\s+(.*$)/gim, '<blockquote>$1</blockquote>');
      
      // Tables (basic table support)
      html = html.replace(/\|(.+)\|/g, (match, content) => {
        const cells = content.split('|').map(cell => cell.trim()).filter(cell => cell);
        if (cells.length === 0) return '';
        
        // Check if it's a separator row
        if (cells.every(cell => /^[-:]+$/.test(cell))) {
          return '';
        }
        
        const isHeader = html.indexOf(match) < html.indexOf('\n|') || html.split('\n').indexOf(match.trim()) === 0;
        const tag = isHeader ? 'th' : 'td';
        return `<tr>${cells.map(cell => `<${tag}>${cell}</${tag}>`).join('')}</tr>`;
      });
      
      // Wrap table rows in table tags
      html = html.replace(/(<tr>.*<\/tr>\n?)+/g, (match) => {
        if (match.includes('<th>')) {
          return `<table><thead>${match}</thead></table>`;
        }
        return `<table><tbody>${match}</tbody></table>`;
      });
      
      // Task lists
      html = html.replace(/^[-*+]\s+\[([ xX])\]\s+(.*$)/gim, '<li style="list-style: none;"><input type="checkbox" disabled' + 
        (RegExp.$1.toLowerCase() === 'x' ? ' checked' : '') + '> $2</li>');
      
      // Unordered lists (process after task lists)
      html = html.replace(/^[-*+]\s+(.*$)/gim, '<li>$1</li>');
      
      // Ordered lists
      html = html.replace(/^\d+\.\s+(.*$)/gim, '<li>$1</li>');
      
      // Wrap consecutive list items in ul/ol tags
      html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => {
        // Check if it contains ordered list markers
        const lines = match.split('\n');
        const firstLine = lines.find(line => line.trim().startsWith('<li>'));
        if (firstLine && /^\d+\./.test(markdown.split('\n').find(line => line.includes(firstLine.replace(/<[^>]+>/g, ''))))) {
          return `<ol>${match}</ol>`;
        }
        return `<ul>${match}</ul>`;
      });
      
      // Bold and italic (process after code to avoid conflicts)
      html = html.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
      html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
      html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');
      html = html.replace(/_(.*?)_/g, '<em>$1</em>');
      
      // Strikethrough
      html = html.replace(/~~(.*?)~~/g, '<s>$1</s>');
      
      // Inline code (process after code blocks)
      html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
      
      // Links
      html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
      
      // Images
      html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1">');
      
      // Paragraphs (wrap non-block elements)
      html = html.split('\n\n').map(block => {
        block = block.trim();
        if (!block) return '';
        if (block.startsWith('<') && (block.startsWith('<h') || block.startsWith('<ul') || 
            block.startsWith('<ol') || block.startsWith('<pre') || block.startsWith('<blockquote') ||
            block.startsWith('<table') || block.startsWith('<hr'))) {
          return block;
        }
        return `<p>${block}</p>`;
      }).join('\n\n');
      
      // Line breaks within paragraphs
      html = html.replace(/\n(?!<[hulop]|<\/[hulop]|<\/?[tb]|<\/?tr|<\/?td|<\/?th|<\/?pre|<\/?blockquote|<\/?hr)/g, '<br>');

      return html;
    } catch (error) {
      console.warn('Markdown conversion failed, using plain text:', error);
      return markdown.replace(/\n/g, '<br>');
    }
  };

  /**
   * Basic PDF export method (fallback) - Original implementation
   */
  _exportAsPDFBasic = async (message) => {
    try {
      // Create a new PDF document
      const pdf = new jsPDF();
      
      // Set font and size
      pdf.setFont('helvetica');
      pdf.setFontSize(12);
      
      // Add title
      const title = `Chat Message - ${new Date(message.timestamp).toLocaleString()}`;
      pdf.setFontSize(16);
      pdf.text(title, 20, 20);
      
      // Add message content with formatting preserved
      pdf.setFontSize(12);
      const content = this._processMarkdownForPDF(message.content);
      
      // Add content starting at y=40
      let yPosition = 40;
      const lineHeight = 7;
      
      for (const line of content) {
        // Check if we need a new page
        if (yPosition > 270) {
          pdf.addPage();
          yPosition = 20;
        }
        
        // Apply formatting based on line type
        let fontStyle = 'normal';
        if (line.bold && line.italic) {
          fontStyle = 'bolditalic';
        } else if (line.bold) {
          fontStyle = 'bold';
        } else if (line.italic) {
          fontStyle = 'italic';
        }
        
        // Set font family (monospace for code)
        const fontFamily = line.monospace ? 'courier' : 'helvetica';
        pdf.setFont(fontFamily, fontStyle);
        
        // Set font size with better scaling
        const fontSize = line.fontSize || 12;
        pdf.setFontSize(fontSize);
        
        // Add background for code blocks (simulate with border)
        if (line.backgroundColor) {
          pdf.setDrawColor(200, 200, 200);
          pdf.setLineWidth(0.1);
        }
        
        // Set line spacing
        const currentLineHeight = line.lineSpacing ? lineHeight * line.lineSpacing : lineHeight;
        
        // Split text to fit page width with proper margin adjustment
        const maxWidth = Math.max(50, 170 - (line.indent || 20)); // Ensure minimum width
        const textLines = pdf.splitTextToSize(line.text, maxWidth);
        
        for (const textLine of textLines) {
          if (yPosition > 270) {
            pdf.addPage();
            yPosition = 20;
          }
          
          // Add left border for blockquotes
          if (line.borderLeft) {
            pdf.setDrawColor(100, 100, 100);
            pdf.setLineWidth(1);
            pdf.line((line.indent || 20) - 5, yPosition - 3, (line.indent || 20) - 5, yPosition + 3);
          }
          
          pdf.text(textLine, line.indent || 20, yPosition);
          yPosition += currentLineHeight;
        }
        
        // Add extra spacing for headers and code blocks
        if (line.lineSpacing && line.lineSpacing > 1) {
          yPosition += lineHeight * 0.5;
        }
        
        // Reset font for next line
        pdf.setFont('helvetica', 'normal');
        pdf.setFontSize(12);
        pdf.setDrawColor(0, 0, 0);
        pdf.setLineWidth(0.2);
      }
      
      // Add metadata if available
      if (message.queryTime || message.iterations || message.executionMode) {
        yPosition += 10;
        if (yPosition > 270) {
          pdf.addPage();
          yPosition = 20;
        }
        
        pdf.setFontSize(10);
        pdf.setTextColor(100, 100, 100);
        
        const metadata = [];
        if (message.queryTime) metadata.push(`Query Time: ${message.queryTime.toFixed(2)}s`);
        if (message.iterations) metadata.push(`Iterations: ${message.iterations}`);
        if (message.executionMode) metadata.push(`Mode: ${message.executionMode.toUpperCase()}`);
        
        pdf.text(metadata.join(' • '), 20, yPosition);
      }
      
      // Generate filename
      const timestamp = new Date(message.timestamp).toISOString().split('T')[0];
      const filename = `chat-message-${timestamp}.pdf`;
      
      // Save the PDF
      pdf.save(filename);
      
      return { success: true, filename };
      
    } catch (error) {
      console.error('Failed to export PDF:', error);
      throw new Error('Failed to export PDF. Please try again.');
    }
  };

  /**
   * Export message content as DOCX
   */
  exportAsDOCX = async (message) => {
    try {
      // Create document
      const doc = new Document({
        sections: [{
          properties: {},
          children: [
            // Title
            new Paragraph({
              text: `Chat Message - ${new Date(message.timestamp).toLocaleString()}`,
              heading: HeadingLevel.HEADING_1,
              alignment: AlignmentType.CENTER,
            }),
            
            // Spacing
            new Paragraph({
              children: [new TextRun({ text: "", break: 1 })],
            }),
            
            // Content
            new Paragraph({
              children: [
                new TextRun({
                  text: markdownToPlainText(message.content),
                  size: 24, // 12pt
                }),
              ],
            }),
            
            // Metadata if available
            ...(message.queryTime || message.iterations || message.executionMode ? [
              new Paragraph({
                children: [new TextRun({ text: "", break: 1 })],
              }),
              new Paragraph({
                children: [
                  new TextRun({
                    text: this._formatMetadata(message),
                    size: 20, // 10pt
                    color: "666666",
                  }),
                ],
              }),
            ] : []),
          ],
        }],
      });

      // Generate the document
      const buffer = await Packer.toBuffer(doc);
      
      // Create blob and download
      const blob = new Blob([buffer], { 
        type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' 
      });
      
      // Generate filename
      const timestamp = new Date(message.timestamp).toISOString().split('T')[0];
      const filename = `chat-message-${timestamp}.docx`;
      
      // Download the file
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      return { success: true, filename };
      
    } catch (error) {
      console.error('Failed to export DOCX:', error);
      throw new Error('Failed to export DOCX. Please try again.');
    }
  };

  /**
   * Export message content for editor (returns formatted content for clipboard/editor)
   */
  exportForEditor = async (message) => {
    try {
      const timestamp = new Date(message.timestamp).toLocaleString();
      const metadata = this._formatMetadata(message);
      
      // Format content for editor with metadata
      const editorContent = `# Chat Message - ${timestamp}

${message.content}

${metadata ? `---\n*${metadata}*` : ''}`;
      
      // Copy to clipboard
      await navigator.clipboard.writeText(editorContent);
      
      return { 
        success: true, 
        content: editorContent,
        message: 'Content copied to clipboard for editor use'
      };
      
    } catch (error) {
      console.error('Failed to export for editor:', error);
      throw new Error('Failed to copy content to clipboard. Please try again.');
    }
  };

  /**
   * Export conversation as a complete document
   */
  exportConversation = async (conversation, format = 'pdf') => {
    try {
      if (format === 'pdf') {
        return await this._exportConversationAsPDF(conversation);
      } else if (format === 'docx') {
        return await this._exportConversationAsDOCX(conversation);
      } else {
        throw new Error(`Unsupported format: ${format}`);
      }
    } catch (error) {
      console.error('Failed to export conversation:', error);
      throw new Error(`Failed to export conversation as ${format.toUpperCase()}. Please try again.`);
    }
  };

  /**
   * Export Markdown content as EPUB via backend API
   */
  exportMarkdownAsEpub = async (markdown, options = {}) => {
    const payload = {
      content: markdown,
      document_id: options.documentId,
      folder_id: options.folderId,
      include_toc: options.includeToc !== false,
      include_cover: !!options.includeCover,
      split_on_headings: options.splitOnHeadings !== false,
      split_on_heading_levels: Array.isArray(options.splitOnHeadingLevels) ? options.splitOnHeadingLevels : [1, 2],
      metadata: options.metadata || {},
      heading_alignments: options.headingAlignments || {},
      indent_paragraphs: options.indentParagraphs !== false,
      no_indent_first_paragraph: options.noIndentFirstParagraph !== false,
    };

    // Direct fetch to receive blob
    const token = localStorage.getItem('auth_token');
    const resp = await fetch((process.env.REACT_APP_API_URL || '') + '/api/export/epub', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      throw new Error('EPUB export failed');
    }
    const blob = await resp.blob();
    const filename = (payload.metadata.title ? `${payload.metadata.title}.epub` : 'export.epub');
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    return { success: true, filename };
  }

  /**
   * Export conversation as PDF with enhanced HTML-based rendering
   */
  _exportConversationAsPDF = async (conversation) => {
    // Create temporary HTML element for rendering
    const tempDiv = document.createElement('div');
    tempDiv.style.position = 'absolute';
    tempDiv.style.left = '-9999px';
    tempDiv.style.top = '-9999px';
    tempDiv.style.width = '210mm'; // A4 width
    tempDiv.style.minHeight = '297mm'; // A4 height
    tempDiv.style.backgroundColor = 'white';
    tempDiv.style.fontFamily = '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif';
    tempDiv.style.fontSize = '14px';
    tempDiv.style.lineHeight = '1.6';
    tempDiv.style.color = '#333';
    tempDiv.style.padding = '20mm';

    // Add conversation title
    const titleDiv = document.createElement('h1');
    titleDiv.textContent = `Conversation: ${conversation.title || 'Untitled'}`;
    titleDiv.style.fontSize = '20px';
    titleDiv.style.fontWeight = 'bold';
    titleDiv.style.marginBottom = '10px';
    titleDiv.style.borderBottom = '2px solid #333';
    titleDiv.style.paddingBottom = '10px';
    tempDiv.appendChild(titleDiv);

    // Add conversation metadata
    const metaDiv = document.createElement('div');
    metaDiv.style.fontSize = '12px';
    metaDiv.style.color = '#666';
    metaDiv.style.marginBottom = '30px';
    metaDiv.textContent = `Created: ${new Date(conversation.created_at).toLocaleString()}`;
    tempDiv.appendChild(metaDiv);

    // Add messages
    for (const message of conversation.messages || []) {
      const messageDiv = document.createElement('div');
      messageDiv.style.marginBottom = '20px';
      messageDiv.style.border = '1px solid #e1e4e8';
      messageDiv.style.borderRadius = '8px';
      messageDiv.style.padding = '16px';
      messageDiv.style.backgroundColor = '#fafbfc';

      // Message header
      const headerDiv = document.createElement('div');
      headerDiv.style.fontSize = '13px';
      headerDiv.style.fontWeight = 'bold';
      headerDiv.style.marginBottom = '8px';
      headerDiv.style.color = message.role === 'user' ? '#0366d6' : '#28a745';
      const role = message.role === 'user' ? 'User' : 'Assistant';
      const timestamp = new Date(message.timestamp).toLocaleString();
      headerDiv.textContent = `${role} - ${timestamp}`;
      messageDiv.appendChild(headerDiv);

      // Message content
      const contentDiv = document.createElement('div');
      contentDiv.style.width = '100%';
      contentDiv.innerHTML = this._markdownToStyledHtml(message.content);
      messageDiv.appendChild(contentDiv);

      tempDiv.appendChild(messageDiv);
    }

    // Add to DOM temporarily
    document.body.appendChild(tempDiv);

    try {
      // Process links for PDF annotations
      const links = this._processLinksForPDF(tempDiv);

      // Use html2canvas to render the HTML as canvas
      const canvas = await html2canvas(tempDiv, {
        scale: 2, // Higher resolution
        useCORS: true,
        allowTaint: false,
        backgroundColor: '#ffffff',
        width: ExportService.CONTENT_WIDTH_PX, // Use content width, not full A4
        height: (() => {
          const contentHeightPx = tempDiv.offsetHeight;
          const pagesNeeded = Math.ceil(contentHeightPx / ExportService.A4_HEIGHT_PX);
          return Math.min(contentHeightPx, ExportService.A4_HEIGHT_PX * pagesNeeded);
        })(),
        scrollX: 0,
        scrollY: 0,
        ignoreElements: (element) => {
          // Skip elements that might cause issues
          return element.tagName === 'SCRIPT' || element.tagName === 'STYLE';
        }
      });

      // Remove temp element
      document.body.removeChild(tempDiv);

      // Create PDF from canvas with proper page handling
      const pdf = new jsPDF('p', 'mm', 'a4');

      // Canvas dimensions in pixels
      const canvasWidth = canvas.width;
      const canvasHeight = canvas.height;

      // PDF dimensions in mm
      const pdfPageWidth = 210; // A4 width
      const pdfPageHeight = 297; // A4 height
      const pdfContentWidth = 170; // 210 - 40mm margins
      const pdfContentHeight = 257; // 297 - 40mm margins

      // Split canvas into pages
      const pixelsPerPage = ExportService.A4_HEIGHT_PX;
      const pages = Math.ceil(canvasHeight / pixelsPerPage);

      for (let page = 0; page < pages; page++) {
        if (page > 0) {
          pdf.addPage();
        }

        // Calculate the portion of canvas for this page
        const startY = page * pixelsPerPage;
        const pageHeightPx = Math.min(pixelsPerPage, canvasHeight - startY);

        // Create a temporary canvas for this page
        const pageCanvas = document.createElement('canvas');
        const pageCtx = pageCanvas.getContext('2d');
        pageCanvas.width = canvasWidth;
        pageCanvas.height = pageHeightPx;

        // Draw the portion of the original canvas
        pageCtx.drawImage(
          canvas,
          0, startY, canvasWidth, pageHeightPx, // Source
          0, 0, canvasWidth, pageHeightPx     // Destination
        );

        // Convert to image data
        const pageImgData = pageCanvas.toDataURL('image/png');

        // Add to PDF with proper positioning (centered with margins)
        const marginLeft = 20; // 20mm left margin
        const marginTop = 20;  // 20mm top margin

        // Calculate the scaled dimensions for PDF
        // Since we're using scale: 1.0 in html2canvas, canvas pixels match CSS pixels
        const scaleFactor = canvasWidth / ExportService.CONTENT_WIDTH_PX; // Should be 1.0 with scale: 1.0
        const pdfImageWidth = pdfContentWidth;
        const pdfImageHeight = (pageHeightPx / pixelsPerPage) * pdfContentHeight;

        pdf.addImage(
          pageImgData,
          'PNG',
          marginLeft, marginTop, // x, y position
          pdfImageWidth, // width
          pdfImageHeight // height scaled proportionally
        );
      }

      // Add clickable links to PDF
      if (links.length > 0) {
        this._addLinksToPDF(pdf, links, 1.0); // scaleFactor matches canvas scale of 1.0
      }

      // Generate filename
      const timestamp = new Date().toISOString().split('T')[0];
      const filename = `conversation-${timestamp}.pdf`;

      // Save the PDF
      pdf.save(filename);

      return { success: true, filename };

    } catch (canvasError) {
      // Clean up and fall back to basic method
      document.body.removeChild(tempDiv);
      return await this._exportConversationAsPDFBasic(conversation);
    }
  };

  /**
   * Basic conversation PDF export method (fallback) - Original implementation
   */
  _exportConversationAsPDFBasic = async (conversation) => {
    const pdf = new jsPDF();
    pdf.setFont('helvetica');
    
    let yPosition = 20;
    const lineHeight = 7;
    
    // Add conversation title
    pdf.setFontSize(18);
    pdf.text(`Conversation: ${conversation.title || 'Untitled'}`, 20, yPosition);
    yPosition += 15;
    
    // Add conversation metadata
    pdf.setFontSize(10);
    pdf.setTextColor(100, 100, 100);
    const metadata = `Created: ${new Date(conversation.created_at).toLocaleString()}`;
    pdf.text(metadata, 20, yPosition);
    yPosition += 15;
    
    // Add messages
    pdf.setFontSize(12);
    pdf.setTextColor(0, 0, 0);
    
    for (const message of conversation.messages || []) {
      // Check if we need a new page
      if (yPosition > 270) {
        pdf.addPage();
        yPosition = 20;
      }
      
      // Add message header
      const role = message.role === 'user' ? 'User' : 'Assistant';
      const timestamp = new Date(message.timestamp).toLocaleString();
      pdf.setFontSize(11);
      pdf.setTextColor(50, 50, 50);
      pdf.text(`${role} - ${timestamp}`, 20, yPosition);
      yPosition += 8;
      
      // Add message content with formatting preserved
      pdf.setFontSize(12);
      pdf.setTextColor(0, 0, 0);
      const content = this._processMarkdownForPDF(message.content);
      
      for (const line of content) {
        // Check if we need a new page
        if (yPosition > 270) {
          pdf.addPage();
          yPosition = 20;
        }
        
        // Apply formatting based on line type
        let fontStyle = 'normal';
        if (line.bold && line.italic) {
          fontStyle = 'bolditalic';
        } else if (line.bold) {
          fontStyle = 'bold';
        } else if (line.italic) {
          fontStyle = 'italic';
        }
        
        // Set font family (monospace for code)
        const fontFamily = line.monospace ? 'courier' : 'helvetica';
        pdf.setFont(fontFamily, fontStyle);
        
        // Set font size with better scaling
        const fontSize = line.fontSize || 12;
        pdf.setFontSize(fontSize);
        
        // Add background for code blocks (simulate with border)
        if (line.backgroundColor) {
          pdf.setDrawColor(200, 200, 200);
          pdf.setLineWidth(0.1);
        }
        
        // Set line spacing
        const currentLineHeight = line.lineSpacing ? lineHeight * line.lineSpacing : lineHeight;
        
        // Split text to fit page width with proper margin adjustment
        const maxWidth = Math.max(50, 170 - (line.indent || 20)); // Ensure minimum width
        const textLines = pdf.splitTextToSize(line.text, maxWidth);
        
        for (const textLine of textLines) {
          if (yPosition > 270) {
            pdf.addPage();
            yPosition = 20;
          }
          
          // Add left border for blockquotes
          if (line.borderLeft) {
            pdf.setDrawColor(100, 100, 100);
            pdf.setLineWidth(1);
            pdf.line((line.indent || 20) - 5, yPosition - 3, (line.indent || 20) - 5, yPosition + 3);
          }
          
          pdf.text(textLine, line.indent || 20, yPosition);
          yPosition += currentLineHeight;
        }
        
        // Add extra spacing for headers and code blocks
        if (line.lineSpacing && line.lineSpacing > 1) {
          yPosition += lineHeight * 0.5;
        }
        
        // Reset font for next line
        pdf.setFont('helvetica', 'normal');
        pdf.setFontSize(12);
        pdf.setDrawColor(0, 0, 0);
        pdf.setLineWidth(0.2);
      }
      
      yPosition += 10; // Space between messages
    }
    
    // Generate filename
    const timestamp = new Date().toISOString().split('T')[0];
    const filename = `conversation-${timestamp}.pdf`;
    pdf.save(filename);
    
    return { success: true, filename };
  };

  /**
   * Export conversation as DOCX
   */
  _exportConversationAsDOCX = async (conversation) => {
    const children = [
      // Title
      new Paragraph({
        text: `Conversation: ${conversation.title || 'Untitled'}`,
        heading: HeadingLevel.HEADING_1,
        alignment: AlignmentType.CENTER,
      }),
      
      // Metadata
      new Paragraph({
        children: [
          new TextRun({
            text: `Created: ${new Date(conversation.created_at).toLocaleString()}`,
            size: 20,
            color: "666666",
          }),
        ],
      }),
      
      new Paragraph({
        children: [new TextRun({ text: "", break: 1 })],
      }),
    ];
    
    // Add messages
    for (const message of conversation.messages || []) {
      const role = message.role === 'user' ? 'User' : 'Assistant';
      const timestamp = new Date(message.timestamp).toLocaleString();
      
      children.push(
        // Message header
        new Paragraph({
          children: [
            new TextRun({
              text: `${role} - ${timestamp}`,
              size: 22,
              color: "333333",
              bold: true,
            }),
          ],
        }),
        
        // Message content
        new Paragraph({
          children: [
            new TextRun({
              text: markdownToPlainText(message.content),
              size: 24,
            }),
          ],
        }),
        
        // Spacing
        new Paragraph({
          children: [new TextRun({ text: "", break: 1 })],
        }),
      );
    }
    
    const doc = new Document({
      sections: [{
        properties: {},
        children,
      }],
    });
    
    const buffer = await Packer.toBuffer(doc);
    const blob = new Blob([buffer], { 
      type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' 
    });
    
    const timestamp = new Date().toISOString().split('T')[0];
    const filename = `conversation-${timestamp}.docx`;
    
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    
    return { success: true, filename };
  };

  /**
   * Format metadata for display
   */
  _formatMetadata = (message) => {
    const metadata = [];
    if (message.queryTime) metadata.push(`Query Time: ${message.queryTime.toFixed(2)}s`);
    if (message.iterations) metadata.push(`Iterations: ${message.iterations}`);
    if (message.executionMode) metadata.push(`Mode: ${message.executionMode.toUpperCase()}`);
    
    return metadata.join(' • ');
  };

  /**
   * Process markdown content for PDF export with comprehensive formatting preserved
   */
  _processMarkdownForPDF = (markdown) => {
    if (!markdown) return [];
    
    const lines = markdown.split('\n');
    const processedLines = [];
    let inCodeBlock = false;
    let codeBlockLanguage = '';
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmedLine = line.trim();
      
      // Handle code block start/end
      if (trimmedLine.startsWith('```')) {
        if (!inCodeBlock) {
          // Starting code block
          inCodeBlock = true;
          codeBlockLanguage = trimmedLine.replace(/```/, '').trim();
          processedLines.push({
            text: `Code Block${codeBlockLanguage ? ` (${codeBlockLanguage})` : ''}:`,
            bold: true,
            fontSize: 10,
            indent: 20
          });
        } else {
          // Ending code block
          inCodeBlock = false;
          codeBlockLanguage = '';
          processedLines.push({
            text: '',
            indent: 20
          });
        }
        continue;
      }
      
      // Handle lines inside code blocks
      if (inCodeBlock) {
        processedLines.push({
          text: line, // Preserve original indentation and formatting
          fontSize: 9,
          monospace: true,
          indent: 30,
          backgroundColor: '#f6f8fa'
        });
        continue;
      }
      
      // Handle headers
      if (trimmedLine.startsWith('#')) {
        const level = Math.min(6, trimmedLine.match(/^#+/)[0].length);
        const text = trimmedLine.replace(/^#+\s*/, '');
        const processedText = this._processInlineFormatting(text);
        processedLines.push({
          text: processedText.text,
          bold: true,
          fontSize: Math.max(10, 18 - level), // h1=17, h2=16, h3=15, etc.
          indent: 20,
          lineSpacing: level <= 2 ? 2 : 1.5
        });
        // Add spacing after headers
        processedLines.push({ text: '', indent: 20 });
        continue;
      }
      
      // Handle horizontal rules
      if (trimmedLine.match(/^[-*_]{3,}$/)) {
        processedLines.push({
          text: '─'.repeat(50),
          indent: 20,
          fontSize: 8
        });
        processedLines.push({ text: '', indent: 20 });
        continue;
      }
      
      // Handle task lists
      if (trimmedLine.match(/^[-*+]\s+\[[xX ]\]\s+/)) {
        const isChecked = trimmedLine.match(/^[-*+]\s+\[[xX]\]\s+/);
        const text = trimmedLine.replace(/^[-*+]\s+\[[xX ]\]\s+/, '');
        const processedText = this._processInlineFormatting(text);
        processedLines.push({
          text: `${isChecked ? '☑' : '☐'} ${processedText.text}`,
          indent: 30,
          bold: processedText.hasBold,
          italic: processedText.hasItalic
        });
        continue;
      }
      
      // Handle bullet points with nesting support
      const bulletMatch = trimmedLine.match(/^(\s*)([-*+])\s+(.+)/);
      if (bulletMatch) {
        const indentLevel = Math.floor(bulletMatch[1].length / 2);
        const text = bulletMatch[3];
        const processedText = this._processInlineFormatting(text);
        processedLines.push({
          text: `• ${processedText.text}`,
          indent: 30 + (indentLevel * 15),
          bold: processedText.hasBold,
          italic: processedText.hasItalic
        });
        continue;
      }
      
      // Handle numbered lists with nesting support
      const numberedMatch = trimmedLine.match(/^(\s*)(\d+)\.\s+(.+)/);
      if (numberedMatch) {
        const indentLevel = Math.floor(numberedMatch[1].length / 2);
        const number = numberedMatch[2];
        const text = numberedMatch[3];
        const processedText = this._processInlineFormatting(text);
        processedLines.push({
          text: `${number}. ${processedText.text}`,
          indent: 30 + (indentLevel * 15),
          bold: processedText.hasBold,
          italic: processedText.hasItalic
        });
        continue;
      }
      
      // Handle blockquotes with nesting support
      const quoteMatch = trimmedLine.match(/^(>+)\s*(.*)/);
      if (quoteMatch) {
        const level = quoteMatch[1].length;
        const text = quoteMatch[2];
        const processedText = this._processInlineFormatting(text);
        processedLines.push({
          text: processedText.text,
          italic: true,
          indent: 30 + ((level - 1) * 15),
          borderLeft: true
        });
        continue;
      }
      
      // Handle tables (basic support)
      if (trimmedLine.includes('|') && !trimmedLine.match(/^\s*\|[\s\-|:]+\|\s*$/)) {
        const cells = trimmedLine.split('|').map(cell => cell.trim()).filter(cell => cell);
        const tableRow = cells.join(' | ');
        processedLines.push({
          text: tableRow,
          fontSize: 10,
          indent: 20,
          monospace: true
        });
        continue;
      }
      
      // Skip table separator lines
      if (trimmedLine.match(/^\s*\|[\s\-|:]+\|\s*$/)) {
        continue;
      }
      
      // Handle regular text with inline formatting
      if (trimmedLine) {
        const processedText = this._processInlineFormatting(trimmedLine);
        processedLines.push({
          text: processedText.text,
          indent: 20,
          bold: processedText.hasBold,
          italic: processedText.hasItalic,
          hasLinks: processedText.hasLinks
        });
      } else {
        // Empty line for spacing
        processedLines.push({
          text: '',
          indent: 20
        });
      }
    }
    
    return processedLines;
  };

  /**
   * Process inline markdown formatting (bold, italic, code, links, strikethrough)
   */
  _processInlineFormatting = (text) => {
    if (!text) return { text: '', hasBold: false, hasItalic: false, hasLinks: false };
    
    let processedText = text;
    let hasBold = false;
    let hasItalic = false;
    let hasLinks = false;
    
    // Process links first to preserve them
    const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g;
    processedText = processedText.replace(linkPattern, (match, linkText, url) => {
      hasLinks = true;
      return `${linkText} (${url})`;
    });
    
    // Process strikethrough
    processedText = processedText.replace(/~~(.+?)~~/g, '$1 [crossed out]');
    
    // Process bold-italic combinations (*** or ___)
    processedText = processedText.replace(/\*\*\*(.+?)\*\*\*/g, (match, content) => {
      hasBold = true;
      hasItalic = true;
      return content;
    });
    processedText = processedText.replace(/___(.+?)___/g, (match, content) => {
      hasBold = true;
      hasItalic = true;
      return content;
    });
    
    // Process bold (**text** or __text__)
    processedText = processedText.replace(/\*\*(.+?)\*\*/g, (match, content) => {
      hasBold = true;
      return content;
    });
    processedText = processedText.replace(/__(.+?)__/g, (match, content) => {
      hasBold = true;
      return content;
    });
    
    // Process italic (*text* or _text_) - be careful not to match underscores in URLs
    processedText = processedText.replace(/(?<!\w)\*([^*]+?)\*(?!\w)/g, (match, content) => {
      hasItalic = true;
      return content;
    });
    processedText = processedText.replace(/(?<!\w)_([^_]+?)_(?!\w)/g, (match, content) => {
      hasItalic = true;
      return content;
    });
    
    // Process inline code
    processedText = processedText.replace(/`([^`]+)`/g, (match, code) => {
      return `[CODE: ${code}]`;
    });
    
    return {
      text: processedText,
      hasBold,
      hasItalic,
      hasLinks
    };
  };

  /**
   * Split text into segments for mixed bold/italic formatting (legacy method - kept for compatibility)
   */
  _splitBoldSegments = (text) => {
    const segments = [];
    let currentText = '';
    let isBold = false;
    let isItalic = false;
    
    // Simple regex to find bold patterns
    const boldPattern = /\*\*(.*?)\*\*/g;
    let match;
    let lastIndex = 0;
    
    while ((match = boldPattern.exec(text)) !== null) {
      // Add text before bold
      if (match.index > lastIndex) {
        const beforeText = text.substring(lastIndex, match.index);
        if (beforeText.trim()) {
          segments.push({
            text: beforeText,
            bold: false,
            italic: false
          });
        }
      }
      
      // Add bold text
      segments.push({
        text: match[1],
        bold: true,
        italic: false
      });
      
      lastIndex = match.index + match[0].length;
    }
    
    // Add remaining text
    if (lastIndex < text.length) {
      const remainingText = text.substring(lastIndex);
      if (remainingText.trim()) {
        segments.push({
          text: remainingText,
          bold: false,
          italic: false
        });
      }
    }
    
    return segments.length > 0 ? segments : [{ text, bold: false, italic: false }];
  };

  /**
   * Export markdown document as PDF with proper formatting
   * Uses native jsPDF text rendering for high quality, selectable text and no canvas size limits
   */
  exportMarkdownAsPDF = async (markdownContent, options = {}) => {
    try {
      console.log('PDF Export: Starting text-based markdown document export...');
      
      // Yield to UI immediately
      await new Promise(resolve => setTimeout(resolve, 10));
      
      // Parse frontmatter to extract metadata
      const parsed = parseFrontmatter(markdownContent);
      const frontmatter = { ...(parsed.data || {}), ...(parsed.lists || {}) };
      const bodyContent = parsed.body || markdownContent;
      
      // Extract title from frontmatter or use filename
      const title = frontmatter.title || frontmatter.Title || options.filename || 'Document';
      const documentTitle = title.replace(/\.[^.]+$/, '');
      
      // Create PDF with US Letter dimensions
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: [ExportService.PAGE_WIDTH_MM, ExportService.PAGE_HEIGHT_MM]
      });

      // Set up layout constants
      const margin = 20; 
      const pageWidth = ExportService.PAGE_WIDTH_MM - (2 * margin);
      const pageHeight = ExportService.PAGE_HEIGHT_MM - (2 * margin);
      let yPosition = margin;

      // 1. Add title header
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(22);
      const titleLines = pdf.splitTextToSize(documentTitle, pageWidth);
      pdf.text(titleLines, margin, yPosition);
      yPosition += titleLines.length * 9 + 5;

      // Add separator line
      pdf.setDrawColor(50, 50, 50);
      pdf.setLineWidth(0.5);
      pdf.line(margin, yPosition, margin + pageWidth, yPosition);
      yPosition += 10;

      // 2. Parse markdown into elements
      console.log('PDF Export: Parsing markdown content...');
      const elements = this._parseMarkdownForTextPDF(bodyContent);
      
      // 3. Render elements
      console.log(`PDF Export: Rendering ${elements.length} elements...`);
      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(12);

      for (let i = 0; i < elements.length; i++) {
        const element = elements[i];
        
        // Yield to UI periodically to prevent hanging
        if (i % 30 === 0) {
          await new Promise(resolve => setTimeout(resolve, 0));
        }

        // Handle different element types
        if (element.type === 'spacer') {
          yPosition += 4;
          continue;
        }

        if (element.type === 'heading') {
          pdf.setFont('helvetica', 'bold');
          const fontSize = element.level === 1 ? 18 : element.level === 2 ? 16 : element.level === 3 ? 14 : 12;
          pdf.setFontSize(fontSize);
          
          // Check if we need a new page for the heading
          if (yPosition > ExportService.PAGE_HEIGHT_MM - margin - 20) {
            pdf.addPage();
            yPosition = margin;
          }
          
          yPosition = this._renderTextWithFormattingImproved(pdf, element.text, margin, yPosition, pageWidth, margin, pageHeight, fontSize);
          yPosition += 2; // Minimal space after header; user can add blank line in markdown
          pdf.setFontSize(12);
          pdf.setFont('helvetica', 'normal');
        }

        else if (element.type === 'codeblock') {
          pdf.setFont('courier', 'normal');
          pdf.setFontSize(10);
          
          const codeLines = pdf.splitTextToSize(element.text, pageWidth - 10);
          const lineHeight = 5;
          
          // If code block is too long, we'll draw it across pages
          for (let j = 0; j < codeLines.length; j++) {
            if (yPosition > ExportService.PAGE_HEIGHT_MM - margin - 10) {
              pdf.addPage();
              yPosition = margin;
            }
            
            // Draw background for this line
            pdf.setFillColor(245, 245, 245);
            pdf.rect(margin - 2, yPosition - 4, pageWidth + 4, lineHeight + 1, 'F');
            
            pdf.text(codeLines[j], margin + 3, yPosition);
            yPosition += lineHeight;
          }
          yPosition += 6;
          pdf.setFont('helvetica', 'normal');
          pdf.setFontSize(12);
        }

        else if (element.type === 'list') {
          const indent = (element.indent || 0) * 8;
          const bullet = element.isOrdered ? `${element.marker}. ` : '• ';
          
          // Check if we need a new page
          if (yPosition > ExportService.PAGE_HEIGHT_MM - margin - 10) {
            pdf.addPage();
            yPosition = margin;
          }
          
          pdf.setFont('helvetica', 'bold');
          pdf.text(bullet, margin + indent, yPosition);
          pdf.setFont('helvetica', 'normal');
          
          yPosition = this._renderTextWithFormattingImproved(pdf, element.text, margin + indent + 6, yPosition, pageWidth - indent - 6, margin, pageHeight);
          yPosition += 2;
        }

        else if (element.type === 'blockquote') {
          // Check if we need a new page
          if (yPosition > ExportService.PAGE_HEIGHT_MM - margin - 10) {
            pdf.addPage();
            yPosition = margin;
          }
          
          const textX = margin + 8;
          const textWidth = pageWidth - 8;
          
          const startY = yPosition - 4;
          yPosition = this._renderTextWithFormattingImproved(pdf, element.text, textX, yPosition, textWidth, margin, pageHeight);
          
          // Draw left border (simplified for multi-page blockquotes)
          pdf.setDrawColor(200, 200, 200);
          pdf.setLineWidth(1);
          pdf.line(margin + 2, startY, margin + 2, yPosition - 2);
          yPosition += 4;
        }

        else if (element.type === 'hr') {
          if (yPosition > ExportService.PAGE_HEIGHT_MM - margin - 10) {
            pdf.addPage();
            yPosition = margin;
          }
          pdf.setDrawColor(200, 200, 200);
          pdf.setLineWidth(0.2);
          pdf.line(margin, yPosition, margin + pageWidth, yPosition);
          yPosition += 8;
        }

        else if (element.type === 'table-row') {
          // Tables are hard in jsPDF text mode, using simplified rendering
          const cellCount = element.cells.length;
          const cellWidth = pageWidth / cellCount;
          
          let maxCellLines = 0;
          const cellContent = element.cells.map(cell => {
            const lines = pdf.splitTextToSize(this._flattenFormattedText(cell), cellWidth - 4);
            if (lines.length > maxCellLines) maxCellLines = lines.length;
            return lines;
          });
          
          const rowHeight = (maxCellLines * 6) + 4;
          
          if (yPosition + rowHeight > ExportService.PAGE_HEIGHT_MM - margin) {
            pdf.addPage();
            yPosition = margin;
          }
          
          let currentX = margin;
          for (let j = 0; j < cellCount; j++) {
            pdf.setDrawColor(200, 200, 200);
            pdf.rect(currentX, yPosition - 5, cellWidth, rowHeight);
            pdf.text(cellContent[j], currentX + 2, yPosition);
            currentX += cellWidth;
          }
          yPosition += rowHeight;
        }

        else {
          // Regular paragraph
          if (yPosition > ExportService.PAGE_HEIGHT_MM - margin - 10) {
            pdf.addPage();
            yPosition = margin;
          }
          yPosition = this._renderTextWithFormattingImproved(pdf, element.text, margin, yPosition, pageWidth, margin, pageHeight);
          yPosition += 4;
        }
      }

      // Add page numbers
      const totalPages = pdf.internal.getNumberOfPages();
      for (let i = 1; i <= totalPages; i++) {
        pdf.setPage(i);
        pdf.setFont('helvetica', 'italic');
        pdf.setFontSize(9);
        pdf.setTextColor(150, 150, 150);
        pdf.text(
          `Page ${i} of ${totalPages}`, 
          ExportService.PAGE_WIDTH_MM / 2, 
          ExportService.PAGE_HEIGHT_MM - 10, 
          { align: 'center' }
        );
      }

      console.log('PDF Export: Saving document...');
      pdf.save(`${documentTitle}.pdf`);
      console.log('PDF Export: Successfully exported markdown document');
      
      return { success: true, filename: `${documentTitle}.pdf` };

    } catch (error) {
      console.error('PDF Export: Failed to export markdown document:', error);
      throw new Error(`Failed to export PDF: ${error.message}`);
    }
  };

  /**
   * Improved version of _renderTextWithFormatting that handles page breaks
   * FIXED: Keeps inline code and formatted text together, moves to next line if needed
   */
  _renderTextWithFormattingImproved = (pdf, segments, x, y, maxWidth, margin, pageHeight, fontSize = 12) => {
    let currentX = x;
    let currentY = y;
    const lineHeight = (fontSize / 2) + 1;

    if (!Array.isArray(segments)) {
      segments = [{ text: segments, type: 'text' }];
    }

    const processedSegments = segments.map(segment => ({
      ...segment,
      text: this._containsEmoji(segment.text) ? this._replaceEmojis(segment.text) : segment.text
    }));

    let lastSegmentText = '';
    for (const segment of processedSegments) {
      let text = segment.text || '';
      const type = segment.type || 'text';

      // Ensure space before inline code when previous segment didn't end with whitespace
      if (type === 'inline-code' && currentX > x && lastSegmentText.length > 0 && !/\s$/.test(lastSegmentText)) {
        text = ' ' + text;
      }
      lastSegmentText = segment.text || '';

      switch (type) {
        case 'bold': pdf.setFont('helvetica', 'bold'); break;
        case 'italic': pdf.setFont('helvetica', 'italic'); break;
        case 'bolditalic': pdf.setFont('helvetica', 'bolditalic'); break;
        case 'inline-code': pdf.setFont('courier', 'normal'); break;
        case 'link': pdf.setTextColor(0, 0, 255); pdf.setFont('helvetica', 'normal'); break;
        default: pdf.setFont('helvetica', 'normal'); break;
      }

      // For inline code and links, try to keep them together on one line
      const availableWidth = maxWidth - (currentX - x);
      const textWidth = pdf.getTextWidth(text);
      
      if ((type === 'inline-code' || type === 'link') && textWidth > availableWidth && currentX > x) {
        // Move to next line to keep inline code/link together
        currentX = x;
        currentY += lineHeight;
        
        if (currentY > pageHeight) {
          pdf.addPage();
          currentY = margin;
        }
      }

      // Now split text to fit (use full maxWidth if we just moved to new line)
      const widthToUse = (currentX === x) ? maxWidth : (maxWidth - (currentX - x));
      const lines = pdf.splitTextToSize(text, widthToUse);
      
      // Render each line
      for (let l = 0; l < lines.length; l++) {
        if (currentY > pageHeight) {
          pdf.addPage();
          currentY = margin;
        }
        
        pdf.text(lines[l], currentX, currentY);
        
        if (l < lines.length - 1) {
          // Move to next line
          currentX = x;
          currentY += lineHeight;
        } else {
          // Last line - update X position
          currentX += pdf.getTextWidth(lines[l]);
          // jsPDF can under-measure trailing spaces; ensure we advance past them so next segment doesn't overwrite
          const trailingSpaces = (lines[l].match(/\s+$/) || [''])[0].length;
          if (trailingSpaces > 0) {
            currentX += pdf.getTextWidth(' ') * trailingSpaces;
          }
        }
      }

      pdf.setTextColor(0, 0, 0);
    }

    return currentY + (currentX === x ? 0 : lineHeight);
  };

  /**
   * Escape HTML to prevent XSS
   */
  _escapeHtml = (text) => {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  };
}

export default new ExportService(); 