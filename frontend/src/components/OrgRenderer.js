import React, { useMemo, useState, useCallback } from 'react';

const TODO_STATES = [
  'TODO', 'NEXT', 'STARTED', 'WAITING', 'HOLD',
  'DONE', 'CANCELED', 'CANCELLED', 'WONTFIX', 'FIXED'
];

/**
 * Parse org-mode links from text: [[link]] or [[link][description]]
 * Returns array of text segments and link objects
 */
function parseLinksInText(text) {
  if (!text) return [{ type: 'text', content: text }];
  
  const segments = [];
  const linkRegex = /\[\[([^\]]+)\](?:\[([^\]]+)\])?\]/g;
  let lastIndex = 0;
  let match;
  
  while ((match = linkRegex.exec(text)) !== null) {
    // Add text before link
    if (match.index > lastIndex) {
      segments.push({ type: 'text', content: text.slice(lastIndex, match.index) });
    }
    
    // Add link
    const linkTarget = match[1];
    const linkDescription = match[2] || linkTarget;
    segments.push({ type: 'link', target: linkTarget, description: linkDescription });
    
    lastIndex = match.index + match[0].length;
  }
  
  // Add remaining text
  if (lastIndex < text.length) {
    segments.push({ type: 'text', content: text.slice(lastIndex) });
  }
  
  return segments.length > 0 ? segments : [{ type: 'text', content: text }];
}

/**
 * Parse link target to determine type and extract components
 */
function parseLinkTarget(target) {
  if (!target) return { type: 'text', value: target };
  
  // HTTP/HTTPS URLs
  if (target.startsWith('http://') || target.startsWith('https://')) {
    return { type: 'url', value: target };
  }
  
  // File links: file:path/to/file.org
  if (target.startsWith('file:')) {
    const filePath = target.slice(5); // Remove 'file:' prefix
    return { type: 'file', value: filePath };
  }
  
  // ID links: id:uuid
  if (target.startsWith('id:')) {
    const id = target.slice(3); // Remove 'id:' prefix
    return { type: 'id', value: id };
  }
  
  // Internal heading links: #heading or *heading
  if (target.startsWith('#') || target.startsWith('*')) {
    const heading = target.slice(1);
    return { type: 'heading', value: heading };
  }
  
  // Plain text - could be a file name or internal link
  // Check if it looks like a file (has extension)
  if (target.match(/\.(org|md|txt)$/i)) {
    return { type: 'file', value: target };
  }
  
  // Otherwise treat as internal heading link
  return { type: 'heading', value: target };
}

/**
 * Link component with click handling for different link types
 */
function OrgLink({ target, description, onNavigate }) {
  const linkInfo = parseLinkTarget(target);
  
  const handleClick = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (linkInfo.type === 'url') {
      // Open URLs in new tab
      window.open(linkInfo.value, '_blank', 'noopener,noreferrer');
    } else if (linkInfo.type === 'file') {
      // File links - notify parent to open document
      if (onNavigate) {
        onNavigate({ type: 'file', path: linkInfo.value });
      }
    } else if (linkInfo.type === 'id') {
      // ID links - notify parent to navigate to heading with ID
      if (onNavigate) {
        onNavigate({ type: 'id', id: linkInfo.value });
      }
    } else if (linkInfo.type === 'heading') {
      // Internal heading links - scroll to heading
      if (onNavigate) {
        onNavigate({ type: 'heading', heading: linkInfo.value });
      }
    }
  }, [linkInfo, onNavigate]);
  
  // Style based on link type
  const linkStyle = {
    color: linkInfo.type === 'url' ? '#1976d2' : '#2e7d32',
    textDecoration: 'none',
    cursor: 'pointer',
    borderBottom: '1px solid',
    borderBottomColor: linkInfo.type === 'url' ? '#1976d2' : '#2e7d32',
    transition: 'opacity 0.2s',
  };
  
  const linkHoverStyle = {
    ...linkStyle,
    opacity: 0.7,
  };
  
  const [isHovered, setIsHovered] = useState(false);
  
  return (
    <a
      href="#"
      onClick={handleClick}
      style={isHovered ? linkHoverStyle : linkStyle}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      title={target}
    >
      {description}
    </a>
  );
}

/**
 * Render text with parsed links
 */
function renderTextWithLinks(text, keyPrefix, onNavigate) {
  const segments = parseLinksInText(text);
  
  return (
    <>
      {segments.map((segment, idx) => {
        if (segment.type === 'link') {
          return (
            <OrgLink
              key={`${keyPrefix}-link-${idx}`}
              target={segment.target}
              description={segment.description}
              onNavigate={onNavigate}
            />
          );
        }
        return <span key={`${keyPrefix}-text-${idx}`}>{segment.content}</span>;
      })}
    </>
  );
}

function parseOrg(content) {
  const lines = (content || '').split('\n');
  const root = { id: 'root', level: 0, title: 'root', tags: [], properties: {}, contentLines: [], children: [] };
  const stack = [root];
  let inCode = false;

  function startNode(level, header) {
    let tags = [];
    let titlePart = header.trim();
    const tagMatch = titlePart.match(/\s+:([A-Za-z0-9_:+-]+):\s*$/);
    if (tagMatch) {
      tags = tagMatch[1].split(':').filter(Boolean);
      titlePart = titlePart.slice(0, tagMatch.index).trim();
    }
    let todoState;
    const firstWord = titlePart.split(/\s+/)[0];
    if (TODO_STATES.includes(firstWord && firstWord.toUpperCase())) {
      todoState = firstWord.toUpperCase();
      titlePart = titlePart.slice(firstWord.length).trim();
    }
    return { id: Math.random().toString(36).slice(2), level, title: titlePart, todoState, tags, properties: {}, contentLines: [], children: [] };
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith('#+BEGIN_SRC')) { inCode = true; stack[stack.length - 1].contentLines.push('```'); continue; }
    if (line.startsWith('#+END_SRC')) { inCode = false; stack[stack.length - 1].contentLines.push('```'); continue; }
    if (inCode) { stack[stack.length - 1].contentLines.push(line); continue; }

    const hMatch = line.match(/^(\*+)\s+(.*)$/);
    if (hMatch) {
      const level = hMatch[1].length;
      const node = startNode(level, hMatch[2]);
      while (stack.length && stack[stack.length - 1].level >= level) stack.pop();
      const parent = stack[stack.length - 1];
      parent.children.push(node);
      stack.push(node);
      continue;
    }

    if (line.trim() === ':PROPERTIES:') {
      const current = stack[stack.length - 1];
      i++;
      while (i < lines.length && lines[i].trim() !== ':END:') {
        const pl = lines[i].trim();
        const pMatch = pl.match(/^:([^:]+):\s*(.*)$/);
        if (pMatch) current.properties[pMatch[1].trim()] = pMatch[2].trim();
        i++;
      }
      continue;
    }
    stack[stack.length - 1].contentLines.push(line);
  }
  return root.children;
}

function renderContentBlock(lines, keyPrefix, onNavigate) {
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line === '```') {
      const code = [];
      i++;
      while (i < lines.length && lines[i] !== '```') { code.push(lines[i]); i++; }
      blocks.push(
        <pre key={`${keyPrefix}-code-${i}`} style={{ background: '#f5f5f5', padding: 12, borderRadius: 4, overflow: 'auto' }}>
          <code>{code.join('\n')}</code>
        </pre>
      );
      i++;
      continue;
    }
    if (line && line.trim().startsWith('|')) {
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        const cells = lines[i].trim().split('|').slice(1, -1).map(c => c.trim());
        rows.push(cells);
        i++;
      }
      if (rows.length > 0) {
        blocks.push(
          <div key={`${keyPrefix}-table-${i}`} style={{ overflowX: 'auto', margin: '8px 0' }}>
            <table style={{ borderCollapse: 'collapse', width: '100%' }}><tbody>
              {rows.map((r, ri) => (
                <tr key={ri}>{r.map((c, ci) => (<td key={ci} style={{ border: '1px solid #ddd', padding: '6px 8px' }}>{renderTextWithLinks(c, `${keyPrefix}-cell-${ri}-${ci}`, onNavigate)}</td>))}</tr>
              ))}
            </tbody></table>
          </div>
        );
      }
      continue;
    }
    if (/^\s*[-+*]\s+/.test(line || '')) {
      const items = [];
      while (i < lines.length && /^\s*[-+*]\s+/.test(lines[i] || '')) {
        const li = lines[i].trim().slice(2);
        const cb = li.match(/^\[( |x|X|-)\]\s+(.*)$/);
        if (cb) items.push({ text: cb[2], checked: /[xX]/.test(cb[1]) });
        else items.push({ text: li });
        i++;
      }
      blocks.push(
        <ul key={`${keyPrefix}-list-${i}`} style={{ margin: '0 0 8px 18px' }}>
          {items.map((it, idx) => (
            <li key={idx} style={{ marginBottom: 4 }}>
              {typeof it.checked === 'boolean' ? (
                <>
                  <input type="checkbox" checked={it.checked} readOnly style={{ marginRight: 6 }} />
                  {renderTextWithLinks(it.text, `${keyPrefix}-li-${idx}`, onNavigate)}
                </>
              ) : (
                renderTextWithLinks(it.text, `${keyPrefix}-li-${idx}`, onNavigate)
              )}
            </li>
          ))}
        </ul>
      );
      continue;
    }
    const para = [];
    while (i < lines.length && lines[i] !== '```' && !(lines[i] && lines[i].trim().startsWith('|')) && !/^\s*[-+*]\s+/.test(lines[i] || '') && !/^(\*+)\s+/.test(lines[i] || '')) {
      para.push(lines[i]); i++;
      if (i < lines.length && lines[i] && lines[i].trim() === ':PROPERTIES:') break;
    }
    const text = para.join('\n').trim();
    if (text) blocks.push(
      <p key={`${keyPrefix}-p-${i}`} style={{ margin: '6px 0', whiteSpace: 'pre-wrap' }}>
        {renderTextWithLinks(text, `${keyPrefix}-p-${i}`, onNavigate)}
      </p>
    );
    else i++;
  }
  return <>{blocks}</>;
}

function NodeView({ node, onNavigate }) {
  const [collapsed, setCollapsed] = useState(false);
  const headingStyle = { display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', marginTop: node.level === 1 ? 12 : 6, marginBottom: 4 };
  const titleStyle = { fontWeight: 600, fontSize: Math.max(18 - (node.level - 1) * 2, 12) };
  const badgeColor = node.todoState === 'DONE' || node.todoState === 'FIXED' ? '#2e7d32' : '#c62828';
  return (
    <div style={{ marginLeft: (node.level - 1) * 12 }} id={`org-heading-${node.id}`}>
      <div style={headingStyle} onClick={() => setCollapsed(!collapsed)}>
        <span style={{ width: 16, display: 'inline-block', textAlign: 'center' }}>{collapsed ? '▶' : '▼'}</span>
        {node.todoState && (
          <span style={{ fontSize: 12, fontWeight: 700, color: badgeColor, border: '1px solid #ddd', borderRadius: 4, padding: '2px 6px' }}>{node.todoState}</span>
        )}
        <span style={titleStyle}>{node.title || '(no title)'}</span>
        {node.tags && node.tags.length > 0 && (
          <span style={{ marginLeft: 8, opacity: 0.8 }}>
            {node.tags.map(tag => (<span key={tag} style={{ border: '1px solid #eee', borderRadius: 4, padding: '1px 6px', marginRight: 4, fontSize: 12 }}>:{tag}:</span>))}
          </span>
        )}
      </div>
      {!collapsed && (
        <div style={{ marginLeft: 16 }}>
          {node.properties && Object.keys(node.properties).length > 0 && (
            <div style={{ margin: '4px 0' }}>
              <table style={{ borderCollapse: 'collapse', fontSize: 12, marginBottom: 6 }}><tbody>
                {Object.entries(node.properties).map(([k, v]) => (
                  <tr key={k}><td style={{ border: '1px solid #eee', padding: '2px 6px', fontWeight: 600 }}>{k}</td><td style={{ border: '1px solid #eee', padding: '2px 6px' }}>{v}</td></tr>
                ))}
              </tbody></table>
            </div>
          )}
          {renderContentBlock(node.contentLines, node.id, onNavigate)}
          {node.children.map(child => (<NodeView key={child.id} node={child} onNavigate={onNavigate} />))}
        </div>
      )}
    </div>
  );
}

export default function OrgRenderer({ content, onNavigate }) {
  const tree = useMemo(() => parseOrg(content), [content]);
  
  // Internal navigation handler for heading links
  const handleInternalNavigation = useCallback((navInfo) => {
    if (navInfo.type === 'heading') {
      // Try to find and scroll to the heading
      const headingText = navInfo.heading.toLowerCase().trim();
      const allHeadings = document.querySelectorAll('[id^="org-heading-"]');
      
      for (const heading of allHeadings) {
        const titleElement = heading.querySelector('span[style*="fontWeight: 600"]');
        if (titleElement && titleElement.textContent.toLowerCase().trim() === headingText) {
          heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
          // Briefly highlight the heading
          const originalBg = heading.style.backgroundColor;
          heading.style.backgroundColor = '#fff3cd';
          setTimeout(() => {
            heading.style.backgroundColor = originalBg;
          }, 1500);
          return;
        }
      }
      console.warn('Heading not found:', navInfo.heading);
    }
    
    // Pass other navigation types to parent
    if (onNavigate) {
      onNavigate(navInfo);
    }
  }, [onNavigate]);
  
  if (!content) return <div />;
  return (
    <div>
      {tree.length === 0 ? (
        <div style={{ whiteSpace: 'pre-wrap' }}>{renderTextWithLinks(content, 'root', handleInternalNavigation)}</div>
      ) : (
        tree.map(node => <NodeView key={node.id} node={node} onNavigate={handleInternalNavigation} />)
      )}
    </div>
  );
}


