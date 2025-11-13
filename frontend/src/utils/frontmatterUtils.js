/**
 * Frontmatter Parsing Utilities - Roosevelt's Shared Cavalry Tools
 * 
 * **BULLY!** Consistent frontmatter parsing across the application!
 */

/**
 * Parse YAML-like frontmatter from markdown text
 * **By George!** Extract metadata like a well-organized cavalry charge!
 */
export function parseFrontmatter(text) {
  try {
    const trimmed = text.startsWith('\ufeff') ? text.slice(1) : text;
    if (!trimmed.startsWith('---\n')) return { data: {}, lists: {}, order: [], raw: '', body: text };
    const end = trimmed.indexOf('\n---', 4);
    if (end === -1) return { data: {}, lists: {}, order: [], raw: '', body: text };
    const yaml = trimmed.slice(4, end).replace(/\r/g, '');
    const body = trimmed.slice(end + 4).replace(/^\n/, '');
    const data = {};
    const lists = {};
    const order = [];
    const lines = yaml.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const m = line.match(/^([A-Za-z0-9_\-]+):\s*(.*)$/);
      if (m) {
        const k = m[1].trim();
        const v = m[2];
        order.push(k);
        if (v && v.trim().length > 0) {
          data[k] = String(v).trim();
        } else {
          const items = [];
          let j = i + 1;
          while (j < lines.length) {
            const ln = lines[j];
            if (/^\s*-\s+/.test(ln)) {
              items.push(ln.replace(/^\s*-\s+/, ''));
              j++;
            } else if (/^\s+$/.test(ln)) {
              j++;
            } else {
              break;
            }
          }
          if (items.length > 0) {
            lists[k] = items;
            i = j - 1;
          } else {
            data[k] = '';
          }
        }
      }
    }
    return { data, lists, order, raw: yaml, body };
  } catch (e) {
    return { data: {}, lists: {}, order: [], raw: '', body: text };
  }
}

/**
 * Build frontmatter YAML block from data object
 * **BULLY!** Construct metadata like a well-drilled regiment!
 */
export function buildFrontmatter(data) {
  if (!data || typeof data !== 'object') return '';
  
  const lines = [];
  for (const [key, value] of Object.entries(data)) {
    if (value !== null && value !== undefined) {
      lines.push(`${key}: ${value}`);
    }
  }
  
  return lines.length > 0 ? `---\n${lines.join('\n')}\n---\n` : '';
}

/**
 * Check if content has frontmatter
 * **By George!** Quick reconnaissance of document structure!
 */
export function hasFrontmatter(text) {
  if (!text || typeof text !== 'string') return false;
  const trimmed = text.startsWith('\ufeff') ? text.slice(1) : text;
  return trimmed.startsWith('---\n') && trimmed.indexOf('\n---', 4) !== -1;
}

/**
 * Extract just the frontmatter data without parsing the full structure
 * **BULLY!** Quick extraction for lightweight operations!
 */
export function extractFrontmatterData(text) {
  const parsed = parseFrontmatter(text);
  return parsed.data || {};
}
