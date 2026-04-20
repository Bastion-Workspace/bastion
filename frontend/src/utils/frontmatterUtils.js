/**
 * Frontmatter parsing utilities for markdown documents.
 */

/**
 * Parse YAML-like frontmatter from markdown text.
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

/** Markdown source for preview: body only, without leading YAML frontmatter when present. */
export function markdownPreviewBody(text) {
  return parseFrontmatter(text || '').body;
}
