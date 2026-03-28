/**
 * Parse, serialize, and locate GitHub-flavored Markdown pipe tables in plain text.
 */

const PIPE_LINE = /^\s*\|.*\|\s*$/;

/**
 * @param {string} line
 * @returns {boolean}
 */
export function isPipeTableLine(line) {
  const t = line.trim();
  return t.length > 0 && t.startsWith('|') && t.endsWith('|');
}

/**
 * Split a pipe row into trimmed cell strings (inner cells only).
 * @param {string} line
 * @returns {string[]|null}
 */
export function splitPipeRow(line) {
  const t = line.trim();
  if (!t.startsWith('|') || !t.endsWith('|')) return null;
  const parts = t.split('|');
  return parts.slice(1, -1).map((c) => c.trim());
}

/**
 * @param {string} cell
 * @returns {'left'|'center'|'right'}
 */
export function separatorCellAlign(cell) {
  const t = String(cell || '').trim();
  if (!/^[\-:]+$/.test(t)) return 'left';
  const left = t.startsWith(':');
  const right = t.endsWith(':');
  if (left && right && t.length > 2) return 'center';
  if (right) return 'right';
  return 'left';
}

/**
 * @param {'left'|'center'|'right'} align
 * @param {number} minInnerWidth
 * @returns {string}
 */
export function alignmentToSeparatorCell(align, minInnerWidth) {
  const w = Math.max(3, minInnerWidth);
  const dashes = '-'.repeat(w);
  if (align === 'center') return `:${dashes}:`;
  if (align === 'right') return `${dashes}:`;
  return dashes;
}

/**
 * @param {import('@codemirror/state').Text} doc
 * @param {number} cursorPos
 * @returns {{ startLine: number, endLine: number, from: number, to: number, text: string } | null}
 */
export function detectTableAtCursor(doc, cursorPos) {
  try {
    const line = doc.lineAt(cursorPos);
    if (!isPipeTableLine(line.text)) return null;

    let start = line.number;
    let end = line.number;

    while (start > 1) {
      const prev = doc.line(start - 1).text;
      if (!isPipeTableLine(prev)) break;
      start--;
    }
    while (end < doc.lines) {
      const next = doc.line(end + 1).text;
      if (!isPipeTableLine(next)) break;
      end++;
    }

    if (end - start + 1 < 2) return null;

    const lines = [];
    for (let n = start; n <= end; n++) {
      lines.push(doc.line(n).text);
    }
    const text = lines.join('\n');
    const from = doc.line(start).from;
    const to = doc.line(end).to;

    return { startLine: start, endLine: end, from, to, text };
  } catch {
    return null;
  }
}

/**
 * @param {string} text
 * @returns {{ headers: string[], alignments: ('left'|'center'|'right')[], rows: string[][] } | null}
 */
export function parseMarkdownTable(text) {
  if (!text || typeof text !== 'string') return null;
  const normalized = text.replace(/\r\n/g, '\n').trim();
  const rawLines = normalized.split('\n').filter((ln) => ln.trim().length > 0);
  if (rawLines.length < 2) return null;

  const headerCells = splitPipeRow(rawLines[0]);
  if (!headerCells || headerCells.length === 0) return null;

  const sepCells = splitPipeRow(rawLines[1]);
  let alignments;
  let dataStart = 2;

  if (sepCells && sepCells.length > 0 && sepCells.every((c) => /^[\-:]+$/.test(c.trim()))) {
    alignments = sepCells.map((c) => separatorCellAlign(c));
    while (alignments.length < headerCells.length) {
      alignments.push('left');
    }
    if (alignments.length > headerCells.length) {
      alignments = alignments.slice(0, headerCells.length);
    }
  } else {
    alignments = headerCells.map(() => 'left');
    dataStart = 1;
  }

  const rows = [];
  for (let i = dataStart; i < rawLines.length; i++) {
    const cells = splitPipeRow(rawLines[i]);
    if (!cells) continue;
    const row = [...cells];
    while (row.length < headerCells.length) row.push('');
    if (row.length > headerCells.length) row.length = headerCells.length;
    rows.push(row);
  }

  return {
    headers: headerCells.map((h) => h ?? ''),
    alignments,
    rows: rows.length > 0 ? rows : [headerCells.map(() => '')],
  };
}

/**
 * Pad string to width (visual approximation; monospace-friendly).
 * @param {string} s
 * @param {number} width
 * @param {'left'|'center'|'right'} align
 */
function padCell(s, width, align) {
  const str = String(s ?? '');
  if (str.length >= width) return str;
  const pad = width - str.length;
  if (align === 'right') return ' '.repeat(pad) + str;
  if (align === 'center') {
    const left = Math.floor(pad / 2);
    const right = pad - left;
    return ' '.repeat(left) + str + ' '.repeat(right);
  }
  return str + ' '.repeat(pad);
}

/**
 * @param {{ headers: string[], alignments: ('left'|'center'|'right')[], rows: string[][] }} model
 * @returns {string}
 */
export function serializeMarkdownTable(model) {
  const headers = (model.headers || []).map((h) => String(h ?? ''));
  const alignments = (model.alignments || []).slice();
  while (alignments.length < headers.length) alignments.push('left');
  if (alignments.length > headers.length) alignments.length = headers.length;

  const rows = (model.rows || []).map((r) => {
    const row = [...(r || [])];
    while (row.length < headers.length) row.push('');
    if (row.length > headers.length) row.length = headers.length;
    return row.map((c) => String(c ?? ''));
  });

  const colCount = Math.max(1, headers.length);
  const colWidths = [];
  for (let j = 0; j < colCount; j++) {
    let w = (headers[j] || '').length;
    for (const row of rows) {
      w = Math.max(w, (row[j] || '').length);
    }
    colWidths.push(Math.max(1, w));
  }

  const sepInner = [];
  for (let j = 0; j < colCount; j++) {
    sepInner.push(alignmentToSeparatorCell(alignments[j] || 'left', colWidths[j]));
  }

  const fmtRow = (cells) =>
    `| ${cells.map((c, j) => padCell(c, colWidths[j], 'left')).join(' | ')} |`;

  const headerRow = fmtRow(headers);
  const sepRow = `| ${sepInner.join(' | ')} |`;
  const bodyRows = rows.map((r) => {
    const padded = r.map((c, j) => padCell(c, colWidths[j], alignments[j] || 'left'));
    return `| ${padded.join(' | ')} |`;
  });

  return [headerRow, sepRow, ...bodyRows].join('\n');
}

/**
 * Default empty table for insert mode.
 */
export function emptyTableModel() {
  return {
    headers: ['Column 1', 'Column 2'],
    alignments: ['left', 'left'],
    rows: [['', '']],
  };
}
