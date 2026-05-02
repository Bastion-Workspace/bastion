/**
 * Markdown to Word (docx) helpers for chat export.
 * Block + inline parsing without extra dependencies.
 */

import {
  Document,
  Paragraph,
  TextRun,
  HeadingLevel,
  AlignmentType,
  ExternalHyperlink,
  Table,
  TableRow,
  TableCell,
  WidthType,
  BorderStyle,
  PageOrientation,
  LevelFormat,
  convertInchesToTwip,
  ShadingType,
  UnderlineType,
} from 'docx';

const PAGE_SIZES = {
  letter: { width: 12240, height: 15840 },
  a4: { width: 11906, height: 16838 },
};

const DEFAULT_MARGIN_TWIP = convertInchesToTwip(1);

const HEADING_MAP = {
  1: HeadingLevel.HEADING_1,
  2: HeadingLevel.HEADING_2,
  3: HeadingLevel.HEADING_3,
  4: HeadingLevel.HEADING_4,
  5: HeadingLevel.HEADING_5,
  6: HeadingLevel.HEADING_6,
};

const BULLET_REF = 'export-bullet';
const ORDERED_REF = 'export-ordered';

/** @typedef {{ fontFamily?: string; fontSizePt?: number; orientation?: 'portrait'|'landscape'; paper?: 'letter'|'a4' }} DocxPageOptions */

/**
 * @returns {DocxPageOptions}
 */
export function getDefaultDocxExportOptions() {
  return {
    fontFamily: 'Calibri',
    fontSizePt: 11,
    orientation: 'portrait',
    paper: 'letter',
  };
}

/**
 * @param {DocxPageOptions} opts
 * @returns {{ font: string; size: number }}
 */
function resolveRunBase(opts) {
  const font = opts?.fontFamily || 'Calibri';
  const pt = opts?.fontSizePt ?? 11;
  return { font, size: Math.round(pt * 2) };
}

/**
 * @param {string} text
 * @param {{ font: string; size: number }} base
 * @param {{ bold?: boolean; italics?: boolean; strike?: boolean }} [mods]
 * @returns {import('docx').ParagraphChild[]}
 */
function parseInlineRuns(text, base, mods = {}) {
  if (!text) return [];
  const { font, size } = base;
  const out = [];
  let i = 0;

  const textRun = (chunk, extra = {}) => {
    if (!chunk) return;
    out.push(
      new TextRun({
        text: chunk,
        font,
        size,
        bold: mods.bold,
        italics: mods.italics,
        strike: mods.strike,
        ...extra,
      }),
    );
  };

  while (i < text.length) {
    const ch = text[i];

    if (ch === '[') {
      const rb = text.indexOf(']', i + 1);
      if (rb !== -1 && text[rb + 1] === '(') {
        const rp = text.indexOf(')', rb + 2);
        if (rp !== -1) {
          const label = text.slice(i + 1, rb);
          const href = text.slice(rb + 2, rp);
          if (/^(https?:\/\/|mailto:)/i.test(href)) {
            out.push(
              new ExternalHyperlink({
                children: [
                  new TextRun({
                    text: label,
                    style: 'Hyperlink',
                    font,
                    size,
                    color: '0563C1',
                    underline: { type: UnderlineType.SINGLE },
                    bold: mods.bold,
                    italics: mods.italics,
                    strike: mods.strike,
                  }),
                ],
                link: href,
              }),
            );
            i = rp + 1;
            continue;
          }
        }
      }
    }

    if (ch === '`') {
      const end = text.indexOf('`', i + 1);
      if (end !== -1) {
        const code = text.slice(i + 1, end);
        out.push(
          new TextRun({
            text: code,
            font: 'Courier New',
            size,
            shading: { fill: 'F0F0F0', type: ShadingType.CLEAR },
            bold: mods.bold,
            italics: mods.italics,
            strike: mods.strike,
          }),
        );
        i = end + 1;
        continue;
      }
    }

    if (text.startsWith('~~', i)) {
      const end = text.indexOf('~~', i + 2);
      if (end !== -1) {
        const inner = text.slice(i + 2, end);
        out.push(...parseInlineRuns(inner, base, { ...mods, strike: true }));
        i = end + 2;
        continue;
      }
    }

    if (text.startsWith('***', i)) {
      const end = text.indexOf('***', i + 3);
      if (end !== -1) {
        const inner = text.slice(i + 3, end);
        out.push(...parseInlineRuns(inner, base, { ...mods, bold: true, italics: true }));
        i = end + 3;
        continue;
      }
    }

    if (text.startsWith('**', i)) {
      const end = text.indexOf('**', i + 2);
      if (end !== -1) {
        const inner = text.slice(i + 2, end);
        out.push(...parseInlineRuns(inner, base, { ...mods, bold: true }));
        i = end + 2;
        continue;
      }
    }

    if (text.startsWith('__', i)) {
      const end = text.indexOf('__', i + 2);
      if (end !== -1) {
        const inner = text.slice(i + 2, end);
        out.push(...parseInlineRuns(inner, base, { ...mods, bold: true }));
        i = end + 2;
        continue;
      }
    }

    if (ch === '*' && text[i + 1] !== '*') {
      const end = text.indexOf('*', i + 1);
      if (end !== -1) {
        const inner = text.slice(i + 1, end);
        out.push(...parseInlineRuns(inner, base, { ...mods, italics: true }));
        i = end + 1;
        continue;
      }
    }

    if (ch === '_' && text[i + 1] !== '_') {
      const end = text.indexOf('_', i + 1);
      if (end !== -1) {
        const inner = text.slice(i + 1, end);
        out.push(...parseInlineRuns(inner, base, { ...mods, italics: true }));
        i = end + 1;
        continue;
      }
    }

    let j = i + 1;
    while (j < text.length) {
      const c = text[j];
      if (
        c === '[' ||
        c === '`' ||
        c === '*' ||
        c === '_' ||
        (c === '~' && text[j + 1] === '~')
      ) {
        break;
      }
      j += 1;
    }
    textRun(text.slice(i, j));
    i = j;
  }

  return out.length ? out : [new TextRun({ text: '', font, size, ...mods })];
}

/**
 * @param {string} line
 */
function isTableDivider(line) {
  const t = line.trim();
  if (!t.includes('|')) return false;
  const cells = t.split('|').map((c) => c.trim()).filter(Boolean);
  if (!cells.length) return false;
  return cells.every((c) => /^:?-{2,}:?$/.test(c));
}

/**
 * @param {string} line
 */
function splitTableRow(line) {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((c) => c.trim());
}

/**
 * @param {string} markdown
 * @param {DocxPageOptions} [options]
 * @returns {{ children: import('docx').Paragraph[] | import('docx').Table[] }}
 */
export function markdownToDocxChildren(markdown, options = {}) {
  const runBase = resolveRunBase(options);
  const lines = (markdown || '').replace(/\r\n/g, '\n').split('\n');
  /** @type {(import('docx').Paragraph | import('docx').Table)[]} */
  const children = [];
  let i = 0;
  let usesLists = false;

  const paraFromLine = (line, extra = {}) => {
    const trimmed = line.trim();
    if (!trimmed) {
      children.push(new Paragraph({ children: [new TextRun({ text: '', ...runBase })] }));
      return;
    }
    const quote = /^>\s?/.test(trimmed);
    const content = quote ? trimmed.replace(/^>\s?/, '') : trimmed;
    const runs = parseInlineRuns(content, runBase);
    children.push(
      new Paragraph({
        children: runs,
        indent: quote ? { left: convertInchesToTwip(0.35) } : undefined,
        spacing: { after: 120 },
        ...extra,
      }),
    );
  };

  while (i < lines.length) {
    const raw = lines[i];
    const line = raw;

    if (line.trim() === '') {
      i += 1;
      continue;
    }

    if (line.trim().startsWith('```')) {
      const fence = line.trim();
      const lang = fence.slice(3).trim();
      i += 1;
      const buf = [];
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        buf.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) i += 1;
      const codeText = buf.join('\n');
      const codeRuns = [];
      const parts = codeText.split('\n');
      parts.forEach((pl, idx) => {
        codeRuns.push(
          new TextRun({
            text: pl,
            font: 'Courier New',
            size: runBase.size,
            shading: { fill: 'F5F5F5', type: ShadingType.CLEAR },
          }),
        );
        if (idx < parts.length - 1) {
          codeRuns.push(new TextRun({ break: 1, font: 'Courier New', size: runBase.size }));
        }
      });
      children.push(
        new Paragraph({
          style: 'CodeBlock',
          shading: { fill: 'F5F5F5', type: ShadingType.CLEAR },
          border: {
            top: { style: BorderStyle.SINGLE, size: 4, color: 'CCCCCC' },
            bottom: { style: BorderStyle.SINGLE, size: 4, color: 'CCCCCC' },
            left: { style: BorderStyle.SINGLE, size: 4, color: 'CCCCCC' },
            right: { style: BorderStyle.SINGLE, size: 4, color: 'CCCCCC' },
          },
          spacing: { before: 120, after: 120 },
          children: codeRuns.length ? codeRuns : [new TextRun({ text: lang ? `(${lang})` : '', font: 'Courier New', size: runBase.size })],
        }),
      );
      continue;
    }

    const atx = /^(#{1,6})\s+(.*)$/.exec(line.trim());
    if (atx) {
      const level = atx[1].length;
      const body = atx[2].trim();
      children.push(
        new Paragraph({
          heading: HEADING_MAP[level] || HeadingLevel.HEADING_6,
          spacing: { before: level <= 2 ? 240 : 160, after: 120 },
          children: parseInlineRuns(body, runBase),
        }),
      );
      i += 1;
      continue;
    }

    if (i + 1 < lines.length) {
      const next = lines[i + 1].trim();
      if (/^=+$/.test(next) && line.trim()) {
        children.push(
          new Paragraph({
            heading: HeadingLevel.HEADING_1,
            spacing: { before: 240, after: 120 },
            children: parseInlineRuns(line.trim(), runBase),
          }),
        );
        i += 2;
        continue;
      }
      if (/^-+$/.test(next) && line.trim()) {
        children.push(
          new Paragraph({
            heading: HeadingLevel.HEADING_2,
            spacing: { before: 200, after: 120 },
            children: parseInlineRuns(line.trim(), runBase),
          }),
        );
        i += 2;
        continue;
      }
    }

    if (/^(-{3,}|\*{3,}|_{3,})\s*$/.test(line.trim())) {
      children.push(
        new Paragraph({
          border: {
            bottom: { color: '999999', space: 1, style: BorderStyle.SINGLE, size: 6 },
          },
          spacing: { after: 200, before: 120 },
          children: [new TextRun({ text: '', ...runBase })],
        }),
      );
      i += 1;
      continue;
    }

    const ul = /^(\s*)([-*+])\s+(.*)$/.exec(line);
    if (ul) {
      usesLists = true;
      children.push(
        new Paragraph({
          numbering: { reference: BULLET_REF, level: 0 },
          spacing: { after: 80 },
          children: parseInlineRuns(ul[3], runBase),
        }),
      );
      i += 1;
      continue;
    }

    const ol = /^(\s*)(\d+)\.\s+(.*)$/.exec(line);
    if (ol) {
      usesLists = true;
      children.push(
        new Paragraph({
          numbering: { reference: ORDERED_REF, level: 0 },
          spacing: { after: 80 },
          children: parseInlineRuns(ol[3], runBase),
        }),
      );
      i += 1;
      continue;
    }

    if (line.includes('|') && i + 1 < lines.length && isTableDivider(lines[i + 1])) {
      const headerCells = splitTableRow(line);
      i += 2;
      const bodyRows = [];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
        bodyRows.push(splitTableRow(lines[i]));
        i += 1;
      }
      const colCount = headerCells.length;
      const colWidth = Math.floor(9000 / Math.max(colCount, 1));
      const makeRow = (cells) =>
        new TableRow({
          children: cells.map(
            (cell) =>
              new TableCell({
                children: [
                  new Paragraph({
                    children: parseInlineRuns(cell, runBase),
                  }),
                ],
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
              }),
          ),
        });
      const tableRows = [makeRow(headerCells)];
      bodyRows.forEach((r) => {
        const padded = [...r];
        while (padded.length < colCount) padded.push('');
        tableRows.push(makeRow(padded.slice(0, colCount)));
      });
      children.push(
        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          columnWidths: Array(colCount).fill(colWidth),
          rows: tableRows,
          borders: {
            top: { style: BorderStyle.SINGLE, size: 4, color: 'CCCCCC' },
            bottom: { style: BorderStyle.SINGLE, size: 4, color: 'CCCCCC' },
            left: { style: BorderStyle.SINGLE, size: 4, color: 'CCCCCC' },
            right: { style: BorderStyle.SINGLE, size: 4, color: 'CCCCCC' },
            insideHorizontal: { style: BorderStyle.SINGLE, size: 4, color: 'EEEEEE' },
            insideVertical: { style: BorderStyle.SINGLE, size: 4, color: 'EEEEEE' },
          },
        }),
      );
      continue;
    }

    const buf = [line];
    i += 1;
    while (i < lines.length && lines[i].trim() !== '') {
      const l = lines[i];
      if (
        l.trim().startsWith('```') ||
        /^(#{1,6})\s/.test(l.trim()) ||
        /^(\s*)([-*+]|\d+\.)\s+/.test(l) ||
        (l.includes('|') && i + 1 < lines.length && isTableDivider(lines[i + 1])) ||
        /^>\s?/.test(l.trim())
      ) {
        break;
      }
      buf.push(l);
      i += 1;
    }
    paraFromLine(buf.join('\n'));
  }

  return { children, usesLists };
}

const listNumberingConfig = {
  config: [
    {
      reference: BULLET_REF,
      levels: [
        {
          level: 0,
          format: LevelFormat.BULLET,
          text: '\u2022',
          alignment: AlignmentType.LEFT,
          style: {
            paragraph: {
              indent: { left: convertInchesToTwip(0.5), hanging: convertInchesToTwip(0.25) },
            },
          },
        },
      ],
    },
    {
      reference: ORDERED_REF,
      levels: [
        {
          level: 0,
          format: LevelFormat.DECIMAL,
          text: '%1.',
          alignment: AlignmentType.LEFT,
          style: {
            paragraph: {
              indent: { left: convertInchesToTwip(0.5), hanging: convertInchesToTwip(0.25) },
            },
          },
        },
      ],
    },
  ],
};

/**
 * @param {(import('docx').Paragraph | import('docx').Table)[]} bodyChildren
 * @param {boolean} usesLists
 * @param {DocxPageOptions} pageOptions
 */
export function buildDocxDocument(bodyChildren, usesLists, pageOptions = {}) {
  const paperKey = pageOptions.paper === 'a4' ? 'a4' : 'letter';
  let { width, height } = PAGE_SIZES[paperKey];
  const orientation =
    pageOptions.orientation === 'landscape' ? PageOrientation.LANDSCAPE : PageOrientation.PORTRAIT;
  if (orientation === PageOrientation.LANDSCAPE) {
    const tmp = width;
    width = height;
    height = tmp;
  }

  const runBase = resolveRunBase(pageOptions);

  return new Document({
    ...(usesLists ? { numbering: listNumberingConfig } : {}),
    styles: {
      default: {
        document: {
          run: {
            font: runBase.font,
            size: runBase.size,
          },
        },
      },
      paragraphStyles: [
        {
          id: 'CodeBlock',
          name: 'Code Block',
          basedOn: 'Normal',
          quickFormat: true,
          run: {
            font: 'Courier New',
            size: runBase.size,
          },
          paragraph: {
            spacing: { before: 60, after: 60 },
          },
        },
      ],
    },
    sections: [
      {
        properties: {
          page: {
            size: {
              width,
              height,
              orientation,
            },
            margin: {
              top: DEFAULT_MARGIN_TWIP,
              right: DEFAULT_MARGIN_TWIP,
              bottom: DEFAULT_MARGIN_TWIP,
              left: DEFAULT_MARGIN_TWIP,
              header: convertInchesToTwip(0.5),
              footer: convertInchesToTwip(0.5),
              gutter: 0,
            },
          },
        },
        children: bodyChildren,
      },
    ],
  });
}
