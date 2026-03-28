/**
 * Strip markdown / org syntax and normalize text for speech synthesis.
 */

const decodeHtmlEntities = (text) => {
  if (!text || typeof text !== 'string') return '';
  let s = text;
  s = s.replace(/&nbsp;/gi, ' ');
  s = s.replace(/&#x([0-9a-fA-F]{1,6});/gi, (_, hex) => {
    const code = parseInt(hex, 16);
    return code >= 0 && code < 0x110000 ? String.fromCodePoint(code) : '';
  });
  s = s.replace(/&#(\d{1,7});/g, (_, dec) => {
    const code = parseInt(dec, 10);
    return code >= 0 && code < 0x110000 ? String.fromCodePoint(code) : '';
  });
  s = s.replace(/&quot;/gi, '"');
  s = s.replace(/&apos;/gi, "'");
  s = s.replace(/&lt;/gi, '<');
  s = s.replace(/&gt;/gi, '>');
  s = s.replace(/&amp;/gi, '&');
  return s;
};

const stripHtmlTags = (text) =>
  String(text || '')
    .replace(/<br\s*\/?>/gi, ' ')
    .replace(/<\/(p|div|h[1-6]|li|tr|blockquote)>/gi, '. ')
    .replace(/<[^>]+>/g, ' ');

/** YAML frontmatter at document start: --- ... --- */
const stripYamlFrontmatter = (input) => {
  const t = input.trimStart();
  if (!t.startsWith('---')) return input;
  const afterFirst = t.slice(3);
  const nl = afterFirst[0] === '\r' ? (afterFirst[1] === '\n' ? 2 : 1) : afterFirst[0] === '\n' ? 1 : 0;
  if (nl === 0) return input;
  const bodyStart = 3 + nl;
  const rest = t.slice(bodyStart);
  const close = rest.search(/\n---\s*(?:\n|$)/);
  if (close === -1) return input;
  const afterFm = rest.slice(close).replace(/^\n---\s*/, '');
  return afterFm.trimStart();
};

/** Table rows: keep cell text, drop pipes; drop alignment rows */
const stripMarkdownTables = (text) =>
  text
    .replace(/^\|[\s\-:|]+\|$/gm, ' ')
    .replace(/^\|[^\n]+\|$/gm, (line) =>
      line
        .split('|')
        .map((c) => c.trim())
        .filter(Boolean)
        .join(' ')
    );

const normalizeWhitespace = (text) =>
  text
    .replace(/\n{2,}/g, '. ')
    .replace(/\s{2,}/g, ' ')
    .trim();

export const stripMarkdownForSpeech = (input) => {
  if (!input || typeof input !== 'string') return '';

  let s = stripYamlFrontmatter(input);
  s = stripHtmlTags(s);
  s = decodeHtmlEntities(s);
  s = stripMarkdownTables(s);
  s = s
    .replace(/^\s*[-*_]{3,}\s*$/gm, ' ')
    .replace(/\[\^[^\]]*\]/g, ' ')
    .replace(/https?:\/\/[^\s)\]>'"<]+/gi, (url) => {
      try {
        const u = new URL(url.replace(/[.,;:!?)]+$/, ''));
        return u.hostname.replace(/^www\./i, '');
      } catch {
        return '';
      }
    })
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/!\[[^\]]*\]\([^)]+\)/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/^\s*\d+\.\s+/gm, '')
    .replace(/^>\s?/gm, '')
    .replace(/[*_~>|]/g, ' ');

  return normalizeWhitespace(s);
};

/** Org-mode: PROPERTIES drawers and pipe tables */
const stripOrgTables = (text) =>
  text
    .replace(/^\|[\-+]+$/gm, ' ')
    .replace(/^\|[^\n]+\|$/gm, (line) =>
      line
        .split('|')
        .map((c) => c.trim())
        .filter(Boolean)
        .join(' ')
    );

export const stripOrgForSpeech = (input) => {
  if (!input || typeof input !== 'string') return '';

  let s = input
    .replace(/:PROPERTIES:\s*[\s\S]*?:END:/gi, ' ')
    .replace(/^#\+BEGIN_[\s\S]*?#\+END_[^\n]*$/gim, ' ')
    .replace(/^#\+[A-Z_]+:\s*/gim, '')
    .replace(/^\*+\s+(TODO|NEXT|STARTED|WAITING|HOLD|DONE|CANCELED|CANCELLED)?\s*/gim, '')
    .replace(/\[\[([^\]]+)\]\[([^\]]+)\]\]/g, '$2')
    .replace(/\[\[([^\]]+)\]\]/g, '$1')
    .replace(/^\s*[-+]\s+/gm, '')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/\/(.*?)\//g, '$1')
    .replace(/=(.*?)=/g, '$1')
    .replace(/~(.*?)~/g, '$1');

  s = stripHtmlTags(s);
  s = decodeHtmlEntities(s);
  s = stripOrgTables(s);
  s = s
    .replace(/^\s*[-*_]{3,}\s*$/gm, ' ')
    .replace(/\[\^[^\]]*\]/g, ' ')
    .replace(/https?:\/\/[^\s)\]>'"<]+/gi, (url) => {
      try {
        const u = new URL(url.replace(/[.,;:!?)]+$/, ''));
        return u.hostname.replace(/^www\./i, '');
      } catch {
        return '';
      }
    })
    .replace(/^\s*\d+\.\s+/gm, '');

  return normalizeWhitespace(s);
};

export const stripTextForSpeech = (input, mode = 'markdown') => {
  if (mode === 'org') return stripOrgForSpeech(input);
  return stripMarkdownForSpeech(input);
};
