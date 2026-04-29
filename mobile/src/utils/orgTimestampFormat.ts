import dayjs from 'dayjs';
import customParseFormat from 'dayjs/plugin/customParseFormat';

dayjs.extend(customParseFormat);

function unescapeHtml(s: string): string {
  return s
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&amp;/gi, '&')
    .trim();
}

/** Strip org active/inactive timestamp wrappers. */
function stripOrgWrappers(s: string): string {
  let t = unescapeHtml(s).trim();
  t = t.replace(/^[\[<]+/, '').replace(/[\]>]+$/, '').trim();
  return t;
}

/**
 * Turn org-parse SCHEDULED/DEADLINE string (e.g. "<2026-04-28 Tue>" or with time) into a short display string.
 */
export function formatOrgPlanningTimestamp(raw: string | null | undefined): string | null {
  if (!raw?.trim()) return null;
  const inner = stripOrgWrappers(raw);
  const dateMatch = inner.match(/^(\d{4}-\d{2}-\d{2})(\s+.+)?$/);
  if (!dateMatch) {
    return inner.length > 56 ? `${inner.slice(0, 53)}…` : inner;
  }
  const ymd = dateMatch[1];
  const tail = (dateMatch[2] || '').trim();
  const d = dayjs(ymd, 'YYYY-MM-DD', true);
  if (!d.isValid()) {
    return inner.length > 56 ? `${inner.slice(0, 53)}…` : inner;
  }
  const dateLabel = d.format('ddd, MMM D, YYYY');
  const timeMatch = tail.match(/\b(\d{1,2}:\d{2}(?::\d{2})?)\b/);
  if (timeMatch) {
    const t = dayjs(`${ymd} ${timeMatch[1]}`, ['YYYY-MM-DD H:mm', 'YYYY-MM-DD H:mm:ss'], true);
    if (t.isValid()) {
      return `${dateLabel} · ${t.format('h:mm A')}`;
    }
    return `${dateLabel} · ${timeMatch[1]}`;
  }
  return dateLabel;
}
