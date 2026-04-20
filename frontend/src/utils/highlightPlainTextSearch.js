import React from 'react';

/**
 * Split plain text into React nodes with <mark> for case-insensitive matches.
 * First match gets data-chat-scroll-target="1" for scroll-into-view.
 */
export function highlightPlainTextSearch(text, query) {
  if (text == null || text === '') return text;
  const q = query != null ? String(query).trim() : '';
  if (!q) return text;

  const str = String(text);
  const escaped = q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(`(${escaped})`, 'gi');
  const matches = [...str.matchAll(re)];
  if (matches.length === 0) return str;

  const out = [];
  let last = 0;
  let key = 0;
  let first = true;
  for (const m of matches) {
    const start = m.index;
    if (start > last) {
      out.push(<React.Fragment key={`t-${key++}`}>{str.slice(last, start)}</React.Fragment>);
    }
    out.push(
      <mark
        key={`m-${key++}`}
        className="chat-in-thread-search-mark"
        {...(first ? { 'data-chat-scroll-target': '1' } : {})}
      >
        {m[0]}
      </mark>
    );
    first = false;
    last = start + m[0].length;
  }
  if (last < str.length) {
    out.push(<React.Fragment key={`t-${key++}`}>{str.slice(last)}</React.Fragment>);
  }
  return out;
}
