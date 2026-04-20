/**
 * Rehype plugin: wrap case-insensitive substring matches in <mark>.
 * Skips text inside <pre> and inline <code>. First mark may get data-chat-scroll-target for scroll-into-view.
 */
export default function rehypeHighlightChatSearch(options = {}) {
  const rawQuery = options.query;
  const addScrollTarget = options.markScrollTarget !== false;

  return (tree) => {
    const query = rawQuery != null ? String(rawQuery).trim() : '';
    if (!query) return tree;

    const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const firstMarkRef = { done: !addScrollTarget };

    function splitText(value) {
      const re = new RegExp(`(${escaped})`, 'gi');
      const matches = [...value.matchAll(re)];
      if (matches.length === 0) return [{ type: 'text', value }];

      const parts = [];
      let last = 0;
      for (const m of matches) {
        const start = m.index;
        if (start > last) {
          parts.push({ type: 'text', value: value.slice(last, start) });
        }
        const isFirst = !firstMarkRef.done;
        if (isFirst) firstMarkRef.done = true;
        parts.push({
          type: 'element',
          tagName: 'mark',
          properties: {
            className: ['chat-in-thread-search-mark'],
            ...(isFirst ? { dataChatScrollTarget: '1' } : {}),
          },
          children: [{ type: 'text', value: m[0] }],
        });
        last = start + m[0].length;
      }
      if (last < value.length) {
        parts.push({ type: 'text', value: value.slice(last) });
      }
      return parts;
    }

    function visitElement(el) {
      if (!el || el.type !== 'element') return;
      if (['pre', 'script', 'style'].includes(el.tagName)) return;

      if (el.children?.length) {
        const next = [];
        for (const child of el.children) {
          if (child.type === 'text' && child.value) {
            next.push(...splitText(child.value));
            continue;
          }
          if (child.type === 'element') {
            if (child.tagName === 'code' && el.tagName !== 'pre') {
              next.push(child);
              continue;
            }
            visitElement(child);
            next.push(child);
            continue;
          }
          next.push(child);
        }
        el.children = next;
      }
    }

    if (tree.type === 'root' && Array.isArray(tree.children)) {
      for (const child of tree.children) {
        if (child.type === 'element') visitElement(child);
      }
    }

    return tree;
  };
}
