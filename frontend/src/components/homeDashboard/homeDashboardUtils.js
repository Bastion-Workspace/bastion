import rssService from '../../services/rssService';

export const WIDGET_TYPES = [
  { type: 'nav_links', label: 'Navigation links' },
  { type: 'markdown_card', label: 'Markdown note' },
  { type: 'rss_headlines', label: 'RSS headlines' },
];

export function newWidgetId() {
  return typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `w-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export function emptyWidget(type) {
  switch (type) {
    case 'nav_links':
      return { type: 'nav_links', id: newWidgetId(), config: { items: [] } };
    case 'markdown_card':
      return {
        type: 'markdown_card',
        id: newWidgetId(),
        config: { title: '', body: '' },
      };
    case 'rss_headlines':
      return {
        type: 'rss_headlines',
        id: newWidgetId(),
        config: { feed_id: null, limit: 8 },
      };
    default:
      return null;
  }
}

export async function loadRssHeadlines(feedId, limit) {
  if (feedId) {
    return rssService.getFeedArticles(feedId, limit);
  }
  const feeds = await rssService.getFeeds();
  if (!feeds?.length) return [];
  const take = Math.min(feeds.length, 4);
  const per = Math.max(1, Math.ceil(limit / take));
  const batches = await Promise.all(
    feeds.slice(0, take).map((f) => rssService.getFeedArticles(f.feed_id, per))
  );
  const merged = batches.flat();
  merged.sort((a, b) => {
    const da = new Date(a.published_date || a.created_at || 0).getTime();
    const db = new Date(b.published_date || b.created_at || 0).getTime();
    return db - da;
  });
  return merged.slice(0, limit);
}

export function widgetTitle(w) {
  const found = WIDGET_TYPES.find((x) => x.type === w.type);
  return found ? found.label : w.type;
}
