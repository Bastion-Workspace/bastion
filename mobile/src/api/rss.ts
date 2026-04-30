import { apiRequest, isApiError } from './client';

export type RssFeed = {
  feed_id: string;
  feed_url: string;
  feed_name: string;
  category: string;
  tags?: string[];
  is_active?: boolean;
  user_id?: string | null;
  last_check?: string | null;
};

export type RssArticle = {
  article_id: string;
  feed_id: string;
  /** Present on cross-feed list responses (joined from rss_feeds). */
  feed_name?: string | null;
  title: string;
  description?: string | null;
  /** Plaintext or markdown-style body when processed (same as web). */
  full_content?: string | null;
  /** Sanitized HTML when available (same as web). */
  full_content_html?: string | null;
  link: string;
  published_date?: string | null;
  is_read?: boolean;
  is_starred?: boolean;
};

/** Per-feed unread counts (feed_id -> count). */
export type RssUnreadMap = Record<string, number>;

export async function listRssFeeds(): Promise<RssFeed[]> {
  return apiRequest<RssFeed[]>('/api/rss/feeds');
}

export async function getRssUnreadByFeed(): Promise<RssUnreadMap> {
  const raw = await apiRequest<RssUnreadMap | unknown>('/api/rss/unread-count');
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
    return raw as RssUnreadMap;
  }
  return {};
}

export async function listFeedArticles(
  feedId: string,
  options?: { limit?: number; readFilter?: 'all' | 'unread' | 'read' }
): Promise<RssArticle[]> {
  const limit = options?.limit ?? 100;
  const readFilter = options?.readFilter ?? 'all';
  const q = new URLSearchParams({
    limit: String(limit),
    read_filter: readFilter,
  });
  return apiRequest<RssArticle[]>(`/api/rss/feeds/${encodeURIComponent(feedId)}/articles?${q}`);
}

function articleSortKey(a: RssArticle): number {
  const d = a.published_date;
  if (!d) return 0;
  const t = Date.parse(String(d));
  return Number.isNaN(t) ? 0 : t;
}

const ALL_FEEDS_FETCH_CONCURRENCY = 8;

/** Merge per-feed lists when the unified `/api/rss/articles` route is missing (older servers). */
async function listAllArticlesViaPerFeed(
  options?: { limit?: number; readFilter?: 'all' | 'unread' | 'read' }
): Promise<RssArticle[]> {
  const limit = options?.limit ?? 200;
  const readFilter = options?.readFilter ?? 'unread';
  const feeds = await listRssFeeds();
  if (feeds.length === 0) return [];
  const feedNameById = new Map(feeds.map((f) => [f.feed_id, f.feed_name]));
  const perFeed = Math.min(120, Math.max(40, Math.ceil((limit * 2) / feeds.length)));
  const merged = new Map<string, RssArticle>();
  for (let i = 0; i < feeds.length; i += ALL_FEEDS_FETCH_CONCURRENCY) {
    const batch = feeds.slice(i, i + ALL_FEEDS_FETCH_CONCURRENCY);
    const lists = await Promise.all(
      batch.map((f) => listFeedArticles(f.feed_id, { limit: perFeed, readFilter }))
    );
    for (const list of lists) {
      for (const a of list) {
        const name = feedNameById.get(a.feed_id);
        merged.set(a.article_id, name ? { ...a, feed_name: name } : a);
      }
    }
  }
  const arr = [...merged.values()];
  arr.sort((x, y) => articleSortKey(y) - articleSortKey(x));
  return arr.slice(0, limit);
}

/** All articles across feeds visible to the user (own + global feeds). */
export async function listAllArticles(
  options?: { limit?: number; readFilter?: 'all' | 'unread' | 'read' }
): Promise<RssArticle[]> {
  const limit = options?.limit ?? 200;
  const readFilter = options?.readFilter ?? 'unread';
  const q = new URLSearchParams({
    limit: String(limit),
    read_filter: readFilter,
  });
  try {
    return await apiRequest<RssArticle[]>(`/api/rss/articles?${q}`);
  } catch (e) {
    if (isApiError(e) && e.status === 404) {
      return listAllArticlesViaPerFeed(options);
    }
    throw e;
  }
}

/** Full article row (large body fields); use when opening the in-app reader. */
export async function getRssArticle(articleId: string): Promise<RssArticle> {
  return apiRequest<RssArticle>(`/api/rss/articles/${encodeURIComponent(articleId)}`);
}

/** Mark every unread article for the current user as read (all feeds). */
export async function markAllUserRead(): Promise<number> {
  const res = await apiRequest<{ count?: number }>('/api/rss/mark-all-read', { method: 'POST' });
  return typeof res?.count === 'number' ? res.count : 0;
}

export async function markArticleRead(articleId: string): Promise<void> {
  await apiRequest(`/api/rss/articles/${encodeURIComponent(articleId)}/read`, {
    method: 'PUT',
  });
}

export async function toggleArticleStar(articleId: string): Promise<boolean> {
  const res = await apiRequest<{ is_starred?: boolean }>(
    `/api/rss/articles/${encodeURIComponent(articleId)}/star`,
    { method: 'PUT' }
  );
  return Boolean(res?.is_starred);
}

export async function markAllFeedRead(feedId: string): Promise<number> {
  const res = await apiRequest<{ count?: number }>(
    `/api/rss/feeds/${encodeURIComponent(feedId)}/mark-all-read`,
    { method: 'POST' }
  );
  return typeof res?.count === 'number' ? res.count : 0;
}
