/**
 * RSS Service for Frontend
 * API calls and state management for RSS feed operations
 */

import apiService from './apiService';

/** Notify listeners (e.g. React Query) that per-feed unread counts changed on the server. */
function notifyRssUnreadCountsChanged() {
    if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('rss-unread-counts-updated'));
    }
}

class RSSService {
    constructor() {
        this.baseUrl = '/api/rss';
        this.feeds = [];
        this.unreadCounts = {};
        this.currentFeed = null;
        this.articles = [];
        this.loading = false;
        this.error = null;
    }

    // RSS Feed Management
    async getFeeds() {
        try {
            this.loading = true;
            this.error = null;
            
            const response = await apiService.get(`${this.baseUrl}/feeds`);
            console.log('🔍 RSS API Response:', response);
            console.log('🔍 RSS Feeds Data:', response);
            this.feeds = response;
            
            // Get unread counts for all feeds
            await this.getUnreadCounts();
            
            return this.feeds;
        } catch (error) {
            this.error = error.message;
            console.error('❌ RSS SERVICE ERROR: Failed to get feeds:', error);
            throw error;
        } finally {
            this.loading = false;
        }
    }

    async getCategorizedFeeds() {
        try {
            this.loading = true;
            this.error = null;
            
            const response = await apiService.get(`${this.baseUrl}/feeds/categorized`);
            console.log('🔍 RSS Categorized API Response:', response);
            
            return response;
        } catch (error) {
            this.error = error.message;
            console.error('❌ RSS SERVICE ERROR: Failed to get categorized feeds:', error);
            throw error;
        } finally {
            this.loading = false;
        }
    }

    async createFeed(feedData, isGlobal = false) {
        try {
            this.loading = true;
            this.error = null;
            
            // Choose the appropriate endpoint based on whether it's a global feed
            const endpoint = isGlobal ? `${this.baseUrl}/feeds/global` : `${this.baseUrl}/feeds`;
            const response = await apiService.post(endpoint, feedData);
            const newFeed = response;
            
            // Ensure feeds array is initialized
            if (!this.feeds) {
                this.feeds = [];
            }
            
            // Add to feeds list
            this.feeds.unshift(newFeed);
            
            // Refresh unread counts
            await this.getUnreadCounts();

            // Backend enqueues an initial Celery poll for new feeds; watch for articles to land.
            try {
                await this.watchFeedRefresh(newFeed.feed_id);
            } catch (e) {
                console.warn('RSS watcher after create timed out or failed:', e);
            }
            
            return newFeed;
        } catch (error) {
            this.error = error.message;
            console.error('❌ RSS SERVICE ERROR: Failed to create feed:', error);
            throw error;
        } finally {
            this.loading = false;
        }
    }

    /**
     * Validate RSS feed URL and return preview metadata (authenticated).
     */
    async validateFeedUrl(feedUrl) {
        const q = new URLSearchParams({ feed_url: feedUrl });
        return apiService.get(`${this.baseUrl}/feeds/validate?${q.toString()}`);
    }

    async deleteFeed(feedId, deleteArticles = false) {
        try {
            this.loading = true;
            this.error = null;
            
            if (deleteArticles) {
                // First get all articles for this feed
                const articles = await this.getFeedArticles(feedId, 1000);
                
                // Delete all articles
                for (const article of articles) {
                    await this.deleteArticle(article.article_id);
                }
            }
            
            // Delete the feed
            await apiService.delete(`${this.baseUrl}/feeds/${feedId}`);
            
            // Ensure feeds array is initialized
            if (!this.feeds) {
                this.feeds = [];
            }
            
            // Remove from feeds list
            this.feeds = this.feeds.filter(feed => feed.feed_id !== feedId);
            
            // Refresh unread counts
            await this.getUnreadCounts();
            
            return true;
        } catch (error) {
            this.error = error.message;
            console.error('❌ RSS SERVICE ERROR: Failed to delete feed:', error);
            throw error;
        } finally {
            this.loading = false;
        }
    }

    async toggleFeedActive(feedId, isActive) {
        const feed = (this.feeds || []).find((f) => f.feed_id === feedId);
        if (!feed) {
            throw new Error('Feed not found in local cache; refresh the list');
        }
        return this.updateFeedMetadata(feedId, {
            feed_url: feed.feed_url,
            feed_name: feed.feed_name,
            category: feed.category,
            tags: Array.isArray(feed.tags) ? feed.tags : [],
            check_interval: feed.check_interval,
            is_active: isActive,
        });
    }

    async updateFeedMetadata(feedId, feedData) {
        try {
            this.loading = true;
            this.error = null;
            const response = await apiService.put(`${this.baseUrl}/feeds/${feedId}`, feedData);
            // Update local cache
            this.feeds = (this.feeds || []).map(f => f.feed_id === feedId ? response : f);
            return response;
        } catch (error) {
            this.error = error.message;
            console.error('❌ RSS SERVICE ERROR: Failed to update feed metadata:', error);
            throw error;
        } finally {
            this.loading = false;
        }
    }

    async refreshFeed(feedId) {
        try {
            this.loading = true;
            this.error = null;
            
            const response = await apiService.post(`${this.baseUrl}/feeds/${feedId}/poll`, {
                force_poll: true
            });
            
            // Update feed's last_check time
            const feed = this.feeds.find(f => f.feed_id === feedId);
            if (feed) {
                feed.last_check = new Date().toISOString();
            }
            
            // Refresh unread counts
            await this.getUnreadCounts();

            // Watch for new articles to arrive and dispatch a refresh-complete event
            try {
                await this.watchFeedRefresh(feedId);
            } catch (e) {
                console.warn('RSS watcher after manual refresh timed out or failed:', e);
            }
            
            return response;
        } catch (error) {
            this.error = error.message;
            console.error('❌ RSS SERVICE ERROR: Failed to refresh feed:', error);
            throw error;
        } finally {
            this.loading = false;
        }
    }

    // RSS Article Management
    /**
     * @param {string} feedId
     * @param {number} [limit]
     * @param {{ readFilter?: 'all'|'unread'|'read' }} [options]
     */
    async getFeedArticles(feedId, limit = 100, options = {}) {
        try {
            this.loading = true;
            this.error = null;

            const q = new URLSearchParams({ limit: String(limit) });
            const rf = options.readFilter;
            if (rf && rf !== 'all') {
                q.set('read_filter', rf);
            }
            const response = await apiService.get(
                `${this.baseUrl}/feeds/${encodeURIComponent(feedId)}/articles?${q.toString()}`
            );
            
            this.articles = response;
            this.currentFeed = this.feeds.find(f => f.feed_id === feedId);
            
            return this.articles;
        } catch (error) {
            this.error = error.message;
            console.error('❌ RSS SERVICE ERROR: Failed to get feed articles:', error);
            throw error;
        } finally {
            this.loading = false;
        }
    }

    /**
     * Watch for feed refresh completion by polling unread counts and article list briefly.
     * If it detects a change, dispatch a window event that interested components can listen for.
     */
    async watchFeedRefresh(feedId, { timeoutMs = 10000, intervalMs = 1000 } = {}) {
        try {
            const start = Date.now();
            // Snapshot current counts
            const beforeCounts = { ...(this.unreadCounts || {}) };
            const beforeUnread = beforeCounts[feedId] || 0;

            // Also snapshot current first article id if we have it cached
            let beforeTopArticleId = null;
            if (Array.isArray(this.articles) && this.currentFeed && this.currentFeed.feed_id === feedId && this.articles.length > 0) {
                beforeTopArticleId = this.articles[0]?.article_id || null;
            }

            while (Date.now() - start < timeoutMs) {
                // Pull latest unread counts
                await this.getUnreadCounts();
                const afterUnread = this.unreadCounts[feedId] || 0;

                // Quick signal: if unread increased, we likely have new items
                if (afterUnread > beforeUnread) {
                    window.dispatchEvent(new CustomEvent('rss-feed-refresh-complete', { detail: { feedId } }));
                    return true;
                }

                // As a fallback, fetch a small page of articles to compare top id
                try {
                    const recent = await apiService.get(
                        `${this.baseUrl}/feeds/${encodeURIComponent(feedId)}/articles?limit=1`
                    );
                    const topId = Array.isArray(recent) && recent.length > 0 ? recent[0].article_id : null;
                    if (topId && topId !== beforeTopArticleId) {
                        window.dispatchEvent(new CustomEvent('rss-feed-refresh-complete', { detail: { feedId } }));
                        return true;
                    }
                } catch (_) {
                    // ignore per-iteration fetch errors
                }

                await new Promise(r => setTimeout(r, intervalMs));
            }
            // Timeout without change is not fatal
            return false;
        } catch (e) {
            // Non-fatal watcher error
            return false;
        }
    }

    async toggleArticleStar(articleId) {
        try {
            const response = await apiService.put(
                `${this.baseUrl}/articles/${encodeURIComponent(articleId)}/star`
            );
            const isStarred = response?.is_starred === true;
            const article = this.articles.find((a) => a.article_id === articleId);
            if (article) {
                article.is_starred = isStarred;
            }
            return response;
        } catch (error) {
            console.error('❌ RSS SERVICE ERROR: Failed to toggle article star:', error);
            throw error;
        }
    }

    async markArticleRead(articleId) {
        try {
            await apiService.put(`${this.baseUrl}/articles/${articleId}/read`);
            
            // Update local state
            const article = this.articles.find(a => a.article_id === articleId);
            if (article) {
                article.is_read = true;
            }
            
            // Refresh unread counts
            await this.getUnreadCounts();
            notifyRssUnreadCountsChanged();

            return true;
        } catch (error) {
            console.error('❌ RSS SERVICE ERROR: Failed to mark article read:', error);
            throw error;
        }
    }

    async markAllArticlesRead(feedId) {
        try {
            this.loading = true;
            this.error = null;

            const response = await apiService.post(
                `${this.baseUrl}/feeds/${encodeURIComponent(feedId)}/mark-all-read`
            );
            await this.getUnreadCounts();
            notifyRssUnreadCountsChanged();
            return response?.count ?? 0;
        } catch (error) {
            this.error = error.message;
            console.error('❌ RSS SERVICE ERROR: Failed to mark all articles read:', error);
            throw error;
        } finally {
            this.loading = false;
        }
    }

    async deleteArticle(articleId) {
        try {
            await apiService.delete(`${this.baseUrl}/articles/${articleId}`);
            
            // Remove from local state
            this.articles = this.articles.filter(a => a.article_id !== articleId);
            
            // Refresh unread counts
            await this.getUnreadCounts();
            notifyRssUnreadCountsChanged();

            return true;
        } catch (error) {
            console.error('❌ RSS SERVICE ERROR: Failed to delete article:', error);
            throw error;
        }
    }

    async deleteAllReadArticles(feedId) {
        try {
            this.loading = true;
            this.error = null;

            const response = await apiService.delete(
                `${this.baseUrl}/feeds/${encodeURIComponent(feedId)}/read-articles`
            );
            await this.getUnreadCounts();
            notifyRssUnreadCountsChanged();
            return response?.count ?? 0;
        } catch (error) {
            this.error = error.message;
            console.error('❌ RSS SERVICE ERROR: Failed to delete all read articles:', error);
            throw error;
        } finally {
            this.loading = false;
        }
    }

    async getImportLocation() {
        return apiService.get(`${this.baseUrl}/settings/import-location`);
    }

    async setImportLocation(folderId) {
        return apiService.put(`${this.baseUrl}/settings/import-location`, {
            folder_id: folderId || null,
        });
    }

    async importArticle(articleId, collectionName = null, targetFolderId = undefined) {
        try {
            this.loading = true;
            this.error = null;

            const payload = {
                article_id: articleId,
                collection_name: collectionName,
                user_id: 'current_user',
            };
            if (targetFolderId !== undefined && targetFolderId !== null && targetFolderId !== '') {
                payload.target_folder_id = targetFolderId;
            }

            const response = await apiService.post(`${this.baseUrl}/articles/${articleId}/import`, payload);
            
            // Update local state
            const article = this.articles.find(a => a.article_id === articleId);
            if (article) {
                article.is_processed = true;
            }
            
            return response;
        } catch (error) {
            this.error = error.message;
            console.error('❌ RSS SERVICE ERROR: Failed to import article:', error);
            throw error;
        } finally {
            this.loading = false;
        }
    }

    // Utility Methods
    async getUnreadCounts() {
        try {
            const response = await apiService.get(`${this.baseUrl}/unread-count`);
            this.unreadCounts = response;
            return this.unreadCounts;
        } catch (error) {
            console.error('❌ RSS SERVICE ERROR: Failed to get unread counts:', error);
            this.unreadCounts = {};
            return {};
        }
    }

    getUnreadCountForFeed(feedId) {
        return this.unreadCounts[feedId] || 0;
    }

    getTotalUnreadCount() {
        return Object.values(this.unreadCounts).reduce((sum, count) => sum + count, 0);
    }

    // Article Filtering and Sorting
    filterArticles(articles, filter = 'unread') {
        const isUnread = (article) => article.is_read !== true;
        switch (filter) {
            case 'unread':
                return articles.filter(isUnread);
            case 'all':
                return articles;
            case 'imported':
                return articles.filter(article => article.is_processed);
            case 'starred':
                return articles.filter((article) => article.is_starred === true);
            default:
                return articles.filter(isUnread);
        }
    }

    sortArticles(articles, sortBy = 'newest') {
        const sorted = [...articles];
        
        switch (sortBy) {
            case 'newest':
                return sorted.sort((a, b) => {
                    const dateA = new Date(a.published_date || a.created_at);
                    const dateB = new Date(b.published_date || b.created_at);
                    return dateB - dateA;
                });
            case 'oldest':
                return sorted.sort((a, b) => {
                    const dateA = new Date(a.published_date || a.created_at);
                    const dateB = new Date(b.published_date || b.created_at);
                    return dateA - dateB;
                });
            case 'title-az':
                return sorted.sort((a, b) => a.title.localeCompare(b.title));
            case 'title-za':
                return sorted.sort((a, b) => b.title.localeCompare(a.title));
            default:
                return sorted;
        }
    }

    // State Management
    getState() {
        return {
            feeds: this.feeds,
            currentFeed: this.currentFeed,
            articles: this.articles,
            unreadCounts: this.unreadCounts,
            loading: this.loading,
            error: this.error
        };
    }

    clearError() {
        this.error = null;
    }

    reset() {
        this.feeds = [];
        this.currentFeed = null;
        this.articles = [];
        this.unreadCounts = {};
        this.loading = false;
        this.error = null;
    }
}

// Create singleton instance
const rssService = new RSSService();
export default rssService;
