/**
 * RSS Article Viewer Component
 * Displays RSS articles with filtering, sorting, and article actions
 */

import React, { useState, useEffect, useLayoutEffect, useMemo, useCallback, useRef } from 'react';
import { useQuery } from 'react-query';
import DOMPurify from 'dompurify';
import rssService from '../services/rssService';
import apiService from '../services/apiService';
import { formatInstantDateTime } from '../utils/userTimeDisplay';
import { useTheme } from '../contexts/ThemeContext';

function escapeForArticleIdSelector(id) {
    if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') {
        return CSS.escape(id);
    }
    return String(id).replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

const RSSArticleViewer = ({
    feedId,
    feedIds,
    viewerTitle,
    onClose,
    initialArticleId = null,
    onInitialArticleConsumed,
}) => {
    const { darkMode } = useTheme();

    const { data: userTimeFormatData } = useQuery(
        'userTimeFormat',
        () => apiService.settings.getUserTimeFormat(),
        { staleTime: 5 * 60 * 1000, refetchOnWindowFocus: false }
    );
    const { data: userTimezoneData } = useQuery(
        'userTimezone',
        () => apiService.getUserTimezone(),
        { staleTime: 5 * 60 * 1000, refetchOnWindowFocus: false }
    );
    const displayTimeFormat = userTimeFormatData?.time_format || '24h';
    const displayTimeZone = userTimezoneData?.timezone || undefined;
    const [articles, setArticles] = useState([]);
    const [filteredArticles, setFilteredArticles] = useState([]);
    const [currentFeed, setCurrentFeed] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    
    // Filter and sort state
    const [filter, setFilter] = useState('unread');
    const [sortBy, setSortBy] = useState('newest');
    
    // Bulk operations state
    const [bulkLoading, setBulkLoading] = useState(false);
    
    // Expanded descriptions state
    const [expandedDescriptions, setExpandedDescriptions] = useState(new Set());

    const articlesContainerRef = useRef(null);
    /** After collapsing a read article in Unread Only, scroll this article id to top of list viewport */
    const scrollAnchorAfterCollapseRef = useRef(null);
    const consumedInitialArticleRef = useRef(false);

    useEffect(() => {
        consumedInitialArticleRef.current = false;
    }, [initialArticleId]);

    const resolvedFeedIds = useMemo(() => {
        if (Array.isArray(feedIds) && feedIds.length > 0) return feedIds;
        if (feedId) return [feedId];
        return [];
    }, [feedId, feedIds]);

    useEffect(() => {
        if (resolvedFeedIds.length > 0) {
            loadFeedArticles();
        } else {
            setArticles([]);
            setCurrentFeed(null);
            setLoading(false);
            setError(null);
        }
    }, [resolvedFeedIds.join(',')]);

    // Live update: when background refresh completes for this feed, reload articles
    useEffect(() => {
        const handler = (e) => {
            try {
                const detail = e?.detail || {};
                if (detail.feedId && resolvedFeedIds.includes(detail.feedId)) {
                    loadFeedArticles();
                }
            } catch (_) {}
        };
        window.addEventListener('rss-feed-refresh-complete', handler);
        return () => window.removeEventListener('rss-feed-refresh-complete', handler);
    }, [resolvedFeedIds.join(',')]);

    useEffect(() => {
        let filtered = rssService.filterArticles(articles, filter);
        if (filter === 'unread' && expandedDescriptions.size > 0) {
            const seen = new Set(filtered.map((a) => a.article_id));
            for (const article of articles) {
                if (
                    article.is_read === true &&
                    expandedDescriptions.has(article.article_id) &&
                    !seen.has(article.article_id)
                ) {
                    filtered.push(article);
                    seen.add(article.article_id);
                }
            }
        }
        filtered = rssService.sortArticles(filtered, sortBy);
        setFilteredArticles(filtered);
    }, [articles, filter, sortBy, expandedDescriptions]);

    useLayoutEffect(() => {
        const anchorId = scrollAnchorAfterCollapseRef.current;
        if (!anchorId) return;
        scrollAnchorAfterCollapseRef.current = null;
        const container = articlesContainerRef.current;
        if (!container) return;
        const el = container.querySelector(
            `[data-article-id="${escapeForArticleIdSelector(anchorId)}"]`
        );
        if (el) {
            el.scrollIntoView({ block: 'start', behavior: 'smooth' });
        }
    }, [filteredArticles]);

    // Deep link: expand and scroll to initialArticleId (e.g. home dashboard headline)
    useEffect(() => {
        if (!initialArticleId || loading || articles.length === 0) return;
        const article = articles.find((a) => a.article_id === initialArticleId);
        if (!article) {
            onInitialArticleConsumed?.();
            return;
        }
        setFilter('all');
        setExpandedDescriptions((prev) => new Set(prev).add(initialArticleId));
    }, [articles, initialArticleId, loading, onInitialArticleConsumed]);

    useLayoutEffect(() => {
        if (!initialArticleId || consumedInitialArticleRef.current) return;
        const inFiltered = filteredArticles.some((a) => a.article_id === initialArticleId);
        if (!inFiltered) return;
        const el = articlesContainerRef.current?.querySelector(
            `[data-article-id="${escapeForArticleIdSelector(initialArticleId)}"]`
        );
        if (el) {
            el.scrollIntoView({ block: 'start', behavior: 'smooth' });
        }
        consumedInitialArticleRef.current = true;
        onInitialArticleConsumed?.();
    }, [filteredArticles, initialArticleId, onInitialArticleConsumed]);

    const loadFeedArticles = async () => {
        try {
            setLoading(true);
            setError(null);

            if (resolvedFeedIds.length === 0) {
                setArticles([]);
                setCurrentFeed(null);
                setLoading(false);
                return;
            }

            const batches = await Promise.all(
                resolvedFeedIds.map((id) => rssService.getFeedArticles(id, 2000))
            );
            const merged = new Map();
            for (const batch of batches) {
                if (!Array.isArray(batch)) continue;
                for (const a of batch) {
                    if (a?.article_id && !merged.has(a.article_id)) {
                        merged.set(a.article_id, a);
                    }
                }
            }
            const articlesData = Array.from(merged.values());
            articlesData.sort((a, b) => {
                const da = new Date(a.published_date || a.created_at || 0).getTime();
                const db = new Date(b.published_date || b.created_at || 0).getTime();
                return db - da;
            });
            setArticles(articlesData);

            if (resolvedFeedIds.length === 1) {
                setCurrentFeed(rssService.currentFeed);
            } else {
                setCurrentFeed({
                    feed_id: '__multi__',
                    feed_name: viewerTitle || 'RSS category',
                });
            }
        } catch (error) {
            setError(error.message || 'Failed to load articles');
            console.error('❌ RSS ARTICLE VIEWER ERROR:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleArticleAction = async (action, articleId) => {
        try {
            switch (action) {
                case 'mark-read':
                    await rssService.markArticleRead(articleId);
                    // Update local state
                    setArticles(prev => prev.map(article => 
                        article.article_id === articleId 
                            ? { ...article, is_read: true }
                            : article
                    ));
                    break;
                    
                case 'import':
                    await rssService.importArticle(articleId);
                    // Update local state
                    setArticles(prev => prev.map(article => 
                        article.article_id === articleId 
                            ? { ...article, is_processed: true }
                            : article
                    ));
                    showToast('Article imported successfully!', 'success');
                    break;
                    
                case 'delete':
                    await rssService.deleteArticle(articleId);
                    setArticles((prev) => prev.filter((article) => article.article_id !== articleId));
                    setExpandedDescriptions((prev) => {
                        const next = new Set(prev);
                        next.delete(articleId);
                        return next;
                    });
                    showToast('Article deleted successfully!', 'success');
                    break;

                case 'toggle-star': {
                    const res = await rssService.toggleArticleStar(articleId);
                    const starred = res?.is_starred === true;
                    setArticles((prev) =>
                        prev.map((article) =>
                            article.article_id === articleId
                                ? { ...article, is_starred: starred }
                                : article
                        )
                    );
                    break;
                }

                case 'expand-description': {
                    const wasExpanded = expandedDescriptions.has(articleId);
                    const target = articles.find((a) => a.article_id === articleId);

                    if (
                        wasExpanded &&
                        filter === 'unread' &&
                        target?.is_read === true &&
                        articlesContainerRef.current
                    ) {
                        const currentCard =
                            articlesContainerRef.current.querySelector(
                                `[data-article-id="${escapeForArticleIdSelector(articleId)}"]`
                            );
                        const next = currentCard?.nextElementSibling;
                        const nextId = next?.getAttribute?.('data-article-id') ?? null;
                        if (nextId) {
                            scrollAnchorAfterCollapseRef.current = nextId;
                        }
                    }

                    setExpandedDescriptions((prev) => {
                        const newSet = new Set(prev);
                        if (newSet.has(articleId)) {
                            newSet.delete(articleId);
                        } else {
                            newSet.add(articleId);
                        }
                        return newSet;
                    });
                    if (!wasExpanded) {
                        if (target && !target.is_read) {
                            await rssService.markArticleRead(articleId);
                            setArticles((prev) =>
                                prev.map((article) =>
                                    article.article_id === articleId
                                        ? { ...article, is_read: true }
                                        : article
                                )
                            );
                        }
                    }
                    break;
                }
                    

                    
                default:
                    break;
            }
        } catch (error) {
            showToast(`Failed to ${action.replace('-', ' ')} article: ${error.message}`, 'error');
        }
    };

    const handleBulkAction = async (action) => {
        let confirmMessage;
        if (action === 'mark-all-read') {
            confirmMessage = 'Mark all unread articles as read?';
        } else if (action === 'delete-all-read') {
            confirmMessage =
                'Delete all read (non-imported) articles? Starred articles will be kept.';
        }

        if (!confirmMessage || !window.confirm(confirmMessage)) return;

        try {
            setBulkLoading(true);

            if (action === 'mark-all-read') {
                for (const fid of resolvedFeedIds) {
                    await rssService.markAllArticlesRead(fid);
                }
                setArticles((prev) => prev.map((article) => ({ ...article, is_read: true })));
                showToast('All articles marked as read!', 'success');
            } else if (action === 'delete-all-read') {
                for (const fid of resolvedFeedIds) {
                    await rssService.deleteAllReadArticles(fid);
                }
                setArticles((prev) =>
                    prev.filter(
                        (article) =>
                            !(
                                article.is_read === true &&
                                !article.is_processed &&
                                !article.is_starred
                            )
                    )
                );
                showToast('All read articles deleted!', 'success');
            }
        } catch (error) {
            showToast(`Failed to ${action.replace('-', ' ')}: ${error.message}`, 'error');
        } finally {
            setBulkLoading(false);
        }
    };

    const handleTitleClick = (link) => {
        window.open(link, '_blank');
    };

    const formatArticleWhen = useCallback(
        (dateString) => {
            if (!dateString) return 'Unknown date';
            const formatted = formatInstantDateTime(dateString, {
                timeFormat: displayTimeFormat,
                timeZone: displayTimeZone,
            });
            return formatted || 'Unknown date';
        },
        [displayTimeFormat, displayTimeZone]
    );

    const stripHtmlTags = (html) => {
        if (!html) return '';
        // Create a temporary div to parse HTML and extract text content
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;
        return tempDiv.textContent || tempDiv.innerText || '';
    };

    const showToast = (message, type = 'info') => {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#4caf50' : type === 'error' ? '#f44336' : '#2196f3'};
            color: white;
            padding: 12px 20px;
            border-radius: 4px;
            z-index: 10000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        `;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            if (document.body.contains(toast)) {
                document.body.removeChild(toast);
            }
        }, 3000);
    };

    if (loading) {
        return (
            <div style={containerStyle}>
                <div style={loadingStyle}>Loading articles...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div style={containerStyle}>
                <div style={errorStyle}>
                    <h3>Error Loading Articles</h3>
                    <p>{error}</p>
                    <button onClick={loadFeedArticles} style={retryButtonStyle}>
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div style={containerStyle}>
            {/* Header */}
            <div style={headerStyle}>
                <div style={headerLeftStyle}>
                    <h2 style={titleStyle}>
                        {viewerTitle || currentFeed?.feed_name || 'RSS Feed'}
                    </h2>
                    <span style={articleCountStyle}>
                        {filteredArticles.length} articles
                    </span>
                </div>
                
                <button onClick={onClose} style={closeButtonStyle}>
                    ×
                </button>
            </div>

            {/* Controls */}
            <div style={controlsStyle}>
                <div style={filterControlsStyle}>
                    <select 
                        value={filter} 
                        onChange={(e) => setFilter(e.target.value)}
                        style={selectStyle}
                    >
                        <option value="unread">Unread Only</option>
                        <option value="all">All Articles</option>
                        <option value="imported">Imported Only</option>
                        <option value="starred">Starred</option>
                    </select>
                    
                    <select 
                        value={sortBy} 
                        onChange={(e) => setSortBy(e.target.value)}
                        style={selectStyle}
                    >
                        <option value="newest">Newest First</option>
                        <option value="oldest">Oldest First</option>
                        <option value="title-az">Title A-Z</option>
                        <option value="title-za">Title Z-A</option>
                    </select>
                </div>
                
                <div style={bulkActionsStyle}>
                    <button
                        onClick={() => handleBulkAction('mark-all-read')}
                        disabled={bulkLoading || filteredArticles.length === 0}
                        style={bulkButtonStyle}
                    >
                        {bulkLoading ? 'Processing...' : 'Mark All Read'}
                    </button>
                    <button
                        onClick={() => handleBulkAction('delete-all-read')}
                        disabled={bulkLoading}
                        style={{ ...bulkButtonStyle, backgroundColor: '#f44336' }}
                    >
                        {bulkLoading ? 'Processing...' : 'Delete All Read'}
                    </button>
                </div>
            </div>

            {/* Articles List */}
            <div ref={articlesContainerRef} style={articlesContainerStyle}>
                {filteredArticles.length === 0 ? (
                    <div style={emptyStateStyle}>
                        <p>No articles found matching the current filter.</p>
                    </div>
                ) : (
                    filteredArticles.map((article) => (
                        <ArticleCard
                            key={article.article_id}
                            article={article}
                            onAction={handleArticleAction}
                            onTitleClick={handleTitleClick}
                            formatArticleWhen={formatArticleWhen}
                            isExpanded={expandedDescriptions.has(article.article_id)}
                            darkMode={darkMode}
                        />
                    ))
                )}
            </div>
        </div>
    );
};

const looksLikeHtml = (s) => typeof s === 'string' && /<[a-z][\s\S]*>/i.test(s);

const RSS_ARTICLE_HTML_PURIFY = {
    ALLOWED_TAGS: [
        'p', 'br', 'strong', 'em', 'b', 'i', 'u', 'a', 'img', 'figure', 'figcaption',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'blockquote',
        'pre', 'code', 'table', 'thead', 'tbody', 'tr', 'th', 'td',
        'div', 'span', 'hr', 'sub', 'sup', 'mark', 'del', 'ins',
    ],
    ALLOWED_ATTR: ['href', 'src', 'alt', 'title', 'target', 'rel', 'loading', 'srcset', 'class'],
    FORBID_ATTR: ['style'],
};

const rssArticleBodyStyle = {
    lineHeight: '1.65',
    fontSize: '14px',
    color: 'var(--text-primary)',
    overflow: 'hidden',
    wordBreak: 'break-word',
};

// Article Card Component
const ArticleCard = ({ article, onAction, onTitleClick, formatArticleWhen, isExpanded, darkMode }) => {
    const [showActions, setShowActions] = useState(false);

    const stripHtmlTags = (html) => {
        if (!html) return '';
        // Create a temporary div to parse HTML and extract text content
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;
        return tempDiv.textContent || tempDiv.innerText || '';
    };

    const descOrBody = article.full_content || article.description || '';
    const previewPlain = stripHtmlTags(descOrBody);
    const expandedHtml =
        article.full_content_html ||
        (looksLikeHtml(article.description) && !article.full_content ? article.description : null);

    const sanitizedExpandedHtml = useMemo(() => {
        if (!expandedHtml) return '';
        return DOMPurify.sanitize(expandedHtml, RSS_ARTICLE_HTML_PURIFY);
    }, [expandedHtml]);

    return (
        <div
            data-article-id={article.article_id}
            style={articleCardStyle}
            onMouseEnter={() => setShowActions(true)}
            onMouseLeave={() => setShowActions(false)}
        >
            <div style={articleContentStyle}>
                <div style={articleTitleRowStyle}>
                    <button
                        type="button"
                        aria-label={article.is_starred ? 'Unstar article' : 'Star article'}
                        title={article.is_starred ? 'Unstar' : 'Star'}
                        onClick={(e) => {
                            e.stopPropagation();
                            onAction('toggle-star', article.article_id);
                        }}
                        style={starIconButtonStyle(article.is_starred === true, darkMode)}
                    >
                        {article.is_starred ? '\u2605' : '\u2606'}
                    </button>
                    <h3
                        style={articleTitleStyle}
                        onClick={() => onTitleClick(article.link)}
                    >
                        {article.title}
                    </h3>
                </div>
                
                {/* Display full content if available, otherwise fall back to description */}
                {(article.full_content_html || article.full_content || article.description) && (
                    <div style={articleDescriptionStyle}>
                        {isExpanded ? (
                            sanitizedExpandedHtml ? (
                                <div
                                    dangerouslySetInnerHTML={{ __html: sanitizedExpandedHtml }}
                                    style={rssArticleBodyStyle}
                                    className="rss-article-content"
                                />
                            ) : (
                                <p
                                    style={{
                                        margin: '0 0 12px 0',
                                        lineHeight: '1.65',
                                        whiteSpace: 'pre-wrap',
                                    }}
                                >
                                    {article.full_content || article.description}
                                </p>
                            )
                        ) : (
                            <p style={{ margin: '0 0 12px 0', lineHeight: '1.5' }}>
                                {previewPlain.length > 300
                                    ? `${previewPlain.substring(0, 300)}...`
                                    : previewPlain}
                            </p>
                        )}
                        
                        {previewPlain.length > 300 && (
                            <button 
                                onClick={() => onAction('expand-description', article.article_id)}
                                style={{
                                    background: 'none',
                                    border: 'none',
                                    color: '#1976d2',
                                    cursor: 'pointer',
                                    fontSize: '12px',
                                    textDecoration: 'underline',
                                    padding: 0
                                }}
                            >
                                {isExpanded ? 'Read less' : 'Read more'}
                            </button>
                        )}
                    </div>
                )}
                
                <div style={articleMetaStyle}>
                    <span style={articleDateStyle}>
                        {formatArticleWhen(article.published_date || article.created_at)}
                    </span>
                    {article.is_processed && (
                        <span style={importedBadgeStyle}>Imported</span>
                    )}
                    {!article.is_read && (
                        <span style={unreadBadgeStyle}>Unread</span>
                    )}
                </div>
            </div>
            
            {/* Hover Actions */}
            {showActions && (
                <div style={actionsStyle}>
                    {!article.is_read && (
                        <button
                            onClick={() => onAction('mark-read', article.article_id)}
                            style={actionButtonStyle}
                        >
                            Mark Read
                        </button>
                    )}

                    <button
                        type="button"
                        onClick={() => onAction('toggle-star', article.article_id)}
                        style={actionButtonStyle}
                    >
                        {article.is_starred ? 'Unstar' : 'Star'}
                    </button>

                    {!article.is_processed && (
                        <button
                            onClick={() => onAction('import', article.article_id)}
                            style={{ ...actionButtonStyle, backgroundColor: '#4caf50' }}
                        >
                            Import
                        </button>
                    )}
                    

                    
                    <button
                        onClick={() => onAction('delete', article.article_id)}
                        style={{ ...actionButtonStyle, backgroundColor: '#f44336' }}
                    >
                        Delete
                    </button>
                </div>
            )}
        </div>
    );
};

// Styles
const containerStyle = {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    backgroundColor: 'var(--bg-secondary)'
};

const headerStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 24px',
    backgroundColor: 'var(--bg-primary)',
    borderBottom: '1px solid var(--border-primary)'
};

const headerLeftStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '12px'
};

const titleStyle = {
    margin: 0,
    fontSize: '20px',
    fontWeight: '600',
    color: 'var(--text-primary)'
};

const articleCountStyle = {
    fontSize: '14px',
    color: 'var(--text-secondary)',
    backgroundColor: 'var(--bg-tertiary)',
    padding: '4px 8px',
    borderRadius: '12px'
};

const closeButtonStyle = {
    background: 'none',
    border: 'none',
    fontSize: '24px',
    cursor: 'pointer',
    padding: '4px',
    borderRadius: '4px',
    color: 'var(--text-secondary)'
};

const controlsStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 24px',
    backgroundColor: 'var(--bg-primary)',
    borderBottom: '1px solid var(--border-primary)'
};

const filterControlsStyle = {
    display: 'flex',
    gap: '12px'
};

const selectStyle = {
    padding: '8px 12px',
    border: '1px solid var(--border-secondary)',
    borderRadius: '4px',
    fontSize: '14px',
    backgroundColor: 'var(--bg-primary)',
    color: 'var(--text-primary)'
};

const bulkActionsStyle = {
    display: 'flex',
    gap: '8px'
};

const bulkButtonStyle = {
    padding: '8px 16px',
    backgroundColor: '#2196f3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '14px'
};

const articlesContainerStyle = {
    flex: 1,
    overflow: 'auto',
    padding: '16px 24px'
};

const articleCardStyle = {
    backgroundColor: 'var(--bg-primary)',
    borderRadius: '8px',
    padding: '20px',
    marginBottom: '16px',
    boxShadow: '0 2px 4px var(--shadow-light)',
    position: 'relative',
    transition: 'box-shadow 0.2s ease'
};

const articleContentStyle = {
    marginRight: '120px' // Space for actions
};

const articleTitleRowStyle = {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '8px',
    marginBottom: '12px',
    width: '100%',
};

const articleTitleStyle = {
    margin: 0,
    flex: 1,
    fontSize: '18px',
    fontWeight: '600',
    color: '#2196f3',
    cursor: 'pointer',
    textDecoration: 'none',
};

function starIconButtonStyle(isStarred, darkMode) {
    return {
        flexShrink: 0,
        marginTop: '2px',
        padding: '4px 6px',
        fontSize: '18px',
        lineHeight: 1,
        border: 'none',
        background: 'transparent',
        cursor: 'pointer',
        color: isStarred ? '#ffb300' : darkMode ? '#888' : '#999',
    };
}

const articleDescriptionStyle = {
    margin: '0 0 12px 0',
    fontSize: '14px',
    color: 'var(--text-secondary)',
    lineHeight: '1.5'
};

const articleMetaStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    fontSize: '12px',
    color: 'var(--text-secondary)'
};

const articleDateStyle = {
    fontSize: '12px',
    color: 'var(--text-secondary)'
};

const importedBadgeStyle = {
    backgroundColor: '#4caf50',
    color: 'white',
    padding: '2px 6px',
    borderRadius: '10px',
    fontSize: '10px'
};

const unreadBadgeStyle = {
    backgroundColor: '#ff9800',
    color: 'white',
    padding: '2px 6px',
    borderRadius: '10px',
    fontSize: '10px'
};

const actionsStyle = {
    position: 'absolute',
    top: '20px',
    right: '20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px'
};

const actionButtonStyle = {
    padding: '6px 12px',
    backgroundColor: '#2196f3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '12px',
    whiteSpace: 'nowrap'
};

const loadingStyle = {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '200px',
    fontSize: '16px',
    color: 'var(--text-secondary)'
};

const errorStyle = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '200px',
    textAlign: 'center',
    color: 'var(--text-primary)'
};

const retryButtonStyle = {
    padding: '8px 16px',
    backgroundColor: '#2196f3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    marginTop: '12px'
};

const emptyStateStyle = {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '200px',
    color: 'var(--text-secondary)',
    fontSize: '16px'
};

export default RSSArticleViewer;
