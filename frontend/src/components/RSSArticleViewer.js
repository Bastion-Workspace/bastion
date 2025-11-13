/**
 * RSS Article Viewer Component
 * Displays RSS articles with filtering, sorting, and article actions
 */

import React, { useState, useEffect } from 'react';
import rssService from '../services/rssService';
import { useTheme } from '../contexts/ThemeContext';

const RSSArticleViewer = ({ feedId, onClose }) => {
    const { darkMode } = useTheme();
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

    useEffect(() => {
        if (feedId) {
            loadFeedArticles();
        }
    }, [feedId]);

    // Live update: when background refresh completes for this feed, reload articles
    useEffect(() => {
        const handler = (e) => {
            try {
                const detail = e?.detail || {};
                if (detail.feedId && detail.feedId === feedId) {
                    loadFeedArticles();
                }
            } catch (_) {}
        };
        window.addEventListener('rss-feed-refresh-complete', handler);
        return () => window.removeEventListener('rss-feed-refresh-complete', handler);
    }, [feedId]);

    useEffect(() => {
        applyFilterAndSort();
    }, [articles, filter, sortBy]);

    const loadFeedArticles = async () => {
        try {
            setLoading(true);
            setError(null);
            
            const articlesData = await rssService.getFeedArticles(feedId, 1000);
            setArticles(articlesData);
            setCurrentFeed(rssService.currentFeed);
        } catch (error) {
            setError(error.message || 'Failed to load articles');
            console.error('❌ RSS ARTICLE VIEWER ERROR:', error);
        } finally {
            setLoading(false);
        }
    };

    const applyFilterAndSort = () => {
        let filtered = rssService.filterArticles(articles, filter);
        filtered = rssService.sortArticles(filtered, sortBy);
        setFilteredArticles(filtered);
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
                    // Remove from local state
                    setArticles(prev => prev.filter(article => article.article_id !== articleId));
                    showToast('Article deleted successfully!', 'success');
                    break;
                    
                case 'expand-description':
                    // Toggle expanded state for this article
                    setExpandedDescriptions(prev => {
                        const newSet = new Set(prev);
                        if (newSet.has(articleId)) {
                            newSet.delete(articleId);
                        } else {
                            newSet.add(articleId);
                        }
                        return newSet;
                    });
                    break;
                    

                    
                default:
                    break;
            }
        } catch (error) {
            showToast(`Failed to ${action.replace('-', ' ')} article: ${error.message}`, 'error');
        }
    };

    const handleBulkAction = async (action) => {
        if (!currentFeed) return;
        
        let confirmMessage;
        if (action === 'mark-all-read') {
            confirmMessage = 'Mark all unread articles as read?';
        } else if (action === 'delete-all-read') {
            confirmMessage = 'Delete all read (non-imported) articles?';
        } else if (action === 'extract-full-content') {
            confirmMessage = 'Extract full content for articles with truncated descriptions? This may take a few minutes.';
        }
            
        if (!window.confirm(confirmMessage)) return;
        
        try {
            setBulkLoading(true);
            
            if (action === 'mark-all-read') {
                await rssService.markAllArticlesRead(currentFeed.feed_id);
                // Update local state
                setArticles(prev => prev.map(article => ({ ...article, is_read: true })));
                showToast('All articles marked as read!', 'success');
            } else if (action === 'delete-all-read') {
                await rssService.deleteAllReadArticles(currentFeed.feed_id);
                // Remove read non-imported articles from local state
                setArticles(prev => prev.filter(article => !(article.is_read && !article.is_processed)));
                showToast('All read articles deleted!', 'success');
            } else if (action === 'extract-full-content') {
                await rssService.extractFullContent();
                showToast('Full content extraction started! Check back in a few minutes.', 'success');
                // Reload articles to show updated content
                await loadFeedArticles();
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

    const formatDate = (dateString) => {
        if (!dateString) return 'Unknown date';
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

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
                        {currentFeed?.feed_name || 'RSS Feed'}
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
                    <button
                        onClick={() => handleBulkAction('extract-full-content')}
                        disabled={bulkLoading}
                        style={{ ...bulkButtonStyle, backgroundColor: '#4caf50' }}
                    >
                        {bulkLoading ? 'Processing...' : 'Extract Full Content'}
                    </button>
                </div>
            </div>

            {/* Articles List */}
            <div style={articlesContainerStyle}>
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
                            formatDate={formatDate}
                            isExpanded={expandedDescriptions.has(article.article_id)}
                            darkMode={darkMode}
                        />
                    ))
                )}
            </div>
        </div>
    );
};

// Article Card Component
const ArticleCard = ({ article, onAction, onTitleClick, formatDate, isExpanded, darkMode }) => {
    const [showActions, setShowActions] = useState(false);

    const stripHtmlTags = (html) => {
        if (!html) return '';
        // Create a temporary div to parse HTML and extract text content
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;
        return tempDiv.textContent || tempDiv.innerText || '';
    };

    return (
        <div 
            style={articleCardStyle}
            onMouseEnter={() => setShowActions(true)}
            onMouseLeave={() => setShowActions(false)}
        >
            <div style={articleContentStyle}>
                <h3 
                    style={articleTitleStyle}
                    onClick={() => onTitleClick(article.link)}
                >
                    {article.title}
                </h3>
                
                {/* Display full content if available, otherwise fall back to description */}
                {(article.full_content_html || article.full_content || article.description) && (
                    <div style={articleDescriptionStyle}>
                        {isExpanded ? (
                            // Show full content with original HTML layout
                            article.full_content_html ? (
                                <div 
                                    dangerouslySetInnerHTML={{ 
                                        __html: article.full_content_html 
                                    }}
                                    style={{
                                        lineHeight: '1.6',
                                        fontSize: '14px',
                                        color: 'var(--text-primary)',
                                        overflow: 'hidden'
                                    }}
                                    className="rss-article-content"
                                />
                            ) : (
                                <p style={{ margin: '0 0 12px 0', lineHeight: '1.5' }}>
                                    {article.full_content || article.description}
                                </p>
                            )
                        ) : (
                            <p style={{ margin: '0 0 12px 0', lineHeight: '1.5' }}>
                                {stripHtmlTags(article.full_content || article.description).substring(0, 300) + '...'}
                            </p>
                        )}
                        
                        {(article.full_content_html || article.full_content || article.description).length > 300 && (
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
                        
                        {(article.full_content_html || article.full_content) && (
                            <div style={{
                                fontSize: '11px',
                                color: 'var(--text-secondary)',
                                marginTop: '8px',
                                padding: '4px 8px',
                                backgroundColor: darkMode ? 'var(--bg-tertiary)' : '#e3f2fd',
                                borderRadius: '4px',
                                display: 'inline-block'
                            }}>
                                ✨ {article.full_content_html ? 'Full content with original layout' : 'Full content extracted'}
                            </div>
                        )}
                    </div>
                )}
                
                <div style={articleMetaStyle}>
                    <span style={articleDateStyle}>
                        {formatDate(article.published_date)}
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

const articleTitleStyle = {
    margin: '0 0 12px 0',
    fontSize: '18px',
    fontWeight: '600',
    color: '#2196f3',
    cursor: 'pointer',
    textDecoration: 'none'
};

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
