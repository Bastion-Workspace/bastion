/**
 * RSS Navigation Component
 * Integrates with the existing file tree sidebar to show RSS feeds
 * with unread counts, context menus, and feed management
 */

import React, { useState, useEffect } from 'react';
import rssService from '../services/rssService';

const RSSNavigation = ({ onFeedClick, onAddFeed }) => {
    const [feeds, setFeeds] = useState([]);
    const [unreadCounts, setUnreadCounts] = useState({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [contextMenu, setContextMenu] = useState(null);
    const [refreshingFeeds, setRefreshingFeeds] = useState(new Set());

    useEffect(() => {
        loadFeeds();
        
        // Set up periodic refresh of unread counts
        const interval = setInterval(() => {
            refreshUnreadCounts();
        }, 30000); // Refresh every 30 seconds
        
        return () => clearInterval(interval);
    }, []);

    const loadFeeds = async () => {
        try {
            setLoading(true);
            setError(null);
            
            const feedsData = await rssService.getFeeds();
            setFeeds(feedsData);
            
            const counts = await rssService.getUnreadCounts();
            setUnreadCounts(counts);
        } catch (error) {
            setError(error.message || 'Failed to load RSS feeds');
            console.error('❌ RSS NAVIGATION ERROR:', error);
        } finally {
            setLoading(false);
        }
    };

    const refreshUnreadCounts = async () => {
        try {
            const counts = await rssService.getUnreadCounts();
            setUnreadCounts(counts);
        } catch (error) {
            console.error('❌ Failed to refresh unread counts:', error);
        }
    };

    const handleFeedClick = (feed) => {
        if (onFeedClick) {
            onFeedClick(feed.feed_id, feed.feed_name);
        }
    };

    const handleContextMenu = (e, feed) => {
        e.preventDefault();
        setContextMenu({
            x: e.clientX,
            y: e.clientY,
            feed: feed
        });
    };

    const closeContextMenu = () => {
        setContextMenu(null);
    };

    const handleRefreshFeed = async (feedId) => {
        try {
            setRefreshingFeeds(prev => new Set(prev).add(feedId));
            
            await rssService.refreshFeed(feedId);
            
            // Refresh unread counts
            await refreshUnreadCounts();
            
            showToast(`Feed refreshed successfully!`, 'success');
        } catch (error) {
            showToast(`Failed to refresh feed: ${error.message}`, 'error');
        } finally {
            setRefreshingFeeds(prev => {
                const newSet = new Set(prev);
                newSet.delete(feedId);
                return newSet;
            });
        }
    };

    const handleDeleteFeed = async (feedId, deleteArticles = false) => {
        const feed = feeds.find(f => f.feed_id === feedId);
        if (!feed) return;

        const confirmMessage = deleteArticles 
            ? `Delete "${feed.feed_name}" and all its articles? This action cannot be undone.`
            : `Delete "${feed.feed_name}"? This will keep imported articles but remove the feed.`;
            
        if (!window.confirm(confirmMessage)) return;

        try {
            await rssService.deleteFeed(feedId, deleteArticles);
            
            // Remove from local state
            setFeeds(prev => prev.filter(f => f.feed_id !== feedId));
            
            // Refresh unread counts
            await refreshUnreadCounts();
            
            showToast(`Feed "${feed.feed_name}" deleted successfully!`, 'success');
        } catch (error) {
            showToast(`Failed to delete feed: ${error.message}`, 'error');
        }
    };

    const formatLastUpdated = (lastCheck) => {
        if (!lastCheck) return 'Never';
        
        const now = new Date();
        const lastCheckDate = new Date(lastCheck);
        const diffMs = now - lastCheckDate;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins} minutes ago`;
        if (diffHours < 24) return `${diffHours} hours ago`;
        if (diffDays < 7) return `${diffDays} days ago`;
        
        return lastCheckDate.toLocaleDateString();
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
            <div style={loadingStyle}>
                <div style={spinnerStyle}>Loading RSS feeds...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div style={errorStyle}>
                <p>Error loading RSS feeds: {error}</p>
                <button onClick={loadFeeds} style={retryButtonStyle}>
                    Retry
                </button>
            </div>
        );
    }

    return (
        <div style={containerStyle}>
            {/* RSS Feeds Header */}
            <div style={headerStyle}>
                <div style={headerContentStyle}>
                    <span style={headerIconStyle}>📰</span>
                    <span style={headerTitleStyle}>RSS Feeds</span>
                    {feeds.length > 0 && (
                        <span style={totalUnreadStyle}>
                            {Object.values(unreadCounts).reduce((sum, count) => sum + count, 0)}
                        </span>
                    )}
                </div>
                
                <button
                    onClick={onAddFeed}
                    style={addButtonStyle}
                    title="Add RSS Feed"
                >
                    +
                </button>
            </div>

            {/* RSS Feeds List */}
            <div style={feedsListStyle}>
                {feeds.length === 0 ? (
                    <div style={emptyStateStyle}>
                        <p>No RSS feeds added yet.</p>
                        <button
                            onClick={onAddFeed}
                            style={addFeedButtonStyle}
                        >
                            Add RSS Feed
                        </button>
                    </div>
                ) : (
                    feeds.map((feed) => (
                        <div
                            key={feed.feed_id}
                            style={feedItemStyle}
                            onClick={() => handleFeedClick(feed)}
                            onContextMenu={(e) => handleContextMenu(e, feed)}
                        >
                            <div style={feedContentStyle}>
                                <div style={feedInfoStyle}>
                                    <span style={feedNameStyle}>{feed.feed_name}</span>
                                    <span style={feedUrlStyle}>{feed.feed_url}</span>
                                </div>
                                
                                <div style={feedMetaStyle}>
                                    <span style={lastUpdatedStyle}>
                                        {formatLastUpdated(feed.last_check)}
                                    </span>
                                    {unreadCounts[feed.feed_id] > 0 && (
                                        <span style={unreadCountStyle}>
                                            {unreadCounts[feed.feed_id]}
                                        </span>
                                    )}
                                    {refreshingFeeds.has(feed.feed_id) && (
                                        <span style={refreshingStyle}>🔄</span>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>

            {/* Context Menu */}
            {contextMenu && (
                <>
                    <div style={contextMenuOverlayStyle} onClick={closeContextMenu} />
                    <div
                        style={{
                            ...contextMenuStyle,
                            left: contextMenu.x,
                            top: contextMenu.y
                        }}
                    >
                        <button
                            onClick={() => {
                                handleRefreshFeed(contextMenu.feed.feed_id);
                                closeContextMenu();
                            }}
                            style={contextMenuItemStyle}
                            disabled={refreshingFeeds.has(contextMenu.feed.feed_id)}
                        >
                            🔄 Refresh Feed
                        </button>
                        <button
                            onClick={() => {
                                handleDeleteFeed(contextMenu.feed.feed_id, false);
                                closeContextMenu();
                            }}
                            style={contextMenuItemStyle}
                        >
                            🗑️ Delete Feed Only
                        </button>
                        <button
                            onClick={() => {
                                handleDeleteFeed(contextMenu.feed.feed_id, true);
                                closeContextMenu();
                            }}
                            style={contextMenuItemStyle}
                        >
                            🗑️ Delete Feed & Articles
                        </button>
                        <div style={contextMenuDividerStyle} />
                        <div style={contextMenuInfoStyle}>
                            <div>Last Updated: {formatLastUpdated(contextMenu.feed.last_check)}</div>
                            <div>Unread: {unreadCounts[contextMenu.feed.feed_id] || 0}</div>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
};

// Styles
const containerStyle = {
    display: 'flex',
    flexDirection: 'column',
    height: '100%'
};

const headerStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px 16px',
    backgroundColor: '#f8f9fa',
    borderBottom: '1px solid #e0e0e0'
};

const headerContentStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
};

const headerIconStyle = {
    fontSize: '16px'
};

const headerTitleStyle = {
    fontSize: '14px',
    fontWeight: '600',
    color: '#333'
};

const totalUnreadStyle = {
    backgroundColor: '#ff9800',
    color: 'white',
    padding: '2px 6px',
    borderRadius: '10px',
    fontSize: '12px',
    fontWeight: '500'
};

const addButtonStyle = {
    background: 'none',
    border: 'none',
    fontSize: '18px',
    cursor: 'pointer',
    padding: '4px 8px',
    borderRadius: '4px',
    color: '#2196f3',
    fontWeight: 'bold'
};

const feedsListStyle = {
    flex: 1,
    overflow: 'auto'
};

const feedItemStyle = {
    padding: '12px 16px',
    borderBottom: '1px solid #f0f0f0',
    cursor: 'pointer',
    transition: 'background-color 0.2s ease'
};

const feedContentStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start'
};

const feedInfoStyle = {
    flex: 1,
    minWidth: 0
};

const feedNameStyle = {
    display: 'block',
    fontSize: '14px',
    fontWeight: '500',
    color: '#333',
    marginBottom: '4px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap'
};

const feedUrlStyle = {
    display: 'block',
    fontSize: '12px',
    color: '#666',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap'
};

const feedMetaStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginLeft: '12px'
};

const lastUpdatedStyle = {
    fontSize: '11px',
    color: '#999',
    whiteSpace: 'nowrap'
};

const unreadCountStyle = {
    backgroundColor: '#ff9800',
    color: 'white',
    padding: '2px 6px',
    borderRadius: '10px',
    fontSize: '11px',
    fontWeight: '500',
    minWidth: '16px',
    textAlign: 'center'
};

const refreshingStyle = {
    fontSize: '12px',
    animation: 'spin 1s linear infinite'
};

const loadingStyle = {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '100px',
    color: '#666'
};

const spinnerStyle = {
    fontSize: '14px'
};

const errorStyle = {
    padding: '16px',
    textAlign: 'center',
    color: '#f44336'
};

const retryButtonStyle = {
    padding: '8px 16px',
    backgroundColor: '#2196f3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    marginTop: '8px'
};

const emptyStateStyle = {
    padding: '24px 16px',
    textAlign: 'center',
    color: '#666'
};

const addFeedButtonStyle = {
    padding: '8px 16px',
    backgroundColor: '#2196f3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    marginTop: '8px',
    fontSize: '12px'
};

const contextMenuOverlayStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 1000
};

const contextMenuStyle = {
    position: 'fixed',
    backgroundColor: 'white',
    border: '1px solid #e0e0e0',
    borderRadius: '4px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
    zIndex: 1001,
    minWidth: '200px',
    padding: '8px 0'
};

const contextMenuItemStyle = {
    display: 'block',
    width: '100%',
    padding: '8px 16px',
    background: 'none',
    border: 'none',
    textAlign: 'left',
    cursor: 'pointer',
    fontSize: '14px',
    color: '#333'
};

const contextMenuDividerStyle = {
    height: '1px',
    backgroundColor: '#e0e0e0',
    margin: '8px 0'
};

const contextMenuInfoStyle = {
    padding: '8px 16px',
    fontSize: '12px',
    color: '#666',
    backgroundColor: '#f8f9fa'
};

// Add CSS animation for spinner
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);

export default RSSNavigation;
