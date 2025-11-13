/**
 * RSS Feed Manager Modal
 * Component for adding and managing RSS feeds
 */

import React, { useState, useEffect } from 'react';
import { useQueryClient } from 'react-query';
import rssService from '../services/rssService';
import { useAuth } from '../contexts/AuthContext';

const RSSFeedManager = ({ isOpen, onClose, onFeedAdded, feedContext = null }) => {
    const queryClient = useQueryClient();
    const { user } = useAuth();
    
    const [feedData, setFeedData] = useState({
        feed_url: '',
        feed_name: '',
        category: 'technology',
        tags: [],
        check_interval: 3600
    });
    
    // Determine scope from context instead of manual selection
    const feedScope = feedContext?.isGlobal ? 'global' : 'user';
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [preview, setPreview] = useState(null);
    const [validating, setValidating] = useState(false);

    const categories = [
        'technology', 'science', 'news', 'business', 'politics', 
        'entertainment', 'sports', 'health', 'education', 'other'
    ];

    const checkIntervals = [
        { value: 900, label: '15 minutes' },
        { value: 1800, label: '30 minutes' },
        { value: 3600, label: '1 hour' },
        { value: 7200, label: '2 hours' },
        { value: 14400, label: '4 hours' },
        { value: 28800, label: '8 hours' },
        { value: 86400, label: '24 hours' }
    ];

    // Check if user can create global feeds
    const canCreateGlobalFeeds = user?.role === 'admin';

    useEffect(() => {
        if (isOpen) {
            resetForm();
        }
    }, [isOpen]);

    const resetForm = () => {
        setFeedData({
            feed_url: '',
            feed_name: '',
            category: 'technology',
            tags: [],
            check_interval: 3600
        });
        setError(null);
        setPreview(null);
    };

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFeedData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleTagsChange = (e) => {
        const tags = e.target.value.split(',').map(tag => tag.trim()).filter(tag => tag);
        setFeedData(prev => ({
            ...prev,
            tags
        }));
    };

    const validateFeedUrl = async () => {
        if (!feedData.feed_url) {
            setError('Please enter a feed URL');
            return false;
        }

        if (!feedData.feed_url.startsWith('http://') && !feedData.feed_url.startsWith('https://')) {
            setError('Feed URL must start with http:// or https://');
            return false;
        }

        return true;
    };

    const previewFeed = async () => {
        if (!(await validateFeedUrl())) return;

        setValidating(true);
        setError(null);

        try {
            // Call the validation API endpoint
            const response = await fetch(`/api/rss/feeds/validate?feed_url=${encodeURIComponent(feedData.feed_url)}`);
            
            if (!response.ok) {
                throw new Error('Failed to validate RSS feed URL');
            }
            
            const result = await response.json();
            
            if (result.status === 'success') {
                setPreview(result.data);
            } else {
                throw new Error('Invalid RSS feed URL');
            }
        } catch (error) {
            setError('Failed to preview feed. Please check the URL and try again.');
        } finally {
            setValidating(false);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        
        if (!(await validateFeedUrl())) return;
        
        if (!feedData.feed_name.trim()) {
            setError('Please enter a feed name');
            return;
        }

        // Validate global feed creation permissions
        if (feedScope === 'global' && !canCreateGlobalFeeds) {
            setError('Only admin users can create global RSS feeds');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const isGlobal = feedScope === 'global';
            const newFeed = await rssService.createFeed(feedData, isGlobal);
            
            // Invalidate React Query cache to refresh the feed list
            queryClient.invalidateQueries(['rss', 'feeds']);
            queryClient.invalidateQueries(['rss', 'unread-counts']);
            
            // Show success message
            const scopeText = isGlobal ? 'global' : 'personal';
            showToast(`RSS feed "${newFeed.feed_name}" added successfully as ${scopeText} feed!`, 'success');
            
            // Close modal and notify parent
            onClose();
            if (onFeedAdded) {
                onFeedAdded(newFeed);
            }
        } catch (error) {
            setError(error.message || 'Failed to add RSS feed');
        } finally {
            setLoading(false);
        }
    };

    const showToast = (message, type = 'info') => {
        // Simple toast implementation - in a real app, you'd use a proper toast library
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#4caf50' : '#2196f3'};
            color: white;
            padding: 12px 20px;
            border-radius: 4px;
            z-index: 10000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        `;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 3000);
    };

    if (!isOpen) return null;

    return (
        <div className="modal-overlay" style={modalOverlayStyle}>
            <div className="modal-content" style={modalContentStyle}>
                <div className="modal-header" style={modalHeaderStyle}>
                    <h2>Add RSS Feed</h2>
                    <button 
                        onClick={onClose}
                        style={closeButtonStyle}
                        disabled={loading}
                    >
                        ×
                    </button>
                </div>

                <form onSubmit={handleSubmit} style={formStyle}>
                    <div className="form-group" style={formGroupStyle}>
                        <label htmlFor="feed_url" style={labelStyle}>
                            Feed URL *
                        </label>
                        <div style={urlInputGroupStyle}>
                            <input
                                type="url"
                                id="feed_url"
                                name="feed_url"
                                value={feedData.feed_url}
                                onChange={handleInputChange}
                                placeholder="https://example.com/feed.xml"
                                style={inputStyle}
                                disabled={loading}
                                required
                            />
                            <button
                                type="button"
                                onClick={previewFeed}
                                disabled={loading || validating || !feedData.feed_url}
                                style={previewButtonStyle}
                            >
                                {validating ? 'Validating...' : 'Preview'}
                            </button>
                        </div>
                    </div>

                    <div className="form-group" style={formGroupStyle}>
                        <label htmlFor="feed_name" style={labelStyle}>
                            Feed Name *
                        </label>
                        <input
                            type="text"
                            id="feed_name"
                            name="feed_name"
                            value={feedData.feed_name}
                            onChange={handleInputChange}
                            placeholder="Enter a name for this feed"
                            style={inputStyle}
                            disabled={loading}
                            required
                        />
                    </div>

                    <div className="form-group" style={formGroupStyle}>
                        <label style={labelStyle}>
                            Feed Scope
                        </label>
                        <div style={{
                            padding: '8px 12px',
                            border: '1px solid #ddd',
                            borderRadius: '4px',
                            backgroundColor: '#f9f9f9',
                            color: '#333',
                            fontSize: '0.9rem'
                        }}>
                            {feedScope === 'global' ? 'Global Feed (Global Documents)' : 'Personal Feed (My Documents)'}
                        </div>
                        <small style={{ color: '#666', fontSize: '0.8rem', marginTop: '4px', display: 'block' }}>
                            {feedScope === 'global' 
                                ? 'Global feeds are visible to all users and appear in Global Documents'
                                : 'Personal feeds are only visible to you and appear in My Documents'
                            }
                        </small>
                    </div>

                    <div className="form-group" style={formGroupStyle}>
                        <label htmlFor="category" style={labelStyle}>
                            Category
                        </label>
                        <select
                            id="category"
                            name="category"
                            value={feedData.category}
                            onChange={handleInputChange}
                            style={selectStyle}
                            disabled={loading}
                        >
                            {categories.map(category => (
                                <option key={category} value={category}>
                                    {category.charAt(0).toUpperCase() + category.slice(1)}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div className="form-group" style={formGroupStyle}>
                        <label htmlFor="tags" style={labelStyle}>
                            Tags (comma-separated)
                        </label>
                        <input
                            type="text"
                            id="tags"
                            name="tags"
                            value={feedData.tags.join(', ')}
                            onChange={handleTagsChange}
                            placeholder="tech, news, ai"
                            style={inputStyle}
                            disabled={loading}
                        />
                    </div>

                    <div className="form-group" style={formGroupStyle}>
                        <label htmlFor="check_interval" style={labelStyle}>
                            Check Interval
                        </label>
                        <select
                            id="check_interval"
                            name="check_interval"
                            value={feedData.check_interval}
                            onChange={handleInputChange}
                            style={selectStyle}
                            disabled={loading}
                        >
                            {checkIntervals.map(interval => (
                                <option key={interval.value} value={interval.value}>
                                    {interval.label}
                                </option>
                            ))}
                        </select>
                    </div>

                    {error && (
                        <div className="error-message" style={errorStyle}>
                            {error}
                        </div>
                    )}

                    {preview && (
                        <div className="feed-preview" style={previewStyle}>
                            <h4>Feed Preview</h4>
                            <p><strong>Title:</strong> {preview.title}</p>
                            <p><strong>Description:</strong> {preview.description}</p>
                            <p><strong>Sample Articles:</strong></p>
                            <ul>
                                {preview.articles.map((article, index) => (
                                    <li key={index}>{article.title}</li>
                                ))}
                            </ul>
                        </div>
                    )}

                    <div className="modal-actions" style={actionsStyle}>
                        <button
                            type="button"
                            onClick={onClose}
                            style={cancelButtonStyle}
                            disabled={loading}
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            style={submitButtonStyle}
                            disabled={loading || !feedData.feed_url || !feedData.feed_name}
                        >
                            {loading ? 'Adding Feed...' : 'Add RSS Feed'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

// Styles
const modalOverlayStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000
};

const modalContentStyle = {
    backgroundColor: 'white',
    borderRadius: '8px',
    padding: '0',
    maxWidth: '600px',
    width: '90%',
    maxHeight: '90vh',
    overflow: 'auto',
    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)'
};

const modalHeaderStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '20px 24px',
    borderBottom: '1px solid #e0e0e0'
};

const closeButtonStyle = {
    background: 'none',
    border: 'none',
    fontSize: '24px',
    cursor: 'pointer',
    padding: '0',
    width: '30px',
    height: '30px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: '4px'
};

const formStyle = {
    padding: '24px'
};

const formGroupStyle = {
    marginBottom: '20px'
};

const labelStyle = {
    display: 'block',
    marginBottom: '8px',
    fontWeight: '500',
    color: '#333'
};

const inputStyle = {
    width: '100%',
    padding: '10px 12px',
    border: '1px solid #ddd',
    borderRadius: '4px',
    fontSize: '14px',
    boxSizing: 'border-box'
};

const selectStyle = {
    width: '100%',
    padding: '10px 12px',
    border: '1px solid #ddd',
    borderRadius: '4px',
    fontSize: '14px',
    backgroundColor: 'white'
};

const urlInputGroupStyle = {
    display: 'flex',
    gap: '10px'
};

const previewButtonStyle = {
    padding: '10px 16px',
    backgroundColor: '#2196f3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    whiteSpace: 'nowrap'
};

const errorStyle = {
    color: '#d32f2f',
    backgroundColor: '#ffebee',
    padding: '12px',
    borderRadius: '4px',
    marginBottom: '20px',
    border: '1px solid #ffcdd2'
};

const previewStyle = {
    backgroundColor: '#f5f5f5',
    padding: '16px',
    borderRadius: '4px',
    marginBottom: '20px',
    border: '1px solid #e0e0e0'
};

const actionsStyle = {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '12px',
    marginTop: '24px',
    paddingTop: '20px',
    borderTop: '1px solid #e0e0e0'
};

const cancelButtonStyle = {
    padding: '10px 20px',
    backgroundColor: '#f5f5f5',
    color: '#333',
    border: '1px solid #ddd',
    borderRadius: '4px',
    cursor: 'pointer'
};

const submitButtonStyle = {
    padding: '10px 20px',
    backgroundColor: '#4caf50',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontWeight: '500'
};

export default RSSFeedManager;
