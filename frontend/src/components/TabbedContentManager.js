/**
 * Tabbed Content Manager
 * Manages multiple tabs for RSS feeds, documents, and other content
 * Maximum 5 tabs with persistence across sessions
 */

import React, { useState, useEffect, forwardRef, useImperativeHandle } from 'react';
import { useTheme } from '@mui/material/styles';
import RSSArticleViewer from './RSSArticleViewer';
import NewsHeadlinesPane from './NewsHeadlinesPane';
import RSSFeedManager from './RSSFeedManager';
import DocumentViewer from './DocumentViewer';
import OrgSearchView from './OrgSearchView';
import OrgAgendaView from './OrgAgendaView';
import OrgTodosView from './OrgTodosView';
import OrgContactsView from './OrgContactsView';
import DataWorkspaceManager from './data_workspace/DataWorkspaceManager';
import FileRelationGraph from './FileRelationGraph';
import { documentDiffStore } from '../services/documentDiffStore';

const TabbedContentManager = forwardRef((props, ref) => {
    const theme = useTheme();
    const [tabs, setTabs] = useState([]);
    const [activeTabId, setActiveTabId] = useState(null);
    const [showFeedManager, setShowFeedManager] = useState(false);
    const [tabDiffCounts, setTabDiffCounts] = useState({}); // { [tabId]: diffCount }
    
    const MAX_TABS = 5;

    // Load tabs from localStorage on component mount
    useEffect(() => {
        const savedTabs = localStorage.getItem('rss-tabs');
        const savedActiveTab = localStorage.getItem('rss-active-tab');
        
        if (savedTabs) {
            try {
                const parsedTabs = JSON.parse(savedTabs);
                setTabs(parsedTabs);
                
                if (savedActiveTab && parsedTabs.find(tab => tab.id === savedActiveTab)) {
                    setActiveTabId(savedActiveTab);
                } else if (parsedTabs.length > 0) {
                    setActiveTabId(parsedTabs[0].id);
                }
            } catch (error) {
                console.error('❌ Failed to parse saved tabs:', error);
                setTabs([]);
            }
        }
    }, []);

    // Save tabs to localStorage whenever tabs change
    useEffect(() => {
        localStorage.setItem('rss-tabs', JSON.stringify(tabs));
    }, [tabs]);

    // Subscribe to diff store changes for badge updates
    useEffect(() => {
        const handleDiffChange = (documentId, changeType) => {
            console.log('🔔 TabbedContentManager: Diff change notification', { documentId, changeType, tabsCount: tabs.length });
            
            // Find all tabs with this documentId
            const matchingTabs = tabs.filter(t => t.type === 'document' && t.documentId === documentId);
            
            console.log('🔔 TabbedContentManager: Found matching tabs:', matchingTabs.length);
            
            matchingTabs.forEach(tab => {
                const diffs = documentDiffStore.getDiffs(documentId);
                const diffCount = diffs && Array.isArray(diffs.operations) ? diffs.operations.length : 0;
                
                console.log('🔔 TabbedContentManager: Updating tab badge', { tabId: tab.id, diffCount });
                
                setTabDiffCounts(prev => ({
                    ...prev,
                    [tab.id]: diffCount
                }));
            });
        };
        
        // Initial load: check all document tabs for existing diffs
        tabs.forEach(tab => {
            if (tab.type === 'document' && tab.documentId) {
                const diffs = documentDiffStore.getDiffs(tab.documentId);
                const diffCount = diffs && Array.isArray(diffs.operations) ? diffs.operations.length : 0;
                if (diffCount > 0) {
                    setTabDiffCounts(prev => ({
                        ...prev,
                        [tab.id]: diffCount
                    }));
                }
            }
        });
        
        documentDiffStore.subscribe(handleDiffChange);
        return () => documentDiffStore.unsubscribe(handleDiffChange);
    }, [tabs]);

    // Save active tab to localStorage whenever it changes
    useEffect(() => {
        if (activeTabId) {
            localStorage.setItem('rss-active-tab', activeTabId);
        }
    }, [activeTabId]);

    const addTab = (tabData) => {
        const newTab = {
            id: generateTabId(),
            ...tabData,
            scrollPosition: 0, // Track scroll position for each tab
            createdAt: Date.now()
        };

        setTabs(prevTabs => {
            let updatedTabs = [...prevTabs];
            
            // If we're at the limit, remove the oldest tab
            if (updatedTabs.length >= MAX_TABS) {
                updatedTabs = updatedTabs.slice(1); // Remove oldest tab
            }
            
            return [...updatedTabs, newTab];
        });
        
        setActiveTabId(newTab.id);
    };
    
    // Update scroll position for a specific tab
    const updateTabScrollPosition = (tabId, scrollPosition) => {
        setTabs(prevTabs => prevTabs.map(tab => 
            tab.id === tabId 
                ? { ...tab, scrollPosition }
                : tab
        ));
    };

    const closeTab = (tabId, options = {}) => {
        const { skipUnsavedCheck = false } = options;
        const tab = tabs.find(t => t.id === tabId);
        if (tab && tab.type === 'document' && tab.documentId && !skipUnsavedCheck) {
            const unsavedKey = `unsaved_content_${tab.documentId}`;
            const hasUnsaved = localStorage.getItem(unsavedKey) !== null;
            if (hasUnsaved) {
                const confirmed = window.confirm(
                    `"${tab.title}" has unsaved changes. Are you sure you want to close this tab? Your changes will be discarded.`
                );
                if (!confirmed) return;
                localStorage.removeItem(unsavedKey);
                localStorage.removeItem(`discard_unsaved_${tab.documentId}`);
            }
        } else if (tab && tab.type === 'document' && tab.documentId && skipUnsavedCheck) {
            const unsavedKey = `unsaved_content_${tab.documentId}`;
            localStorage.removeItem(unsavedKey);
            localStorage.removeItem(`discard_unsaved_${tab.documentId}`);
        }
        setTabs(prevTabs => {
            const updatedTabs = prevTabs.filter(t => t.id !== tabId);
            if (activeTabId === tabId) {
                setActiveTabId(updatedTabs.length > 0 ? updatedTabs[updatedTabs.length - 1].id : null);
            }
            return updatedTabs;
        });
    };

    const closeDocumentTab = (documentId) => {
        const docTabs = tabs.filter(t => t.type === 'document' && t.documentId === documentId);
        docTabs.forEach(t => closeTab(t.id, { skipUnsavedCheck: true }));
    };

    const updateDocumentTabTitle = (documentId, newTitle) => {
        setTabs(prevTabs => prevTabs.map(tab =>
            tab.type === 'document' && tab.documentId === documentId ? { ...tab, title: newTitle } : tab
        ));
    };

    const openRSSFeed = (feedId, feedName) => {
        // Check if tab already exists for this feed
        const existingTab = tabs.find(tab => tab.type === 'rss-feed' && tab.feedId === feedId);
        
        if (existingTab) {
            setActiveTabId(existingTab.id);
            return;
        }

        addTab({
            type: 'rss-feed',
            title: feedName,
            feedId: feedId,
            icon: '📰'
        });
    };

    const openNewsHeadlines = () => {
        const existingTab = tabs.find(tab => tab.type === 'news-headlines');
        if (existingTab) {
            setActiveTabId(existingTab.id);
            return;
        }
        addTab({
            type: 'news-headlines',
            title: 'News',
            icon: '📰'
        });
    };

    const openDocument = (documentId, documentName, options = {}) => {
        // Check if tab already exists for this document
        const existingTab = tabs.find(tab => tab.type === 'document' && tab.documentId === documentId);
        
        if (existingTab) {
            setActiveTabId(existingTab.id);
            // If scroll parameters are provided, we'll need to update the tab to include them
            if (options.scrollToLine || options.scrollToHeading) {
                setTabs(prevTabs => prevTabs.map(tab => 
                    tab.id === existingTab.id 
                        ? { ...tab, scrollToLine: options.scrollToLine, scrollToHeading: options.scrollToHeading }
                        : tab
                ));
            }
            return;
        }

        addTab({
            type: 'document',
            title: documentName,
            documentId: documentId,
            icon: '📄',
            scrollToLine: options.scrollToLine,
            scrollToHeading: options.scrollToHeading
        });
    };

    const openNote = (noteId, noteName) => {
        // Check if tab already exists for this note
        const existingTab = tabs.find(tab => tab.type === 'note' && tab.noteId === noteId);
        
        if (existingTab) {
            setActiveTabId(existingTab.id);
            return;
        }

        addTab({
            type: 'note',
            title: noteName,
            noteId: noteId,
            icon: '📝'
        });
    };

    // ROOSEVELT'S ORG VIEW OPENER
    const openOrgView = (viewType) => {
        // Map view types to tab configurations
        const viewConfigs = {
            'agenda': { title: 'Agenda', icon: '📅', type: 'org-agenda' },
            'search': { title: 'Search', icon: '🔍', type: 'org-search' },
            'todos': { title: 'TODOs', icon: '✅', type: 'org-todos' },
            'contacts': { title: 'Contacts', icon: '👤', type: 'org-contacts' },
            'tags': { title: 'Tags', icon: '🏷️', type: 'org-tags' }
        };

        const config = viewConfigs[viewType];
        if (!config) {
            console.error('Unknown org view type:', viewType);
            return;
        }

        // Check if tab already exists for this view type
        const existingTab = tabs.find(tab => tab.type === config.type);
        
        if (existingTab) {
            setActiveTabId(existingTab.id);
            return;
        }

        addTab({
            type: config.type,
            title: config.title,
            icon: config.icon
        });
    };

    const openDataWorkspace = (workspaceId) => {
        // Check if tab already exists for this workspace
        const existingTab = tabs.find(tab => tab.type === 'data-workspace' && tab.workspaceId === workspaceId);
        
        if (existingTab) {
            setActiveTabId(existingTab.id);
            return;
        }

        addTab({
            type: 'data-workspace',
            title: 'Data Workspace',
            icon: '📊',
            workspaceId: workspaceId
        });
    };

    const openFileGraph = (scope = 'all', folderId = null) => {
        const existingTab = tabs.find(tab => tab.type === 'file-graph');
        if (existingTab) {
            setActiveTabId(existingTab.id);
            return;
        }
        addTab({
            type: 'file-graph',
            title: 'File Relations',
            icon: '🔗',
            scope: scope || 'all',
            folderId: folderId || null
        });
    };

    const generateTabId = () => {
        return 'tab_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    };

    const getTabContent = (tab) => {
        switch (tab.type) {
            case 'news-headlines':
                return (
                    <NewsHeadlinesPane
                        onOpenArticle={(newsId) => {
                            try {
                                if (window?.history && typeof window.history.pushState === 'function') {
                                    window.history.pushState({}, '', `/news/${newsId}`);
                                    window.dispatchEvent(new PopStateEvent('popstate'));
                                } else {
                                    window.location.href = `/news/${newsId}`;
                                }
                            } catch {
                                window.location.href = `/news/${newsId}`;
                            }
                        }}
                    />
                );
            case 'rss-feed':
                return (
                    <RSSArticleViewer
                        feedId={tab.feedId}
                        onClose={() => closeTab(tab.id)}
                    />
                );
            case 'document':
                return (
                    <DocumentViewer
                        documentId={tab.documentId}
                        onClose={() => closeTab(tab.id)}
                        scrollToLine={tab.scrollToLine}
                        scrollToHeading={tab.scrollToHeading}
                        initialScrollPosition={tab.scrollPosition || 0}
                        onScrollChange={(scrollPos) => updateTabScrollPosition(tab.id, scrollPos)}
                    />
                );
            case 'note':
                return (
                    <div style={placeholderStyle}>
                        <h2>Note Editor</h2>
                        <p>Note: {tab.title}</p>
                        <p>Note ID: {tab.noteId}</p>
                        <p>Note editor component would be rendered here.</p>
                    </div>
                );
            case 'org-agenda':
                return (
                    <OrgAgendaView
                        onOpenDocument={(result) => {
                            console.log('📅 ROOSEVELT: Opening document from agenda:', result);
                            // Open document with scroll parameters
                            openDocument(result.documentId, result.documentName, {
                                scrollToLine: result.scrollToLine,
                                scrollToHeading: result.scrollToHeading
                            });
                        }}
                    />
                );
            case 'org-search':
                return (
                    <OrgSearchView
                        onOpenDocument={(result) => {
                            console.log('🔍 ROOSEVELT: Opening document from search:', result);
                            // Open document with scroll parameters
                            openDocument(result.documentId, result.documentName, {
                                scrollToLine: result.scrollToLine,
                                scrollToHeading: result.scrollToHeading
                            });
                        }}
                    />
                );
            case 'org-todos':
                return (
                    <OrgTodosView
                        onOpenDocument={(result) => {
                            console.log('✅ ROOSEVELT: Opening document from TODOs:', result);
                            // Open document with scroll parameters
                            openDocument(result.documentId, result.documentName, {
                                scrollToLine: result.scrollToLine,
                                scrollToHeading: result.scrollToHeading
                            });
                        }}
                    />
                );
            case 'org-contacts':
                return (
                    <OrgContactsView
                        onOpenDocument={(result) => {
                            console.log('👤 ROOSEVELT: Opening document from Contacts:', result);
                            // Open document with scroll parameters
                            openDocument(result.documentId, result.documentName, {
                                scrollToLine: result.scrollToLine,
                                scrollToHeading: result.scrollToHeading
                            });
                        }}
                    />
                );
            case 'org-tags':
                return (
                    <div style={placeholderStyle}>
                        <div style={{ maxWidth: 800, margin: '0 auto', textAlign: 'left' }}>
                            <h2 style={{ display: 'flex', alignItems: 'center', gap: 8, color: theme.palette.text.primary }}>
                                <span>🏷️</span> Tags Browser
                            </h2>
                            <p style={{ color: theme.palette.text.secondary, marginBottom: 24 }}>
                                Browse and explore content by org-mode tags.
                            </p>
                            <div style={{ 
                                background: theme.palette.mode === 'dark' ? 'rgba(255, 193, 7, 0.15)' : '#fff3cd', 
                                padding: 16, 
                                borderRadius: 8, 
                                border: `1px solid ${theme.palette.mode === 'dark' ? 'rgba(255, 193, 7, 0.5)' : '#ffc107'}`,
                                marginBottom: 16,
                                color: theme.palette.text.primary
                            }}>
                                <strong>🚧 Coming Soon!</strong>
                                <p style={{ marginTop: 8, marginBottom: 0 }}>
                                    This feature is under active development. It will provide:
                                </p>
                                <ul style={{ marginTop: 8, marginBottom: 0 }}>
                                    <li>Tag cloud showing all tags in use</li>
                                    <li>Click tags to see all items with that tag</li>
                                    <li>Combine multiple tags for refined filtering</li>
                                    <li>Tag statistics and usage patterns</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                );
            case 'data-workspace':
                return (
                    <DataWorkspaceManager
                        workspaceId={tab.workspaceId}
                        onClose={() => closeTab(tab.id)}
                    />
                );
            case 'file-graph':
                return (
                    <FileRelationGraph
                        onOpenDocument={(docId, docName) => openDocument(docId, docName)}
                        scope={tab.scope}
                        folderId={tab.folderId}
                    />
                );
            default:
                return (
                    <div style={placeholderStyle}>
                        <h2>Unknown Tab Type</h2>
                        <p>Tab type: {tab.type}</p>
                    </div>
                );
        }
    };

    // Expose methods to parent component via ref
    useImperativeHandle(ref, () => ({
        openRSSFeed,
        openDocument,
        openNewsHeadlines,
        openOrgView,
        openDataWorkspace,
        openFileGraph,
        closeDocumentTab,
        updateDocumentTabTitle
    }), [tabs]);

    const activeTab = tabs.find(tab => tab.id === activeTabId);

    // Theme-aware styles
    const containerStyle = {
        display: 'flex',
        flexDirection: 'column',
        height: '100%'
    };

    const tabBarStyle = {
        display: 'flex',
        backgroundColor: theme.palette.background.paper,
        borderBottom: `1px solid ${theme.palette.divider}`,
        padding: '0 16px',
        alignItems: 'center',
        minHeight: '44px',
        position: 'sticky',
        top: 0,
        zIndex: 1
    };

    const tabListStyle = {
        display: 'flex',
        flex: 1,
        overflow: 'hidden'
    };

    const tabStyle = {
        display: 'flex',
        alignItems: 'center',
        padding: '8px 16px',
        borderRight: `1px solid ${theme.palette.divider}`,
        cursor: 'pointer',
        backgroundColor: 'transparent',
        minWidth: '120px',
        maxWidth: '200px',
        position: 'relative',
        transition: 'background-color 0.2s ease'
    };

    const activeTabStyle = {
        backgroundColor: theme.palette.background.default,
        boxShadow: `inset 0 -2px 0 ${theme.palette.primary.main}`
    };

    const tabIconStyle = {
        marginRight: '8px',
        fontSize: '16px'
    };

    const tabTitleStyle = {
        flex: 1,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        fontSize: '13px',
        fontWeight: '500',
        color: theme.palette.text.primary
    };

    const closeTabButtonStyle = {
        border: 'none',
        background: 'none',
        fontSize: '16px',
        cursor: 'pointer',
        padding: '2px 6px',
        borderRadius: '4px',
        color: theme.palette.text.secondary,
        marginLeft: '8px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
    };

    const tabActionsStyle = {
        display: 'flex',
        gap: '8px',
        marginLeft: '16px'
    };

    const contentStyle = {
        flex: 1,
        overflow: 'hidden',
        backgroundColor: theme.palette.background.default
    };

    const placeholderStyle = {
        padding: '24px',
        textAlign: 'center',
        color: theme.palette.text.secondary
    };

    const emptyStateStyle = {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        textAlign: 'center',
        color: theme.palette.text.secondary
    };

    return (
        <div style={containerStyle}>
            {/* Tab Bar - visually merged with content via shared border and background */}
            <div style={tabBarStyle}>
                <div style={tabListStyle}>
                    {tabs.map((tab) => {
                        const diffCount = tabDiffCounts[tab.id] || 0;
                        const hasPendingDiffs = diffCount > 0;
                        
                        return (
                        <div
                            key={tab.id}
                            style={{
                                ...tabStyle,
                                ...(activeTabId === tab.id ? activeTabStyle : {})
                            }}
                            onClick={() => setActiveTabId(tab.id)}
                        >
                            <span style={tabIconStyle}>{tab.icon}</span>
                            <span style={tabTitleStyle}>{tab.title}</span>
                            
                            {/* Diff badge */}
                            {hasPendingDiffs && (
                                <span style={{
                                    backgroundColor: '#ff9800',
                                    color: 'white',
                                    borderRadius: '10px',
                                    padding: '2px 6px',
                                    fontSize: '11px',
                                    fontWeight: 'bold',
                                    marginLeft: '6px',
                                    minWidth: '18px',
                                    textAlign: 'center',
                                    display: 'inline-block'
                                }}>
                                    {diffCount}
                                </span>
                            )}
                            
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    closeTab(tab.id);
                                }}
                                style={closeTabButtonStyle}
                            >
                                ×
                            </button>
                        </div>
                        );
                    })}
                </div>
                
                <div style={tabActionsStyle} />
            </div>

            {/* Tab Content */}
            <div style={contentStyle}>
                {activeTab ? (
                    getTabContent(activeTab)
                ) : (
                    <div style={emptyStateStyle}>
                        {/* Empty state - blank content area */}
                    </div>
                )}
            </div>

            {/* RSS Feed Manager Modal */}
            <RSSFeedManager
                isOpen={showFeedManager}
                onClose={() => setShowFeedManager(false)}
                onFeedAdded={(newFeed) => {
                    setShowFeedManager(false);
                    // The feed will be added to the navigation, and clicking it will open a tab
                }}
            />
        </div>
    );
});

TabbedContentManager.displayName = 'TabbedContentManager';

export default TabbedContentManager;
