/**
 * Tabbed Content Manager
 * Manages multiple tabs for RSS feeds, documents, and other content
 * Maximum 5 tabs with persistence across sessions
 */

import React, { useState, useEffect, forwardRef, useImperativeHandle, useCallback } from 'react';
import { useQuery } from 'react-query';
import { useTheme, alpha } from '@mui/material/styles';
import { useAuth } from '../contexts/AuthContext';
import apiService from '../services/apiService';
import { UI_WALLPAPER_QUERY_KEY } from '../config/uiWallpaperBuiltins';
import { isUiWallpaperConfigActive, MAIN_WORKSPACE_WALLPAPER_TINT_ALPHA } from '../theme/wallpaperPaneSx';
import RSSArticleViewer from './RSSArticleViewer';
import RSSFeedManager from './RSSFeedManager';
import DocumentViewer from './DocumentViewer';
import OrgSearchView from './OrgSearchView';
import OrgAgendaView from './OrgAgendaView';
import OrgTodosView from './OrgTodosView';
import OrgContactsView from './OrgContactsView';
import OrgTagsView from './OrgTagsView';
import DataWorkspaceManager from './data_workspace/DataWorkspaceManager';
import FileRelationGraph from './graph/FileRelationGraph';
import EntityRelationGraph from './graph/EntityRelationGraph';
import UnifiedKnowledgeGraph from './graph/UnifiedKnowledgeGraph';
import MapViewTabContent from './maps/MapViewTabContent';
import ArtifactViewerTab from './ArtifactViewerTab';
import OpdsHubTab from './OpdsHubTab';
import EbookReaderTab from './EbookReaderTab';
import { documentDiffStore } from '../services/documentDiffStore';
import { lockAndRemove } from '../services/encryptionSessionRegistry';
import { devLog } from '../utils/devConsole';

const TabbedContentManager = forwardRef((props, ref) => {
    const {
        onActiveDocumentIdChange,
        documentsFileTreeCollapsed = false,
        documentsIsMobile = false,
    } = props;
    const theme = useTheme();
    const { isAuthenticated, loading: authLoading } = useAuth();
    const { data: uiWallpaperData } = useQuery(
        [UI_WALLPAPER_QUERY_KEY],
        () => apiService.settings.getUserUiWallpaper(),
        { enabled: isAuthenticated && !authLoading, staleTime: 60_000 }
    );
    const wallpaperTintOnMainShell = isUiWallpaperConfigActive(uiWallpaperData?.config);
    const darkMode = theme.palette.mode === 'dark';
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
            devLog('🔔 TabbedContentManager: Diff change notification', { documentId, changeType, tabsCount: tabs.length });
            
            // Find all tabs with this documentId
            const matchingTabs = tabs.filter(t => t.type === 'document' && t.documentId === documentId);
            
            devLog('🔔 TabbedContentManager: Found matching tabs:', matchingTabs.length);
            
            matchingTabs.forEach(tab => {
                const diffs = documentDiffStore.getDiffs(documentId);
                const diffCount = diffs && Array.isArray(diffs.operations) ? diffs.operations.length : 0;
                
                devLog('🔔 TabbedContentManager: Updating tab badge', { tabId: tab.id, diffCount });
                
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

    // Notify parent when the active document tab changes (e.g. file tree highlight)
    useEffect(() => {
        if (typeof onActiveDocumentIdChange !== 'function') return;
        const active = tabs.find((tab) => tab.id === activeTabId);
        const docId =
            active && active.type === 'document' && active.documentId != null && active.documentId !== ''
                ? String(active.documentId)
                : null;
        onActiveDocumentIdChange(docId);
    }, [tabs, activeTabId, onActiveDocumentIdChange]);

    // Chat reads editor_ctx_cache when sending messages. Clear it whenever the user is not
    // on a real document tab so RSS/graph/etc. (or an empty tab bar) does not keep sending
    // a previously open manuscript as active_editor.
    useEffect(() => {
        if (typeof window === 'undefined') return;
        const active = activeTabId ? tabs.find((t) => t.id === activeTabId) : null;
        const onDocumentTab =
            !!active &&
            active.type === 'document' &&
            active.documentId != null &&
            String(active.documentId).trim() !== '';
        if (onDocumentTab) return;
        try {
            localStorage.removeItem('editor_ctx_cache');
        } catch (_) {
            /* ignore */
        }
    }, [tabs, activeTabId]);

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

    const updateFileGraphTabState = (tabId, updates) => {
        setTabs(prevTabs => prevTabs.map(tab =>
            tab.id === tabId ? { ...tab, ...updates } : tab
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
        const updatedTabsPreview = tabs.filter(t => t.id !== tabId);
        if (tab && tab.type === 'document' && tab.documentId) {
            const stillOpen = updatedTabsPreview.some(
                (t) => t.type === 'document' && t.documentId === tab.documentId
            );
            if (!stillOpen) {
                void lockAndRemove(tab.documentId);
            }
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

    const updateArtifactTabTitle = (artifactId, newTitle) => {
        setTabs(prevTabs => prevTabs.map(tab =>
            tab.type === 'artifact' && tab.artifactId === artifactId ? { ...tab, title: newTitle } : tab
        ));
    };

    const openRSSFeed = (feedId, feedName, articleId = null) => {
        const existingTab = tabs.find(tab => tab.type === 'rss-feed' && tab.feedId === feedId);

        if (existingTab) {
            setActiveTabId(existingTab.id);
            if (articleId) {
                setTabs(prevTabs =>
                    prevTabs.map(tab =>
                        tab.id === existingTab.id
                            ? { ...tab, initialArticleId: articleId }
                            : tab
                    )
                );
            }
            return;
        }

        addTab({
            type: 'rss-feed',
            title: feedName,
            feedId: feedId,
            icon: '📰',
            ...(articleId ? { initialArticleId: articleId } : {}),
        });
    };

    const openRSSCategory = (categoryTitle, feedIds, scope = 'user') => {
        if (!Array.isArray(feedIds) || feedIds.length === 0) return;
        const feedIdsKey = [...feedIds].map(String).sort().join('|');
        const existingTab = tabs.find(
            (tab) => tab.type === 'rss-category' && tab.feedIdsKey === feedIdsKey
        );
        if (existingTab) {
            setActiveTabId(existingTab.id);
            return;
        }
        const suffix = scope === 'global' ? ' (global)' : '';
        addTab({
            type: 'rss-category',
            title: `${categoryTitle}${suffix}`,
            feedIds: [...feedIds],
            feedIdsKey,
            scope,
            icon: '📁',
        });
    };

    const openDocument = useCallback((documentId, documentName, options = {}) => {
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
    }, [tabs]);

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
            'todos': { title: 'ToDos', icon: '✅', type: 'org-todos' },
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

    const openDataWorkspace = (workspaceId, workspaceName = null) => {
        // Check if tab already exists for this workspace
        const existingTab = tabs.find(tab => tab.type === 'data-workspace' && tab.workspaceId === workspaceId);
        
        if (existingTab) {
            setActiveTabId(existingTab.id);
            return;
        }

        addTab({
            type: 'data-workspace',
            title: workspaceName || 'Data Workspace',
            icon: '📊',
            workspaceId: workspaceId
        });
    };

    const openArtifact = (artifactId, title = null) => {
        if (!artifactId) return;
        const existingTab = tabs.find(tab => tab.type === 'artifact' && tab.artifactId === artifactId);
        if (existingTab) {
            setActiveTabId(existingTab.id);
            return;
        }
        addTab({
            type: 'artifact',
            title: title || 'Artifact',
            icon: '📚',
            artifactId,
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

    const openEntityGraph = () => {
        const existingTab = tabs.find(tab => tab.type === 'entity-graph');
        if (existingTab) {
            setActiveTabId(existingTab.id);
            return;
        }
        addTab({
            type: 'entity-graph',
            title: 'Entity Graph',
            icon: '🕸️'
        });
    };

    const openUnifiedGraph = () => {
        const existingTab = tabs.find(tab => tab.type === 'unified-graph');
        if (existingTab) {
            setActiveTabId(existingTab.id);
            return;
        }
        addTab({
            type: 'unified-graph',
            title: 'Unified Knowledge Graph',
            icon: '🕸️'
        });
    };

    const openMapView = (config = {}) => {
        const { title = 'Map', layers = [], style = 'auto', center, zoom } = config;
        const existingTab = tabs.find(tab => tab.type === 'map-view' && tab.title === title);
        if (existingTab) {
            setActiveTabId(existingTab.id);
            return;
        }
        addTab({
            type: 'map-view',
            title: title,
            icon: '🗺️',
            layers,
            style,
            center,
            zoom
        });
    };

    const openOPDSHub = useCallback(() => {
        const existingTab = tabs.find((t) => t.type === 'opds-hub');
        if (existingTab) {
            setActiveTabId(existingTab.id);
            return;
        }
        addTab({
            type: 'opds-hub',
            title: 'OPDS',
            icon: '📚',
        });
    }, [tabs]);

    const openEbookFromOpds = useCallback((payload) => {
        const { catalogId, acquisitionUrl, title, digest } = payload || {};
        if (!catalogId || !acquisitionUrl) return;
        addTab({
            type: 'ebook-reader',
            title: title || 'EPUB',
            icon: '📖',
            catalogId,
            acquisitionUrl,
            ebookTitle: title || 'EPUB',
            digest: digest || undefined,
        });
    }, []);

    const generateTabId = () => {
        return 'tab_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    };

    const getTabContent = (tab) => {
        switch (tab.type) {
            case 'rss-feed':
                return (
                    <RSSArticleViewer
                        feedId={tab.feedId}
                        initialArticleId={tab.initialArticleId}
                        onInitialArticleConsumed={() => {
                            setTabs(prevTabs =>
                                prevTabs.map(t =>
                                    t.id === tab.id ? { ...t, initialArticleId: undefined } : t
                                )
                            );
                        }}
                        onClose={() => closeTab(tab.id)}
                    />
                );
            case 'rss-category':
                return (
                    <RSSArticleViewer
                        feedIds={tab.feedIds}
                        viewerTitle={tab.title}
                        onClose={() => closeTab(tab.id)}
                    />
                );
            case 'document':
                return (
                    <DocumentViewer
                        key={tab.id}
                        documentId={tab.documentId}
                        onClose={() => closeTab(tab.id)}
                        scrollToLine={tab.scrollToLine}
                        scrollToHeading={tab.scrollToHeading}
                        initialScrollPosition={tab.scrollPosition || 0}
                        onScrollChange={(scrollPos) => updateTabScrollPosition(tab.id, scrollPos)}
                        onOpenDocument={openDocument}
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
                            devLog('Opening document from agenda:', result);
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
                            devLog('Opening document from search:', result);
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
                            devLog('Opening document from TODOs:', result);
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
                            devLog('Opening document from Contacts:', result);
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
                    <OrgTagsView
                        onOpenDocument={(result) => {
                            openDocument(result.documentId, result.documentName, {
                                scrollToLine: result.scrollToLine,
                                scrollToHeading: result.scrollToHeading
                            });
                        }}
                    />
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
                        persistedState={{
                            scope: tab.scope,
                            folderId: tab.folderId,
                            searchFilter: tab.searchFilter,
                            viewport: tab.viewport,
                        }}
                        onStateChange={(state) => updateFileGraphTabState(tab.id, state)}
                    />
                );
            case 'entity-graph':
                return (
                    <EntityRelationGraph
                        onOpenDocument={(docId, docName) => openDocument(docId, docName)}
                        persistedState={{
                            searchFilter: tab.searchFilter,
                            entityTypeFilter: tab.entityTypeFilter,
                            entityLimit: tab.entityLimit,
                            viewport: tab.viewport,
                        }}
                        onStateChange={(state) => updateFileGraphTabState(tab.id, state)}
                    />
                );
            case 'unified-graph':
                return (
                    <UnifiedKnowledgeGraph
                        onOpenDocument={(docId, docName) => openDocument(docId, docName)}
                        persistedState={tab}
                        onStateChange={(state) => updateFileGraphTabState(tab.id, state)}
                    />
                );
            case 'map-view':
                return (
                    <MapViewTabContent
                        tab={tab}
                        darkMode={darkMode}
                    />
                );
            case 'artifact':
                return (
                    <ArtifactViewerTab
                        artifactId={tab.artifactId}
                        onClose={() => closeTab(tab.id)}
                    />
                );
            case 'opds-hub':
                return (
                    <OpdsHubTab
                        onClose={() => closeTab(tab.id)}
                        onOpenEbook={(payload) => openEbookFromOpds(payload)}
                    />
                );
            case 'ebook-reader':
                return (
                    <EbookReaderTab
                        catalogId={tab.catalogId}
                        acquisitionUrl={tab.acquisitionUrl}
                        title={tab.ebookTitle || 'EPUB'}
                        digest={tab.digest}
                        documentsFileTreeCollapsed={documentsFileTreeCollapsed}
                        documentsIsMobile={documentsIsMobile}
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
        openRSSCategory,
        openDocument,
        openOrgView,
        openDataWorkspace,
        openArtifact,
        openFileGraph,
        openEntityGraph,
        openUnifiedGraph,
        openMapView,
        openOPDSHub,
        openEbookFromOpds,
        closeDocumentTab,
        updateDocumentTabTitle,
        updateArtifactTabTitle,
        /** True if a document tab for this id still exists (park session on viewer unmount when switching tabs). */
        shouldParkEncryptionSession: (documentId) => {
            if (!documentId) return false;
            return tabs.some((t) => t.type === 'document' && t.documentId === documentId);
        },
    }), [tabs, openOPDSHub, openEbookFromOpds]);

    // Keep global ref in sync so sidebar always gets latest API (avoids stale closure after closing tabs)
    useEffect(() => {
        if (typeof window !== 'undefined' && ref && ref.current) {
            window.tabbedContentManagerRef = ref.current;
        }
    }, [ref, tabs, activeTabId]);

    const activeTab = tabs.find(tab => tab.id === activeTabId);

    // Theme-aware styles
    const containerStyle = {
        display: 'flex',
        flexDirection: 'column',
        height: '100%'
    };

    const tabBarStyle = {
        display: 'flex',
        boxSizing: 'border-box',
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
        backgroundColor: alpha(theme.palette.background.default, 0.9),
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

    const isGraphTab = activeTab && ['file-graph', 'entity-graph', 'unified-graph'].includes(activeTab.type);
    const contentStyle = {
        flex: 1,
        overflow: 'hidden',
        minWidth: 0,
        backgroundColor: wallpaperTintOnMainShell
            ? 'transparent'
            : alpha(theme.palette.background.default, MAIN_WORKSPACE_WALLPAPER_TINT_ALPHA),
        ...(isGraphTab ? { display: 'flex', flexDirection: 'column' } : {})
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
