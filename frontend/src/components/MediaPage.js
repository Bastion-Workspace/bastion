import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert,
  Divider,
  IconButton,
  Tabs,
  Tab,
  Checkbox,
  Menu,
  MenuItem,
  ListItemIcon,
  ListSubheader,
  TextField,
  InputAdornment,
  Button,
} from '@mui/material';
import {
  Album,
  Person,
  PlaylistPlay,
  ArrowBack,
  LibraryMusic,
  Headphones,
  Podcasts,
  Add,
  Remove,
  Search,
  Clear,
  Tv,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import apiService from '../services/apiService';
import { useMusic } from '../contexts/MediaContext';
import EmbyBrowseView from './video/EmbyBrowseView';
import SplitResizeHandle from './common/SplitResizeHandle';
import { solidSurfaceBg } from '../theme/wallpaperPaneSx';

const MEDIA_PAGE_SIDEBAR_MIN = 220;
const MEDIA_PAGE_SIDEBAR_MAX = 600;
const MEDIA_PAGE_SIDEBAR_DEFAULT = 250;

/** One-line artist credit for sidebar lists (VA / compilation albums). */
function formatCompactAlbumArtist(artist, { maxNames = 2, maxChars = 72 } = {}) {
  if (!artist || typeof artist !== 'string') return '';
  const s = artist.trim();
  if (!s) return '';
  let parts;
  if (/[•·]/.test(s)) {
    parts = s.split(/\s*[•·]\s*/).map((p) => p.trim()).filter(Boolean);
  } else if (s.includes(',')) {
    parts = s.split(/\s*,\s*/).map((p) => p.trim()).filter(Boolean);
  } else {
    parts = [s];
  }
  if (parts.length <= 1) {
    return s.length > maxChars ? `${s.slice(0, maxChars - 1).trimEnd()}…` : s;
  }
  const shown = parts.slice(0, maxNames).join(', ');
  const extra = parts.length - maxNames;
  let out = extra > 0 ? `${shown} +${extra}` : shown;
  if (out.length > maxChars) {
    out = `${out.slice(0, maxChars - 1).trimEnd()}…`;
  }
  return out;
}

const listSecondaryTypographyProps = {
  noWrap: true,
  sx: { overflow: 'hidden', textOverflow: 'ellipsis' },
};

function idKeyEq(a, b) {
  if (a == null || b == null) return false;
  return String(a) === String(b);
}

/** Subsonic often omits per-track cover; use album id when enqueueing. */
function enrichTracksWithAlbumCover(tracks, albumCoverId) {
  if (!tracks?.length || !albumCoverId || !String(albumCoverId).trim()) return tracks;
  const cid = String(albumCoverId).trim();
  return tracks.map((t) => ({
    ...t,
    cover_art_id: (t.cover_art_id && String(t.cover_art_id).trim()) || cid,
  }));
}

const MediaPage = () => {
  // Load persisted state from localStorage on mount
  const loadPersistedState = () => {
    try {
      const saved = localStorage.getItem('mediaPageState');
      if (saved) {
        const parsed = JSON.parse(saved);
        return {
          activeTab: parsed.activeTab ?? 0,
          selectedView: parsed.selectedView ?? 'albums',
          selectedItem: parsed.selectedItem ?? null,
          selectedItemType: parsed.selectedItemType ?? null,
          selectedArtist: parsed.selectedArtist ?? null,
          selectedSeries: parsed.selectedSeries ?? null,
          selectedAuthor: parsed.selectedAuthor ?? null,
          searchQuery: parsed.searchQuery ?? '',
          itemsToShow: parsed.itemsToShow ?? 200,
        };
      }
    } catch (e) {
      console.error('Failed to load media page state:', e);
    }
    return {
      activeTab: 0,
      selectedView: 'albums',
      selectedItem: null,
      selectedItemType: null,
      selectedArtist: null,
      selectedSeries: null,
      selectedAuthor: null,
      searchQuery: '',
      itemsToShow: 200,
    };
  };

  const initialState = loadPersistedState();
  
  const [activeTab, setActiveTab] = useState(initialState.activeTab); // 0: Music, 1: Audiobooks, 2: Podcasts
  const [selectedView, setSelectedView] = useState(initialState.selectedView); // 'albums', 'artists', 'playlists'
  const [selectedItem, setSelectedItem] = useState(initialState.selectedItem);
  const [selectedItemType, setSelectedItemType] = useState(initialState.selectedItemType);
  const [selectedArtist, setSelectedArtist] = useState(initialState.selectedArtist); // Track selected artist for hierarchical nav
  const [selectedSeries, setSelectedSeries] = useState(initialState.selectedSeries); // Track selected series for Audiobooks
  const [selectedAuthor, setSelectedAuthor] = useState(initialState.selectedAuthor); // Track selected author for Audiobooks series navigation
  const [selectedTracks, setSelectedTracks] = useState(new Set()); // Multi-select tracks
  const [contextMenu, setContextMenu] = useState(null); // Context menu state
  const [searchQuery, setSearchQuery] = useState(initialState.searchQuery); // Search filter
  const [itemsToShow, setItemsToShow] = useState(initialState.itemsToShow); // Pagination: show first 200 items
  const [embyMusicMode, setEmbyMusicMode] = useState(false);
  const [trackSortField, setTrackSortField] = useState(() => {
    // Load from localStorage
    try {
      return localStorage.getItem('mediaTrackSortField') || 'track_number';
    } catch {
      return 'track_number';
    }
  });
  const [trackSortDirection, setTrackSortDirection] = useState(() => {
    // Load from localStorage
    try {
      return localStorage.getItem('mediaTrackSortDirection') || 'asc';
    } catch {
      return 'asc';
    }
  });
  const { playTrack, shuffleMode } = useMusic();

  const [mediaSidebarWidth, setMediaSidebarWidth] = useState(() => {
    try {
      const saved = localStorage.getItem('mediaPageSidebarWidth');
      if (saved) {
        const n = parseInt(saved, 10);
        if (!Number.isNaN(n)) {
          return Math.min(
            MEDIA_PAGE_SIDEBAR_MAX,
            Math.max(MEDIA_PAGE_SIDEBAR_MIN, n)
          );
        }
      }
    } catch (_) {}
    return MEDIA_PAGE_SIDEBAR_DEFAULT;
  });
  const [mediaSidebarResizing, setMediaSidebarResizing] = useState(false);
  const mediaSidebarResizingRef = useRef(false);

  useEffect(() => {
    try {
      localStorage.setItem('mediaPageSidebarWidth', String(mediaSidebarWidth));
    } catch (_) {}
  }, [mediaSidebarWidth]);

  const handleMediaSidebarResizeMove = useCallback((e) => {
    if (!mediaSidebarResizingRef.current) return;
    const next = e.clientX;
    setMediaSidebarWidth(
      Math.min(MEDIA_PAGE_SIDEBAR_MAX, Math.max(MEDIA_PAGE_SIDEBAR_MIN, next))
    );
  }, []);

  const handleMediaSidebarResizeEnd = useCallback(() => {
    mediaSidebarResizingRef.current = false;
    setMediaSidebarResizing(false);
  }, []);

  const handleMediaSidebarResizeStart = useCallback((e) => {
    e.preventDefault();
    mediaSidebarResizingRef.current = true;
    setMediaSidebarResizing(true);
  }, []);

  useEffect(() => {
    if (!mediaSidebarResizing) return undefined;
    document.addEventListener('mousemove', handleMediaSidebarResizeMove);
    document.addEventListener('mouseup', handleMediaSidebarResizeEnd);
    document.body.style.cursor = 'ew-resize';
    document.body.style.userSelect = 'none';
    return () => {
      document.removeEventListener('mousemove', handleMediaSidebarResizeMove);
      document.removeEventListener('mouseup', handleMediaSidebarResizeEnd);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [mediaSidebarResizing, handleMediaSidebarResizeMove, handleMediaSidebarResizeEnd]);

  // Persist state to localStorage whenever relevant state changes
  useEffect(() => {
    try {
      const stateToSave = {
        activeTab,
        selectedView,
        selectedItem,
        selectedItemType,
        selectedArtist,
        selectedSeries,
        selectedAuthor,
        searchQuery,
        itemsToShow,
      };
      localStorage.setItem('mediaPageState', JSON.stringify(stateToSave));
    } catch (e) {
      console.error('Failed to save media page state:', e);
    }
  }, [activeTab, selectedView, selectedItem, selectedItemType, selectedArtist, selectedSeries, selectedAuthor, searchQuery, itemsToShow]);

  // Fetch all configured sources
  const { data: sourcesData } = useQuery(
    'mediaSources',
    () => apiService.music.getSources(),
    {
      retry: false,
      refetchOnWindowFocus: false,
    }
  );

  const sources = sourcesData?.sources || [];
  const subsonicSource = sources.find(s => s.service_type === 'subsonic');
  const audiobookshelfSource = sources.find(s => s.service_type === 'audiobookshelf');
  const embySource = sources.find(s => s.service_type === 'emby');
  const hasSubsonic = !!subsonicSource;
  const hasAudiobookshelf = !!audiobookshelfSource;
  const hasEmby = !!embySource;

  const availableTabs = useMemo(
    () =>
      [
        { label: 'Music', icon: <LibraryMusic />, enabled: hasSubsonic, serviceType: 'subsonic' },
        { label: 'Audiobooks', icon: <Headphones />, enabled: hasAudiobookshelf, serviceType: 'audiobookshelf' },
        { label: 'Podcasts', icon: <Podcasts />, enabled: hasAudiobookshelf, serviceType: 'audiobookshelf' },
        { label: 'Video', icon: <Tv />, enabled: hasEmby, serviceType: 'emby', isEmby: true },
      ].filter((t) => t.enabled),
    [hasSubsonic, hasAudiobookshelf, hasEmby]
  );

  const tabMeta = availableTabs[activeTab];
  const isMusicTab = tabMeta?.label === 'Music';
  const isAudiobookTab = tabMeta?.label === 'Audiobooks';
  const isPodcastTab = tabMeta?.label === 'Podcasts';
  const isEmbyTab = Boolean(tabMeta?.isEmby);
  const serviceType = tabMeta?.serviceType ?? null;

  // Fetch library for the active tab's service
  const { data: library, isLoading: loadingLibrary, error: libraryError } = useQuery(
    ['mediaLibrary', serviceType],
    () => {
      if (!serviceType) {
        console.warn('getLibrary called without serviceType');
        return Promise.resolve({ albums: [], artists: [], playlists: [], last_sync_at: null });
      }
      console.log(`Fetching library for serviceType: ${serviceType}`);
      return apiService.music.getLibrary(serviceType);
    },
    {
      enabled: !!serviceType && sources.length > 0,
      refetchOnWindowFocus: false,
    }
  );

  // Debug logging
  React.useEffect(() => {
    if (library) {
      console.log(`📚 Library data for ${serviceType} (tab ${activeTab}):`, {
        albums: library.albums?.length || 0,
        artists: library.artists?.length || 0,
        playlists: library.playlists?.length || 0,
        selectedView,
        firstAlbum: library.albums?.[0],
        firstArtist: library.artists?.[0],
        firstPlaylist: library.playlists?.[0],
      });
    }
    if (libraryError) {
      console.error('❌ Library fetch error:', libraryError);
    }
  }, [library, libraryError, serviceType, activeTab, selectedView]);

  // Fetch series for selected author (Audiobooks only)
  const { data: seriesData, isLoading: loadingSeries } = useQuery(
    ['authorSeries', selectedAuthor, serviceType],
    () => apiService.music.getSeriesByAuthor(selectedAuthor, serviceType),
    {
      enabled: !!selectedAuthor && !!serviceType && isAudiobookTab,
      refetchOnWindowFocus: false,
    }
  );

  // Books for selected author (Audiobooks author drill-in; mirrors Music "albums by artist")
  const { data: authorBooksData, isLoading: loadingAuthorBooks } = useQuery(
    ['authorBooks', selectedAuthor, serviceType],
    () => apiService.music.getAlbumsByArtist(selectedAuthor, serviceType),
    {
      enabled: !!selectedAuthor && !!serviceType && isAudiobookTab && !selectedSeries,
      refetchOnWindowFocus: false,
    }
  );

  // Fetch albums for selected artist (or series if in Audiobooks)
  const { data: artistAlbumsData, isLoading: loadingArtistAlbums } = useQuery(
    ['artistAlbums', selectedArtist, selectedSeries, selectedAuthor, serviceType],
    () => {
      if (isAudiobookTab && selectedSeries && selectedAuthor) {
        // For Audiobooks, if series is selected, get books in that series
        const author = library?.artists?.find(a => a.id === selectedAuthor);
        return apiService.music.getAlbumsBySeries(selectedSeries, author?.name || '', serviceType);
      } else if (selectedArtist) {
        // Otherwise, get albums by artist
        return apiService.music.getAlbumsByArtist(selectedArtist, serviceType);
      }
      return Promise.resolve({ albums: [] });
    },
    {
      enabled: (!!selectedArtist || (!!selectedSeries && !!selectedAuthor)) && !!serviceType,
      refetchOnWindowFocus: false,
    }
  );

  // Fetch tracks for selected item
  const { data: tracksData, isLoading: loadingTracks, refetch: refetchTracks } = useQuery(
    ['mediaTracks', selectedItem, selectedItemType, serviceType],
    () => apiService.music.getTracks(selectedItem, selectedItemType, serviceType),
    {
      enabled: !!selectedItem && !!selectedItemType && selectedItemType !== 'artist' && !!serviceType,
      refetchOnWindowFocus: false,
    }
  );

  const resolvedHeaderCoverArtId = useMemo(() => {
    if (!selectedItem || !serviceType) return null;
    if (selectedItemType === 'album') {
      const album =
        library?.albums?.find((a) => idKeyEq(a.id, selectedItem)) ||
        artistAlbumsData?.albums?.find((a) => idKeyEq(a.id, selectedItem)) ||
        authorBooksData?.albums?.find((a) => idKeyEq(a.id, selectedItem));
      let cid =
        (album?.cover_art_id && String(album.cover_art_id).trim()) || null;
      if (!cid && tracksData?.tracks?.length) {
        const t = tracksData.tracks.find(
          (tr) => tr.cover_art_id && String(tr.cover_art_id).trim()
        );
        cid = (t?.cover_art_id && String(t.cover_art_id).trim()) || null;
      }
      return cid;
    }
    if (selectedItemType === 'playlist') {
      const pl = library?.playlists?.find((p) => idKeyEq(p.id, selectedItem));
      const meta = pl?.metadata && typeof pl.metadata === 'object' ? pl.metadata : {};
      const fromMeta =
        meta.coverArt ||
        meta.cover_art ||
        meta.coverArtId ||
        (meta.playlist && meta.playlist.coverArt);
      const raw = fromMeta || pl?.cover_art_id;
      return raw && String(raw).trim() ? String(raw).trim() : null;
    }
    return null;
  }, [
    selectedItem,
    selectedItemType,
    library?.albums,
    library?.playlists,
    artistAlbumsData?.albums,
    authorBooksData?.albums,
    tracksData?.tracks,
    serviceType,
  ]);

  const headerAlbumArtUrl = useMemo(() => {
    if (!resolvedHeaderCoverArtId) return null;
    return apiService.music.getCoverArtUrl(resolvedHeaderCoverArtId, serviceType, 200);
  }, [resolvedHeaderCoverArtId, serviceType]);

  const handleViewChange = (view) => {
    setSelectedView(view);
    setSelectedItem(null);
    setSelectedItemType(null);
    setSelectedArtist(null);
    setSelectedSeries(null);
    setSelectedAuthor(null);
    setSearchQuery(''); // Clear search when changing views
    setItemsToShow(200); // Reset pagination
  };

  const handleItemClick = (item, type) => {
    setSelectedTracks(new Set());

    if (type === 'artist') {
      if (isAudiobookTab) {
        setSelectedAuthor(item.id);
        setSelectedSeries(null);
        setSelectedArtist(null);
        setSelectedItem(null);
        setSelectedItemType(null);
      } else {
        setSelectedArtist(item.id);
        setSelectedItem(null);
        setSelectedItemType(null);
        setSelectedSeries(null);
        setSelectedAuthor(null);
      }
    } else {
      setSelectedItem(item.id);
      setSelectedItemType(type);
      setSelectedArtist(null);
      setSelectedSeries(null);
      setSelectedAuthor(null);
    }
  };

  const handleAlbumFromArtistClick = (album) => {
    // When clicking an album from artist view, show tracks
    // Keep selectedArtist so the sidebar stays on the artist's albums view
    setSelectedItem(album.id);
    setSelectedItemType('album');
    // Don't clear selectedArtist - keep it so sidebar stays consistent
  };

  const handleBackToArtists = () => {
    setSelectedArtist(null);
    setSelectedItem(null);
    setSelectedItemType(null);
    setSelectedSeries(null);
    setSelectedAuthor(null);
  };

  const handleItemDoubleClick = async (item, type) => {
    // Fetch tracks and play
    try {
      const tracks = await apiService.music.getTracks(item.id, type, serviceType);
      if (tracks.tracks && tracks.tracks.length > 0) {
        // If shuffle is enabled, pick a random track to start with
        const startTrack = shuffleMode && tracks.tracks.length > 1
          ? tracks.tracks[Math.floor(Math.random() * tracks.tracks.length)]
          : tracks.tracks[0];
        const albumCover =
          type === 'album' && item.cover_art_id
            ? String(item.cover_art_id).trim()
            : null;
        const meta = item.metadata && typeof item.metadata === 'object' ? item.metadata : {};
        const playlistCover =
          type === 'playlist'
            ? String(
                meta.coverArt ||
                  meta.cover_art ||
                  meta.coverArtId ||
                  item.cover_art_id ||
                  ''
              ).trim() || null
            : null;
        const fallbackCover = albumCover || playlistCover;
        const queueTracks = fallbackCover
          ? enrichTracksWithAlbumCover(tracks.tracks, fallbackCover)
          : tracks.tracks;
        playTrack(startTrack, queueTracks, item.id);
      }
    } catch (error) {
      console.error('Failed to play item:', error);
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
    setEmbyMusicMode(false);
    setSelectedItem(null);
    setSelectedItemType(null);
    setSelectedArtist(null);
    setSelectedSeries(null);
    setSelectedAuthor(null);
    const tab = availableTabs[newValue];
    if (!tab) return;
    if (tab.label === 'Music' || tab.isEmby) {
      setSelectedView('albums');
    } else if (tab.label === 'Audiobooks') {
      setSelectedView('albums');
    } else if (tab.label === 'Podcasts') {
      setSelectedView('playlists');
    }
  };

  const handleTrackDoubleClick = (track) => {
    const tracks = tracksData?.tracks || [];
    const queueTracks =
      resolvedHeaderCoverArtId &&
      (selectedItemType === 'album' || selectedItemType === 'playlist')
        ? enrichTracksWithAlbumCover(tracks, resolvedHeaderCoverArtId)
        : tracks;
    playTrack(track, queueTracks, selectedItem);
  };

  const formatDuration = (seconds) => {
    if (!seconds || seconds === 0) return '0:00';
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    if (hours > 0) {
      return `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Multi-select handlers
  const handleSelectTrack = (trackId) => {
    const newSelected = new Set(selectedTracks);
    if (newSelected.has(trackId)) {
      newSelected.delete(trackId);
    } else {
      newSelected.add(trackId);
    }
    setSelectedTracks(newSelected);
  };

  const handleSelectAllTracks = () => {
    if (selectedTracks.size === tracksData?.tracks?.length) {
      setSelectedTracks(new Set());
    } else {
      const allTrackIds = new Set(tracksData?.tracks?.map(t => t.id) || []);
      setSelectedTracks(allTrackIds);
    }
  };

  // Context menu handlers
  const handleContextMenu = (event, track) => {
    event.preventDefault();
    setContextMenu({
      mouseX: event.clientX - 2,
      mouseY: event.clientY - 4,
      track,
    });
  };

  const handleCloseContextMenu = () => {
    setContextMenu(null);
  };

  const handleAddToPlaylist = async (playlistId) => {
    try {
      // Get selected track IDs or the right-clicked track
      const trackIds = selectedTracks.size > 0 
        ? Array.from(selectedTracks)
        : [contextMenu.track.id];
      
      // Call API to add tracks to playlist
      await apiService.music.addTracksToPlaylist(playlistId, trackIds, serviceType);
      
      console.log(`Added ${trackIds.length} track(s) to playlist`);
      
      handleCloseContextMenu();
      setSelectedTracks(new Set());
    } catch (error) {
      console.error('Failed to add tracks to playlist:', error);
      alert(`Failed to add tracks to playlist: ${error.message || error}`);
    }
  };

  const handleRemoveFromPlaylist = async () => {
    try {
      // Get selected track IDs or the right-clicked track
      const trackIds = selectedTracks.size > 0 
        ? Array.from(selectedTracks)
        : [contextMenu.track.id];
      
      // Call API to remove tracks from playlist
      await apiService.music.removeTracksFromPlaylist(selectedItem, trackIds, serviceType);
      
      console.log(`Removed ${trackIds.length} track(s) from playlist`);
      
      handleCloseContextMenu();
      setSelectedTracks(new Set());
      
      // Refresh the track list
      refetchTracks();
    } catch (error) {
      console.error('Failed to remove tracks from playlist:', error);
      alert(`Failed to remove tracks from playlist: ${error.message || error}`);
    }
  };

  const renderSidebar = () => {
    if (loadingLibrary) {
      return (
        <Box display="flex" justifyContent="center" p={3}>
          <CircularProgress />
        </Box>
      );
    }

    if (libraryError) {
      return (
        <Alert severity="error" sx={{ m: 2 }}>
          Failed to load library. Please check your music service configuration in Settings.
        </Alert>
      );
    }

    if (!library || (!library.albums?.length && !library.artists?.length && !library.playlists?.length)) {
      const serviceName = isMusicTab ? 'SubSonic' : isAudiobookTab || isPodcastTab ? 'Audiobookshelf' : 'Emby';
      const kind = isMusicTab ? 'music' : isAudiobookTab ? 'audiobook' : isPodcastTab ? 'podcast' : 'media';
      return (
        <Alert severity="info" sx={{ m: 2 }}>
          No {kind} library found. Configure your {serviceName} server in Settings &gt; Media and refresh the cache.
        </Alert>
      );
    }

    // If an author is selected in Audiobooks, show their books and optional series
    if (isAudiobookTab && selectedAuthor && !selectedSeries) {
      const author = library?.artists?.find(a => a.id === selectedAuthor);
      const series = seriesData?.series || [];
      const authorBooks = authorBooksData?.albums ?? [];

      return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
          <Box sx={{ flexShrink: 0 }}>
            <List>
              <ListItem disablePadding>
                <ListItemButton onClick={handleBackToArtists} sx={{ py: 0.5 }}>
                  <ArrowBack sx={{ mr: 1 }} />
                  <ListItemText primary="Back to Authors" />
                </ListItemButton>
              </ListItem>
            </List>
            <Divider sx={{ my: 1 }} />
            <List>
              <ListItem disablePadding>
                <ListItemText
                  primary={<strong>{author?.name || 'Author'}</strong>}
                  secondary="Books"
                  sx={{ px: 2, py: 1 }}
                />
              </ListItem>
            </List>
            <Divider sx={{ my: 1 }} />
          </Box>

          <Box
            sx={{
              flex: 1,
              overflowY: 'auto',
              overflowX: 'hidden',
              minHeight: 0,
              position: 'relative',
              '&::-webkit-scrollbar': {
                width: '8px',
              },
              '&::-webkit-scrollbar-track': {
                backgroundColor: 'transparent',
              },
              '&::-webkit-scrollbar-thumb': {
                backgroundColor: 'rgba(0,0,0,0.2)',
                borderRadius: '4px',
              },
            }}
          >
            {loadingAuthorBooks && authorBooks.length === 0 ? (
              <Box display="flex" justifyContent="center" p={2}>
                <CircularProgress size={24} />
              </Box>
            ) : (
              <List sx={{ padding: 0 }} subheader={<li />}>
                <ListSubheader component="div" sx={{ lineHeight: 2, bgcolor: 'background.paper' }}>
                  Books
                </ListSubheader>
                {authorBooks.length > 0 ? (
                  authorBooks.map((album) => (
                    <ListItem key={album.id} disablePadding>
                      <ListItemButton
                        selected={selectedItem === album.id}
                        onClick={() => handleAlbumFromArtistClick(album)}
                        onDoubleClick={() => handleItemDoubleClick(album, 'album')}
                        sx={{ py: 0.5 }}
                      >
                        <ListItemText
                          primary={album.title}
                          secondary={formatCompactAlbumArtist(album.artist)}
                          secondaryTypographyProps={listSecondaryTypographyProps}
                        />
                      </ListItemButton>
                    </ListItem>
                  ))
                ) : !loadingAuthorBooks ? (
                  <ListItem>
                    <ListItemText primary="No books found" secondary="Try refreshing the media cache" />
                  </ListItem>
                ) : null}

                {loadingSeries && series.length === 0 ? (
                  <Box display="flex" justifyContent="center" p={2}>
                    <CircularProgress size={24} />
                  </Box>
                ) : series.length > 0 ? (
                  <>
                    <ListSubheader component="div" sx={{ lineHeight: 2, bgcolor: 'background.paper' }}>
                      Series
                    </ListSubheader>
                    {series.map((seriesItem) => (
                      <ListItem key={seriesItem.id} disablePadding>
                        <ListItemButton
                          onClick={() => {
                            setSelectedSeries(seriesItem.name);
                            setSelectedItem(null);
                            setSelectedItemType(null);
                          }}
                          sx={{ py: 0.5 }}
                        >
                          <ListItemText
                            primary={seriesItem.name}
                            secondary={`${seriesItem.book_count || 0} books`}
                          />
                        </ListItemButton>
                      </ListItem>
                    ))}
                  </>
                ) : null}
              </List>
            )}
          </Box>
        </Box>
      );
    }

    // If a series is selected in Audiobooks, show books in that series
    if (isAudiobookTab && selectedSeries && selectedAuthor) {
      const author = library?.artists?.find(a => a.id === selectedAuthor);
      const albums = artistAlbumsData?.albums || [];
      
      return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
          {/* Back button and header - fixed */}
          <Box sx={{ flexShrink: 0 }}>
            <List>
              <ListItem disablePadding>
                <ListItemButton 
                  onClick={() => {
                    setSelectedSeries(null);
                    setSelectedItem(null);
                    setSelectedItemType(null);
                  }}
                  sx={{ py: 0.5 }}
                >
                  <ArrowBack sx={{ mr: 1 }} />
                  <ListItemText primary={`Back to ${author?.name || 'Author'}'s Series`} />
                </ListItemButton>
              </ListItem>
            </List>
            <Divider sx={{ my: 1 }} />
            <List>
              <ListItem disablePadding>
                <ListItemText 
                  primary={<strong>{selectedSeries}</strong>}
                  secondary={`by ${author?.name || 'Author'}`}
                  sx={{ px: 2, py: 1 }}
                />
              </ListItem>
            </List>
            <Divider sx={{ my: 1 }} />
          </Box>
          
          {/* Albums list - scrollable */}
          <Box 
            sx={{ 
              flex: 1, 
              overflowY: 'auto',
              overflowX: 'hidden',
              minHeight: 0,
              position: 'relative',
              '&::-webkit-scrollbar': {
                width: '8px',
              },
              '&::-webkit-scrollbar-track': {
                backgroundColor: 'transparent',
              },
              '&::-webkit-scrollbar-thumb': {
                backgroundColor: 'rgba(0,0,0,0.2)',
                borderRadius: '4px',
              },
            }}
          >
            {loadingArtistAlbums ? (
              <Box display="flex" justifyContent="center" p={2}>
                <CircularProgress size={24} />
              </Box>
            ) : albums.length > 0 ? (
              <List sx={{ padding: 0 }}>
                {albums.map((album) => (
                  <ListItem key={album.id} disablePadding>
                    <ListItemButton onClick={() => handleAlbumFromArtistClick(album)} sx={{ py: 0.5 }}>
                      <ListItemText primary={album.title} />
                    </ListItemButton>
                  </ListItem>
                ))}
              </List>
            ) : (
              <List>
                <ListItem>
                  <ListItemText primary="No books found in this series" />
                </ListItem>
              </List>
            )}
          </Box>
        </Box>
      );
    }

    // If an artist is selected (Music or Emby music mode), show their albums
    if (selectedArtist && (isMusicTab || (isEmbyTab && embyMusicMode))) {
      const artist = library?.artists?.find(a => a.id === selectedArtist);
      return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
          {/* Back button and header - fixed */}
          <Box sx={{ flexShrink: 0 }}>
            <List>
              <ListItem disablePadding>
                <ListItemButton onClick={handleBackToArtists} sx={{ py: 0.5 }}>
                  <ArrowBack sx={{ mr: 1 }} />
                  <ListItemText primary="Back to Artists" />
                </ListItemButton>
              </ListItem>
            </List>
            <Divider sx={{ my: 1 }} />
            <List>
              <ListItem disablePadding>
                <ListItemText 
                  primary={artist?.name || 'Artist'} 
                  secondary="Albums"
                  sx={{ px: 2, py: 1 }}
                />
              </ListItem>
            </List>
            <Divider sx={{ my: 1 }} />
          </Box>
          
          {/* Albums list - scrollable */}
          <Box 
            sx={{ 
              flex: 1, 
              overflowY: 'auto',
              overflowX: 'hidden',
              minHeight: 0,
              position: 'relative',
              '&::-webkit-scrollbar': {
                width: '8px',
              },
              '&::-webkit-scrollbar-track': {
                backgroundColor: 'transparent',
              },
              '&::-webkit-scrollbar-thumb': {
                backgroundColor: 'rgba(0,0,0,0.2)',
                borderRadius: '4px',
              },
            }}
          >
            {loadingArtistAlbums ? (
              <Box display="flex" justifyContent="center" p={2}>
                <CircularProgress size={24} />
              </Box>
            ) : (
              <List sx={{ padding: 0 }}>
                {(artistAlbumsData?.albums || []).map((album) => (
                  <ListItem key={album.id} disablePadding>
                    <ListItemButton
                      selected={selectedItem === album.id}
                      onClick={() => handleAlbumFromArtistClick(album)}
                      onDoubleClick={() => handleItemDoubleClick(album, 'album')}
                      sx={{ py: 0.5 }}
                    >
                      <ListItemText
                        primary={album.title}
                        secondary={formatCompactAlbumArtist(album.artist)}
                        secondaryTypographyProps={listSecondaryTypographyProps}
                      />
                    </ListItemButton>
                  </ListItem>
                ))}
              </List>
            )}
          </Box>
        </Box>
      );
    }

    // Context-aware navigation based on active tab
    let navItems = [];
    let items = {};
    
    if (isMusicTab || (isEmbyTab && embyMusicMode)) {
      // Music tab - Albums, Artists, Playlists
      navItems = [
        { key: 'albums', label: 'Albums', icon: <Album /> },
        { key: 'artists', label: 'Artists', icon: <Person /> },
        { key: 'playlists', label: 'Playlists', icon: <PlaylistPlay /> },
      ];
      items = {
        albums: library.albums || [],
        artists: library.artists || [],
        playlists: library.playlists || [],
      };
    } else if (isAudiobookTab) {
      // Audiobooks tab - Books (albums), Authors (artists)
      navItems = [
        { key: 'albums', label: 'Books', icon: <Album /> },
        { key: 'artists', label: 'Authors', icon: <Person /> },
      ];
      items = {
        albums: library.albums || [],
        artists: library.artists || [],
        playlists: [],
      };
    } else {
      // Podcasts tab - Shows (playlists)
      navItems = [
        { key: 'playlists', label: 'Shows', icon: <PlaylistPlay /> },
      ];
      items = {
        albums: [],
        artists: [],
        playlists: library.playlists || [],
      };
    }

    const currentItems = items[selectedView] || [];
    
    // Filter items based on search query (using regular JS, not hook)
    let filteredItems = currentItems;
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filteredItems = currentItems.filter(item => {
        const title = (item.title || item.name || '').toLowerCase();
        const artist = (item.artist || '').toLowerCase();
        return title.includes(query) || artist.includes(query);
      });
    }
    
    // Paginate: show only first N items
    const displayedItems = filteredItems.slice(0, itemsToShow);
    const hasMore = filteredItems.length > itemsToShow;
    const totalCount = filteredItems.length;
    
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
        {/* Navigation buttons - fixed at top */}
        <Box sx={{ flexShrink: 0 }}>
          <List>
            {navItems.map((navItem) => (
              <ListItem key={navItem.key} disablePadding>
                <ListItemButton
                  selected={selectedView === navItem.key}
                  onClick={() => handleViewChange(navItem.key)}
                  sx={{ py: 0.5 }}
                >
                  {React.cloneElement(navItem.icon, { sx: { mr: 1 } })}
                  <ListItemText primary={navItem.label} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Box>
        
        <Divider sx={{ my: 1, flexShrink: 0 }} />
        
        {/* Search box - fixed below nav */}
        {currentItems.length > 50 && (
          <Box sx={{ px: 1, pb: 1, flexShrink: 0 }}>
            <TextField
              fullWidth
              size="small"
              placeholder={`Search ${selectedView}...`}
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setItemsToShow(200); // Reset pagination when searching
              }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search fontSize="small" />
                  </InputAdornment>
                ),
                endAdornment: searchQuery && (
                  <InputAdornment position="end">
                    <IconButton
                      size="small"
                      onClick={() => {
                        setSearchQuery('');
                        setItemsToShow(200);
                      }}
                    >
                      <Clear fontSize="small" />
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />
            {searchQuery && (
              <Typography variant="caption" color="text.secondary" sx={{ px: 1, pt: 0.5, display: 'block' }}>
                Showing {displayedItems.length} of {totalCount}
              </Typography>
            )}
          </Box>
        )}
        
        {/* Items list - only this section scrolls */}
        <Box 
          sx={{ 
            flex: 1, 
            overflowY: 'auto',
            overflowX: 'hidden',
            minHeight: 0, // Critical for flex scrolling
            position: 'relative',
            '&::-webkit-scrollbar': {
              width: '8px',
            },
            '&::-webkit-scrollbar-track': {
              backgroundColor: 'transparent',
            },
            '&::-webkit-scrollbar-thumb': {
              backgroundColor: 'rgba(0,0,0,0.2)',
              borderRadius: '4px',
            },
          }}
        >
          <List sx={{ padding: 0 }}>
            {displayedItems.length === 0 ? (
              <ListItem>
                <ListItemText 
                  primary={searchQuery ? 'No matches found' : 'No items'} 
                  secondary={searchQuery ? `Try a different search term` : null}
                />
              </ListItem>
            ) : (
              displayedItems.map((item) => (
                <ListItem key={`${selectedView}-${item.id}`} disablePadding>
                  <ListItemButton
                    selected={selectedItem === item.id || selectedArtist === item.id}
                    onClick={() => handleItemClick(item, selectedView === 'albums' ? 'album' : selectedView === 'playlists' ? 'playlist' : 'artist')}
                    onDoubleClick={() => {
                      if (selectedView === 'albums' || selectedView === 'playlists') {
                        handleItemDoubleClick(item, selectedView === 'albums' ? 'album' : 'playlist');
                      }
                    }}
                    sx={{ py: 0.5, display: 'flex', alignItems: 'center' }}
                  >
                    {selectedView === 'albums' && item.cover_art_id ? (
                      <Box
                        component="img"
                        src={apiService.music.getCoverArtUrl(item.cover_art_id, serviceType, 64)}
                        alt=""
                        loading="lazy"
                        sx={{
                          width: 32,
                          height: 32,
                          borderRadius: 0.5,
                          mr: 1,
                          flexShrink: 0,
                          objectFit: 'cover',
                        }}
                      />
                    ) : null}
                    <ListItemText
                      primary={item.title || item.name}
                      secondary={
                        selectedView === 'albums'
                          ? formatCompactAlbumArtist(item.artist)
                          : null
                      }
                      secondaryTypographyProps={
                        selectedView === 'albums' ? listSecondaryTypographyProps : undefined
                      }
                      sx={{ flex: 1, minWidth: 0 }}
                    />
                  </ListItemButton>
                </ListItem>
              ))
            )}
          </List>
          
          {/* Load More button */}
          {hasMore && (
            <Box sx={{ p: 1, textAlign: 'center' }}>
              <Button
                size="small"
                onClick={() => setItemsToShow(prev => prev + 200)}
                variant="outlined"
              >
                Load More ({Math.min(200, totalCount - itemsToShow)} more)
              </Button>
            </Box>
          )}
        </Box>
      </Box>
    );
  };

  const renderTrackList = () => {
    if (!selectedItem) {
      return (
        <Box display="flex" alignItems="center" justifyContent="center" height="100%" p={3}>
          <Typography variant="body1" color="text.secondary">
            Select an album or playlist to view tracks
          </Typography>
        </Box>
      );
    }

    if (loadingTracks) {
      return (
        <Box display="flex" justifyContent="center" p={3}>
          <CircularProgress />
        </Box>
      );
    }

    if (!tracksData || !tracksData.tracks || tracksData.tracks.length === 0) {
      return (
        <Alert severity="info" sx={{ m: 2 }}>
          No tracks found
        </Alert>
      );
    }

    const isInPlaylist = selectedItemType === 'playlist';
    const isPodcast = isPodcastTab;
    const allSelected = selectedTracks.size === tracksData.tracks.length && tracksData.tracks.length > 0;

    // Sort tracks
    const sortedTracks = [...(tracksData.tracks || [])].sort((a, b) => {
      let aVal, bVal;
      
      switch (trackSortField) {
        case 'title':
          aVal = (a.title || '').toLowerCase();
          bVal = (b.title || '').toLowerCase();
          break;
        case 'published_date':
          aVal = a.metadata?.published_date || a.metadata?.publishedAt || '';
          bVal = b.metadata?.published_date || b.metadata?.publishedAt || '';
          break;
        case 'duration':
          aVal = a.duration || 0;
          bVal = b.duration || 0;
          break;
        case 'track_number':
        default:
          aVal = a.track_number || 0;
          bVal = b.track_number || 0;
          break;
      }
      
      if (aVal < bVal) return trackSortDirection === 'asc' ? -1 : 1;
      if (aVal > bVal) return trackSortDirection === 'asc' ? 1 : -1;
      return 0;
    });

    const handleSort = (field) => {
      const newDirection = trackSortField === field && trackSortDirection === 'asc' ? 'desc' : 'asc';
      setTrackSortField(field);
      setTrackSortDirection(newDirection);
      // Save to localStorage
      try {
        localStorage.setItem('mediaTrackSortField', field);
        localStorage.setItem('mediaTrackSortDirection', newDirection);
      } catch (e) {
        console.error('Failed to save sort preferences:', e);
      }
    };

    const formatDate = (dateString) => {
      if (!dateString) return '-';
      try {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
      } catch {
        return dateString;
      }
    };

    const SortableHeader = ({ field, children, width }) => (
      <TableCell 
        width={width} 
        align={field === 'duration' || field === 'published_date' ? 'right' : 'left'}
        sx={{ cursor: 'pointer', userSelect: 'none' }}
        onClick={() => handleSort(field)}
      >
        <Box display="flex" alignItems="center" gap={0.5}>
          {children}
          {trackSortField === field && (
            <Typography variant="caption" color="text.secondary">
              {trackSortDirection === 'asc' ? '↑' : '↓'}
            </Typography>
          )}
        </Box>
      </TableCell>
    );

    return (
      <>
        <TableContainer
          component={Paper}
          variant="outlined"
          sx={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
        >
          <Table size="small" sx={{ width: '100%' }}>
            <TableHead>
              <TableRow>
                <TableCell padding="checkbox">
                  <Checkbox
                    indeterminate={selectedTracks.size > 0 && selectedTracks.size < tracksData.tracks.length}
                    checked={allSelected}
                    onChange={handleSelectAllTracks}
                  />
                </TableCell>
                {!isPodcast && <SortableHeader field="track_number" width="5%">#</SortableHeader>}
                <SortableHeader field="title">Title</SortableHeader>
                {!isPodcast && <TableCell>Artist</TableCell>}
                {!isPodcast && <TableCell>Album</TableCell>}
                {isPodcast && <SortableHeader field="published_date" width="15%">Published</SortableHeader>}
                <SortableHeader field="duration" width="10%">Duration</SortableHeader>
              </TableRow>
            </TableHead>
            <TableBody>
              {sortedTracks.map((track, index) => {
                const isSelected = selectedTracks.has(track.id);
                return (
                  <TableRow
                    key={`${track.id}-${index}`}
                    hover
                    selected={isSelected}
                    onDoubleClick={() => handleTrackDoubleClick(track)}
                    onContextMenu={(e) => handleContextMenu(e, track)}
                    sx={{ cursor: 'pointer' }}
                  >
                    <TableCell padding="checkbox">
                      <Checkbox
                        checked={isSelected}
                        onChange={() => handleSelectTrack(track.id)}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </TableCell>
                    {!isPodcast && <TableCell>{track.track_number || index + 1}</TableCell>}
                    <TableCell>{track.title}</TableCell>
                    {!isPodcast && <TableCell>{track.artist || '-'}</TableCell>}
                    {!isPodcast && <TableCell>{track.album || '-'}</TableCell>}
                    {isPodcast && <TableCell align="right">{formatDate(track.metadata?.published_date || track.metadata?.publishedAt)}</TableCell>}
                    <TableCell align="right">{formatDuration(track.duration)}</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>

        {/* Context Menu */}
        <Menu
          open={Boolean(contextMenu)}
          onClose={handleCloseContextMenu}
          anchorReference="anchorPosition"
          anchorPosition={
            contextMenu !== null
              ? { top: contextMenu.mouseY, left: contextMenu.mouseX }
              : undefined
          }
        >
          {isInPlaylist ? (
            <MenuItem onClick={handleRemoveFromPlaylist}>
              <ListItemIcon>
                <Remove fontSize="small" />
              </ListItemIcon>
              Remove from Playlist
            </MenuItem>
          ) : (
            <>
              <MenuItem disabled>
                <ListItemIcon>
                  <Add fontSize="small" />
                </ListItemIcon>
                Add to Playlist
              </MenuItem>
              {library?.playlists?.map((playlist) => (
                <MenuItem 
                  key={playlist.id} 
                  onClick={() => handleAddToPlaylist(playlist.id)}
                  sx={{ pl: 4 }}
                >
                  {playlist.name}
                </MenuItem>
              ))}
            </>
          )}
        </Menu>
      </>
    );
  };

  // Keep activeTab in range and selectedView valid for the tab at that index (sources / tab strip can
  // change without handleTabChange, e.g. after refresh — otherwise persisted "playlists" on Music
  // or an invalid view for Audiobooks/Podcasts shows the wrong sidebar list).
  React.useEffect(() => {
    if (availableTabs.length === 0) {
      return;
    }
    const inRange = activeTab < availableTabs.length;
    if (!inRange) {
      setActiveTab(0);
    }
    const tabIndex = inRange ? activeTab : 0;
    const tab = availableTabs[tabIndex];
    if (!tab) return;
    if (tab.isEmby && !embyMusicMode) return;

    let validViews;
    if (tab.label === 'Podcasts') {
      validViews = ['playlists'];
    } else if (tab.label === 'Audiobooks') {
      validViews = ['albums', 'artists'];
    } else {
      validViews = ['albums', 'artists', 'playlists'];
    }
    if (!validViews.includes(selectedView)) {
      setSelectedView(validViews[0]);
    }
  }, [activeTab, availableTabs, embyMusicMode, selectedView]);

  if (availableTabs.length === 0) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <Alert severity="info">
          No media sources configured. Please configure a media source in Settings.
        </Alert>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        minWidth: 0,
        width: '100%',
        overflow: 'hidden',
      }}
    >
      {/* Category bar — match Documents / Agent Factory tab strip (44px, paper, chat header alignment) */}
      <Box
        sx={{
          flexShrink: 0,
          boxSizing: 'border-box',
          height: 44,
          minHeight: 44,
          pl: 0.5,
          pr: 2,
          display: 'flex',
          alignItems: 'center',
          backgroundColor: (theme) => solidSurfaceBg(theme),
          borderBottom: '1px solid',
          borderColor: 'divider',
        }}
      >
        {isEmbyTab && embyMusicMode && (
          <Button
            size="small"
            variant="outlined"
            sx={{ mr: 1, flexShrink: 0 }}
            onClick={() => setEmbyMusicMode(false)}
          >
            Video library
          </Button>
        )}
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          aria-label="media tabs"
          textColor="inherit"
          sx={{
            flex: 1,
            minHeight: 44,
            '& .MuiTabs-scroller': {
              overflow: 'hidden !important',
            },
            '& .MuiTabs-indicator': {
              display: 'none',
            },
            '& .MuiTabs-flexContainer': {
              alignItems: 'stretch',
              minHeight: 44,
              gap: 3,
            },
            '& .MuiTab-root': {
              minWidth: 'auto',
              minHeight: 44,
              maxHeight: 44,
              boxSizing: 'border-box',
              px: 1.5,
              py: 0,
              textTransform: 'none',
              fontSize: 13,
              fontWeight: 500,
              color: 'text.primary',
              opacity: 1,
              borderRadius: 0.75,
              '&.Mui-selected': {
                color: 'text.primary',
                fontWeight: 600,
                backgroundColor: 'background.default',
                boxShadow: (t) => `inset 0 -2px 0 ${t.palette.primary.main}`,
              },
              '&.Mui-disabled': { opacity: 0.45 },
            },
            '& .MuiTab-iconWrapper': {
              marginBottom: '0 !important',
              marginRight: 1,
              color: 'primary.main',
            },
          }}
        >
          {availableTabs.map((tab, index) => (
            <Tab
              key={tab.label}
              label={tab.label}
              icon={tab.icon}
              iconPosition="start"
              disabled={!tab.enabled}
            />
          ))}
        </Tabs>
      </Box>

      {isEmbyTab && !embyMusicMode ? (
        <EmbyBrowseView onOpenMusicLibrary={() => setEmbyMusicMode(true)} />
      ) : (
        <Box
          sx={{
            display: 'flex',
            flex: 1,
            minHeight: 0,
            minWidth: 0,
            width: '100%',
            overflow: 'hidden',
            position: 'relative',
            alignItems: 'stretch',
          }}
        >
          {/* Sidebar */}
          <Box
            sx={{
              width: mediaSidebarWidth,
              borderRight: 1,
              borderColor: 'divider',
              flexShrink: 0,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
              minHeight: 0,
              height: '100%',
              position: 'relative',
              backgroundColor: (theme) => solidSurfaceBg(theme),
              boxShadow: (theme) =>
                theme.palette.mode === 'dark'
                  ? '2px 0 12px rgba(0,0,0,0.35)'
                  : '2px 0 8px rgba(0,0,0,0.1)',
            }}
          >
            {renderSidebar()}
            <SplitResizeHandle
              edge="trailing"
              isResizing={mediaSidebarResizing}
              onMouseDown={handleMediaSidebarResizeStart}
            />
          </Box>

          {/* Main Content */}
          <Box
            sx={{
              flex: '1 1 0%',
              minWidth: 0,
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              minHeight: 0,
              position: 'relative',
              width: '100%',
              backgroundColor: (theme) => solidSurfaceBg(theme),
            }}
          >
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1.5,
                px: 2,
                pt: 0.5,
                pb: 0.5,
                flexShrink: 0,
                borderBottom: 1,
                borderColor: 'divider',
                backgroundColor: (theme) => solidSurfaceBg(theme),
              }}
            >
              {headerAlbumArtUrl ? (
                <Box
                  component="img"
                  src={headerAlbumArtUrl}
                  alt=""
                  loading="lazy"
                  sx={{
                    width: 100,
                    height: 100,
                    borderRadius: 1,
                    objectFit: 'cover',
                    flexShrink: 0,
                  }}
                />
              ) : null}
              <Typography variant="h6" sx={{ fontWeight: 500, my: 0 }}>
                {selectedArtist
                  ? library?.artists?.find((a) => a.id === selectedArtist)?.name || (isAudiobookTab ? 'Author' : 'Artist')
                  : selectedItem
                  ? selectedView === 'albums'
                    ? library?.albums?.find((a) => idKeyEq(a.id, selectedItem))?.title ||
                      artistAlbumsData?.albums?.find((a) => idKeyEq(a.id, selectedItem))?.title ||
                      authorBooksData?.albums?.find((a) => idKeyEq(a.id, selectedItem))?.title ||
                      (isAudiobookTab ? 'Book' : 'Album')
                    : selectedView === 'playlists'
                    ? library?.playlists?.find((p) => idKeyEq(p.id, selectedItem))?.name || (isPodcastTab ? 'Show' : 'Playlist')
                    : 'Tracks'
                  : isMusicTab
                  ? 'Music Library'
                  : isEmbyTab && embyMusicMode
                  ? 'Emby music'
                  : isAudiobookTab
                  ? 'Audiobooks'
                  : isPodcastTab
                  ? 'Podcasts'
                  : 'Media'}
              </Typography>
            </Box>
            <Box
              sx={{
                flex: '1 1 0%',
                overflow: 'auto',
                overflowX: 'hidden',
                minHeight: 0,
                minWidth: 0,
                width: '100%',
                p: 2,
                pt: 0,
                backgroundColor: (theme) => solidSurfaceBg(theme),
                '&::-webkit-scrollbar': {
                  width: '8px',
                },
                '&::-webkit-scrollbar-track': {
                  backgroundColor: 'transparent',
                },
                '&::-webkit-scrollbar-thumb': {
                  backgroundColor: 'rgba(0,0,0,0.2)',
                  borderRadius: '4px',
                },
              }}
            >
              {renderTrackList()}
            </Box>
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default MediaPage;

