import React, { useState } from 'react';
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
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import apiService from '../services/apiService';
import { useMusic } from '../contexts/MediaContext';
import DeezerSearch from './music/DeezerSearch';

const MediaPage = () => {
  const [activeTab, setActiveTab] = useState(0); // 0: Music, 1: Audiobooks, 2: Podcasts
  const [selectedView, setSelectedView] = useState('albums'); // 'albums', 'artists', 'playlists'
  const [selectedItem, setSelectedItem] = useState(null);
  const [selectedItemType, setSelectedItemType] = useState(null);
  const [selectedArtist, setSelectedArtist] = useState(null); // Track selected artist for hierarchical nav
  const [selectedSeries, setSelectedSeries] = useState(null); // Track selected series for Audiobooks
  const [selectedAuthor, setSelectedAuthor] = useState(null); // Track selected author for Audiobooks series navigation
  const [selectedTracks, setSelectedTracks] = useState(new Set()); // Multi-select tracks
  const [contextMenu, setContextMenu] = useState(null); // Context menu state
  const [searchQuery, setSearchQuery] = useState(''); // Search filter
  const [itemsToShow, setItemsToShow] = useState(200); // Pagination: show first 200 items
  const [searchDialogOpen, setSearchDialogOpen] = useState(false); // Deezer search dialog
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
  const { playTrack } = useMusic();

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
  const deezerSource = sources.find(s => s.service_type === 'deezer');

  // Determine which service type to use based on active tab
  const getServiceType = () => {
    if (activeTab === 0) {
      // Music tab - use SubSonic if available, then Deezer, otherwise fall back to first available source
      if (subsonicSource) return 'subsonic';
      if (deezerSource) return 'deezer';
      return sources.length > 0 ? sources[0].service_type : null;
    }
    if (activeTab === 1 || activeTab === 2) {
      // Audiobooks/Podcasts tabs - use Audiobookshelf if available, otherwise fall back to first available source
      if (audiobookshelfSource) return 'audiobookshelf';
      return sources.length > 0 ? sources[0].service_type : null;
    }
    return null;
  };

  const serviceType = getServiceType();
  
  // Check if current service supports search (streaming services)
  const isStreamingService = serviceType === 'deezer';

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
      enabled: !!selectedAuthor && !!serviceType && activeTab === 1, // Only for Audiobooks tab
      refetchOnWindowFocus: false,
    }
  );

  // Fetch albums for selected artist (or series if in Audiobooks)
  const { data: artistAlbumsData, isLoading: loadingArtistAlbums } = useQuery(
    ['artistAlbums', selectedArtist, selectedSeries, selectedAuthor, serviceType],
    () => {
      if (activeTab === 1 && selectedSeries && selectedAuthor) {
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
    // Clear selected tracks when changing items
    setSelectedTracks(new Set());
    
    if (type === 'artist') {
      if (activeTab === 1) {
        // For Audiobooks, show series for the author
        setSelectedAuthor(item.id);
        setSelectedSeries(null);
        setSelectedArtist(null);
        setSelectedItem(null);
        setSelectedItemType(null);
      } else {
        // For Music, show albums for the artist
        setSelectedArtist(item.id);
        setSelectedItem(null);
        setSelectedItemType(null);
        setSelectedSeries(null);
        setSelectedAuthor(null);
      }
    } else {
      // For albums and playlists, show tracks
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
        playTrack(tracks.tracks[0], tracks.tracks, item.id);
      }
    } catch (error) {
      console.error('Failed to play item:', error);
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
    setSelectedItem(null);
    setSelectedItemType(null);
    setSelectedArtist(null);
    // Reset view based on tab
    if (newValue === 0) {
      setSelectedView('albums'); // Music: albums
    } else if (newValue === 1) {
      setSelectedView('albums'); // Audiobooks: books (as albums)
    } else {
      setSelectedView('playlists'); // Podcasts: playlists
    }
  };

  const handleTrackDoubleClick = (track) => {
    const tracks = tracksData?.tracks || [];
    playTrack(track, tracks, selectedItem);
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
      const serviceName = activeTab === 0 ? 'SubSonic' : activeTab === 1 ? 'Audiobookshelf' : 'Audiobookshelf';
      return (
        <Alert severity="info" sx={{ m: 2 }}>
          No {activeTab === 0 ? 'music' : activeTab === 1 ? 'audiobook' : 'podcast'} library found. Configure your {serviceName} server in Settings > Media and refresh the cache.
        </Alert>
      );
    }

    // If an author is selected in Audiobooks, show their series
    if (activeTab === 1 && selectedAuthor && !selectedSeries) {
      const author = library?.artists?.find(a => a.id === selectedAuthor);
      const series = seriesData?.series || [];
      
      return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
          {/* Back button and header - fixed */}
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
                  secondary="Series"
                  sx={{ px: 2, py: 1 }}
                />
              </ListItem>
            </List>
            <Divider sx={{ my: 1 }} />
          </Box>
          
          {/* Series list - scrollable */}
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
            {loadingSeries ? (
              <Box display="flex" justifyContent="center" p={2}>
                <CircularProgress size={24} />
              </Box>
            ) : series.length > 0 ? (
              <List sx={{ padding: 0 }}>
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
              </List>
            ) : (
              <List>
                <ListItem>
                  <ListItemText primary="No series found" secondary="This author has no series" />
                </ListItem>
              </List>
            )}
          </Box>
        </Box>
      );
    }

    // If a series is selected in Audiobooks, show books in that series
    if (activeTab === 1 && selectedSeries && selectedAuthor) {
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

    // If an artist is selected (Music tab), show their albums
    if (selectedArtist && activeTab === 0) {
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
                        secondary={album.artist}
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
    
    if (activeTab === 0) {
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
    } else if (activeTab === 1) {
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
                <ListItem key={item.id} disablePadding>
                  <ListItemButton
                    selected={selectedItem === item.id || selectedArtist === item.id}
                    onClick={() => handleItemClick(item, selectedView === 'albums' ? 'album' : selectedView === 'playlists' ? 'playlist' : 'artist')}
                    onDoubleClick={() => {
                      if (selectedView === 'albums' || selectedView === 'playlists') {
                        handleItemDoubleClick(item, selectedView === 'albums' ? 'album' : 'playlist');
                      }
                    }}
                    sx={{ py: 0.5 }}
                  >
                    <ListItemText
                      primary={item.title || item.name}
                      secondary={selectedView === 'albums' ? item.artist : null}
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
    const isPodcast = activeTab === 2; // Podcasts tab
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
        <TableContainer component={Paper} variant="outlined">
          <Table>
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
                    key={track.id}
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

  // Check if sources are configured
  const hasSubsonic = !!subsonicSource;
  const hasAudiobookshelf = !!audiobookshelfSource;
  const hasDeezer = !!deezerSource;

  // Filter available tabs based on configured sources
  const availableTabs = [
    { label: 'Music', icon: <LibraryMusic />, enabled: hasSubsonic || hasDeezer },
    { label: 'Audiobooks', icon: <Headphones />, enabled: hasAudiobookshelf },
    { label: 'Podcasts', icon: <Podcasts />, enabled: hasAudiobookshelf },
  ].filter(tab => tab.enabled);

  // Adjust activeTab if current tab is not available
  React.useEffect(() => {
    if (availableTabs.length === 0) {
      return; // No sources configured
    }
    if (activeTab >= availableTabs.length) {
      setActiveTab(0);
    }
  }, [activeTab, availableTabs.length]);

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
    <Box sx={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
      {/* Tabs - right against the top, no scrollbar */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', flexShrink: 0, display: 'flex', alignItems: 'center', px: 1.3125 }}>
        <Tabs 
          value={activeTab} 
          onChange={handleTabChange} 
          aria-label="media tabs"
          sx={{
            flex: 1,
            minHeight: 48,
            '& .MuiTabs-scroller': {
              overflow: 'hidden !important',
            },
            '& .MuiTabs-indicator': {
              display: 'none',
            },
          }}
        >
          {availableTabs.map((tab, index) => (
            <Tab
              key={index}
              label={tab.label}
              icon={tab.icon}
              iconPosition="start"
              disabled={!tab.enabled}
              sx={{
                minHeight: 48,
                textTransform: 'none',
                fontWeight: activeTab === index ? 600 : 400,
              }}
            />
          ))}
        </Tabs>
        {isStreamingService && (
          <Box sx={{ pr: 0.5 }}>
            <Button
              variant="outlined"
              startIcon={<Search />}
              onClick={() => setSearchDialogOpen(true)}
              size="small"
            >
              Search
            </Button>
          </Box>
        )}
      </Box>

      <Box sx={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden', position: 'relative' }}>
        {/* Sidebar */}
        <Box
          sx={{
            width: 250,
            borderRight: 1,
            borderColor: 'divider',
            flexShrink: 0,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            minHeight: 0,
            height: '100%',
          }}
        >
          {renderSidebar()}
        </Box>

        {/* Main Content */}
        <Box 
          sx={{ 
            flexGrow: 1, 
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            minHeight: 0,
            position: 'relative',
          }}
        >
          <Box sx={{ px: 2, pt: 0.5, pb: 0.5, flexShrink: 0, borderBottom: 1, borderColor: 'divider' }}>
            <Typography variant="h6" sx={{ fontWeight: 500, my: 0 }}>
              {selectedArtist
                ? library?.artists?.find((a) => a.id === selectedArtist)?.name || (activeTab === 1 ? 'Author' : 'Artist')
                : selectedItem
                ? selectedView === 'albums'
                  ? library?.albums?.find((a) => a.id === selectedItem)?.title || (activeTab === 1 ? 'Book' : 'Album')
                  : selectedView === 'playlists'
                  ? library?.playlists?.find((p) => p.id === selectedItem)?.name || (activeTab === 2 ? 'Show' : 'Playlist')
                  : 'Tracks'
                : activeTab === 0
                ? 'Music Library'
                : activeTab === 1
                ? 'Audiobooks'
                : 'Podcasts'}
            </Typography>
          </Box>
          <Box 
            sx={{ 
              flexGrow: 1, 
              overflow: 'auto',
              overflowX: 'hidden',
              minHeight: 0, // Important for flex scrolling
              p: 2, 
              pt: 0,
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
      
      {/* Deezer Search Dialog */}
      {isStreamingService && (
        <DeezerSearch
          open={searchDialogOpen}
          onClose={() => setSearchDialogOpen(false)}
          serviceType={serviceType}
        />
      )}
      
      {/* Deezer Search Dialog */}
      {isStreamingService && (
        <DeezerSearch
          open={searchDialogOpen}
          onClose={() => setSearchDialogOpen(false)}
          serviceType={serviceType}
        />
      )}
    </Box>
  );
};

export default MediaPage;

