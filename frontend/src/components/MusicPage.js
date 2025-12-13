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
} from '@mui/material';
import {
  Album,
  Person,
  PlaylistPlay,
  ArrowBack,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import apiService from '../services/apiService';
import { useMusic } from '../contexts/MusicContext';

const MusicPage = () => {
  const [selectedView, setSelectedView] = useState('albums'); // 'albums', 'artists', 'playlists'
  const [selectedItem, setSelectedItem] = useState(null);
  const [selectedItemType, setSelectedItemType] = useState(null);
  const [selectedArtist, setSelectedArtist] = useState(null); // Track selected artist for hierarchical nav
  const { playTrack } = useMusic();

  // Fetch library
  const { data: library, isLoading: loadingLibrary, error: libraryError } = useQuery(
    'musicLibrary',
    () => apiService.music.getLibrary(),
    {
      enabled: true,
      refetchOnWindowFocus: false,
    }
  );

  // Fetch albums for selected artist
  const { data: artistAlbumsData, isLoading: loadingArtistAlbums } = useQuery(
    ['artistAlbums', selectedArtist],
    () => apiService.music.getAlbumsByArtist(selectedArtist),
    {
      enabled: !!selectedArtist,
      refetchOnWindowFocus: false,
    }
  );

  // Fetch tracks for selected item
  const { data: tracksData, isLoading: loadingTracks } = useQuery(
    ['musicTracks', selectedItem, selectedItemType],
    () => apiService.music.getTracks(selectedItem, selectedItemType),
    {
      enabled: !!selectedItem && !!selectedItemType && selectedItemType !== 'artist',
      refetchOnWindowFocus: false,
    }
  );

  const handleViewChange = (view) => {
    setSelectedView(view);
    setSelectedItem(null);
    setSelectedItemType(null);
    setSelectedArtist(null);
  };

  const handleItemClick = (item, type) => {
    if (type === 'artist') {
      // For artists, show their albums instead of tracks
      setSelectedArtist(item.id);
      setSelectedItem(null);
      setSelectedItemType(null);
    } else {
      // For albums and playlists, show tracks
      setSelectedItem(item.id);
      setSelectedItemType(type);
      setSelectedArtist(null);
    }
  };

  const handleAlbumFromArtistClick = (album) => {
    // When clicking an album from artist view, show tracks
    setSelectedItem(album.id);
    setSelectedItemType('album');
    setSelectedArtist(null);
  };

  const handleBackToArtists = () => {
    setSelectedArtist(null);
    setSelectedItem(null);
    setSelectedItemType(null);
  };

  const handleItemDoubleClick = async (item, type) => {
    // Fetch tracks and play
    try {
      const tracks = await apiService.music.getTracks(item.id, type);
      if (tracks.tracks && tracks.tracks.length > 0) {
        playTrack(tracks.tracks[0], tracks.tracks, item.id);
      }
    } catch (error) {
      console.error('Failed to play item:', error);
    }
  };

  const handleTrackDoubleClick = (track) => {
    const tracks = tracksData?.tracks || [];
    playTrack(track, tracks, selectedItem);
  };

  const formatDuration = (seconds) => {
    if (!seconds) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
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
      return (
        <Alert severity="info" sx={{ m: 2 }}>
          No music library found. Configure your SubSonic server in Settings > Media.
        </Alert>
      );
    }

    // If an artist is selected, show their albums
    if (selectedArtist) {
      const artist = library?.artists?.find(a => a.id === selectedArtist);
      return (
        <List>
          <ListItem disablePadding>
            <ListItemButton onClick={handleBackToArtists}>
              <ArrowBack sx={{ mr: 1 }} />
              <ListItemText primary="Back to Artists" />
            </ListItemButton>
          </ListItem>
          <Divider sx={{ my: 1 }} />
          <ListItem disablePadding>
            <ListItemText 
              primary={artist?.name || 'Artist'} 
              secondary="Albums"
              sx={{ px: 2, py: 1 }}
            />
          </ListItem>
          {loadingArtistAlbums ? (
            <Box display="flex" justifyContent="center" p={2}>
              <CircularProgress size={24} />
            </Box>
          ) : (
            (artistAlbumsData?.albums || []).map((album) => (
              <ListItem key={album.id} disablePadding>
                <ListItemButton
                  selected={selectedItem === album.id}
                  onClick={() => handleAlbumFromArtistClick(album)}
                  onDoubleClick={() => handleItemDoubleClick(album, 'album')}
                >
                  <ListItemText
                    primary={album.title}
                    secondary={album.artist}
                  />
                </ListItemButton>
              </ListItem>
            ))
          )}
        </List>
      );
    }

    // Normal view - show albums, artists, or playlists
    const items = {
      albums: library.albums || [],
      artists: library.artists || [],
      playlists: library.playlists || [],
    };

    const currentItems = items[selectedView] || [];

    return (
      <List>
        <ListItem disablePadding>
          <ListItemButton
            selected={selectedView === 'albums'}
            onClick={() => handleViewChange('albums')}
          >
            <Album sx={{ mr: 1 }} />
            <ListItemText primary="Albums" />
          </ListItemButton>
        </ListItem>
        <ListItem disablePadding>
          <ListItemButton
            selected={selectedView === 'artists'}
            onClick={() => handleViewChange('artists')}
          >
            <Person sx={{ mr: 1 }} />
            <ListItemText primary="Artists" />
          </ListItemButton>
        </ListItem>
        <ListItem disablePadding>
          <ListItemButton
            selected={selectedView === 'playlists'}
            onClick={() => handleViewChange('playlists')}
          >
            <PlaylistPlay sx={{ mr: 1 }} />
            <ListItemText primary="Playlists" />
          </ListItemButton>
        </ListItem>
        <Divider sx={{ my: 1 }} />
        {currentItems.map((item) => (
          <ListItem key={item.id} disablePadding>
            <ListItemButton
              selected={selectedItem === item.id || selectedArtist === item.id}
              onClick={() => handleItemClick(item, selectedView === 'albums' ? 'album' : selectedView === 'playlists' ? 'playlist' : 'artist')}
              onDoubleClick={() => {
                if (selectedView === 'albums' || selectedView === 'playlists') {
                  handleItemDoubleClick(item, selectedView === 'albums' ? 'album' : 'playlist');
                }
              }}
            >
              <ListItemText
                primary={item.title || item.name}
                secondary={selectedView === 'albums' ? item.artist : null}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
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

    return (
      <TableContainer component={Paper} variant="outlined">
        <Table>
          <TableHead>
            <TableRow>
              <TableCell width="5%">#</TableCell>
              <TableCell>Title</TableCell>
              <TableCell>Artist</TableCell>
              <TableCell>Album</TableCell>
              <TableCell width="10%" align="right">Duration</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {tracksData.tracks.map((track, index) => (
              <TableRow
                key={track.id}
                hover
                onDoubleClick={() => handleTrackDoubleClick(track)}
                sx={{ cursor: 'pointer' }}
              >
                <TableCell>{track.track_number || index + 1}</TableCell>
                <TableCell>{track.title}</TableCell>
                <TableCell>{track.artist || '-'}</TableCell>
                <TableCell>{track.album || '-'}</TableCell>
                <TableCell align="right">{formatDuration(track.duration)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  return (
    <Box sx={{ display: 'flex', height: '100%', overflow: 'hidden', position: 'relative' }}>
      {/* Sidebar */}
      <Box
        sx={{
          width: 250,
          borderRight: 1,
          borderColor: 'divider',
          flexShrink: 0,
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <Box
          sx={{
            overflowY: 'auto',
            overflowX: 'hidden',
            height: '100%',
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
          {renderSidebar()}
        </Box>
      </Box>

      {/* Main Content */}
      <Box 
        sx={{ 
          flexGrow: 1, 
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          position: 'relative',
        }}
      >
        <Box sx={{ p: 2, flexShrink: 0 }}>
          <Typography variant="h5" gutterBottom>
            {selectedArtist
              ? library?.artists?.find((a) => a.id === selectedArtist)?.name || 'Artist'
              : selectedItem
              ? selectedView === 'albums'
                ? library?.albums?.find((a) => a.id === selectedItem)?.title || 'Album'
                : selectedView === 'playlists'
                ? library?.playlists?.find((p) => p.id === selectedItem)?.name || 'Playlist'
                : 'Tracks'
              : 'Music Library'}
          </Typography>
        </Box>
        <Box 
          sx={{ 
            flexGrow: 1, 
            overflow: 'auto', 
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
  );
};

export default MusicPage;

