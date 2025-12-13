import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemAvatar,
  Avatar,
  CircularProgress,
  Alert,
  InputAdornment,
  IconButton,
  Chip,
} from '@mui/material';
import {
  Search,
  Clear,
  MusicNote,
  Album,
  Person,
  PlayArrow,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import apiService from '../../services/apiService';
import { useMusic } from '../../contexts/MediaContext';

const DeezerSearch = ({ open, onClose, serviceType }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState(0); // 0: Tracks, 1: Albums, 2: Artists
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const { playTrack } = useMusic();

  // Debounce search query
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(searchQuery);
    }, 300);

    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Search tracks
  const { data: tracksData, isLoading: loadingTracks } = useQuery(
    ['deezerSearchTracks', debouncedQuery, serviceType],
    () => apiService.music.searchTracks(debouncedQuery, serviceType, 25),
    {
      enabled: !!debouncedQuery && activeTab === 0 && !!serviceType,
      retry: false,
    }
  );

  // Search albums
  const { data: albumsData, isLoading: loadingAlbums } = useQuery(
    ['deezerSearchAlbums', debouncedQuery, serviceType],
    () => apiService.music.searchAlbums(debouncedQuery, serviceType, 25),
    {
      enabled: !!debouncedQuery && activeTab === 1 && !!serviceType,
      retry: false,
    }
  );

  // Search artists
  const { data: artistsData, isLoading: loadingArtists } = useQuery(
    ['deezerSearchArtists', debouncedQuery, serviceType],
    () => apiService.music.searchArtists(debouncedQuery, serviceType, 25),
    {
      enabled: !!debouncedQuery && activeTab === 2 && !!serviceType,
      retry: false,
    }
  );

  const tracks = tracksData?.tracks || [];
  const albums = albumsData?.albums || [];
  const artists = artistsData?.artists || [];

  const isLoading = loadingTracks || loadingAlbums || loadingArtists;

  const handlePlayTrack = (track) => {
    if (playTrack && track.id) {
      playTrack(track.id, serviceType);
    }
  };

  const formatDuration = (seconds) => {
    if (!seconds) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getCoverArt = (item) => {
    if (item.cover_art_id) {
      return item.cover_art_id;
    }
    if (item.metadata?.cover_medium) {
      return item.metadata.cover_medium;
    }
    if (item.metadata?.cover) {
      return item.metadata.cover;
    }
    return null;
  };

  const handleClose = () => {
    setSearchQuery('');
    setDebouncedQuery('');
    setActiveTab(0);
    onClose();
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Typography variant="h6">Search Deezer</Typography>
          <IconButton onClick={handleClose} size="small">
            <Clear />
          </IconButton>
        </Box>
      </DialogTitle>
      <DialogContent>
        <Box mb={2}>
          <TextField
            fullWidth
            placeholder="Search for tracks, albums, or artists..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <Search />
                </InputAdornment>
              ),
              endAdornment: searchQuery && (
                <InputAdornment position="end">
                  <IconButton size="small" onClick={() => setSearchQuery('')}>
                    <Clear />
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
        </Box>

        <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)} sx={{ mb: 2 }}>
          <Tab icon={<MusicNote />} label="Tracks" />
          <Tab icon={<Album />} label="Albums" />
          <Tab icon={<Person />} label="Artists" />
        </Tabs>

        {isLoading && (
          <Box display="flex" justifyContent="center" py={4}>
            <CircularProgress />
          </Box>
        )}

        {!isLoading && debouncedQuery && (
          <>
            {activeTab === 0 && (
              <>
                {tracks.length === 0 ? (
                  <Alert severity="info">No tracks found</Alert>
                ) : (
                  <List>
                    {tracks.map((track) => (
                      <ListItem key={track.id} disablePadding>
                        <ListItemButton onClick={() => handlePlayTrack(track)}>
                          <ListItemAvatar>
                            <Avatar
                              src={getCoverArt(track)}
                              variant="rounded"
                              sx={{ width: 56, height: 56 }}
                            >
                              <MusicNote />
                            </Avatar>
                          </ListItemAvatar>
                          <ListItemText
                            primary={track.title}
                            secondary={
                              <Box>
                                <Typography variant="body2" color="text.secondary">
                                  {track.artist} • {track.album}
                                </Typography>
                                {track.duration && (
                                  <Chip
                                    label={formatDuration(track.duration)}
                                    size="small"
                                    sx={{ mt: 0.5 }}
                                  />
                                )}
                              </Box>
                            }
                          />
                          <IconButton edge="end" onClick={(e) => { e.stopPropagation(); handlePlayTrack(track); }}>
                            <PlayArrow />
                          </IconButton>
                        </ListItemButton>
                      </ListItem>
                    ))}
                  </List>
                )}
              </>
            )}

            {activeTab === 1 && (
              <>
                {albums.length === 0 ? (
                  <Alert severity="info">No albums found</Alert>
                ) : (
                  <List>
                    {albums.map((album) => (
                      <ListItem key={album.id} disablePadding>
                        <ListItemButton>
                          <ListItemAvatar>
                            <Avatar
                              src={getCoverArt(album)}
                              variant="rounded"
                              sx={{ width: 56, height: 56 }}
                            >
                              <Album />
                            </Avatar>
                          </ListItemAvatar>
                          <ListItemText
                            primary={album.title}
                            secondary={
                              <Typography variant="body2" color="text.secondary">
                                {album.artist}
                              </Typography>
                            }
                          />
                        </ListItemButton>
                      </ListItem>
                    ))}
                  </List>
                )}
              </>
            )}

            {activeTab === 2 && (
              <>
                {artists.length === 0 ? (
                  <Alert severity="info">No artists found</Alert>
                ) : (
                  <List>
                    {artists.map((artist) => (
                      <ListItem key={artist.id} disablePadding>
                        <ListItemButton>
                          <ListItemAvatar>
                            <Avatar
                              src={getCoverArt(artist)}
                              sx={{ width: 56, height: 56 }}
                            >
                              <Person />
                            </Avatar>
                          </ListItemAvatar>
                          <ListItemText
                            primary={artist.name}
                            secondary={
                              artist.metadata?.nb_album && (
                                <Typography variant="body2" color="text.secondary">
                                  {artist.metadata.nb_album} albums
                                </Typography>
                              )
                            }
                          />
                        </ListItemButton>
                      </ListItem>
                    ))}
                  </List>
                )}
              </>
            )}
          </>
        )}

        {!debouncedQuery && !isLoading && (
          <Box textAlign="center" py={4}>
            <Typography variant="body2" color="text.secondary">
              Enter a search query to find music on Deezer
            </Typography>
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

export default DeezerSearch;

