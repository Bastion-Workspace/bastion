import React, { useState } from 'react';
import {
  Box,
  Typography,
  List,
  ListItemButton,
  ListItemText,
  Divider,
  CircularProgress,
  Alert,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TableContainer,
  Paper,
  Button,
} from '@mui/material';
import { ArrowBack, PlayArrow } from '@mui/icons-material';
import { useQuery } from 'react-query';
import apiService from '../../services/apiService';
import { useVideo } from '../../contexts/VideoContext';
import EmbyPosterCard from './EmbyPosterCard';

function itemId(it) {
  return it?.Id || it?.id;
}

function itemName(it) {
  return it?.Name || it?.name || '';
}

function progressRatio(it) {
  const ud = it?.UserData || it?.userData;
  if (!ud?.PlaybackPositionTicks || !it?.RunTimeTicks) return null;
  return ud.PlaybackPositionTicks / it.RunTimeTicks;
}

export default function EmbyBrowseView({ onOpenMusicLibrary }) {
  const { openVideo } = useVideo();
  const [nav, setNav] = useState({ kind: 'hub' });
  const [selectedLibrary, setSelectedLibrary] = useState(null);
  const [selectedSeries, setSelectedSeries] = useState(null);
  const [selectedSeason, setSelectedSeason] = useState(null);

  const { data: libData, isLoading: loadingLibs } = useQuery(
    ['embyLibraries'],
    () => apiService.emby.getLibraries(),
    { retry: false, refetchOnWindowFocus: false }
  );

  const libraries = libData?.libraries || [];

  const { data: resumeData } = useQuery(
    ['embyResume'],
    () => apiService.emby.getResumeItems(40),
    { retry: false, refetchOnWindowFocus: false }
  );
  const resumeItems = resumeData?.Items || [];

  const movieQuery = useQuery(
    ['embyItems', 'movies', selectedLibrary?.id],
    () =>
      apiService.emby.getItems({
        parent_id: selectedLibrary.id,
        item_types: 'Movie',
        sort_by: 'SortName',
        limit: 200,
        start_index: 0,
      }),
    {
      enabled: nav.kind === 'movies' && !!selectedLibrary?.id,
      refetchOnWindowFocus: false,
    }
  );

  const seriesGridQuery = useQuery(
    ['embyItems', 'series', selectedLibrary?.id],
    () =>
      apiService.emby.getItems({
        parent_id: selectedLibrary.id,
        item_types: 'Series',
        sort_by: 'SortName',
        limit: 200,
        start_index: 0,
      }),
    {
      enabled: nav.kind === 'tv' && !!selectedLibrary?.id,
      refetchOnWindowFocus: false,
    }
  );

  const seasonsQuery = useQuery(
    ['embySeasons', itemId(selectedSeries)],
    () => apiService.emby.getSeasons(itemId(selectedSeries)),
    {
      enabled: nav.kind === 'series' && !!itemId(selectedSeries),
      refetchOnWindowFocus: false,
    }
  );

  const episodesQuery = useQuery(
    ['embyEpisodes', itemId(selectedSeries), itemId(selectedSeason)],
    () => apiService.emby.getEpisodes(itemId(selectedSeries), itemId(selectedSeason)),
    {
      enabled: nav.kind === 'episodes' && !!itemId(selectedSeries) && !!itemId(selectedSeason),
      refetchOnWindowFocus: false,
    }
  );

  const poster = (item, w = 320) => {
    const id = itemId(item);
    if (!id) return '';
    const tag = item.PrimaryImageTag || item.ImageTags?.Primary;
    return apiService.emby.getImageUrl(id, 'Primary', w, 0, tag || null);
  };

  const resetToHub = () => {
    setNav({ kind: 'hub' });
    setSelectedLibrary(null);
    setSelectedSeries(null);
    setSelectedSeason(null);
  };

  const openLibrary = (lib) => {
    const t = (lib.collection_type || lib.collectionType || '').toLowerCase();
    setSelectedLibrary(lib);
    setSelectedSeries(null);
    setSelectedSeason(null);
    if (t === 'music') {
      onOpenMusicLibrary();
      return;
    }
    if (t === 'movies' || t === 'boxsets') {
      setNav({ kind: 'movies' });
      return;
    }
    if (t === 'tvshows') {
      setNav({ kind: 'tv' });
    }
  };

  const renderEmbySidebar = () => {
    if (loadingLibs) {
      return (
        <Box display="flex" justifyContent="center" p={2}>
          <CircularProgress size={28} />
        </Box>
      );
    }

    return (
      <List dense sx={{ py: 0 }}>
        {resumeItems.length > 0 && (
          <>
            <ListItemButton
              selected={nav.kind === 'resume'}
              onClick={() => {
                setNav({ kind: 'resume' });
                setSelectedLibrary(null);
                setSelectedSeries(null);
                setSelectedSeason(null);
              }}
            >
              <ListItemText primary="Continue watching" secondary={`${resumeItems.length} items`} />
            </ListItemButton>
            <Divider />
          </>
        )}
        {libraries.map((lib) => (
          <ListItemButton
            key={lib.id}
            selected={selectedLibrary?.id === lib.id && ['movies', 'tv', 'series', 'episodes'].includes(nav.kind)}
            onClick={() => openLibrary(lib)}
          >
            <ListItemText primary={lib.name} secondary={lib.collection_type || ''} />
          </ListItemButton>
        ))}
        {nav.kind !== 'hub' && (
          <>
            <Divider sx={{ my: 1 }} />
            <ListItemButton onClick={resetToHub}>
              <ArrowBack sx={{ mr: 1, fontSize: 18 }} />
              <ListItemText primary="Home" />
            </ListItemButton>
          </>
        )}
      </List>
    );
  };

  const mainHub = () => (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Emby library
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Choose a library in the sidebar. Movies and TV open here; Music opens the album and track browser.
      </Typography>
      {resumeItems.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            Continue watching
          </Typography>
          <Grid container spacing={2}>
            {resumeItems.slice(0, 12).map((it) => (
              <Grid item xs={6} sm={4} md={3} lg={2} key={itemId(it)}>
                <EmbyPosterCard
                  title={itemName(it)}
                  subtitle={it.SeriesName || it.ProductionYear || ''}
                  imageUrl={poster(it)}
                  played={it.UserData?.Played}
                  progressRatio={progressRatio(it)}
                  onClick={() => openVideo(it)}
                />
              </Grid>
            ))}
          </Grid>
        </Box>
      )}
    </Box>
  );

  const mainResume = () => (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Continue watching
      </Typography>
      <Grid container spacing={2}>
        {resumeItems.map((it) => (
          <Grid item xs={6} sm={4} md={3} lg={2} key={itemId(it)}>
            <EmbyPosterCard
              title={itemName(it)}
              subtitle={it.SeriesName || ''}
              imageUrl={poster(it)}
              played={it.UserData?.Played}
              progressRatio={progressRatio(it)}
              onClick={() => openVideo(it)}
            />
          </Grid>
        ))}
      </Grid>
    </Box>
  );

  const mainMovies = () => {
    if (movieQuery.isLoading) {
      return (
        <Box display="flex" justifyContent="center" p={4}>
          <CircularProgress />
        </Box>
      );
    }
    if (movieQuery.error) {
      return (
        <Alert severity="error" sx={{ m: 2 }}>
          Failed to load movies
        </Alert>
      );
    }
    const items = movieQuery.data?.Items || [];
    return (
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          {selectedLibrary?.name || 'Movies'}
        </Typography>
        <Grid container spacing={2}>
          {items.map((it) => (
            <Grid item xs={6} sm={4} md={3} lg={2} key={itemId(it)}>
              <EmbyPosterCard
                title={itemName(it)}
                subtitle={it.ProductionYear ? String(it.ProductionYear) : ''}
                imageUrl={poster(it)}
                played={it.UserData?.Played}
                progressRatio={progressRatio(it)}
                onClick={() => openVideo(it)}
              />
            </Grid>
          ))}
        </Grid>
        {!items.length && <Typography color="text.secondary">No movies found.</Typography>}
      </Box>
    );
  };

  const mainTvGrid = () => {
    if (seriesGridQuery.isLoading) {
      return (
        <Box display="flex" justifyContent="center" p={4}>
          <CircularProgress />
        </Box>
      );
    }
    const items = seriesGridQuery.data?.Items || [];
    return (
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          {selectedLibrary?.name || 'TV shows'}
        </Typography>
        <Grid container spacing={2}>
          {items.map((it) => (
            <Grid item xs={6} sm={4} md={3} lg={2} key={itemId(it)}>
              <EmbyPosterCard
                title={itemName(it)}
                subtitle={it.ProductionYear ? String(it.ProductionYear) : ''}
                imageUrl={poster(it)}
                onClick={() => {
                  setSelectedSeries(it);
                  setSelectedSeason(null);
                  setNav({ kind: 'series' });
                }}
              />
            </Grid>
          ))}
        </Grid>
        {!items.length && <Typography color="text.secondary">No series found.</Typography>}
      </Box>
    );
  };

  const mainSeries = () => {
    const seasons = seasonsQuery.data?.Items || [];
    return (
      <Box sx={{ p: 2, display: 'flex', gap: 2, flexDirection: { xs: 'column', md: 'row' } }}>
        <Box sx={{ width: { xs: '100%', md: 220 }, flexShrink: 0 }}>
          <Typography variant="subtitle1" gutterBottom>
            {itemName(selectedSeries)}
          </Typography>
          <Button
            size="small"
            startIcon={<ArrowBack />}
            onClick={() => {
              setSelectedSeries(null);
              setSelectedSeason(null);
              setNav({ kind: 'tv' });
            }}
            sx={{ mb: 1 }}
          >
            All series
          </Button>
          {seasonsQuery.isLoading ? (
            <CircularProgress size={24} />
          ) : (
            <List dense>
              {seasons.map((s) => (
                <ListItemButton
                  key={itemId(s)}
                  selected={itemId(selectedSeason) === itemId(s)}
                  onClick={() => {
                    setSelectedSeason(s);
                    setNav({ kind: 'episodes' });
                  }}
                >
                  <ListItemText primary={itemName(s) || `Season ${s.IndexNumber}`} />
                </ListItemButton>
              ))}
            </List>
          )}
        </Box>
        <Box sx={{ flex: 1, minWidth: 0 }}>
          {nav.kind === 'episodes' && selectedSeason ? (
            <EpisodeTable
              episodes={episodesQuery.data?.Items || []}
              loading={episodesQuery.isLoading}
              poster={poster}
              onPlay={openVideo}
            />
          ) : (
            <Typography color="text.secondary">Select a season to see episodes.</Typography>
          )}
        </Box>
      </Box>
    );
  };

  let main = mainHub();
  if (nav.kind === 'resume') main = mainResume();
  else if (nav.kind === 'movies') main = mainMovies();
  else if (nav.kind === 'tv') main = mainTvGrid();
  else if (nav.kind === 'series' || nav.kind === 'episodes') main = mainSeries();

  return (
    <Box sx={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden' }}>
      <Box
        sx={{
          width: 250,
          borderRight: 1,
          borderColor: 'divider',
          flexShrink: 0,
          overflowY: 'auto',
          minHeight: 0,
        }}
      >
        {renderEmbySidebar()}
      </Box>
      <Box sx={{ flex: 1, minWidth: 0, overflow: 'auto', minHeight: 0 }}>{main}</Box>
    </Box>
  );
}

function EpisodeTable({ episodes, loading, poster, onPlay }) {
  if (loading) {
    return (
      <Box display="flex" justifyContent="center" p={4}>
        <CircularProgress />
      </Box>
    );
  }
  return (
    <TableContainer component={Paper} variant="outlined">
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell width={72} />
            <TableCell>#</TableCell>
            <TableCell>Title</TableCell>
            <TableCell width={100}>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {episodes.map((ep) => (
            <TableRow key={itemId(ep)} hover>
              <TableCell>
                <Box
                  component="img"
                  src={poster(ep, 120)}
                  alt=""
                  sx={{ width: 64, height: 36, objectFit: 'cover', borderRadius: 0.5 }}
                />
              </TableCell>
              <TableCell>{ep.IndexNumber}</TableCell>
              <TableCell>{itemName(ep)}</TableCell>
              <TableCell>
                <Button size="small" startIcon={<PlayArrow />} onClick={() => onPlay(ep)}>
                  Play
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
