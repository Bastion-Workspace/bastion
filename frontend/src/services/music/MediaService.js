import ApiServiceBase from '../base/ApiServiceBase';

class MediaService extends ApiServiceBase {
  // Configuration methods
  saveConfig = async (config) => {
    return this.post('/api/music/config', config);
  }

  getConfig = async (serviceType = null) => {
    const url = serviceType 
      ? `/api/music/config?service_type=${encodeURIComponent(serviceType)}`
      : '/api/music/config';
    return this.get(url);
  }

  getSources = async () => {
    return this.get('/api/music/sources');
  }

  deleteConfig = async (serviceType = null) => {
    const url = serviceType
      ? `/api/music/config?service_type=${encodeURIComponent(serviceType)}`
      : '/api/music/config';
    return this.delete(url);
  }

  testConnection = async (serviceType = null) => {
    const url = serviceType
      ? `/api/music/test-connection?service_type=${encodeURIComponent(serviceType)}`
      : '/api/music/test-connection';
    return this.post(url);
  }

  // Cache management
  refreshCache = async (serviceType = null) => {
    const url = serviceType
      ? `/api/music/refresh?service_type=${encodeURIComponent(serviceType)}`
      : '/api/music/refresh';
    return this.post(url);
  }

  // Library methods
  getLibrary = async (serviceType = null) => {
    const url = serviceType
      ? `/api/music/library?service_type=${encodeURIComponent(serviceType)}`
      : '/api/music/library';
    return this.get(url);
  }

  getTracks = async (parentId, parentType = 'album', serviceType = null) => {
    let url = `/api/music/tracks/${parentId}?parent_type=${parentType}`;
    if (serviceType) {
      url += `&service_type=${encodeURIComponent(serviceType)}`;
    }
    return this.get(url);
  }

  getAlbumsByArtist = async (artistId, serviceType = null) => {
    let url = `/api/music/albums/artist/${artistId}`;
    if (serviceType) {
      url += `?service_type=${encodeURIComponent(serviceType)}`;
    }
    return this.get(url);
  }

  getSeriesByAuthor = async (authorId, serviceType = null) => {
    let url = `/api/music/series/author/${authorId}`;
    if (serviceType) {
      url += `?service_type=${encodeURIComponent(serviceType)}`;
    }
    return this.get(url);
  }

  getAlbumsBySeries = async (seriesName, authorName, serviceType = null) => {
    let url = `/api/music/albums/series/${encodeURIComponent(seriesName)}?author_name=${encodeURIComponent(authorName)}`;
    if (serviceType) {
      url += `&service_type=${encodeURIComponent(serviceType)}`;
    }
    return this.get(url);
  }

  // Streaming
  getStreamUrl = async (trackId, serviceType = null) => {
    // Use proxy endpoint for better format support and CORS handling
    let url = `${window.location.origin}/api/music/stream-proxy/${trackId}`;
    if (serviceType) {
      url += `?service_type=${encodeURIComponent(serviceType)}`;
    }
    return url;
  }

  // Playlist management
  addTracksToPlaylist = async (playlistId, trackIds, serviceType = null) => {
    let url = `/api/music/playlist/${playlistId}/add-tracks`;
    if (serviceType) {
      url += `?service_type=${encodeURIComponent(serviceType)}`;
    }
    return this.post(url, { track_ids: trackIds });
  }

  removeTracksFromPlaylist = async (playlistId, trackIds, serviceType = null) => {
    let url = `/api/music/playlist/${playlistId}/remove-tracks`;
    if (serviceType) {
      url += `?service_type=${encodeURIComponent(serviceType)}`;
    }
    return this.post(url, { track_ids: trackIds });
  }

  // Search methods
  searchTracks = async (query, serviceType, limit = 25) => {
    return this.post('/api/music/search/tracks', { query, service_type: serviceType, limit });
  }

  searchAlbums = async (query, serviceType, limit = 25) => {
    return this.post('/api/music/search/albums', { query, service_type: serviceType, limit });
  }

  searchArtists = async (query, serviceType, limit = 25) => {
    return this.post('/api/music/search/artists', { query, service_type: serviceType, limit });
  }
}

export default new MediaService();

