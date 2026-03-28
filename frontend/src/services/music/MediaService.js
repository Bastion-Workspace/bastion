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
  getStreamUrl = (trackId, serviceType = null, parentId = null) => {
    // Use proxy endpoint for streaming. Include token in query so the HTML5 audio
    // element can load the URL directly (no fetch + blob), enabling near-gapless playback.
    // parentId is required for AudioBookShelf podcast episodes (library item id).
    // Base must match ApiServiceBase (VITE_API_URL); window.location alone breaks when the API
    // is on another origin or dev proxies JSON but audio used the wrong host.
    const apiBase = (this.baseURL || '').replace(/\/$/, '');
    const path = `/api/music/stream-proxy/${trackId}`;
    const base = apiBase ? `${apiBase}${path}` : path;
    const params = new URLSearchParams();
    if (serviceType) params.set('service_type', serviceType);
    if (parentId) params.set('parent_id', parentId);
    const token = typeof localStorage !== 'undefined'
      ? (localStorage.getItem('auth_token') || localStorage.getItem('token'))
      : null;
    if (token) params.set('token', token);
    const query = params.toString();
    return query ? `${base}?${query}` : base;
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

