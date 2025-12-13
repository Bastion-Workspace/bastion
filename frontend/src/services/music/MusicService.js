import ApiServiceBase from '../base/ApiServiceBase';

class MusicService extends ApiServiceBase {
  // Configuration methods
  saveConfig = async (config) => {
    return this.post('/api/music/config', config);
  }

  getConfig = async () => {
    return this.get('/api/music/config');
  }

  deleteConfig = async () => {
    return this.delete('/api/music/config');
  }

  testConnection = async () => {
    return this.post('/api/music/test-connection');
  }

  // Cache management
  refreshCache = async () => {
    return this.post('/api/music/refresh');
  }

  // Library methods
  getLibrary = async () => {
    return this.get('/api/music/library');
  }

  getTracks = async (parentId, parentType = 'album') => {
    return this.get(`/api/music/tracks/${parentId}?parent_type=${parentType}`);
  }

  getAlbumsByArtist = async (artistId) => {
    return this.get(`/api/music/albums/artist/${artistId}`);
  }

  // Streaming
  getStreamUrl = async (trackId) => {
    // Use proxy endpoint for better format support and CORS handling
    return `${window.location.origin}/api/music/stream-proxy/${trackId}`;
  }
}

export default new MusicService();

