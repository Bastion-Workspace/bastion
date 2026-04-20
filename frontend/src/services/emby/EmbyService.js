import ApiServiceBase from '../base/ApiServiceBase';

class EmbyService extends ApiServiceBase {
  _authQuery() {
    const token =
      typeof localStorage !== 'undefined'
        ? localStorage.getItem('auth_token') || localStorage.getItem('token')
        : null;
    return token ? `token=${encodeURIComponent(token)}` : '';
  }

  getLibraries = () => this.get('/api/emby/libraries');

  getItems = (params = {}) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '') q.set(k, String(v));
    });
    const qs = q.toString();
    return this.get(qs ? `/api/emby/items?${qs}` : '/api/emby/items');
  };

  getLatestItems = (parentId, limit = 24) =>
    this.get(`/api/emby/items/latest?parent_id=${encodeURIComponent(parentId)}&limit=${limit}`);

  getResumeItems = (limit = 50) =>
    this.get(`/api/emby/items/resume?limit=${limit}`);

  getItemDetail = (itemId) => this.get(`/api/emby/items/${encodeURIComponent(itemId)}`);

  getSeasons = (seriesId) =>
    this.get(`/api/emby/shows/${encodeURIComponent(seriesId)}/seasons`);

  getEpisodes = (seriesId, seasonId) =>
    this.get(
      `/api/emby/shows/${encodeURIComponent(seriesId)}/episodes?season_id=${encodeURIComponent(seasonId)}`
    );

  getPlaybackInfo = (itemId, body = null) =>
    this.post(`/api/emby/items/${encodeURIComponent(itemId)}/playback-info`, body || {});

  getVideoStreamUrl = (itemId, mediaSourceId, playSessionId, extra = {}) => {
    const apiBase = (this.baseURL || '').replace(/\/$/, '');
    const path = `/api/emby/video-stream/${encodeURIComponent(itemId)}`;
    const base = apiBase ? `${apiBase}${path}` : path;
    const params = new URLSearchParams({
      media_source_id: mediaSourceId,
      play_session_id: playSessionId,
      ...extra,
    });
    const t = this._authQuery();
    const q = params.toString();
    return t ? `${base}?${q}&${t}` : `${base}?${q}`;
  };

  getHlsMasterUrl = (itemId, mediaSourceId, playSessionId, extra = {}) => {
    const apiBase = (this.baseURL || '').replace(/\/$/, '');
    const path = `/api/emby/hls/${encodeURIComponent(itemId)}/master.m3u8`;
    const base = apiBase ? `${apiBase}${path}` : path;
    const params = new URLSearchParams({
      media_source_id: mediaSourceId,
      play_session_id: playSessionId,
      ...extra,
    });
    const t = this._authQuery();
    const qs = params.toString();
    return t ? `${base}?${qs}&${t}` : `${base}?${qs}`;
  };

  getImageUrl = (itemId, imageType = 'Primary', maxWidth = 400, index = 0, tag = null) => {
    const apiBase = (this.baseURL || '').replace(/\/$/, '');
    const path = `/api/emby/image/${encodeURIComponent(itemId)}/${encodeURIComponent(imageType)}`;
    const params = new URLSearchParams({ max_width: String(maxWidth), index: String(index) });
    if (tag) params.set('tag', tag);
    const t = this._authQuery();
    const qs = params.toString();
    const suffix = t ? `${qs}&${t}` : qs;
    return apiBase ? `${apiBase}${path}?${suffix}` : `${path}?${suffix}`;
  };

  reportPlaybackStart = (payload) => this.post('/api/emby/playback/start', payload);

  reportPlaybackProgress = (payload) => this.post('/api/emby/playback/progress', payload);

  reportPlaybackStopped = (payload) => this.post('/api/emby/playback/stopped', payload);

  search = (query, limit = 40) =>
    this.get(`/api/emby/search?q=${encodeURIComponent(query)}&limit=${limit}`);
}

export default new EmbyService();
