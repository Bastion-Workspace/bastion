import { assertApiBaseUrl } from './config';
import { apiRequest } from './client';
import { getStoredToken } from '../session/tokenStore';

export type MusicTrack = {
  id: string;
  title: string;
  artist?: string | null;
  album?: string | null;
  duration?: number | null;
  track_number?: number | null;
  cover_art_id?: string | null;
  metadata?: Record<string, unknown> | null;
  service_type?: string | null;
};

export type MusicAlbum = {
  id: string;
  title: string;
  artist?: string | null;
  cover_art_id?: string | null;
  track_count?: number | null;
  metadata?: Record<string, unknown> | null;
};

export type MusicArtist = {
  id: string;
  name: string;
  album_count?: number | null;
  metadata?: Record<string, unknown> | null;
};

export type MusicPlaylist = {
  id: string;
  name: string;
  track_count?: number | null;
  metadata?: Record<string, unknown> | null;
};

export type MusicLibraryResponse = {
  albums: MusicAlbum[];
  artists: MusicArtist[];
  playlists: MusicPlaylist[];
  last_sync_at?: string | null;
};

export type MusicTracksResponse = {
  tracks: MusicTrack[];
  parent_id: string;
  parent_type: string;
};

export type MusicServiceConfigResponse = {
  server_url: string;
  username: string;
  auth_type: string;
  service_type: string;
  service_name?: string | null;
  is_active?: boolean;
  has_config?: boolean;
  last_sync_at?: string | null;
  sync_status?: string | null;
  total_albums?: number;
  total_artists?: number;
  total_playlists?: number;
  total_tracks?: number;
};

export type MediaSourceListResponse = {
  sources: MusicServiceConfigResponse[];
};

function q(serviceType?: string | null): string {
  if (!serviceType) return '';
  return `?service_type=${encodeURIComponent(serviceType)}`;
}

export async function getMediaSources(): Promise<MediaSourceListResponse> {
  return apiRequest<MediaSourceListResponse>('/api/music/sources');
}

export async function getMusicConfig(serviceType?: string | null): Promise<MusicServiceConfigResponse> {
  const suffix = serviceType ? `?service_type=${encodeURIComponent(serviceType)}` : '';
  return apiRequest<MusicServiceConfigResponse>(`/api/music/config${suffix}`);
}

export async function getLibrary(serviceType?: string | null): Promise<MusicLibraryResponse> {
  return apiRequest<MusicLibraryResponse>(`/api/music/library${q(serviceType)}`);
}

export async function getAlbumsByArtist(
  artistId: string,
  serviceType?: string | null
): Promise<MusicLibraryResponse> {
  return apiRequest<MusicLibraryResponse>(`/api/music/albums/artist/${encodeURIComponent(artistId)}${q(serviceType)}`);
}

export async function getTracks(
  parentId: string,
  parentType: 'album' | 'playlist',
  serviceType?: string | null
): Promise<MusicTracksResponse> {
  const qs = new URLSearchParams();
  qs.set('parent_type', parentType);
  if (serviceType) qs.set('service_type', serviceType);
  return apiRequest<MusicTracksResponse>(
    `/api/music/tracks/${encodeURIComponent(parentId)}?${qs.toString()}`
  );
}

export type SearchRequestBody = {
  query: string;
  service_type: string;
  limit?: number;
};

export async function searchTracks(body: SearchRequestBody): Promise<MusicTracksResponse> {
  return apiRequest<MusicTracksResponse>('/api/music/search/tracks', {
    method: 'POST',
    body: JSON.stringify({
      query: body.query,
      service_type: body.service_type,
      limit: body.limit ?? 25,
    }),
  });
}

export async function searchAlbums(body: SearchRequestBody): Promise<MusicLibraryResponse> {
  return apiRequest<MusicLibraryResponse>('/api/music/search/albums', {
    method: 'POST',
    body: JSON.stringify({
      query: body.query,
      service_type: body.service_type,
      limit: body.limit ?? 25,
    }),
  });
}

export async function searchArtists(body: SearchRequestBody): Promise<MusicLibraryResponse> {
  return apiRequest<MusicLibraryResponse>('/api/music/search/artists', {
    method: 'POST',
    body: JSON.stringify({
      query: body.query,
      service_type: body.service_type,
      limit: body.limit ?? 25,
    }),
  });
}

export async function refreshMusicCache(serviceType?: string | null): Promise<Record<string, unknown>> {
  return apiRequest<Record<string, unknown>>(`/api/music/refresh${q(serviceType)}`, { method: 'POST' });
}

/**
 * Absolute URL for stream-proxy with JWT query param (HTML5 / RNTP / downloads).
 */
export async function getStreamProxyUrl(
  trackId: string,
  options?: { serviceType?: string | null; parentId?: string | null }
): Promise<string> {
  const base = assertApiBaseUrl();
  const token = await getStoredToken();
  const params = new URLSearchParams();
  if (token) params.set('token', token);
  if (options?.serviceType) params.set('service_type', options.serviceType);
  if (options?.parentId) params.set('parent_id', options.parentId);
  const qs = params.toString();
  return `${base}/api/music/stream-proxy/${encodeURIComponent(trackId)}${qs ? `?${qs}` : ''}`;
}
