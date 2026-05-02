import { apiRequest } from './client';

export type EmbyLibrary = {
  id: string;
  name: string;
  collection_type?: string;
  item_id?: string;
  raw?: Record<string, unknown>;
};

export type EmbyItem = {
  Id: string;
  Name?: string;
  Type?: string;
  ParentId?: string;
  AlbumId?: string;
  IndexNumber?: number;
  ParentIndexNumber?: number;
  ImageTags?: { Primary?: string };
  SeriesName?: string;
  SeasonName?: string;
  RunTimeTicks?: number;
  UserData?: {
    PlaybackPositionTicks?: number;
    PlayedPercentage?: number;
  };
};

export type EmbyPlaybackStartBody = {
  ItemId: string;
  MediaSourceId?: string | null;
  PlaySessionId?: string | null;
  PositionTicks?: number;
  PlayMethod?: string;
  IsPaused?: boolean;
  AudioStreamIndex?: number | null;
};

export type EmbyPlaybackProgressBody = EmbyPlaybackStartBody & {
  EventName?: string;
};

export type EmbyPlaybackStoppedBody = {
  ItemId: string;
  MediaSourceId?: string | null;
  PlaySessionId?: string | null;
  PositionTicks?: number;
};

export async function getEmbyLibraries(): Promise<{ libraries: EmbyLibrary[] }> {
  return apiRequest<{ libraries: EmbyLibrary[] }>('/api/emby/libraries');
}

export async function getEmbyResumeItems(limit = 50): Promise<{ Items: EmbyItem[] }> {
  const qs = new URLSearchParams({ limit: String(limit) });
  return apiRequest<{ Items: EmbyItem[] }>(`/api/emby/items/resume?${qs.toString()}`);
}

export async function getEmbyLatestItems(
  parentId: string,
  limit = 24
): Promise<{ Items: EmbyItem[] }> {
  const qs = new URLSearchParams({ parent_id: parentId, limit: String(limit) });
  return apiRequest<{ Items: EmbyItem[] }>(`/api/emby/items/latest?${qs.toString()}`);
}

export async function getEmbySearch(q: string, limit = 40): Promise<{ Items: EmbyItem[] }> {
  const qs = new URLSearchParams({ q: q.trim(), limit: String(limit) });
  return apiRequest<{ Items: EmbyItem[] }>(`/api/emby/search?${qs.toString()}`);
}

export async function getEmbyItems(opts: {
  parentId?: string;
  itemTypes?: string;
  recursive?: boolean;
  limit?: number;
  sortBy?: string;
  sortOrder?: string;
  startIndex?: number;
}): Promise<{ Items: EmbyItem[] }> {
  const qs = new URLSearchParams();
  if (opts.parentId) qs.set('parent_id', opts.parentId);
  if (opts.itemTypes) qs.set('item_types', opts.itemTypes);
  if (opts.recursive) qs.set('recursive', 'true');
  qs.set('limit', String(opts.limit ?? 100));
  if (opts.sortBy) qs.set('sort_by', opts.sortBy);
  if (opts.sortOrder) qs.set('sort_order', opts.sortOrder);
  if (opts.startIndex != null) qs.set('start_index', String(opts.startIndex));
  return apiRequest<{ Items: EmbyItem[] }>(`/api/emby/items?${qs.toString()}`);
}

export async function getEmbySeasons(seriesId: string): Promise<{ Items: EmbyItem[] }> {
  return apiRequest<{ Items: EmbyItem[] }>(`/api/emby/shows/${encodeURIComponent(seriesId)}/seasons`);
}

export async function getEmbyEpisodes(seriesId: string, seasonId: string): Promise<{ Items: EmbyItem[] }> {
  const qs = new URLSearchParams({ season_id: seasonId });
  return apiRequest<{ Items: EmbyItem[] }>(
    `/api/emby/shows/${encodeURIComponent(seriesId)}/episodes?${qs.toString()}`
  );
}

/** Emby PlaybackInfo DeviceProfile: guides direct play vs server transcode. */
export const MOBILE_DEVICE_PROFILE: Record<string, unknown> = {
  MaxStreamingBitrate: 40_000_000,
  DirectPlayProfiles: [
    {
      Type: 'Video',
      Container: 'mp4,mov',
      VideoCodec: 'h264',
      AudioCodec: 'aac,mp3,opus',
    },
  ],
  TranscodingProfiles: [
    {
      Type: 'Video',
      Container: 'ts',
      VideoCodec: 'h264',
      AudioCodec: 'aac',
      Protocol: 'hls',
      MaxAudioChannels: '2',
    },
  ],
  SubtitleProfiles: [{ Format: 'vtt', Method: 'External' }],
};

export type MobileSourcePick = {
  mediaSourceId: string;
  playSessionId: string;
  useHls: boolean;
  playMethod: 'DirectPlay' | 'DirectStream' | 'Transcode';
  audioStreamIndex: number | null;
};

function defaultMediaSourceFromPlaybackInfo(
  info: Record<string, unknown>
): Record<string, unknown> | null {
  const sources = (info.MediaSources ?? info.mediaSources) as Record<string, unknown>[] | undefined;
  if (!sources?.length) return null;
  const def = sources.find((s) => s.IsDefault === true || s.isDefault === true);
  return def ?? sources[0];
}

function defaultAudioFromMediaSource(ms: Record<string, unknown>): { index: number | null } {
  const streams = (ms.MediaStreams ?? ms.mediaStreams) as Record<string, unknown>[] | undefined;
  const audioStreams = (streams ?? []).filter((s) => (s.Type ?? s.type) === 'Audio');
  if (!audioStreams.length) return { index: null };
  const defIdx = ms.DefaultAudioStreamIndex ?? ms.defaultAudioStreamIndex;
  const chosen =
    defIdx != null
      ? audioStreams.find((s) => (s.Index ?? s.index) === defIdx) ?? audioStreams[0]
      : audioStreams[0];
  const idx = chosen?.Index ?? chosen?.index;
  if (idx != null && Number.isFinite(Number(idx))) return { index: Number(idx) };
  return { index: null };
}

export function pickMobileSourceAndMode(
  info: Record<string, unknown>,
  options?: { preferDirectOverHls?: boolean }
): MobileSourcePick | null {
  const ms = defaultMediaSourceFromPlaybackInfo(info);
  if (!ms) return null;
  const msid = ms.Id ?? ms.id;
  if (typeof msid !== 'string' || !msid) return null;
  const sessionId = String(
    info.PlaySessionId ?? info.playSessionId ?? ms.PlaySessionId ?? ms.playSessionId ?? ''
  );
  if (!sessionId) return null;
  const transUrl = String(ms.TranscodingUrl ?? ms.transcodingUrl ?? '').toLowerCase();
  let useHls = transUrl.includes('m3u8');
  if (options?.preferDirectOverHls && useHls) {
    useHls = false;
  }
  const supportsDirectPlay = ms.SupportsDirectPlay === true || ms.supportsDirectPlay === true;
  const playMethod: MobileSourcePick['playMethod'] = useHls
    ? 'Transcode'
    : supportsDirectPlay
      ? 'DirectPlay'
      : 'DirectStream';
  const { index: audioStreamIndex } = defaultAudioFromMediaSource(ms);
  return { mediaSourceId: msid, playSessionId: sessionId, useHls, playMethod, audioStreamIndex };
}

/** POST playback-info; returns Emby JSON (PascalCase) with MediaSources and PlaySessionId. */
export async function postEmbyPlaybackInfo(
  itemId: string,
  body?: {
    max_streaming_bitrate?: number;
    start_time_ticks?: number;
    device_profile?: Record<string, unknown> | null;
  }
): Promise<Record<string, unknown>> {
  return apiRequest<Record<string, unknown>>(
    `/api/emby/items/${encodeURIComponent(itemId)}/playback-info`,
    {
      method: 'POST',
      body: JSON.stringify({
        max_streaming_bitrate: body?.max_streaming_bitrate ?? 40_000_000,
        start_time_ticks: body?.start_time_ticks ?? 0,
        device_profile: body?.device_profile ?? MOBILE_DEVICE_PROFILE,
      }),
    }
  );
}

export function parseEmbyPlaybackInfoSession(data: Record<string, unknown>): {
  playSessionId: string;
  mediaSourceId: string;
} | null {
  const pick = pickMobileSourceAndMode(data);
  if (!pick) return null;
  return { playSessionId: pick.playSessionId, mediaSourceId: pick.mediaSourceId };
}

export async function reportEmbyPlaybackStart(body: EmbyPlaybackStartBody): Promise<void> {
  const payload: Record<string, unknown> = {
    ItemId: body.ItemId,
    MediaSourceId: body.MediaSourceId ?? undefined,
    PlaySessionId: body.PlaySessionId ?? undefined,
    PositionTicks: body.PositionTicks ?? 0,
    PlayMethod: body.PlayMethod ?? 'DirectStream',
    IsPaused: body.IsPaused ?? false,
  };
  if (body.AudioStreamIndex != null) payload.AudioStreamIndex = body.AudioStreamIndex;
  await apiRequest<{ success?: boolean }>('/api/emby/playback/start', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function reportEmbyPlaybackProgress(body: EmbyPlaybackProgressBody): Promise<void> {
  const payload: Record<string, unknown> = {
    ItemId: body.ItemId,
    MediaSourceId: body.MediaSourceId ?? undefined,
    PlaySessionId: body.PlaySessionId ?? undefined,
    PositionTicks: body.PositionTicks ?? 0,
    PlayMethod: body.PlayMethod ?? 'DirectStream',
    IsPaused: body.IsPaused ?? false,
    EventName: body.EventName ?? 'TimeUpdate',
  };
  if (body.AudioStreamIndex != null) payload.AudioStreamIndex = body.AudioStreamIndex;
  await apiRequest<{ success?: boolean }>('/api/emby/playback/progress', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function reportEmbyPlaybackStopped(body: EmbyPlaybackStoppedBody): Promise<void> {
  await apiRequest<{ success?: boolean }>('/api/emby/playback/stopped', {
    method: 'POST',
    body: JSON.stringify({
      ItemId: body.ItemId,
      MediaSourceId: body.MediaSourceId ?? undefined,
      PlaySessionId: body.PlaySessionId ?? undefined,
      PositionTicks: body.PositionTicks ?? 0,
    }),
  });
}

/**
 * Bastion-proxied Emby image URL (JWT query). Caller must supply a valid API base and token.
 */
export function buildEmbyImageUrl(
  itemId: string,
  baseUrl: string,
  token: string,
  options?: { maxWidth?: number; imageType?: string; index?: number; tag?: string | null }
): string {
  const params = new URLSearchParams();
  params.set('token', token);
  const mw = options?.maxWidth ?? 400;
  params.set('max_width', String(mw));
  if (options?.index != null) params.set('index', String(options.index));
  if (options?.tag) params.set('tag', options.tag);
  const imageType = options?.imageType ?? 'Primary';
  const qs = params.toString();
  return `${baseUrl.replace(/\/$/, '')}/api/emby/image/${encodeURIComponent(itemId)}/${encodeURIComponent(imageType)}?${qs}`;
}

const root = (baseUrl: string) => baseUrl.replace(/\/$/, '');

/**
 * Bastion-proxied Emby HLS master playlist URL (JWT query).
 * Segments are rewritten server-side to relay through Bastion.
 */
export function buildEmbyHlsUrl(
  itemId: string,
  baseUrl: string,
  token: string,
  opts: { mediaSourceId: string; playSessionId: string; audioStreamIndex?: number | null }
): string {
  const params = new URLSearchParams();
  params.set('token', token);
  params.set('media_source_id', opts.mediaSourceId);
  params.set('play_session_id', opts.playSessionId);
  if (opts.audioStreamIndex != null) params.set('audio_stream_index', String(opts.audioStreamIndex));
  return `${root(baseUrl)}/api/emby/hls/${encodeURIComponent(itemId)}/master.m3u8?${params.toString()}`;
}

/**
 * Bastion-proxied progressive video stream URL (JWT query).
 */
export function buildEmbyVideoStreamUrl(
  itemId: string,
  baseUrl: string,
  token: string,
  opts: {
    mediaSourceId: string;
    playSessionId: string;
    startTimeTicks?: number;
    static?: boolean;
    audioStreamIndex?: number | null;
  }
): string {
  const params = new URLSearchParams();
  params.set('token', token);
  params.set('media_source_id', opts.mediaSourceId);
  params.set('play_session_id', opts.playSessionId);
  if (opts.startTimeTicks != null && opts.startTimeTicks > 0) {
    params.set('start_time_ticks', String(opts.startTimeTicks));
  }
  params.set('static', opts.static === false ? 'false' : 'true');
  if (opts.audioStreamIndex != null) params.set('audio_stream_index', String(opts.audioStreamIndex));
  return `${root(baseUrl)}/api/emby/video-stream/${encodeURIComponent(itemId)}?${params.toString()}`;
}
