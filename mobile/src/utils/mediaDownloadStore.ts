import * as FileSystem from 'expo-file-system';
import type { MusicTrack } from '../api/media';
import { getStreamProxyUrl } from '../api/media';

const DIR_NAME = 'media/audio/';
const INDEX_FILE = 'media/index.json';

function cacheRoot(): string {
  const base = FileSystem.documentDirectory;
  if (!base) {
    throw new Error('documentDirectory is not available');
  }
  return `${base}${DIR_NAME}`;
}

async function ensureDir(): Promise<string> {
  const root = cacheRoot();
  const info = await FileSystem.getInfoAsync(root);
  if (!info.exists) {
    await FileSystem.makeDirectoryAsync(root, { intermediates: true });
  }
  return root;
}

function indexPath(): string {
  const base = FileSystem.documentDirectory;
  if (!base) throw new Error('documentDirectory is not available');
  return `${base}${INDEX_FILE}`;
}

export type MediaDownloadEntry = {
  track_id: string;
  title: string;
  artist?: string | null;
  album?: string | null;
  duration?: number | null;
  service_type?: string | null;
  parent_id?: string | null;
  /** Local file URI (file://...) */
  local_uri: string;
  file_size?: number | null;
  downloaded_at: string;
};

async function readIndex(): Promise<MediaDownloadEntry[]> {
  const p = indexPath();
  const info = await FileSystem.getInfoAsync(p);
  if (!info.exists) return [];
  try {
    const raw = await FileSystem.readAsStringAsync(p);
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? (parsed as MediaDownloadEntry[]) : [];
  } catch {
    return [];
  }
}

async function writeIndex(entries: MediaDownloadEntry[]): Promise<void> {
  const dir = FileSystem.documentDirectory;
  if (!dir) throw new Error('documentDirectory is not available');
  const mediaDir = `${dir}media/`;
  const dInfo = await FileSystem.getInfoAsync(mediaDir);
  if (!dInfo.exists) {
    await FileSystem.makeDirectoryAsync(mediaDir, { intermediates: true });
  }
  await FileSystem.writeAsStringAsync(indexPath(), JSON.stringify(entries), {
    encoding: FileSystem.EncodingType.UTF8,
  });
}

function localFileUri(trackId: string): string {
  return `${cacheRoot()}${encodeURIComponent(trackId)}`;
}

export async function isDownloaded(trackId: string): Promise<boolean> {
  const list = await readIndex();
  return list.some((e) => e.track_id === trackId);
}

export async function getLocalPath(trackId: string): Promise<string | null> {
  const list = await readIndex();
  const hit = list.find((e) => e.track_id === trackId);
  if (!hit) return null;
  const info = await FileSystem.getInfoAsync(hit.local_uri);
  if (!info.exists) return null;
  return hit.local_uri;
}

export async function listDownloads(): Promise<MediaDownloadEntry[]> {
  const list = await readIndex();
  const out: MediaDownloadEntry[] = [];
  for (const e of list) {
    const info = await FileSystem.getInfoAsync(e.local_uri);
    if (info.exists) {
      out.push({
        ...e,
        file_size: typeof info.size === 'number' ? info.size : e.file_size,
      });
    }
  }
  return out.sort((a, b) => (a.downloaded_at < b.downloaded_at ? 1 : -1));
}

export async function removeDownload(trackId: string): Promise<void> {
  const list = await readIndex();
  const entry = list.find((e) => e.track_id === trackId);
  if (entry) {
    try {
      const info = await FileSystem.getInfoAsync(entry.local_uri);
      if (info.exists) {
        await FileSystem.deleteAsync(entry.local_uri, { idempotent: true });
      }
    } catch {
      // ignore
    }
  }
  await writeIndex(list.filter((e) => e.track_id !== trackId));
}

export async function clearAllDownloads(): Promise<void> {
  const list = await readIndex();
  for (const e of list) {
    try {
      const info = await FileSystem.getInfoAsync(e.local_uri);
      if (info.exists) {
        await FileSystem.deleteAsync(e.local_uri, { idempotent: true });
      }
    } catch {
      // ignore
    }
  }
  await writeIndex([]);
}

export type DownloadProgress = {
  loaded: number;
  total: number;
  percent: number;
};

/**
 * Download track audio to app documents. Uses stream-proxy URL with token.
 */
export async function downloadTrack(
  track: MusicTrack,
  options: {
    serviceType?: string | null;
    parentId?: string | null;
    onProgress?: (p: DownloadProgress) => void;
  } = {}
): Promise<MediaDownloadEntry> {
  await ensureDir();
  const dest = localFileUri(track.id);
  const url = await getStreamProxyUrl(track.id, {
    serviceType: options.serviceType ?? track.service_type ?? undefined,
    parentId: options.parentId ?? undefined,
  });

  options.onProgress?.({ loaded: 0, total: 0, percent: 0 });

  const result = await (
    FileSystem as typeof FileSystem & {
      downloadAsync: (uri: string, fileUri: string) => Promise<{ uri: string; status: number }>;
    }
  ).downloadAsync(url, dest);
  if (!result?.uri) {
    throw new Error('Download failed');
  }

  options.onProgress?.({
    loaded: result.status === 200 ? 1 : 0,
    total: 1,
    percent: 100,
  });

  let size: number | null = null;
  try {
    const fi = await FileSystem.getInfoAsync(result.uri);
    if (fi.exists && typeof fi.size === 'number') size = fi.size;
  } catch {
    // ignore
  }

  const entry: MediaDownloadEntry = {
    track_id: track.id,
    title: track.title,
    artist: track.artist,
    album: track.album,
    duration: track.duration ?? undefined,
    service_type: track.service_type ?? options.serviceType ?? undefined,
    parent_id: options.parentId ?? undefined,
    local_uri: result.uri,
    file_size: size,
    downloaded_at: new Date().toISOString(),
  };

  const list = await readIndex();
  const next = list.filter((e) => e.track_id !== track.id);
  next.push(entry);
  await writeIndex(next);
  return entry;
}
