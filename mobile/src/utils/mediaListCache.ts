import * as FileSystem from 'expo-file-system';

const CACHE_DIR_REL = 'media/list-cache/';

/** Library index responses (albums, artists, playlists). */
export const TTL_LIBRARY_MS = 5 * 60 * 1000;

/** Album/playlist track lists and artist album lists. */
export const TTL_TRACKS_MS = 15 * 60 * 1000;

type CacheEnvelope<T> = {
  data: T;
  fetchedAt: number;
};

function cacheDir(): string {
  const base = FileSystem.documentDirectory;
  if (!base) throw new Error('documentDirectory is not available');
  return `${base}${CACHE_DIR_REL}`;
}

function filePathForKey(key: string): string {
  const safe = encodeURIComponent(key).replace(/%/g, '_');
  return `${cacheDir()}${safe}.json`;
}

async function ensureCacheDir(): Promise<void> {
  const dir = cacheDir();
  const info = await FileSystem.getInfoAsync(dir);
  if (!info.exists) {
    await FileSystem.makeDirectoryAsync(dir, { intermediates: true });
  }
}

/**
 * Returns cached data if present. `stale` is true when older than ttlMs (caller may revalidate).
 */
export async function getCachedEntry<T>(key: string, ttlMs: number): Promise<{ data: T; stale: boolean } | null> {
  try {
    const p = filePathForKey(key);
    const info = await FileSystem.getInfoAsync(p);
    if (!info.exists) return null;
    const raw = await FileSystem.readAsStringAsync(p);
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== 'object' || !('data' in parsed) || !('fetchedAt' in parsed)) {
      return null;
    }
    const env = parsed as CacheEnvelope<T>;
    if (typeof env.fetchedAt !== 'number') return null;
    const age = Date.now() - env.fetchedAt;
    return { data: env.data, stale: age > ttlMs };
  } catch {
    return null;
  }
}

export async function setCachedEntry<T>(key: string, data: T): Promise<void> {
  await ensureCacheDir();
  const env: CacheEnvelope<T> = { data, fetchedAt: Date.now() };
  await FileSystem.writeAsStringAsync(filePathForKey(key), JSON.stringify(env), {
    encoding: FileSystem.EncodingType.UTF8,
  });
}

/** Removes all persisted list cache entries (e.g. after manual library refresh). */
export async function clearMediaListCache(): Promise<void> {
  try {
    const dir = cacheDir();
    const info = await FileSystem.getInfoAsync(dir);
    if (!info.exists) return;
    await FileSystem.deleteAsync(dir, { idempotent: true });
  } catch {
    // ignore
  }
}
