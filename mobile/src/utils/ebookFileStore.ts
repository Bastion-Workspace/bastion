import * as FileSystem from 'expo-file-system';

const CACHE_DIR_NAME = 'ebooks/';
const META_EXT = '.json';
const EPUB_EXT = '.epub';
const DEFAULT_QUOTA = 8;

function cacheRoot(): string {
  const base = FileSystem.documentDirectory;
  if (!base) {
    throw new Error('documentDirectory is not available');
  }
  return `${base}${CACHE_DIR_NAME}`;
}

async function ensureCacheDir(): Promise<string> {
  const root = cacheRoot();
  const info = await FileSystem.getInfoAsync(root);
  if (!info.exists) {
    await FileSystem.makeDirectoryAsync(root, { intermediates: true });
  }
  return root;
}

export type EbookCacheMeta = {
  digest: string;
  title: string;
  catalog_id: string;
  acquisition_url: string;
  cached_at: string;
};

function epubPath(digest: string): string {
  return `${cacheRoot()}${digest}${EPUB_EXT}`;
}

/** Whether a cached EPUB file exists for this digest (no file read). */
export async function ebookCacheExists(digest: string): Promise<boolean> {
  const epub = epubPath(digest);
  const info = await FileSystem.getInfoAsync(epub);
  return info.exists;
}

/** Local file URI for the cached EPUB; same path `ebookCachePut` writes. */
export function ebookCacheGetUri(digest: string): string {
  return epubPath(digest);
}

function metaPath(digest: string): string {
  return `${cacheRoot()}${digest}${META_EXT}`;
}

export async function ebookCachePut(
  digest: string,
  data: ArrayBuffer,
  meta: Omit<EbookCacheMeta, 'digest' | 'cached_at'>
): Promise<void> {
  const root = await ensureCacheDir();
  const b64 = arrayBufferToBase64(data);
  const fullMeta: EbookCacheMeta = {
    digest,
    ...meta,
    cached_at: new Date().toISOString(),
  };
  await FileSystem.writeAsStringAsync(`${root}${digest}${EPUB_EXT}`, b64, {
    encoding: FileSystem.EncodingType.Base64,
  });
  await FileSystem.writeAsStringAsync(`${root}${digest}${META_EXT}`, JSON.stringify(fullMeta), {
    encoding: FileSystem.EncodingType.UTF8,
  });
}

export async function ebookCacheGet(digest: string): Promise<{ data: ArrayBuffer; meta: EbookCacheMeta } | null> {
  const epub = epubPath(digest);
  const metaFile = metaPath(digest);
  const epubInfo = await FileSystem.getInfoAsync(epub);
  if (!epubInfo.exists) {
    return null;
  }
  const b64 = await FileSystem.readAsStringAsync(epub, { encoding: FileSystem.EncodingType.Base64 });
  const data = base64ToArrayBuffer(b64);
  let meta: EbookCacheMeta = {
    digest,
    title: 'Book',
    catalog_id: '',
    acquisition_url: '',
    cached_at: '',
  };
  try {
    const metaInfo = await FileSystem.getInfoAsync(metaFile);
    if (metaInfo.exists) {
      const raw = await FileSystem.readAsStringAsync(metaFile);
      meta = { ...meta, ...JSON.parse(raw) };
    }
  } catch {
    // keep defaults
  }
  return { data, meta };
}

export async function ebookCacheDelete(digest: string): Promise<void> {
  const epub = epubPath(digest);
  const metaFile = metaPath(digest);
  for (const p of [epub, metaFile]) {
    try {
      const info = await FileSystem.getInfoAsync(p);
      if (info.exists) {
        await FileSystem.deleteAsync(p, { idempotent: true });
      }
    } catch {
      // ignore
    }
  }
}

export async function ebookCacheList(): Promise<EbookCacheMeta[]> {
  const root = cacheRoot();
  const info = await FileSystem.getInfoAsync(root);
  if (!info.exists || !info.isDirectory) {
    return [];
  }
  const entries = await FileSystem.readDirectoryAsync(root);
  const out: EbookCacheMeta[] = [];
  for (const name of entries) {
    if (!name.endsWith(META_EXT)) continue;
    const digest = name.slice(0, -META_EXT.length);
    try {
      const raw = await FileSystem.readAsStringAsync(`${root}${name}`);
      out.push({ ...JSON.parse(raw), digest });
    } catch {
      out.push({
        digest,
        title: 'Book',
        catalog_id: '',
        acquisition_url: '',
        cached_at: '',
      });
    }
  }
  return out.sort((a, b) => (a.cached_at < b.cached_at ? 1 : -1));
}

export async function ebookCacheEnforceQuota(maxEntries: number = DEFAULT_QUOTA): Promise<void> {
  const list = await ebookCacheList();
  if (list.length <= maxEntries) return;
  const victims = list.slice(maxEntries);
  for (const v of victims) {
    await ebookCacheDelete(v.digest);
  }
}

/** Used by the EPUB WebView reader to embed bytes in inline HTML (avoids file:// fetch). */
export function arrayBufferToBase64(buffer: ArrayBuffer): string {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    const sub = bytes.subarray(i, i + chunk);
    binary += String.fromCharCode.apply(null, Array.from(sub) as unknown as number[]);
  }
  return btoa(binary);
}

function base64ToArrayBuffer(b64: string): ArrayBuffer {
  const binary = atob(b64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}
