/**
 * Ephemeral EPUB blob cache (IndexedDB), keyed by KOReader partial MD5 digest.
 */
const DB_NAME = 'bastion_ebook_cache';
const STORE = 'blobs';
const DB_VERSION = 1;

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => reject(req.error);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: 'digest' });
      }
    };
    req.onsuccess = () => resolve(req.result);
  });
}

export async function ebookCachePut(digest, arrayBuffer, meta = {}) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put({
      digest,
      data: arrayBuffer,
      updatedAt: Date.now(),
      ...meta,
    });
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function ebookCacheGet(digest) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).get(digest);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
}

export async function ebookCacheDelete(digest) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).delete(digest);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/** Evict oldest entries beyond maxEntries (by updatedAt). */
export async function ebookCacheEnforceQuota(maxEntries = 8) {
  const db = await openDb();
  const all = await new Promise((resolve, reject) => {
    const out = [];
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).openCursor();
    req.onerror = () => reject(req.error);
    req.onsuccess = (e) => {
      const c = e.target.result;
      if (c) {
        out.push(c.value);
        c.continue();
      } else {
        resolve(out);
      }
    };
  });
  all.sort((a, b) => (a.updatedAt || 0) - (b.updatedAt || 0));
  const overflow = all.length - maxEntries;
  if (overflow <= 0) return;
  const dbw = await openDb();
  return new Promise((resolve, reject) => {
    const tx = dbw.transaction(STORE, 'readwrite');
    for (let i = 0; i < overflow; i += 1) {
      tx.objectStore(STORE).delete(all[i].digest);
    }
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
