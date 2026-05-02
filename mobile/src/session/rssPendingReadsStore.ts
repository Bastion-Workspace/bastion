import * as SecureStore from 'expo-secure-store';

const KEY = 'bastion_rss_pending_reads_v1';
const MAX_IDS = 500;

async function readIds(): Promise<string[]> {
  try {
    const raw = await SecureStore.getItemAsync(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((x): x is string => typeof x === 'string' && x.length > 0 && x.length < 256);
  } catch {
    return [];
  }
}

async function writeIds(ids: string[]): Promise<void> {
  try {
    const uniq = [...new Set(ids)].slice(-MAX_IDS);
    await SecureStore.setItemAsync(KEY, JSON.stringify(uniq));
  } catch {
    /* ignore */
  }
}

/** Queue an article id so mark-as-read can complete after app suspend or retry on next launch. */
export async function enqueuePendingMarkRead(articleId: string): Promise<void> {
  if (!articleId || articleId.length > 256) return;
  const cur = await readIds();
  if (cur.includes(articleId)) return;
  await writeIds([...cur, articleId]);
}

export async function dequeuePendingMarkRead(articleId: string): Promise<void> {
  const cur = await readIds();
  await writeIds(cur.filter((x) => x !== articleId));
}

/** Returns pending ids and clears the queue (caller should attempt API and re-enqueue failures). */
export async function takeAllPendingMarkReads(): Promise<string[]> {
  const cur = await readIds();
  if (cur.length === 0) return [];
  try {
    await SecureStore.deleteItemAsync(KEY);
  } catch {
    /* ignore */
  }
  return cur;
}
