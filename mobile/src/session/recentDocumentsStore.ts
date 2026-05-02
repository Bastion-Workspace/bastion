import * as SecureStore from 'expo-secure-store';

const KEY = 'bastion_recent_documents_v1';
const MAX_ENTRIES = 20;

export type RecentDocumentEntry = {
  document_id: string;
  title: string;
  opened_at: string;
};

function isValidEntry(x: unknown): x is RecentDocumentEntry {
  if (!x || typeof x !== 'object') return false;
  const o = x as Record<string, unknown>;
  return (
    typeof o.document_id === 'string' &&
    o.document_id.length > 0 &&
    o.document_id.length < 128 &&
    typeof o.title === 'string' &&
    typeof o.opened_at === 'string'
  );
}

export async function loadRecentDocuments(): Promise<RecentDocumentEntry[]> {
  try {
    const raw = await SecureStore.getItemAsync(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isValidEntry);
  } catch {
    return [];
  }
}

export async function recordRecentDocument(documentId: string, title: string): Promise<void> {
  const id = (documentId || '').trim();
  if (!id || id.includes('..') || id === '[id]') return;
  const t = (title || '').trim() || 'Document';
  try {
    const prev = await loadRecentDocuments();
    const entry: RecentDocumentEntry = {
      document_id: id,
      title: t.slice(0, 500),
      opened_at: new Date().toISOString(),
    };
    const without = prev.filter((e) => e.document_id !== id);
    const next = [entry, ...without].slice(0, MAX_ENTRIES);
    await SecureStore.setItemAsync(KEY, JSON.stringify(next));
  } catch {
    /* ignore storage errors */
  }
}
