import * as SecureStore from 'expo-secure-store';

const KEY = 'app:todosListScopeV1';

export type TodosListScopePersisted = {
  listMode: 'all' | 'inbox' | 'file';
  singleFilePath: string | null;
};

function isPersistedScope(v: unknown): v is TodosListScopePersisted {
  if (!v || typeof v !== 'object') return false;
  const o = v as Record<string, unknown>;
  const m = o.listMode;
  if (m !== 'all' && m !== 'inbox' && m !== 'file') return false;
  const p = o.singleFilePath;
  if (p != null && typeof p !== 'string') return false;
  return true;
}

export async function loadTodosListScope(): Promise<TodosListScopePersisted | null> {
  try {
    const raw = await SecureStore.getItemAsync(KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (!isPersistedScope(parsed)) return null;
    if (parsed.listMode === 'file' && (!parsed.singleFilePath || !parsed.singleFilePath.trim())) {
      return { listMode: 'all', singleFilePath: null };
    }
    return parsed;
  } catch {
    return null;
  }
}

export async function saveTodosListScope(scope: TodosListScopePersisted): Promise<void> {
  try {
    await SecureStore.setItemAsync(KEY, JSON.stringify(scope));
  } catch {
    /* ignore */
  }
}
