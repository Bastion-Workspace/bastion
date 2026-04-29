import { apiRequest } from './client';

export type TodoStatesPayload = {
  active: string[];
  done: string[];
  all: string[];
};

function normalizeUpper(list: string[] | undefined): string[] {
  return [...new Set((list ?? []).map((s) => String(s).trim().toUpperCase()).filter(Boolean))].sort();
}

/**
 * TODO keywords from the user's org-mode settings (GET /api/org/settings/todo-states).
 */
export async function getTodoStates(): Promise<TodoStatesPayload> {
  const res = await apiRequest<{ success?: boolean; states?: Partial<TodoStatesPayload> }>(
    '/api/org/settings/todo-states'
  );
  const raw = res.states;
  if (!raw) {
    return { active: [], done: [], all: [] };
  }
  return {
    active: normalizeUpper(raw.active as string[] | undefined),
    done: normalizeUpper(raw.done as string[] | undefined),
    all: normalizeUpper(raw.all as string[] | undefined),
  };
}
