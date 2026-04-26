import { apiRequest } from './client';

export type TodoListOptions = {
  scope?: string;
  states?: string[];
  tags?: string[];
  query?: string;
  limit?: number;
  includeArchives?: boolean;
};

export type TodoListResult = {
  success?: boolean;
  results?: unknown[];
  count?: number;
  files_searched?: number;
};

export async function listTodos(options: TodoListOptions = {}): Promise<TodoListResult> {
  const params = new URLSearchParams();
  params.set('scope', options.scope ?? 'all');
  if (options.states?.length) params.set('states', options.states.join(','));
  if (options.tags?.length) params.set('tags', options.tags.join(','));
  if (options.query) params.set('query', options.query);
  if (options.limit != null) params.set('limit', String(options.limit));
  if (options.includeArchives) params.set('include_archives', 'true');
  return apiRequest<TodoListResult>(`/api/todos?${params.toString()}`);
}

export type ToggleTodoBody = {
  file_path: string;
  line_number: number;
  heading_text?: string | null;
};

export async function toggleTodo(body: ToggleTodoBody): Promise<unknown> {
  return apiRequest('/api/todos/toggle', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export type UpdateTodoBody = ToggleTodoBody & {
  new_state?: string;
  new_text?: string;
  add_tags?: string[];
  remove_tags?: string[];
  scheduled?: string | null;
  deadline?: string | null;
  priority?: string | null;
};

export async function updateTodo(body: UpdateTodoBody): Promise<unknown> {
  return apiRequest('/api/todos/update', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}
