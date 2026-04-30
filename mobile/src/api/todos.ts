import { apiRequest } from './client';

/** One row from GET /api/todos — matches org search / org_todo_service list shape. */
export type OrgTodoListItem = {
  filename: string;
  file_path: string;
  heading: string;
  line_number: number;
  level?: number;
  /** Ancestor headings from org hierarchy (same as web OrgTodosView). */
  parent_path?: string[];
  /** Org heading levels for each segment in parent_path. */
  parent_levels?: number[];
  todo_state?: string | null;
  tags?: string[];
  preview?: string;
  body_preview?: string;
  creation_timestamp?: string | null;
  notes?: { timestamp: string; text: string }[];
  scheduled?: string | null;
  deadline?: string | null;
  priority?: string | null;
  document_id?: string | null;
  closed?: string | null;
};

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
  results?: OrgTodoListItem[];
  count?: number;
  files_searched?: number;
  total_matches?: number;
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
