import type { OrgTodoListItem } from '../api/todos';

export type TodoTreeNode = {
  heading: string | null;
  level: number;
  todos: OrgTodoListItem[];
  children: TodoTreeNode[];
  path: string[];
  isOrphan: boolean;
};

export function buildHierarchicalTree(todos: OrgTodoListItem[]): TodoTreeNode {
  const tree: TodoTreeNode = {
    heading: null,
    level: 0,
    todos: [],
    children: [],
    path: [],
    isOrphan: true,
  };

  const findOrCreateNode = (
    parentNode: TodoTreeNode,
    pathSegment: string,
    level: number,
    fullPath: string[]
  ): TodoTreeNode => {
    let node = parentNode.children.find((child) => child.heading === pathSegment);
    if (!node) {
      node = {
        heading: pathSegment,
        level,
        todos: [],
        children: [],
        path: fullPath,
        isOrphan: false,
      };
      parentNode.children.push(node);
    }
    return node;
  };

  for (const todo of todos) {
    const parentPath = todo.parent_path ?? [];
    const parentLevels = todo.parent_levels ?? [];

    if (parentPath.length === 0) {
      tree.todos.push(todo);
    } else {
      let currentNode = tree;
      for (let i = 0; i < parentPath.length; i++) {
        const pathSegment = parentPath[i];
        const level = parentLevels[i] ?? i + 1;
        const fullPath = parentPath.slice(0, i + 1);
        currentNode = findOrCreateNode(currentNode, pathSegment, level, fullPath);
      }
      currentNode.todos.push(todo);
    }
  }

  return tree;
}

export function countTodosInSubtree(node: TodoTreeNode): number {
  let n = node.todos.length;
  for (const c of node.children) {
    n += countTodosInSubtree(c);
  }
  return n;
}

export function collectPathKeys(node: TodoTreeNode, out: Set<string>): void {
  if (node.path.length > 0) {
    out.add(JSON.stringify(node.path));
  }
  for (const c of node.children) {
    collectPathKeys(c, out);
  }
}

export function pathKey(path: string[]): string {
  return JSON.stringify(path);
}

export type FlatTodoRow =
  | {
      kind: 'file';
      key: string;
      filePath: string;
      filename: string;
      expanded: boolean;
      todoCount: number;
    }
  | {
      kind: 'section';
      key: string;
      filePath: string;
      path: string[];
      heading: string;
      depth: number;
      todoCount: number;
      expanded: boolean;
    }
  | { kind: 'todo'; key: string; filePath: string; item: OrgTodoListItem; depth: number };

export function groupTodosByFileOrdered(todos: OrgTodoListItem[]): Map<string, OrgTodoListItem[]> {
  const map = new Map<string, OrgTodoListItem[]>();
  for (const t of todos) {
    const fp = t.file_path?.trim();
    if (!fp) continue;
    const arr = map.get(fp);
    if (arr) arr.push(t);
    else map.set(fp, [t]);
  }
  return map;
}

function flattenNode(
  node: TodoTreeNode,
  ctx: {
    filePath: string;
    fileExpanded: boolean;
    expandedPaths: Set<string>;
    baseIndent: number;
    rows: FlatTodoRow[];
  }
): void {
  const total = countTodosInSubtree(node);
  if (total === 0) return;

  const nodeIndent = node.isOrphan
    ? ctx.baseIndent
    : node.level === 1
      ? ctx.baseIndent
      : ctx.baseIndent + (node.level - 1);

  const isPathExpanded =
    node.path.length === 0 ? true : ctx.expandedPaths.has(pathKey(node.path));

  if (node.heading) {
    ctx.rows.push({
      kind: 'section',
      key: `${ctx.filePath}::section::${pathKey(node.path)}`,
      filePath: ctx.filePath,
      path: [...node.path],
      heading: node.heading,
      depth: nodeIndent,
      todoCount: total,
      expanded: isPathExpanded,
    });
  }

  const showOrphanTodos = node.isOrphan && ctx.fileExpanded;
  const showNestedTodos = !node.isOrphan && isPathExpanded;

  if ((showOrphanTodos || showNestedTodos) && node.todos.length > 0) {
    for (const item of node.todos) {
      ctx.rows.push({
        kind: 'todo',
        key: `${ctx.filePath}::todo::${item.line_number}::${item.heading?.slice(0, 48) ?? ''}`,
        filePath: ctx.filePath,
        item,
        depth: nodeIndent + 1,
      });
    }
  }

  if (isPathExpanded && node.children.length > 0) {
    for (const child of node.children) {
      flattenNode(child, ctx);
    }
  }
}

/**
 * Flatten grouped todos for a FlatList: optional per-file headers, then section rows + todo rows
 * following the same hierarchy as the web OrgTodosView tree.
 */
export function flattenTodosForList(
  todos: OrgTodoListItem[],
  expandedFiles: Set<string>,
  expandedPaths: Set<string>
): FlatTodoRow[] {
  const byFile = groupTodosByFileOrdered(todos);
  const rows: FlatTodoRow[] = [];
  const showFileHeaders = byFile.size > 1;

  for (const [filePath, items] of byFile) {
    const tree = buildHierarchicalTree(items);
    const fileTotal = countTodosInSubtree(tree);
    if (fileTotal === 0) continue;

    const filename = items[0]?.filename || filePath.split('/').pop() || filePath;
    const fileExpanded = !showFileHeaders || expandedFiles.has(filePath);

    if (showFileHeaders) {
      rows.push({
        kind: 'file',
        key: `file::${filePath}`,
        filePath,
        filename,
        expanded: expandedFiles.has(filePath),
        todoCount: fileTotal,
      });
    }

    if (fileExpanded) {
      flattenNode(tree, {
        filePath,
        fileExpanded,
        expandedPaths,
        baseIndent: 0,
        rows,
      });
    }
  }

  return rows;
}
