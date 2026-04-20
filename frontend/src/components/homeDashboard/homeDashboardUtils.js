import rssService from '../../services/rssService';

export const WIDGET_TYPES = [
  { type: 'nav_links', label: 'Navigation links' },
  { type: 'markdown_card', label: 'Markdown note' },
  { type: 'scratchpad', label: 'Scratch pad' },
  { type: 'rss_headlines', label: 'RSS headlines' },
  { type: 'org_agenda', label: 'Org agenda' },
  { type: 'folder_shortcuts', label: 'Folder shortcuts' },
  { type: 'pinned_documents', label: 'Pinned documents' },
  { type: 'folder_images', label: 'Folder images' },
  { type: 'artifact_embed', label: 'Saved artifact' },
];

/** Default grid cell per widget type (12-column grid). */
export function defaultGridForWidgetType(type) {
  switch (type) {
    case 'org_agenda':
      return { x: 0, y: 0, w: 6, h: 8 };
    case 'folder_shortcuts':
      return { x: 0, y: 0, w: 6, h: 4 };
    case 'pinned_documents':
      return { x: 0, y: 0, w: 6, h: 6 };
    case 'folder_images':
      return { x: 0, y: 0, w: 6, h: 10 };
    case 'rss_headlines':
      return { x: 0, y: 0, w: 6, h: 6 };
    case 'markdown_card':
      return { x: 0, y: 0, w: 6, h: 5 };
    case 'scratchpad':
      return { x: 0, y: 0, w: 6, h: 6 };
    case 'artifact_embed':
      return { x: 0, y: 0, w: 6, h: 8 };
    case 'nav_links':
    default:
      return { x: 0, y: 0, w: 12, h: 3 };
  }
}

/**
 * Assign grid positions in a two-column flow (left then right, then next row).
 */
export function assignDefaultGridsToWidgets(widgets) {
  let maxY = 0;
  widgets.forEach((w) => {
    if (w?.grid && typeof w.grid.y === 'number' && typeof w.grid.h === 'number') {
      maxY = Math.max(maxY, w.grid.y + w.grid.h);
    }
  });
  let leftY = maxY;
  let rightY = maxY;
  let idx = 0;
  return widgets.map((w) => {
    if (w.grid && typeof w.grid.x === 'number') {
      return w;
    }
    const base = defaultGridForWidgetType(w.type);
    const isLeft = idx % 2 === 0;
    idx += 1;
    if (isLeft) {
      const g = { x: 0, y: leftY, w: Math.min(base.w, 6), h: base.h };
      leftY += base.h;
      return { ...w, grid: { ...g } };
    }
    const g = { x: 6, y: rightY, w: Math.min(base.w, 6), h: base.h };
    rightY += base.h;
    return { ...w, grid: { ...g } };
  });
}

/** Strip grid and order widgets by (y, x) for stack mode. */
export function widgetsForStackMode(widgets) {
  const sorted = [...widgets].sort((a, b) => {
    const ay = a.grid?.y ?? 0;
    const by = b.grid?.y ?? 0;
    if (ay !== by) return ay - by;
    const ax = a.grid?.x ?? 0;
    const bx = b.grid?.x ?? 0;
    return ax - bx;
  });
  return sorted.map(({ grid, ...rest }) => rest);
}

export function newWidgetId() {
  return typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `w-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export function emptyWidget(type) {
  switch (type) {
    case 'nav_links':
      return { type: 'nav_links', id: newWidgetId(), config: { items: [] } };
    case 'markdown_card':
      return {
        type: 'markdown_card',
        id: newWidgetId(),
        config: { title: '', body: '' },
      };
    case 'scratchpad':
      return {
        type: 'scratchpad',
        id: newWidgetId(),
        config: { show_labels: true },
      };
    case 'rss_headlines':
      return {
        type: 'rss_headlines',
        id: newWidgetId(),
        config: { feed_id: null, limit: 8, read_filter: 'unread' },
      };
    case 'org_agenda':
      return {
        type: 'org_agenda',
        id: newWidgetId(),
        config: {
          days_ahead: 7,
          include_scheduled: true,
          include_deadlines: true,
          include_appointments: true,
        },
      };
    case 'folder_shortcuts':
      return {
        type: 'folder_shortcuts',
        id: newWidgetId(),
        config: { items: [] },
      };
    case 'pinned_documents':
      return {
        type: 'pinned_documents',
        id: newWidgetId(),
        config: { limit: 10, show_preview: false },
      };
    case 'folder_images':
      return {
        type: 'folder_images',
        id: newWidgetId(),
        config: { folder_id: null, scan_limit: 500, include_subfolders: false },
      };
    case 'artifact_embed':
      return {
        type: 'artifact_embed',
        id: newWidgetId(),
        config: { artifact_id: null },
      };
    default:
      return null;
  }
}

/**
 * @param {string|null} feedId
 * @param {number} limit
 * @param {'all'|'unread'|'read'} [readFilter]
 */
export async function loadRssHeadlines(feedId, limit, readFilter = 'all') {
  const rf = readFilter === 'read' || readFilter === 'unread' ? readFilter : 'all';
  const opts = rf === 'all' ? {} : { readFilter: rf };
  if (feedId) {
    return rssService.getFeedArticles(feedId, limit, opts);
  }
  const feeds = await rssService.getFeeds();
  if (!feeds?.length) return [];
  const take = Math.min(feeds.length, 4);
  const per = Math.max(1, Math.ceil(limit / take));
  const batches = await Promise.all(
    feeds.slice(0, take).map((f) => rssService.getFeedArticles(f.feed_id, per, opts))
  );
  const merged = batches.flat();
  merged.sort((a, b) => {
    const da = new Date(a.published_date || a.created_at || 0).getTime();
    const db = new Date(b.published_date || b.created_at || 0).getTime();
    return db - da;
  });
  return merged.slice(0, limit);
}

export function widgetTitle(w) {
  const found = WIDGET_TYPES.find((x) => x.type === w.type);
  return found ? found.label : w.type;
}

/**
 * Card header label: for `artifact_embed`, use the saved artifact title when known
 * (from `savedArtifactService.list()` keyed by id). Otherwise same as `widgetTitle`.
 * @param {object} widget
 * @param {Map<string, string>|undefined} idToTitle - artifact id → trimmed title
 */
export function resolveWidgetCardTitle(widget, idToTitle) {
  if (!widget || widget.type !== 'artifact_embed') {
    return widgetTitle(widget);
  }
  const aid = widget.config?.artifact_id;
  if (!aid || !String(aid).trim()) {
    return widgetTitle(widget);
  }
  const key = String(aid);
  const t = idToTitle?.get?.(key);
  if (t && String(t).trim()) {
    return String(t).trim();
  }
  return widgetTitle(widget);
}

/** Build react-grid-layout layout array from widgets. */
export function gridLayoutFromWidgets(widgets) {
  return (widgets || []).map((w) => {
    const g = w.grid || defaultGridForWidgetType(w.type);
    return {
      i: w.id,
      x: g.x,
      y: g.y,
      w: g.w,
      h: g.h,
      minW: 2,
      minH: w.type === 'scratchpad' ? 3 : 2,
    };
  });
}

/** Merge RGL layout change into widgets (by id). */
export function applyGridLayoutToWidgets(widgets, layout) {
  const byId = new Map(layout.map((l) => [String(l.i), l]));
  return widgets.map((w) => {
    const l = byId.get(String(w.id));
    if (!l) return w;
    return {
      ...w,
      grid: { x: l.x, y: l.y, w: l.w, h: l.h },
    };
  });
}

/**
 * Prefer a layout array that includes every widget id exactly once (avoids stale `lg`
 * after add-widget or when the active breakpoint updates before `allLayouts.lg`).
 */
export function pickCanonicalGridLayout(currentLayout, allLayouts, widgets) {
  const widgetIds = (widgets || []).map((w) => String(w.id));
  const n = widgetIds.length;
  const idSet = new Set(widgetIds);
  const valid = (layout) => {
    if (!layout?.length || layout.length !== n) return false;
    const seen = new Set();
    for (const item of layout) {
      const id = String(item.i);
      if (!idSet.has(id) || seen.has(id)) return false;
      seen.add(id);
    }
    return true;
  };
  const order = [
    currentLayout,
    allLayouts?.lg,
    allLayouts?.md,
    allLayouts?.sm,
    allLayouts?.xs,
    allLayouts?.xxs,
  ];
  for (const layout of order) {
    if (valid(layout)) return layout;
  }
  return null;
}

/** Flatten folder tree for pickers: [{ folder_id, label }] */
export function flattenFolderTree(nodes, prefix = '') {
  if (!nodes?.length) return [];
  const out = [];
  for (const n of nodes) {
    const name = n.name || n.folder_name || 'Folder';
    const label = prefix ? `${prefix} / ${name}` : name;
    const id = n.folder_id;
    if (id) out.push({ folder_id: id, label });
    if (n.children?.length) {
      out.push(...flattenFolderTree(n.children, label));
    }
  }
  return out;
}
