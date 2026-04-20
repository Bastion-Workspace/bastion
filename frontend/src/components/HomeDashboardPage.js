import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { Alert, CircularProgress, Container, Stack } from '@mui/material';
import { Responsive, WidthProvider } from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

import apiService from '../services/apiService';
import rssService from '../services/rssService';
import savedArtifactService from '../services/savedArtifactService';
import {
  WIDGET_TYPES,
  assignDefaultGridsToWidgets,
  applyGridLayoutToWidgets,
  defaultGridForWidgetType,
  emptyWidget,
  gridLayoutFromWidgets,
  pickCanonicalGridLayout,
  resolveWidgetCardTitle,
  widgetsForStackMode,
} from './homeDashboard/homeDashboardUtils';
import {
  MarkdownCardView,
  NavLinksView,
  RssHeadlinesBlock,
  WidgetCard,
  WidgetEditor,
} from './homeDashboard/HomeDashboardPanels';
import {
  OrgAgendaBlock,
  FolderShortcutsView,
  PinnedDocumentsBlockWithAdd,
} from './homeDashboard/homeDashboardWidgetBlocks';
import { FolderImageSlideshowBlock } from './homeDashboard/homeDashboardFolderImageSlideshow';
import ArtifactEmbedBlock from './homeDashboard/ArtifactEmbedBlock';
import HomeDashboardChrome, { HomeDashboardDialogs } from './homeDashboard/HomeDashboardChrome';
import ScratchPadBlock from './homeDashboard/ScratchPadBlock';

const ResponsiveGridLayout = WidthProvider(Responsive);

const GRID_BREAKPOINTS = { lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 };
const GRID_COLS = { lg: 12, md: 12, sm: 12, xs: 8, xxs: 4 };

function normalizeLayoutDraft(raw) {
  const o = typeof raw === 'object' && raw ? raw : {};
  const mode = o.layout_mode === 'grid' ? 'grid' : 'stack';
  let widgets = Array.isArray(o.widgets) ? [...o.widgets] : [];
  if (mode === 'grid' && widgets.some((w) => !w.grid)) {
    widgets = assignDefaultGridsToWidgets(widgets);
  }
  return {
    schema_version: o.schema_version ?? 1,
    layout_mode: mode,
    widgets,
  };
}

export default function HomeDashboardPage() {
  const navigate = useNavigate();
  const { dashboardId: dashboardIdParam } = useParams();
  const queryClient = useQueryClient();

  const [editMode, setEditMode] = useState(false);
  const [draft, setDraft] = useState(null);
  const [saveError, setSaveError] = useState(null);
  const [addMenuAnchor, setAddMenuAnchor] = useState(null);

  const [createOpen, setCreateOpen] = useState(false);
  const [duplicateFromCurrent, setDuplicateFromCurrent] = useState(false);
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameName, setRenameName] = useState('');
  const [deleteOpen, setDeleteOpen] = useState(false);
  const { data: listData, isLoading: listLoading, error: listError } = useQuery(
    ['homeDashboards'],
    () => apiService.listHomeDashboards(),
    { staleTime: 30 * 1000 }
  );

  const dashboards = listData?.dashboards || [];
  const dashboardId = dashboardIdParam || null;

  const defaultDashboard = useMemo(
    () => dashboards.find((d) => d.is_default) || dashboards[0],
    [dashboards]
  );

  const idValid = dashboardId && dashboards.some((d) => d.id === dashboardId);
  const effectiveId = idValid ? dashboardId : null;

  useEffect(() => {
    if (listLoading || !dashboards.length) return;
    if (!dashboardIdParam) {
      const target = defaultDashboard?.id;
      if (target) navigate(`/home/${target}`, { replace: true });
      return;
    }
    if (!idValid && defaultDashboard?.id) {
      navigate(`/home/${defaultDashboard.id}`, { replace: true });
    }
  }, [listLoading, dashboards, dashboardIdParam, defaultDashboard, idValid, navigate]);

  const { data: serverLayout, isLoading: layoutLoading, error: layoutError } = useQuery(
    ['homeDashboardLayout', effectiveId],
    () => apiService.getHomeDashboardLayout(effectiveId),
    {
      enabled: Boolean(effectiveId),
      staleTime: 30 * 1000,
    }
  );

  const { data: feeds } = useQuery(['homeDashboardFeeds'], () => rssService.getFeeds(), {
    staleTime: 5 * 60 * 1000,
    enabled: editMode,
  });

  const startEdit = useCallback(() => {
    if (serverLayout) {
      const parsed = JSON.parse(JSON.stringify(serverLayout));
      setDraft(normalizeLayoutDraft(parsed));
      setEditMode(true);
      setSaveError(null);
    }
  }, [serverLayout]);

  const cancelEdit = useCallback(() => {
    setDraft(null);
    setEditMode(false);
    setSaveError(null);
    setAddMenuAnchor(null);
  }, []);

  const saveMutation = useMutation(
    ({ id, body }) => apiService.putHomeDashboardLayout(id, body),
    {
      onSuccess: (savedLayout, variables) => {
        if (savedLayout && variables?.id) {
          queryClient.setQueryData(['homeDashboardLayout', variables.id], savedLayout);
        }
        queryClient.invalidateQueries(['homeDashboardLayout', variables.id]);
        queryClient.invalidateQueries(['homeDashboardRss']);
        setEditMode(false);
        setDraft(null);
        setSaveError(null);
        setAddMenuAnchor(null);
      },
      onError: (err) => {
        const d = err?.response?.data?.detail;
        let msg = err?.message || 'Save failed';
        if (typeof d === 'string') msg = d;
        else if (Array.isArray(d) && d.length) {
          msg = d.map((x) => x.msg || JSON.stringify(x)).join('; ');
        }
        setSaveError(msg);
      },
    }
  );

  const createMutation = useMutation(
    async ({ name, duplicate }) => {
      const prev = queryClient.getQueryData(['homeDashboards']);
      const payload = {
        name: name?.trim() || undefined,
        duplicate_from_id: duplicate ? effectiveId : undefined,
      };
      const res = await apiService.createHomeDashboard(payload);
      const oldIds = new Set((prev?.dashboards || []).map((d) => d.id));
      const created = res.dashboards.find((d) => !oldIds.has(d.id));
      return { res, createdId: created?.id };
    },
    {
      onSuccess: ({ res, createdId }) => {
        queryClient.setQueryData(['homeDashboards'], res);
        setCreateOpen(false);
        if (createdId) navigate(`/home/${createdId}`);
      },
    }
  );

  const renameMutation = useMutation(
    ({ id, name }) => apiService.patchHomeDashboard(id, { name }),
    {
      onSuccess: (res) => {
        queryClient.setQueryData(['homeDashboards'], res);
        setRenameOpen(false);
      },
    }
  );

  const setDefaultMutation = useMutation(
    (id) => apiService.patchHomeDashboard(id, { is_default: true }),
    {
      onSuccess: (res) => {
        queryClient.setQueryData(['homeDashboards'], res);
      },
    }
  );

  const deleteMutation = useMutation((id) => apiService.deleteHomeDashboard(id), {
    onSuccess: (res) => {
      queryClient.setQueryData(['homeDashboards'], res);
      setDeleteOpen(false);
      const nextDefault = res.dashboards.find((d) => d.is_default) || res.dashboards[0];
      if (nextDefault) navigate(`/home/${nextDefault.id}`, { replace: true });
    },
  });

  const layout = editMode ? draft : serverLayout;
  const layoutNormalized = useMemo(
    () => normalizeLayoutDraft(layout),
    [layout]
  );
  const isGrid = layoutNormalized.layout_mode === 'grid';

  const artifactEmbedIds = useMemo(() => {
    const widgets = layoutNormalized.widgets || [];
    const ids = new Set();
    for (const w of widgets) {
      if (w.type === 'artifact_embed' && w.config?.artifact_id) {
        ids.add(String(w.config.artifact_id));
      }
    }
    return Array.from(ids);
  }, [layoutNormalized.widgets]);

  const { data: savedArtifactsListData } = useQuery(
    ['savedArtifactsList'],
    () => savedArtifactService.list(),
    {
      enabled: artifactEmbedIds.length > 0,
      staleTime: 60 * 1000,
    }
  );

  const savedArtifactsTitleMap = useMemo(() => {
    const m = new Map();
    for (const a of savedArtifactsListData?.artifacts || []) {
      if (a?.id) {
        m.set(String(a.id), String(a.title || '').trim());
      }
    }
    return m;
  }, [savedArtifactsListData?.artifacts]);

  const handleLayoutModeChange = useCallback((_, mode) => {
    if (!mode) return;
    setDraft((prev) => {
      if (!prev) return prev;
      if (mode === 'grid') {
        return {
          ...prev,
          layout_mode: 'grid',
          widgets: assignDefaultGridsToWidgets(prev.widgets),
        };
      }
      return {
        ...prev,
        layout_mode: 'stack',
        widgets: widgetsForStackMode(prev.widgets),
      };
    });
  }, []);

  const gridLayouts = useMemo(() => {
    const gl = gridLayoutFromWidgets(layoutNormalized.widgets);
    return { lg: gl, md: gl, sm: gl, xs: gl, xxs: gl };
  }, [layoutNormalized.widgets]);

  const onGridLayoutChange = useCallback((currentLayout, allLayouts) => {
    setDraft((prev) => {
      if (!prev || prev.layout_mode !== 'grid') return prev;
      const l = pickCanonicalGridLayout(currentLayout, allLayouts, prev.widgets);
      if (!l) return prev;
      return {
        ...prev,
        widgets: applyGridLayoutToWidgets(prev.widgets, l),
      };
    });
  }, []);

  const updateWidgetAt = (index, w) => {
    setDraft((prev) => {
      if (!prev) return prev;
      const widgets = [...prev.widgets];
      widgets[index] = w;
      return { ...prev, widgets };
    });
  };

  const removeWidget = (index) => {
    setDraft((prev) => {
      if (!prev) return prev;
      return { ...prev, widgets: prev.widgets.filter((_, i) => i !== index) };
    });
  };

  const moveWidget = (index, delta) => {
    setDraft((prev) => {
      if (!prev) return prev;
      const j = index + delta;
      if (j < 0 || j >= prev.widgets.length) return prev;
      const widgets = [...prev.widgets];
      [widgets[index], widgets[j]] = [widgets[j], widgets[index]];
      return { ...prev, widgets };
    });
  };

  const addWidget = (type) => {
    const base = emptyWidget(type);
    if (!base) return;
    setDraft((prev) => {
      if (!prev) return prev;
      if (prev.layout_mode === 'grid') {
        let maxY = 0;
        prev.widgets.forEach((x) => {
          const g = x.grid || defaultGridForWidgetType(x.type);
          maxY = Math.max(maxY, g.y + g.h);
        });
        const def = defaultGridForWidgetType(type);
        const nw = { ...base, grid: { x: 0, y: maxY, w: def.w, h: def.h } };
        return { ...prev, widgets: [...prev.widgets, nw] };
      }
      return { ...prev, widgets: [...prev.widgets, base] };
    });
  };

  const handleSave = () => {
    if (!draft || !effectiveId) return;
    setSaveError(null);
    for (const w of draft.widgets) {
      if (w.type === 'markdown_card') {
        const body = w.config?.body ?? '';
        if (body.length > 50000) {
          setSaveError('A markdown note exceeds the maximum length.');
          return;
        }
      }
      if (w.type === 'nav_links') {
        for (const it of w.config?.items || []) {
          const hasP = !!(it.path && String(it.path).trim());
          const hasH = !!(it.href && String(it.href).trim());
          if (hasP === hasH) {
            setSaveError('Each navigation link needs exactly one of path or external URL.');
            return;
          }
        }
      }
      if (w.type === 'folder_images') {
        const fid = w.config?.folder_id;
        if (!fid || !String(fid).trim()) {
          setSaveError('Folder images widget needs a folder selected.');
          return;
        }
      }
    }
    saveMutation.mutate({ id: effectiveId, body: draft });
  };

  useEffect(() => {
    if (!editMode) setSaveError(null);
  }, [editMode]);

  const renderWidgetView = (w) => {
    if (w.type === 'nav_links') return <NavLinksView config={w.config} navigate={navigate} />;
    if (w.type === 'markdown_card') return <MarkdownCardView config={w.config} />;
    if (w.type === 'scratchpad') {
      return <ScratchPadBlock showLabels={w.config?.show_labels !== false} />;
    }
    if (w.type === 'rss_headlines') return <RssHeadlinesBlock config={w.config} navigate={navigate} />;
    if (w.type === 'org_agenda') return <OrgAgendaBlock config={w.config} navigate={navigate} />;
    if (w.type === 'folder_shortcuts') return <FolderShortcutsView config={w.config} navigate={navigate} />;
    if (w.type === 'pinned_documents') return <PinnedDocumentsBlockWithAdd config={w.config} />;
    if (w.type === 'folder_images') return <FolderImageSlideshowBlock config={w.config} navigate={navigate} />;
    if (w.type === 'artifact_embed') return <ArtifactEmbedBlock config={w.config} />;
    return null;
  };

  const currentMeta = dashboards.find((d) => d.id === effectiveId);
  const isDefaultDashboard = Boolean(currentMeta?.is_default);
  const canDeleteDashboard = dashboards.length > 1;

  const openRename = () => {
    setRenameName(currentMeta?.name || '');
    setRenameOpen(true);
  };

  if (listLoading || (dashboardIdParam && !idValid && dashboards.length > 0)) {
    return (
      <Container maxWidth="md" sx={{ py: 4, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Container>
    );
  }

  if (listError) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Alert severity="error">Could not load dashboards.</Alert>
      </Container>
    );
  }

  if (!dashboards.length) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Alert severity="warning">No dashboards available.</Alert>
      </Container>
    );
  }

  if (!listLoading && dashboards.length > 0 && !effectiveId) {
    return (
      <Container maxWidth="md" sx={{ py: 4, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Container>
    );
  }

  if (layoutLoading && effectiveId) {
    return (
      <Container maxWidth="md" sx={{ py: 4, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Container>
    );
  }

  if (layoutError) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Alert severity="error">Could not load this dashboard layout.</Alert>
      </Container>
    );
  }

  const widgets = layoutNormalized.widgets || [];

  return (
    <Container maxWidth={isGrid ? false : 'md'} sx={{ py: 3 }}>
      <HomeDashboardChrome
        dashboards={dashboards}
        dashboardId={effectiveId}
        onSelectDashboard={(id) => navigate(`/home/${id}`)}
        editMode={editMode}
        onStartEdit={startEdit}
        onCancelEdit={cancelEdit}
        onSaveLayout={handleSave}
        saveLayoutLoading={saveMutation.isLoading}
        onOpenCreate={() => {
          setDuplicateFromCurrent(false);
          setCreateOpen(true);
        }}
        onOpenRename={openRename}
        onSetDefault={() => effectiveId && setDefaultMutation.mutate(effectiveId)}
        onOpenDelete={() => setDeleteOpen(true)}
        canDeleteDashboard={canDeleteDashboard}
        isDefaultDashboard={isDefaultDashboard}
        onAddWidgetClick={(e) => setAddMenuAnchor(e.currentTarget)}
        addMenuAnchor={addMenuAnchor}
        onCloseAddMenu={() => setAddMenuAnchor(null)}
        onPickWidgetType={(t) => addWidget(t)}
        widgetTypeOptions={WIDGET_TYPES}
        layoutMode={editMode ? draft?.layout_mode || 'stack' : layoutNormalized.layout_mode}
        onLayoutModeChange={handleLayoutModeChange}
        showLayoutModeToggle={editMode && Boolean(draft)}
      />
      {saveError ? (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setSaveError(null)}>
          {saveError}
        </Alert>
      ) : null}

      {isGrid ? (
        <ResponsiveGridLayout
          className="home-dashboard-grid"
          layouts={gridLayouts}
          breakpoints={GRID_BREAKPOINTS}
          cols={GRID_COLS}
          rowHeight={36}
          margin={[12, 12]}
          containerPadding={[0, 0]}
          onLayoutChange={editMode ? onGridLayoutChange : undefined}
          isDraggable={editMode}
          isResizable={editMode}
          draggableCancel="button,a,input,textarea,.MuiButton-root,.MuiIconButton-root,.MuiChip-root,.MuiSelect-root,.MuiMenuItem-root"
        >
          {widgets.map((w, i) => (
            <div key={w.id}>
              <WidgetCard
                title={resolveWidgetCardTitle(w, savedArtifactsTitleMap)}
                editMode={editMode}
                onDelete={() => removeWidget(i)}
                onUp={() => moveWidget(i, -1)}
                onDown={() => moveWidget(i, 1)}
                disableUp={i === 0}
                disableDown={i === widgets.length - 1}
                hideStackReorder
                cardContentOverflow={w.type === 'scratchpad' ? 'hidden' : 'auto'}
              >
                {editMode ? (
                  <WidgetEditor widget={w} feeds={feeds} onChange={(nw) => updateWidgetAt(i, nw)} />
                ) : (
                  renderWidgetView(w)
                )}
              </WidgetCard>
            </div>
          ))}
        </ResponsiveGridLayout>
      ) : (
        <Stack spacing={2}>
          {widgets.map((w, i) => (
            <WidgetCard
              key={w.id}
              title={resolveWidgetCardTitle(w, savedArtifactsTitleMap)}
              editMode={editMode}
              onDelete={() => removeWidget(i)}
              onUp={() => moveWidget(i, -1)}
              onDown={() => moveWidget(i, 1)}
              disableUp={i === 0}
              disableDown={i === widgets.length - 1}
              cardContentOverflow={w.type === 'scratchpad' ? 'hidden' : 'auto'}
            >
              {editMode ? (
                <WidgetEditor widget={w} feeds={feeds} onChange={(nw) => updateWidgetAt(i, nw)} />
              ) : (
                renderWidgetView(w)
              )}
            </WidgetCard>
          ))}
        </Stack>
      )}

      <HomeDashboardDialogs
        createOpen={createOpen}
        onCloseCreate={() => setCreateOpen(false)}
        onSubmitCreate={(name, dup) => createMutation.mutate({ name, duplicate: dup })}
        createLoading={createMutation.isLoading}
        duplicateByDefault={duplicateFromCurrent}
        onDuplicateByDefaultChange={setDuplicateFromCurrent}
        renameOpen={renameOpen}
        onCloseRename={() => setRenameOpen(false)}
        renameName={renameName}
        onRenameNameChange={setRenameName}
        onSubmitRename={() => {
          if (effectiveId && renameName.trim()) {
            renameMutation.mutate({ id: effectiveId, name: renameName.trim() });
          }
        }}
        renameLoading={renameMutation.isLoading}
        deleteOpen={deleteOpen}
        onCloseDelete={() => setDeleteOpen(false)}
        onConfirmDelete={() => effectiveId && deleteMutation.mutate(effectiveId)}
        deleteLoading={deleteMutation.isLoading}
        deleteTargetName={currentMeta?.name || ''}
      />
    </Container>
  );
}
