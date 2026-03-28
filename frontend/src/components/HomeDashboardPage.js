import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { Alert, CircularProgress, Container, Stack } from '@mui/material';
import apiService from '../services/apiService';
import rssService from '../services/rssService';
import { WIDGET_TYPES, emptyWidget, widgetTitle } from './homeDashboard/homeDashboardUtils';
import {
  MarkdownCardView,
  NavLinksView,
  RssHeadlinesBlock,
  WidgetCard,
  WidgetEditor,
} from './homeDashboard/HomeDashboardPanels';
import HomeDashboardChrome, { HomeDashboardDialogs } from './homeDashboard/HomeDashboardChrome';

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
      setDraft(JSON.parse(JSON.stringify(serverLayout)));
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
      onSuccess: (_, variables) => {
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
    const w = emptyWidget(type);
    if (!w) return;
    setDraft((prev) => {
      if (!prev) return prev;
      return { ...prev, widgets: [...prev.widgets, w] };
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
    }
    saveMutation.mutate({ id: effectiveId, body: draft });
  };

  useEffect(() => {
    if (!editMode) setSaveError(null);
  }, [editMode]);

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

  return (
    <Container maxWidth="md" sx={{ py: 3 }}>
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
      />
      {saveError ? (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setSaveError(null)}>
          {saveError}
        </Alert>
      ) : null}
      <Stack spacing={2}>
        {(layout?.widgets || []).map((w, i) => (
          <WidgetCard
            key={w.id}
            title={widgetTitle(w)}
            editMode={editMode}
            onDelete={() => removeWidget(i)}
            onUp={() => moveWidget(i, -1)}
            onDown={() => moveWidget(i, 1)}
            disableUp={i === 0}
            disableDown={i === (layout?.widgets?.length || 0) - 1}
          >
            {editMode ? (
              <WidgetEditor widget={w} feeds={feeds} onChange={(nw) => updateWidgetAt(i, nw)} />
            ) : w.type === 'nav_links' ? (
              <NavLinksView config={w.config} navigate={navigate} />
            ) : w.type === 'markdown_card' ? (
              <MarkdownCardView config={w.config} />
            ) : w.type === 'rss_headlines' ? (
              <RssHeadlinesBlock config={w.config} navigate={navigate} />
            ) : null}
          </WidgetCard>
        ))}
      </Stack>

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
