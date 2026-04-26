import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Collapse,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControl,
  FormControlLabel,
  IconButton,
  InputLabel,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  Radio,
  RadioGroup,
  Select,
  Tab,
  Tabs,
  TextField,
  Typography,
  useTheme,
} from '@mui/material';
import {
  ChevronRight,
  ExpandMore,
  Folder,
  InsertDriveFile,
  Article,
  Refresh,
  Delete as DeleteIcon,
  Save,
  Close,
  LinkOff,
} from '@mui/icons-material';
import CodeMirror from '@uiw/react-codemirror';
import { keymap } from '@codemirror/view';
import { Prec } from '@codemirror/state';
import apiService from '../services/apiService';
import { buildCodeSpaceEditorExtensions } from './codeSpaces/codeSpaceEditorExtensions';

const CODE_WORKSPACE_CHAT_CACHE_KEY = 'code_workspace_ctx_cache';
const MAX_OPEN_TABS = 10;
const MAX_FILE_EDIT_BYTES = 2 * 1024 * 1024;

/** API may return last_file_tree as an object or a JSON string (depends on DB codec). */
function normalizedLastFileTree(raw) {
  if (raw == null) return null;
  if (typeof raw === 'string') {
    try {
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === 'object' ? parsed : null;
    } catch (e) {
      return null;
    }
  }
  if (typeof raw === 'object') return raw;
  return null;
}

function newTabId() {
  return `tab-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function tabKey(workspaceId, relPath) {
  return `${workspaceId}\0${relPath}`;
}

function FileTreeBranch({ nodes, workspaceId, depth, openDirs, toggleDir, onOpenFile }) {
  if (!Array.isArray(nodes) || nodes.length === 0) return null;
  return (
    <List dense disablePadding sx={{ pl: depth ? 0.5 : 0 }}>
      {nodes.map((n) => {
        const rel = n.path || n.name || '';
        const key = `${workspaceId}-${rel}-${depth}`;
        const isDir = !!n.is_dir;
        const isOpen = openDirs.has(rel);
        if (isDir) {
          return (
            <React.Fragment key={key}>
              <ListItem disablePadding secondaryAction={null}>
                <ListItemButton dense sx={{ py: 0.25, pl: 0.5 }} onClick={() => toggleDir(rel)}>
                  <ListItemIcon sx={{ minWidth: 28 }}>
                    {isOpen ? <ExpandMore fontSize="small" /> : <ChevronRight fontSize="small" />}
                  </ListItemIcon>
                  <ListItemIcon sx={{ minWidth: 32 }}>
                    <Folder fontSize="small" color="warning" />
                  </ListItemIcon>
                  <ListItemText primary={n.name} primaryTypographyProps={{ variant: 'body2', noWrap: true }} />
                </ListItemButton>
              </ListItem>
              <Collapse in={isOpen} timeout="auto" unmountOnExit>
                <FileTreeBranch
                  nodes={n.children}
                  workspaceId={workspaceId}
                  depth={depth + 1}
                  openDirs={openDirs}
                  toggleDir={toggleDir}
                  onOpenFile={onOpenFile}
                />
              </Collapse>
            </React.Fragment>
          );
        }
        return (
          <ListItem key={key} disablePadding>
            <ListItemButton
              dense
              sx={{ py: 0.25, pl: depth ? 3 : 1 }}
              onClick={() => onOpenFile(workspaceId, rel, n.name)}
            >
              <ListItemIcon sx={{ minWidth: 32 }}>
                <InsertDriveFile fontSize="small" color="action" />
              </ListItemIcon>
              <ListItemText primary={n.name} primaryTypographyProps={{ variant: 'body2', noWrap: true }} />
            </ListItemButton>
          </ListItem>
        );
      })}
    </List>
  );
}

const CodeSpacesPage = () => {
  const theme = useTheme();
  const darkMode = theme.palette.mode === 'dark';

  const [loading, setLoading] = useState(true);
  const [workspaces, setWorkspaces] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const selected = useMemo(
    () => workspaces.find((w) => w.id === selectedId) || null,
    [workspaces, selectedId]
  );

  const [detailsById, setDetailsById] = useState({});
  const [detailLoadingMap, setDetailLoadingMap] = useState({});

  const [expandedWorkspaceIds, setExpandedWorkspaceIds] = useState(() => new Set());
  const [treeOpenDirs, setTreeOpenDirs] = useState({});

  const [ctxMenu, setCtxMenu] = useState(null);
  const [rulesDialog, setRulesDialog] = useState(null);
  const [rulesEdit, setRulesEdit] = useState('');
  const [rulesSaving, setRulesSaving] = useState(false);

  const [openTabs, setOpenTabs] = useState([]);
  const openTabsRef = useRef(openTabs);
  openTabsRef.current = openTabs;
  const [activeTabId, setActiveTabId] = useState(null);

  const [createOpen, setCreateOpen] = useState(false);
  const [createName, setCreateName] = useState('');
  const [pathMode, setPathMode] = useState('existing');
  const [createPath, setCreatePath] = useState('');
  const [createParentPath, setCreateParentPath] = useState('');
  const [createFolderName, setCreateFolderName] = useState('');
  const [createDeviceId, setCreateDeviceId] = useState('');
  const [createRulesText, setCreateRulesText] = useState('');
  const [devicesLoading, setDevicesLoading] = useState(false);
  const [connectedDevices, setConnectedDevices] = useState([]);
  const [createSubmitting, setCreateSubmitting] = useState(false);

  const loadList = useCallback(async () => {
    setLoading(true);
    try {
      const data = await apiService.get('/api/code-workspaces');
      setWorkspaces(data?.workspaces || []);
      if (!selectedId && (data?.workspaces || []).length) {
        setSelectedId((data.workspaces[0] || {}).id);
      }
    } finally {
      setLoading(false);
    }
  }, [selectedId]);

  const loadDetailFor = useCallback(async (id) => {
    if (!id) return;
    setDetailLoadingMap((m) => ({ ...m, [id]: true }));
    try {
      const data = await apiService.get(`/api/code-workspaces/${id}`);
      setDetailsById((prev) => ({ ...prev, [id]: data }));
    } finally {
      setDetailLoadingMap((m) => ({ ...m, [id]: false }));
    }
  }, []);

  useEffect(() => {
    loadList();
  }, [loadList]);

  useEffect(() => {
    if (selectedId) loadDetailFor(selectedId);
  }, [selectedId, loadDetailFor]);

  useEffect(() => {
    if (!selectedId) return;
    try {
      localStorage.setItem(
        CODE_WORKSPACE_CHAT_CACHE_KEY,
        JSON.stringify({ code_workspace_id: selectedId })
      );
    } catch (e) {
      /* ignore */
    }
  }, [selectedId]);

  const loadDevicesForDialog = useCallback(async () => {
    setDevicesLoading(true);
    try {
      const data = await apiService.codeWorkspaces.connectedDevices();
      const list = data?.devices || [];
      setConnectedDevices(list);
      if (list.length === 1) {
        setCreateDeviceId(list[0].device_id || '');
      } else {
        setCreateDeviceId('');
      }
    } catch (e) {
      setConnectedDevices([]);
      setCreateDeviceId('');
    } finally {
      setDevicesLoading(false);
    }
  }, []);

  useEffect(() => {
    if (createOpen) {
      loadDevicesForDialog();
    }
  }, [createOpen, loadDevicesForDialog]);

  const onCreate = useCallback(async () => {
    const settings = {};
    if (createRulesText.trim()) settings.rules_text = createRulesText.trim();
    const payload = {
      name: createName.trim(),
      device_id: createDeviceId.trim() || undefined,
      settings,
    };
    if (pathMode === 'existing') {
      payload.workspace_path = createPath.trim();
    } else {
      payload.parent_path = createParentPath.trim();
      payload.folder_name = createFolderName.trim();
    }
    setCreateSubmitting(true);
    try {
      await apiService.post('/api/code-workspaces', payload);
      setCreateOpen(false);
      setCreateName('');
      setCreatePath('');
      setCreateParentPath('');
      setCreateFolderName('');
      setCreateDeviceId('');
      setCreateRulesText('');
      setPathMode('existing');
      await loadList();
    } finally {
      setCreateSubmitting(false);
    }
  }, [
    createName,
    createPath,
    createParentPath,
    createFolderName,
    createDeviceId,
    createRulesText,
    pathMode,
    loadList,
  ]);

  const createDisabled =
    !createName.trim() ||
    createSubmitting ||
    devicesLoading ||
    !connectedDevices.length ||
    (pathMode === 'existing' ? !createPath.trim() : !createParentPath.trim() || !createFolderName.trim()) ||
    (connectedDevices.length > 1 && !createDeviceId.trim());

  const onDelete = useCallback(
    async (id) => {
      if (!id) return;
      await apiService.delete(`/api/code-workspaces/${id}`);
      try {
        const raw = localStorage.getItem(CODE_WORKSPACE_CHAT_CACHE_KEY);
        const parsed = raw ? JSON.parse(raw) : null;
        if (parsed?.code_workspace_id === id) {
          localStorage.removeItem(CODE_WORKSPACE_CHAT_CACHE_KEY);
        }
      } catch (e) {
        /* ignore */
      }
      setOpenTabs((tabs) => {
        const next = tabs.filter((t) => t.workspaceId !== id);
        setActiveTabId((cur) => {
          if (!next.length) return null;
          if (cur && next.some((t) => t.id === cur)) return cur;
          return next[0].id;
        });
        return next;
      });
      setDetailsById((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      setExpandedWorkspaceIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      if (selectedId === id) {
        setSelectedId(null);
      }
      await loadList();
    },
    [loadList, selectedId]
  );

  const onRefreshTree = useCallback(
    async (id) => {
      if (!id) return;
      setDetailLoadingMap((m) => ({ ...m, [id]: true }));
      try {
        await apiService.post(`/api/code-workspaces/${id}/refresh-tree`, {});
        await loadDetailFor(id);
      } finally {
        setDetailLoadingMap((m) => ({ ...m, [id]: false }));
      }
    },
    [loadDetailFor]
  );

  const toggleWorkspaceExpand = useCallback(
    (id) => {
      setExpandedWorkspaceIds((prev) => {
        const next = new Set(prev);
        if (next.has(id)) {
          next.delete(id);
        } else {
          next.add(id);
          if (!detailsById[id]) {
            void loadDetailFor(id);
          }
        }
        return next;
      });
    },
    [detailsById, loadDetailFor]
  );

  const toggleTreeDir = useCallback((workspaceId, relPath) => {
    setTreeOpenDirs((prev) => {
      const key = workspaceId;
      const set = new Set(prev[key] || []);
      if (set.has(relPath)) set.delete(relPath);
      else set.add(relPath);
      return { ...prev, [key]: set };
    });
  }, []);

  const openContextMenu = useCallback((event, workspaceId) => {
    event.preventDefault();
    setCtxMenu({ mouseX: event.clientX, mouseY: event.clientY, workspaceId });
  }, []);

  const closeContextMenu = useCallback(() => setCtxMenu(null), []);

  const openRulesDialog = useCallback(
    async (workspaceId) => {
      closeContextMenu();
      try {
        const d = detailsById[workspaceId] || (await apiService.get(`/api/code-workspaces/${workspaceId}`));
        setDetailsById((prev) => ({ ...prev, [workspaceId]: d }));
        const rt = (d?.settings && d.settings.rules_text) || '';
        setRulesEdit(typeof rt === 'string' ? rt : '');
        setRulesDialog(workspaceId);
      } catch (e) {
        /* ignore */
      }
    },
    [closeContextMenu, detailsById]
  );

  const onSaveRules = useCallback(async () => {
    if (!rulesDialog) return;
    setRulesSaving(true);
    try {
      await apiService.put(`/api/code-workspaces/${rulesDialog}`, {
        settings: { rules_text: rulesEdit },
      });
      await loadDetailFor(rulesDialog);
      setRulesDialog(null);
    } finally {
      setRulesSaving(false);
    }
  }, [rulesDialog, rulesEdit, loadDetailFor]);

  const clearChatBinding = useCallback(() => {
    try {
      localStorage.removeItem(CODE_WORKSPACE_CHAT_CACHE_KEY);
    } catch (e) {
      /* ignore */
    }
  }, []);

  const openFile = useCallback(async (workspaceId, relPath, title) => {
    const tKey = tabKey(workspaceId, relPath);
    const tabsNow = openTabsRef.current;
    const existing = tabsNow.find((t) => tabKey(t.workspaceId, t.relPath) === tKey);
    if (existing) {
      setActiveTabId(existing.id);
      return;
    }
    if (tabsNow.length >= MAX_OPEN_TABS) {
      return;
    }
    const id = newTabId();
    const tab = {
      id,
      workspaceId,
      relPath,
      title: title || relPath.split('/').pop(),
      content: '',
      dirty: false,
      loading: true,
      loadError: null,
      tooLarge: false,
    };
    setOpenTabs((tabs) => [...tabs, tab]);
    setActiveTabId(id);
    try {
      const data = await apiService.codeWorkspaces.readFile(workspaceId, relPath);
      const size = data?.size_bytes ?? (data?.content || '').length;
      if (size > MAX_FILE_EDIT_BYTES) {
        setOpenTabs((tabs) =>
          tabs.map((t) =>
            t.id === id
              ? {
                  ...t,
                  loading: false,
                  tooLarge: true,
                  content: '',
                  loadError: `File is about ${Math.round(size / 1024)} KB; open in a local editor (limit ${Math.round(MAX_FILE_EDIT_BYTES / 1024)} KB).`,
                }
              : t
          )
        );
        return;
      }
      const content = data?.content ?? '';
      setOpenTabs((tabs) =>
        tabs.map((t) =>
          t.id === id ? { ...t, loading: false, content, dirty: false, loadError: null, tooLarge: false } : t
        )
      );
    } catch (e) {
      const detail = e?.response?.data?.detail;
      const msg =
        typeof detail === 'string' ? detail : Array.isArray(detail) ? JSON.stringify(detail) : e?.message;
      setOpenTabs((tabs) =>
        tabs.map((t) => (t.id === id ? { ...t, loading: false, loadError: msg || 'Failed to load file' } : t))
      );
    }
  }, []);

  const updateActiveContent = useCallback(
    (value) => {
      if (!activeTabId) return;
      setOpenTabs((tabs) =>
        tabs.map((t) => (t.id === activeTabId ? { ...t, content: value, dirty: true } : t))
      );
    },
    [activeTabId]
  );

  const saveActiveTab = useCallback(async () => {
    const tab = openTabs.find((t) => t.id === activeTabId);
    if (!tab || tab.loading || tab.loadError || tab.tooLarge || !tab.dirty) return;
    const tabId = tab.id;
    try {
      await apiService.codeWorkspaces.writeFile(tab.workspaceId, {
        path: tab.relPath,
        content: tab.content,
      });
      setOpenTabs((tabs) => tabs.map((t) => (t.id === tabId ? { ...t, dirty: false } : t)));
    } catch (e) {
      const detail = e?.response?.data?.detail;
      const msg = typeof detail === 'string' ? detail : e?.message;
      setOpenTabs((tabs) =>
        tabs.map((t) => (t.id === tabId ? { ...t, loadError: msg || 'Save failed' } : t))
      );
    }
  }, [activeTabId, openTabs]);

  const saveActiveTabRef = useRef(saveActiveTab);
  saveActiveTabRef.current = saveActiveTab;

  const closeTab = useCallback(
    (tabId) => {
      setOpenTabs((tabs) => {
        const idx = tabs.findIndex((t) => t.id === tabId);
        if (idx === -1) return tabs;
        const next = tabs.filter((t) => t.id !== tabId);
        setActiveTabId((cur) => {
          if (cur !== tabId) return cur;
          if (!next.length) return null;
          return next[Math.max(0, idx - 1)].id;
        });
        return next;
      });
    },
    []
  );

  const activeTab = useMemo(() => openTabs.find((t) => t.id === activeTabId) || null, [openTabs, activeTabId]);

  const cmExtensions = useMemo(() => {
    if (!activeTab || activeTab.tooLarge || activeTab.loadError) return [];
    const name = activeTab.title || activeTab.relPath || 'file.txt';
    return [
      ...buildCodeSpaceEditorExtensions(name, darkMode),
      Prec.highest(
        keymap.of([
          {
            key: 'Mod-s',
            run: () => {
              void saveActiveTabRef.current();
              return true;
            },
          },
        ])
      ),
    ];
  }, [activeTab, darkMode]);

  const selectedDetail = selectedId ? detailsById[selectedId] : null;

  return (
    <Box sx={{ display: 'flex', height: '100%', gap: 2, p: 2, minHeight: 0 }}>
      <Card sx={{ width: 380, flexShrink: 0, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <CardContent sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', py: 1.5 }}>
          <Typography variant="h6">Code Spaces</Typography>
          <Button variant="contained" size="small" onClick={() => setCreateOpen(true)}>
            New
          </Button>
        </CardContent>
        <Divider />
        <Box sx={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
          {loading ? (
            <Box sx={{ p: 2, display: 'flex', justifyContent: 'center' }}>
              <CircularProgress size={24} />
            </Box>
          ) : (
            <List dense disablePadding>
              {workspaces.map((w) => {
                const expanded = expandedWorkspaceIds.has(w.id);
                const detail = detailsById[w.id];
                const treeNorm = normalizedLastFileTree(detail?.last_file_tree);
                const rows = treeNorm?.tree;
                const rowLoading = !!detailLoadingMap[w.id];
                const openDirs = new Set(treeOpenDirs[w.id] || []);

                return (
                  <Box key={w.id}>
                    <ListItem
                      disablePadding
                      secondaryAction={
                        <IconButton size="small" edge="end" onClick={() => toggleWorkspaceExpand(w.id)} aria-label="expand">
                          {expanded ? <ExpandMore /> : <ChevronRight />}
                        </IconButton>
                      }
                      onContextMenu={(e) => openContextMenu(e, w.id)}
                    >
                      <ListItemButton
                        selected={w.id === selectedId}
                        onClick={() => setSelectedId(w.id)}
                        sx={{ pr: 5 }}
                      >
                        <ListItemText
                          primary={w.name}
                          secondary={w.workspace_path}
                          primaryTypographyProps={{ noWrap: true }}
                          secondaryTypographyProps={{ noWrap: true }}
                        />
                      </ListItemButton>
                    </ListItem>
                    <Collapse in={expanded} timeout="auto" unmountOnExit>
                      <Box sx={{ pl: 1, pr: 1, pb: 1 }}>
                        {rowLoading && !rows ? (
                          <Box sx={{ display: 'flex', justifyContent: 'center', py: 1 }}>
                            <CircularProgress size={20} />
                          </Box>
                        ) : rows && rows.length ? (
                          <FileTreeBranch
                            nodes={rows}
                            workspaceId={w.id}
                            depth={0}
                            openDirs={openDirs}
                            toggleDir={(rel) => toggleTreeDir(w.id, rel)}
                            onOpenFile={openFile}
                          />
                        ) : (
                          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', py: 0.5 }}>
                            No tree cached. Right-click → Refresh tree.
                          </Typography>
                        )}
                      </Box>
                    </Collapse>
                  </Box>
                );
              })}
              {!workspaces.length ? (
                <Box sx={{ p: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    No code spaces yet.
                  </Typography>
                </Box>
              ) : null}
            </List>
          )}
        </Box>
      </Card>

      <Card sx={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <CardContent sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2, py: 1.5 }}>
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="h6" noWrap>
              {selected?.name || 'Select a code space'}
            </Typography>
            <Typography variant="body2" color="text.secondary" noWrap>
              {selected?.workspace_path || ''}
            </Typography>
            {selectedDetail?.device_id ? (
              <Typography variant="caption" color="text.secondary" display="block" noWrap>
                Device: {selectedDetail.device_id}
              </Typography>
            ) : null}
          </Box>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
            <Button variant="text" size="small" startIcon={<LinkOff />} onClick={clearChatBinding}>
              Clear chat binding
            </Button>
            <Button
              variant="contained"
              size="small"
              startIcon={<Save />}
              disabled={!activeTab || activeTab.loading || !!activeTab.loadError || activeTab.tooLarge || !activeTab.dirty}
              onClick={() => void saveActiveTab()}
            >
              Save
            </Button>
          </Box>
        </CardContent>
        <Divider />
        {openTabs.length > 0 ? (
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs
              value={
                activeTabId && openTabs.some((t) => t.id === activeTabId)
                  ? activeTabId
                  : openTabs[0]?.id
              }
              onChange={(_, v) => setActiveTabId(v)}
              variant="scrollable"
              scrollButtons="auto"
            >
              {openTabs.map((t) => (
                <Tab
                  key={t.id}
                  value={t.id}
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, maxWidth: 200 }}>
                      <Typography variant="body2" noWrap component="span">
                        {t.title}
                        {t.dirty ? ' •' : ''}
                      </Typography>
                      <IconButton
                        component="span"
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          closeTab(t.id);
                        }}
                      >
                        <Close fontSize="inherit" />
                      </IconButton>
                    </Box>
                  }
                />
              ))}
            </Tabs>
          </Box>
        ) : null}
        <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', p: 0 }}>
          {!activeTab ? (
            <Box sx={{ p: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Expand a code space and open a file to edit. Use Save or Ctrl/Cmd+S to write changes to your machine via
                the local proxy.
              </Typography>
            </Box>
          ) : (
            <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
              {activeTab.loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flex: 1 }}>
                  <CircularProgress size={28} />
                </Box>
              ) : activeTab.loadError ? (
                <Alert severity="error" sx={{ m: 2 }}>
                  {activeTab.loadError}
                </Alert>
              ) : activeTab.tooLarge ? (
                <Alert severity="warning" sx={{ m: 2 }}>
                  {activeTab.loadError || 'File too large for this editor.'}
                </Alert>
              ) : (
                <Box sx={{ flex: 1, minHeight: 320, display: 'flex', flexDirection: 'column' }}>
                  <CodeMirror
                    value={activeTab.content}
                    height="calc(100vh - 240px)"
                    style={{ flex: 1, minHeight: 280 }}
                    extensions={cmExtensions}
                    onChange={updateActiveContent}
                    basicSetup={false}
                  />
                </Box>
              )}
            </Box>
          )}
        </Box>
      </Card>

      <Menu
        open={ctxMenu !== null}
        onClose={closeContextMenu}
        anchorReference="anchorPosition"
        anchorPosition={ctxMenu ? { top: ctxMenu.mouseY, left: ctxMenu.mouseX } : undefined}
      >
        <MenuItem
          onClick={() => {
            const id = ctxMenu?.workspaceId;
            closeContextMenu();
            if (id) void openRulesDialog(id);
          }}
        >
          <ListItemIcon>
            <Article fontSize="small" />
          </ListItemIcon>
          <ListItemText>Edit project rules…</ListItemText>
        </MenuItem>
        <MenuItem
          onClick={() => {
            const id = ctxMenu?.workspaceId;
            closeContextMenu();
            if (id) void onRefreshTree(id);
          }}
        >
          <ListItemIcon>
            <Refresh fontSize="small" />
          </ListItemIcon>
          <ListItemText>Refresh tree</ListItemText>
        </MenuItem>
        <MenuItem
          onClick={() => {
            const id = ctxMenu?.workspaceId;
            closeContextMenu();
            if (id && window.confirm('Delete this code space?')) void onDelete(id);
          }}
        >
          <ListItemIcon>
            <DeleteIcon fontSize="small" color="error" />
          </ListItemIcon>
          <ListItemText>Delete</ListItemText>
        </MenuItem>
      </Menu>

      <Dialog open={!!rulesDialog} onClose={() => setRulesDialog(null)} maxWidth="sm" fullWidth>
        <DialogTitle>Project rules (chat agents)</DialogTitle>
        <DialogContent>
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
            Shown to agents when this code space is selected.
          </Typography>
          <TextField
            multiline
            minRows={6}
            fullWidth
            value={rulesEdit}
            onChange={(e) => setRulesEdit(e.target.value)}
            placeholder="e.g. Use TypeScript strict mode; prefer functional components; ..."
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRulesDialog(null)}>Cancel</Button>
          <Button variant="contained" disabled={rulesSaving} onClick={() => void onSaveRules()}>
            Save
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={createOpen} onClose={() => setCreateOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>New Code Space</DialogTitle>
        <DialogContent sx={{ pt: 1 }}>
          {devicesLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
              <CircularProgress size={28} />
            </Box>
          ) : !connectedDevices.length ? (
            <Typography color="error" variant="body2" sx={{ mt: 1 }}>
              No local proxy connected. Start the Bastion local proxy on your machine and try again.
            </Typography>
          ) : null}
          <TextField
            label="Name"
            fullWidth
            value={createName}
            onChange={(e) => setCreateName(e.target.value)}
            sx={{ mt: 1 }}
          />
          <FormControl fullWidth sx={{ mt: 2 }} disabled={connectedDevices.length < 2}>
            <InputLabel id="cw-device-label">Device</InputLabel>
            <Select
              labelId="cw-device-label"
              label="Device"
              value={createDeviceId}
              onChange={(e) => setCreateDeviceId(e.target.value)}
            >
              {connectedDevices.length > 1 ? (
                <MenuItem value="">
                  <em>Select device</em>
                </MenuItem>
              ) : null}
              {connectedDevices.map((d) => (
                <MenuItem key={d.device_id} value={d.device_id}>
                  {d.device_id}
                  {Array.isArray(d.capabilities) && d.capabilities.length
                    ? ` (${d.capabilities.length} caps)`
                    : ''}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Typography variant="subtitle2" sx={{ mt: 2, mb: 0.5 }}>
            Workspace path on device
          </Typography>
          <RadioGroup row value={pathMode} onChange={(e) => setPathMode(e.target.value)}>
            <FormControlLabel value="existing" control={<Radio size="small" />} label="Existing path" />
            <FormControlLabel value="new" control={<Radio size="small" />} label="Create folder" />
          </RadioGroup>
          {pathMode === 'existing' ? (
            <TextField
              label="Workspace path"
              fullWidth
              value={createPath}
              onChange={(e) => setCreatePath(e.target.value)}
              sx={{ mt: 1 }}
              placeholder="/home/user/projects/my-repo"
            />
          ) : (
            <>
              <TextField
                label="Parent directory"
                fullWidth
                value={createParentPath}
                onChange={(e) => setCreateParentPath(e.target.value)}
                sx={{ mt: 1 }}
                placeholder="/home/user/projects"
              />
              <TextField
                label="New folder name"
                fullWidth
                value={createFolderName}
                onChange={(e) => setCreateFolderName(e.target.value)}
                sx={{ mt: 2 }}
                placeholder="my-repo"
                helperText="Single segment only (no slashes)."
              />
            </>
          )}
          <TextField
            label="Project rules (optional)"
            fullWidth
            multiline
            minRows={3}
            value={createRulesText}
            onChange={(e) => setCreateRulesText(e.target.value)}
            sx={{ mt: 2 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateOpen(false)}>Cancel</Button>
          <Button variant="contained" disabled={createDisabled} onClick={onCreate}>
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default CodeSpacesPage;
