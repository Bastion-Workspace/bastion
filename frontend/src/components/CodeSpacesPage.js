import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControl,
  FormControlLabel,
  InputLabel,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  MenuItem,
  Radio,
  RadioGroup,
  Select,
  TextField,
  Typography,
  CircularProgress,
} from '@mui/material';
import apiService from '../services/apiService';

const CODE_WORKSPACE_CHAT_CACHE_KEY = 'code_workspace_ctx_cache';

function renderTree(nodes, depth = 0) {
  if (!Array.isArray(nodes) || nodes.length === 0) return null;
  return (
    <Box sx={{ pl: depth ? 2 : 0 }}>
      {nodes.map((n) => {
        const key = `${n.path || n.name || 'node'}-${depth}`;
        const isDir = !!n.is_dir;
        return (
          <Box key={key} sx={{ py: 0.25 }}>
            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
              {isDir ? `${n.name}/` : n.name}
            </Typography>
            {isDir ? renderTree(n.children, depth + 1) : null}
          </Box>
        );
      })}
    </Box>
  );
}

const CodeSpacesPage = () => {
  const [loading, setLoading] = useState(true);
  const [workspaces, setWorkspaces] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const selected = useMemo(
    () => workspaces.find((w) => w.id === selectedId) || null,
    [workspaces, selectedId]
  );

  const [detailLoading, setDetailLoading] = useState(false);
  const [detail, setDetail] = useState(null);
  const [rulesEdit, setRulesEdit] = useState('');
  const [rulesSaving, setRulesSaving] = useState(false);

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

  const loadDetail = useCallback(async (id) => {
    if (!id) return;
    setDetailLoading(true);
    try {
      const data = await apiService.get(`/api/code-workspaces/${id}`);
      setDetail(data);
      const rt = (data?.settings && data.settings.rules_text) || '';
      setRulesEdit(typeof rt === 'string' ? rt : '');
    } finally {
      setDetailLoading(false);
    }
  }, []);

  useEffect(() => {
    loadList();
  }, [loadList]);

  useEffect(() => {
    if (selectedId) loadDetail(selectedId);
  }, [selectedId, loadDetail]);

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
      setSelectedId(null);
      setDetail(null);
      await loadList();
    },
    [loadList]
  );

  const onRefreshTree = useCallback(
    async (id) => {
      if (!id) return;
      setDetailLoading(true);
      try {
        await apiService.post(`/api/code-workspaces/${id}/refresh-tree`, {});
        await loadDetail(id);
      } finally {
        setDetailLoading(false);
      }
    },
    [loadDetail]
  );

  const onSaveRules = useCallback(async () => {
    if (!selectedId) return;
    setRulesSaving(true);
    try {
      await apiService.put(`/api/code-workspaces/${selectedId}`, {
        settings: { rules_text: rulesEdit },
      });
      await loadDetail(selectedId);
    } finally {
      setRulesSaving(false);
    }
  }, [selectedId, rulesEdit, loadDetail]);

  const clearChatBinding = useCallback(() => {
    try {
      localStorage.removeItem(CODE_WORKSPACE_CHAT_CACHE_KEY);
    } catch (e) {
      /* ignore */
    }
  }, []);

  return (
    <Box sx={{ display: 'flex', height: '100%', gap: 2, p: 2 }}>
      <Card sx={{ width: 360, flexShrink: 0, display: 'flex', flexDirection: 'column' }}>
        <CardContent sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="h6">Code Spaces</Typography>
          <Button variant="contained" size="small" onClick={() => setCreateOpen(true)}>
            New
          </Button>
        </CardContent>
        <Divider />
        <Box sx={{ flex: 1, overflow: 'auto' }}>
          {loading ? (
            <Box sx={{ p: 2, display: 'flex', justifyContent: 'center' }}>
              <CircularProgress size={24} />
            </Box>
          ) : (
            <List dense disablePadding>
              {workspaces.map((w) => (
                <ListItem key={w.id} disablePadding>
                  <ListItemButton selected={w.id === selectedId} onClick={() => setSelectedId(w.id)}>
                    <ListItemText
                      primary={w.name}
                      secondary={w.workspace_path}
                      primaryTypographyProps={{ noWrap: true }}
                      secondaryTypographyProps={{ noWrap: true }}
                    />
                  </ListItemButton>
                </ListItem>
              ))}
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

      <Card sx={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
        <CardContent sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2 }}>
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="h6" noWrap>
              {selected?.name || 'Select a code space'}
            </Typography>
            <Typography variant="body2" color="text.secondary" noWrap>
              {selected?.workspace_path || ''}
            </Typography>
            {selected?.device_id ? (
              <Typography variant="caption" color="text.secondary" display="block" noWrap>
                Device: {selected.device_id}
              </Typography>
            ) : null}
          </Box>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
            <Button variant="text" size="small" onClick={clearChatBinding}>
              Clear chat binding
            </Button>
            <Button
              variant="outlined"
              size="small"
              disabled={!selectedId || detailLoading}
              onClick={() => onRefreshTree(selectedId)}
            >
              Refresh tree
            </Button>
            <Button
              variant="outlined"
              color="error"
              size="small"
              disabled={!selectedId || detailLoading}
              onClick={() => onDelete(selectedId)}
            >
              Delete
            </Button>
          </Box>
        </CardContent>
        <Divider />
        <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
          {detailLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', pt: 4 }}>
              <CircularProgress size={28} />
            </Box>
          ) : !detail ? (
            <Typography variant="body2" color="text.secondary">
              Select a code space to view details.
            </Typography>
          ) : (
            <>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Project rules (chat agents)
              </Typography>
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                Shown to custom agents when this code space is selected. Selection is stored locally for chat.
              </Typography>
              <TextField
                multiline
                minRows={4}
                fullWidth
                value={rulesEdit}
                onChange={(e) => setRulesEdit(e.target.value)}
                placeholder="e.g. Use TypeScript strict mode; prefer functional components; ..."
              />
              <Button
                sx={{ mt: 1 }}
                variant="outlined"
                size="small"
                disabled={rulesSaving}
                onClick={onSaveRules}
              >
                Save rules
              </Button>
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Cached file tree
              </Typography>
              {detail?.last_file_tree?.tree ? (
                renderTree(detail.last_file_tree.tree)
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No cached tree. Click Refresh tree.
                </Typography>
              )}
            </>
          )}
        </Box>
      </Card>

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
