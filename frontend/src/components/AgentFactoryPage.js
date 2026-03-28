/**
 * Agent Factory page: three-panel layout (agent list sidebar, section-card editor, chat sidebar).
 * Main content area supports browser-style tabs; each entity opened from the sidebar becomes a tab.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  FormControlLabel,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  Tabs,
  Tab,
  IconButton,
} from '@mui/material';
import { SmartToy, PlayArrow, Storage, Build, Close, Group } from '@mui/icons-material';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import apiService from '../services/apiService';
import { getSelectableChatModels } from '../utils/chatSelectableModels';
import AgentListSidebar from './agent-factory/AgentListSidebar';
import AgentEditor from './agent-factory/AgentEditor';
import PlaybookEditor from './agent-factory/PlaybookEditor';
import DataSourceEditor from './agent-factory/DataSourceEditor';
import SkillEditor from './agent-factory/SkillEditor';
import LineEditor from './agent-factory/LineEditor';

const AF_TABS_STORAGE_KEY = 'af-tabs';
const AF_LAST_PATH_KEY = 'af-last-path';

function parseAgentFactorySection(pathname) {
  if (pathname.includes('/line/')) return 'line';
  if (pathname.includes('/playbook/')) return 'playbook';
  if (pathname.includes('/datasource/')) return 'datasource';
  if (pathname.includes('/skill/')) return 'skill';
  if (pathname.includes('/agent/')) return 'agent';
  return null;
}

function getTabUrl(type, entityId) {
  if (type === 'skill') return `/agent-factory/skill/${entityId}`;
  if (type === 'line') return `/agent-factory/line/${entityId}`;
  return `/agent-factory/${type}/${entityId}`;
}

export default function AgentFactoryPage() {
  const { id: selectedId, skillId: selectedSkillId } = useParams();
  const selectedIdResolved = selectedSkillId ?? selectedId;
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const section = parseAgentFactorySection(pathname);
  const queryClient = useQueryClient();
  const [createOpen, setCreateOpen] = useState(false);
  const [createForm, setCreateForm] = useState({
    name: '',
    handle: '',
    description: '',
    model_preference: '',
    default_playbook_id: '',
    chat_visible: true,
  });

  const [tabs, setTabs] = useState(() => {
    try {
      const raw = localStorage.getItem(AF_TABS_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  });

  const createProfileMutation = useMutation(
    (body) => apiService.agentFactory.createProfile(body),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('agentFactoryProfiles');
        setCreateOpen(false);
        setCreateForm({ name: '', handle: '', description: '', model_preference: '', default_playbook_id: '', chat_visible: true });
        const id = data?.id;
        if (id) navigate(`/agent-factory/agent/${id}`);
      },
    }
  );

  const { data: playbooks = [] } = useQuery(
    'agentFactoryPlaybooks',
    () => apiService.agentFactory.listPlaybooks(),
    { retry: false }
  );

  const { data: profiles = [] } = useQuery(
    'agentFactoryProfiles',
    () => apiService.agentFactory.listProfiles(),
    { retry: false }
  );
  const { data: connectors = [] } = useQuery(
    'agentFactoryConnectors',
    () => apiService.agentFactory.listConnectors(),
    { retry: false }
  );
  const { data: skillsList = [] } = useQuery(
    'agentFactorySkills',
    () => apiService.agentFactory.listSkills({ include_builtin: true }),
    { retry: false }
  );
  const { data: linesList = [] } = useQuery(
    'agentFactoryLines',
    () => apiService.agentFactory.listLines(),
    { retry: false }
  );

  const getEntityTitle = useCallback(
    (type, entityId) => {
      if (!entityId) return '';
      switch (type) {
        case 'agent':
          return profiles.find((p) => p.id === entityId)?.name || profiles.find((p) => p.id === entityId)?.handle || entityId;
        case 'playbook':
          return playbooks.find((p) => p.id === entityId)?.name || entityId;
        case 'datasource':
          return connectors.find((c) => c.id === entityId)?.name || entityId;
        case 'skill':
          return skillsList.find((s) => s.id === entityId)?.name || skillsList.find((s) => s.slug === entityId)?.name || entityId;
        case 'line':
          return linesList.find((l) => l.id === entityId)?.name || entityId;
        default:
          return entityId;
      }
    },
    [profiles, playbooks, connectors, skillsList, linesList]
  );

  const { data: enabledData } = useQuery(
    'enabledModels',
    () => apiService.getEnabledModels(),
    { staleTime: 300000 }
  );
  const { data: availableData } = useQuery(
    'availableModels',
    () => apiService.getAvailableModels(),
    { staleTime: 300000 }
  );

  const chatModels = getSelectableChatModels(enabledData);
  const getModelLabel = (id) => availableData?.models?.find((m) => m.id === id)?.name || id;

  const activeTabKey = section && selectedIdResolved ? `${section}:${selectedIdResolved}` : null;

  // Persist last-visited entity URL so "Agent Factory" nav can reopen it
  useEffect(() => {
    if (pathname.startsWith('/agent-factory') && pathname !== '/agent-factory' && (section && selectedIdResolved)) {
      try {
        localStorage.setItem(AF_LAST_PATH_KEY, pathname);
      } catch (_) {}
    }
  }, [pathname, section, selectedIdResolved]);

  // When landing on /agent-factory exactly, go to last tab if any; otherwise stay on blank
  const hasRedirectedRef = React.useRef(false);
  useEffect(() => {
    if (pathname !== '/agent-factory') {
      hasRedirectedRef.current = false;
      return;
    }
    if (hasRedirectedRef.current) return;
    try {
      const last = localStorage.getItem(AF_LAST_PATH_KEY);
      if (last && last !== '/agent-factory') {
        hasRedirectedRef.current = true;
        navigate(last, { replace: true });
      }
    } catch (_) {}
  }, [pathname, navigate]);

  useEffect(() => {
    if (!section || !selectedIdResolved) return;
    const key = `${section}:${selectedIdResolved}`;
    setTabs((prev) =>
      prev.some((t) => t.key === key) ? prev : [...prev, { key, type: section, entityId: selectedIdResolved }]
    );
  }, [section, selectedIdResolved]);

  useEffect(() => {
    try {
      localStorage.setItem(AF_TABS_STORAGE_KEY, JSON.stringify(tabs));
    } catch (_) {}
  }, [tabs]);

  const handleTabClick = (tab) => {
    const url = getTabUrl(tab.type, tab.entityId);
    navigate(url);
  };

  const handleCloseTab = (e, tabKey) => {
    e.stopPropagation();
    const idx = tabs.findIndex((t) => t.key === tabKey);
    if (idx === -1) return;
    const wasActive = tabKey === activeTabKey;
    const nextTabs = tabs.filter((t) => t.key !== tabKey);
    setTabs(nextTabs);
    if (wasActive && nextTabs.length > 0) {
      const nextTab = nextTabs[Math.min(idx, nextTabs.length - 1)];
      navigate(getTabUrl(nextTab.type, nextTab.entityId));
    } else if (wasActive) {
      try {
        localStorage.removeItem(AF_LAST_PATH_KEY);
      } catch (_) {}
      navigate('/agent-factory');
    }
  };

  const handleCloseTabForEntity = useCallback(
    (type, entityId) => {
      const tabKey = `${type}:${entityId}`;
      const idx = tabs.findIndex((t) => t.key === tabKey);
      if (idx === -1) return;
      const wasActive = tabKey === activeTabKey;
      const nextTabs = tabs.filter((t) => t.key !== tabKey);
      setTabs(nextTabs);
      if (wasActive && nextTabs.length > 0) {
        const nextTab = nextTabs[Math.min(idx, nextTabs.length - 1)];
        navigate(getTabUrl(nextTab.type, nextTab.entityId));
      } else if (wasActive) {
        try {
          localStorage.removeItem(AF_LAST_PATH_KEY);
        } catch (_) {}
        navigate('/agent-factory');
      }
    },
    [tabs, activeTabKey, navigate]
  );

  const handleOpenCreate = () => setCreateOpen(true);

  const handleCreateSave = () => {
    const handleVal = (createForm.handle || '').trim() || null;
    createProfileMutation.mutate({
      name: createForm.name,
      handle: handleVal,
      description: createForm.description || null,
      model_preference: createForm.model_preference || null,
      default_playbook_id: createForm.default_playbook_id || null,
      persona_mode: 'default',
      persona_id: null,
      chat_visible: createForm.chat_visible ?? true,
    });
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flex: 1,
        minHeight: 0,
        overflow: 'hidden',
      }}
    >
      <AgentListSidebar onOpenCreate={handleOpenCreate} onCloseEntityTab={handleCloseTabForEntity} />

      <Box
        sx={{
          flex: 1,
          minWidth: 0,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <Box
          sx={{
            flexShrink: 0,
            boxSizing: 'border-box',
            height: 44,
            minHeight: 44,
            px: 2,
            display: 'flex',
            alignItems: 'center',
            backgroundColor: 'background.paper',
            borderBottom: '1px solid',
            borderColor: 'divider',
          }}
        >
          {tabs.length > 0 ? (
            <>
              <Tabs
                value={activeTabKey ?? false}
                variant="scrollable"
                scrollButtons="auto"
                allowScrollButtonsMobile
                textColor="inherit"
                sx={{
                  flex: 1,
                  minHeight: 44,
                  '& .MuiTabs-scroller': { overflow: 'hidden !important' },
                  '& .MuiTabs-indicator': { display: 'none' },
                  '& .MuiTabs-flexContainer': { gap: 0, alignItems: 'stretch', minHeight: 44 },
                  '& .MuiTab-root': {
                    minHeight: 44,
                    maxHeight: 44,
                    boxSizing: 'border-box',
                    padding: '0 12px',
                    minWidth: 120,
                    maxWidth: 220,
                    textTransform: 'none',
                    borderRight: 1,
                    borderColor: 'divider',
                    overflow: 'hidden',
                    alignItems: 'center',
                    color: 'text.primary',
                    opacity: 1,
                    '&.Mui-selected': {
                      color: 'text.primary',
                      backgroundColor: 'background.default',
                      boxShadow: (t) => `inset 0 -2px 0 ${t.palette.primary.main}`,
                    },
                  },
                  '& .MuiTab-wrapper': {
                    width: '100%',
                    maxWidth: '100%',
                    minWidth: 0,
                    overflow: 'hidden',
                  },
                }}
              >
                {tabs.map((tab) => {
                  const TabIcon =
                    tab.type === 'agent'
                      ? SmartToy
                      : tab.type === 'playbook'
                        ? PlayArrow
                        : tab.type === 'datasource'
                          ? Storage
                          : tab.type === 'line'
                            ? Group
                            : Build;
                  const label = getEntityTitle(tab.type, tab.entityId) || tab.entityId;
                  return (
                    <Tab
                      key={tab.key}
                      value={tab.key}
                      label={
                        <Box
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 0.75,
                            minWidth: 0,
                            maxWidth: '100%',
                            overflow: 'hidden',
                          }}
                        >
                          <TabIcon sx={{ fontSize: 16, flexShrink: 0, color: 'primary.main' }} />
                          <Box
                            component="span"
                            sx={{
                              flex: 1,
                              minWidth: 0,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                              fontSize: 13,
                              fontWeight: 500,
                              lineHeight: 1.2,
                              color: 'text.primary',
                            }}
                          >
                            {label}
                          </Box>
                          <IconButton
                            size="small"
                            onClick={(e) => handleCloseTab(e, tab.key)}
                            sx={{
                              p: 0.25,
                              flexShrink: 0,
                              ml: -0.25,
                              color: 'text.secondary',
                              '&:hover': { color: 'text.primary', bgcolor: 'action.hover' },
                            }}
                            aria-label="Close tab"
                          >
                            <Close fontSize="small" />
                          </IconButton>
                        </Box>
                      }
                      onClick={() => handleTabClick(tab)}
                    />
                  );
                })}
              </Tabs>
            </>
          ) : (
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'row',
                alignItems: 'center',
                gap: 1.5,
                minWidth: 0,
                width: '100%',
                overflow: 'hidden',
              }}
            >
              <Typography variant="subtitle1" component="div" sx={{ fontWeight: 600, flexShrink: 0 }}>
                Agent Factory
              </Typography>
              <Typography variant="body2" color="text.secondary" noWrap sx={{ minWidth: 0 }}>
                Build and configure custom agents
              </Typography>
            </Box>
          )}
        </Box>
        <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {section === 'playbook' && selectedIdResolved && <PlaybookEditor playbookId={selectedIdResolved} onCloseEntityTab={handleCloseTabForEntity} />}
          {section === 'datasource' && selectedIdResolved && <DataSourceEditor connectorId={selectedIdResolved} onCloseEntityTab={handleCloseTabForEntity} />}
          {section === 'skill' && selectedSkillId && <SkillEditor skillId={selectedSkillId} onCloseEntityTab={handleCloseTabForEntity} />}
          {section === 'line' && selectedIdResolved && <LineEditor lineId={selectedIdResolved} onCloseEntityTab={handleCloseTabForEntity} />}
          {((section === 'agent' || (selectedIdResolved && !section)) && selectedIdResolved) && <AgentEditor profileId={selectedIdResolved} onCloseEntityTab={handleCloseTabForEntity} />}
        </Box>
      </Box>

      <Dialog open={createOpen} onClose={() => setCreateOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create agent</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Name"
            fullWidth
            value={createForm.name}
            onChange={(e) => setCreateForm((f) => ({ ...f, name: e.target.value }))}
          />
          <TextField
            margin="dense"
            label="Handle (optional)"
            fullWidth
            value={createForm.handle}
            onChange={(e) => setCreateForm((f) => ({ ...f, handle: e.target.value }))}
            placeholder="Leave blank for schedule/Run-only — not @mentionable"
          />
          <FormControlLabel
            control={
              <Switch
                checked={createForm.chat_visible !== false}
                onChange={(e) => setCreateForm((f) => ({ ...f, chat_visible: e.target.checked }))}
                disabled={!(createForm.handle || '').trim()}
                color="primary"
              />
            }
            label="Show in chat @ menu"
            labelPlacement="start"
            sx={{ mt: 1, mb: 0, alignSelf: 'flex-start' }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ mt: -0.5, display: 'block' }}>
            When off, this agent is still addressable by other agents in lines. Requires a handle.
          </Typography>
          <TextField
            margin="dense"
            label="Description"
            fullWidth
            multiline
            value={createForm.description}
            onChange={(e) => setCreateForm((f) => ({ ...f, description: e.target.value }))}
          />
          <FormControl fullWidth margin="dense">
            <InputLabel>Model preference</InputLabel>
            <Select
              value={createForm.model_preference}
              label="Model preference"
              onChange={(e) => setCreateForm((f) => ({ ...f, model_preference: e.target.value }))}
            >
              <MenuItem value="">— Default</MenuItem>
              {chatModels.map((modelId) => (
                <MenuItem key={modelId} value={modelId}>
                  {getModelLabel(modelId)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl fullWidth margin="dense">
            <InputLabel>Playbook</InputLabel>
            <Select
              value={createForm.default_playbook_id}
              label="Playbook"
              onChange={(e) => setCreateForm((f) => ({ ...f, default_playbook_id: e.target.value }))}
            >
              <MenuItem value="">—</MenuItem>
              {playbooks.map((pb) => (
                <MenuItem key={pb.id} value={pb.id}>{pb.name}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateOpen(false)}>Cancel</Button>
          <Button
            onClick={handleCreateSave}
            variant="contained"
            disabled={!createForm.name || createProfileMutation.isLoading}
          >
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
