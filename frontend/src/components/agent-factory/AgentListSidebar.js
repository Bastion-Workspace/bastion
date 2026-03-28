/**
 * Agent Factory left panel: collapsible sections (Agents, Playbooks, Skills, Data Connections).
 * Each section has user-defined categories (folders), drag-and-drop, and persisted expand/collapse.
 */

import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Typography,
  TextField,
  InputAdornment,
  List,
  ListItemButton,
  ListItemText,
  Button,
  CircularProgress,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Collapse,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Menu,
  Chip,
  Badge,
} from '@mui/material';
import {
  Search as SearchIcon,
  Pause,
  PlayArrow,
  Delete,
  ExpandMore,
  ExpandLess,
  Lock,
  ContentCopy,
  MoreVert,
  SmartToy,
  Storage,
  Build,
  Group,
  Close,
} from '@mui/icons-material';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';
import QuickTeamWizard from './QuickTeamWizard';
import TeamEditor from './TeamEditor';

const SIDEBAR_WIDTH = 280;
const AF_SIDEBAR_STATE_KEY = 'agentFactorySidebarState';
const UNCATEGORIZED = 'Uncategorized';

const SECTION_KEYS = { agents: 'agents', playbooks: 'playbooks', skills: 'skills', dataSources: 'connectors' };

function AfSectionHeaderTitle({ Icon, children }) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0 }}>
      <Icon sx={{ fontSize: 18, color: 'primary.main', flexShrink: 0 }} aria-hidden />
      <Typography variant="subtitle2" fontWeight={600} component="span">
        {children}
      </Typography>
    </Box>
  );
}

function parseAgentFactoryPath(pathname) {
  const lineMatch = pathname.match(/\/agent-factory\/line\/([^/]+)/);
  if (lineMatch) return { section: 'line', id: lineMatch[1] };
  const match = pathname.match(/\/agent-factory\/(agent|playbook|datasource|skill)\/([^/]+)/);
  if (match) return { section: match[1], id: match[2] };
  const legacyMatch = pathname.match(/^\/agent-factory\/([^/]+)$/);
  if (legacyMatch) return { section: 'agent', id: legacyMatch[1] };
  return { section: null, id: null };
}

function StatusDot({ status }) {
  const color =
    status === 'active' ? 'success.main' :
    status === 'paused' ? 'warning.main' :
    status === 'error' ? 'error.main' : 'grey.400';
  return (
    <Box
      sx={{
        width: 8,
        height: 8,
        borderRadius: '50%',
        bgcolor: color,
        flexShrink: 0,
        mt: 0.6,
      }}
      aria-label={status}
    />
  );
}

const AF_TOOLBAR_ICON_BTN_SX = {
  p: '2px',
  color: 'text.secondary',
  borderRadius: 1,
  '&:hover': { color: 'text.primary', bgcolor: 'action.hover' },
};

/**
 * Compact toolbar: search toggle → optional field → menu trigger.
 * Collapsed: two small icon buttons side by side. Expanded: short text field between them.
 */
function AfSectionSearchRow({
  searchOpen,
  onSearchOpenChange,
  value,
  onValueChange,
  placeholder,
  children,
}) {
  const inputRef = useRef(null);
  const hasQuery = Boolean((value || '').trim());

  useEffect(() => {
    if (searchOpen && inputRef.current) {
      inputRef.current.focus({ preventScroll: true });
    }
  }, [searchOpen]);

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 0.25,
        mb: 0.5,
        minWidth: 0,
        width: '100%',
      }}
    >
      <Tooltip title={searchOpen ? 'Hide search' : hasQuery ? 'Search (filter on)' : 'Search'}>
        <Badge color="primary" variant="dot" invisible={!hasQuery || searchOpen} sx={{ '& .MuiBadge-badge': { minWidth: 6, height: 6 } }}>
          <IconButton
            size="small"
            onClick={() => onSearchOpenChange(!searchOpen)}
            aria-label="Toggle search"
            aria-expanded={searchOpen}
            sx={{ ...AF_TOOLBAR_ICON_BTN_SX, flexShrink: 0 }}
          >
            <SearchIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Badge>
      </Tooltip>
      {searchOpen && (
        <TextField
          inputRef={inputRef}
          size="small"
          placeholder={placeholder}
          value={value}
          onChange={(e) => onValueChange(e.target.value)}
          sx={{
            flex: 1,
            minWidth: 0,
            '& .MuiOutlinedInput-root': {
              height: 28,
              fontSize: '0.75rem',
              pl: 1,
            },
            '& .MuiOutlinedInput-notchedOutline': { borderColor: 'divider' },
          }}
          InputProps={{
            endAdornment: hasQuery ? (
              <InputAdornment position="end" sx={{ mr: 0.25 }}>
                <IconButton size="small" aria-label="Clear search" edge="end" onClick={() => onValueChange('')} sx={AF_TOOLBAR_ICON_BTN_SX}>
                  <Close sx={{ fontSize: 16 }} />
                </IconButton>
              </InputAdornment>
            ) : undefined,
          }}
        />
      )}
      <Box sx={{ flexShrink: 0, display: 'flex', alignItems: 'center' }}>{children}</Box>
    </Box>
  );
}

function AfSectionMenuTrigger({ setMenuAnchor, tooltipTitle = 'List actions' }) {
  return (
    <Tooltip title={tooltipTitle}>
      <IconButton
        size="small"
        onClick={(e) => setMenuAnchor(e.currentTarget)}
        aria-label={tooltipTitle}
        aria-haspopup="true"
        sx={AF_TOOLBAR_ICON_BTN_SX}
      >
        <MoreVert sx={{ fontSize: 18 }} />
      </IconButton>
    </Tooltip>
  );
}

export default function AgentListSidebar({ onOpenCreate, onCloseEntityTab }) {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const { section: selectedSection, id: selectedId } = parseAgentFactoryPath(pathname);
  const [agentSearch, setAgentSearch] = useState('');
  const [playbookSearch, setPlaybookSearch] = useState('');
  const [connectorSearch, setConnectorSearch] = useState('');
  const [skillSearch, setSkillSearch] = useState('');
  const [lineSearch, setLineSearch] = useState('');
  const [agentSearchOpen, setAgentSearchOpen] = useState(false);
  const [playbookSearchOpen, setPlaybookSearchOpen] = useState(false);
  const [skillSearchOpen, setSkillSearchOpen] = useState(false);
  const [connectorSearchOpen, setConnectorSearchOpen] = useState(false);
  const [lineSearchOpen, setLineSearchOpen] = useState(false);
  const [quickLineWizardOpen, setQuickLineWizardOpen] = useState(false);
  const [createLineOpen, setCreateLineOpen] = useState(false);
  const [deleteLineConfirm, setDeleteLineConfirm] = useState(null);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [deletePlaybookConfirm, setDeletePlaybookConfirm] = useState(null);
  const [deleteConnectorConfirm, setDeleteConnectorConfirm] = useState(null);
  const [createPlaybookOpen, setCreatePlaybookOpen] = useState(false);
  const [newPlaybookName, setNewPlaybookName] = useState('');
  const [importFileInputKey, setImportFileInputKey] = useState(0);
  const [importPlaybooksFileInputKey, setImportPlaybooksFileInputKey] = useState(0);
  const [importSkillsFileInputKey, setImportSkillsFileInputKey] = useState(0);
  const [importConnectorsFileInputKey, setImportConnectorsFileInputKey] = useState(0);
  const [templateDialogOpen, setTemplateDialogOpen] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [addCategoryOpen, setAddCategoryOpen] = useState(false);
  const [addCategorySection, setAddCategorySection] = useState('agents');
  const [newCategoryName, setNewCategoryName] = useState('');
  const [agentsMenuAnchor, setAgentsMenuAnchor] = useState(null);
  const [playbooksMenuAnchor, setPlaybooksMenuAnchor] = useState(null);
  const [skillsMenuAnchor, setSkillsMenuAnchor] = useState(null);
  const [connectorsMenuAnchor, setConnectorsMenuAnchor] = useState(null);
  const [linesMenuAnchor, setLinesMenuAnchor] = useState(null);
  const queryClient = useQueryClient();

  const persistedState = useMemo(() => {
    try {
      const raw = localStorage.getItem(AF_SIDEBAR_STATE_KEY);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }, []);

  const [agentsOpen, setAgentsOpenState] = useState(() => persistedState?.sections?.agents ?? true);
  const [playbooksOpen, setPlaybooksOpenState] = useState(() => persistedState?.sections?.playbooks ?? true);
  const [skillsOpen, setSkillsOpenState] = useState(() => persistedState?.sections?.skills ?? true);
  const [dataSourcesOpen, setDataSourcesOpenState] = useState(() => persistedState?.sections?.dataSources ?? true);
  const [linesOpen, setLinesOpenState] = useState(() => persistedState?.sections?.lines ?? true);
  const [categoryExpanded, setCategoryExpanded] = useState(() => persistedState?.categories ?? { agents: {}, playbooks: {}, skills: {}, connectors: {} });

  const persistSidebarState = useCallback((updates) => {
    let nextSections = updates.sections;
    let nextCategories = updates.categories;
    if (nextSections) {
      setAgentsOpenState((p) => (nextSections.agents !== undefined ? nextSections.agents : p));
      setPlaybooksOpenState((p) => (nextSections.playbooks !== undefined ? nextSections.playbooks : p));
      setSkillsOpenState((p) => (nextSections.skills !== undefined ? nextSections.skills : p));
      setDataSourcesOpenState((p) => (nextSections.dataSources !== undefined ? nextSections.dataSources : p));
      setLinesOpenState((p) => (nextSections.lines !== undefined ? nextSections.lines : p));
    }
    if (nextCategories) setCategoryExpanded(nextCategories);
    try {
      const raw = localStorage.getItem(AF_SIDEBAR_STATE_KEY);
      const current = raw ? JSON.parse(raw) : { sections: {}, categories: {} };
      const next = {
        sections: { ...current.sections, ...(nextSections || {}) },
        categories: nextCategories || current.categories,
      };
      localStorage.setItem(AF_SIDEBAR_STATE_KEY, JSON.stringify(next));
    } catch (e) {
      console.warn('Failed to persist sidebar state', e);
    }
  }, []);

  const setAgentsOpen = useCallback((fn) => {
    setAgentsOpenState((prev) => {
      const next = typeof fn === 'function' ? fn(prev) : fn;
      persistSidebarState({ sections: { agents: next } });
      return next;
    });
  }, [persistSidebarState]);
  const setPlaybooksOpen = useCallback((fn) => {
    setPlaybooksOpenState((prev) => {
      const next = typeof fn === 'function' ? fn(prev) : fn;
      persistSidebarState({ sections: { playbooks: next } });
      return next;
    });
  }, [persistSidebarState]);
  const setSkillsOpen = useCallback((fn) => {
    setSkillsOpenState((prev) => {
      const next = typeof fn === 'function' ? fn(prev) : fn;
      persistSidebarState({ sections: { skills: next } });
      return next;
    });
  }, [persistSidebarState]);
  const setDataSourcesOpen = useCallback((fn) => {
    setDataSourcesOpenState((prev) => {
      const next = typeof fn === 'function' ? fn(prev) : fn;
      persistSidebarState({ sections: { dataSources: next } });
      return next;
    });
  }, [persistSidebarState]);
  const setLinesOpen = useCallback((fn) => {
    setLinesOpenState((prev) => {
      const next = typeof fn === 'function' ? fn(prev) : fn;
      persistSidebarState({ sections: { lines: next } });
      return next;
    });
  }, [persistSidebarState]);

  const toggleCategoryExpanded = useCallback((sectionKey, categoryName) => {
    setCategoryExpanded((prev) => {
      const section = prev[sectionKey] || {};
      const next = { ...prev, [sectionKey]: { ...section, [categoryName]: !section[categoryName] } };
      persistSidebarState({ categories: next });
      return next;
    });
  }, [persistSidebarState]);

  const isCategoryExpanded = useCallback((sectionKey, categoryName) => {
    const section = categoryExpanded[sectionKey] || {};
    return section[categoryName] !== false;
  }, [categoryExpanded]);

  const { data: profiles = [], isLoading: profilesLoading } = useQuery(
    'agentFactoryProfiles',
    () => apiService.agentFactory.listProfiles(),
    { retry: false }
  );
  const { data: defaultChatPref } = useQuery(
    'defaultChatAgentProfile',
    () => apiService.settings.getDefaultChatAgentProfile(),
    { retry: false }
  );
  const defaultChatProfileId = defaultChatPref?.agent_profile_id || null;
  const { data: playbooks = [], isLoading: playbooksLoading } = useQuery(
    'agentFactoryPlaybooks',
    () => apiService.agentFactory.listPlaybooks(),
    { retry: false }
  );
  const { data: connectors = [], isLoading: connectorsLoading } = useQuery(
    'agentFactoryConnectors',
    () => apiService.agentFactory.listConnectors(),
    { retry: false }
  );
  const { data: skillsList = [], isLoading: skillsLoading } = useQuery(
    'agentFactorySkills',
    () => apiService.agentFactory.listSkills({ include_builtin: true }),
    { retry: false }
  );
  const { data: linesList = [], isLoading: linesLoading } = useQuery(
    'agentFactoryLines',
    () => apiService.agentFactory.listLines(),
    { retry: false }
  );
  const { data: connectorTemplates = [], isLoading: templatesLoading } = useQuery(
    'agentFactoryConnectorTemplates',
    () => apiService.agentFactory.getConnectorTemplates(),
    { enabled: templateDialogOpen, retry: false }
  );
  const { data: sidebarCategories = [] } = useQuery(
    'agentFactorySidebarCategories',
    () => apiService.agentFactory.listSidebarCategories(),
    { retry: false }
  );

  const createSidebarCategoryMutation = useMutation(
    (body) => apiService.agentFactory.createSidebarCategory(body),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('agentFactorySidebarCategories');
        setAddCategoryOpen(false);
        setNewCategoryName('');
      },
    }
  );

  const updateProfileCategoryMutation = useMutation(
    ({ profileId, category }) => apiService.agentFactory.updateProfile(profileId, { category }),
    { onSuccess: () => queryClient.invalidateQueries('agentFactoryProfiles') }
  );
  const updatePlaybookCategoryMutation = useMutation(
    ({ playbookId, category }) => apiService.agentFactory.updatePlaybook(playbookId, { category }),
    { onSuccess: () => queryClient.invalidateQueries('agentFactoryPlaybooks') }
  );
  const updateSkillCategoryMutation = useMutation(
    ({ skillId, category }) => apiService.agentFactory.updateSkill(skillId, { category }),
    { onSuccess: () => queryClient.invalidateQueries('agentFactorySkills') }
  );
  const updateConnectorCategoryMutation = useMutation(
    ({ connectorId, category }) => apiService.agentFactory.updateConnector(connectorId, { category }),
    { onSuccess: () => queryClient.invalidateQueries('agentFactoryConnectors') }
  );

  const pauseMutation = useMutation(
    (profileId) => apiService.agentFactory.pauseProfile(profileId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('agentFactoryProfiles');
        queryClient.invalidateQueries('defaultChatAgentProfile');
      },
    }
  );
  const resumeMutation = useMutation(
    (profileId) => apiService.agentFactory.resumeProfile(profileId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('agentFactoryProfiles');
        queryClient.invalidateQueries('defaultChatAgentProfile');
      },
    }
  );
  const deleteMutation = useMutation(
    (profileId) => apiService.agentFactory.deleteProfile(profileId),
    {
      onSuccess: (_, profileId) => {
        queryClient.invalidateQueries('agentFactoryProfiles');
        queryClient.invalidateQueries('defaultChatAgentProfile');
        onCloseEntityTab?.('agent', profileId);
        setDeleteConfirm(null);
      },
    }
  );
  const deletePlaybookMutation = useMutation(
    (playbookId) => apiService.agentFactory.deletePlaybook(playbookId),
    {
      onSuccess: (_, playbookId) => {
        queryClient.invalidateQueries('agentFactoryPlaybooks');
        onCloseEntityTab?.('playbook', playbookId);
        setDeletePlaybookConfirm(null);
      },
    }
  );
  const clonePlaybookMutation = useMutation(
    (playbookId) => apiService.agentFactory.clonePlaybook(playbookId),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('agentFactoryPlaybooks');
        if (data?.id) navigate(`/agent-factory/playbook/${data.id}`);
      },
    }
  );
  const deleteConnectorMutation = useMutation(
    (connectorId) => apiService.agentFactory.deleteConnector(connectorId),
    {
      onSuccess: (_, connectorId) => {
        queryClient.invalidateQueries('agentFactoryConnectors');
        onCloseEntityTab?.('datasource', connectorId);
        setDeleteConnectorConfirm(null);
      },
    }
  );
  const deleteLineMutation = useMutation(
    (lineId) => apiService.agentFactory.deleteLine(lineId),
    {
      onSuccess: (_, lineId) => {
        queryClient.invalidateQueries('agentFactoryLines');
        onCloseEntityTab?.('line', lineId);
        setDeleteLineConfirm(null);
      },
    }
  );
  const createPlaybookMutation = useMutation(
    (body) => apiService.agentFactory.createPlaybook(body),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('agentFactoryPlaybooks');
        setCreatePlaybookOpen(false);
        setNewPlaybookName('');
        if (data?.id) navigate(`/agent-factory/playbook/${data.id}`);
      },
    }
  );
  const createConnectorMutation = useMutation(
    (body) => apiService.agentFactory.createConnector(body),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('agentFactoryConnectors');
        if (data?.id) navigate(`/agent-factory/datasource/${data.id}`);
      },
    }
  );
  const createConnectorFromTemplateMutation = useMutation(
    (body) => apiService.agentFactory.createConnectorFromTemplate(body),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('agentFactoryConnectors');
        setTemplateDialogOpen(false);
        setSelectedTemplate('');
        if (data?.id) navigate(`/agent-factory/datasource/${data.id}`);
      },
    }
  );
  const importBundleMutation = useMutation(
    (yamlString) => apiService.agentFactory.importAgentBundle(yamlString),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('agentFactoryProfiles');
        queryClient.invalidateQueries('agentFactoryPlaybooks');
        queryClient.invalidateQueries('agentFactoryConnectors');
        setImportFileInputKey((k) => k + 1);
        if (data?.id) navigate(`/agent-factory/agent/${data.id}`);
      },
    }
  );
  const importSkillsMutation = useMutation(
    (yamlString) => apiService.agentFactory.importSkills(yamlString),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('agentFactorySkills');
        setImportSkillsFileInputKey((k) => k + 1);
      },
    }
  );
  const importConnectorsMutation = useMutation(
    (yamlString) => apiService.agentFactory.importConnectors(yamlString),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('agentFactoryConnectors');
        setImportConnectorsFileInputKey((k) => k + 1);
      },
    }
  );
  const importPlaybookMutation = useMutation(
    (body) => apiService.agentFactory.importPlaybook(body),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('agentFactoryPlaybooks');
        setImportPlaybooksFileInputKey((k) => k + 1);
        if (data?.id) navigate(`/agent-factory/playbook/${data.id}`);
      },
    }
  );

  const filteredProfiles = agentSearch.trim()
    ? profiles.filter(
        (p) =>
          (p.name || '').toLowerCase().includes(agentSearch.toLowerCase()) ||
          (p.handle || '').toLowerCase().includes(agentSearch.toLowerCase())
      )
    : profiles;
  const userPlaybooks = playbooks.filter((pb) => !pb.is_template || !!pb.is_builtin);
  const filteredPlaybooks = playbookSearch.trim()
    ? userPlaybooks.filter(
        (p) =>
          (p.name || '').toLowerCase().includes(playbookSearch.toLowerCase())
      )
    : userPlaybooks;
  const filteredConnectors = connectorSearch.trim()
    ? connectors.filter(
        (c) =>
          (c.name || '').toLowerCase().includes(connectorSearch.toLowerCase())
      )
    : connectors;
  const filteredSkills = skillSearch.trim()
    ? skillsList.filter(
        (s) =>
          (s.name || '').toLowerCase().includes(skillSearch.toLowerCase()) ||
          (s.slug || '').toLowerCase().includes(skillSearch.toLowerCase())
      )
    : skillsList;
  const filteredLines = lineSearch.trim()
    ? linesList.filter((l) => (l.name || '').toLowerCase().includes(lineSearch.toLowerCase()))
    : linesList;
  const handleSelectAgent = (profile) => {
    navigate(`/agent-factory/agent/${profile.id}`);
  };
  const handleSelectPlaybook = (playbook) => {
    navigate(`/agent-factory/playbook/${playbook.id}`);
  };
  const handleSelectConnector = (connector) => {
    navigate(`/agent-factory/datasource/${connector.id}`);
  };

  const handleCreateAgent = () => {
    if (onOpenCreate) onOpenCreate();
    else navigate('/agent-factory');
  };

  const handleImportAgent = (e) => {
    const file = e?.target?.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const text = typeof reader.result === 'string' ? reader.result : '';
      if (text) importBundleMutation.mutate(text);
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const handleExportSkills = async () => {
    try {
      const res = await apiService.agentFactory.exportSkills();
      const text = await res.text();
      const blob = new Blob([text], { type: 'application/x-yaml' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'skills.yaml';
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export skills failed:', err);
    }
  };

  const handleImportSkills = (e) => {
    const file = e?.target?.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const text = typeof reader.result === 'string' ? reader.result : '';
      if (text) importSkillsMutation.mutate(text);
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const handleCreateSkill = async () => {
    try {
      const slug = `custom-skill-${Date.now().toString(36)}`;
      const created = await apiService.agentFactory.createSkill({
        name: 'New skill',
        slug,
        procedure: 'Describe how to use tools and perform this task. Required tools will be auto-bound to the step.',
        required_tools: [],
        optional_tools: [],
      });
      queryClient.invalidateQueries('agentFactorySkills');
      if (created?.id) navigate(`/agent-factory/skill/${created.id}`);
    } catch (err) {
      console.error('Create skill failed:', err);
    }
  };

  const handleExportConnectors = async () => {
    try {
      const res = await apiService.agentFactory.exportConnectors();
      const text = await res.text();
      const blob = new Blob([text], { type: 'application/x-yaml' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'connectors.yaml';
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export connectors failed:', err);
    }
  };

  const handleImportConnectors = (e) => {
    const file = e?.target?.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const text = typeof reader.result === 'string' ? reader.result : '';
      if (text) importConnectorsMutation.mutate(text);
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const handleImportPlaybook = (e) => {
    const file = e?.target?.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const text = typeof reader.result === 'string' ? reader.result : '';
        const playbookData = JSON.parse(text);
        importPlaybookMutation.mutate(playbookData);
      } catch (err) {
        console.error('Failed to parse playbook JSON:', err);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const handleCreatePlaybook = () => {
    if (!newPlaybookName.trim()) return;
    createPlaybookMutation.mutate({
      name: newPlaybookName.trim(),
      definition: { steps: [], run_context: 'interactive' },
    });
  };

  const handleToggleActive = (e, profile) => {
    e.stopPropagation();
    if (profile.is_active) {
      pauseMutation.mutate(profile.id);
    } else {
      resumeMutation.mutate(profile.id);
    }
  };

  const stepCount = (pb) => (pb.definition?.steps?.length ?? 0);

  const apiSectionKey = (sectionKey) => (sectionKey === 'dataSources' ? 'connectors' : sectionKey);

  const getEffectiveCategories = useCallback((sectionKey, items, getItemCategory) => {
    const apiSection = apiSectionKey(sectionKey);
    const fromApi = (sidebarCategories || [])
      .filter((c) => c.section === apiSection)
      .map((c) => c.name);
    const fromItems = [...new Set((items || []).map(getItemCategory).filter(Boolean))];
    const combined = [...new Set([...fromApi, ...fromItems])];
    const hasUncategorized = (items || []).some((i) => !getItemCategory(i) || getItemCategory(i) === UNCATEGORIZED);
    if (hasUncategorized && !combined.includes(UNCATEGORIZED)) combined.push(UNCATEGORIZED);
    if (items?.length === 0 && !combined.includes(UNCATEGORIZED)) combined.push(UNCATEGORIZED);
    const named = combined.filter((c) => c !== UNCATEGORIZED);
    named.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
    if (combined.includes(UNCATEGORIZED)) named.push(UNCATEGORIZED);
    return named;
  }, [sidebarCategories]);

  const groupByCategory = useCallback((items, getItemCategory, getSortLabel) => {
    const map = {};
    (items || []).forEach((item) => {
      const cat = getItemCategory(item) || UNCATEGORIZED;
      if (!map[cat]) map[cat] = [];
      map[cat].push(item);
    });
    const label = getSortLabel || ((i) => i.name || '');
    Object.keys(map).forEach((k) => {
      map[k].sort((a, b) =>
        String(label(a) || '').localeCompare(String(label(b) || ''), undefined, { sensitivity: 'base' })
      );
    });
    return map;
  }, []);

  const handleDragEnd = useCallback((result, sectionKey, items, getItemCategory, updateMutation) => {
    if (!result.destination || result.source.droppableId === result.destination.droppableId) return;
    const destId = result.destination.droppableId;
    const categoryName = destId.includes('::') ? destId.split('::')[1] : destId;
    const targetCategory = categoryName === UNCATEGORIZED ? null : categoryName;
    const itemId = result.draggableId.replace(/^[^:]+::/, '');
    const item = items.find((i) => i.id === itemId);
    if (!item || (getItemCategory(item) || UNCATEGORIZED) === categoryName) return;
    updateMutation.mutate({ [sectionKey === 'agents' ? 'profileId' : sectionKey === 'playbooks' ? 'playbookId' : sectionKey === 'skills' ? 'skillId' : 'connectorId']: itemId, category: targetCategory });
  }, []);

  const agentsCategories = useMemo(() => getEffectiveCategories('agents', filteredProfiles, (p) => p.category), [getEffectiveCategories, filteredProfiles]);
  const agentsByCategory = useMemo(
    () => groupByCategory(filteredProfiles, (p) => p.category, (p) => p.name || p.handle || ''),
    [groupByCategory, filteredProfiles]
  );
  const playbooksCategories = useMemo(() => getEffectiveCategories('playbooks', filteredPlaybooks, (p) => p.category), [getEffectiveCategories, filteredPlaybooks]);
  const playbooksByCategory = useMemo(
    () => groupByCategory(filteredPlaybooks, (p) => p.category, (p) => p.name || ''),
    [groupByCategory, filteredPlaybooks]
  );
  const skillsCategories = useMemo(() => getEffectiveCategories('skills', filteredSkills, (s) => s.category), [getEffectiveCategories, filteredSkills]);
  const skillsByCategory = useMemo(
    () => groupByCategory(filteredSkills, (s) => s.category, (s) => s.name || s.slug || ''),
    [groupByCategory, filteredSkills]
  );
  const connectorsCategories = useMemo(() => getEffectiveCategories('connectors', filteredConnectors, (c) => c.category), [getEffectiveCategories, filteredConnectors]);
  const connectorsByCategory = useMemo(
    () => groupByCategory(filteredConnectors, (c) => c.category, (c) => c.name || ''),
    [groupByCategory, filteredConnectors]
  );

  return (
    <Box
      sx={{
        width: SIDEBAR_WIDTH,
        minWidth: SIDEBAR_WIDTH,
        borderRight: 1,
        borderColor: 'divider',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        bgcolor: 'background.paper',
      }}
    >
      <Box sx={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
        {/* Agents section */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              px: 2,
              py: 1,
              cursor: 'pointer',
              '&:hover': { bgcolor: 'action.hover' },
            }}
            onClick={() => setAgentsOpen((o) => !o)}
          >
            <AfSectionHeaderTitle Icon={SmartToy}>Agents</AfSectionHeaderTitle>
            {agentsOpen ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
          </Box>
          <Collapse in={agentsOpen}>
            <Box sx={{ px: 2, pb: 2 }}>
              <input
                type="file"
                accept=".yaml,.yml"
                key={importFileInputKey}
                onChange={handleImportAgent}
                style={{ display: 'none' }}
                id="agent-import-file"
              />
              <AfSectionSearchRow
                searchOpen={agentSearchOpen}
                onSearchOpenChange={setAgentSearchOpen}
                value={agentSearch}
                onValueChange={setAgentSearch}
                placeholder="Search agents…"
              >
                <AfSectionMenuTrigger setMenuAnchor={setAgentsMenuAnchor} />
              </AfSectionSearchRow>
              <Menu
                anchorEl={agentsMenuAnchor}
                open={!!agentsMenuAnchor}
                onClose={() => setAgentsMenuAnchor(null)}
              >
                <MenuItem onClick={() => { setAgentsMenuAnchor(null); handleCreateAgent(); }}>
                  Create new
                </MenuItem>
                <MenuItem onClick={() => { setAgentsMenuAnchor(null); document.getElementById('agent-import-file')?.click(); }} disabled={importBundleMutation.isLoading}>
                  Import
                </MenuItem>
                <MenuItem onClick={() => { setAgentsMenuAnchor(null); setAddCategorySection('agents'); setNewCategoryName(''); setAddCategoryOpen(true); }}>
                  Add category
                </MenuItem>
              </Menu>
            </Box>
            {profilesLoading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
                <CircularProgress size={24} />
              </Box>
            )}
            {!profilesLoading && (
              <DragDropContext
                onDragEnd={(result) => handleDragEnd(result, 'agents', filteredProfiles, (p) => p.category, updateProfileCategoryMutation)}
              >
                {agentsCategories.map((cat) => {
                  const expanded = isCategoryExpanded('agents', cat);
                  const items = agentsByCategory[cat] || [];
                  return (
                    <Box key={cat} sx={{ borderTop: 1, borderColor: 'divider' }}>
                      <Box
                        sx={{ display: 'flex', alignItems: 'center', px: 1.5, py: 0.5, cursor: 'pointer', '&:hover': { bgcolor: 'action.hover' } }}
                        onClick={() => toggleCategoryExpanded('agents', cat)}
                      >
                        {expanded ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
                        <Typography variant="caption" fontWeight={500} sx={{ ml: 0.5 }}>{cat}</Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ ml: 0.5 }}>({items.length})</Typography>
                      </Box>
                      <Collapse in={expanded}>
                        <Droppable droppableId={`agents::${cat}`}>
                          {(provided) => (
                            <List dense disablePadding ref={provided.innerRef} {...provided.droppableProps} sx={{ pb: 0.5 }}>
                              {items.map((p, idx) => {
                                const isChatDefault =
                                  (defaultChatProfileId && defaultChatProfileId === p.id) ||
                                  (!defaultChatProfileId && !!p.is_builtin);
                                return (
                                <Draggable key={p.id} draggableId={`agent::${p.id}`} index={idx}>
                                  {(dragProvided) => (
                                    <ListItemButton
                                      ref={dragProvided.innerRef}
                                      {...dragProvided.draggableProps}
                                      {...dragProvided.dragHandleProps}
                                      selected={selectedSection === 'agent' && selectedId === p.id}
                                      onClick={() => handleSelectAgent(p)}
                                      sx={{
                                        pr: 0.5,
                                        ...(isChatDefault
                                          ? {
                                              borderLeft: '3px solid',
                                              borderLeftColor: 'primary.main',
                                              bgcolor: 'action.selected',
                                            }
                                          : {}),
                                      }}
                                    >
                                      <StatusDot status={p.status || (p.is_active ? 'active' : 'paused')} />
                                      <ListItemText
                                        primary={
                                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 0 }}>
                                            <Typography component="span" variant="body2" noWrap sx={{ flex: 1, minWidth: 0 }}>
                                              {p.name || p.handle || 'Unnamed'}
                                            </Typography>
                                            {isChatDefault && (
                                              <Chip size="small" color="primary" label="Default" sx={{ height: 22, flexShrink: 0 }} />
                                            )}
                                          </Box>
                                        }
                                        secondary={
                                          <>
                                            {p.handle ? `@${p.handle}` : 'Schedule only'}
                                            {p.budget?.monthly_limit_usd != null && p.budget.monthly_limit_usd > 0 && (
                                              <Typography component="span" variant="caption" display="block" color="text.secondary">
                                                ${Number(p.budget.current_period_spend_usd || 0).toFixed(2)} / ${Number(p.budget.monthly_limit_usd).toFixed(2)}
                                              </Typography>
                                            )}
                                          </>
                                        }
                                        primaryTypographyProps={{ noWrap: true }}
                                        secondaryTypographyProps={{ noWrap: true }}
                                        sx={{ ml: 1, mr: 0.5 }}
                                      />
                                      <Tooltip title={p.is_active ? 'Pause agent' : 'Resume agent'}>
                                        <IconButton size="small" onClick={(e) => { e.stopPropagation(); handleToggleActive(e, p); }} disabled={pauseMutation.isLoading || resumeMutation.isLoading} aria-label={p.is_active ? 'Pause' : 'Resume'}>
                                          {p.is_active ? <Pause fontSize="small" /> : <PlayArrow fontSize="small" />}
                                        </IconButton>
                                      </Tooltip>
                                      {p.is_locked && !p.is_builtin && <Tooltip title="Locked"><Lock fontSize="small" sx={{ color: 'text.secondary', mr: 0.25 }} /></Tooltip>}
                                      {p.is_builtin && <Tooltip title="Built-in"><Lock fontSize="small" sx={{ color: 'text.secondary', mr: 0.25 }} /></Tooltip>}
                                      <Tooltip title={p.is_builtin ? 'Built-in agent cannot be deleted' : p.is_locked ? 'Unlock in editor to delete' : 'Delete agent'}>
                                        <span>
                                          <IconButton size="small" onClick={(e) => { e.stopPropagation(); if (!p.is_locked && !p.is_builtin) setDeleteConfirm(p); }} disabled={!!p.is_locked || !!p.is_builtin} aria-label="Delete agent" sx={{ color: 'error.main', opacity: (p.is_locked || p.is_builtin) ? 0.4 : 0.6, '&:hover': { opacity: (p.is_locked || p.is_builtin) ? 0.4 : 1 } }}>
                                            <Delete fontSize="small" />
                                          </IconButton>
                                        </span>
                                      </Tooltip>
                                    </ListItemButton>
                                  )}
                                </Draggable>
                              );
                              })}
                              {provided.placeholder}
                              {items.length === 0 && (
                                <ListItemButton disabled>
                                  <ListItemText primary={cat === UNCATEGORIZED && !agentSearch.trim() ? 'No agents yet' : 'Empty'} />
                                </ListItemButton>
                              )}
                            </List>
                          )}
                        </Droppable>
                      </Collapse>
                    </Box>
                  );
                })}
              </DragDropContext>
            )}
          </Collapse>
        </Box>

        {/* Playbooks section */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              px: 2,
              py: 1,
              cursor: 'pointer',
              '&:hover': { bgcolor: 'action.hover' },
            }}
            onClick={() => setPlaybooksOpen((o) => !o)}
          >
            <AfSectionHeaderTitle Icon={PlayArrow}>Playbooks</AfSectionHeaderTitle>
            {playbooksOpen ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
          </Box>
          <Collapse in={playbooksOpen}>
            <Box sx={{ px: 2, pb: 2 }}>
              <input
                type="file"
                accept=".json"
                key={importPlaybooksFileInputKey}
                onChange={handleImportPlaybook}
                style={{ display: 'none' }}
                id="playbooks-import-file"
              />
              <AfSectionSearchRow
                searchOpen={playbookSearchOpen}
                onSearchOpenChange={setPlaybookSearchOpen}
                value={playbookSearch}
                onValueChange={setPlaybookSearch}
                placeholder="Search playbooks…"
              >
                <AfSectionMenuTrigger setMenuAnchor={setPlaybooksMenuAnchor} />
              </AfSectionSearchRow>
              <Menu
                anchorEl={playbooksMenuAnchor}
                open={!!playbooksMenuAnchor}
                onClose={() => setPlaybooksMenuAnchor(null)}
              >
                <MenuItem onClick={() => { setPlaybooksMenuAnchor(null); setCreatePlaybookOpen(true); }}>
                  Create new
                </MenuItem>
                <MenuItem onClick={() => { setPlaybooksMenuAnchor(null); document.getElementById('playbooks-import-file')?.click(); }} disabled={importPlaybookMutation.isLoading}>
                  Import
                </MenuItem>
                <MenuItem onClick={() => { setPlaybooksMenuAnchor(null); setAddCategorySection('playbooks'); setNewCategoryName(''); setAddCategoryOpen(true); }}>
                  Add category
                </MenuItem>
              </Menu>
            </Box>
            {playbooksLoading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
                <CircularProgress size={24} />
              </Box>
            )}
            {!playbooksLoading && (
              <DragDropContext onDragEnd={(result) => handleDragEnd(result, 'playbooks', filteredPlaybooks, (p) => p.category, updatePlaybookCategoryMutation)}>
                {playbooksCategories.map((cat) => {
                  const expanded = isCategoryExpanded('playbooks', cat);
                  const items = playbooksByCategory[cat] || [];
                  return (
                    <Box key={cat} sx={{ borderTop: 1, borderColor: 'divider' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', px: 1.5, py: 0.5, cursor: 'pointer', '&:hover': { bgcolor: 'action.hover' } }} onClick={() => toggleCategoryExpanded('playbooks', cat)}>
                        {expanded ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
                        <Typography variant="caption" fontWeight={500} sx={{ ml: 0.5 }}>{cat}</Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ ml: 0.5 }}>({items.length})</Typography>
                      </Box>
                      <Collapse in={expanded}>
                        <Droppable droppableId={`playbooks::${cat}`}>
                          {(provided) => (
                            <List dense disablePadding ref={provided.innerRef} {...provided.droppableProps} sx={{ pb: 0.5 }}>
                              {items.map((pb, idx) => (
                                <Draggable key={pb.id} draggableId={`playbook::${pb.id}`} index={idx}>
                                  {(dragProvided) => (
                                    <ListItemButton ref={dragProvided.innerRef} {...dragProvided.draggableProps} {...dragProvided.dragHandleProps} selected={selectedSection === 'playbook' && selectedId === pb.id} onClick={() => handleSelectPlaybook(pb)} sx={{ pr: 0.5 }}>
                                      <ListItemText primary={pb.name || 'Unnamed'} secondary={`${stepCount(pb)} step(s)`} primaryTypographyProps={{ noWrap: true }} secondaryTypographyProps={{ noWrap: true }} sx={{ mr: 0.5 }} />
                                      {pb.is_locked && !pb.is_builtin && <Tooltip title="Locked"><Lock fontSize="small" sx={{ color: 'text.secondary', mr: 0.25 }} /></Tooltip>}
                                      {pb.is_builtin && <Tooltip title="Built-in"><Lock fontSize="small" sx={{ color: 'text.secondary', mr: 0.25 }} /></Tooltip>}
                                      <Tooltip title="Clone playbook">
                                        <IconButton size="small" onClick={(e) => { e.stopPropagation(); clonePlaybookMutation.mutate(pb.id); }} disabled={clonePlaybookMutation.isLoading} aria-label="Clone playbook" sx={{ color: 'text.secondary', opacity: 0.6, '&:hover': { opacity: 1 } }}>
                                          <ContentCopy fontSize="small" />
                                        </IconButton>
                                      </Tooltip>
                                      <Tooltip title={pb.is_builtin ? 'Built-in playbook cannot be deleted' : pb.is_locked ? 'Unlock in editor to delete' : 'Delete playbook'}>
                                        <span>
                                          <IconButton size="small" onClick={(e) => { e.stopPropagation(); if (!pb.is_locked && !pb.is_builtin) setDeletePlaybookConfirm(pb); }} disabled={!!pb.is_locked || !!pb.is_builtin} aria-label="Delete playbook" sx={{ color: 'error.main', opacity: (pb.is_locked || pb.is_builtin) ? 0.4 : 0.6, '&:hover': { opacity: (pb.is_locked || pb.is_builtin) ? 0.4 : 1 } }}>
                                            <Delete fontSize="small" />
                                          </IconButton>
                                        </span>
                                      </Tooltip>
                                    </ListItemButton>
                                  )}
                                </Draggable>
                              ))}
                              {provided.placeholder}
                              {items.length === 0 && <ListItemButton disabled><ListItemText primary={cat === UNCATEGORIZED && !playbookSearch.trim() ? 'No playbooks yet' : 'Empty'} /></ListItemButton>}
                            </List>
                          )}
                        </Droppable>
                      </Collapse>
                    </Box>
                  );
                })}
              </DragDropContext>
            )}
          </Collapse>
        </Box>

        {/* Skills section */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              px: 2,
              py: 1,
              cursor: 'pointer',
              '&:hover': { bgcolor: 'action.hover' },
            }}
            onClick={() => setSkillsOpen((o) => !o)}
          >
            <AfSectionHeaderTitle Icon={Build}>Skills</AfSectionHeaderTitle>
            {skillsOpen ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
          </Box>
          <Collapse in={skillsOpen}>
            <Box sx={{ px: 2, pb: 2 }}>
              <input
                type="file"
                accept=".yaml,.yml"
                key={importSkillsFileInputKey}
                onChange={handleImportSkills}
                style={{ display: 'none' }}
                id="skills-import-file"
              />
              <AfSectionSearchRow
                searchOpen={skillSearchOpen}
                onSearchOpenChange={setSkillSearchOpen}
                value={skillSearch}
                onValueChange={setSkillSearch}
                placeholder="Search skills…"
              >
                <AfSectionMenuTrigger setMenuAnchor={setSkillsMenuAnchor} />
              </AfSectionSearchRow>
              <Menu
                anchorEl={skillsMenuAnchor}
                open={!!skillsMenuAnchor}
                onClose={() => setSkillsMenuAnchor(null)}
              >
                <MenuItem onClick={() => { setSkillsMenuAnchor(null); handleCreateSkill(); }}>
                  Create new
                </MenuItem>
                <MenuItem onClick={() => { setSkillsMenuAnchor(null); document.getElementById('skills-import-file')?.click(); }} disabled={importSkillsMutation.isLoading}>
                  Import
                </MenuItem>
                <MenuItem onClick={() => { setSkillsMenuAnchor(null); handleExportSkills(); }}>
                  Export all (YAML)
                </MenuItem>
                <MenuItem onClick={() => { setSkillsMenuAnchor(null); setAddCategorySection('skills'); setNewCategoryName(''); setAddCategoryOpen(true); }}>
                  Add category
                </MenuItem>
              </Menu>
            </Box>
            {skillsLoading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
                <CircularProgress size={24} />
              </Box>
            )}
            {!skillsLoading && (
              <DragDropContext onDragEnd={(result) => handleDragEnd(result, 'skills', filteredSkills, (s) => s.category, updateSkillCategoryMutation)}>
                {skillsCategories.map((cat) => {
                  const expanded = isCategoryExpanded('skills', cat);
                  const items = skillsByCategory[cat] || [];
                  return (
                    <Box key={cat} sx={{ borderTop: 1, borderColor: 'divider' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', px: 1.5, py: 0.5, cursor: 'pointer', '&:hover': { bgcolor: 'action.hover' } }} onClick={() => toggleCategoryExpanded('skills', cat)}>
                        {expanded ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
                        <Typography variant="caption" fontWeight={500} sx={{ ml: 0.5 }}>{cat}</Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ ml: 0.5 }}>({items.length})</Typography>
                      </Box>
                      <Collapse in={expanded}>
                        <Droppable droppableId={`skills::${cat}`}>
                          {(provided) => (
                            <List dense disablePadding ref={provided.innerRef} {...provided.droppableProps} sx={{ pb: 0.5 }}>
                              {items.map((s, idx) => (
                                <Draggable key={s.id} draggableId={`skill::${s.id}`} index={idx}>
                                  {(dragProvided) => (
                                    <ListItemButton ref={dragProvided.innerRef} {...dragProvided.draggableProps} {...dragProvided.dragHandleProps} selected={selectedSection === 'skill' && selectedId === s.id} onClick={() => navigate(`/agent-factory/skill/${s.id}`)} sx={{ pr: 0.5 }}>
                                      {s.is_builtin && <Tooltip title="Built-in"><Lock fontSize="small" sx={{ color: 'text.secondary', mr: 0.5 }} /></Tooltip>}
                                      <ListItemText primary={s.name || s.slug} secondary={s.category ? `${s.category}` : null} primaryTypographyProps={{ noWrap: true }} secondaryTypographyProps={{ noWrap: true }} sx={{ mr: 0.5 }} />
                                    </ListItemButton>
                                  )}
                                </Draggable>
                              ))}
                              {provided.placeholder}
                              {items.length === 0 && <ListItemButton disabled><ListItemText primary={cat === UNCATEGORIZED && !skillSearch.trim() ? 'No skills' : 'Empty'} /></ListItemButton>}
                            </List>
                          )}
                        </Droppable>
                      </Collapse>
                    </Box>
                  );
                })}
              </DragDropContext>
            )}
          </Collapse>
        </Box>

        {/* Data Connections section */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              px: 2,
              py: 1,
              cursor: 'pointer',
              '&:hover': { bgcolor: 'action.hover' },
            }}
            onClick={() => setDataSourcesOpen((o) => !o)}
          >
            <AfSectionHeaderTitle Icon={Storage}>Data Connections</AfSectionHeaderTitle>
            {dataSourcesOpen ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
          </Box>
          <Collapse in={dataSourcesOpen}>
            <Box sx={{ px: 2, pb: 2 }}>
              <input
                type="file"
                accept=".yaml,.yml"
                key={importConnectorsFileInputKey}
                onChange={handleImportConnectors}
                style={{ display: 'none' }}
                id="connectors-import-file"
              />
              <AfSectionSearchRow
                searchOpen={connectorSearchOpen}
                onSearchOpenChange={setConnectorSearchOpen}
                value={connectorSearch}
                onValueChange={setConnectorSearch}
                placeholder="Search connections…"
              >
                <AfSectionMenuTrigger setMenuAnchor={setConnectorsMenuAnchor} />
              </AfSectionSearchRow>
              <Menu
                anchorEl={connectorsMenuAnchor}
                open={!!connectorsMenuAnchor}
                onClose={() => setConnectorsMenuAnchor(null)}
              >
                <MenuItem onClick={() => {
                  setConnectorsMenuAnchor(null);
                  createConnectorMutation.mutate({
                    name: 'New connector',
                    description: '',
                    connector_type: 'rest',
                    definition: { base_url: '', auth: { type: 'none' }, endpoints: {} },
                    requires_auth: false,
                    auth_fields: [],
                  });
                }} disabled={createConnectorMutation.isLoading}>
                  Create new
                </MenuItem>
                <MenuItem onClick={() => { setConnectorsMenuAnchor(null); setTemplateDialogOpen(true); }}>
                  Add from template
                </MenuItem>
                <MenuItem onClick={() => { setConnectorsMenuAnchor(null); document.getElementById('connectors-import-file')?.click(); }} disabled={importConnectorsMutation.isLoading}>
                  Import
                </MenuItem>
                <MenuItem onClick={() => { setConnectorsMenuAnchor(null); handleExportConnectors(); }}>
                  Export all (YAML)
                </MenuItem>
                <MenuItem onClick={() => { setConnectorsMenuAnchor(null); setAddCategorySection('connectors'); setNewCategoryName(''); setAddCategoryOpen(true); }}>
                  Add category
                </MenuItem>
              </Menu>
            </Box>
            {connectorsLoading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
                <CircularProgress size={24} />
              </Box>
            )}
            {!connectorsLoading && (
              <DragDropContext onDragEnd={(result) => handleDragEnd(result, 'connectors', filteredConnectors, (c) => c.category, updateConnectorCategoryMutation)}>
                {connectorsCategories.map((cat) => {
                  const expanded = isCategoryExpanded('connectors', cat);
                  const items = connectorsByCategory[cat] || [];
                  return (
                    <Box key={cat} sx={{ borderTop: 1, borderColor: 'divider' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', px: 1.5, py: 0.5, cursor: 'pointer', '&:hover': { bgcolor: 'action.hover' } }} onClick={() => toggleCategoryExpanded('connectors', cat)}>
                        {expanded ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
                        <Typography variant="caption" fontWeight={500} sx={{ ml: 0.5 }}>{cat}</Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ ml: 0.5 }}>({items.length})</Typography>
                      </Box>
                      <Collapse in={expanded}>
                        <Droppable droppableId={`connectors::${cat}`}>
                          {(provided) => (
                            <List dense disablePadding ref={provided.innerRef} {...provided.droppableProps} sx={{ pb: 0.5 }}>
                              {items.map((c, idx) => (
                                <Draggable key={c.id} draggableId={`connector::${c.id}`} index={idx}>
                                  {(dragProvided) => (
                                    <ListItemButton ref={dragProvided.innerRef} {...dragProvided.draggableProps} {...dragProvided.dragHandleProps} selected={selectedSection === 'datasource' && selectedId === c.id} onClick={() => handleSelectConnector(c)} sx={{ pr: 0.5 }}>
                                      <ListItemText primary={c.name || 'Unnamed'} secondary={c.connector_type + (c.endpoint_count != null ? ` · ${c.endpoint_count} endpoint(s)` : '')} primaryTypographyProps={{ noWrap: true }} secondaryTypographyProps={{ noWrap: true }} sx={{ mr: 0.5 }} />
                                      {c.is_locked && <Tooltip title="Locked"><Lock fontSize="small" sx={{ color: 'text.secondary', mr: 0.25 }} /></Tooltip>}
                                      <Tooltip title={c.is_locked ? 'Unlock in editor to delete' : 'Delete data connection'}>
                                        <span>
                                          <IconButton size="small" onClick={(e) => { e.stopPropagation(); if (!c.is_locked) setDeleteConnectorConfirm(c); }} disabled={!!c.is_locked} aria-label="Delete data connection" sx={{ color: 'error.main', opacity: c.is_locked ? 0.4 : 0.6, '&:hover': { opacity: c.is_locked ? 0.4 : 1 } }}>
                                            <Delete fontSize="small" />
                                          </IconButton>
                                        </span>
                                      </Tooltip>
                                    </ListItemButton>
                                  )}
                                </Draggable>
                              ))}
                              {provided.placeholder}
                              {items.length === 0 && <ListItemButton disabled><ListItemText primary={cat === UNCATEGORIZED && !connectorSearch.trim() ? 'No data connections yet' : 'Empty'} /></ListItemButton>}
                            </List>
                          )}
                        </Droppable>
                      </Collapse>
                    </Box>
                  );
                })}
              </DragDropContext>
            )}
          </Collapse>
        </Box>

        {/* Lines section */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              px: 2,
              py: 1,
              cursor: 'pointer',
              '&:hover': { bgcolor: 'action.hover' },
            }}
            onClick={() => setLinesOpen((o) => !o)}
          >
            <AfSectionHeaderTitle Icon={Group}>Lines</AfSectionHeaderTitle>
            {linesOpen ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
          </Box>
          <Collapse in={linesOpen}>
            <Box sx={{ px: 2, pb: 2 }}>
              <AfSectionSearchRow
                searchOpen={lineSearchOpen}
                onSearchOpenChange={setLineSearchOpen}
                value={lineSearch}
                onValueChange={setLineSearch}
                placeholder="Search lines…"
              >
                <AfSectionMenuTrigger setMenuAnchor={setLinesMenuAnchor} />
              </AfSectionSearchRow>
              <Menu
                anchorEl={linesMenuAnchor}
                open={!!linesMenuAnchor}
                onClose={() => setLinesMenuAnchor(null)}
              >
                <MenuItem onClick={() => { setLinesMenuAnchor(null); setQuickLineWizardOpen(true); }}>
                  Quick line
                </MenuItem>
                <MenuItem onClick={() => { setLinesMenuAnchor(null); setCreateLineOpen(true); }}>
                  New line
                </MenuItem>
              </Menu>
            </Box>
            {linesLoading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
                <CircularProgress size={24} />
              </Box>
            )}
            {!linesLoading && (
              <List dense disablePadding>
                {filteredLines.length === 0 ? (
                  <ListItemButton disabled>
                    <ListItemText primary={lineSearch.trim() ? 'No matches' : 'No lines yet'} />
                  </ListItemButton>
                ) : (
                  filteredLines.map((line) => (
                    <ListItemButton
                      key={line.id}
                      selected={selectedSection === 'line' && selectedId === line.id}
                      onClick={() => navigate(`/agent-factory/line/${line.id}`)}
                      sx={{ pr: 0.5 }}
                    >
                      <StatusDot status={line.status === 'active' ? 'active' : line.status === 'paused' ? 'paused' : 'default'} />
                      <ListItemText
                        primary={line.name || 'Unnamed line'}
                        secondary={`${line.member_count ?? 0} agent${(line.member_count ?? 0) !== 1 ? 's' : ''}`}
                        primaryTypographyProps={{ noWrap: true }}
                        secondaryTypographyProps={{ noWrap: true }}
                        sx={{ ml: 1, mr: 0.5 }}
                      />
                      <Tooltip title="Delete line">
                        <span>
                          <IconButton
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation();
                              setDeleteLineConfirm(line);
                            }}
                            aria-label="Delete line"
                            sx={{ color: 'error.main', opacity: 0.6, '&:hover': { opacity: 1 } }}
                          >
                            <Delete fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                    </ListItemButton>
                  ))
                )}
              </List>
            )}
          </Collapse>
        </Box>
      </Box>

      <QuickTeamWizard
        open={quickLineWizardOpen}
        onClose={() => setQuickLineWizardOpen(false)}
        onSuccess={(line) => {
          queryClient.invalidateQueries('agentFactoryLines');
          if (line?.id) navigate(`/agent-factory/line/${line.id}`);
          setQuickLineWizardOpen(false);
        }}
      />

      <Dialog open={createLineOpen} onClose={() => setCreateLineOpen(false)} maxWidth="sm" fullWidth>
        <DialogContent sx={{ pt: 2 }}>
          <TeamEditor
            onSuccess={(data) => {
              queryClient.invalidateQueries('agentFactoryLines');
              setCreateLineOpen(false);
              if (data?.id) navigate(`/agent-factory/line/${data.id}`);
            }}
            onCancel={() => setCreateLineOpen(false)}
          />
        </DialogContent>
      </Dialog>

      {/* Delete agent confirmation */}
      <Dialog open={!!deleteConfirm} onClose={() => !deleteMutation.isLoading && setDeleteConfirm(null)}>
        <DialogTitle>Delete agent</DialogTitle>
        <DialogContent>
          <Typography>
            Permanently delete <strong>{deleteConfirm?.name || deleteConfirm?.handle || 'this agent'}</strong>?
            This will also remove its data connections, skills, and execution history.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirm(null)} disabled={deleteMutation.isLoading}>
            Cancel
          </Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => deleteConfirm?.id && deleteMutation.mutate(deleteConfirm.id)}
            disabled={deleteMutation.isLoading}
          >
            {deleteMutation.isLoading ? 'Deleting…' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete playbook confirmation */}
      <Dialog open={!!deletePlaybookConfirm} onClose={() => !deletePlaybookMutation.isLoading && setDeletePlaybookConfirm(null)}>
        <DialogTitle>Delete playbook</DialogTitle>
        <DialogContent>
          <Typography>
            Permanently delete playbook <strong>{deletePlaybookConfirm?.name || 'this playbook'}</strong>?
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeletePlaybookConfirm(null)} disabled={deletePlaybookMutation.isLoading}>
            Cancel
          </Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => deletePlaybookConfirm?.id && deletePlaybookMutation.mutate(deletePlaybookConfirm.id)}
            disabled={deletePlaybookMutation.isLoading}
          >
            {deletePlaybookMutation.isLoading ? 'Deleting…' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete data connection confirmation */}
      <Dialog open={!!deleteConnectorConfirm} onClose={() => !deleteConnectorMutation.isLoading && setDeleteConnectorConfirm(null)}>
        <DialogTitle>Delete data connection</DialogTitle>
        <DialogContent>
          <Typography>
            Permanently delete data connection <strong>{deleteConnectorConfirm?.name || 'this connection'}</strong>?
            This will also remove it from all agent profiles.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConnectorConfirm(null)} disabled={deleteConnectorMutation.isLoading}>
            Cancel
          </Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => deleteConnectorConfirm?.id && deleteConnectorMutation.mutate(deleteConnectorConfirm.id)}
            disabled={deleteConnectorMutation.isLoading}
          >
            {deleteConnectorMutation.isLoading ? 'Deleting…' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={!!deleteLineConfirm} onClose={() => !deleteLineMutation.isLoading && setDeleteLineConfirm(null)}>
        <DialogTitle>Delete line</DialogTitle>
        <DialogContent>
          <Typography>
            Permanently delete line <strong>{deleteLineConfirm?.name || 'this line'}</strong>?
            This removes members, goals, tasks, and timeline data for this line.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteLineConfirm(null)} disabled={deleteLineMutation.isLoading}>
            Cancel
          </Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => deleteLineConfirm?.id && deleteLineMutation.mutate(deleteLineConfirm.id)}
            disabled={deleteLineMutation.isLoading}
          >
            {deleteLineMutation.isLoading ? 'Deleting…' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Create playbook dialog */}
      <Dialog open={createPlaybookOpen} onClose={() => setCreatePlaybookOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>Create playbook</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Name"
            fullWidth
            value={newPlaybookName}
            onChange={(e) => setNewPlaybookName(e.target.value)}
            placeholder="My playbook"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreatePlaybookOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleCreatePlaybook}
            disabled={!newPlaybookName.trim() || createPlaybookMutation.isLoading}
          >
            Create
          </Button>
        </DialogActions>
      </Dialog>

      {/* Add category dialog */}
      <Dialog open={addCategoryOpen} onClose={() => !createSidebarCategoryMutation.isLoading && setAddCategoryOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>Add category</DialogTitle>
        <DialogContent>
          <FormControl fullWidth margin="dense">
            <InputLabel>Section</InputLabel>
            <Select
              value={addCategorySection}
              label="Section"
              onChange={(e) => setAddCategorySection(e.target.value)}
              disabled={createSidebarCategoryMutation.isLoading}
            >
              <MenuItem value="agents">Agents</MenuItem>
              <MenuItem value="playbooks">Playbooks</MenuItem>
              <MenuItem value="skills">Skills</MenuItem>
              <MenuItem value="connectors">Data Connections</MenuItem>
            </Select>
          </FormControl>
          <TextField
            autoFocus
            margin="dense"
            label="Category name"
            fullWidth
            value={newCategoryName}
            onChange={(e) => setNewCategoryName(e.target.value)}
            placeholder="e.g. Research"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddCategoryOpen(false)} disabled={createSidebarCategoryMutation.isLoading}>Cancel</Button>
          <Button
            variant="contained"
            onClick={() => {
              if (newCategoryName.trim()) createSidebarCategoryMutation.mutate({ section: addCategorySection, name: newCategoryName.trim() });
            }}
            disabled={!newCategoryName.trim() || createSidebarCategoryMutation.isLoading}
          >
            Add
          </Button>
        </DialogActions>
      </Dialog>

      {/* Add from template dialog */}
      <Dialog open={templateDialogOpen} onClose={() => !createConnectorFromTemplateMutation.isLoading && setTemplateDialogOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>Add from template</DialogTitle>
        <DialogContent>
          <FormControl fullWidth margin="dense">
            <InputLabel>Template</InputLabel>
            <Select
              value={selectedTemplate}
              label="Template"
              onChange={(e) => setSelectedTemplate(e.target.value)}
              disabled={templatesLoading}
            >
              <MenuItem value="">—</MenuItem>
              {(connectorTemplates || []).map((t) => (
                <MenuItem key={t.name || t.id} value={t.name || ''}>
                  {t.name || 'Unnamed'}
                  {t.description ? ` — ${t.description}` : ''}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTemplateDialogOpen(false)} disabled={createConnectorFromTemplateMutation.isLoading}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={() => selectedTemplate && createConnectorFromTemplateMutation.mutate({ template_name: selectedTemplate })}
            disabled={!selectedTemplate || createConnectorFromTemplateMutation.isLoading}
          >
            {createConnectorFromTemplateMutation.isLoading ? 'Creating…' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>

    </Box>
  );
}
