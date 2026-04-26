/**
 * Standalone playbook editor: name, description, workflow steps.
 * Shows usage warning when the playbook is used by agents.
 */

import React, { useState, useCallback, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  TextField,
  Button,
  Paper,
  FormControlLabel,
  Checkbox,
  Switch,
  IconButton,
  Tooltip,
  Drawer,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
} from '@mui/material';
import { Add, PlayArrow, History, Restore, Lock, Download, ContentCopy } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';
import agentFactoryService from '../../services/agentFactoryService';
import UsageWarningBanner from './UsageWarningBanner';
import WorkflowSection from './WorkflowSection';
import AddStepPanel from './AddStepPanel';
import StepConfigDrawer from './StepConfigDrawer';
import BrowserAuthCapture from './BrowserAuthCapture';
import {
  SESSION_STORAGE_CLIPBOARD_KEY,
  CLIPBOARD_MIME,
  buildClipboardPayload,
  isValidClipboardPayload,
  preparePasteSteps,
  hasDuplicateWireKeysInSteps,
  deepCloneSteps,
} from '../../utils/playbookStepTransfer';

function stepTreeHasSkillRefs(steps) {
  let found = false;
  const visit = (n) => {
    if (!n || typeof n !== 'object') return;
    const sk = n.skill_ids || n.skills;
    if (Array.isArray(sk) && sk.length) found = true;
    for (const k of ['steps', 'parallel_steps', 'then_steps', 'else_steps']) {
      if (Array.isArray(n[k])) n[k].forEach(visit);
    }
  };
  if (Array.isArray(steps)) steps.forEach(visit);
  return found;
}

async function readStepsClipboardPayload() {
  try {
    const raw = sessionStorage.getItem(SESSION_STORAGE_CLIPBOARD_KEY);
    if (raw) {
      const p = JSON.parse(raw);
      if (isValidClipboardPayload(p)) return p;
    }
  } catch {
    /* ignore */
  }
  if (typeof navigator !== 'undefined' && navigator.clipboard?.readText) {
    try {
      const text = await navigator.clipboard.readText();
      const p = JSON.parse(text);
      if (isValidClipboardPayload(p)) return p;
    } catch {
      /* ignore */
    }
  }
  return null;
}

async function writeStepsClipboardPayload(payload) {
  const json = JSON.stringify(payload);
  try {
    sessionStorage.setItem(SESSION_STORAGE_CLIPBOARD_KEY, json);
  } catch {
    /* ignore */
  }
  if (typeof navigator !== 'undefined' && navigator.clipboard?.write) {
    try {
      await navigator.clipboard.write([
        new ClipboardItem({
          [CLIPBOARD_MIME]: new Blob([json], { type: CLIPBOARD_MIME }),
          'text/plain': new Blob([json], { type: 'text/plain' }),
        }),
      ]);
      return;
    } catch {
      /* fall through */
    }
  }
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(json);
  }
}

/** Matches orchestrator engines/playbook_limits.MAX_PARALLEL_SUBSTEPS */
const MAX_PARALLEL_SUBSTEPS = 10;

function sanitizeDefinition(def) {
  if (!def || typeof def !== 'object')
    return { steps: [], run_context: 'interactive' };
  const STEP_KEYS = [
    '_step_id',
    'step_type', 'action', 'output_key', 'inputs', 'params',
    'prompt_template', 'prompt', 'model_override', 'output_schema',
    'timeout_minutes', 'on_reject', 'preview_from',
    'output_destination', 'max_iterations', 'steps', 'name',
    'available_tools', 'tool_packs', 'heading_level',  // kept for backward compat shim
    'condition', 'parallel_steps', 'branch_condition', 'then_steps', 'else_steps',
    'skill_ids', 'skills',
    'connection_policy', 'restricted_connections', 'discovery_mode',
    'auto_discover_skills', 'max_auto_skills',
    'skill_discovery_mode', 'max_discovered_skills',
    'dynamic_tool_discovery', 'max_skill_acquisitions',
    'site_domain', 'login_url', 'verify_url', 'verify_selector',
    'phases', 'search_tools', 'strategy', 'criteria', 'pass_threshold',
    'on_pass', 'on_fail', 'max_retries', 'target', 'next',
    'subagents', 'delegation_mode',
    'samples', 'selection_strategy', 'selection_criteria',
    'fan_out',
    'user_facts_policy',
    'persona_policy',
    'agent_memory_policy',
    'history_policy',
    'exclusive',
  ];
  const PHASE_KEYS = [
    'name', 'type', 'prompt', 'search_tools', 'available_tools', 'strategy',
    'criteria', 'pass_threshold', 'on_pass', 'on_fail', 'max_retries', 'target', 'next',
  ];
  function sanitizePhase(p) {
    if (!p || typeof p !== 'object') return p;
    const clean = {};
    PHASE_KEYS.forEach((k) => { if (p[k] !== undefined) clean[k] = p[k]; });
    return clean;
  }
  function sanitizeStep(s) {
    if (!s || typeof s !== 'object') return s;
    const clean = {};
    STEP_KEYS.forEach((k) => { if (s[k] !== undefined) clean[k] = s[k]; });
    if (Array.isArray(s.steps)) {
      clean.steps = s.steps.map(sanitizeStep);
    }
    if (Array.isArray(s.parallel_steps)) {
      clean.parallel_steps = s.parallel_steps.slice(0, MAX_PARALLEL_SUBSTEPS).map(sanitizeStep);
    }
    if (Array.isArray(s.then_steps)) {
      clean.then_steps = s.then_steps.map(sanitizeStep);
    }
    if (Array.isArray(s.else_steps)) {
      clean.else_steps = s.else_steps.map(sanitizeStep);
    }
    if (Array.isArray(s.phases)) {
      clean.phases = s.phases.map(sanitizePhase);
    }
    return clean;
  }
  const steps = Array.isArray(def.steps)
    ? def.steps.map(sanitizeStep)
    : [];
  return {
    steps,
    run_context: def.run_context || 'interactive',
  };
}

export default function PlaybookEditor({ playbookId }) {
  const [searchParams] = useSearchParams();
  const profileIdFromUrl = searchParams.get('profile_id') || null;
  const queryClient = useQueryClient();
  const [localPlaybook, setLocalPlaybook] = useState(null);
  const [addStepOpen, setAddStepOpen] = useState(false);
  const [newStepType, setNewStepType] = useState('tool');
  const [newStepAction, setNewStepAction] = useState('');
  const [newStepOutputKey, setNewStepOutputKey] = useState('');
  const [newStepAvailableTools, setNewStepAvailableTools] = useState([]);
  const [configDrawerOpen, setConfigDrawerOpen] = useState(false);
  const [selectedStepIndex, setSelectedStepIndex] = useState(null);
  const [selectedStep, setSelectedStep] = useState(null);
  const [selectedStepPath, setSelectedStepPath] = useState(null);
  const [testQuery, setTestQuery] = useState('');
  const [testResult, setTestResult] = useState('');
  const [testLoading, setTestLoading] = useState(false);
  const [testPersistConversation, setTestPersistConversation] = useState(true);
  const [pendingAuth, setPendingAuth] = useState(null);
  const [testConversationId, setTestConversationId] = useState(null);
  const [versionDrawerOpen, setVersionDrawerOpen] = useState(false);
  const [restoreVersionConfirm, setRestoreVersionConfirm] = useState(null);
  const [exportLoading, setExportLoading] = useState(false);
  const [pastePreviewDialog, setPastePreviewDialog] = useState(null);
  const [pasteUnsafeConfirm, setPasteUnsafeConfirm] = useState(false);
  const [snackbar, setSnackbar] = useState(null);
  const [selectionResetSignal, setSelectionResetSignal] = useState(0);

  const { data: playbook, isLoading: playbookLoading, error: playbookError } = useQuery(
    ['agentFactoryPlaybook', playbookId],
    () => apiService.agentFactory.getPlaybook(playbookId),
    { enabled: !!playbookId, retry: false }
  );
  const { data: usageAgents = [] } = useQuery(
    ['agentFactoryPlaybookUsage', playbookId],
    () => apiService.agentFactory.getPlaybookUsage(playbookId),
    { enabled: !!playbookId, staleTime: 30_000, retry: false }
  );
  const { data: versions = [] } = useQuery(
    ['agentFactoryPlaybookVersions', playbookId],
    () => apiService.agentFactory.listPlaybookVersions(playbookId),
    { enabled: !!playbookId && versionDrawerOpen, retry: false }
  );

  const effectiveProfileId = profileIdFromUrl || (usageAgents?.[0]?.id ?? null);
  const { data: actions = [] } = useQuery(
    ['agentFactoryActions', effectiveProfileId],
    () => apiService.agentFactory.getActions(effectiveProfileId),
    { staleTime: 60_000, retry: false }
  );

  const actionList = Array.isArray(actions) ? actions : [];
  const currentPlaybook = localPlaybook ?? playbook;

  const { data: sharedWithMe = [] } = useQuery(
    'agentFactorySharedWithMe',
    () => agentFactoryService.listSharedWithMe(),
    { staleTime: 30_000, retry: false }
  );
  const shareIdForPlaybook = (sharedWithMe || []).find(
    (s) => s.artifact_type === 'playbook' && s.artifact_id === playbookId
  )?.id;
  const copySharedToMineMutation = useMutation(
    (shareId) => agentFactoryService.copySharedToMine(shareId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('agentFactoryProfiles');
        queryClient.invalidateQueries('agentFactoryPlaybooks');
        queryClient.invalidateQueries('agentFactorySkills');
        queryClient.invalidateQueries('agentFactorySharedWithMe');
      },
    }
  );

  const updatePlaybookMutation = useMutation(
    ({ id, body }) => apiService.agentFactory.updatePlaybook(id, body),
    {
      onSuccess: (data, { id }) => {
        if (data?.id) {
          queryClient.setQueryData(['agentFactoryPlaybook', id], data);
          queryClient.setQueryData('agentFactoryPlaybooks', (old) => {
            if (!Array.isArray(old)) return old;
            return old.map((p) => (p.id === id ? { ...p, ...data } : p));
          });
        } else {
          queryClient.invalidateQueries(['agentFactoryPlaybook', id]);
          queryClient.invalidateQueries('agentFactoryPlaybooks');
        }
        setLocalPlaybook(null);
      },
    }
  );
  const restoreVersionMutation = useMutation(
    ({ playbookId: pid, versionId }) => apiService.agentFactory.restorePlaybookVersion(pid, versionId),
    {
      onSuccess: (data, { playbookId: pid }) => {
        if (data?.id && pid) queryClient.setQueryData(['agentFactoryPlaybook', pid], data);
        if (pid) queryClient.invalidateQueries(['agentFactoryPlaybook', pid]);
        if (pid) queryClient.invalidateQueries(['agentFactoryPlaybookVersions', pid]);
        setRestoreVersionConfirm(null);
        setVersionDrawerOpen(false);
        setLocalPlaybook(null);
      },
    }
  );

  useEffect(() => {
    setLocalPlaybook(null);
    setAddStepOpen(false);
    setConfigDrawerOpen(false);
    setSelectedStepPath(null);
  }, [playbookId]);

  const handlePlaybookChange = useCallback(
    (next) => {
      setLocalPlaybook(next);
      if (playbookId && next?.definition != null) {
        const cleanDef = sanitizeDefinition(next.definition);
        updatePlaybookMutation.mutate({
          id: playbookId,
          body: {
            definition: cleanDef,
            name: next.name ?? currentPlaybook?.name,
            description: next.description ?? currentPlaybook?.description,
          },
        });
      }
    },
    [playbookId, currentPlaybook, updatePlaybookMutation]
  );

  const applyPastedSteps = useCallback(
    (mergedSteps) => {
      const base = localPlaybook ?? playbook;
      if (!base) return;
      handlePlaybookChange({
        ...base,
        definition: { ...base.definition, steps: mergedSteps },
      });
      setSelectionResetSignal((s) => s + 1);
    },
    [localPlaybook, playbook, handlePlaybookChange]
  );

  const handleCopySteps = useCallback(async (stepsSlice) => {
    if (!stepsSlice?.length) return;
    if (hasDuplicateWireKeysInSteps(stepsSlice)) {
      setSnackbar('Cannot copy: duplicate wire keys within selection');
      return;
    }
    const payload = buildClipboardPayload(deepCloneSteps(stepsSlice));
    try {
      await writeStepsClipboardPayload(payload);
    } catch {
      setSnackbar('Could not write clipboard');
      return;
    }
    setSnackbar(`Copied ${stepsSlice.length} step(s)`);
  }, []);

  const handlePasteRequest = useCallback(
    async (insertIndex) => {
      const base = localPlaybook ?? playbook;
      if (!base?.definition) return;
      const payload = await readStepsClipboardPayload();
      if (!payload) {
        setSnackbar('No playbook steps in clipboard');
        return;
      }
      const targetSteps = base.definition.steps ?? [];
      const result = preparePasteSteps({
        clipboardSteps: payload.steps,
        targetTopLevelSteps: targetSteps,
        insertIndex,
      });
      const hasRemap = Object.keys(result.remap).length > 0;
      const hasIssues = result.referenceIssues.length > 0;
      if (hasRemap || hasIssues) {
        setPasteUnsafeConfirm(false);
        setPastePreviewDialog({
          mergedSteps: result.mergedSteps,
          remap: result.remap,
          referenceIssues: result.referenceIssues,
          insertIndex,
          stepCount: payload.steps.length,
          hasSkillRefs: stepTreeHasSkillRefs(payload.steps),
        });
        return;
      }
      applyPastedSteps(result.mergedSteps);
      setSnackbar('Pasted steps');
    },
    [localPlaybook, playbook, applyPastedSteps]
  );

  const handleConfirmPastePreview = useCallback(() => {
    if (!pastePreviewDialog) return;
    const { mergedSteps, referenceIssues } = pastePreviewDialog;
    if (referenceIssues.length > 0 && !pasteUnsafeConfirm) return;
    applyPastedSteps(mergedSteps);
    setPastePreviewDialog(null);
    setPasteUnsafeConfirm(false);
    setSnackbar(referenceIssues.length > 0 ? 'Pasted steps (references may need fixes)' : 'Pasted steps');
  }, [pastePreviewDialog, pasteUnsafeConfirm, applyPastedSteps]);

  const handleNameBlur = useCallback(() => {
    if (!playbookId || !currentPlaybook) return;
    const name = (localPlaybook ?? currentPlaybook).name ?? currentPlaybook.name;
    if (name === currentPlaybook.name) return;
    updatePlaybookMutation.mutate({
      id: playbookId,
      body: { name: name || currentPlaybook.name },
    });
    setLocalPlaybook(null);
  }, [playbookId, currentPlaybook, localPlaybook, updatePlaybookMutation]);

  const handleDescriptionBlur = useCallback(() => {
    if (!playbookId || !currentPlaybook) return;
    const description = (localPlaybook ?? currentPlaybook).description ?? currentPlaybook.description;
    if (description === currentPlaybook.description) return;
    updatePlaybookMutation.mutate({
      id: playbookId,
      body: { description: description ?? currentPlaybook.description },
    });
    setLocalPlaybook(null);
  }, [playbookId, currentPlaybook, localPlaybook, updatePlaybookMutation]);

  const handleAddStep = useCallback(() => {
    setNewStepType('tool');
    setNewStepAction('');
    setNewStepOutputKey('');
    setNewStepAvailableTools([]);
    setAddStepOpen(true);
  }, []);

  const handleStepClick = useCallback((idx, step, meta) => {
    if (currentPlaybook?.is_locked) return;
    setSelectedStepIndex(idx);
    setSelectedStep(step);
    setSelectedStepPath(
      meta?.parentLoopIndex != null
        ? { parentLoopIndex: meta.parentLoopIndex }
        : meta?.parentParallelIndex != null
          ? { parentParallelIndex: meta.parentParallelIndex }
          : meta?.parentBranchIndex != null
            ? { parentBranchIndex: meta.parentBranchIndex, branchPath: meta.branchPath }
            : null
    );
    setConfigDrawerOpen(true);
  }, [currentPlaybook?.is_locked]);

  const handleStepConfigSave = useCallback(
    (idx, updatedStep, path) => {
      if (!currentPlaybook?.definition?.steps) return;
      const steps = [...currentPlaybook.definition.steps];
      if (path?.parentLoopIndex != null) {
        const parent = steps[path.parentLoopIndex];
        if (parent?.step_type === 'loop' && Array.isArray(parent.steps)) {
          const childSteps = [...parent.steps];
          childSteps[idx] = updatedStep;
          steps[path.parentLoopIndex] = { ...parent, steps: childSteps };
        }
      } else if (path?.parentParallelIndex != null) {
        const parent = steps[path.parentParallelIndex];
        if (parent?.step_type === 'parallel' && Array.isArray(parent.parallel_steps)) {
          const childSteps = [...parent.parallel_steps];
          childSteps[idx] = updatedStep;
          steps[path.parentParallelIndex] = { ...parent, parallel_steps: childSteps };
        }
      } else if (path?.parentBranchIndex != null && path?.branchPath) {
        const parent = steps[path.parentBranchIndex];
        if (parent?.step_type === 'branch') {
          const pathKey = path.branchPath === 'then' ? 'then_steps' : 'else_steps';
          const childSteps = [...(parent[pathKey] || [])];
          childSteps[idx] = updatedStep;
          steps[path.parentBranchIndex] = { ...parent, [pathKey]: childSteps };
        }
      } else {
        steps[idx] = updatedStep;
      }
      handlePlaybookChange({
        ...currentPlaybook,
        definition: { ...currentPlaybook.definition, steps },
      });
      setConfigDrawerOpen(false);
      setSelectedStepPath(null);
    },
    [currentPlaybook, handlePlaybookChange]
  );

  const handleAddParallelChild = useCallback(
    (parallelIndex) => {
      if (!currentPlaybook?.definition?.steps) return;
      const steps = [...currentPlaybook.definition.steps];
      const parent = steps[parallelIndex];
      if (parent?.step_type !== 'parallel') return;
      const parallelSteps = [...(parent.parallel_steps || [])];
      if (parallelSteps.length >= MAX_PARALLEL_SUBSTEPS) return;
      const newChild = { _step_id: crypto.randomUUID(), step_type: 'tool', action: '', output_key: '', name: '', inputs: {} };
      parallelSteps.push(newChild);
      const childIndex = parallelSteps.length - 1;
      steps[parallelIndex] = { ...parent, parallel_steps: parallelSteps };
      handlePlaybookChange({
        ...currentPlaybook,
        definition: { ...currentPlaybook.definition, steps },
      });
      if (!currentPlaybook?.is_locked) {
        setSelectedStepIndex(childIndex);
        setSelectedStep(newChild);
        setSelectedStepPath({ parentParallelIndex: parallelIndex });
        setConfigDrawerOpen(true);
      }
    },
    [currentPlaybook, handlePlaybookChange]
  );

  const handleGroupAsParallel = useCallback(
    (sortedIndices) => {
      if (!currentPlaybook?.definition?.steps || sortedIndices.length < 2) return;
      if (sortedIndices.length > MAX_PARALLEL_SUBSTEPS) return;
      const steps = [...currentPlaybook.definition.steps];
      const extracted = sortedIndices.map((i) => steps[i]).filter(Boolean);
      const firstIdx = sortedIndices[0];
      const parallelStep = {
        _step_id: crypto.randomUUID(),
        step_type: 'parallel',
        parallel_steps: extracted,
        output_key: `parallel_${firstIdx + 1}`,
        name: `parallel_${firstIdx + 1}`,
      };
      const next = steps.filter((_, i) => !sortedIndices.includes(i));
      next.splice(firstIdx, 0, parallelStep);
      handlePlaybookChange({
        ...currentPlaybook,
        definition: { ...currentPlaybook.definition, steps: next },
      });
    },
    [currentPlaybook, handlePlaybookChange]
  );

  const handleUngroupParallel = useCallback(
    (parallelIndex) => {
      if (!currentPlaybook?.definition?.steps) return;
      const steps = [...currentPlaybook.definition.steps];
      const par = steps[parallelIndex];
      if (par?.step_type !== 'parallel' || !Array.isArray(par.parallel_steps) || par.parallel_steps.length === 0) return;
      const next = [...steps.slice(0, parallelIndex), ...par.parallel_steps, ...steps.slice(parallelIndex + 1)];
      handlePlaybookChange({
        ...currentPlaybook,
        definition: { ...currentPlaybook.definition, steps: next },
      });
    },
    [currentPlaybook, handlePlaybookChange]
  );

  const handleAddBranchChild = useCallback(
    (branchIndex, path) => {
      if (!currentPlaybook?.definition?.steps) return;
      const steps = [...currentPlaybook.definition.steps];
      const parent = steps[branchIndex];
      if (parent?.step_type !== 'branch') return;
      const pathKey = path === 'then' ? 'then_steps' : 'else_steps';
      const pathSteps = [...(parent[pathKey] || [])];
      const newChild = { _step_id: crypto.randomUUID(), step_type: 'tool', action: '', output_key: '', name: '', inputs: {} };
      pathSteps.push(newChild);
      const childIndex = pathSteps.length - 1;
      steps[branchIndex] = { ...parent, [pathKey]: pathSteps };
      handlePlaybookChange({
        ...currentPlaybook,
        definition: { ...currentPlaybook.definition, steps },
      });
      if (!currentPlaybook?.is_locked) {
        setSelectedStepIndex(childIndex);
        setSelectedStep(newChild);
        setSelectedStepPath({ parentBranchIndex: branchIndex, branchPath: path });
        setConfigDrawerOpen(true);
      }
    },
    [currentPlaybook, handlePlaybookChange]
  );

  const handleReorder = useCallback(
    (fromIdx, toIdx) => {
      if (!currentPlaybook?.definition?.steps) return;
      const steps = [...currentPlaybook.definition.steps];
      const [removed] = steps.splice(fromIdx, 1);
      steps.splice(toIdx, 0, removed);
      handlePlaybookChange({
        ...currentPlaybook,
        definition: { ...currentPlaybook.definition, steps },
      });
    },
    [currentPlaybook, handlePlaybookChange]
  );

  const handleReorderParallelChild = useCallback(
    (parallelIndex, fromIdx, toIdx) => {
      if (!currentPlaybook?.definition?.steps) return;
      const steps = [...currentPlaybook.definition.steps];
      const parent = steps[parallelIndex];
      if (parent?.step_type !== 'parallel') return;
      const children = [...(parent.parallel_steps || [])];
      const [removed] = children.splice(fromIdx, 1);
      children.splice(toIdx, 0, removed);
      steps[parallelIndex] = { ...parent, parallel_steps: children };
      handlePlaybookChange({
        ...currentPlaybook,
        definition: { ...currentPlaybook.definition, steps },
      });
    },
    [currentPlaybook, handlePlaybookChange]
  );

  const handleRemoveParallelChild = useCallback(
    (parallelIndex, childIdx) => {
      if (!currentPlaybook?.definition?.steps) return;
      const steps = [...currentPlaybook.definition.steps];
      const parent = steps[parallelIndex];
      if (parent?.step_type !== 'parallel') return;
      const children = [...(parent.parallel_steps || [])].filter((_, i) => i !== childIdx);
      steps[parallelIndex] = { ...parent, parallel_steps: children };
      handlePlaybookChange({
        ...currentPlaybook,
        definition: { ...currentPlaybook.definition, steps },
      });
    },
    [currentPlaybook, handlePlaybookChange]
  );

  const handleAddStepSave = useCallback(() => {
    if (!currentPlaybook) return;
    const steps = [...(currentPlaybook.definition?.steps ?? [])];
    const outputKey = newStepOutputKey.trim() || `step_${steps.length + 1}`;
    let newStep;
    if (newStepType === 'tool') {
      if (!newStepAction) return;
      newStep = { step_type: 'tool', action: newStepAction, output_key: outputKey, name: outputKey, inputs: {} };
    } else if (newStepType === 'llm_task') {
      newStep = {
        step_type: 'llm_task',
        action: newStepAction || 'llm_task',
        output_key: outputKey,
        name: outputKey,
        prompt_template: 'Summarize or analyze the following:\n\n{context}',
        inputs: {},
      };
    } else if (newStepType === 'llm_agent') {
      newStep = {
        step_type: 'llm_agent',
        action: 'llm_agent',
        output_key: outputKey,
        name: outputKey,
        prompt_template: 'Use the available tools to: {query}',
        skill_ids: [],
        max_iterations: 3,
        inputs: {},
      };
    } else if (newStepType === 'deep_agent') {
      newStep = {
        step_type: 'deep_agent',
        output_key: outputKey,
        name: outputKey,
        phases: [],
        inputs: {},
      };
    } else if (newStepType === 'parallel') {
      newStep = { step_type: 'parallel', parallel_steps: [], output_key: outputKey, name: outputKey };
    } else if (newStepType === 'branch') {
      newStep = { step_type: 'branch', branch_condition: '', then_steps: [], else_steps: [], output_key: outputKey, name: outputKey };
    } else if (newStepType === 'browser_authenticate') {
      newStep = {
        step_type: 'browser_authenticate',
        name: outputKey,
        output_key: outputKey,
        site_domain: '',
        login_url: '',
        verify_url: '',
        verify_selector: '',
      };
    } else {
      newStep = {
        step_type: 'approval',
        output_key: outputKey,
        name: outputKey,
        prompt: 'Approve to continue?',
        timeout_minutes: 30,
        on_reject: 'stop',
      };
    }
    newStep._step_id = crypto.randomUUID();
    steps.push(newStep);
    handlePlaybookChange({
      ...currentPlaybook,
      definition: { ...currentPlaybook.definition, steps },
    });
    setAddStepOpen(false);
    if ((newStepType === 'parallel' || newStepType === 'branch' || newStepType === 'deep_agent') && !currentPlaybook?.is_locked) {
      setSelectedStepIndex(steps.length - 1);
      setSelectedStep(newStep);
      setConfigDrawerOpen(true);
    }
  }, [newStepType, newStepAction, newStepOutputKey, newStepAvailableTools, currentPlaybook, handlePlaybookChange]);

  const runTest = useCallback(async (overrideConversationId = null, overrideQuery = null) => {
    if (!effectiveProfileId) return;
    // If runTest is used directly as an onClick handler, React will pass a SyntheticEvent.
    // Guard against accidentally serializing DOM/event objects into the request payload.
    if (overrideConversationId && typeof overrideConversationId === 'object') {
      overrideConversationId = null;
    }
    if (overrideQuery && typeof overrideQuery === 'object') {
      overrideQuery = null;
    }
    setTestLoading(true);
    setTestResult('');
    setPendingAuth(null);
    const token = apiService.getToken?.();
    const conversationId = overrideConversationId || `agent-run-${crypto.randomUUID?.() ?? Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    if (!overrideConversationId) setTestConversationId(conversationId);
    const query = overrideQuery !== null ? overrideQuery : testQuery;
    (async () => {
      try {
        const response = await fetch('/api/async/orchestrator/stream', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
          body: JSON.stringify({
            query,
            conversation_id: conversationId,
            session_id: 'playbook-test',
            agent_profile_id: effectiveProfileId,
            persist_conversation: testPersistConversation,
          }),
        });
        if (!response.ok) {
          setTestResult(`HTTP ${response.status}: ${await response.text()}`);
          setTestLoading(false);
          return;
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let full = '';
        let hadPendingAuth = false;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.type === 'content' && data.content) full += data.content;
                if (data.type === 'complete' && data.content) full += data.content;
                if (data.type === 'permission_request' && data.pending_auth) {
                  hadPendingAuth = true;
                  setPendingAuth(data.pending_auth);
                  setTestResult(full || 'Authentication required. Log in in the browser panel below, then click "I\'m Logged In".');
                }
              } catch (_) {}
            }
          }
        }
        if (!hadPendingAuth) setTestResult(full || '(No content in stream)');
      } catch (e) {
        setTestResult('Error: ' + (e.message || String(e)));
      }
      setTestLoading(false);
    })();
  }, [effectiveProfileId, testQuery]);

  const handleBrowserAuthComplete = useCallback(() => {
    setPendingAuth(null);
    runTest(testConversationId, 'yes');
  }, [runTest, testConversationId]);

  const handleBrowserAuthCancel = useCallback(() => {
    setPendingAuth(null);
    setTestResult('Authentication cancelled.');
  }, []);

  if (!playbookId) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography color="text.secondary">
          Select a playbook from the sidebar or create one to edit.
        </Typography>
      </Box>
    );
  }

  if (playbookLoading || playbookError) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
        {playbookError && (
          <Alert severity="error">
            Failed to load playbook. It may have been deleted.
          </Alert>
        )}
        {playbookLoading && <CircularProgress />}
      </Box>
    );
  }

  const displayPlaybook = currentPlaybook || { name: '', description: '', definition: { steps: [] } };
  const isShared = displayPlaybook.ownership === 'shared';
  const isLocked = !!displayPlaybook.is_locked || !!displayPlaybook.is_builtin || !!displayPlaybook.is_template || isShared;

  const handleLockToggle = (e) => {
    const locked = e.target.checked;
    updatePlaybookMutation.mutate({ id: playbookId, body: { is_locked: locked } });
    setLocalPlaybook((prev) => (prev ? { ...prev, is_locked: locked } : null));
  };

  const handleExport = async () => {
    if (!playbookId || exportLoading) return;
    setExportLoading(true);
    try {
      const playbookData = await apiService.agentFactory.exportPlaybook(playbookId);
      const jsonString = JSON.stringify(playbookData, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const playbookName = displayPlaybook.name || 'playbook';
      const sanitizedName = playbookName.replace(/[^a-z0-9]/gi, '-').toLowerCase();
      a.download = `${sanitizedName}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export playbook failed:', err);
    } finally {
      setExportLoading(false);
    }
  };

  const connectorContextAgent = effectiveProfileId && !profileIdFromUrl ? usageAgents?.find((a) => String(a.id) === String(effectiveProfileId)) : null;

  return (
    <Box sx={{ p: 2, overflow: 'auto', maxWidth: 720, flex: 1, minHeight: 0 }}>
      {isShared && (
        <Alert
          severity="info"
          sx={{ mb: 2 }}
          action={
            shareIdForPlaybook && (
              <Button
                color="inherit"
                size="small"
                startIcon={<ContentCopy />}
                onClick={() => copySharedToMineMutation.mutate(shareIdForPlaybook)}
                disabled={copySharedToMineMutation.isLoading}
              >
                {copySharedToMineMutation.isLoading ? 'Copying…' : 'Make my own copy'}
              </Button>
            )
          }
        >
          This playbook is shared by {displayPlaybook.owner_display_name || displayPlaybook.owner_username || 'another user'}. You can use it as-is or make your own copy to customize.
        </Alert>
      )}
      <UsageWarningBanner resourceLabel="This playbook" agents={usageAgents} />
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2, flexWrap: 'wrap', gap: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Tooltip title="Version history">
            <IconButton
              onClick={() => setVersionDrawerOpen(true)}
              aria-label="Version history"
              disabled={!playbookId}
            >
              <History />
            </IconButton>
          </Tooltip>
          <Button
            size="small"
            variant="outlined"
            startIcon={<Download />}
            onClick={handleExport}
            disabled={exportLoading || !playbookId}
          >
            Export
          </Button>
        </Box>
        <FormControlLabel
          control={
            <Switch
              checked={isLocked}
              onChange={handleLockToggle}
              disabled={updatePlaybookMutation.isLoading || !!displayPlaybook.is_builtin || !!displayPlaybook.is_template}
              color="primary"
            />
          }
          label={displayPlaybook.is_builtin ? 'Built-in' : displayPlaybook.is_template ? 'Template' : (isLocked ? 'Locked' : 'Unlocked')}
          labelPlacement="start"
        />
      </Box>
      <Drawer
        anchor="right"
        open={versionDrawerOpen}
        onClose={() => setVersionDrawerOpen(false)}
        PaperProps={{ sx: { width: 320, maxWidth: '100%' } }}
      >
        <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Typography variant="h6">Version history</Typography>
          <Typography variant="caption" color="text.secondary">Restore a previous definition. Current version is saved first.</Typography>
        </Box>
        <List dense>
          {versions.length === 0 && !playbookId && null}
          {versions.length === 0 && playbookId && (
            <ListItem><ListItemText primary="No versions yet" secondary="Versions are created when you edit the playbook." /></ListItem>
          )}
          {versions.map((v) => (
            <ListItem key={v.id} disablePadding>
              <ListItemButton
                onClick={() => setRestoreVersionConfirm(v)}
              >
                <ListItemText
                  primary={`Version ${v.version_number}${v.label ? ` — ${v.label}` : ''}`}
                  secondary={v.created_at ? new Date(v.created_at).toLocaleString() : null}
                />
                <Restore fontSize="small" sx={{ color: 'text.secondary' }} />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Drawer>
      <Dialog open={!!restoreVersionConfirm} onClose={() => !restoreVersionMutation.isLoading && setRestoreVersionConfirm(null)}>
        <DialogTitle>Restore version</DialogTitle>
        <DialogContent>
          <Typography>
            Restore playbook to version <strong>{restoreVersionConfirm?.version_number}</strong>? Current definition will be saved as a new version first.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRestoreVersionConfirm(null)} disabled={restoreVersionMutation.isLoading}>Cancel</Button>
          <Button
            variant="contained"
            onClick={() => restoreVersionConfirm && playbookId && restoreVersionMutation.mutate({ playbookId, versionId: restoreVersionConfirm.id })}
            disabled={restoreVersionMutation.isLoading}
          >
            {restoreVersionMutation.isLoading ? 'Restoring…' : 'Restore'}
          </Button>
        </DialogActions>
      </Dialog>
      <Dialog
        open={!!pastePreviewDialog}
        onClose={() => {
          setPastePreviewDialog(null);
          setPasteUnsafeConfirm(false);
        }}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Paste steps</DialogTitle>
        <DialogContent>
          {pastePreviewDialog && (
            <>
              <Typography variant="body2" sx={{ mb: 1 }}>
                Insert {pastePreviewDialog.stepCount} step(s) at position {pastePreviewDialog.insertIndex + 1} (0-based index {pastePreviewDialog.insertIndex}).
              </Typography>
              {pastePreviewDialog.hasSkillRefs && (
                <Alert severity="info" sx={{ mb: 2 }}>
                  Copied steps reference skills. Ensure this agent profile has the same skills assigned if needed.
                </Alert>
              )}
              {Object.keys(pastePreviewDialog.remap).length > 0 && (
                <>
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>Renamed wire keys (avoid collisions)</Typography>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Previous</TableCell>
                        <TableCell>New</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(pastePreviewDialog.remap).map(([from, to]) => (
                        <TableRow key={from}>
                          <TableCell>{from}</TableCell>
                          <TableCell>{to}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </>
              )}
              {pastePreviewDialog.referenceIssues.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Alert severity="warning" sx={{ mb: 1 }}>
                    Some references may not resolve at runtime until you fix wiring.
                  </Alert>
                  <Box component="ul" sx={{ m: 0, pl: 2, maxHeight: 200, overflow: 'auto' }}>
                    {pastePreviewDialog.referenceIssues.map((iss, i) => (
                      <li key={i}>
                        <Typography variant="caption" component="span">
                          {iss.path ? `${iss.path}: ` : ''}{iss.message}
                        </Typography>
                      </li>
                    ))}
                  </Box>
                  <FormControlLabel
                    control={(
                      <Checkbox
                        checked={pasteUnsafeConfirm}
                        onChange={(e) => setPasteUnsafeConfirm(e.target.checked)}
                      />
                    )}
                    label="Paste anyway"
                  />
                </Box>
              )}
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setPastePreviewDialog(null); setPasteUnsafeConfirm(false); }}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleConfirmPastePreview}
            disabled={!!pastePreviewDialog?.referenceIssues?.length && !pasteUnsafeConfirm}
          >
            Paste
          </Button>
        </DialogActions>
      </Dialog>
      <Snackbar
        open={!!snackbar}
        autoHideDuration={5000}
        onClose={() => setSnackbar(null)}
        message={snackbar}
      />
      {connectorContextAgent && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          Data connector actions from: <strong>{connectorContextAgent.name || connectorContextAgent.handle || 'Agent'}</strong>
        </Typography>
      )}

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <TextField
            fullWidth
            label="Name"
            value={(localPlaybook ?? displayPlaybook).name ?? ''}
            onChange={(e) => setLocalPlaybook((p) => ({ ...(p ?? displayPlaybook), name: e.target.value }))}
            onBlur={handleNameBlur}
            placeholder="Playbook name"
            sx={{ mb: 2 }}
            disabled={isLocked}
          />
          <TextField
            fullWidth
            label="Description"
            multiline
            minRows={2}
            value={(localPlaybook ?? displayPlaybook).description ?? ''}
            onChange={(e) => setLocalPlaybook((p) => ({ ...(p ?? displayPlaybook), description: e.target.value }))}
            onBlur={handleDescriptionBlur}
            placeholder="Optional description"
            disabled={isLocked}
          />
        </CardContent>
      </Card>

      <WorkflowSection
        key={playbookId || 'none'}
        playbook={displayPlaybook}
        actions={actionList}
        onPlaybookChange={handlePlaybookChange}
        onAddStep={handleAddStep}
        onStepClick={handleStepClick}
        onReorder={handleReorder}
        onAddParallelChild={handleAddParallelChild}
        onReorderParallelChild={handleReorderParallelChild}
        onRemoveParallelChild={handleRemoveParallelChild}
        onGroupAsParallel={handleGroupAsParallel}
        onUngroupParallel={handleUngroupParallel}
        onAddBranchChild={handleAddBranchChild}
        maxParallelSubsteps={MAX_PARALLEL_SUBSTEPS}
        addStepPanelOpen={addStepOpen}
        readOnly={isLocked}
        onCopySteps={isLocked ? undefined : handleCopySteps}
        onPasteRequest={isLocked ? undefined : handlePasteRequest}
        selectionResetSignal={selectionResetSignal}
      />
      {addStepOpen && (
        <AddStepPanel
          stepType={newStepType}
          setStepType={setNewStepType}
          action={newStepAction}
          setAction={setNewStepAction}
          outputKey={newStepOutputKey}
          setOutputKey={setNewStepOutputKey}
          actions={actionList}
          onAdd={handleAddStepSave}
          onCancel={() => setAddStepOpen(false)}
          addDisabled={newStepType === 'tool' ? !newStepAction : false}
          availableTools={newStepAvailableTools}
          setAvailableTools={setNewStepAvailableTools}
        />
      )}
      <StepConfigDrawer
        open={configDrawerOpen}
        onClose={() => { setConfigDrawerOpen(false); setSelectedStepPath(null); }}
        step={selectedStep}
        stepIndex={selectedStepIndex}
        steps={
          selectedStepPath?.parentLoopIndex != null && displayPlaybook?.definition?.steps?.[selectedStepPath.parentLoopIndex]
            ? (displayPlaybook.definition.steps[selectedStepPath.parentLoopIndex].steps ?? [])
            : selectedStepPath?.parentParallelIndex != null && displayPlaybook?.definition?.steps?.[selectedStepPath.parentParallelIndex]
              ? (displayPlaybook.definition.steps[selectedStepPath.parentParallelIndex].parallel_steps ?? [])
              : selectedStepPath?.parentBranchIndex != null && selectedStepPath?.branchPath && displayPlaybook?.definition?.steps?.[selectedStepPath.parentBranchIndex]
                ? (displayPlaybook.definition.steps[selectedStepPath.parentBranchIndex][selectedStepPath.branchPath === 'then' ? 'then_steps' : 'else_steps'] ?? [])
                : (displayPlaybook?.definition?.steps ?? [])
        }
        stepPath={selectedStepPath}
        actions={actionList}
        playbookInputs={displayPlaybook?.definition?.inputs ?? []}
        onSave={handleStepConfigSave}
        profileId={effectiveProfileId}
        readOnly={isLocked}
      />

      <Card variant="outlined" sx={{ mt: 2 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Test run
          </Typography>
          {effectiveProfileId ? (
            <>
              <TextField
                fullWidth
                multiline
                minRows={2}
                label="Query (optional)"
                value={testQuery}
                onChange={(e) => setTestQuery(e.target.value)}
                placeholder="Leave blank for tool-only runs; add text if steps use {query}"
                sx={{ mb: 2 }}
              />
              <Button
                variant="contained"
                startIcon={testLoading ? <CircularProgress size={20} /> : <PlayArrow />}
                disabled={testLoading}
                onClick={() => runTest()}
                sx={{ mb: 2 }}
              >
                Run
              </Button>
              <FormControlLabel
                control={(
                  <Checkbox
                    checked={testPersistConversation}
                    onChange={(e) => setTestPersistConversation(e.target.checked)}
                  />
                )}
                label="Save test run to chat (conversation history)"
                sx={{ mb: 2 }}
              />
              {testResult !== '' && (
                <Paper
                  variant="outlined"
                  sx={{
                    p: 2,
                    whiteSpace: 'pre-wrap',
                    fontFamily: 'monospace',
                    fontSize: '0.875rem',
                    maxHeight: 300,
                    overflow: 'auto',
                  }}
                >
                  {testResult}
                </Paper>
              )}
              {pendingAuth && (
                <Box sx={{ mt: 2 }}>
                  <BrowserAuthCapture
                    pendingAuth={pendingAuth}
                    onComplete={handleBrowserAuthComplete}
                    onCancel={handleBrowserAuthCancel}
                  />
                </Box>
              )}
            </>
          ) : (
            <Typography variant="body2" color="text.secondary">
              Assign this playbook to an agent to enable test runs.
            </Typography>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}
