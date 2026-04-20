/**
 * Agent Factory Playbook section: step card stack with drag-to-reorder.
 * Playbook validation: wiring issues plus exclusive/catch-all hints for conditional steps.
 */

import React, { useMemo, useState, useEffect } from 'react';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import {
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
  IconButton,
  Button,
  Box,
  Alert,
  Chip,
  Paper,
  Checkbox,
  Tooltip,
} from '@mui/material';
import { Add, Delete, DragIndicator, Build, Psychology, SmartToy, Gavel, Repeat, Send, ViewModule, CallSplit, Lock, AccountTree, ContentCopy, ContentPaste } from '@mui/icons-material';
import {
  validatePlaybookWiring,
  validateExclusiveConditions,
  indexActionsByName,
  getOutputFieldsForStep,
} from '../../utils/agentFactoryTypeWiring';
import { isContiguousSortedIndices } from '../../utils/playbookStepTransfer';

export default function WorkflowSection({
  playbook,
  actions = [],
  onPlaybookChange,
  onAddStep,
  onRemoveStep,
  onReorder,
  onStepClick,
  onAddParallelChild,
  onReorderParallelChild,
  onRemoveParallelChild,
  onGroupAsParallel,
  onUngroupParallel,
  onAddBranchChild,
  readOnly = false,
  addStepPanelOpen = false,
  onCopySteps,
  onPasteRequest,
  selectionResetSignal = 0,
  maxParallelSubsteps = 10,
}) {
  const steps = playbook?.definition?.steps ?? [];
  const [selectedIndices, setSelectedIndices] = useState([]);
  const actionsByName = useMemo(() => indexActionsByName(actions), [actions]);
  const wiringErrors = useMemo(() => {
    const wiring = validatePlaybookWiring(steps, actionsByName);
    const exclusive = validateExclusiveConditions(steps);
    return [...wiring, ...exclusive];
  }, [steps, actionsByName]);

  const sortedSelected = useMemo(() => [...selectedIndices].sort((a, b) => a - b), [selectedIndices]);
  const isAdjacent = sortedSelected.length < 2 || sortedSelected.every((v, i) => i === 0 || v === sortedSelected[i - 1] + 1);
  const copyContiguous = isContiguousSortedIndices(sortedSelected);
  const singleSelectedParallel = sortedSelected.length === 1 && (steps[sortedSelected[0]]?.step_type || steps[sortedSelected[0]]?.type) === 'parallel';

  const pasteInsertIndex = useMemo(() => {
    if (!selectedIndices.length) return steps.length;
    return Math.max(...selectedIndices) + 1;
  }, [selectedIndices, steps.length]);

  useEffect(() => {
    setSelectedIndices([]);
  }, [selectionResetSignal]);

  const handleToggleSelect = (idx) => {
    setSelectedIndices((prev) =>
      prev.includes(idx) ? prev.filter((i) => i !== idx) : [...prev, idx]
    );
  };

  const handleGroupAsParallel = () => {
    if (!isAdjacent || sortedSelected.length < 2 || !onGroupAsParallel) return;
    onGroupAsParallel(sortedSelected);
    setSelectedIndices([]);
  };

  const handleUngroupParallel = () => {
    if (!singleSelectedParallel || !onUngroupParallel) return;
    onUngroupParallel(sortedSelected[0]);
    setSelectedIndices([]);
  };

  const handleDragEnd = (result) => {
    const { source, destination } = result;
    if (!destination) return;
    if (source.droppableId === destination.droppableId && source.index === destination.index) return;

    if (source.droppableId === 'playbook-steps' && destination.droppableId === 'playbook-steps') {
      const fromIdx = source.index;
      const toIdx = destination.index;
      if (onReorder) {
        onReorder(fromIdx, toIdx);
        return;
      }
      const next = [...steps];
      const [removed] = next.splice(fromIdx, 1);
      next.splice(toIdx, 0, removed);
      onPlaybookChange({ ...playbook, definition: { ...playbook.definition, steps: next } });
      return;
    }

    if (
      source.droppableId.startsWith('parallel-')
      && source.droppableId === destination.droppableId
      && onReorderParallelChild
    ) {
      const parentIdx = parseInt(source.droppableId.replace('parallel-', ''), 10);
      if (!Number.isNaN(parentIdx)) {
        onReorderParallelChild(parentIdx, source.index, destination.index);
      }
    }
  };

  const handleRemove = (idx) => {
    if (onRemoveStep) {
      onRemoveStep(idx);
      return;
    }
    const next = [...steps];
    next.splice(idx, 1);
    onPlaybookChange({ ...playbook, definition: { ...playbook.definition, steps: next } });
  };

  const handleCopySelectedSteps = () => {
    if (!onCopySteps || !copyContiguous || sortedSelected.length === 0) return;
    const first = sortedSelected[0];
    const last = sortedSelected[sortedSelected.length - 1];
    onCopySteps(steps.slice(first, last + 1));
  };

  const handlePasteClick = () => {
    if (!onPasteRequest) return;
    onPasteRequest(pasteInsertIndex);
  };

  const stepTypeInfo = (step) => {
    const type = step.step_type || step.type || 'tool';
    const icons = { tool: Build, llm_task: Psychology, llm_agent: SmartToy, deep_agent: AccountTree, approval: Gavel, browser_authenticate: Lock, loop: Repeat, output: Send, parallel: ViewModule, branch: CallSplit };
    const labels = { tool: 'Tool', llm_task: 'LLM Task', llm_agent: 'LLM Agent', deep_agent: 'Deep Agent', approval: 'Approval', browser_authenticate: 'Browser Auth', loop: 'Loop', output: 'Output', parallel: 'Parallel', branch: 'Branch' };
    const Icon = icons[type] || Build;
    const bg = type === 'llm_task' ? 'rgba(25, 118, 210, 0.08)' : type === 'llm_agent' ? 'rgba(102, 187, 106, 0.08)' : type === 'approval' ? 'rgba(237, 108, 2, 0.08)' : type === 'browser_authenticate' ? 'rgba(93, 64, 55, 0.08)' : type === 'loop' ? 'rgba(156, 39, 176, 0.06)' : type === 'output' ? 'rgba(0, 150, 136, 0.06)' : type === 'parallel' ? 'rgba(66, 165, 245, 0.06)' : type === 'branch' ? 'rgba(255, 152, 0, 0.06)' : undefined;
    return { type, Icon, label: labels[type], bg };
  };

  /**
   * Shared step card body (header row + detail captions). Wrap with Paper + stepCardPaperSx at call sites.
   */
  const renderStepCard = (step, {
    indexLabel,
    readOnly: ro,
    showCheckbox,
    isChecked,
    onToggleCheckbox,
    dragHandleProps,
    isDragging: _isDragging,
    onOpen,
    onDelete,
    parallelChildCount,
  }) => {
    const { Icon, label } = stepTypeInfo(step);
    const actionName = step.action || step.name || (step.step_type || step.type) || 'step';
    const outputKey = step.output_key || '';
    const inputSummary = step.inputs && Object.keys(step.inputs).length > 0
      ? Object.entries(step.inputs).map(([k, v]) => `${k}: ${String(v).slice(0, 20)}${String(v).length > 20 ? '…' : ''}`).join(', ')
      : null;
    const outputFields = getOutputFieldsForStep(step, actionsByName);
    const outputSummary = outputKey
      ? (outputFields.length > 0
        ? `${outputKey} → ${outputFields.map((f) => f.name).join(', ')}`
        : outputKey)
      : null;
    const modelSummary = (step.step_type || step.type) === 'llm_task' && step.model_override
      ? step.model_override
      : null;
    const toolsSummary = (step.step_type || step.type) === 'llm_agent' && Array.isArray(step.available_tools) && step.available_tools.length > 0
      ? `${step.available_tools.length} tool${step.available_tools.length !== 1 ? 's' : ''}`
      : null;
    const subagentsSummary = ['llm_agent', 'deep_agent'].includes(step.step_type || step.type)
      && Array.isArray(step.subagents) && step.subagents.length > 0
      ? `${step.subagents.length} subagent${step.subagents.length !== 1 ? 's' : ''}`
      : null;
    const phasesSummary = (step.step_type || step.type) === 'deep_agent' && Array.isArray(step.phases) && step.phases.length > 0
      ? `${step.phases.length} phase${step.phases.length !== 1 ? 's' : ''}`
      : null;
    const isLoop = (step.step_type || step.type) === 'loop';
    const isParallel = (step.step_type || step.type) === 'parallel';
    const isBranch = (step.step_type || step.type) === 'branch';
    const hasCondition = !!(step.condition && String(step.condition).trim());
    const conditionLabel = hasCondition
      ? (String(step.condition).length > 32 ? String(step.condition).slice(0, 32) + '…' : step.condition)
      : null;
    const pc = typeof parallelChildCount === 'number' ? parallelChildCount : (step.parallel_steps || []).length;

    return (
      <>
        <ListItemButton
          onClick={onOpen}
          sx={{ py: 1.5, pr: 0 }}
        >
          {!ro && showCheckbox && (
            <Checkbox
              size="small"
              checked={isChecked}
              onChange={onToggleCheckbox}
              onClick={(e) => e.stopPropagation()}
              sx={{ p: 0.25, mr: 0.5 }}
              aria-label="Select step"
            />
          )}
          {!ro && dragHandleProps && (
            <Box
              {...dragHandleProps}
              sx={{ display: 'flex', alignItems: 'center', pr: 0.5, cursor: 'grab', color: 'text.secondary' }}
              aria-label="Drag to reorder"
            >
              <DragIndicator fontSize="small" />
            </Box>
          )}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1, minWidth: 0 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexShrink: 0 }}>
              <Icon sx={{ fontSize: 18, color: 'text.secondary' }} />
              <Typography variant="caption" fontWeight={600} color="text.secondary">
                {indexLabel}
              </Typography>
            </Box>
            <Chip size="small" label={label} sx={{ fontWeight: 500 }} />
            <Typography variant="body2" noWrap sx={{ flex: 1, minWidth: 0 }}>
              {actionName}
            </Typography>
            {isLoop && (
              <Chip size="small" variant="outlined" label={`max ${step.max_iterations ?? 3} iterations`} sx={{ flexShrink: 0 }} />
            )}
            {isParallel && (
              <Chip size="small" variant="outlined" label={`${pc} step${pc !== 1 ? 's' : ''}`} sx={{ flexShrink: 0 }} />
            )}
            {isBranch && (
              <Tooltip title={step.branch_condition || 'Set condition in config'} enterDelay={400} placement="top">
                <Chip size="small" variant="outlined" label={step.branch_condition ? String(step.branch_condition).slice(0, 24) + (String(step.branch_condition).length > 24 ? '…' : '') : 'IF …'} sx={{ flexShrink: 0, maxWidth: 200 }} />
              </Tooltip>
            )}
            {outputKey && (
              <Chip size="small" variant="outlined" label={`→ ${outputKey}`} sx={{ flexShrink: 0 }} />
            )}
            {hasCondition && conditionLabel && (
              <Tooltip title={step.condition} enterDelay={400} placement="top">
                <Chip size="small" label={`IF ${conditionLabel}`} sx={{ flexShrink: 0, fontStyle: 'italic', maxWidth: 180 }} />
              </Tooltip>
            )}
          </Box>
          {!ro && onDelete && (
            <IconButton size="small" onClick={(e) => { e.stopPropagation(); onDelete(); }} aria-label="Remove step">
              <Delete />
            </IconButton>
          )}
        </ListItemButton>
        {(inputSummary || outputSummary || modelSummary || toolsSummary || subagentsSummary || phasesSummary) && (
          <Box sx={{ px: 2, pb: 1, pt: 0, display: 'flex', flexDirection: 'column', gap: 0.25 }}>
            {inputSummary && (
              <Typography variant="caption" color="text.secondary" noWrap>
                Inputs: {inputSummary}
              </Typography>
            )}
            {outputSummary && (
              <Typography variant="caption" color="text.secondary" noWrap>
                Outputs: {outputSummary}
              </Typography>
            )}
            {modelSummary && (
              <Typography variant="caption" color="text.secondary" noWrap>
                Model: {modelSummary}
              </Typography>
            )}
            {toolsSummary && (
              <Typography variant="caption" color="text.secondary" noWrap>
                {toolsSummary}
              </Typography>
            )}
            {subagentsSummary && (
              <Typography variant="caption" color="text.secondary" noWrap>
                {subagentsSummary}
              </Typography>
            )}
            {phasesSummary && (
              <Typography variant="caption" color="text.secondary" noWrap>
                {phasesSummary}
              </Typography>
            )}
          </Box>
        )}
      </>
    );
  };

  const stepCardPaperSx = (step, isDragging) => {
    const { bg } = stepTypeInfo(step);
    const hasCondition = !!(step.condition && String(step.condition).trim());
    return {
      width: '100%',
      bgcolor: bg,
      borderRadius: 1,
      overflow: 'hidden',
      boxShadow: isDragging ? 2 : 0,
      ...(hasCondition && { borderStyle: 'dashed', borderWidth: 1.5 }),
    };
  };

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2, gap: 1, flexWrap: 'wrap' }}>
          <Typography variant="h6">
            Playbook steps
          </Typography>
          {!readOnly && onPasteRequest && (
            <Tooltip title="Insert after selection, or at end if none selected">
              <Button size="small" variant="outlined" startIcon={<ContentPaste />} onClick={handlePasteClick}>
                Paste steps
              </Button>
            </Tooltip>
          )}
        </Box>
        {wiringErrors.length > 0 && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Playbook validation: {wiringErrors.length} issue{wiringErrors.length !== 1 ? 's' : ''} found
            </Typography>
            <Box component="ul" sx={{ m: 0, pl: 2 }}>
              {wiringErrors.slice(0, 5).map((err, i) => (
                <li key={i}>
                  Step &quot;{err.stepName}&quot; / {err.inputKey}: {err.message}
                </li>
              ))}
              {wiringErrors.length > 5 && (
                <li>… and {wiringErrors.length - 5} more</li>
              )}
            </Box>
          </Alert>
        )}
        {(sortedSelected.length >= 2 || singleSelectedParallel || (sortedSelected.length >= 1 && copyContiguous && onCopySteps)) && !readOnly && (
          <Paper variant="outlined" sx={{ p: 1, mb: 2, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
            {sortedSelected.length >= 1 && copyContiguous && onCopySteps && (
              <Tooltip title={copyContiguous ? 'Copy step JSON to clipboard (same browser tab)' : 'Select adjacent steps to copy'}>
                <span>
                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<ContentCopy />}
                    onClick={handleCopySelectedSteps}
                    disabled={!copyContiguous}
                  >
                    Copy
                  </Button>
                </span>
              </Tooltip>
            )}
            {sortedSelected.length >= 2 && (
              <>
                <Typography variant="body2" color="text.secondary">
                  {sortedSelected.length} steps selected
                </Typography>
                {!isAdjacent && (
                  <Typography variant="caption" color="warning.main">
                    Select adjacent steps to group
                  </Typography>
                )}
                <Button
                  size="small"
                  variant="contained"
                  startIcon={<ViewModule />}
                  onClick={handleGroupAsParallel}
                  disabled={!isAdjacent}
                >
                  Group as Parallel
                </Button>
              </>
            )}
            {singleSelectedParallel && (
              <Button
                size="small"
                variant="outlined"
                startIcon={<CallSplit />}
                onClick={handleUngroupParallel}
              >
                Ungroup
              </Button>
            )}
            <Button size="small" onClick={() => setSelectedIndices([])}>
              Cancel
            </Button>
          </Paper>
        )}
        <DragDropContext onDragEnd={handleDragEnd}>
          <Droppable droppableId="playbook-steps" type="playbook-main">
            {(provided) => (
              <List
                ref={provided.innerRef}
                {...provided.droppableProps}
                dense
                sx={{ borderRadius: 1 }}
              >
                {steps.map((step, idx) => {
                  const isLoop = (step.step_type || step.type) === 'loop';
                  const isParallel = (step.step_type || step.type) === 'parallel';
                  const isBranch = (step.step_type || step.type) === 'branch';
                  const childSteps = isLoop ? (step.steps || []) : [];
                  const parallelSteps = isParallel ? (step.parallel_steps || []) : [];
                  const thenSteps = isBranch ? (step.then_steps || []) : [];
                  const elseSteps = isBranch ? (step.else_steps || []) : [];
                  return (
                    <Draggable key={idx} draggableId={`step-${idx}`} index={idx} isDragDisabled={readOnly}>
                      {(dragProvided, dragSnapshot) => (
                        <ListItem
                          ref={dragProvided.innerRef}
                          {...dragProvided.draggableProps}
                          disablePadding
                          sx={{ mb: 1, flexDirection: 'column', alignItems: 'stretch' }}
                        >
                          <Paper variant="outlined" sx={stepCardPaperSx(step, dragSnapshot.isDragging)}>
                            {renderStepCard(step, {
                              indexLabel: String(idx + 1),
                              readOnly,
                              showCheckbox: true,
                              isChecked: selectedIndices.includes(idx),
                              onToggleCheckbox: () => handleToggleSelect(idx),
                              dragHandleProps: readOnly ? null : dragProvided.dragHandleProps,
                              isDragging: dragSnapshot.isDragging,
                              onOpen: () => !readOnly && onStepClick && onStepClick(idx, step),
                              onDelete: readOnly ? null : () => handleRemove(idx),
                              parallelChildCount: parallelSteps.length,
                            })}
                            {isLoop && childSteps.length > 0 && (
                              <Box sx={{ borderLeft: 2, borderColor: 'divider', ml: 2, mr: 1, mb: 1, pl: 1.5 }}>
                                {childSteps.map((child, cIdx) => {
                                  const childInfo = stepTypeInfo(child);
                                  const ChildIcon = childInfo.Icon;
                                  const childName = child.name || child.output_key || child.action || `Step ${cIdx + 1}`;
                                  return (
                                    <ListItemButton
                                      key={cIdx}
                                      onClick={() => !readOnly && onStepClick && onStepClick(cIdx, child, { parentLoopIndex: idx })}
                                      sx={{ py: 0.75, borderRadius: 1 }}
                                    >
                                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                                        <ChildIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
                                        <Chip size="small" label={childInfo.label} sx={{ fontWeight: 500 }} />
                                        <Typography variant="body2" noWrap sx={{ flex: 1 }}>
                                          {childName}
                                        </Typography>
                                      </Box>
                                    </ListItemButton>
                                  );
                                })}
                              </Box>
                            )}
                            {isParallel && (
                              <Box sx={{ borderLeft: 2, borderColor: 'info.main', ml: 2, mr: 1, mb: 1, pl: 1.5, pr: 1 }}>
                                <Droppable droppableId={`parallel-${idx}`} type={`parallel-${idx}`}>
                                  {(pDropProvided) => (
                                    <Box ref={pDropProvided.innerRef} {...pDropProvided.droppableProps}>
                                      {parallelSteps.map((child, cIdx) => {
                                        const dragId = child._step_id != null
                                          ? `par-${idx}-${child._step_id}`
                                          : `par-${idx}-i${cIdx}`;
                                        return (
                                          <Draggable
                                            key={dragId}
                                            draggableId={dragId}
                                            index={cIdx}
                                            isDragDisabled={readOnly || !onReorderParallelChild}
                                          >
                                            {(cDragProvided, cDragSnapshot) => (
                                              <Box
                                                ref={cDragProvided.innerRef}
                                                {...cDragProvided.draggableProps}
                                                sx={{ mb: 1 }}
                                              >
                                                <Paper variant="outlined" sx={stepCardPaperSx(child, cDragSnapshot.isDragging)}>
                                                  {renderStepCard(child, {
                                                    indexLabel: String.fromCharCode(65 + cIdx),
                                                    readOnly,
                                                    showCheckbox: false,
                                                    isChecked: false,
                                                    onToggleCheckbox: () => {},
                                                    dragHandleProps: readOnly || !onReorderParallelChild ? null : cDragProvided.dragHandleProps,
                                                    isDragging: cDragSnapshot.isDragging,
                                                    onOpen: () => !readOnly && onStepClick && onStepClick(cIdx, child, { parentParallelIndex: idx }),
                                                    onDelete: readOnly || !onRemoveParallelChild ? null : () => onRemoveParallelChild(idx, cIdx),
                                                    parallelChildCount: undefined,
                                                  })}
                                                </Paper>
                                              </Box>
                                            )}
                                          </Draggable>
                                        );
                                      })}
                                      {pDropProvided.placeholder}
                                      {!readOnly && (
                                        <Button
                                          size="small"
                                          variant="outlined"
                                          startIcon={<Add />}
                                          disabled={parallelSteps.length >= maxParallelSubsteps}
                                          onClick={() => onAddParallelChild && onAddParallelChild(idx)}
                                          sx={{ mt: 0.5 }}
                                          fullWidth
                                        >
                                          Add to group
                                        </Button>
                                      )}
                                    </Box>
                                  )}
                                </Droppable>
                              </Box>
                            )}
                            {isBranch && (
                              <Box sx={{ px: 2, pb: 1.5, pt: 0 }}>
                                <Typography variant="caption" fontWeight={600} color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                                  IF {step.branch_condition || '(set condition in config)'}
                                </Typography>
                                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                                  <Box sx={{ flex: '1 1 200px', minWidth: 0, borderLeft: 2, borderColor: 'success.main', pl: 1.5 }}>
                                    <Typography variant="caption" color="success.dark" fontWeight={600}>THEN</Typography>
                                    {(thenSteps || []).map((child, cIdx) => {
                                      const childInfo = stepTypeInfo(child);
                                      const ThenIcon = childInfo.Icon;
                                      const childName = child.name || child.output_key || child.action || `Step ${cIdx + 1}`;
                                      return (
                                        <ListItemButton
                                          key={cIdx}
                                          onClick={() => !readOnly && onStepClick && onStepClick(cIdx, child, { parentBranchIndex: idx, branchPath: 'then' })}
                                          sx={{ py: 0.5, px: 0, minHeight: 32 }}
                                        >
                                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, width: '100%' }}>
                                            <ThenIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
                                            <Typography variant="body2" noWrap sx={{ flex: 1 }}>{childName}</Typography>
                                          </Box>
                                        </ListItemButton>
                                      );
                                    })}
                                    {!readOnly && (
                                      <Button size="small" startIcon={<Add />} onClick={() => onAddBranchChild && onAddBranchChild(idx, 'then')} sx={{ mt: 0.5 }}>
                                        Add to THEN
                                      </Button>
                                    )}
                                  </Box>
                                  <Box sx={{ flex: '1 1 200px', minWidth: 0, borderLeft: 2, borderColor: 'divider', pl: 1.5 }}>
                                    <Typography variant="caption" color="text.secondary" fontWeight={600}>ELSE</Typography>
                                    {(elseSteps || []).map((child, cIdx) => {
                                      const childInfo = stepTypeInfo(child);
                                      const ElseIcon = childInfo.Icon;
                                      const childName = child.name || child.output_key || child.action || `Step ${cIdx + 1}`;
                                      return (
                                        <ListItemButton
                                          key={cIdx}
                                          onClick={() => !readOnly && onStepClick && onStepClick(cIdx, child, { parentBranchIndex: idx, branchPath: 'else' })}
                                          sx={{ py: 0.5, px: 0, minHeight: 32 }}
                                        >
                                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, width: '100%' }}>
                                            <ElseIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
                                            <Typography variant="body2" noWrap sx={{ flex: 1 }}>{childName}</Typography>
                                          </Box>
                                        </ListItemButton>
                                      );
                                    })}
                                    {!readOnly && (
                                      <Button size="small" startIcon={<Add />} onClick={() => onAddBranchChild && onAddBranchChild(idx, 'else')} sx={{ mt: 0.5 }}>
                                        Add to ELSE
                                      </Button>
                                    )}
                                  </Box>
                                </Box>
                              </Box>
                            )}
                          </Paper>
                        </ListItem>
                      )}
                    </Draggable>
                  );
                })}
                {provided.placeholder}
                {steps.length === 0 && (
                  <ListItem>
                    <ListItemText
                      primary="No steps"
                      secondary="Add a tool, LLM task, or approval step below"
                    />
                  </ListItem>
                )}
              </List>
            )}
          </Droppable>
        </DragDropContext>
        {!readOnly && !addStepPanelOpen && (
          <Box sx={{ mt: 2 }}>
            <Button
              variant="outlined"
              startIcon={<Add />}
              onClick={onAddStep}
              fullWidth
            >
              Add step
            </Button>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}
