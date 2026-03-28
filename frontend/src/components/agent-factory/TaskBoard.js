/**
 * Kanban-style task board: backlog | assigned | in_progress | review | done
 */

import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import { DragIndicator, Edit, Delete } from '@mui/icons-material';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';

const COLUMNS = [
  { id: 'backlog', label: 'Backlog' },
  { id: 'assigned', label: 'Assigned' },
  { id: 'in_progress', label: 'In progress' },
  { id: 'review', label: 'Review' },
  { id: 'done', label: 'Done' },
];

function formatTs(iso) {
  if (!iso) return null;
  try {
    return new Date(iso).toLocaleString(undefined, {
      dateStyle: 'medium',
      timeStyle: 'short',
    });
  } catch (_) {
    return iso;
  }
}

function TaskHoverDetails({ task, agentNameById, goalTitleById }) {
  const assignee = task.assigned_agent_id
    ? agentNameById[task.assigned_agent_id] ||
      task.assigned_agent_name ||
      task.assigned_agent_id
    : null;
  const creator = task.created_by_agent_id
    ? agentNameById[task.created_by_agent_id] || task.created_by_agent_id
    : null;
  const goalTitle = task.goal_id ? goalTitleById[task.goal_id] || task.goal_id : null;

  return (
    <Box sx={{ textAlign: 'left', maxWidth: 400 }}>
      <Typography variant="subtitle2" component="div" fontWeight={700} sx={{ mb: 0.75, lineHeight: 1.35 }}>
        {task.title}
      </Typography>
      {task.description ? (
        <Typography
          variant="body2"
          sx={{
            mb: 1,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            opacity: 0.95,
            lineHeight: 1.45,
          }}
        >
          {task.description}
        </Typography>
      ) : (
        <Typography variant="caption" component="div" sx={{ mb: 1, opacity: 0.85, fontStyle: 'italic' }}>
          No description
        </Typography>
      )}
      <Typography variant="caption" component="div" sx={{ display: 'block', lineHeight: 1.65 }}>
        <strong>Assigned to:</strong> {assignee || '—'}
        <br />
        <strong>Created by:</strong>{' '}
        {creator || 'Not attributed to an agent (e.g. created in dashboard)'}
        {goalTitle ? (
          <>
            <br />
            <strong>Goal:</strong> {goalTitle}
          </>
        ) : null}
        <br />
        <strong>Status:</strong> {task.status || 'backlog'} · <strong>Priority:</strong> P{task.priority ?? 0}
        {task.due_date ? (
          <>
            <br />
            <strong>Due:</strong> {task.due_date}
          </>
        ) : null}
        {task.created_at ? (
          <>
            <br />
            <strong>Created:</strong> {formatTs(task.created_at)}
          </>
        ) : null}
        {task.updated_at && task.updated_at !== task.created_at ? (
          <>
            <br />
            <strong>Updated:</strong> {formatTs(task.updated_at)}
          </>
        ) : null}
      </Typography>
    </Box>
  );
}

function TaskCard({
  task,
  onEdit,
  onDelete,
  onTransition,
  assignedAgentColor,
  agentNameById,
  goalTitleById,
  compact = false,
}) {
  const status = task.status || 'backlog';
  const canMoveLeft = status !== 'backlog';
  const canMoveRight = status !== 'done';

  const moveLeft = () => {
    const idx = COLUMNS.findIndex((c) => c.id === status);
    if (idx > 0) onTransition?.(task.id, COLUMNS[idx - 1].id);
  };
  const moveRight = () => {
    const idx = COLUMNS.findIndex((c) => c.id === status);
    if (idx >= 0 && idx < COLUMNS.length - 1) onTransition?.(task.id, COLUMNS[idx + 1].id);
  };

  return (
    <Tooltip
      title={
        <TaskHoverDetails task={task} agentNameById={agentNameById} goalTitleById={goalTitleById} />
      }
      placement="top"
      enterDelay={compact ? 250 : 400}
      enterNextDelay={300}
      componentsProps={{
        tooltip: {
          sx: {
            maxWidth: 440,
            maxHeight: 320,
            overflow: 'auto',
            py: 1.25,
            px: 1.5,
          },
        },
      }}
    >
      <Card
        variant="outlined"
        sx={{
          mb: compact ? 0.5 : 1,
          cursor: 'default',
          '&:hover': { bgcolor: 'action.hover' },
          ...(assignedAgentColor && { borderLeft: 3, borderLeftColor: assignedAgentColor }),
        }}
      >
        <CardContent
          sx={{
            py: compact ? 0.5 : 1,
            px: compact ? 1 : 1.5,
            '&:last-child': { pb: compact ? 0.5 : 1 },
          }}
        >
          {compact ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, minWidth: 0 }}>
              <Typography
                variant="body2"
                noWrap
                sx={{ flex: 1, minWidth: 0, fontSize: '0.8125rem', lineHeight: 1.35 }}
              >
                {task.title}
              </Typography>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ flexShrink: 0, fontVariantNumeric: 'tabular-nums' }}
              >
                P{task.priority ?? 0}
              </Typography>
            </Box>
          ) : (
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 0.5 }}>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                {canMoveLeft && (
                  <IconButton size="small" onClick={moveLeft} title="Move left" sx={{ p: 0.25 }}>
                    <DragIndicator sx={{ transform: 'rotate(-90deg)', fontSize: 18 }} />
                  </IconButton>
                )}
                {canMoveRight && (
                  <IconButton size="small" onClick={moveRight} title="Move right" sx={{ p: 0.25 }}>
                    <DragIndicator sx={{ transform: 'rotate(90deg)', fontSize: 18 }} />
                  </IconButton>
                )}
              </Box>
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Typography variant="subtitle2" noWrap title={task.title}>
                  {task.title}
                </Typography>
                {task.assigned_agent_id && (
                  <Typography variant="caption" color="text.secondary">
                    → {task.assigned_agent_name || task.assigned_agent_id}
                  </Typography>
                )}
                <Box sx={{ mt: 0.5, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                  <Chip size="small" label={`P${task.priority ?? 0}`} variant="outlined" />
                  {task.due_date && (
                    <Chip size="small" label={task.due_date} variant="outlined" />
                  )}
                </Box>
              </Box>
              <Box>
                {onEdit && (
                  <IconButton size="small" onClick={() => onEdit(task)} title="Edit">
                    <Edit fontSize="small" />
                  </IconButton>
                )}
                {onDelete && (
                  <IconButton size="small" onClick={() => onDelete(task)} title="Delete" color="error">
                    <Delete fontSize="small" />
                  </IconButton>
                )}
              </Box>
            </Box>
          )}
        </CardContent>
      </Card>
    </Tooltip>
  );
}

/** Approximate row height for compact cards (px) for scroll cap */
const COMPACT_TASK_ROW_PX = 40;

export default function TaskBoard({
  tasks = [],
  onEditTask,
  onDeleteTask,
  onTransitionTask,
  agentColorMap = {},
  agentNameById = {},
  goalTitleById = {},
  compact = false,
  /** When set with `compact`, column body scrolls after this many visible rows */
  maxTasksPerColumn = null,
}) {
  const byStatus = (status) => (tasks || []).filter((t) => (t.status || 'backlog') === status);

  const onDragEnd = (result) => {
    if (!result.destination || result.destination.droppableId === result.source.droppableId) return;
    onTransitionTask?.(result.draggableId, result.destination.droppableId);
  };

  const columnBodySx =
    compact && maxTasksPerColumn != null && maxTasksPerColumn > 0
      ? {
          minHeight: 32,
          maxHeight: maxTasksPerColumn * COMPACT_TASK_ROW_PX,
          overflowY: 'auto',
          overflowX: 'hidden',
          pr: 0.25,
        }
      : { minHeight: 40 };

  const columns = (
    <Box sx={{ display: 'flex', gap: compact ? 1.5 : 2, overflowX: 'auto', pb: 1 }}>
      {COLUMNS.map((col) => (
        <Card
          key={col.id}
          variant="outlined"
          sx={{ minWidth: compact ? 200 : 220, maxWidth: compact ? 240 : 280 }}
        >
          <CardContent sx={{ py: compact ? 1 : 1.5, px: compact ? 1 : 1.5 }}>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: compact ? 0.75 : 1 }}>
              {col.label} ({byStatus(col.id).length})
            </Typography>
            {compact ? (
              // Compact mode: no drag-and-drop. DragDropContext registers global window listeners
              // that intercept ALL mousedown events (including native scrollbar drags) and show
              // a "not-allowed" cursor. Skip it entirely; use the arrow-button transition UI instead.
              <Box sx={columnBodySx}>
                {byStatus(col.id).map((task) => (
                  <TaskCard
                    key={task.id}
                    task={task}
                    onEdit={onEditTask}
                    onDelete={onDeleteTask}
                    onTransition={onTransitionTask}
                    assignedAgentColor={task.assigned_agent_id ? agentColorMap[task.assigned_agent_id] : undefined}
                    agentNameById={agentNameById}
                    goalTitleById={goalTitleById}
                    compact={compact}
                  />
                ))}
              </Box>
            ) : (
              <Droppable droppableId={col.id}>
                {(provided) => (
                  <Box ref={provided.innerRef} {...provided.droppableProps} sx={columnBodySx}>
                    {byStatus(col.id).map((task, index) => (
                      <Draggable key={task.id} draggableId={task.id} index={index}>
                        {(provided) => (
                          <Box
                            ref={provided.innerRef}
                            {...provided.draggableProps}
                            sx={{
                              display: 'flex',
                              alignItems: 'flex-start',
                              gap: 0.5,
                              width: '100%',
                              minWidth: 0,
                            }}
                          >
                            <Box
                              {...provided.dragHandleProps}
                              sx={{
                                cursor: 'grab',
                                display: 'flex',
                                alignItems: 'center',
                                flexShrink: 0,
                                py: 0.25,
                                px: 0.125,
                                touchAction: 'none',
                              }}
                              aria-label="Drag to move task"
                            >
                              <DragIndicator sx={{ fontSize: 20, color: 'action.active', opacity: 0.85 }} />
                            </Box>
                            <Box sx={{ flex: 1, minWidth: 0 }}>
                              <TaskCard
                                task={task}
                                onEdit={onEditTask}
                                onDelete={onDeleteTask}
                                onTransition={onTransitionTask}
                                assignedAgentColor={task.assigned_agent_id ? agentColorMap[task.assigned_agent_id] : undefined}
                                agentNameById={agentNameById}
                                goalTitleById={goalTitleById}
                                compact={compact}
                              />
                            </Box>
                          </Box>
                        )}
                      </Draggable>
                    ))}
                    {provided.placeholder}
                  </Box>
                )}
              </Droppable>
            )}
          </CardContent>
        </Card>
      ))}
    </Box>
  );

  if (compact) {
    return columns;
  }

  return (
    <DragDropContext onDragEnd={onDragEnd}>
      {columns}
    </DragDropContext>
  );
}
