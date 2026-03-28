/**
 * Goal tree: hierarchical goals with title, status, progress, assigned agent.
 */

import React from 'react';
import { Box, Typography, Card, CardContent, Chip, LinearProgress } from '@mui/material';
import { Flag } from '@mui/icons-material';

function GoalTreeNode({ node, depth = 0, onEdit, onDelete }) {
  const title = node.title || 'Untitled';
  const status = node.status || 'active';
  const progress = Math.min(100, Math.max(0, node.progress_pct ?? 0));
  const children = node.children || [];
  const hasActions = typeof onEdit === 'function' || typeof onDelete === 'function';

  return (
    <Box sx={{ ml: depth ? 3 : 0, mb: 1 }}>
      <Card variant="outlined" sx={{ maxWidth: 420 }}>
        <CardContent sx={{ py: 1, px: 1.5, '&:last-child': { pb: 1 } }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
            <Flag fontSize="small" color="action" />
            <Typography variant="subtitle2" sx={{ flex: 1, minWidth: 0 }}>
              {title}
            </Typography>
            <Chip size="small" label={status} variant="outlined" />
            {hasActions && (
              <Box sx={{ display: 'flex', gap: 0.5 }}>
                {onEdit && (
                  <Chip size="small" label="Edit" onClick={() => onEdit(node)} variant="outlined" />
                )}
                {onDelete && (
                  <Chip size="small" label="Delete" onClick={() => onDelete(node)} color="error" variant="outlined" />
                )}
              </Box>
            )}
          </Box>
          {node.assigned_agent_id && (
            <Typography variant="caption" color="text.secondary" display="block">
              Assigned: {node.assigned_agent_name || node.assigned_agent_id}
            </Typography>
          )}
          <Box sx={{ mt: 0.5 }}>
            <LinearProgress variant="determinate" value={progress} sx={{ height: 6, borderRadius: 1 }} />
            <Typography variant="caption" color="text.secondary">{progress}%</Typography>
          </Box>
        </CardContent>
      </Card>
      {children.length > 0 && (
        <Box sx={{ mt: 0.5, borderLeft: 1, borderColor: 'divider', ml: 1.5, pl: 1 }}>
          {children.map((child) => (
            <GoalTreeNode
              key={child.id}
              node={child}
              depth={depth + 1}
              onEdit={onEdit}
              onDelete={onDelete}
            />
          ))}
        </Box>
      )}
    </Box>
  );
}

export default function GoalTreeView({ tree, onEditGoal, onDeleteGoal }) {
  const roots = Array.isArray(tree) ? tree : [];

  if (roots.length === 0) {
    return (
      <Box sx={{ py: 2, textAlign: 'center' }}>
        <Typography color="text.secondary">No goals yet. Create a goal to get started.</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
        Goal tree
      </Typography>
      {roots.map((root) => (
        <GoalTreeNode
          key={root.id}
          node={root}
          onEdit={onEditGoal}
          onDelete={onDeleteGoal}
        />
      ))}
    </Box>
  );
}
