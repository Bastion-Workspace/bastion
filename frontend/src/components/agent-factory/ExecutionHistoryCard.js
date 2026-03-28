/**
 * Recent execution history for an agent profile.
 * Click an execution to open a detail drawer with step-level trace (query, status, duration, steps, metadata).
 */

import React, { useState, useCallback } from 'react';
import {
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
  CircularProgress,
  Box,
  Drawer,
  IconButton,
  Alert,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
  Menu,
  MenuItem,
} from '@mui/material';
import { Close, ExpandMore, Build, Psychology, SmartToy, Gavel, Lock, Repeat, ViewModule, CallSplit, Delete, DeleteSweep, FileDownload, AccountTree } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';

const STEP_TYPE_ICONS = {
  tool: Build,
  llm_task: Psychology,
  llm_agent: SmartToy,
  deep_agent: AccountTree,
  approval: Gavel,
  browser_authenticate: Lock,
  loop: Repeat,
  parallel: ViewModule,
  branch: CallSplit,
};
const STEP_TYPE_LABELS = {
  tool: 'Tool',
  llm_task: 'LLM Task',
  llm_agent: 'LLM Agent',
  deep_agent: 'Deep Agent',
  approval: 'Approval',
  browser_authenticate: 'Browser Auth',
  loop: 'Loop',
  parallel: 'Parallel',
  branch: 'Branch',
};

const DRAWER_MIN_WIDTH = 400;
const DRAWER_MAX_WIDTH = 1200;
const DRAWER_DEFAULT_WIDTH = 560;

function formatDuration(ms) {
  if (ms == null) return '—';
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatDate(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    const now = new Date();
    const diff = now - d;
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return d.toLocaleDateString();
  } catch (_) {
    return iso;
  }
}

/** Build a readable Markdown string from execution detail for export. */
function executionToMarkdown(detail) {
  if (!detail) return '';
  const lines = [
    '# Execution trace',
    '',
    '## Query',
    detail.query || '—',
    '',
    '## Summary',
    `- **Status:** ${detail.status}`,
    `- **Duration:** ${formatDuration(detail.duration_ms)}`,
    detail.started_at ? `- **Started:** ${new Date(detail.started_at).toLocaleString()}` : '',
    detail.error_details ? `- **Error:** ${detail.error_details}` : '',
    '',
  ].filter(Boolean);

  if (detail.steps && detail.steps.length > 0) {
    lines.push('## Step trace', '');
    detail.steps.forEach((step, idx) => {
      const label = STEP_TYPE_LABELS[step.step_type] || step.step_type || 'Step';
      lines.push(`### ${idx + 1}. ${step.step_name || label} (${label})`, '');
      lines.push(`- **Status:** ${step.status}`);
      if (step.duration_ms != null) lines.push(`- **Duration:** ${formatDuration(step.duration_ms)}`);
      if (step.error_details) lines.push(`- **Error:** ${step.error_details}`);
      lines.push('');

      if (step.inputs && Object.keys(step.inputs).length > 0) {
        lines.push('#### Inputs', '');
        if (typeof step.inputs.prompt === 'string') {
          lines.push(step.inputs.prompt, '');
        }
        lines.push('```json', JSON.stringify(step.inputs, null, 2), '```', '');
      }

      if (step.step_type === 'llm_agent' && step.tool_call_trace && step.tool_call_trace.length > 0) {
        lines.push('#### Tool calls', '');
        step.tool_call_trace.forEach((tc, i) => {
          lines.push(`- **${tc.tool_name || 'tool'}** (iter ${tc.iteration})${tc.duration_ms != null ? ` — ${formatDuration(tc.duration_ms)}` : ''}${tc.status === 'failed' ? ' — failed' : ''}`);
          if (tc.args && Object.keys(tc.args).length > 0) {
            lines.push('  ```json', JSON.stringify(tc.args, null, 2), '  ```');
          }
          if (tc.result != null && tc.result !== '') {
            lines.push('  Result: ' + (tc.result.length > 200 ? tc.result.slice(0, 200) + '…' : tc.result));
          }
        });
        lines.push('');
      } else if (step.step_type === 'llm_agent') {
        lines.push('#### Tool calls', 'No tool calls', '');
      }

      if (step.step_type === 'deep_agent' && step.outputs && Array.isArray(step.outputs.phase_trace) && step.outputs.phase_trace.length > 0) {
        lines.push('#### Phase trace', '');
        step.outputs.phase_trace.forEach((pt, i) => {
          const parts = [`- **${pt.phase || `Phase ${i + 1}`}** (${pt.type || '—'})`];
          if (pt.duration_ms != null) parts.push(formatDuration(pt.duration_ms));
          if (pt.score != null) parts.push(`score: ${(pt.score * 100).toFixed(0)}%`);
          if (pt.pass === true) parts.push('pass');
          if (pt.pass === false) parts.push('retry');
          lines.push(parts.join(' — '));
        });
        lines.push('');
      }

      if (step.outputs && Object.keys(step.outputs).length > 0) {
        lines.push('#### Outputs', '');
        const out = step.outputs;
        if (out.formatted) lines.push(out.formatted, '');
        lines.push('```json', JSON.stringify(step.outputs, null, 2), '```', '');
      }
    });
  }

  if (detail.metadata && Object.keys(detail.metadata).length > 0) {
    lines.push('## Metadata', '');
    lines.push('```json', JSON.stringify(detail.metadata, null, 2), '```');
  }
  return lines.join('\n');
}

export default function ExecutionHistoryCard({ profileId }) {
  const [detailExecutionId, setDetailExecutionId] = useState(null);
  const [clearAllConfirm, setClearAllConfirm] = useState(false);
  const [deleteOneConfirm, setDeleteOneConfirm] = useState(null);
  const [drawerWidth, setDrawerWidth] = useState(DRAWER_DEFAULT_WIDTH);
  const [exportMenuAnchor, setExportMenuAnchor] = useState(null);
  const [isResizing, setIsResizing] = useState(false);
  const queryClient = useQueryClient();

  const { data: executions = [], isLoading } = useQuery(
    ['agentFactoryExecutions', profileId],
    () => apiService.agentFactory.listProfileExecutions(profileId, 10),
    { enabled: !!profileId, retry: false }
  );

  const { data: executionDetail, isLoading: detailLoading } = useQuery(
    ['agentFactoryExecutionDetail', profileId, detailExecutionId],
    () => apiService.agentFactory.getExecution(profileId, detailExecutionId),
    { enabled: !!profileId && !!detailExecutionId, retry: false }
  );

  const deleteExecutionMutation = useMutation(
    ({ profileId: pid, executionId: eid }) => apiService.agentFactory.deleteExecution(pid, eid),
    {
      onSuccess: (_, { profileId: pid }) => {
        queryClient.invalidateQueries(['agentFactoryExecutions', pid]);
        if (detailExecutionId === deleteOneConfirm?.id) setDetailExecutionId(null);
        setDeleteOneConfirm(null);
      },
    }
  );
  const clearExecutionsMutation = useMutation(
    (pid) => apiService.agentFactory.clearExecutions(pid),
    {
      onSuccess: (_, pid) => {
        queryClient.invalidateQueries(['agentFactoryExecutions', pid]);
        setDetailExecutionId(null);
        setClearAllConfirm(false);
      },
    }
  );

  const handleResizeMove = useCallback((e) => {
    if (!isResizing) return;
    const x = e.clientX != null ? e.clientX : e.touches?.[0]?.clientX;
    if (x == null) return;
    const maxW = Math.min(DRAWER_MAX_WIDTH, (window.innerWidth || 1024) * 0.95);
    const newWidth = Math.round(Math.min(maxW, Math.max(DRAWER_MIN_WIDTH, window.innerWidth - x)));
    setDrawerWidth(newWidth);
  }, [isResizing]);
  const handleResizeEnd = useCallback(() => setIsResizing(false), []);
  React.useEffect(() => {
    if (!isResizing) return;
    window.addEventListener('mousemove', handleResizeMove);
    window.addEventListener('mouseup', handleResizeEnd);
    return () => {
      window.removeEventListener('mousemove', handleResizeMove);
      window.removeEventListener('mouseup', handleResizeEnd);
    };
  }, [isResizing, handleResizeMove, handleResizeEnd]);

  const handleExportCopy = useCallback(() => {
    if (!executionDetail) return;
    const md = executionToMarkdown(executionDetail);
    navigator.clipboard.writeText(md).then(() => setExportMenuAnchor(null)).catch(() => {});
  }, [executionDetail]);
  const handleExportDownload = useCallback(() => {
    if (!executionDetail) return;
    const md = executionToMarkdown(executionDetail);
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `execution-trace-${executionDetail.id || 'export'}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
    setExportMenuAnchor(null);
  }, [executionDetail]);

  if (!profileId) return null;

  return (
    <>
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6">Recent runs</Typography>
            {executions.length > 0 && (
              <Tooltip title="Clear all runs">
                <IconButton
                  size="small"
                  onClick={() => setClearAllConfirm(true)}
                  disabled={clearExecutionsMutation.isLoading}
                  aria-label="Clear all"
                  color="inherit"
                  sx={{ opacity: 0.7 }}
                >
                  <DeleteSweep fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
          </Box>
          {isLoading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
              <CircularProgress size={24} />
            </Box>
          )}
          {!isLoading && (
            <List dense disablePadding>
              {executions.slice(0, 10).map((ex) => (
                <ListItem key={ex.id} disablePadding sx={{ borderBottom: 1, borderColor: 'divider' }} secondaryAction={
                  <Tooltip title="Delete run">
                    <IconButton
                      size="small"
                      edge="end"
                      onClick={(e) => { e.stopPropagation(); setDeleteOneConfirm(ex); }}
                      disabled={deleteExecutionMutation.isLoading}
                      aria-label="Delete"
                      sx={{ opacity: 0.6 }}
                    >
                      <Delete fontSize="small" />
                    </IconButton>
                  </Tooltip>
                }>
                  <ListItemButton onClick={() => setDetailExecutionId(ex.id)}>
                    <ListItemText
                      primary={ex.query ? `${ex.query.slice(0, 60)}${ex.query.length > 60 ? '…' : ''}` : '—'}
                      secondary={
                        <>
                          {formatDate(ex.started_at)} · {ex.status} · {formatDuration(ex.duration_ms)}
                          {ex.steps_completed != null && ex.steps_total != null && (
                            <> · {ex.steps_completed}/{ex.steps_total} steps</>
                          )}
                          {ex.cost_usd != null && ex.cost_usd > 0 && (
                            <> · ${Number(ex.cost_usd).toFixed(4)}</>
                          )}
                          {ex.error_details && (
                            <Typography component="span" variant="caption" color="error" display="block">
                              {ex.error_details.slice(0, 80)}…
                            </Typography>
                          )}
                        </>
                      }
                      primaryTypographyProps={{ noWrap: true }}
                    />
                  </ListItemButton>
                </ListItem>
              ))}
              {executions.length === 0 && (
                <ListItem>
                  <ListItemText primary="No runs yet" secondary="Use Test below to run this agent" />
                </ListItem>
              )}
            </List>
          )}
        </CardContent>
      </Card>

      <Drawer
        anchor="right"
        open={!!detailExecutionId}
        onClose={() => setDetailExecutionId(null)}
        PaperProps={{
          sx: {
            width: drawerWidth,
            minWidth: DRAWER_MIN_WIDTH,
            maxWidth: '95vw',
            position: 'relative',
          },
        }}
      >
        <Box
          onMouseDown={() => setIsResizing(true)}
          onTouchStart={() => setIsResizing(true)}
          sx={{
            position: 'absolute',
            left: 0,
            top: 0,
            bottom: 0,
            width: 6,
            cursor: 'col-resize',
            zIndex: 1,
            '&:hover': { bgcolor: 'action.hover' },
          }}
          aria-label="Resize drawer"
        />
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Typography variant="h6">Execution details</Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Tooltip title="Export trace">
              <IconButton
                size="small"
                onClick={(e) => setExportMenuAnchor(e.currentTarget)}
                disabled={!executionDetail}
                aria-label="Export"
              >
                <FileDownload fontSize="small" />
              </IconButton>
            </Tooltip>
            <Menu
              anchorEl={exportMenuAnchor}
              open={!!exportMenuAnchor}
              onClose={() => setExportMenuAnchor(null)}
              anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
              transformOrigin={{ vertical: 'top', horizontal: 'right' }}
            >
              <MenuItem onClick={handleExportCopy}>Copy as Markdown</MenuItem>
              <MenuItem onClick={handleExportDownload}>Download .md file</MenuItem>
            </Menu>
            <IconButton onClick={() => setDetailExecutionId(null)} aria-label="Close">
              <Close />
            </IconButton>
          </Box>
        </Box>
        <Box sx={{ p: 2, pb: 4, overflow: 'auto', maxHeight: '100%' }}>
          {detailLoading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress size={32} />
            </Box>
          )}
          {!detailLoading && executionDetail && (
            <>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Query
              </Typography>
              <Typography variant="body2" sx={{ mb: 2, wordBreak: 'break-word' }}>
                {executionDetail.query || '—'}
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                <Chip size="small" label={executionDetail.status} />
                <Chip size="small" label={formatDuration(executionDetail.duration_ms)} />
                {executionDetail.started_at && (
                  <Chip size="small" label={new Date(executionDetail.started_at).toLocaleString()} />
                )}
                {executionDetail.cost_usd != null && executionDetail.cost_usd > 0 && (
                  <Chip size="small" label={`$${Number(executionDetail.cost_usd).toFixed(4)}`} />
                )}
                {(executionDetail.tokens_input > 0 || executionDetail.tokens_output > 0) && (
                  <Chip size="small" label={`${(executionDetail.tokens_input || 0) + (executionDetail.tokens_output || 0)} tok`} />
                )}
                {executionDetail.model_used && (
                  <Chip size="small" label={executionDetail.model_used} variant="outlined" />
                )}
              </Box>
              {executionDetail.error_details && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                    {executionDetail.error_details}
                  </Typography>
                </Alert>
              )}

              {executionDetail.steps && executionDetail.steps.length > 0 && (
                <>
                  <Typography variant="subtitle2" color="text.secondary" sx={{ mt: 2, mb: 1 }}>
                    Step trace
                  </Typography>
                  {executionDetail.steps.map((step, idx) => {
                    const StepIcon = STEP_TYPE_ICONS[step.step_type] || Build;
                    const label = STEP_TYPE_LABELS[step.step_type] || step.step_type || 'Step';
                    const statusColor = step.status === 'failed' ? 'error' : step.status === 'skipped' ? 'default' : 'success';
                    return (
                      <Accordion key={idx} defaultExpanded={step.status === 'failed'} disableGutters sx={{ mb: 1, '&:before': { display: 'none' } }}>
                        <AccordionSummary expandIcon={<ExpandMore />} sx={{ minHeight: 48 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                            <StepIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
                            <Typography variant="body2" sx={{ flex: 1 }}>{step.step_name || `Step ${step.step_index}`}</Typography>
                            <Chip size="small" label={step.status} color={statusColor} sx={{ mr: 0.5 }} />
                            {step.duration_ms != null && (
                              <Typography variant="caption" color="text.secondary">{formatDuration(step.duration_ms)}</Typography>
                            )}
                          </Box>
                        </AccordionSummary>
                        <AccordionDetails sx={{ pt: 0, flexDirection: 'column', gap: 1 }}>
                          {step.step_type && (
                            <Typography variant="caption" color="text.secondary">{label}{step.action_name ? ` · ${step.action_name}` : ''}</Typography>
                          )}
                          {step.error_details && (
                            <Alert severity="error" sx={{ py: 0.5 }}>
                              <Typography variant="caption" component="pre" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{step.error_details}</Typography>
                            </Alert>
                          )}
                          {step.inputs && Object.keys(step.inputs).length > 0 && (
                            <Box>
                              <Typography variant="caption" color="text.secondary" display="block">Inputs</Typography>
                              <Box component="pre" sx={{ p: 1, bgcolor: 'action.hover', borderRadius: 1, fontSize: '0.7rem', overflow: 'auto', maxHeight: 220, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                                {JSON.stringify(step.inputs, null, 2)}
                              </Box>
                            </Box>
                          )}
                          {step.step_type === 'llm_agent' && (
                            <Box>
                              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>Tool calls</Typography>
                              {step.tool_call_trace && step.tool_call_trace.length > 0 ? (
                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                  {step.tool_call_trace.map((tc, tcIdx) => (
                                    <Accordion key={tcIdx} disableGutters sx={{ '&:before': { display: 'none' } }}>
                                      <AccordionSummary expandIcon={<ExpandMore />} sx={{ minHeight: 40 }}>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap' }}>
                                          <Chip size="small" label={tc.tool_name || 'tool'} sx={{ fontFamily: 'monospace', fontSize: '0.7rem' }} />
                                          <Chip size="small" variant="outlined" label={`iter ${tc.iteration}`} />
                                          {tc.duration_ms != null && (
                                            <Typography variant="caption" color="text.secondary">{formatDuration(tc.duration_ms)}</Typography>
                                          )}
                                          {tc.status === 'failed' && (
                                            <Chip size="small" color="error" label="failed" />
                                          )}
                                        </Box>
                                      </AccordionSummary>
                                      <AccordionDetails sx={{ pt: 0, flexDirection: 'column', gap: 0.5 }}>
                                        {tc.error && (
                                          <Alert severity="error" sx={{ py: 0.5 }}>
                                            <Typography variant="caption" component="pre" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{tc.error}</Typography>
                                          </Alert>
                                        )}
                                        {tc.args && Object.keys(tc.args).length > 0 && (
                                          <Box>
                                            <Typography variant="caption" color="text.secondary" display="block">Args</Typography>
                                            <Box component="pre" sx={{ p: 0.75, bgcolor: 'action.hover', borderRadius: 1, fontSize: '0.65rem', overflow: 'auto', maxHeight: 80, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                                              {JSON.stringify(tc.args, null, 2)}
                                            </Box>
                                          </Box>
                                        )}
                                        {tc.result != null && tc.result !== '' && (
                                          <Box>
                                            <Typography variant="caption" color="text.secondary" display="block">Result</Typography>
                                            <Box component="pre" sx={{ p: 0.75, bgcolor: 'action.hover', borderRadius: 1, fontSize: '0.65rem', overflow: 'auto', maxHeight: 100, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                                              {tc.result}
                                            </Box>
                                          </Box>
                                        )}
                                      </AccordionDetails>
                                    </Accordion>
                                  ))}
                                </Box>
                              ) : (
                                <Typography variant="caption" color="text.secondary" fontStyle="italic">No tool calls</Typography>
                              )}
                            </Box>
                          )}
                          {step.step_type === 'deep_agent' && step.outputs && Array.isArray(step.outputs.phase_trace) && step.outputs.phase_trace.length > 0 && (
                            <Box>
                              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>Phase trace</Typography>
                              <List dense disablePadding sx={{ bgcolor: 'action.hover', borderRadius: 1, py: 0.5 }}>
                                {step.outputs.phase_trace.map((pt, ptIdx) => (
                                  <ListItem key={ptIdx} disablePadding sx={{ py: 0.25, px: 1 }}>
                                    <ListItemText
                                      primary={
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap' }}>
                                          <Typography variant="caption" fontWeight={600}>{pt.phase || `Phase ${ptIdx + 1}`}</Typography>
                                          <Chip size="small" variant="outlined" label={pt.type || '—'} sx={{ fontFamily: 'monospace', fontSize: '0.65rem' }} />
                                          {pt.duration_ms != null && (
                                            <Typography variant="caption" color="text.secondary">{formatDuration(pt.duration_ms)}</Typography>
                                          )}
                                          {pt.score != null && (
                                            <Typography variant="caption" color="text.secondary">score: {(pt.score * 100).toFixed(0)}%</Typography>
                                          )}
                                          {pt.pass === true && <Chip size="small" color="success" label="pass" />}
                                          {pt.pass === false && <Chip size="small" color="default" label="retry" />}
                                        </Box>
                                      }
                                    />
                                  </ListItem>
                                ))}
                              </List>
                            </Box>
                          )}
                          {step.outputs && Object.keys(step.outputs).length > 0 && (
                            <Box>
                              <Typography variant="caption" color="text.secondary" display="block">Outputs</Typography>
                              <Box component="pre" sx={{ p: 1, bgcolor: 'action.hover', borderRadius: 1, fontSize: '0.7rem', overflow: 'auto', maxHeight: 160, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                                {JSON.stringify(step.outputs, null, 2)}
                              </Box>
                            </Box>
                          )}
                        </AccordionDetails>
                      </Accordion>
                    );
                  })}
                </>
              )}

              {executionDetail.metadata && Object.keys(executionDetail.metadata).length > 0 && (
                <Accordion disableGutters sx={{ mt: 2, '&:before': { display: 'none' } }}>
                  <AccordionSummary expandIcon={<ExpandMore />}>Raw metadata</AccordionSummary>
                  <AccordionDetails>
                    <Box
                      component="pre"
                      sx={{
                        p: 1,
                        bgcolor: 'action.hover',
                        borderRadius: 1,
                        fontSize: '0.75rem',
                        overflow: 'auto',
                        maxHeight: 200,
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                      }}
                    >
                      {JSON.stringify(executionDetail.metadata, null, 2)}
                    </Box>
                  </AccordionDetails>
                </Accordion>
              )}
            </>
          )}
        </Box>
      </Drawer>

      <Dialog open={clearAllConfirm} onClose={() => !clearExecutionsMutation.isLoading && setClearAllConfirm(false)}>
        <DialogTitle>Clear all runs</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Permanently delete all execution history for this agent? This cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setClearAllConfirm(false)} disabled={clearExecutionsMutation.isLoading}>Cancel</Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => profileId && clearExecutionsMutation.mutate(profileId)}
            disabled={clearExecutionsMutation.isLoading}
          >
            {clearExecutionsMutation.isLoading ? 'Clearing…' : 'Clear all'}
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={!!deleteOneConfirm} onClose={() => !deleteExecutionMutation.isLoading && setDeleteOneConfirm(null)}>
        <DialogTitle>Delete run</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Permanently delete this execution? This cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteOneConfirm(null)} disabled={deleteExecutionMutation.isLoading}>Cancel</Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => deleteOneConfirm && profileId && deleteExecutionMutation.mutate({ profileId, executionId: deleteOneConfirm.id })}
            disabled={deleteExecutionMutation.isLoading}
          >
            {deleteExecutionMutation.isLoading ? 'Deleting…' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
