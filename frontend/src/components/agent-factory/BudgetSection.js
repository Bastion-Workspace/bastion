/**
 * Budget section: monthly limit, current spend, warning threshold, enforce hard limit.
 * Shown in Agent Editor when an agent profile is selected.
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  FormControlLabel,
  Switch,
  LinearProgress,
  Alert,
} from '@mui/material';
import { useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';

function formatUsd(val) {
  if (val == null || val === '') return '—';
  const n = Number(val);
  if (Number.isNaN(n)) return '—';
  if (n === 0) return '$0';
  if (n < 0.01) return '<$0.01';
  return `$${n.toFixed(2)}`;
}

export default function BudgetSection({ profileId, budget }) {
  const queryClient = useQueryClient();
  const [monthlyLimit, setMonthlyLimit] = useState(
    budget?.monthly_limit_usd != null ? String(budget.monthly_limit_usd) : ''
  );
  const [warningPct, setWarningPct] = useState(budget?.warning_threshold_pct ?? 80);
  const [enforceHard, setEnforceHard] = useState(budget?.enforce_hard_limit ?? true);

  const updateBudgetMutation = useMutation(
    ({ id, body }) => apiService.agentFactory.updateProfileBudget(id, body),
    {
      onSuccess: (_, { id }) => {
        queryClient.invalidateQueries(['agentFactoryProfile', id]);
        queryClient.invalidateQueries('agentFactoryProfiles');
      },
    }
  );

  const handleSave = () => {
    if (!profileId) return;
    const limit = monthlyLimit.trim() === '' ? null : parseFloat(monthlyLimit);
    if (limit !== null && (Number.isNaN(limit) || limit < 0)) return;
    updateBudgetMutation.mutate({
      id: profileId,
      body: {
        monthly_limit_usd: limit,
        warning_threshold_pct: warningPct,
        enforce_hard_limit: enforceHard,
      },
    });
  };

  const spend = budget?.current_period_spend_usd ?? 0;
  const limit = budget?.monthly_limit_usd;
  const hasLimit = limit != null && limit > 0;
  const pct = hasLimit ? Math.min(100, (spend / limit) * 100) : 0;
  const overLimit = hasLimit && spend >= limit;

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Budget
        </Typography>
        {budget != null && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              This period: {formatUsd(spend)}
              {hasLimit ? ` / ${formatUsd(limit)}` : ' (no limit)'}
            </Typography>
            {hasLimit && (
              <LinearProgress
                variant="determinate"
                value={pct}
                color={overLimit ? 'error' : pct >= (budget.warning_threshold_pct ?? 80) ? 'warning' : 'primary'}
                sx={{ mt: 0.5, height: 6, borderRadius: 1 }}
              />
            )}
            {overLimit && (
              <Alert severity="error" sx={{ mt: 1 }}>
                Monthly limit reached. Scheduled runs are paused until next period or limit is increased.
              </Alert>
            )}
          </Box>
        )}
        <TextField
          fullWidth
          size="small"
          label="Monthly limit (USD)"
          type="number"
          inputProps={{ min: 0, step: 1 }}
          value={monthlyLimit}
          onChange={(e) => setMonthlyLimit(e.target.value)}
          placeholder="Leave empty for unlimited"
          sx={{ mb: 2 }}
        />
        <TextField
          fullWidth
          size="small"
          label="Warn at (%)"
          type="number"
          inputProps={{ min: 1, max: 100 }}
          value={warningPct}
          onChange={(e) => setWarningPct(parseInt(e.target.value, 10) || 80)}
          sx={{ mb: 1 }}
        />
        <FormControlLabel
          control={
            <Switch
              checked={enforceHard}
              onChange={(e) => setEnforceHard(e.target.checked)}
              color="primary"
            />
          }
          label="Pause when limit reached"
          sx={{ mb: 2 }}
        />
        <Button
          variant="contained"
          size="small"
          onClick={handleSave}
          disabled={updateBudgetMutation.isLoading}
        >
          {updateBudgetMutation.isLoading ? 'Saving…' : 'Save budget'}
        </Button>
      </CardContent>
    </Card>
  );
}
