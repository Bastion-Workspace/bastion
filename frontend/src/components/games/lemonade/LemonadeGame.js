import React, { useState, useEffect, useCallback } from 'react';
import { Box, IconButton, Typography, Button } from '@mui/material';
import { ArrowBack, Refresh } from '@mui/icons-material';
import { newGame, runDay, getForecast, TOTAL_DAYS } from './lemonadeEngine';
import LemonadeDay from './LemonadeDay';
import LemonadeResults from './LemonadeResults';
import LemonadeChart from './LemonadeChart';

const STORAGE_KEY_PREFIX = 'bastion_lemonade_';

const LemonadeGame = ({ userId, onBack }) => {
  const [state, setState] = useState(null);

  const loadOrNew = useCallback(() => {
    try {
      const raw = localStorage.getItem(`${STORAGE_KEY_PREFIX}${userId}`);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed.phase !== 'gameover' || parsed.day <= TOTAL_DAYS) {
          setState(parsed);
          return;
        }
      }
    } catch (_) {}
    setState(newGame());
  }, [userId]);

  useEffect(() => {
    loadOrNew();
  }, [loadOrNew]);

  useEffect(() => {
    if (!state || state.phase !== 'planning' || state.forecast) return;
    setState((prev) => ({ ...prev, forecast: getForecast(prev.day).forecast }));
  }, [state?.phase, state?.day, state?.forecast]);

  useEffect(() => {
    if (!state || !userId) return;
    try {
      localStorage.setItem(`${STORAGE_KEY_PREFIX}${userId}`, JSON.stringify(state));
    } catch (_) {}
  }, [state, userId]);

  const handleOpenStand = (choices) => {
    const next = runDay(state, choices);
    setState(next);
  };

  const handleNextDay = () => {
    setState((prev) => ({
      ...prev,
      phase: 'planning',
      lastResult: null,
    }));
  };

  if (!state) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography>Loading...</Typography>
      </Box>
    );
  }

  if (state.phase === 'gameover') {
    const totalProfit = state.history.reduce((sum, h) => sum + h.profit, 0);
    return (
      <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, alignSelf: 'stretch' }}>
          <IconButton onClick={onBack} size="small" aria-label="Back">
            <ArrowBack />
          </IconButton>
          <Typography variant="h6">Game Over</Typography>
        </Box>
        <Typography>30 days complete. Final cash: ${state.money.toFixed(2)}</Typography>
        <Typography color="text.secondary">Total profit over 30 days: ${totalProfit.toFixed(2)}</Typography>
        <LemonadeChart history={state.history} />
        <Button variant="contained" startIcon={<Refresh />} onClick={loadOrNew}>
          New Game
        </Button>
      </Box>
    );
  }

  const forecast = state.forecast || getForecast(state.day).forecast;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 1, borderBottom: 1, borderColor: 'divider' }}>
        <IconButton onClick={onBack} size="small" aria-label="Back to games">
          <ArrowBack />
        </IconButton>
        <Typography variant="h6">Lemonade Stand</Typography>
        <Button size="small" startIcon={<Refresh />} onClick={loadOrNew} sx={{ ml: 'auto' }}>
          New Game
        </Button>
      </Box>
      <Box sx={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
        {state.phase === 'planning' && (
          <LemonadeDay state={state} forecast={forecast} onOpenStand={handleOpenStand} />
        )}
        {state.phase === 'results' && (
          <>
            <LemonadeResults lastResult={state.lastResult} onNext={handleNextDay} />
            <Box sx={{ p: 2 }}>
              <LemonadeChart history={state.history} />
            </Box>
          </>
        )}
      </Box>
    </Box>
  );
};

export default LemonadeGame;
