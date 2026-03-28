import React, { useState, useEffect, useCallback } from 'react';
import { Box, IconButton, Typography, Button } from '@mui/material';
import { ArrowBack, Refresh, Undo } from '@mui/icons-material';
import SolitaireBoard from './SolitaireBoard';
import { newGame, drawFromStock, flipTableauCard, isWon } from './solitaireEngine';

const STORAGE_KEY_PREFIX = 'bastion_solitaire_';

const SolitaireGame = ({ userId, onBack }) => {
  const [state, setState] = useState(null);
  const [selection, setSelection] = useState(null);
  const [history, setHistory] = useState([]);

  const loadOrNew = useCallback(() => {
    try {
      const raw = localStorage.getItem(`${STORAGE_KEY_PREFIX}${userId}`);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (!parsed.won) {
          setState(parsed);
          setHistory([]);
          return;
        }
      }
    } catch (_) {}
    setState(newGame());
    setHistory([]);
  }, [userId]);

  useEffect(() => {
    loadOrNew();
  }, [loadOrNew]);

  useEffect(() => {
    if (!state || !userId) return;
    try {
      localStorage.setItem(`${STORAGE_KEY_PREFIX}${userId}`, JSON.stringify(state));
    } catch (_) {}
  }, [state, userId]);

  const handleDraw = () => {
    if (!state) return;
    const next = drawFromStock(state);
    setHistory((h) => [...h, state]);
    setState(next);
    setSelection(null);
  };

  const handleFlip = (colIndex) => {
    if (!state) return;
    setState(flipTableauCard(state, colIndex));
  };

  const handleMove = (nextState) => {
    setHistory((h) => [...h, state]);
    let final = nextState;
    if (isWon(nextState)) final = { ...nextState, won: true };
    setState(final);
    setSelection(null);
  };

  const handleUndo = () => {
    if (history.length === 0) return;
    const prev = history[history.length - 1];
    setHistory((h) => h.slice(0, -1));
    setState(prev);
    setSelection(null);
  };

  if (!state) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography>Loading...</Typography>
      </Box>
    );
  }

  if (state.won) {
    return (
      <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, alignSelf: 'stretch' }}>
          <IconButton onClick={onBack} size="small" aria-label="Back">
            <ArrowBack />
          </IconButton>
          <Typography variant="h6">You won!</Typography>
        </Box>
        <Typography color="text.secondary">Moves: {state.moves}</Typography>
        <Button variant="contained" startIcon={<Refresh />} onClick={loadOrNew}>
          New Game
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 1, borderBottom: 1, borderColor: 'divider' }}>
        <IconButton onClick={onBack} size="small" aria-label="Back to games">
          <ArrowBack />
        </IconButton>
        <Typography variant="h6">Solitaire</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
          Moves: {state.moves}
        </Typography>
        <IconButton onClick={handleUndo} disabled={history.length === 0} size="small" aria-label="Undo">
          <Undo />
        </IconButton>
        <Button size="small" startIcon={<Refresh />} onClick={loadOrNew} sx={{ ml: 'auto' }}>
          New Game
        </Button>
      </Box>
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <SolitaireBoard
          state={state}
          selection={selection}
          onSelect={setSelection}
          onDraw={handleDraw}
          onMove={handleMove}
          onFlip={handleFlip}
        />
      </Box>
    </Box>
  );
};

export default SolitaireGame;
