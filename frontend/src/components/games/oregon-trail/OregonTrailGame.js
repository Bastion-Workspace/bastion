import React, { useState, useEffect, useCallback } from 'react';
import { Box, Typography } from '@mui/material';
import apiService from '../../../services/apiService';
import SetupScreen from './SetupScreen';
import GameScreen from './GameScreen';
import OregonUiToolbar from './OregonUiToolbar';
import { OREGON_UI_STORAGE_KEY, useOregonTrailStyles } from './oregonTrailStyles';

const readStoredVariant = () => {
  try {
    const v = localStorage.getItem(OREGON_UI_STORAGE_KEY);
    if (v === 'bbs' || v === 'app') return v;
  } catch (_) {}
  return 'app';
};

const OregonTrailGame = ({ userId, onBack }) => {
  const [phase, setPhase] = useState('menu');
  const [gameId, setGameId] = useState(null);
  const [gameState, setGameState] = useState(null);
  const [saves, setSaves] = useState([]);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [uiVariant, setUiVariantState] = useState(readStoredVariant);

  const setUiVariant = useCallback((v) => {
    setUiVariantState(v);
    try {
      localStorage.setItem(OREGON_UI_STORAGE_KEY, v);
    } catch (_) {}
  }, []);

  const styles = useOregonTrailStyles(uiVariant);

  const loadSaves = useCallback(async () => {
    try {
      const res = await apiService.get('/api/games/oregon-trail/saves');
      setSaves(res?.saves || []);
    } catch {
      setSaves([]);
    }
  }, []);

  const loadModels = useCallback(async () => {
    try {
      const res = await apiService.get('/api/games/oregon-trail/models');
      setModels(res?.models || []);
    } catch {
      setModels([]);
    }
  }, []);

  useEffect(() => {
    loadSaves();
    loadModels();
  }, [loadSaves, loadModels]);

  const handleNewGame = async ({ leaderName, partyNames, profession, modelId }) => {
    setLoading(true);
    setError('');
    try {
      const res = await apiService.post('/api/games/oregon-trail/new', {
        leader_name: leaderName,
        party_names: partyNames,
        profession,
        model_id: modelId,
      });
      if (res?.game_id) {
        setGameId(res.game_id);
        setGameState(res);
        setPhase('playing');
      } else {
        setError(res?.detail || 'Failed to start game');
      }
    } catch (e) {
      setError(e.message || 'Failed to start game');
    } finally {
      setLoading(false);
    }
  };

  const handleResume = async (gid) => {
    setLoading(true);
    try {
      const res = await apiService.get(`/api/games/oregon-trail/${gid}`);
      if (res?.game_id) {
        setGameId(gid);
        setGameState(res);
        setPhase('playing');
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleQuit = () => {
    setPhase('menu');
    setGameId(null);
    setGameState(null);
    loadSaves();
  };

  if (phase === 'setup') {
    return (
      <SetupScreen
        models={models}
        loading={loading}
        error={error}
        onStart={handleNewGame}
        onBack={() => setPhase('menu')}
        uiVariant={uiVariant}
        onUiVariantChange={setUiVariant}
      />
    );
  }

  if (phase === 'playing' && gameState) {
    return (
      <GameScreen
        gameId={gameId}
        initialState={gameState}
        onQuit={handleQuit}
        uiVariant={uiVariant}
        onUiVariantChange={setUiVariant}
      />
    );
  }

  const activeSaves = saves.filter((s) => !s.is_finished);

  return (
    <Box sx={styles.containerSx}>
      <OregonUiToolbar
        onBack={onBack}
        title="Oregon Trail"
        uiVariant={uiVariant}
        onUiVariantChange={setUiVariant}
        styles={styles}
      />

      <Box sx={styles.screenSx({ p: 2 })}>
        <pre
          style={{
            margin: 0,
            fontFamily: 'inherit',
            fontSize: 'inherit',
            lineHeight: 1.4,
          }}
        >
          {TITLE_ART}
        </pre>

        <Box sx={{ mt: 2 }}>
          <TerminalButton styles={styles} onClick={() => setPhase('setup')}>
            [N] New Game
          </TerminalButton>
          {activeSaves.map((s, i) => (
            <TerminalButton key={s.game_id} styles={styles} onClick={() => handleResume(s.game_id)}>
              [{i + 1}] Resume: {s.leader_name} — Day {s.day_number}, {s.miles_traveled} mi
            </TerminalButton>
          ))}
        </Box>

        {error && (
          <Typography sx={{ color: styles.colors.error, mt: 1, fontFamily: styles.mono }}>
            {error}
          </Typography>
        )}
        {loading && (
          <Typography sx={{ color: styles.colors.loading, mt: 1, fontFamily: styles.mono }}>
            Loading...
          </Typography>
        )}
      </Box>
    </Box>
  );
};

const TerminalButton = ({ onClick, children, styles }) => (
  <Box
    onClick={onClick}
    role="button"
    tabIndex={0}
    onKeyDown={(e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        onClick();
      }
    }}
    sx={{
      cursor: 'pointer',
      fontFamily: styles.mono,
      py: 0.5,
      px: 1,
      my: 0.5,
      borderRadius: 1,
      transition: 'background 0.15s',
      '&:hover': { bgcolor: styles.terminalButtonHover },
    }}
  >
    {children}
  </Box>
);

const TITLE_ART = `
   ╔═══════════════════════════════════════════════╗
   ║          THE OREGON TRAIL — 1848              ║
   ║                                               ║
   ║   2,170 miles from Independence, Missouri     ║
   ║        to Oregon's Willamette Valley           ║
   ╚═══════════════════════════════════════════════╝`;

export default OregonTrailGame;
