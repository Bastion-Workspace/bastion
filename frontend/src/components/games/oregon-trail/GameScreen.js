import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Box, Typography, TextField, CircularProgress, ToggleButton, ToggleButtonGroup } from '@mui/material';
import apiService from '../../../services/apiService';
import OregonUiToolbar from './OregonUiToolbar';
import { useOregonTrailStyles } from './oregonTrailStyles';

const GameScreen = ({ gameId, initialState, onQuit, uiVariant, onUiVariantChange }) => {
  const styles = useOregonTrailStyles(uiVariant);
  const [gs, setGs] = useState(initialState);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [talkMode, setTalkMode] = useState(false);
  const [journal, setJournal] = useState(null);
  const logRef = useRef(null);
  const inputRef = useRef(null);
  const [log, setLog] = useState(() => {
    const n = initialState?.narrative;
    if (!n || !String(n).trim()) return [];
    return String(n)
      .split(/\n\n+/)
      .map((s) => s.trim())
      .filter(Boolean);
  });

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  useEffect(() => {
    if (!loading && inputRef.current) inputRef.current.focus();
  }, [loading]);

  const appendLog = useCallback((text) => {
    if (text) setLog((prev) => [...prev, text]);
  }, []);

  const doAction = async (action, detail, quantity) => {
    const gid = (gameId && String(gameId).trim()) || '';
    if (!gid || gid === 'undefined') {
      appendLog('Error: no game id — go back and start or resume a game.');
      return;
    }
    setLoading(true);
    try {
      const body = { action };
      if (detail) body.detail = detail;
      if (quantity != null) body.quantity = quantity;
      const res = await apiService.post(`/api/games/oregon-trail/${encodeURIComponent(gid)}/action`, body);
      if (res?.narrative) {
        const chunks = String(res.narrative)
          .split(/\n\n+/)
          .map((s) => s.trim())
          .filter(Boolean);
        if (chunks.length) setLog((prev) => [...prev, ...chunks]);
      }
      if (res?.game_id) setGs(res);
    } catch (e) {
      appendLog(`Error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const doTalk = async (message) => {
    const gid = (gameId && String(gameId).trim()) || '';
    if (!gid || gid === 'undefined') return;
    setLoading(true);
    try {
      const res = await apiService.post(`/api/games/oregon-trail/${encodeURIComponent(gid)}/talk`, { message });
      if (res?.dialogue) appendLog(`${res.npc_name || 'Stranger'}: ${res.dialogue}`);
      if (res?.game_id) setGs(res);
    } catch (e) {
      appendLog(`Error: ${e.message}`);
    } finally {
      setLoading(false);
      setTalkMode(false);
    }
  };

  const showJournal = async () => {
    const gid = (gameId && String(gameId).trim()) || '';
    if (!gid || gid === 'undefined') return;
    try {
      const res = await apiService.get(`/api/games/oregon-trail/${encodeURIComponent(gid)}/journal`);
      setJournal(res?.journal || []);
    } catch {
      setJournal([]);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const val = input.trim();
    if (!val) return;
    setInput('');

    if (val.toLowerCase() === 'quit' || val.toLowerCase() === '/quit') {
      onQuit();
      return;
    }
    if (val.toLowerCase() === 'journal') {
      showJournal();
      return;
    }

    if (talkMode) {
      appendLog(`You: ${val}`);
      doTalk(val);
      return;
    }

    const lower = val.toLowerCase();
    if (lower.startsWith('>') || lower.startsWith('custom ')) {
      const detail = val.replace(/^>|^custom\s+/i, '').trim();
      appendLog(`> ${detail}`);
      doAction('custom', detail);
      return;
    }
    if (lower.startsWith('talk')) {
      if (lower === 'talk') {
        setTalkMode(true);
        appendLog('[Talk mode — type what you say, or "cancel"]');
        return;
      }
      appendLog(`You: ${val.slice(5)}`);
      doTalk(val.slice(5));
      return;
    }
    if (lower.startsWith('pace ')) {
      doAction('pace', lower.slice(5).trim());
      return;
    }
    if (lower.startsWith('rations ')) {
      doAction('rations', lower.slice(8).trim());
      return;
    }
    if (lower.startsWith('trade ')) {
      const parts = lower.slice(6).trim().split(/\s+/);
      const qty = parts.length > 1 && !isNaN(parts[parts.length - 1]) ? parseInt(parts.pop(), 10) : null;
      doAction('trade', parts.join(' '), qty);
      return;
    }
    if (lower.startsWith('buy_') || lower.startsWith('rest ') || lower === 'status') {
      const parts = lower.split(/\s+/);
      const qty = parts.length > 1 && !isNaN(parts[1]) ? parseInt(parts[1], 10) : null;
      doAction(parts[0], null, qty);
      return;
    }

    doAction(lower);
  };

  const variantToggle = (
    <ToggleButtonGroup
      size="small"
      value={uiVariant}
      exclusive
      onChange={(_, v) => {
        if (v) onUiVariantChange(v);
      }}
      aria-label="Oregon Trail display style"
    >
      <ToggleButton value="app">App</ToggleButton>
      <ToggleButton value="bbs" sx={styles.isBbs ? { fontFamily: styles.mono } : undefined}>
        Terminal
      </ToggleButton>
    </ToggleButtonGroup>
  );

  if (journal !== null) {
    return (
      <Box sx={styles.containerSx}>
        <OregonUiToolbar
          onBack={() => setJournal(null)}
          title="Trail Journal"
          uiVariant={uiVariant}
          onUiVariantChange={onUiVariantChange}
          styles={styles}
        />
        <Box sx={styles.screenSx({ p: 1.5 })} ref={logRef}>
          {journal.length === 0 && (
            <Typography sx={{ fontFamily: styles.mono, color: styles.colors.loading }}>
              No entries yet.
            </Typography>
          )}
          {journal.map((e, i) => (
            <Box key={i} sx={{ mb: 1.5 }}>
              <Typography
                sx={{ fontFamily: styles.mono, color: styles.colors.journalMeta, fontSize: '12px' }}
              >
                Day {e.day} — {e.game_date} ({e.location})
              </Typography>
              <Typography sx={{ fontFamily: styles.mono, fontSize: '13px', whiteSpace: 'pre-wrap' }}>
                {e.text}
              </Typography>
            </Box>
          ))}
        </Box>
      </Box>
    );
  }

  const actions = gs?.available_actions || [];
  const party = gs?.party || [];
  const res = gs?.resources || {};
  const finished = gs?.is_finished;
  const c = styles.colors;

  const partyColor = (m) => {
    if (!m.is_alive) return c.partyDead;
    if (m.status === 'good') return c.partyGood;
    if (m.status === 'fair') return c.partyFair;
    return c.partyPoor;
  };

  return (
    <Box sx={{ ...styles.containerSx, p: 1.5 }}>
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>{variantToggle}</Box>

      <Box
        sx={{
          borderBottom: 1,
          borderColor: styles.isBbs ? '#1a3a1a' : 'divider',
          pb: 1,
          mb: 1,
          fontSize: '12px',
          fontFamily: styles.mono,
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 0.5 }}>
          <span>
            Day {gs?.day_number} — {gs?.game_date}
          </span>
          <span style={{ color: c.location }}>{gs?.location}</span>
          <span>
            {gs?.miles_traveled}/{gs?.total_miles} mi
          </span>
        </Box>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 0.5 }}>
          <span>Weather: {gs?.weather?.replace(/_/g, ' ')}</span>
          <span>Pace: {gs?.pace}</span>
          <span>Rations: {gs?.rations?.replace(/_/g, ' ')}</span>
        </Box>
        <ProgressBar miles={gs?.miles_traveled || 0} total={gs?.total_miles || 2170} color={c.progress} />
        <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', mt: 0.5 }}>
          {party.map((m) => (
            <span key={m.name} style={{ color: partyColor(m) }}>
              {m.name}({m.is_alive ? m.status?.replace(/_/g, ' ') : 'Dead'})
            </span>
          ))}
        </Box>
        <Box sx={{ mt: 0.5, color: c.resources }}>
          Food:{res.food} Ammo:{res.ammunition} Parts:{res.spare_parts} Clothes:{res.clothing} Oxen:
          {res.oxen} ${(res.money || 0).toFixed(2)}
        </Box>
      </Box>

      <Box sx={styles.screenSx({ p: 1.5 })} ref={logRef}>
        {log.map((entry, i) => (
          <Typography
            key={i}
            sx={{
              fontFamily: styles.mono,
              fontSize: '13px',
              mb: 1,
              whiteSpace: 'pre-wrap',
              lineHeight: 1.5,
            }}
          >
            {entry}
          </Typography>
        ))}
        {loading && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: c.loading }}>
            <CircularProgress size={14} sx={{ color: styles.circularProgressColor }} />
            <Typography sx={{ fontFamily: styles.mono, fontSize: '12px' }}>Working...</Typography>
          </Box>
        )}
      </Box>

      {!finished && actions.length > 0 && (
        <Box
          sx={{
            py: 0.5,
            fontSize: '12px',
            fontFamily: styles.mono,
            color: c.hint,
            display: 'flex',
            flexWrap: 'wrap',
            gap: 1,
          }}
        >
          {actions.map((a) => (
            <Box
              component="span"
              key={a.key}
              onClick={() => {
                if (!loading) doAction(a.key);
              }}
              sx={{
                cursor: loading ? 'default' : 'pointer',
                color: c.hint,
                '&:hover': { color: loading ? c.hint : c.hintHover },
              }}
            >
              [{a.key}]{a.label && ` ${a.label}`}
            </Box>
          ))}
          <Box component="span" sx={{ color: c.hint }}>
            [&gt;] custom action
          </Box>
          <Box component="span" sx={{ color: c.hint }}>
            [quit] menu
          </Box>
        </Box>
      )}

      {!finished ? (
        <form onSubmit={handleSubmit} style={{ display: 'flex', gap: 8, marginTop: 4 }}>
          <Typography
            sx={{
              fontFamily: styles.mono,
              lineHeight: '40px',
              color: talkMode ? c.talkPrompt : c.accent,
            }}
          >
            {talkMode ? 'Say>' : '>'}
          </Typography>
          <TextField
            inputRef={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
            autoFocus
            fullWidth
            size="small"
            placeholder={talkMode ? 'Type what you say...' : 'Enter command or "> custom action"'}
            sx={styles.inputSx}
          />
        </form>
      ) : (
        <Box sx={{ textAlign: 'center', py: 2 }}>
          <Typography
            sx={{
              fontFamily: styles.mono,
              fontSize: '16px',
              fontWeight: 'bold',
              color: gs?.phase === 'victory' ? styles.victoryColor : c.error,
            }}
          >
            {gs?.phase === 'victory'
              ? `YOU REACHED OREGON! Score: ${gs.final_score}`
              : 'GAME OVER'}
          </Typography>
          <Typography
            onClick={onQuit}
            sx={{
              cursor: 'pointer',
              mt: 1,
              fontFamily: styles.mono,
              color: styles.menuReturnColor,
              '&:hover': { color: styles.menuReturnHover },
            }}
          >
            [Return to menu]
          </Typography>
        </Box>
      )}
    </Box>
  );
};

const ProgressBar = ({ miles, total, color }) => {
  const pct = Math.min(100, Math.round((miles / total) * 100));
  const width = 30;
  const filled = Math.round((width * pct) / 100);
  const bar = '█'.repeat(filled) + '░'.repeat(width - filled);
  return (
    <Box sx={{ fontFamily: 'monospace', fontSize: '11px', mt: 0.5, color }}>
      [{bar}] {pct}%
    </Box>
  );
};

export default GameScreen;
