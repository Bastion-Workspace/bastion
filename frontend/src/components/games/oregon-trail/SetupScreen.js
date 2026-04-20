import React, { useState } from 'react';
import { Box, Typography, TextField, Select, MenuItem, Button, CircularProgress } from '@mui/material';
import OregonUiToolbar from './OregonUiToolbar';
import { useOregonTrailStyles } from './oregonTrailStyles';

const PROFESSIONS = [
  { value: 'banker', label: 'Banker', desc: '$1,600 starting money' },
  { value: 'carpenter', label: 'Carpenter', desc: '$800, 2x score multiplier' },
  { value: 'farmer', label: 'Farmer', desc: '$400, 3x score multiplier' },
];

const SetupScreen = ({ models, loading, error, onStart, onBack, uiVariant, onUiVariantChange }) => {
  const styles = useOregonTrailStyles(uiVariant);
  const [leaderName, setLeaderName] = useState('');
  const [companion1, setCompanion1] = useState('');
  const [companion2, setCompanion2] = useState('');
  const [companion3, setCompanion3] = useState('');
  const [profession, setProfession] = useState('banker');
  const [modelId, setModelId] = useState(models[0] || '');

  const handleSubmit = (e) => {
    e.preventDefault();
    const partyNames = [companion1, companion2, companion3].filter((n) => n.trim());
    onStart({
      leaderName: leaderName.trim() || 'Pioneer',
      partyNames: partyNames.length ? partyNames : ['Mary', 'Tom', 'Sara'],
      profession,
      modelId: modelId || models[0] || '',
    });
  };

  const inputSx = styles.inputSx;

  return (
    <Box sx={styles.containerSx}>
      <OregonUiToolbar
        onBack={onBack}
        title="New Game"
        uiVariant={uiVariant}
        onUiVariantChange={onUiVariantChange}
        styles={styles}
      />

      <Box sx={styles.screenSx({ p: 3, maxWidth: 600 })}>
        <form onSubmit={handleSubmit}>
          <Typography sx={{ mb: 2, fontFamily: styles.mono }}>═══ Party Setup ═══</Typography>

          <TextField
            label="Your name"
            value={leaderName}
            onChange={(e) => setLeaderName(e.target.value)}
            placeholder="Pioneer"
            fullWidth
            size="small"
            sx={{ ...inputSx, mb: 2 }}
          />

          <Typography
            sx={{
              mb: 1,
              fontFamily: styles.mono,
              color: styles.colors.accentMuted,
              fontSize: '13px',
            }}
          >
            Companions (up to 3):
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
            <TextField
              placeholder="Mary"
              value={companion1}
              onChange={(e) => setCompanion1(e.target.value)}
              size="small"
              sx={inputSx}
            />
            <TextField
              placeholder="Tom"
              value={companion2}
              onChange={(e) => setCompanion2(e.target.value)}
              size="small"
              sx={inputSx}
            />
            <TextField
              placeholder="Sara"
              value={companion3}
              onChange={(e) => setCompanion3(e.target.value)}
              size="small"
              sx={inputSx}
            />
          </Box>

          <Typography sx={{ mb: 1, fontFamily: styles.mono }}>═══ Profession ═══</Typography>
          <Select
            value={profession}
            onChange={(e) => setProfession(e.target.value)}
            fullWidth
            size="small"
            sx={{ ...styles.selectSx, mb: 2 }}
            MenuProps={{ PaperProps: { sx: styles.menuPaperSx } }}
          >
            {PROFESSIONS.map((p) => (
              <MenuItem key={p.value} value={p.value} sx={{ fontFamily: styles.mono }}>
                {p.label} — {p.desc}
              </MenuItem>
            ))}
          </Select>

          {models.length > 1 && (
            <>
              <Typography sx={{ mb: 1, fontFamily: styles.mono }}>═══ Narrator Model ═══</Typography>
              <Select
                value={modelId || models[0] || ''}
                onChange={(e) => setModelId(e.target.value)}
                fullWidth
                size="small"
                sx={{ ...styles.selectSx, mb: 2 }}
                MenuProps={{ PaperProps: { sx: styles.menuPaperSx } }}
              >
                {models.map((m) => (
                  <MenuItem key={m} value={m} sx={{ fontFamily: styles.mono }}>
                    {m.split('/').pop()}
                  </MenuItem>
                ))}
              </Select>
            </>
          )}

          {error && (
            <Typography sx={{ color: styles.colors.error, mb: 1, fontFamily: styles.mono }}>
              {error}
            </Typography>
          )}

          <Button
            type="submit"
            variant={styles.isBbs ? 'outlined' : 'contained'}
            color="primary"
            disabled={loading}
            fullWidth
            sx={{
              fontFamily: styles.mono,
              mt: 1,
              ...(styles.isBbs ? styles.primaryButtonSx : {}),
            }}
          >
            {loading ? (
              <>
                <CircularProgress size={16} sx={{ color: styles.circularProgressColor, mr: 1 }} />
                Starting...
              </>
            ) : (
              'Hit the Trail!'
            )}
          </Button>
        </form>
      </Box>
    </Box>
  );
};

export default SetupScreen;
