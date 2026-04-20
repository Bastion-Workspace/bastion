import { useMemo } from 'react';
import { useTheme } from '@mui/material/styles';

export const OREGON_UI_STORAGE_KEY = 'bastion_oregon_trail_ui_variant';

/** @typedef {'app' | 'bbs'} OregonUiVariant */

/**
 * @param {import('@mui/material/styles').Theme} theme
 * @param {OregonUiVariant} variant
 */
function buildStyles(theme, variant) {
  const isBbs = variant === 'bbs';
  const mono = '"Roboto Mono", "Courier New", Courier, monospace';

  if (isBbs) {
    const scanlineOverlay = {
      content: '""',
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background:
        'repeating-linear-gradient(0deg, rgba(0,0,0,0.15) 0px, rgba(0,0,0,0.15) 1px, transparent 1px, transparent 3px)',
      pointerEvents: 'none',
      zIndex: 1,
    };
    return {
      isBbs: true,
      mono,
      containerSx: {
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: '#0a0a0a',
        color: '#33ff33',
        fontFamily: mono,
        fontSize: '14px',
        overflow: 'hidden',
        p: 2,
      },
      screenSx: (extra = {}) => ({
        flex: 1,
        overflow: 'auto',
        border: '1px solid #1a3a1a',
        borderRadius: 1,
        bgcolor: '#050505',
        '&::-webkit-scrollbar': { width: 6 },
        '&::-webkit-scrollbar-thumb': { bgcolor: '#1a3a1a', borderRadius: 3 },
        position: 'relative',
        '&::before': scanlineOverlay,
        ...extra,
      }),
      inputSx: {
        '& .MuiInputBase-root': {
          fontFamily: mono,
          color: '#33ff33',
          bgcolor: '#0d0d0d',
        },
        '& .MuiInputLabel-root': { color: '#1a8a1a' },
        '& .MuiOutlinedInput-notchedOutline': { borderColor: '#1a3a1a' },
        '& .MuiOutlinedInput-root:hover .MuiOutlinedInput-notchedOutline': { borderColor: '#33ff33' },
        '& .MuiOutlinedInput-root.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: '#33ff33' },
      },
      selectSx: {
        fontFamily: mono,
        color: '#33ff33',
        bgcolor: '#0d0d0d',
        '& .MuiOutlinedInput-notchedOutline': { borderColor: '#1a3a1a' },
        '& .MuiSvgIcon-root': { color: '#33ff33' },
      },
      menuPaperSx: { bgcolor: '#111', color: '#33ff33' },
      colors: {
        accent: '#33ff33',
        accentMuted: '#1a8a1a',
        location: '#88ccff',
        partyGood: '#33ff33',
        partyFair: '#cccc33',
        partyPoor: '#ff9933',
        partyDead: '#555',
        resources: '#aaa',
        hint: '#1a8a1a',
        hintHover: '#33ff33',
        error: '#ff6b6b',
        loading: '#888',
        journalMeta: '#88cc88',
        talkPrompt: '#cccc33',
        progress: '#33ff33',
      },
      primaryButtonSx: {
        fontFamily: mono,
        color: '#33ff33',
        borderColor: '#33ff33',
        '&:hover': { borderColor: '#66ff66', bgcolor: 'rgba(0,255,0,0.08)' },
        '&.Mui-disabled': { color: '#1a3a1a', borderColor: '#1a3a1a' },
      },
      circularProgressColor: '#33ff33',
      terminalButtonHover: 'rgba(0,255,0,0.1)',
      victoryColor: '#33ff33',
      menuReturnColor: '#888',
      menuReturnHover: '#33ff33',
    };
  }

  const divider = theme.palette.divider;
  const screenBg =
    theme.palette.mode === 'dark' ? theme.palette.grey[900] : theme.palette.grey[100];

  return {
    isBbs: false,
    mono,
    containerSx: {
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      bgcolor: 'background.default',
      color: 'text.primary',
      fontFamily: theme.typography.fontFamily,
      fontSize: theme.typography.body2?.fontSize ?? '0.875rem',
      overflow: 'hidden',
      p: 2,
    },
    screenSx: (extra = {}) => ({
      flex: 1,
      overflow: 'auto',
      border: `1px solid ${divider}`,
      borderRadius: 1,
      bgcolor: screenBg,
      '&::-webkit-scrollbar': { width: 6 },
      '&::-webkit-scrollbar-thumb': {
        bgcolor: theme.palette.action.hover,
        borderRadius: 3,
      },
      position: 'relative',
      ...extra,
    }),
    inputSx: {
      '& .MuiInputBase-root': {
        fontFamily: mono,
      },
    },
    selectSx: {
      fontFamily: mono,
    },
    menuPaperSx: {},
    colors: {
      accent: theme.palette.primary.main,
      accentMuted: theme.palette.text.secondary,
      location: theme.palette.info.main,
      partyGood: theme.palette.success.main,
      partyFair: theme.palette.warning.main,
      partyPoor: theme.palette.warning.dark,
      partyDead: theme.palette.action.disabled,
      resources: theme.palette.text.secondary,
      hint: theme.palette.text.secondary,
      hintHover: theme.palette.primary.main,
      error: theme.palette.error.main,
      loading: theme.palette.text.secondary,
      journalMeta: theme.palette.text.secondary,
      talkPrompt: theme.palette.warning.main,
      progress: theme.palette.primary.main,
    },
    primaryButtonSx: {},
    circularProgressColor: theme.palette.primary.main,
    terminalButtonHover: theme.palette.action.hover,
    victoryColor: theme.palette.success.main,
    menuReturnColor: theme.palette.text.secondary,
    menuReturnHover: theme.palette.primary.main,
  };
}

/**
 * @param {OregonUiVariant} variant
 */
export function useOregonTrailStyles(variant) {
  const theme = useTheme();
  return useMemo(() => buildStyles(theme, variant), [theme, variant]);
}
