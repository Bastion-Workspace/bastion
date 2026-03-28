import React from 'react';
import { Box } from '@mui/material';

const SUIT_SYMBOLS = { s: '♠', h: '♥', d: '♦', c: '♣' };

const SolitaireCard = ({ card, selected, onClick, stacked, style, width = 56, height = 76, disableHover = false }) => {
  const isRed = card.suit === 'h' || card.suit === 'd';
  const symbol = SUIT_SYMBOLS[card.suit];

  const noSelect = { userSelect: 'none', WebkitUserSelect: 'none' };
  const hoverStyle = disableHover ? {} : { '&:hover': { bgcolor: 'primary.dark' } };
  const hoverStyleFaceUp = disableHover ? {} : { '&:hover': { bgcolor: 'action.hover' } };

  if (!card.faceUp) {
    return (
      <Box
        onClick={onClick}
        sx={{
          width,
          minWidth: width,
          height,
          minHeight: height,
          borderRadius: 1,
          bgcolor: 'primary.main',
          border: selected ? '2px solid' : '1px solid',
          borderColor: selected ? 'warning.main' : 'divider',
          boxShadow: 1,
          cursor: 'pointer',
          ...hoverStyle,
          ...noSelect,
          ...style,
        }}
      />
    );
  }

  const cornerFontSize = stacked ? 10 : 11;
  const cornerSymbolSize = stacked ? 10 : 12;

  return (
    <Box
      onClick={onClick}
      sx={{
        width,
        minWidth: width,
        height,
        minHeight: height,
        borderRadius: 1,
        bgcolor: 'background.paper',
        border: selected ? '2px solid' : '1px solid',
        borderColor: selected ? 'warning.main' : 'divider',
        boxShadow: 1,
        cursor: 'pointer',
        position: 'relative',
        color: isRed ? 'error.main' : 'text.primary',
        ...hoverStyleFaceUp,
        ...noSelect,
        ...style,
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          top: 2,
          left: 3,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          lineHeight: 1.1,
          fontSize: cornerFontSize,
          fontWeight: 700,
        }}
      >
        <Box component="span">{card.rank}</Box>
        <Box component="span" sx={{ fontSize: cornerSymbolSize }}>{symbol}</Box>
      </Box>
      <Box
        sx={{
          position: 'absolute',
          bottom: 2,
          right: 3,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          lineHeight: 1.1,
          fontSize: cornerFontSize,
          fontWeight: 700,
          transform: 'rotate(180deg)',
        }}
      >
        <Box component="span">{card.rank}</Box>
        <Box component="span" sx={{ fontSize: cornerSymbolSize }}>{symbol}</Box>
      </Box>
    </Box>
  );
};

export default SolitaireCard;
