import React from 'react';
import { Box } from '@mui/material';
import SolitaireCard from './SolitaireCard';
import {
  drawFromStock,
  moveTableauToTableau,
  moveTableauToFoundation,
  moveWasteToTableau,
  moveWasteToFoundation,
  moveFoundationToTableau,
  canMoveFromTableau,
  findTableauDropColumn,
  findFoundationDropIndex,
} from './solitaireEngine';

const CARD_WIDTH = 56;
const CARD_HEIGHT = 76;
const STACK_OFFSET = 16;

const SolitaireBoard = ({ state, selection, onSelect, onDraw, onMove, onFlip }) => {
  const getSelectedCard = () => {
    if (!selection) return null;
    if (selection.source === 'waste') {
      return state.waste.length ? state.waste[state.waste.length - 1] : null;
    }
    const col = state.tableau[selection.colIndex];
    if (!col || selection.cardIndex >= col.length) return null;
    return col[selection.cardIndex];
  };

  const handleStockClick = () => {
    onDraw();
  };

  const handleWasteClick = () => {
    if (state.waste.length === 0) return;
    const card = state.waste[state.waste.length - 1];
    if (selection && selection.source === 'waste') {
      const toCol = findTableauDropColumn(card, state.tableau);
      if (toCol >= 0) {
        onMove(moveWasteToTableau(state, toCol));
        onSelect(null);
        return;
      }
      const fIdx = findFoundationDropIndex(card, state.foundations);
      if (fIdx >= 0) {
        onMove(moveWasteToFoundation(state));
        onSelect(null);
        return;
      }
    }
    onSelect({ source: 'waste', colIndex: 0, cardIndex: 0 });
  };

  const handleFoundationClick = (foundIndex) => {
    if (selection && state.foundations[foundIndex].length > 0) {
      const card = state.foundations[foundIndex][state.foundations[foundIndex].length - 1];
      const toCol = findTableauDropColumn(card, state.tableau);
      if (toCol >= 0) {
        onMove(moveFoundationToTableau(state, foundIndex, toCol));
        onSelect(null);
      }
      return;
    }
    if (!selection) return;
    const card = getSelectedCard();
    if (!card) return;
    const fIdx = findFoundationDropIndex(card, state.foundations);
    if (fIdx !== foundIndex) return;
    if (selection.source === 'waste') onMove(moveWasteToFoundation(state));
    else onMove(moveTableauToFoundation(state, selection.colIndex));
    onSelect(null);
  };

  const handleTableauClick = (colIndex, cardIndex) => {
    const col = state.tableau[colIndex];
    if (cardIndex === col.length) {
      if (selection) {
        const card = getSelectedCard();
        if (card && card.rank === 'K') {
          if (selection.source === 'waste') onMove(moveWasteToTableau(state, colIndex));
          else onMove(moveTableauToTableau(state, selection.colIndex, selection.cardIndex, colIndex));
          onSelect(null);
        }
      }
      return;
    }
    const card = col[cardIndex];
    if (!card.faceUp) {
      if (cardIndex === col.length - 1 && onFlip) onFlip(colIndex);
      return;
    }
    if (!canMoveFromTableau(state.tableau, colIndex, cardIndex)) return;
    if (selection && (selection.source !== 'tableau' || selection.colIndex !== colIndex || selection.cardIndex !== cardIndex)) {
      const selCard = getSelectedCard();
      if (selCard && selection.source === 'tableau') {
        const next = moveTableauToTableau(state, selection.colIndex, selection.cardIndex, colIndex);
        if (next !== state) {
          onMove(next);
          onSelect(null);
        }
        return;
      }
      if (selCard && selection.source === 'waste') {
        const toCol = findTableauDropColumn(selCard, state.tableau);
        if (toCol === colIndex) {
          onMove(moveWasteToTableau(state, colIndex));
          onSelect(null);
        }
        return;
      }
    }
    onSelect({ source: 'tableau', colIndex, cardIndex });
  };

  return (
    <Box
      sx={{
        p: 1,
        userSelect: 'none',
        WebkitUserSelect: 'none',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-start',
        minHeight: 0,
      }}
    >
      <Box sx={{ display: 'flex', gap: 1, mb: 1.5, flexWrap: 'wrap' }}>
        <Box onClick={handleStockClick} sx={{ cursor: 'pointer' }}>
          {state.stock.length > 0 ? (
            <SolitaireCard card={{ suit: 's', rank: '?', faceUp: false }} onClick={() => {}} width={CARD_WIDTH} height={CARD_HEIGHT} />
          ) : (
            <Box sx={{ width: CARD_WIDTH, height: CARD_HEIGHT, borderRadius: 1, border: '1px dashed', borderColor: 'divider' }} />
          )}
        </Box>
        <Box onClick={handleWasteClick} sx={{ cursor: 'pointer' }}>
          {state.waste.length > 0 ? (
            <SolitaireCard
              card={state.waste[state.waste.length - 1]}
              selected={selection?.source === 'waste'}
              onClick={() => {}}
              width={CARD_WIDTH}
              height={CARD_HEIGHT}
            />
          ) : (
            <Box sx={{ width: CARD_WIDTH, height: CARD_HEIGHT, borderRadius: 1, border: '1px dashed', borderColor: 'divider' }} />
          )}
        </Box>
        <Box sx={{ width: 32 }} />
        {[0, 1, 2, 3].map((f) => (
          <Box key={f} onClick={() => handleFoundationClick(f)} sx={{ cursor: 'pointer' }}>
            {state.foundations[f].length > 0 ? (
              <SolitaireCard card={state.foundations[f][state.foundations[f].length - 1]} onClick={() => {}} width={CARD_WIDTH} height={CARD_HEIGHT} />
            ) : (
              <Box sx={{ width: CARD_WIDTH, height: CARD_HEIGHT, borderRadius: 1, border: '1px dashed', borderColor: 'divider' }} />
            )}
          </Box>
        ))}
      </Box>
      <Box sx={{ display: 'flex', gap: 0.5 }}>
        {state.tableau.map((col, colIndex) => {
          const lastCardBottom = col.length === 0 ? 0 : (col.length - 1) * STACK_OFFSET + CARD_HEIGHT;
          return (
            <Box
              key={colIndex}
              sx={{
                minWidth: CARD_WIDTH,
                position: 'relative',
                minHeight: col.length === 0 ? CARD_HEIGHT : lastCardBottom + CARD_HEIGHT,
                isolation: 'isolate',
              }}
            >
              {col.length === 0 ? (
                <Box
                  onClick={() => handleTableauClick(colIndex, 0)}
                  sx={{ height: CARD_HEIGHT, borderRadius: 1, border: '1px dashed', borderColor: 'divider', cursor: 'pointer' }}
                />
              ) : (
                col.map((card, cardIndex) => {
                  const cardHeight = cardIndex < col.length - 1 ? CARD_HEIGHT - 4 : CARD_HEIGHT;
                  return (
                    <Box
                      key={`${colIndex}-${cardIndex}`}
                      onClick={() => handleTableauClick(colIndex, cardIndex)}
                      sx={{
                        position: 'absolute',
                        left: 0,
                        top: cardIndex * STACK_OFFSET,
                        zIndex: cardIndex,
                        width: CARD_WIDTH,
                        height: cardHeight,
                        transform: 'translateZ(0)',
                        '&:hover > *': {
                          bgcolor: 'action.hover',
                        },
                      }}
                    >
                      <SolitaireCard
                        card={card}
                        selected={selection?.source === 'tableau' && selection.colIndex === colIndex && selection.cardIndex === cardIndex}
                        stacked={cardIndex < col.length - 1}
                        width={CARD_WIDTH}
                        height={cardHeight}
                        disableHover
                      />
                    </Box>
                  );
                })
              )}
              {col.length > 0 && (
                <Box
                  onClick={() => handleTableauClick(colIndex, col.length)}
                  sx={{
                    position: 'absolute',
                    left: 0,
                    top: lastCardBottom,
                    width: CARD_WIDTH,
                    height: CARD_HEIGHT,
                    zIndex: col.length,
                  }}
                />
              )}
            </Box>
          );
        })}
      </Box>
    </Box>
  );
};

export default SolitaireBoard;
