import React from 'react';
import { useParams, useNavigate, Navigate } from 'react-router-dom';
import { Box, Card, CardContent, CardActionArea, Typography, Grid, Chip } from '@mui/material';
import { useAuth } from '../../contexts/AuthContext';
import { useCapabilities } from '../../contexts/CapabilitiesContext';
import SolitaireGame from './solitaire/SolitaireGame';
import LemonadeGame from './lemonade/LemonadeGame';

const GAMES = [
  {
    id: 'solitaire',
    title: 'Solitaire',
    description: 'Classic Klondike. Build foundations and clear the tableau. Click a card to select, then click a destination to move.',
    storageKey: 'bastion_solitaire',
  },
  {
    id: 'lemonade',
    title: 'Lemonade Stand',
    description: 'Run your stand for 30 days. Set price, buy ingredients, buy upgrades, and see how the weather affects your sales.',
    storageKey: 'bastion_lemonade',
  },
];

const GamesPage = () => {
  const { gameId } = useParams();
  const navigate = useNavigate();
  const { user } = useAuth();
  const { isAdmin, has } = useCapabilities();
  const userId = user?.user_id ?? 'anonymous';

  if (!isAdmin && !has('feature.games.view')) {
    return <Navigate to="/documents" replace />;
  }

  const hasSavedGame = (storageKey) => {
    try {
      const raw = localStorage.getItem(`${storageKey}_${userId}`);
      if (!raw) return false;
      const data = JSON.parse(raw);
      if (data.won) return false;
      if (data.phase === 'gameover') return false;
      return true;
    } catch {
      return false;
    }
  };

  if (gameId === 'solitaire') {
    return (
      <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <SolitaireGame userId={userId} onBack={() => navigate('/games')} />
      </Box>
    );
  }
  if (gameId === 'lemonade') {
    return (
      <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <LemonadeGame userId={userId} onBack={() => navigate('/games')} />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h4" sx={{ mb: 2 }}>
        Games
      </Typography>
      <Grid container spacing={2}>
        {GAMES.map((game) => (
          <Grid item xs={12} sm={6} md={4} key={game.id}>
            <Card variant="outlined" sx={{ height: '100%' }}>
              <CardActionArea
                onClick={() => navigate(`/games/${game.id}`)}
                sx={{ height: '100%', display: 'flex', alignItems: 'stretch' }}
              >
                <CardContent sx={{ width: '100%' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Typography variant="h6">{game.title}</Typography>
                    {hasSavedGame(game.storageKey) && (
                      <Chip label="Resume" color="primary" size="small" />
                    )}
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {game.description}
                  </Typography>
                </CardContent>
              </CardActionArea>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default GamesPage;
