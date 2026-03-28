import React, { useState } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  Slider,
  FormControlLabel,
  RadioGroup,
  Radio,
  Divider,
} from '@mui/material';
import { WEATHER_LABELS, UPGRADES, LEMON_COST, CUP_COST, ICE_COST, TOTAL_DAYS } from './lemonadeEngine';

const LemonadeDay = ({ state, forecast, onOpenStand }) => {
  const [pricePerCup, setPricePerCup] = useState(0.25);
  const [buyLemons, setBuyLemons] = useState(0);
  const [buyCups, setBuyCups] = useState(0);
  const [buyIce, setBuyIce] = useState(0);
  const [buyUpgradeId, setBuyUpgradeId] = useState('');

  const costLemons = buyLemons * LEMON_COST;
  const costCups = buyCups * CUP_COST;
  const costIce = buyIce * ICE_COST;
  const upgradeCost = buyUpgradeId ? (UPGRADES.find((u) => u.id === buyUpgradeId)?.cost ?? 0) : 0;
  const totalSpend = costLemons + costCups + costIce + upgradeCost;
  const canAfford = state.money >= totalSpend;
  const ownedUpgrades = state.upgrades || [];

  const handleSubmit = () => {
    if (!canAfford) return;
    onOpenStand({
      pricePerCup,
      buyLemons,
      buyCups,
      buyIce,
      buyUpgradeId: buyUpgradeId || undefined,
    });
  };

  return (
    <Box sx={{ p: 2, maxWidth: 480, mx: 'auto' }}>
      <Typography variant="h6" gutterBottom>
        Day {state.day} of {TOTAL_DAYS}
      </Typography>
      <Typography color="text.secondary" gutterBottom>
        Cash: ${state.money.toFixed(2)} · Lemons: {state.lemons} · Cups: {state.cups} · Ice: {state.ice}
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Forecast: {WEATHER_LABELS[forecast] || forecast}
      </Typography>
      <Divider sx={{ my: 2 }} />

      <Typography gutterBottom>Price per cup ($)</Typography>
      <Slider
        value={pricePerCup}
        onChange={(_, v) => setPricePerCup(v)}
        min={0.01}
        max={1.5}
        step={0.01}
        valueLabelDisplay="auto"
        valueLabelFormat={(v) => `$${v.toFixed(2)}`}
        sx={{ mb: 2 }}
      />

      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2 }}>
        <TextField
          label="Buy lemons"
          type="number"
          inputProps={{ min: 0, step: 1 }}
          value={buyLemons}
          onChange={(e) => setBuyLemons(Math.max(0, parseInt(e.target.value, 10) || 0))}
          size="small"
          sx={{ width: 100 }}
          helperText={`$${(buyLemons * LEMON_COST).toFixed(2)}`}
        />
        <TextField
          label="Buy cups"
          type="number"
          inputProps={{ min: 0, step: 1 }}
          value={buyCups}
          onChange={(e) => setBuyCups(Math.max(0, parseInt(e.target.value, 10) || 0))}
          size="small"
          sx={{ width: 100 }}
          helperText={`$${(buyCups * CUP_COST).toFixed(2)}`}
        />
        <TextField
          label="Buy ice"
          type="number"
          inputProps={{ min: 0, step: 1 }}
          value={buyIce}
          onChange={(e) => setBuyIce(Math.max(0, parseInt(e.target.value, 10) || 0))}
          size="small"
          sx={{ width: 100 }}
          helperText={`$${(buyIce * ICE_COST).toFixed(2)}`}
        />
      </Box>

      {UPGRADES.filter((u) => !ownedUpgrades.includes(u.id)).length > 0 && (
        <>
          <Typography gutterBottom>Buy upgrade (once)</Typography>
          <RadioGroup row value={buyUpgradeId} onChange={(e) => setBuyUpgradeId(e.target.value)}>
            <FormControlLabel value="" control={<Radio />} label="None" />
            {UPGRADES.filter((u) => !ownedUpgrades.includes(u.id)).map((u) => (
              <FormControlLabel
                key={u.id}
                value={u.id}
                control={<Radio />}
                label={`${u.name} ($${u.cost})`}
              />
            ))}
          </RadioGroup>
        </>
      )}

      <Box sx={{ mt: 2 }}>
        <Typography color="text.secondary">Total spend: ${totalSpend.toFixed(2)}</Typography>
        {!canAfford && totalSpend > 0 && (
          <Typography color="error">Not enough cash.</Typography>
        )}
      </Box>

      <Button
        variant="contained"
        onClick={handleSubmit}
        disabled={!canAfford}
        fullWidth
        sx={{ mt: 2 }}
      >
        Open Stand
      </Button>
    </Box>
  );
};

export default LemonadeDay;
