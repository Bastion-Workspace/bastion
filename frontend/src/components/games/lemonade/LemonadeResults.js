import React from 'react';
import { Box, Typography, Button, Card, CardContent } from '@mui/material';
import { WEATHER_LABELS } from './lemonadeEngine';

const LemonadeResults = ({ lastResult, onNext }) => {
  if (!lastResult) return null;
  const { day, weather, customers, revenue, cost, profit, eventMessage } = lastResult;

  return (
    <Box sx={{ p: 2, maxWidth: 400, mx: 'auto' }}>
      <Typography variant="h6" gutterBottom>
        Day {day} Results
      </Typography>
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography color="text.secondary">Weather: {WEATHER_LABELS[weather] || weather}</Typography>
          <Typography>Customers: {customers}</Typography>
          <Typography>Revenue: ${revenue.toFixed(2)}</Typography>
          <Typography>Cost of goods: ${cost.toFixed(2)}</Typography>
          <Typography fontWeight="bold" color={profit >= 0 ? 'success.main' : 'error.main'}>
            Profit: ${profit.toFixed(2)}
          </Typography>
          {eventMessage && (
            <Typography variant="body2" color="primary" sx={{ mt: 1, fontStyle: 'italic' }}>
              {eventMessage}
            </Typography>
          )}
        </CardContent>
      </Card>
      <Button variant="contained" onClick={onNext} fullWidth>
        Next Day
      </Button>
    </Box>
  );
};

export default LemonadeResults;
