import React from 'react';
import { Box, Typography } from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

const LemonadeChart = ({ history }) => {
  if (!history || history.length === 0) {
    return (
      <Box sx={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography color="text.secondary">No data yet. Open your stand to see results.</Typography>
      </Box>
    );
  }

  const data = history.map((h) => ({
    day: h.day,
    profit: Math.round(h.profit * 100) / 100,
    revenue: Math.round(h.revenue * 100) / 100,
    customers: h.customers,
  }));

  return (
    <Box sx={{ width: '100%', height: 260 }}>
      <Typography variant="subtitle2" gutterBottom>
        Daily results
      </Typography>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="day" type="number" domain={[1, 30]} />
          <YAxis />
          <Tooltip formatter={(v) => (typeof v === 'number' ? (v < 10 ? v.toFixed(2) : v) : v)} />
          <Legend />
          <Line type="monotone" dataKey="profit" stroke="#2e7d32" name="Profit ($)" dot />
          <Line type="monotone" dataKey="revenue" stroke="#1976d2" name="Revenue ($)" dot />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default LemonadeChart;
