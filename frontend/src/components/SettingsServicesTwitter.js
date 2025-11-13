import React from 'react';
import { Box, Card, CardContent, Typography, TextField, Grid, Switch, FormControlLabel, Button, Alert, CircularProgress } from '@mui/material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../services/apiService';

const SettingsServicesTwitter = () => {
  const queryClient = useQueryClient();
  const { data, isLoading } = useQuery('twitterConfig', () => apiService.get('/api/services/twitter/config'));

  const [enabled, setEnabled] = React.useState(false);
  const [bearer, setBearer] = React.useState('');
  const [interval, setInterval] = React.useState(5);
  const [xUserId, setXUserId] = React.useState('');
  const [backfill, setBackfill] = React.useState(60);
  const [includeReplies, setIncludeReplies] = React.useState(true);
  const [includeRetweets, setIncludeRetweets] = React.useState(true);
  const [includeQuotes, setIncludeQuotes] = React.useState(true);
  const [snackbar, setSnackbar] = React.useState({ open: false, message: '', severity: 'success' });

  React.useEffect(() => {
    if (data) {
      setEnabled(!!data.enabled);
      setBearer(data.bearer_token || '');
      setInterval(data.poll_interval_minutes ?? 5);
      setXUserId(data.user_id || '');
      setBackfill(data.backfill_days ?? 60);
      setIncludeReplies(!!data.include_replies);
      setIncludeRetweets(!!data.include_retweets);
      setIncludeQuotes(!!data.include_quotes);
    }
  }, [data]);

  const saveMutation = useMutation((payload) => apiService.post('/api/services/twitter/config', payload), {
    onSuccess: () => {
      queryClient.invalidateQueries('twitterConfig');
      setSnackbar({ open: true, message: 'Twitter settings saved', severity: 'success' });
    },
    onError: (e) => setSnackbar({ open: true, message: e?.response?.data?.detail || 'Failed to save', severity: 'error' })
  });

  const toggleMutation = useMutation((enabled) => apiService.post('/api/services/twitter/toggle', { enabled }), {
    onSuccess: () => queryClient.invalidateQueries('twitterConfig')
  });

  const testMutation = useMutation(() => apiService.post('/api/services/twitter/test', {}), {
    onSuccess: (res) => setSnackbar({ open: true, message: res?.message || 'OK', severity: res?.success ? 'success' : 'error' }),
    onError: (e) => setSnackbar({ open: true, message: e?.response?.data?.detail || 'Failed', severity: 'error' })
  });

  if (isLoading) {
    return (
      <Card><CardContent><Box display="flex" alignItems="center" gap={2}><CircularProgress size={20} /><Typography>Loading Twitter settings…</Typography></Box></CardContent></Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Twitter Ingestion</Typography>
        <Alert severity="info" sx={{ mb: 2 }}>Enable background ingestion of your Following feed, including replies, retweets, and quotes. Initial backfill ~60 days.</Alert>

        <FormControlLabel
          control={<Switch checked={enabled} onChange={(e) => { setEnabled(e.target.checked); toggleMutation.mutate(e.target.checked); }} />}
          label="Enable Twitter Ingestion"
        />

        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={12} md={8}>
            <TextField
              fullWidth
              type="password"
              label="Bearer Token"
              value={bearer}
              onChange={(e) => setBearer(e.target.value)}
              helperText="Twitter API v2 Bearer token"
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Your X User ID"
              value={xUserId}
              onChange={(e) => setXUserId(e.target.value)}
              helperText="Numeric user ID for your account (for Following/home)"
            />
          </Grid>
          <Grid item xs={12} md={2}>
            <TextField fullWidth type="number" label="Interval (min)" value={interval} onChange={(e) => setInterval(parseInt(e.target.value || '5', 10))} />
          </Grid>
          <Grid item xs={12} md={2}>
            <TextField fullWidth type="number" label="Backfill (days)" value={backfill} onChange={(e) => setBackfill(parseInt(e.target.value || '60', 10))} />
          </Grid>
          <Grid item xs={12} md={4}>
            <FormControlLabel control={<Switch checked={includeReplies} onChange={(e) => setIncludeReplies(e.target.checked)} />} label="Include replies" />
          </Grid>
          <Grid item xs={12} md={4}>
            <FormControlLabel control={<Switch checked={includeRetweets} onChange={(e) => setIncludeRetweets(e.target.checked)} />} label="Include retweets" />
          </Grid>
          <Grid item xs={12} md={4}>
            <FormControlLabel control={<Switch checked={includeQuotes} onChange={(e) => setIncludeQuotes(e.target.checked)} />} label="Include quotes" />
          </Grid>
        </Grid>

        <Box mt={2} display="flex" gap={2}>
          <Button variant="contained" onClick={() => saveMutation.mutate({ enabled, bearer_token: bearer, user_id: xUserId, poll_interval_minutes: interval, backfill_days: backfill, include_replies: includeReplies, include_retweets: includeRetweets, include_quotes: includeQuotes })} disabled={saveMutation.isLoading}>
            {saveMutation.isLoading ? 'Saving…' : 'Save Settings'}
          </Button>
          <Button variant="outlined" onClick={() => testMutation.mutate()} disabled={testMutation.isLoading}>
            {testMutation.isLoading ? 'Testing…' : 'Test Connection'}
          </Button>
        </Box>

        {snackbar.open && (
          <Box mt={2}>
            <Alert severity={snackbar.severity} onClose={() => setSnackbar({ ...snackbar, open: false })}>{snackbar.message}</Alert>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default SettingsServicesTwitter;


