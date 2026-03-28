import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Typography,
  Box,
  CircularProgress,
  Alert
} from '@mui/material';
import { Code as CodeIcon } from '@mui/icons-material';
import dataWorkspaceService from '../../services/dataWorkspaceService';

const RunSqlDialog = ({ open, onClose, workspaceId, onSuccess }) => {
  const [sql, setSql] = useState('');
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleRun = async () => {
    if (!sql.trim()) return;
    setError(null);
    setResult(null);
    setRunning(true);
    try {
      const data = await dataWorkspaceService.executeSql(workspaceId, {
        query: sql.trim(),
        limit: 1000
      });
      setResult(data);
      if (onSuccess) onSuccess(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Query failed');
    } finally {
      setRunning(false);
    }
  };

  const handleClose = () => {
    setSql('');
    setResult(null);
    setError(null);
    onClose();
  };

  const rowsAffected = result?.rows_affected ?? 0;
  const hasReturning = result?.returning_rows?.length > 0;
  const hasResults = result?.results?.length > 0;

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <CodeIcon /> Run SQL
      </DialogTitle>
      <DialogContent>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          Run SQL in this workspace. Use CREATE TABLE to add tables, INSERT/UPDATE/DELETE to modify data. Tables are created in the workspace schema.
        </Typography>
        <TextField
          autoFocus
          multiline
          minRows={6}
          maxRows={16}
          fullWidth
          placeholder={'CREATE TABLE portfolio (\n  cash_balance NUMERIC(15,2) DEFAULT 1000000.00,\n  total_invested NUMERIC(15,2) DEFAULT 0.00,\n  ...\n);'}
          value={sql}
          onChange={(e) => setSql(e.target.value)}
          variant="outlined"
          margin="dense"
          sx={{ fontFamily: 'monospace', fontSize: '0.9rem' }}
          disabled={running}
        />
        {error && (
          <Alert severity="error" sx={{ mt: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
        {result && !error && (
          <Box sx={{ mt: 2 }}>
            {result.error_message ? (
              <Alert severity="warning">{result.error_message}</Alert>
            ) : (
              <>
                {rowsAffected > 0 && (
                  <Typography variant="body2" color="primary.main" sx={{ mb: 1 }}>
                    Rows affected: {rowsAffected}
                  </Typography>
                )}
                {hasReturning && (
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    Returned {result.returning_rows.length} row(s).
                  </Typography>
                )}
                {hasResults && (
                  <Typography variant="body2" color="text.secondary">
                    Returned {result.results.length} row(s). Execution time: {result.execution_time_ms} ms
                  </Typography>
                )}
                {!rowsAffected && !hasReturning && !hasResults && !result.error_message && (
                  <Typography variant="body2" color="text.secondary">
                    Command completed. Execution time: {result.execution_time_ms} ms
                  </Typography>
                )}
              </>
            )}
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose}>Close</Button>
        <Button
          variant="contained"
          onClick={handleRun}
          disabled={running || !sql.trim()}
          startIcon={running ? <CircularProgress size={18} /> : <CodeIcon />}
        >
          {running ? 'Running…' : 'Run'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default RunSqlDialog;
