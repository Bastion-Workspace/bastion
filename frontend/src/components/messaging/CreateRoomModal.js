/**
 * Create Room Modal — new messaging rooms with local or federated users.
 */

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Autocomplete,
  Chip,
  CircularProgress,
  Alert,
  Box,
  Typography,
} from '@mui/material';
import { Hub } from '@mui/icons-material';
import apiService from '../../services/apiService';
import { useMessaging } from '../../contexts/MessagingContext';

const FEDERATED_ADDRESS_RE = /^[^\s@]+@[^\s@.]+\.[^\s@]+$/;

const CreateRoomModal = ({ open, onClose }) => {
  const { createRoom, loadRooms, selectRoom } = useMessaging();
  const [loading, setLoading] = useState(false);
  const [users, setUsers] = useState([]);
  const [loadingUsers, setLoadingUsers] = useState(false);
  const [selectedUsers, setSelectedUsers] = useState([]);
  const [roomName, setRoomName] = useState('');
  const [error, setError] = useState(null);
  const [remoteSuggestion, setRemoteSuggestion] = useState(null);
  const [remoteLoading, setRemoteLoading] = useState(false);
  const resolveTimerRef = useRef(null);

  useEffect(() => {
    if (open) {
      loadUsers();
    } else {
      setSelectedUsers([]);
      setRoomName('');
      setError(null);
      setRemoteSuggestion(null);
      setRemoteLoading(false);
      if (resolveTimerRef.current) clearTimeout(resolveTimerRef.current);
    }
  }, [open]);

  const loadUsers = async () => {
    setLoadingUsers(true);
    setError(null);

    try {
      const response = await apiService.get('/api/messaging/users');
      setUsers(response.users || []);
    } catch (err) {
      console.error('❌ Failed to load users:', err);
      setError('Failed to load users. Please try again.');
    } finally {
      setLoadingUsers(false);
    }
  };

  const scheduleResolveRemote = useCallback((inputValue) => {
    if (resolveTimerRef.current) clearTimeout(resolveTimerRef.current);
    const t = (inputValue || '').trim();
    if (!FEDERATED_ADDRESS_RE.test(t)) {
      setRemoteSuggestion(null);
      setRemoteLoading(false);
      return;
    }
    resolveTimerRef.current = setTimeout(async () => {
      setRemoteLoading(true);
      try {
        const r = await apiService.federation.resolveRemoteUser(t);
        if (r?.found) {
          setRemoteSuggestion({
            kind: 'federated_remote',
            address: t,
            display_name: r.display_name,
            username: r.username,
            avatar_url: r.avatar_url,
          });
        } else {
          setRemoteSuggestion(null);
        }
      } catch {
        setRemoteSuggestion(null);
      } finally {
        setRemoteLoading(false);
      }
    }, 450);
  }, []);

  const mergedOptions = useMemo(() => {
    const base = [...users];
    if (
      remoteSuggestion &&
      !base.some((u) => u.user_id && remoteSuggestion.username && u.username === remoteSuggestion.username)
    ) {
      return [remoteSuggestion, ...base];
    }
    return base;
  }, [users, remoteSuggestion]);

  const handleCreateRoom = async () => {
    if (selectedUsers.length === 0) {
      setError('Please select at least one user to chat with');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const remote = selectedUsers.find((u) => u.kind === 'federated_remote');
      if (remote) {
        const res = await apiService.federation.createUserDmRoom({
          remote_user_address: remote.address,
        });
        await loadRooms();
        const rid = res?.room?.room_id;
        if (rid) selectRoom(rid);
        onClose();
        return;
      }

      const participantIds = selectedUsers.map((user) => user.user_id);
      const finalRoomName =
        selectedUsers.length === 1 && !roomName ? null : roomName || null;

      await createRoom(participantIds, finalRoomName);
      onClose();
    } catch (err) {
      console.error('❌ Failed to create room:', err);
      setError(
        err.response?.data?.detail || err.message || 'Failed to create room. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleCreateRoom();
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Start New Conversation</DialogTitle>

      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
          {error && (
            <Alert severity="error" onClose={() => setError(null)}>
              {error}
            </Alert>
          )}

          <Typography variant="caption" color="text.secondary">
            For a federated DM, type <strong>username@remote-host</strong> (host must match an active
            federation peer). Otherwise search local users.
          </Typography>

          <Autocomplete
            multiple
            options={mergedOptions}
            loading={loadingUsers || remoteLoading}
            getOptionLabel={(option) => {
              if (option.kind === 'federated_remote') {
                const dn = option.display_name || option.username || option.address;
                return `${dn} (${option.address})`;
              }
              return option.display_name || option.username || 'Unknown';
            }}
            isOptionEqualToValue={(a, b) => {
              if (a.kind === 'federated_remote' && b.kind === 'federated_remote') {
                return a.address === b.address;
              }
              return a.user_id && b.user_id && a.user_id === b.user_id;
            }}
            filterOptions={(opts, state) => {
              const q = (state.inputValue || '').toLowerCase().trim();
              if (!q) return opts;
              return opts.filter((o) => {
                if (o.kind === 'federated_remote') {
                  return (
                    (o.address || '').toLowerCase().includes(q) ||
                    (o.username || '').toLowerCase().includes(q) ||
                    (o.display_name || '').toLowerCase().includes(q)
                  );
                }
                return (
                  (o.username || '').toLowerCase().includes(q) ||
                  (o.display_name || '').toLowerCase().includes(q)
                );
              });
            }}
            value={selectedUsers}
            onChange={(event, newValue) => {
              const hasFed = newValue.some((u) => u.kind === 'federated_remote');
              if (hasFed) {
                const fed = newValue.find((u) => u.kind === 'federated_remote');
                setSelectedUsers(fed ? [fed] : []);
                return;
              }
              setSelectedUsers(newValue);
            }}
            onInputChange={(event, value, reason) => {
              if (reason === 'input') scheduleResolveRemote(value);
            }}
            renderOption={(props, option) => {
              if (option.kind === 'federated_remote') {
                return (
                  <li {...props} key={`fed-${option.address}`}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                      <Hub fontSize="small" color="action" />
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="body2">
                          {option.display_name || option.username}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {option.address}
                        </Typography>
                      </Box>
                      <Chip size="small" label="Remote" variant="outlined" />
                    </Box>
                  </li>
                );
              }
              return (
                <li {...props} key={option.user_id}>
                  {option.display_name || option.username}
                </li>
              );
            }}
            renderInput={(params) => (
              <TextField
                {...params}
                label="Select Users"
                placeholder="Search or type user@remote-host…"
                InputProps={{
                  ...params.InputProps,
                  endAdornment: (
                    <>
                      {loadingUsers || remoteLoading ? <CircularProgress size={20} /> : null}
                      {params.InputProps.endAdornment}
                    </>
                  ),
                }}
              />
            )}
            renderTags={(value, getTagProps) =>
              value.map((option, index) => (
                <Chip
                  label={
                    option.kind === 'federated_remote'
                      ? option.display_name || option.username || option.address
                      : option.display_name || option.username
                  }
                  {...getTagProps({ index })}
                  size="small"
                  icon={option.kind === 'federated_remote' ? <Hub /> : undefined}
                />
              ))
            }
            disabled={loadingUsers || loading}
          />

          {selectedUsers.length > 1 && !selectedUsers.some((u) => u.kind === 'federated_remote') && (
            <TextField
              label="Room Name (optional)"
              value={roomName}
              onChange={(e) => setRoomName(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Give your group chat a name"
              disabled={loading}
              helperText="For group chats, you can set a custom name"
            />
          )}

          {selectedUsers.length === 1 &&
            roomName &&
            !selectedUsers.some((u) => u.kind === 'federated_remote') && (
              <Typography variant="caption" color="text.secondary">
                💡 Custom names for 1:1 chats will be visible to both participants
              </Typography>
            )}
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <Button
          onClick={handleCreateRoom}
          variant="contained"
          disabled={loading || selectedUsers.length === 0}
          startIcon={loading && <CircularProgress size={16} />}
        >
          {loading ? 'Creating...' : 'Start Chat'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default CreateRoomModal;
