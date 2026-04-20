import React, { useState, useEffect, useCallback } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  IconButton,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Avatar,
  Chip,
  Autocomplete,
  TextField,
  Alert,
  CircularProgress,
  Box,
  Divider,
  Tooltip,
} from '@mui/material';
import {
  Close as CloseIcon,
  Share as ShareIcon,
  Delete as DeleteIcon,
  PersonAdd as PersonAddIcon,
} from '@mui/icons-material';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';
import agentFactoryService from '../../services/agentFactoryService';

const ARTIFACT_TYPE_LABELS = {
  agent_profile: 'Agent',
  playbook: 'Playbook',
  skill: 'Skill',
};

export default function ArtifactShareDialog({ open, onClose, artifactType, artifactId, artifactName }) {
  const queryClient = useQueryClient();
  const [selectedUser, setSelectedUser] = useState(null);
  const [inputValue, setInputValue] = useState('');
  const [error, setError] = useState('');

  const { data: shareableUsers = [], isLoading: usersLoading } = useQuery(
    ['shareableUsers'],
    () => apiService.getShareableUsers(),
    {
      enabled: open,
      staleTime: 30000,
      select: (data) => data?.users || data || [],
    }
  );

  const { data: existingShares = [], isLoading: sharesLoading, refetch: refetchShares } = useQuery(
    ['artifactShares', artifactType, artifactId],
    () => agentFactoryService.listArtifactShares(artifactType, artifactId),
    { enabled: open && !!artifactId }
  );

  const shareMutation = useMutation(
    (body) => agentFactoryService.shareArtifact(body),
    {
      onSuccess: () => {
        setSelectedUser(null);
        setInputValue('');
        setError('');
        refetchShares();
        queryClient.invalidateQueries('agentFactorySharedWithMe');
      },
      onError: (err) => {
        setError(err?.response?.data?.detail || err?.message || 'Failed to share');
      },
    }
  );

  const revokeMutation = useMutation(
    (shareId) => agentFactoryService.revokeShare(shareId),
    {
      onSuccess: () => {
        refetchShares();
        queryClient.invalidateQueries('agentFactorySharedWithMe');
      },
    }
  );

  const handleShare = useCallback(() => {
    if (!selectedUser) return;
    setError('');
    shareMutation.mutate({
      artifact_type: artifactType,
      artifact_id: artifactId,
      shared_with_user_id: selectedUser.user_id,
    });
  }, [selectedUser, artifactType, artifactId, shareMutation]);

  useEffect(() => {
    if (!open) {
      setSelectedUser(null);
      setInputValue('');
      setError('');
    }
  }, [open]);

  const sharedUserIds = new Set((existingShares || []).map((s) => s.shared_with_user_id));
  const availableUsers = (shareableUsers || []).filter((u) => !sharedUserIds.has(u.user_id));
  const typeLabel = ARTIFACT_TYPE_LABELS[artifactType] || 'Item';

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1, pr: 6 }}>
        <ShareIcon fontSize="small" />
        <Typography variant="h6" component="span" noWrap>
          Share {typeLabel}: {artifactName}
        </Typography>
        <IconButton onClick={onClose} sx={{ position: 'absolute', right: 8, top: 8 }}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        <Box sx={{ display: 'flex', gap: 1, mb: 2, alignItems: 'flex-start' }}>
          <Autocomplete
            sx={{ flex: 1 }}
            options={availableUsers}
            loading={usersLoading}
            value={selectedUser}
            onChange={(_, val) => setSelectedUser(val)}
            inputValue={inputValue}
            onInputChange={(_, val) => setInputValue(val)}
            getOptionLabel={(opt) =>
              opt?.display_name
                ? `${opt.display_name} (${opt.username})`
                : opt?.username || ''
            }
            isOptionEqualToValue={(opt, val) => opt?.user_id === val?.user_id}
            renderInput={(params) => (
              <TextField
                {...params}
                label="Search users"
                placeholder="Type a name or username..."
                size="small"
                InputProps={{
                  ...params.InputProps,
                  startAdornment: (
                    <>
                      <PersonAddIcon fontSize="small" sx={{ mr: 0.5, color: 'text.secondary' }} />
                      {params.InputProps.startAdornment}
                    </>
                  ),
                }}
              />
            )}
            renderOption={(props, opt) => (
              <li {...props} key={opt.user_id}>
                <Avatar sx={{ width: 28, height: 28, mr: 1, fontSize: '0.8rem' }}>
                  {(opt.display_name || opt.username || '?')[0].toUpperCase()}
                </Avatar>
                <Box>
                  <Typography variant="body2">{opt.display_name || opt.username}</Typography>
                  {opt.display_name && (
                    <Typography variant="caption" color="text.secondary">{opt.username}</Typography>
                  )}
                </Box>
              </li>
            )}
            noOptionsText="No users available"
          />
          <Button
            variant="contained"
            size="small"
            disabled={!selectedUser || shareMutation.isLoading}
            onClick={handleShare}
            sx={{ minWidth: 80, height: 40 }}
          >
            {shareMutation.isLoading ? <CircularProgress size={20} /> : 'Share'}
          </Button>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
            {error}
          </Alert>
        )}

        <Divider sx={{ mb: 1 }} />
        <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
          Shared with {existingShares?.length || 0} user{(existingShares?.length || 0) !== 1 ? 's' : ''}
        </Typography>

        {sharesLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
            <CircularProgress size={24} />
          </Box>
        ) : (existingShares || []).length === 0 ? (
          <Typography variant="body2" color="text.secondary" sx={{ py: 2, textAlign: 'center' }}>
            Not shared with anyone yet.
          </Typography>
        ) : (
          <List dense disablePadding>
            {(existingShares || []).map((share) => (
              <ListItem
                key={share.id}
                secondaryAction={
                  <Tooltip title="Revoke access">
                    <IconButton
                      edge="end"
                      size="small"
                      onClick={() => revokeMutation.mutate(share.id)}
                      disabled={revokeMutation.isLoading}
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                }
              >
                <ListItemAvatar>
                  <Avatar sx={{ width: 32, height: 32, fontSize: '0.85rem' }}>
                    {(share.shared_with_display_name || share.shared_with_username || '?')[0].toUpperCase()}
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={share.shared_with_display_name || share.shared_with_username}
                  secondary={share.shared_with_username}
                />
                <Chip label="use" size="small" variant="outlined" sx={{ mr: 4 }} />
              </ListItem>
            ))}
          </List>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}
