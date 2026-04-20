import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Autocomplete,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  FormLabel,
  Radio,
  RadioGroup,
  FormControlLabel,
  TextField,
  Typography,
  Divider,
  Stack,
  IconButton,
} from '@mui/material';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import apiService from '../services/apiService';

/**
 * Manage shares for a document or folder (owner only).
 * @param {object} props
 * @param {boolean} props.open
 * @param {function} props.onClose
 * @param {'document'|'folder'} props.targetType
 * @param {string} [props.documentId]
 * @param
 * {string} [props.folderId]
 * @param {function} [props.onSaved]
 */
export default function DocumentSharingDialog({
  open,
  onClose,
  targetType,
  documentId,
  folderId,
  onSaved,
}) {
  const [loading, setLoading] = useState(false);
  const [users, setUsers] = useState([]);
  const [shares, setShares] = useState([]);
  const [pickedUser, setPickedUser] = useState(null);
  const [shareTypeNew, setShareTypeNew] = useState('read');
  const [error, setError] = useState(null);

  const title = useMemo(
    () => (targetType === 'folder' ? 'Share folder' : 'Share document'),
    [targetType]
  );

  const load = useCallback(async () => {
    if (!open) return;
    setError(null);
    setLoading(true);
    try {
      const [u, s] = await Promise.all([
        apiService.getShareableUsers(),
        targetType === 'folder'
          ? apiService.getFolderShares(folderId)
          : apiService.getDocumentShares(documentId),
      ]);
      setUsers(u?.users || []);
      setShares(s?.shares || []);
    } catch (e) {
      setError(e?.message || 'Failed to load sharing');
      setUsers([]);
      setShares([]);
    } finally {
      setLoading(false);
    }
  }, [open, targetType, documentId, folderId]);

  useEffect(() => {
    load();
  }, [load]);

  const handleAdd = async () => {
    if (!pickedUser?.user_id) return;
    setError(null);
    setLoading(true);
    try {
      const body = { shared_with_user_id: pickedUser.user_id, share_type: shareTypeNew };
      if (targetType === 'folder') {
        await apiService.createFolderShare(folderId, body);
      } else {
        await apiService.createDocumentShare(documentId, body);
      }
      setPickedUser(null);
      setShareTypeNew('read');
      await load();
      onSaved?.();
    } catch (e) {
      setError(e?.message || 'Failed to create share');
    } finally {
      setLoading(false);
    }
  };

  const handleChangeShareType = async (shareId, newType) => {
    setError(null);
    try {
      await apiService.updateShare(shareId, { share_type: newType });
      await load();
      onSaved?.();
    } catch (e) {
      setError(e?.message || 'Failed to update share');
    }
  };

  const handleRevoke = async (shareId) => {
    if (!window.confirm('Remove this person\'s access?')) return;
    setError(null);
    try {
      await apiService.revokeShare(shareId);
      await load();
      onSaved?.();
    } catch (e) {
      setError(e?.message || 'Failed to revoke share');
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>{title}</DialogTitle>
      <DialogContent>
        {targetType === 'folder' && (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            People you add can access this folder and everything inside it (depending on read or write).
          </Typography>
        )}
        {error && (
          <Typography color="error" variant="body2" sx={{ mb: 1 }}>
            {error}
          </Typography>
        )}
        <Stack spacing={2}>
          <Autocomplete
            options={users}
            loading={loading}
            getOptionLabel={(o) => o.username || o.user_id || ''}
            value={pickedUser}
            onChange={(_e, v) => setPickedUser(v)}
            renderInput={(params) => <TextField {...params} label="Add user" variant="outlined" />}
          />
          <FormControl>
            <FormLabel>Permission</FormLabel>
            <RadioGroup
              row
              value={shareTypeNew}
              onChange={(e) => setShareTypeNew(e.target.value)}
            >
              <FormControlLabel value="read" control={<Radio size="small" />} label="Read" />
              <FormControlLabel value="write" control={<Radio size="small" />} label="Write" />
            </RadioGroup>
          </FormControl>
          <Button variant="contained" onClick={handleAdd} disabled={loading || !pickedUser}>
            Add share
          </Button>
          <Divider />
          <Typography variant="subtitle2">Shared with</Typography>
          {shares.length === 0 && (
            <Typography variant="body2" color="text.secondary">
              No one yet.
            </Typography>
          )}
          {shares.map((s) => (
            <Stack
              key={s.share_id}
              direction="row"
              alignItems="center"
              spacing={1}
              sx={{ flexWrap: 'wrap' }}
            >
              <Typography sx={{ minWidth: 120 }}>
                {s.shared_with_username || s.shared_with_user_id}
              </Typography>
              <FormControl size="small">
                <RadioGroup
                  row
                  value={s.share_type}
                  onChange={(e) => handleChangeShareType(s.share_id, e.target.value)}
                >
                  <FormControlLabel value="read" control={<Radio size="small" />} label="Read" />
                  <FormControlLabel value="write" control={<Radio size="small" />} label="Write" />
                </RadioGroup>
              </FormControl>
              <IconButton
                size="small"
                aria-label="remove share"
                onClick={() => handleRevoke(s.share_id)}
              >
                <DeleteOutlineIcon fontSize="small" />
              </IconButton>
            </Stack>
          ))}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}
