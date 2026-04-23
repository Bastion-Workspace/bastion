import React, { useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Divider,
  IconButton,
  InputAdornment,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
} from '@mui/material';
import { Block, ContentCopy, Hub, LockOpen, Refresh, Sync } from '@mui/icons-material';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import apiService from '../services/apiService';

function connectivityLabel(mode) {
  if (mode === 'asymmetric_caller') return 'Outbound poll (this instance)';
  if (mode === 'asymmetric_listener') return 'Outbox (peer polls us)';
  return 'Direct';
}

export default function FederationSettings() {
  const queryClient = useQueryClient();
  const [peerUrl, setPeerUrl] = useState('');
  const [probeResult, setProbeResult] = useState(null);
  const [regenOpen, setRegenOpen] = useState(false);
  const [removePeerTarget, setRemovePeerTarget] = useState(null);
  const [copyOk, setCopyOk] = useState(false);
  const [fedPeerId, setFedPeerId] = useState('');
  const [fedRemoteAddr, setFedRemoteAddr] = useState('');
  const [fedRoomName, setFedRoomName] = useState('');

  const identityQuery = useQuery(
    'federationIdentity',
    () => apiService.federation.getIdentity(),
    { retry: false }
  );

  const err0 = identityQuery.error;
  const status404Early = err0?.response?.status === 404;
  const detail0 = err0?.response?.data?.detail || '';
  const federationDisabledEarly =
    status404Early &&
    typeof detail0 === 'string' &&
    detail0.toLowerCase().includes('not enabled');

  const peersQuery = useQuery('federationPeers', () => apiService.federation.listPeers(), {
    enabled: !federationDisabledEarly && identityQuery.status !== 'loading',
    retry: false,
  });

  const initMutation = useMutation(() => apiService.federation.initializeIdentity(), {
    onSuccess: () => {
      queryClient.invalidateQueries('federationIdentity');
      queryClient.invalidateQueries('federationPeers');
    },
  });

  const regenMutation = useMutation(() => apiService.federation.regenerateIdentity(), {
    onSuccess: () => {
      queryClient.invalidateQueries('federationIdentity');
      queryClient.invalidateQueries('federationPeers');
      setRegenOpen(false);
    },
  });

  const probeMutation = useMutation(
    (url) => apiService.federation.probePeer(url),
    {
      onSuccess: (data) => setProbeResult(data.remote || data),
    }
  );

  const pairMutation = useMutation(
    (url) => apiService.federation.initiatePairing(url),
    {
      onSuccess: () => {
        setPeerUrl('');
        setProbeResult(null);
        queryClient.invalidateQueries('federationPeers');
      },
    }
  );

  const patchMutation = useMutation(
    ({ peerId, status }) => apiService.federation.patchPeer(peerId, status),
    {
      onSuccess: () => queryClient.invalidateQueries('federationPeers'),
    }
  );

  const deletePeerMutation = useMutation(
    (peerId) => apiService.federation.deletePeer(peerId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('federationPeers');
        queryClient.invalidateQueries('federationFederatedUsers');
        setRemovePeerTarget(null);
      },
    }
  );

  const syncMutation = useMutation(() => apiService.federation.syncOutbox(), {
    onSuccess: () => queryClient.invalidateQueries('federationPeers'),
  });

  const createFedRoomMutation = useMutation(
    (payload) => apiService.federation.createFederatedRoom(payload),
    {
      onSuccess: () => {
        setFedRemoteAddr('');
        setFedRoomName('');
        queryClient.invalidateQueries('federationPeers');
      },
    }
  );

  const federatedUsersQuery = useQuery(
    'federationFederatedUsers',
    () => apiService.federation.listFederatedUsers(),
    {
      enabled: !federationDisabledEarly && identityQuery.isSuccess,
      retry: false,
    }
  );

  const toggleBlockFedMutation = useMutation(
    ({ federatedUserId, block }) =>
      block
        ? apiService.federation.blockFederatedUser(federatedUserId)
        : apiService.federation.unblockFederatedUser(federatedUserId),
    {
      onSuccess: () => queryClient.invalidateQueries('federationFederatedUsers'),
    }
  );

  const err = identityQuery.error;
  const status404 = err?.response?.status === 404;
  const detail = err?.response?.data?.detail || '';
  const federationDisabled = federationDisabledEarly;
  const notInitialized =
    status404 && typeof detail === 'string' && detail.toLowerCase().includes('not initialized');

  const copyKey = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopyOk(true);
      setTimeout(() => setCopyOk(false), 2000);
    } catch {
      /* ignore */
    }
  };

  if (identityQuery.isLoading) {
    return (
      <Box display="flex" justifyContent="center" p={4}>
        <CircularProgress />
      </Box>
    );
  }

  if (federationDisabled) {
    return (
      <Alert severity="info" sx={{ m: 2 }}>
        Federation is disabled on this server. Set <code>FEDERATION_ENABLED=true</code> in the
        backend environment and restart.
      </Alert>
    );
  }

  if (identityQuery.isError && !notInitialized && !federationDisabled) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {typeof detail === 'string' ? detail : 'Failed to load federation identity'}
      </Alert>
    );
  }

  const identity = identityQuery.data;
  const peers = peersQuery.data?.peers || [];

  const inboundPending = peers.filter((p) => p.is_inbound && p.status === 'pending');
  const outboundPending = peers.filter((p) => !p.is_inbound && p.status === 'pending');

  return (
    <Box sx={{ p: 2, maxWidth: 1100 }}>
      <Box display="flex" alignItems="center" gap={1} mb={2}>
        <Hub color="primary" />
        <Typography variant="h5">Federation</Typography>
        <Chip size="small" label="Admin" color="warning" />
      </Box>

      {(notInitialized || !identity) && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Generate an instance keypair so other Bastion deployments can verify this server.
        </Alert>
      )}

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            This instance
          </Typography>
          {identity ? (
            <>
              <Typography variant="body2" color="text.secondary">
                URL: {identity.instance_url}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Display name: {identity.display_name}
              </Typography>
              <Typography variant="body2" sx={{ mt: 1, wordBreak: 'break-all' }}>
                Public key: {identity.public_key}
              </Typography>
              <Box mt={1} display="flex" gap={1} flexWrap="wrap">
                <Tooltip title={copyOk ? 'Copied' : 'Copy public key'}>
                  <IconButton onClick={() => copyKey(identity.public_key)} size="small">
                    <ContentCopy fontSize="small" />
                  </IconButton>
                </Tooltip>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<Refresh />}
                  onClick={() => setRegenOpen(true)}
                >
                  Regenerate keypair
                </Button>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<Sync />}
                  onClick={() => syncMutation.mutate()}
                  disabled={syncMutation.isLoading}
                >
                  Sync outbox (asymmetric)
                </Button>
              </Box>
              {syncMutation.isSuccess && (
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                  Pulled {syncMutation.data?.pulled ?? 0} event(s).
                </Typography>
              )}
            </>
          ) : (
            <Button
              variant="contained"
              onClick={() => initMutation.mutate()}
              disabled={initMutation.isLoading}
            >
              {initMutation.isLoading ? 'Generating…' : 'Generate keypair'}
            </Button>
          )}
        </CardContent>
      </Card>

      {identity && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Federated users (moderation)
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Block or unblock remote identities seen on federated chats. Blocked users are ignored for
              inbound federation events.
            </Typography>
            {federatedUsersQuery.isLoading && <CircularProgress size={24} />}
            {federatedUsersQuery.isError && (
              <Alert severity="warning">
                {federatedUsersQuery.error?.response?.data?.detail ||
                  federatedUsersQuery.error?.message ||
                  'Could not load federated users'}
              </Alert>
            )}
            {federatedUsersQuery.data?.users?.length > 0 && (
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Display name</TableCell>
                    <TableCell>Address</TableCell>
                    <TableCell>Peer</TableCell>
                    <TableCell>Presence</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {federatedUsersQuery.data.users.map((u) => (
                    <TableRow key={u.federated_user_id}>
                      <TableCell>{u.display_name || '—'}</TableCell>
                      <TableCell sx={{ wordBreak: 'break-all' }}>{u.federated_address}</TableCell>
                      <TableCell sx={{ maxWidth: 180, wordBreak: 'break-all' }}>
                        {u.peer_display_name || u.peer_url}
                      </TableCell>
                      <TableCell>{u.presence_status || '—'}</TableCell>
                      <TableCell align="right">
                        {u.is_blocked ? (
                          <Button
                            size="small"
                            startIcon={<LockOpen />}
                            disabled={toggleBlockFedMutation.isLoading}
                            onClick={() =>
                              toggleBlockFedMutation.mutate({
                                federatedUserId: u.federated_user_id,
                                block: false,
                              })
                            }
                          >
                            Unblock
                          </Button>
                        ) : (
                          <Button
                            size="small"
                            color="warning"
                            startIcon={<Block />}
                            disabled={toggleBlockFedMutation.isLoading}
                            onClick={() =>
                              toggleBlockFedMutation.mutate({
                                federatedUserId: u.federated_user_id,
                                block: true,
                              })
                            }
                          >
                            Block
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
            {federatedUsersQuery.isSuccess &&
              (!federatedUsersQuery.data?.users || federatedUsersQuery.data.users.length === 0) && (
                <Typography variant="body2" color="text.secondary">
                  No federated users recorded yet.
                </Typography>
              )}
          </CardContent>
        </Card>
      )}

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Add peer
          </Typography>
          <TextField
            fullWidth
            label="Peer base URL"
            placeholder="https://other-bastion.example.com"
            value={peerUrl}
            onChange={(e) => setPeerUrl(e.target.value)}
            sx={{ mb: 2 }}
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <Button
                    size="small"
                    onClick={() => peerUrl.trim() && probeMutation.mutate(peerUrl.trim())}
                    disabled={probeMutation.isLoading || !peerUrl.trim()}
                  >
                    Fetch identity
                  </Button>
                </InputAdornment>
              ),
            }}
          />
          {probeMutation.isError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {probeMutation.error?.response?.data?.detail ||
                probeMutation.error?.message ||
                'Probe failed'}
            </Alert>
          )}
          {probeResult && (
            <Alert severity="success" sx={{ mb: 2 }}>
              Remote: {probeResult.display_name} — key {String(probeResult.public_key).slice(0, 24)}
              …
            </Alert>
          )}
          <Button
            variant="contained"
            disabled={!peerUrl.trim() || pairMutation.isLoading || !identity}
            onClick={() => pairMutation.mutate(peerUrl.trim())}
          >
            {pairMutation.isLoading ? 'Sending…' : 'Send pairing request'}
          </Button>
          {pairMutation.isError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {pairMutation.error?.response?.data?.detail ||
                pairMutation.error?.message ||
                'Pairing failed'}
            </Alert>
          )}
          {outboundPending.length > 0 && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              Waiting on remote admin: {outboundPending.map((p) => p.peer_url).join(', ')}. Use
              &quot;Sync outbox&quot; if the peer cannot reach this instance directly.
            </Typography>
          )}
        </CardContent>
      </Card>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            New federated room
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Creates a federated chat on this instance and sends a signed invite to the peer. Use the
            remote user&apos;s <strong>username@peer-host</strong> (host must match the peer&apos;s URL).
          </Typography>
          <FormControl fullWidth sx={{ mb: 2 }} size="small">
            <InputLabel id="fed-peer-label">Peer</InputLabel>
            <Select
              labelId="fed-peer-label"
              label="Peer"
              value={fedPeerId}
              onChange={(e) => setFedPeerId(e.target.value)}
            >
              {peers
                .filter((p) => p.status === 'active')
                .map((p) => (
                  <MenuItem key={p.peer_id} value={p.peer_id}>
                    {p.display_name || p.peer_url}
                  </MenuItem>
                ))}
            </Select>
          </FormControl>
          <TextField
            fullWidth
            size="small"
            label="Remote user address"
            placeholder="alice@other-bastion.example.com"
            value={fedRemoteAddr}
            onChange={(e) => setFedRemoteAddr(e.target.value)}
            sx={{ mb: 2 }}
          />
          <TextField
            fullWidth
            size="small"
            label="Room name (optional)"
            value={fedRoomName}
            onChange={(e) => setFedRoomName(e.target.value)}
            sx={{ mb: 2 }}
          />
          <Button
            variant="contained"
            disabled={
              !fedPeerId ||
              !fedRemoteAddr.trim() ||
              createFedRoomMutation.isLoading ||
              !identity
            }
            onClick={() =>
              createFedRoomMutation.mutate({
                peer_id: fedPeerId,
                remote_user_address: fedRemoteAddr.trim(),
                room_name: fedRoomName.trim() || undefined,
              })
            }
          >
            {createFedRoomMutation.isLoading ? 'Creating…' : 'Create & send invite'}
          </Button>
          {createFedRoomMutation.isError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {createFedRoomMutation.error?.response?.data?.detail ||
                createFedRoomMutation.error?.message ||
                'Failed'}
            </Alert>
          )}
          {createFedRoomMutation.isSuccess && (
            <Alert severity="success" sx={{ mt: 2 }}>
              Federated room created. Room id: {createFedRoomMutation.data?.room?.room_id}. If the
              peer uses asymmetric connectivity, run &quot;Sync outbox&quot; on both sides after the
              remote admin accepts.
            </Alert>
          )}
        </CardContent>
      </Card>

      {inboundPending.length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Pending inbound requests
            </Typography>
            <Divider sx={{ mb: 2 }} />
            {inboundPending.map((p) => (
              <Box
                key={p.peer_id}
                display="flex"
                alignItems="center"
                justifyContent="space-between"
                flexWrap="wrap"
                gap={1}
                mb={2}
              >
                <Box>
                  <Typography fontWeight="medium">{p.display_name || p.peer_url}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    {p.peer_url}
                  </Typography>
                  {p.connectivity_mode === 'asymmetric_listener' && (
                    <Typography variant="caption" color="text.secondary" display="block">
                      This peer is not publicly reachable — outbox delivery will be used.
                    </Typography>
                  )}
                </Box>
                <Box display="flex" gap={1}>
                  <Button
                    variant="contained"
                    color="success"
                    size="small"
                    onClick={() => patchMutation.mutate({ peerId: p.peer_id, status: 'active' })}
                    disabled={patchMutation.isLoading}
                  >
                    Approve
                  </Button>
                  <Button
                    variant="outlined"
                    color="error"
                    size="small"
                    onClick={() => patchMutation.mutate({ peerId: p.peer_id, status: 'revoked' })}
                    disabled={patchMutation.isLoading}
                  >
                    Reject
                  </Button>
                </Box>
              </Box>
            ))}
          </CardContent>
        </Card>
      )}

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            All peers
          </Typography>
          {peersQuery.isError && (
            <Alert severity="error">
              {peersQuery.error?.response?.data?.detail || peersQuery.error?.message}
            </Alert>
          )}
          {!peersQuery.isError && (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>URL</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Connectivity</TableCell>
                  <TableCell align="right">Outbox pending</TableCell>
                  <TableCell>Last sync</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {peers.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={7}>
                      <Typography variant="body2" color="text.secondary">
                        No peers yet.
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
                {peers.map((p) => (
                  <TableRow key={p.peer_id}>
                    <TableCell>{p.display_name || '—'}</TableCell>
                    <TableCell sx={{ maxWidth: 280, wordBreak: 'break-all' }}>{p.peer_url}</TableCell>
                    <TableCell>
                      <Chip size="small" label={p.status} />
                    </TableCell>
                    <TableCell>
                      <Chip size="small" variant="outlined" label={connectivityLabel(p.connectivity_mode)} />
                    </TableCell>
                    <TableCell align="right">
                      {p.outbox_pending_count != null ? p.outbox_pending_count : '—'}
                    </TableCell>
                    <TableCell sx={{ maxWidth: 200, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {p.last_sync_at || '—'}
                    </TableCell>
                    <TableCell align="right">
                      {p.status === 'active' && (
                        <>
                          <Button
                            size="small"
                            onClick={() => patchMutation.mutate({ peerId: p.peer_id, status: 'suspended' })}
                            disabled={patchMutation.isLoading}
                          >
                            Suspend
                          </Button>
                          <Button
                            size="small"
                            color="error"
                            onClick={() => patchMutation.mutate({ peerId: p.peer_id, status: 'revoked' })}
                            disabled={patchMutation.isLoading}
                          >
                            Revoke
                          </Button>
                        </>
                      )}
                      {p.status === 'revoked' && (
                        <Button
                          size="small"
                          color="error"
                          variant="outlined"
                          onClick={() =>
                            setRemovePeerTarget({
                              peer_id: p.peer_id,
                              peer_url: p.peer_url,
                              display_name: p.display_name,
                            })
                          }
                          disabled={deletePeerMutation.isLoading}
                        >
                          Remove
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Dialog open={regenOpen} onClose={() => setRegenOpen(false)}>
        <DialogTitle>Regenerate federation keypair?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Existing peers may need to re-pair if you regenerate this instance&apos;s keys.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRegenOpen(false)}>Cancel</Button>
          <Button
            color="warning"
            variant="contained"
            onClick={() => regenMutation.mutate()}
            disabled={regenMutation.isLoading}
          >
            Regenerate
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog
        open={Boolean(removePeerTarget)}
        onClose={() => !deletePeerMutation.isLoading && setRemovePeerTarget(null)}
      >
        <DialogTitle>Remove revoked peer?</DialogTitle>
        <DialogContent>
          <DialogContentText component="div">
            <Typography variant="body2" paragraph>
              This permanently deletes the peer record for{' '}
              <strong>{removePeerTarget?.display_name || removePeerTarget?.peer_url || 'this peer'}</strong>.
              The URL can be added again for a fresh pairing.
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Historical federated messages may lose stored federated sender linkage for users tied
              to this peer. Federated rooms may show incomplete federation metadata until you clean
              up or re-pair.
            </Typography>
            {deletePeerMutation.isError && (
              <Alert severity="error" sx={{ mt: 1 }}>
                {deletePeerMutation.error?.response?.data?.detail ||
                  deletePeerMutation.error?.message ||
                  'Remove failed'}
              </Alert>
            )}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRemovePeerTarget(null)} disabled={deletePeerMutation.isLoading}>
            Cancel
          </Button>
          <Button
            color="error"
            variant="contained"
            onClick={() =>
              removePeerTarget?.peer_id && deletePeerMutation.mutate(removePeerTarget.peer_id)
            }
            disabled={deletePeerMutation.isLoading || !removePeerTarget?.peer_id}
          >
            {deletePeerMutation.isLoading ? 'Removing…' : 'Remove peer'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
