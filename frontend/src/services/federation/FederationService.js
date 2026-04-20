import ApiServiceBase from '../base/ApiServiceBase';

class FederationService extends ApiServiceBase {
  getIdentity = () => this.get('/api/federation/identity');

  initializeIdentity = () => this.post('/api/federation/identity/initialize', {});

  regenerateIdentity = () => this.post('/api/federation/identity/regenerate', {});

  probePeer = (peerUrl) => this.post('/api/federation/peers/probe', { peer_url: peerUrl });

  initiatePairing = (peerUrl) => this.post('/api/federation/peer-request', { peer_url: peerUrl });

  listPeers = () => this.get('/api/federation/peers');

  patchPeer = (peerId, status) =>
    this.patch(`/api/federation/peers/${encodeURIComponent(peerId)}`, { status });

  syncOutbox = () => this.post('/api/federation/sync', {});

  createFederatedRoom = (body) => this.post('/api/federation/rooms', body);

  resolveRemoteUser = (address) =>
    this.get(`/api/federation/users/resolve-remote?address=${encodeURIComponent(address)}`);

  createUserDmRoom = (body) => this.post('/api/federation/rooms/dm', body);

  listFederatedUsers = () => this.get('/api/federation/federated-users');

  blockFederatedUser = (federatedUserId) =>
    this.post(`/api/federation/federated-users/${encodeURIComponent(federatedUserId)}/block`, {});

  unblockFederatedUser = (federatedUserId) =>
    this.post(`/api/federation/federated-users/${encodeURIComponent(federatedUserId)}/unblock`, {});
}

const federationService = new FederationService();
export default federationService;
