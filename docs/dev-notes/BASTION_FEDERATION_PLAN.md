# Bastion-to-Bastion Federation Plan

Cross-instance federation that allows users on separate Bastion deployments to discover each other, create shared rooms, and exchange messages in real time. This document covers the architecture, data model changes, and a phased implementation plan.

---

## Motivation

Every Bastion instance today is an island — users on `bastion.org-a.com` cannot message users on `bastion.org-b.com`. For teams that span organizations, or for self-hosters who want to communicate without routing everything through a centralized service, direct Bastion-to-Bastion federation fills this gap without requiring third-party platforms.

---

## Design Principles

1. **Admin-controlled trust**: federation is opt-in and established by admins exchanging credentials, not automatic discovery.
2. **Fits existing patterns**: the `connections-service` provider model and `external_connections` trust model are extended, not replaced.
3. **Minimal schema surgery**: the room messaging system gains a new room type and one new table; existing tables are not structurally altered.
4. **Cryptographic identity**: shared symmetric secrets are replaced with asymmetric keypairs so no secret ever leaves an instance.
5. **Custom protocol first**: a purpose-built Bastion Federation Protocol (BFP) keeps v1 scope manageable; Matrix federation can be layered in later.
6. **Topology-agnostic**: federation works when both instances are publicly reachable (bidirectional push) and when only one is publicly reachable (the firewalled instance initiates all connections). Only one side needs a public URL.

---

## Architecture Overview

```
Instance A                             Instance B
──────────────────────────             ──────────────────────────
Frontend                               Frontend
  └─ Federated room UI                   └─ Federated room UI
        │                                       │
Backend (messaging_api)                Backend (messaging_api)
  └─ messaging_service                    └─ messaging_service
        │ detect federated room                  │ inbound federation endpoint
        │                                       │
connections-service                    connections-service
  └─ BastionFederationProvider           └─ (passive; relays to backend)
        │ sign + POST                           │
        └───────────────────────────────────────┘
              HTTPS + Ed25519 signature
```

Messages flow through the existing `messaging_service` and `websocket_manager` on both ends. The `BastionFederationProvider` in `connections-service` handles outbound signing and delivery; a new `federation_api.py` in the backend handles inbound verification and ingestion.

---

## Connectivity Topologies

Federation supports two network topologies. The topology is negotiated during pairing and recorded on each side's `federation_peers` row so the provider knows which delivery strategy to use.

### Bidirectional (both instances publicly reachable)

The default and simplest mode. Both instances push messages to each other via direct HTTPS POST. No polling or persistent connections are needed.

```
Instance A ──POST──▶ Instance B
Instance A ◀──POST── Instance B
```

### Asymmetric (one instance behind a firewall)

When one instance has no public ingress (behind a corporate firewall, NAT, VPN-only network, etc.), it can still federate as long as the peer is publicly reachable. The firewalled instance initiates all connections outbound; the public instance holds pending messages in an outbox until the firewalled instance picks them up.

```
Firewalled Instance A ──POST──▶ Public Instance B       (outbound delivery: works normally)
Firewalled Instance A ──GET───▶ Public Instance B       (inbound pickup: A polls B's outbox)
```

Two delivery mechanisms are available for the inbound direction:

| Mechanism | How it works | Latency | Best for |
|-----------|-------------|---------|----------|
| **Outbox polling** | Firewalled instance periodically calls `GET /api/federation/outbox`. Public instance queues messages in a `federation_outbox` table. | Seconds (tunable poll interval) | Simple deployments, unreliable connections |
| **Persistent stream** | Firewalled instance opens a long-lived `WSS /api/federation/stream` connection outbound. Public instance pushes messages over it in real time. | Sub-second | Low-latency requirements, stable connections |

Both mechanisms can run together — the WebSocket stream provides real-time delivery while the outbox endpoint serves as a catch-up mechanism for messages missed during reconnection gaps.

**Topology detection during pairing:** When Instance A initiates pairing with Instance B, Instance A's backend attempts to determine whether Instance B can reach it back by including its own public URL in the peer-request. Instance B tries a lightweight callback (`GET {instance_a_url}/api/federation/identity`). If the callback fails, Instance B marks the peer as `connectivity_mode = 'asymmetric_caller'` (meaning A can only call out) and reports this in the pairing acceptance. Both sides store the negotiated mode.

---

## Data Model

### New table: `federation_peers`

```sql
CREATE TABLE federation_peers (
    peer_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    peer_url           TEXT NOT NULL UNIQUE,          -- canonical base URL of remote instance
    peer_public_key    TEXT NOT NULL,                  -- Ed25519 public key (base64)
    our_key_id         UUID REFERENCES encryption_keys(key_id),  -- our signing keypair reference
    display_name       TEXT,                           -- human label for this peer (e.g. "Acme Corp")
    status             TEXT NOT NULL DEFAULT 'pending', -- pending | active | suspended | revoked
    connectivity_mode  TEXT NOT NULL DEFAULT 'bidirectional',
        -- bidirectional:       both sides push directly (default)
        -- asymmetric_caller:   this instance cannot receive inbound; it polls/streams from the peer
        -- asymmetric_listener: the peer cannot receive inbound; this instance holds an outbox for it
    allowed_scopes     TEXT[] DEFAULT ARRAY['messaging'],
    initiated_by       UUID REFERENCES users(user_id),
    created_at         TIMESTAMPTZ DEFAULT NOW(),
    activated_at       TIMESTAMPTZ,
    metadata           JSONB DEFAULT '{}'
);
```

### New table: `federated_users`

Represents a remote user who has sent or received messages on this instance. Acts as the local shadow identity.

```sql
CREATE TABLE federated_users (
    federated_user_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    peer_id             UUID NOT NULL REFERENCES federation_peers(peer_id) ON DELETE CASCADE,
    remote_user_id      TEXT NOT NULL,              -- user_id as known on the remote instance
    federated_address   TEXT NOT NULL UNIQUE,       -- e.g. alice@bastion.org-b.com
    display_name        TEXT,
    avatar_url          TEXT,
    last_seen_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (peer_id, remote_user_id)
);
```

### Changes to existing tables

| Table | Change |
|-------|--------|
| `chat_rooms` | Add `room_type` value `'federated'`; add nullable `federation_metadata JSONB` (stores `peer_id`, `remote_room_id`, `sync_cursor`). |
| `chat_messages` | Add nullable `federated_sender_id UUID REFERENCES federated_users`. When set, `sender_id` may be NULL (the message is from a remote user). |
| `users` | No change — remote identities live in `federated_users`, not the main users table. |

### New table: `federation_outbox`

Holds pending outbound messages for peers that cannot receive direct pushes (i.e., peers with `connectivity_mode = 'asymmetric_listener'`). The firewalled peer drains the outbox via polling or a persistent stream.

```sql
CREATE TABLE federation_outbox (
    outbox_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    peer_id         UUID NOT NULL REFERENCES federation_peers(peer_id) ON DELETE CASCADE,
    event_type      TEXT NOT NULL,       -- 'message' | 'room_invite' | 'peer_accept' | 'read_receipt' | ...
    payload         JSONB NOT NULL,      -- signed BFP envelope
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    picked_up_at    TIMESTAMPTZ          -- set when the peer acknowledges receipt
);
CREATE INDEX idx_federation_outbox_pending
    ON federation_outbox (peer_id, created_at)
    WHERE picked_up_at IS NULL;
```

A background task periodically prunes rows where `picked_up_at` is older than a configurable retention window (default 7 days).

### Instance keypair

Stored in the existing `encryption_keys` table (or a new `instance_keypair` settings key) as an Ed25519 keypair generated on first federation setup. The public key is served at `GET /api/federation/identity`.

---

## New Components

### `backend/api/federation_api.py`

| Endpoint | Auth | Purpose |
|----------|------|---------|
| `GET /api/federation/identity` | Public | Returns `{instance_url, public_key, display_name, version}`. Used during pairing. |
| `POST /api/federation/peer-request` | Admin JWT | Initiate or accept a peer relationship. |
| `POST /api/federation/message` | Ed25519 signature (peer) | Receive a signed inbound message from a remote instance. |
| `POST /api/federation/room-invite` | Ed25519 signature (peer) | Receive a room invitation (remote user wants to start a federated room). |
| `GET /api/federation/peers` | Admin JWT | List configured peers and their status. |
| `PATCH /api/federation/peers/{peer_id}` | Admin JWT | Update status (activate / suspend / revoke). |
| `GET /api/federation/outbox` | Ed25519 signature (peer) | Drain pending outbox events for the calling peer (asymmetric mode). Returns events oldest-first; caller ACKs with the last received `outbox_id`. |
| `WSS /api/federation/stream` | Ed25519 signature (peer) | Persistent outbound stream for a firewalled peer. The peer opens this connection; the server pushes events in real time. Falls back to outbox polling if the connection drops. |

Inbound message verification:

```python
# Pseudocode
def verify_federation_request(request, peer_url):
    peer = get_peer_by_url(peer_url)
    if not peer or peer.status != 'active':
        raise Forbidden()
    signature = request.headers['X-Bastion-Signature']
    body_bytes = await request.body()
    verify_ed25519(peer.peer_public_key, body_bytes, signature)  # raises on failure
```

### `connections-service/providers/bastion_federation_provider.py`

Implements `BaseMessagingProvider` for outbound delivery to peer instances.

Key responsibilities:
- Hold the instance's Ed25519 private key for signing outbound payloads.
- `send_message(chat_id, text, ...)` → signs payload → delivery path depends on `connectivity_mode`:
  - **`bidirectional`** — direct `POST {peer_url}/api/federation/message` (default).
  - **`asymmetric_listener`** — the peer cannot receive pushes; write the signed payload to `federation_outbox` for later pickup.
  - **`asymmetric_caller`** — this instance cannot receive pushes; outbound delivery still uses direct POST (we *can* reach the peer). Inbound messages arrive via the poll/stream path described below.
- `start()` / `stop()`:
  - In `bidirectional` mode these are lightweight (no long-lived socket).
  - In `asymmetric_caller` mode, `start()` opens a persistent `WSS` stream to the peer (or starts a polling loop against `GET /api/federation/outbox`). `stop()` tears down the connection.
- Retry with exponential backoff on transient failures; write a delivery receipt or error to `federation_metadata` on the room.

### `backend/services/federation_service.py`

Orchestrates federation logic:
- Keypair generation and retrieval.
- Pairing handshake (initiate → fetch remote identity → connectivity probe → save peer as `pending` → notify admin).
- Connectivity probe: during pairing, attempts a callback to the peer's `/api/federation/identity` to determine reachability and negotiate `connectivity_mode`.
- Federated user lookup / creation (`get_or_create_federated_user`).
- Routing inbound messages to `messaging_service` with a synthetic sender.
- Constructing the outbound signed payload from a `ChatMessage`.
- Outbox management: enqueue events for `asymmetric_listener` peers; serve outbox drain requests; prune acknowledged entries.

---

## Message Wire Format

```json
{
  "bfp_version": "1",
  "from_instance": "https://bastion.org-a.com",
  "from_user": "alice@bastion.org-a.com",
  "from_user_display_name": "Alice",
  "room_id": "<remote_room_id_on_receiver>",
  "message_id": "<uuid>",
  "sent_at": "2026-04-13T12:00:00Z",
  "content": {
    "type": "text",
    "text": "Hey, can you review this?"
  },
  "attachments": []
}
```

The entire JSON body is signed with the sending instance's Ed25519 private key; the signature is placed in the `X-Bastion-Signature` header (base64url-encoded).

---

## Admin Pairing Flow

### Bidirectional pairing (both publicly reachable)

```
Admin A (Settings → Federation → Add Peer)
  1. Enters peer URL: https://bastion.org-b.com
  2. Backend calls GET https://bastion.org-b.com/api/federation/identity
  3. Response: {public_key, display_name, version}
  4. Admin A reviews and confirms → POST /api/federation/peer-request
     (status=pending, stores public key)
  5. Backend sends a signed peer-request to https://bastion.org-b.com/api/federation/peer-request
     Request includes Instance A's public URL for the connectivity probe.
  6. Instance B attempts GET {instance_a_url}/api/federation/identity (connectivity probe).
     Probe succeeds → Instance B records connectivity_mode = 'bidirectional'.

Admin B (Settings → Federation → Pending Requests)
  7. Sees incoming request from bastion.org-a.com with display name
  8. Reviews and approves → PATCH /api/federation/peers/{peer_id} {status: active}
  9. Backend signs and POSTs acceptance back to org-a

Both sides now have status=active, connectivity_mode=bidirectional peer records.
```

### Asymmetric pairing (Instance A behind a firewall)

The firewalled instance must be the initiator — it is the only side that can make outbound calls.

```
Admin A — firewalled (Settings → Federation → Add Peer)
  1. Enters peer URL: https://bastion.org-b.com
  2. Backend calls GET https://bastion.org-b.com/api/federation/identity
  3. Response: {public_key, display_name, version}
  4. Admin A reviews and confirms → POST /api/federation/peer-request
     Request includes Instance A's public URL (or explicitly declares "no public ingress").
  5. Instance B attempts the connectivity probe → GET {instance_a_url}/api/federation/identity.
     Probe fails (timeout / unreachable) → Instance B records connectivity_mode = 'asymmetric_listener'
     (Instance B will hold an outbox for A). Instance A records connectivity_mode = 'asymmetric_caller'.

Admin B (Settings → Federation → Pending Requests)
  6. Sees incoming request with note: "This peer is not publicly reachable — outbox delivery will be used."
  7. Reviews and approves → writes acceptance to federation_outbox (cannot POST it back).
  8. Instance A picks up the acceptance on its next poll/stream cycle.

Both sides now have status=active peer records with the negotiated asymmetric connectivity mode.
```

No shared secret is ever transmitted — only each instance's public key. Private keys never leave their respective instances.

---

## End-to-End Message Flow

### Bidirectional (default)

**User A sends a message in a federated room:**

```
1. User A types → frontend calls POST /api/messaging/rooms/{room_id}/messages
2. messaging_service saves message locally (sender_id = User A's local UUID)
3. messaging_service detects room_type = 'federated'; reads federation_metadata.peer_id
4. messaging_service calls connections_service_client.send_outbound_message(
       provider="bastion_federation",
       peer_url=peer.peer_url,
       room_id=federation_metadata.remote_room_id,
       message=payload
   )
5. connections-service BastionFederationProvider signs payload with instance A private key
6. HTTP POST to https://instance-b.com/api/federation/message
7. Instance B federation_api verifies signature against stored instance A public key
8. federation_service.get_or_create_federated_user("alice@bastion.org-a.com")
9. messaging_service.create_message(room_id=local_room_id, federated_sender_id=..., content=...)
10. websocket_manager broadcasts to participants in the local room on Instance B
11. User B's browser receives the message in real time
```

When User B replies, the same flow runs in reverse — Instance B POSTs directly to Instance A.

### Asymmetric (Instance A behind a firewall, Instance B public)

**User A (firewalled) sends a message → Instance B:**

Steps 1–6 are identical to the bidirectional flow. Instance A can reach Instance B directly, so outbound delivery is a normal POST.

**User B (public) sends a message → Instance A (firewalled):**

```
1–3. Same as bidirectional: message saved locally on Instance B, federated room detected.
4.   BastionFederationProvider checks connectivity_mode = 'asymmetric_listener' for this peer.
5.   Instead of POSTing, the provider writes the signed payload to federation_outbox
     (peer_id = Instance A, event_type = 'message').
6.   Instance A's provider (running in asymmetric_caller mode) picks up the message via either:
     a. Persistent WSS stream — Instance B pushes the event immediately over the open connection.
     b. Outbox poll — Instance A calls GET /api/federation/outbox on its next poll cycle.
7.   Instance A's federation_api processes the event as if it were a normal inbound POST:
     verify signature → get_or_create_federated_user → create_message → websocket broadcast.
8.   Instance A ACKs the outbox event (the next poll includes the last processed outbox_id,
     or the WSS stream receives an ACK frame). Instance B marks the row picked_up_at = NOW().
```

From the user's perspective, the experience is identical in both topologies. The only observable difference is slightly higher latency in poll mode (bounded by the poll interval, default 5 seconds).

---

## Phased Implementation Plan

### Phase 1 — Foundation (Identity + Admin Pairing UI)

**Goal:** Instances can establish trust. No messages yet.

**Scope:**
- Generate and persist instance Ed25519 keypair on first setup (or via admin action).
- `GET /api/federation/identity` endpoint (public).
- `federation_peers` table migration (includes `connectivity_mode` column).
- `federation_outbox` table migration.
- `POST /api/federation/peer-request` and `PATCH /api/federation/peers/{peer_id}` endpoints.
- Connectivity probe during pairing: attempt callback to the initiator's URL; negotiate and persist `connectivity_mode` on both sides.
- Admin settings tab: **Federation** panel — "This instance's identity," "Add peer," "Pending requests," "Active peers." Active peers display shows connectivity mode (e.g., "Direct" vs "Outbox — this peer polls us").
- `federation_service.py` with keypair management, pairing handshake logic, and connectivity negotiation.
- Basic Ed25519 signing and verification utility (`backend/utils/federation_crypto.py`).

**Deliverable:** Admin on Instance A can initiate a pairing with Instance B; Admin B can accept. Both instances show `status=active` in their federation peers list, with the correct connectivity mode. This works whether Instance A is publicly reachable or behind a firewall.

---

### Phase 2 — Federated Rooms + Message Delivery

**Goal:** Users can create a shared room and exchange text messages across instances.

**Scope:**
- `federated_users` table migration.
- `chat_rooms.room_type` extended to include `'federated'`; add `federation_metadata JSONB` column.
- `chat_messages.federated_sender_id` nullable column.
- `POST /api/federation/message` and `POST /api/federation/room-invite` inbound endpoints.
- `GET /api/federation/outbox` endpoint for asymmetric peers to drain pending events.
- `WSS /api/federation/stream` endpoint for real-time push to connected asymmetric peers.
- `BastionFederationProvider` in `connections-service`:
  - Outbound signing + delivery (direct POST for bidirectional; outbox write for asymmetric_listener peers).
  - Inbound pickup for asymmetric_caller mode: polling loop or persistent WebSocket stream.
  - Retry logic for all modes.
- `messaging_service` updated to detect federated rooms and route outbound messages via provider.
- `federation_service.get_or_create_federated_user()` for inbound identity resolution.
- `federation_service` outbox enqueue/drain/prune logic.
- Frontend: "New federated room" flow — select active peer, enter remote username → sends room invite.
- Frontend: Display federated sender name and instance badge in room message list.

**Deliverable:** User A on Instance A can invite User B on Instance B to a room. Both can send and receive text messages in real time. This works in both bidirectional and asymmetric topologies.

---

### Phase 3 — Federated User Discovery

**Goal:** Users can search for remote users by federated address without needing an admin to create the room.

**Scope:**
- `GET /api/federation/users/resolve?address=alice@bastion.org-b.com` — proxied lookup: Instance A calls Instance B's user directory endpoint, returns display name + avatar.
- `GET /api/federation/users` on each instance returns a (scoped) directory of users who have opted in to federated discovery (`users.federation_discoverable` boolean flag).
- Frontend: User search in "New Message" modal accepts `@user@instance` format.
- Direct Messages extended to support federated direct rooms (`room_type='federated'`, `room_sub_type='direct'`).

**Deliverable:** Users can type `@alice@bastion.org-b.com` in the new message dialog and start a DM without admin involvement.

---

### Phase 4 — Hardening and Feature Parity

**Goal:** Federation is production-ready and feature-complete for messaging.

**Scope:**
- **File/image attachments** in federated messages: attachments are proxied or referenced via signed time-limited URLs.
- **Reactions** forwarded to remote instance (new wire message type `reaction`).
- **Read receipts** (optional, respects privacy settings): `POST /api/federation/read-receipt`.
- **Presence** (optional): instances can poll or subscribe to presence updates for federated contacts.
- **Delivery status** indicators in the room UI: local saved, delivered to peer instance, failed.
- **Peer suspension / revocation**: messaging_service drops messages for suspended peers; UI shows "Federation suspended" banner in affected rooms.
- **Rate limiting and abuse controls**: per-peer inbound message rate limits; admin can block specific federated users.
- **Audit log entries** for all federation events (peer added, revoked, inbound message received).
- End-to-end tests covering the full sign → deliver → verify → broadcast path.

**Deliverable:** Federation is stable and trustworthy enough for production use across organizations.

---

### Phase 5 — Matrix Protocol Compatibility (Stretch)

**Goal:** Bastion rooms can federate with the broader Matrix ecosystem.

This phase replaces or supplements BFP with the Matrix federation protocol, allowing Bastion users to communicate with any Matrix homeserver (Element, Conduit, etc.).

**Considerations:**
- Matrix federation is a full homeserver protocol — substantial implementation effort.
- `matrix-nio` (async Python) is the best available client-side library; a full server implementation would need Synapse or Conduit as a sidecar.
- The simplest approach is a Matrix **application service** bridge: run a small Matrix homeserver (Conduit) alongside Bastion, and bridge `chat_rooms` ↔ Matrix rooms via the AS API. This avoids reimplementing Matrix federation from scratch.
- Federated users from Matrix would appear as `federated_users` with `peer_id` pointing to a special "Matrix bridge" peer.

**Deliverable:** Bastion users can exchange messages with Matrix users on external homeservers.

---

## Key Files Affected

| File | Change |
|------|--------|
| `backend/sql/migrations/` | New migration: `federation_peers` (with `connectivity_mode`), `federation_outbox`, `federated_users`, schema changes to `chat_rooms`, `chat_messages` |
| `backend/api/federation_api.py` | New file (includes outbox drain + WebSocket stream endpoints) |
| `backend/services/federation_service.py` | New file (pairing, connectivity probe, outbox management) |
| `backend/utils/federation_crypto.py` | New file (Ed25519 sign/verify utilities) |
| `backend/services/messaging/messaging_service.py` | Detect federated rooms; call outbound provider |
| `backend/clients/connections_service_client.py` | Add `send_outbound_message` variant for federation provider |
| `connections-service/providers/bastion_federation_provider.py` | New file (bidirectional push, outbox write, poll/stream inbound) |
| `connections-service/service/provider_router.py` | Register `BastionFederationProvider` |
| `connections-service/service/grpc_service.py` | Expose `RegisterFederationPeer` gRPC method if needed |
| `frontend/src/components/SettingsPage.js` | Add Federation admin tab (displays connectivity mode per peer) |
| `frontend/src/components/messaging/` | New federated room UI components |
| `backend/main.py` | Register `federation_api` router |

---

## Open Questions

1. **Privacy / opt-in**: Should users be able to opt out of federation entirely (refuse all federated room invites)? Per-user setting or admin-only control?
2. **Message retention across instances**: If User A deletes a message locally, should a deletion event be sent to Instance B? Define semantics for soft delete propagation.
3. **E2EE**: The existing `room_encryption_keys` stub is scoped to local users. Federated E2EE requires a proper key exchange protocol (e.g., Signal's X3DH or MLS). Defer to Phase 4 or later.
4. **Instance URL mutability**: What happens if Instance B changes its canonical URL? Need a procedure for re-keying a peer without losing room history.
5. **Scope beyond messaging**: Could federation extend to shared agent access, document sharing, or team membership? Kept out of scope for now via `allowed_scopes` on the peer record.
6. **Both instances firewalled**: If neither instance is publicly reachable, federation is not possible without a relay. Should we document this as an unsupported topology, or plan for an optional relay service (e.g., a lightweight public relay that both instances connect to outbound)?
7. **Outbox retention and backpressure**: If a firewalled peer goes offline for an extended period, the outbox grows unbounded. Define a maximum outbox size or age limit, and the behavior when the limit is hit (drop oldest? suspend the peer? notify the admin?).

---

## References

- Existing channel provider pattern: `connections-service/providers/base_messaging_provider.py`
- Channels roadmap (Matrix stretch goal): `docs/dev-notes/CHANNELS_ROADMAP_AND_DEPLOYMENT.md`
- Messaging system audit: `docs/dev-notes/MESSAGING_IMPROVEMENTS.md`
- OpenFang comparison (federation gap noted): `docs/dev-notes/OPENFANG_COMPARISON.md`
- External connections service: `backend/services/external_connections_service.py`
