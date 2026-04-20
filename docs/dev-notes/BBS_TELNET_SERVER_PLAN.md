# BBS Telnet Server — Architecture & Implementation Plan

## Overview

A telnet-accessible BBS (Bulletin Board System) that provides a text-mode interface to Bastion's full service suite. Users connect via any telnet client — including real MS-DOS machines with mTCP, PuTTY, vintage terminals, or `telnet host 2323` from any Unix shell — and interact with the same AI chat, documents, RSS feeds, data workspaces, and agent system available through the web UI.

**Core principles:**
- **100% optional** — no existing service depends on the BBS server; omitting it from `docker-compose.yml` has zero impact on any other feature
- **Zero backend changes** — the BBS server is a pure consumer of existing HTTP APIs and the internal service API
- **Per-user isolation** — authenticates against Bastion's auth system; every user sees only their own data, enforced by the same RLS policies and `user_id` scoping as the web UI
- **Transport-agnostic reuse** — follows the same integration pattern as the Telegram/Discord bots in `connections-service`, calling the same internal API endpoints
- **Classic BBS UX** — numbered menus, ANSI color, box drawing, `--More--` pagination, slash commands

---

## Deployment: Completely Optional

The BBS server is a standalone Docker container with no reverse dependencies. To enable it, add the service block to `docker-compose.yml`. To disable it, remove or comment out the block. No other container references the BBS server; no database migration is required; no configuration flag in any other service mentions it.

```yaml
# Optional: Telnet BBS server
bbs-server:
  build:
    context: ./bbs-server
    dockerfile: Dockerfile
  container_name: ${COMPOSE_PROJECT_NAME:-bastion}-bbs
  ports:
    - "${BBS_TELNET_PORT:-2323}:2323"
  environment:
    - BACKEND_URL=http://backend:8000
    - INTERNAL_SERVICE_KEY=${INTERNAL_SERVICE_KEY}
    - BBS_NAME=${BBS_NAME:-Bastion BBS}
    - BBS_MAX_CONNECTIONS=${BBS_MAX_CONNECTIONS:-20}
    - BBS_IDLE_TIMEOUT=${BBS_IDLE_TIMEOUT:-600}
  depends_on:
    backend:
      condition: service_healthy
  networks:
    - bastion-net
  restart: unless-stopped
  logging:
    driver: "journald"
    options:
      tag: "bastion-bbs"
```

**No changes required in:**
- `backend/` — no new endpoints, no new config, no new dependencies
- `frontend/` — no UI changes
- `connections-service/` — no new providers
- `llm-orchestrator/` — no new routes or tools
- Database schema — no migrations

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    BBS SERVER CONTAINER                    │
│                  (Python asyncio service)                  │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Telnet Listener (port 2323)                        │  │
│  │  • asyncio.start_server or telnetlib3               │  │
│  │  • NAWS negotiation (detect terminal size)          │  │
│  │  • Echo suppression for password entry              │  │
│  │  • One task per connection                          │  │
│  └────────────────────┬────────────────────────────────┘  │
│                       ▼                                    │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Session Manager                                    │  │
│  │  • Per-connection state: user_id, JWT, term size    │  │
│  │  • Active conversation_id tracking                  │  │
│  │  • Idle timeout enforcement                         │  │
│  │  • Max concurrent connections cap                   │  │
│  └────────────────────┬────────────────────────────────┘  │
│                       ▼                                    │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Menu System                                        │  │
│  │  • Login screen → Main menu → Sub-areas             │  │
│  │  • Slash command router (same as Telegram/Discord)  │  │
│  │  • ANSI rendering, box drawing, color themes        │  │
│  │  • Paginated output with --More-- prompts           │  │
│  │  • Markdown-to-ANSI conversion                      │  │
│  └────────────────────┬────────────────────────────────┘  │
│                       ▼                                    │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Backend Client (HTTP)                              │  │
│  │  • Pattern: identical to connections-service        │  │
│  │    BackendClient (httpx async calls)                │  │
│  │  • Auth: X-Internal-Service-Key for internal API    │  │
│  │  • User JWT for user-facing API endpoints           │  │
│  │  • Calls existing endpoints — no new ones needed    │  │
│  └─────────────────────────────────────────────────────┘  │
└──────────────────────────┬────────────────────────────────┘
                           │
                           │  HTTP over Docker internal network
                           ▼
                ┌──────────────────────┐
                │   Backend (FastAPI)   │  ← Unchanged
                │   Port 8000           │
                └──────────┬───────────┘
                           │  gRPC
                           ▼
                ┌──────────────────────┐
                │  LLM Orchestrator    │  ← Unchanged
                │  Port 50051          │
                └──────────────────────┘
```

### Communication Flow (Chat Example)

```
DOS User types message in telnet client
  → TCP bytes arrive at BBS server port 2323
  → Session reads line, identifies as regular chat message (not a / command)
  → Backend client POSTs to /api/internal/external-chat
    {user_id, conversation_id, query, platform: "bbs", ...}
  → Backend stores user message, calls orchestrator via gRPC StreamChat
  → Orchestrator runs agent pipeline, streams response
  → Backend accumulates response, stores assistant message, returns JSON
  → BBS server receives {response, images, conversation_id}
  → ANSI renderer strips markdown, word-wraps at terminal width
  → Paginator displays with --More-- if response exceeds screen height
  → User sees response, types next message
```

### Communication Flow (Slash Commands)

Slash commands are handled locally in the BBS server, identical to how `channel_listener_manager.py` handles them for Telegram/Discord:

| Command | BBS Server Action | Backend API Called |
|---------|-------------------|-------------------|
| `/newchat` | Generate new conversation_id, update session state | `POST /api/internal/external-chat` (start_new_conversation=True) |
| `/chats` | List recent conversations | `GET /api/internal/user-conversations` |
| `/loadchat N` | Switch active conversation | `GET /api/internal/validate-conversation` |
| `/chat` | Show current conversation info | `GET /api/internal/validate-conversation` |
| `/model` | List available models | `POST /api/internal/external-chat-models` |
| `/model N` | Set model for conversation | `POST /api/internal/external-chat-set-model` |

These are the exact same internal API endpoints the connections-service bots already call. The BBS server is just another client.

---

## BBS Areas → Bastion Service Mapping

### Area 1: Chat with AI (Main Board)

The primary interaction area. Directly reuses the external-chat internal API.

**Backend endpoints used:**
- `POST /api/internal/external-chat` — send message, receive response
- `POST /api/internal/external-chat-models` — list models
- `POST /api/internal/external-chat-set-model` — set model
- `GET /api/internal/user-conversations` — list conversations
- `GET /api/internal/validate-conversation` — validate/switch conversation
- `POST /api/internal/connection-active-conversation` — persist selection

**Platform identifier:** `"bbs"` (passed in the `platform` field, alongside existing `"telegram"` and `"discord"`)

**Image handling:** Images in responses are not displayable in a text terminal. The BBS renderer should:
- Strip `![alt](url)` markdown image references from response text
- Optionally show `[Image: {alt_text}]` placeholder
- Structured `images` from metadata are silently dropped

### Area 2: Message Boards (Conversation Browser)

Browse and read conversation history. Uses existing user-facing API with JWT auth.

**Backend endpoints used:**
- `GET /api/internal/user-conversations` — list conversations with title, message count, last updated
- Conversation message history — read via the orchestrator context (or a direct DB query endpoint if added later)

**Display format:**
```
═══ Conversations ══════════════════════════
 #  │ Title                     │ Msgs │ When
────┼───────────────────────────┼──────┼──────
  1 │ Quantum computing rese... │   14 │  2h ago
  2 │ Code review: auth service │    8 │  1d ago
  3 │ Weekly planning notes     │   22 │  3d ago

[R]ead #  [N]ew chat  [B]ack
```

### Area 3: File Library (Document Browser)

Browse user documents and folders, read document content in the terminal.

**Backend endpoints used (with user JWT):**
- `GET /api/user/documents` — list documents
- `GET /api/documents/{doc_id}/content` — read document content
- `POST /api/user/documents/search` — search documents

**Display format:**
```
═══ File Library ═══════════════════════════
Folder: /Research    (12 files)

 #  │ Filename                │ Type │  Size
────┼─────────────────────────┼──────┼──────
  1 │ meeting-notes.md        │  MD  │  4KB
  2 │ architecture-plan.md    │  MD  │ 12KB
  3 │ quarterly-report.pdf    │ PDF  │ 89KB

[V]iew #  [S]earch  [U]p dir  [B]ack
```

**Content display:** Markdown files render with ANSI formatting (bold, headers as bright text). PDF and binary files show a "not displayable in terminal" message with metadata.

### Area 4: News Reader (RSS)

Browse RSS feeds and read articles in the terminal.

**Backend endpoints used (with user JWT):**
- `GET /api/rss/feeds` — list subscribed feeds
- `GET /api/rss/feeds/{feed_id}/articles` — list articles in a feed
- `GET /api/rss/unread-count` — unread badge count
- `PUT /api/rss/articles/{article_id}/read` — mark as read
- `PUT /api/rss/articles/{article_id}/star` — star/unstar

**Display format:**
```
═══ News Reader ════════════════════════════
Feeds (5 unread total):

 #  │ Feed                     │ Unread
────┼──────────────────────────┼────────
  1 │ Hacker News              │     3
  2 │ Ars Technica             │     2
  3 │ The Verge                │     0
  4 │ Wired                    │     0

[R]ead #  [B]ack
```

Article view renders HTML content as plain text with line wrapping. Images become `[Image]` placeholders.

### Area 5: Database Explorer (Data Workspace)

Browse data workspaces, view tables, run SQL queries — displayed as ASCII tables.

**Backend endpoints used (with user JWT):**
- `GET /api/data/workspaces` — list workspaces
- `GET /api/data/workspaces/{id}/databases` — list databases
- `GET /api/data/databases/{id}/tables` — list tables
- `GET /api/data/tables/{id}/data` — view table data
- `POST /api/data/workspaces/{id}/query/sql` — run SQL query

**Display format:**
```
═══ Database Explorer ══════════════════════
Workspace: Sales Analytics

Tables:
  1 │ customers    │ 1,247 rows
  2 │ orders       │ 8,903 rows
  3 │ products     │   156 rows

[V]iew #  [Q]uery SQL  [B]ack

SQL> SELECT name, total FROM customers
     ORDER BY total DESC LIMIT 5

 name          │ total
───────────────┼──────────
 Acme Corp     │ $45,200
 Widget Inc    │ $38,100
 ...
(5 rows)
```

### Area 6: Settings

Model selection and user profile, per the existing APIs.

**Backend endpoints used:**
- `POST /api/internal/external-chat-models` — list models
- `POST /api/internal/external-chat-set-model` — set model
- `GET /api/auth/me` — current user info

### Area 7: SysOp Panel (Admin Only)

Visible only to users with `role: "admin"`. Exposes user management.

**Backend endpoints used (with admin JWT):**
- `GET /api/auth/users` — list users
- `POST /api/auth/users` — create user
- `PUT /api/auth/users/{id}` — update user
- `DELETE /api/auth/users/{id}` — delete user

---

## Authentication Strategy

The BBS server uses a **dual-auth approach:**

### 1. Internal Service Key (for internal API)

Same as the connections-service: the BBS container has `INTERNAL_SERVICE_KEY` in its environment. Used for `X-Internal-Service-Key` header on `/api/internal/*` endpoints. The `user_id` is passed in the request body to scope operations to the logged-in user.

### 2. User JWT (for user-facing API)

On login, the BBS server calls `POST /api/auth/login` with the user's credentials. The returned JWT is stored in the session and used as `Authorization: Bearer {token}` for user-facing endpoints (`/api/user/documents`, `/api/rss/*`, `/api/data/*`, etc.).

### Login Flow

```
1. User connects via telnet
2. BBS displays welcome screen with ANSI art
3. Prompts for username (with echo)
4. Prompts for password (echo suppressed via telnet IAC WILL ECHO)
5. BBS server POSTs to /api/auth/login {username, password}
6. On success: store JWT + user_id in session, show main menu
7. On failure: show error, allow retry (max 3 attempts then disconnect)
8. Account lockout is enforced by auth_service (MAX_FAILED_LOGINS)
```

### Session Lifecycle

- JWT stored in-memory in the session object (never written to disk)
- Token refresh: if `JWT_EXPIRATION_MINUTES > 0`, the session periodically calls `POST /api/auth/refresh` to renew the token before expiry
- Idle timeout: configurable via `BBS_IDLE_TIMEOUT` env var (default 600s); warns user at 60s before disconnect
- Explicit logout: user selects Goodbye from main menu; session ends, connection closed

---

## File Structure

```
bbs-server/
├── Dockerfile                  # Slim Python image, pip install deps
├── requirements.txt            # httpx, telnetlib3 (or asyncio only)
├── config.py                   # Environment config (BACKEND_URL, etc.)
├── main.py                     # Entry point: start telnet listener
├── server.py                   # Telnet server, connection handler
├── session.py                  # Per-connection session state
├── backend_client.py           # HTTP client (mirrors connections-service pattern)
├── menu_system.py              # Menu framework: main menu, sub-menus
├── areas/
│   ├── __init__.py
│   ├── chat_area.py            # AI chat (slash commands, message loop)
│   ├── conversations_area.py   # Conversation browser
│   ├── documents_area.py       # Document/file browser
│   ├── rss_area.py             # RSS feed reader
│   ├── data_area.py            # Data workspace explorer
│   ├── settings_area.py        # Model selection, profile
│   └── admin_area.py           # SysOp admin panel
├── rendering/
│   ├── __init__.py
│   ├── ansi.py                 # ANSI escape codes, color themes
│   ├── boxes.py                # Box drawing (╔═╗║╚╝ frames)
│   ├── tables.py               # ASCII table renderer
│   ├── markdown_strip.py       # Markdown → ANSI text conversion
│   ├── paginator.py            # --More-- pagination
│   └── wordwrap.py             # Word-wrap at terminal width
└── assets/
    └── welcome.ans             # ANSI art for welcome screen (optional)
```

Every file stays well under the 500-line limit. The `areas/` modules each handle one BBS section (~80-120 lines each). The `rendering/` modules are pure-function utilities with no backend dependencies.

---

## Rendering: Markdown → ANSI

Agent responses come back as Markdown. The BBS must convert to ANSI terminal formatting:

| Markdown | ANSI Rendering |
|----------|---------------|
| `**bold**` | `\033[1m` (bright/bold) |
| `*italic*` | `\033[3m` (italic, if terminal supports) or unchanged |
| `# Heading` | `\033[1;33m` (bold yellow) + uppercase |
| `## Heading` | `\033[1;36m` (bold cyan) |
| `` `code` `` | `\033[32m` (green) |
| ```` ```code block``` ```` | Green text, indented 2 spaces |
| `![alt](url)` | `[Image: alt]` or stripped entirely |
| `[text](url)` | `text (url)` — show URL inline |
| `- list item` | `  • list item` |
| `1. item` | `  1. item` |
| `> blockquote` | `  │ quote text` (with box-drawing pipe) |

### Color Themes

Configurable via environment variable `BBS_THEME`:

- **green** (default): Green text on black, bright green for emphasis — classic hacker aesthetic
- **amber**: Amber/yellow text on black — vintage CRT look
- **blue**: Blue/cyan text on black — corporate BBS style
- **none**: No ANSI colors (for dumb terminals or accessibility)

---

## Telnet Protocol Considerations

### Terminal Size Detection (NAWS)

Negotiate NAWS (RFC 1073) on connection to detect terminal dimensions. Default to 80x24 if negotiation fails. Use detected dimensions for word-wrap width and pagination line count.

### Echo Control

During password entry, send `IAC WILL ECHO` to suppress client-side echo, then `IAC WONT ECHO` after. This prevents the password from appearing on screen.

### Character Encoding

Assume CP437 for maximum DOS compatibility. UTF-8 box-drawing characters (`╔═╗║╚╝`) have CP437 equivalents (`╔═╗║╚╝` are in the CP437 range 179-218). For terminals that don't support box drawing, fall back to `+-|` ASCII art.

### Line Endings

Send `\r\n` (CR+LF) for all output lines — required by the telnet spec and expected by DOS telnet clients.

---

## Concurrency Model

- `asyncio.start_server` handles TCP connections
- Each connection spawns one `asyncio.Task` running the session loop
- All backend HTTP calls use `httpx.AsyncClient` (non-blocking)
- `BBS_MAX_CONNECTIONS` caps concurrent sessions (default 20)
- No shared mutable state between sessions — each session has its own `BBSSession` instance with its own JWT, user_id, conversation_id, and menu state

---

## Dependencies

Minimal — the BBS server is a thin client:

```
# bbs-server/requirements.txt
httpx>=0.27.0
telnetlib3>=2.0.0       # Telnet protocol negotiation (NAWS, echo control)
```

Optional:
```
html2text>=2024.2.26    # For RSS article HTML → plain text conversion
```

No LangGraph, no gRPC, no database drivers, no Redis. The BBS server talks only HTTP to the backend.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_URL` | `http://backend:8000` | Backend API base URL (Docker service name) |
| `INTERNAL_SERVICE_KEY` | (required) | Shared key for `/api/internal/*` endpoints |
| `BBS_TELNET_PORT` | `2323` | Telnet listen port |
| `BBS_NAME` | `Bastion BBS` | BBS name shown in welcome screen |
| `BBS_MAX_CONNECTIONS` | `20` | Maximum simultaneous telnet sessions |
| `BBS_IDLE_TIMEOUT` | `600` | Seconds of inactivity before disconnect |
| `BBS_THEME` | `green` | Color theme: green, amber, blue, none |
| `BBS_MOTD` | (empty) | Message of the day, shown after login |

---

## Interaction with Existing Services

### What the BBS Server Calls

| Service | Protocol | Endpoints Used |
|---------|----------|---------------|
| Backend (FastAPI) | HTTP | `/api/auth/login`, `/api/auth/me`, `/api/auth/refresh` |
| Backend (internal) | HTTP | `/api/internal/external-chat`, `/api/internal/external-chat-models`, `/api/internal/external-chat-set-model`, `/api/internal/user-conversations`, `/api/internal/validate-conversation`, `/api/internal/connection-active-conversation` |
| Backend (user API) | HTTP | `/api/user/documents`, `/api/documents/{id}/content`, `/api/rss/*`, `/api/data/*` |

### What Calls the BBS Server

Nothing. No other service references or depends on the BBS server. It is a leaf node in the dependency graph.

### Database Impact

None. The BBS server has no database of its own. All state is in-memory (session objects) or delegated to the backend (conversations, messages, documents, etc.). Conversations created via BBS are stored by the backend in the same `conversations` and `conversation_messages` tables as any other source — tagged with `source_platform: "bbs"`.

---

## DOS Client Connectivity

### Real MS-DOS Machines

For actual MS-DOS 5.0+ machines to connect:

1. **NIC + packet driver** — install the appropriate packet driver for the network card
2. **mTCP stack** — install mTCP (actively maintained DOS TCP/IP stack, http://www.brutman.com/mTCP/)
3. **mTCP telnet client** — `TELNET bastion.local 2323`
4. **Alternative: serial bridge** — DOS terminal emulator (Kermit, Telix, QBASIC) over COM port → modern machine running `socat` or similar bridging to TCP

### DOSBox / 86Box / RetroPC Emulators

For emulated DOS environments, most emulators support virtual networking. DOSBox-X has built-in NE2000 emulation that works with mTCP's packet driver.

### Modern Clients

Any telnet client works: PuTTY, Windows `telnet`, macOS/Linux `telnet` or `nc`, Terminal.app, iTerm2, etc. This makes the BBS useful as a lightweight CLI interface even for users who don't care about the DOS aesthetic.

---

## Security Considerations

- **No TLS on telnet** — telnet is inherently unencrypted. For production use over untrusted networks, either:
  - Add SSH support (via `asyncssh` library) on a separate port as an alternative
  - Run behind a TLS-terminating proxy (e.g., `stunnel`)
  - Accept the risk for LAN-only deployments
- **Password transmitted in cleartext** — same as any telnet login. SSH alternative addresses this.
- **Internal service key** — same security model as connections-service; the key never leaves the Docker network
- **JWT storage** — in-memory only; never logged, never written to disk
- **Account lockout** — enforced by the backend's `auth_service` (MAX_FAILED_LOGINS), not by the BBS server
- **Idle timeout** — prevents abandoned sessions from holding resources
- **Connection limit** — `BBS_MAX_CONNECTIONS` prevents resource exhaustion

---

## Future Enhancements (Not In Initial Scope)

- **SSH support** — add `asyncssh` listener on port 2222 for encrypted access
- **ANSI art gallery** — load `.ANS` files for welcome screens, menu headers, area transitions
- **Door games** — launch specific Agent Factory agent profiles as interactive "doors" (e.g., a trivia agent, a storytelling agent)
- **File upload/download** — ZMODEM or XMODEM protocol support for document transfer
- **Who's online** — show other connected BBS users (requires minimal shared state)
- **Inter-user messaging** — BBS-to-BBS chat between connected users
- **Bulletin boards** — read/write message areas backed by Bastion documents (shared folders)
- **SysOp chat** — admin can "page" into a user's session for real-time assistance

---

## Implementation Order

1. **Phase 1: Server + Auth + Chat** — Telnet listener, login flow, main menu, AI chat area with slash commands. This alone is a functional BBS.
2. **Phase 2: Conversations + Documents** — Conversation browser, document file library with content viewer.
3. **Phase 3: RSS + Data** — News reader area, data workspace explorer with SQL query support.
4. **Phase 4: Polish** — ANSI art, color themes, admin panel, idle timeout warnings, terminal size detection.

Each phase is independently deployable. Phase 1 delivers a working BBS in ~500 lines across 5-6 files.

---

## Estimated Effort

| Component | Files | Lines | Notes |
|-----------|-------|-------|-------|
| Server core + config | 3 | ~250 | main.py, server.py, config.py |
| Session + auth flow | 1 | ~150 | session.py |
| Backend HTTP client | 1 | ~200 | backend_client.py (mirrors connections-service pattern) |
| Menu framework | 1 | ~120 | menu_system.py |
| Chat area | 1 | ~120 | areas/chat_area.py |
| Conversation browser | 1 | ~100 | areas/conversations_area.py |
| Document browser | 1 | ~120 | areas/documents_area.py |
| RSS reader | 1 | ~100 | areas/rss_area.py |
| Data explorer | 1 | ~120 | areas/data_area.py |
| Settings + admin | 2 | ~120 | areas/settings_area.py, areas/admin_area.py |
| ANSI rendering suite | 6 | ~300 | rendering/*.py |
| Dockerfile + requirements | 2 | ~30 | Container setup |
| **Total** | **~21** | **~1,730** | No file exceeds 250 lines |

---

## Key Design Decision: Internal API vs User API

The BBS server uses **both** API tiers:

- **Internal API** (`/api/internal/*` with `X-Internal-Service-Key`) for chat, model selection, and conversation management — this is the same path Telegram/Discord bots use, and it handles the full orchestrator cycle (store message → call LLM → store response → return)
- **User API** (`/api/*` with `Authorization: Bearer {jwt}`) for documents, RSS, data workspaces, and settings — these endpoints already exist for the web frontend and require user-level auth

This hybrid approach means the BBS server needs no new backend endpoints. Every API it calls already exists and is battle-tested by either the web frontend or the connections-service bots.
