# Local Proxy Daemon — Architecture & Implementation Plan

## Overview

A lightweight daemon that runs on the user's local machine and exposes local capabilities (shell, filesystem, screenshots, clipboard, desktop notifications, etc.) as tools that Bastion agents can invoke. The daemon connects **outbound** to Bastion over a persistent WebSocket, registers its capabilities, receives tool invocations, executes them locally, and returns results — giving agents controlled access to the user's physical workstation.

**Core principles:**
- The daemon connects outbound to Bastion — works behind NAT with zero port forwarding
- Agents invoke local proxy tools identically to any other tool (same Action I/O Registry pattern)
- A three-layer security model ensures agents cannot exceed their authorized scope
- The daemon is stateless and restartable — all state lives in Bastion
- Capability set is configurable per-device via a local policy file

---

## Architecture Overview

```
┌──────────────────────────┐            ┌──────────────────────┐
│   LLM Orchestrator       │            │  User's Machine      │
│                          │            │                      │
│  Agent calls tool:       │            │  Local Proxy Daemon  │
│  local_shell_execute()   │            │  (bastion-proxy)     │
│         │                │            │         │            │
│         ▼                │            │         ▼            │
│  backend_tool_client ────┼── gRPC ──▶ │  Backend (FastAPI)   │
│                          │            │         │            │
│                          │            │         ▼            │
│                          │            │  WebSocket dispatch  │
│                          │            │  to connected device │
│                          │            │         │            │
│                          │            │         ▼            │
│  ◀── result ─────────────┼────────────┼  Daemon executes     │
│                          │            │  locally, returns    │
│  Agent sees result,      │            │  result over WS      │
│  continues reasoning     │            │                      │
└──────────────────────────┘            └──────────────────────┘
```

### Invocation Chain

```
Agent LLM generates tool_call
  → orchestrator tool wrapper (Zone 1 thin wrapper)
  → backend_tool_client gRPC call (Zone 2 bridge)
  → backend grpc_tool_service.InvokeDeviceTool handler
  → WebSocket dispatch to connected device via websocket_manager
  → daemon receives invocation, executes locally
  → daemon sends result back over WebSocket
  → backend resolves pending Future, returns gRPC response
  → orchestrator receives result as ToolMessage
  → agent continues reasoning
```

---

## Zone Placement: Zone 5 — Local Device

This introduces a new tool placement zone:

| Zone | Description | Location |
|------|-------------|----------|
| 1 | Orchestrator (in-memory) | `orchestrator/tools/` |
| 2 | Tools Service (Bastion infrastructure) | `backend/services/grpc_tool_service.py` |
| 3 | Connections Service (third-party APIs) | `connections-service/providers/` |
| 4 | Plugins (SaaS integrations) | `orchestrator/plugins/integrations/` |
| **5** | **Local Proxy (user's machine)** | **Daemon on user machine + backend bridge** |

**Decision criteria:** If the tool requires execution on the user's physical machine (local filesystem, local processes, display, audio, peripherals), it belongs in Zone 5.

---

## Component Design

### 1. Local Proxy Daemon (standalone package)

Runs on the user's machine. Distributed as `pip install bastion-local-proxy` with a `bastion-proxy` CLI entry point.

**Responsibilities:**
- Maintain persistent WebSocket connection to Bastion backend
- Register device capabilities on connect
- Receive tool invocations, execute locally, return results
- Enforce local security policy (allowlists, deny patterns, confirmation dialogs)
- Automatic reconnection with exponential backoff

**Structure:**

```
bastion-local-proxy/
├── bastion_proxy/
│   ├── __init__.py
│   ├── cli.py                    # Entry point: bastion-proxy start
│   ├── daemon.py                 # Main daemon loop, WebSocket client
│   ├── config.py                 # Load ~/.bastion/local-proxy.yml
│   ├── policy.py                 # Security policy enforcement
│   ├── capabilities/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseCapability interface
│   │   ├── shell.py              # Shell command execution
│   │   ├── filesystem.py         # File read/write/list
│   │   ├── screenshot.py         # Screen capture
│   │   ├── clipboard.py          # Clipboard read/write
│   │   ├── system_info.py        # CPU, RAM, disk, OS, network
│   │   ├── desktop_notify.py     # OS desktop notifications
│   │   ├── processes.py          # List/manage processes
│   │   ├── browser.py            # Open URLs
│   │   └── audio.py              # Audio playback / TTS
│   └── protocol.py               # WebSocket message format
├── setup.py / pyproject.toml
└── README.md
```

**Startup flow:**

```
bastion-proxy start --url https://bastion.example.com --token <device_token>
  1. Load config from ~/.bastion/local-proxy.yml
  2. Discover enabled capabilities from config
  3. Connect WebSocket to /api/ws/device?token=<device_token>
  4. Send register message with device_id + capability list
  5. Enter listen loop: receive invocations, execute, respond
```

**Capability interface:**

```python
class BaseCapability:
    name: str
    description: str
    risk_level: str  # "low", "medium", "high"
    
    async def execute(self, args: dict) -> dict:
        """Execute the capability. Returns {result: ..., formatted: ...}"""
        raise NotImplementedError
    
    def validate_args(self, args: dict) -> bool:
        """Validate args against local policy before execution."""
        raise NotImplementedError
```

### 2. Backend WebSocket Device Endpoint

New file: `backend/api/device_proxy_api.py`

**Responsibilities:**
- Accept persistent WebSocket connections from local proxy daemons
- Authenticate via device token (linked to a user)
- Register device in WebSocket manager's device registry
- Route tool invocations from gRPC to the correct device WebSocket
- Collect responses and resolve pending Futures

**Key endpoint:**

```
WS /api/ws/device?token=<device_token>
```

**Protocol messages (daemon → backend):**

```json
{"type": "register", "device_id": "adams-workstation", "capabilities": ["shell_execute", "read_file", ...]}
{"type": "result", "request_id": "uuid", "result": {"stdout": "...", "exit_code": 0}, "formatted": "..."}
{"type": "error", "request_id": "uuid", "error": "Permission denied by local policy"}
{"type": "heartbeat"}
```

**Protocol messages (backend → daemon):**

```json
{"type": "invoke", "request_id": "uuid", "tool": "shell_execute", "args": {"command": "ls -la"}}
{"type": "ping"}
```

### 3. WebSocket Manager Extension

Modify: `backend/utils/websocket_manager.py`

Add device connection tracking and invocation dispatch:

```python
# New data structures
device_connections: Dict[str, Dict[str, WebSocket]]   # user_id -> {device_id -> ws}
device_capabilities: Dict[str, Dict[str, List[str]]]  # user_id -> {device_id -> [capabilities]}
pending_invocations: Dict[str, asyncio.Future]         # request_id -> Future

# New methods
async def register_device(user_id, device_id, websocket, capabilities)
async def unregister_device(user_id, device_id)
async def invoke_device_tool(user_id, device_id, tool, args, timeout=30) -> dict
async def resolve_device_invocation(request_id, result)
async def get_user_devices(user_id) -> List[dict]
```

### 4. gRPC Tool Service Handler

Modify: `backend/services/grpc_tool_service.py`

New RPC: `InvokeDeviceTool`

```protobuf
message InvokeDeviceToolRequest {
    string user_id = 1;
    string device_id = 2;    // empty = default/only device
    string tool = 3;
    string args_json = 4;
    int32 timeout_seconds = 5;
}

message InvokeDeviceToolResponse {
    bool success = 1;
    string result_json = 2;
    string error = 3;
    string formatted = 4;
}
```

Handler dispatches to `websocket_manager.invoke_device_tool()` and awaits the Future.

### 5. Orchestrator Tool Wrappers

New file: `llm-orchestrator/orchestrator/tools/local_proxy_tools.py`

Thin gRPC wrappers registered in the Action I/O Registry:

```python
# Tools exposed to agents:
local_shell_execute(command: str) -> dict
local_read_file(path: str) -> dict
local_write_file(path: str, content: str) -> dict
local_list_directory(path: str) -> dict
local_screenshot() -> dict
local_clipboard_read() -> dict
local_clipboard_write(content: str) -> dict
local_system_info() -> dict
local_desktop_notify(title: str, message: str) -> dict
local_list_processes() -> dict
local_open_url(url: str) -> dict
```

Each follows the standard unified return format with `formatted` field.

---

## Capability Catalog

| Capability | Description | Risk | Default |
|-----------|-------------|------|---------|
| `shell_execute` | Run shell commands | **High** | Off |
| `read_file` | Read files from local filesystem | Medium | On (restricted paths) |
| `write_file` | Write/create files | Medium | Off |
| `list_directory` | List directory contents | Low | On (restricted paths) |
| `screenshot` | Capture screen or active window | Medium | Off |
| `clipboard_read` | Read clipboard contents | Low | On |
| `clipboard_write` | Write to clipboard | Low | On |
| `open_url` | Open URL in default browser | Low | On |
| `open_application` | Launch an application | Medium | Off |
| `system_info` | CPU, RAM, disk, OS, network info | Low | On |
| `desktop_notify` | Show OS desktop notification | Low | On |
| `list_processes` | List running processes | Low | On |
| `audio_play` | Play audio file or TTS | Low | Off |
| `keyboard_type` | Type text (automation) | **High** | Off |
| `mouse_click` | Click at screen coordinates | **High** | Off |

---

## Security Model

Three independent layers, each enforced at a different point.

### Layer 1: Local Policy (daemon-side, user controls)

The daemon reads `~/.bastion/local-proxy.yml` at startup. The user configures exactly what's allowed:

```yaml
device_id: "adams-workstation"
bastion_url: "https://bastion.example.com"

capabilities:
  shell_execute:
    enabled: true
    mode: allowlist                 # allowlist | denylist | ask | unrestricted
    allowed_commands: ["ls", "cat", "grep", "find", "df", "free", "uname", "ps"]
    denied_patterns: ["rm -rf", "sudo", "mkfs", "dd if=", "chmod 777"]
    max_output_bytes: 1048576       # 1MB output cap
  
  read_file:
    enabled: true
    allowed_paths:
      - "/home/adam/Documents"
      - "/home/adam/Projects"
    denied_paths:
      - "/home/adam/.ssh"
      - "/home/adam/.gnupg"
      - "/home/adam/.config/bastion"
    max_file_bytes: 10485760        # 10MB file cap
  
  write_file:
    enabled: true
    allowed_paths:
      - "/home/adam/Documents/bastion-output"
  
  screenshot:
    enabled: true
    mode: ask                       # Always confirm with desktop dialog
  
  clipboard_read:
    enabled: true
  
  system_info:
    enabled: true

# Global settings
confirmation_mode: high_risk_only   # never | high_risk_only | always
confirmation_timeout_seconds: 30
max_concurrent_invocations: 3
```

### Layer 2: Desktop Confirmation (HITL on the machine)

For capabilities with `mode: ask` or when `confirmation_mode` triggers, the daemon shows a native desktop dialog before executing:

```
┌─────────────────────────────────────────┐
│  Bastion Agent Request                  │
│                                         │
│  Agent "research" wants to execute:     │
│  shell_execute: grep -r "TODO" ~/Proj   │
│                                         │
│  [Allow]  [Deny]  [Allow this session]  │
└─────────────────────────────────────────┘
```

Implementation options: `tkinter` (built-in), `zenity` (Linux), `osascript` (macOS), `ctypes MessageBox` (Windows).

### Layer 3: Agent-Side Tool Access (orchestrator)

The orchestrator's tool registry and skill definitions control which agents/skills can access local proxy tools. A research agent may get `read_file` and `system_info` but not `shell_execute`. An automation agent may get the full set.

This is enforced by existing patterns: skills declare their `tools` list, and the automation engine only resolves those tools.

---

## Connection Protocol

### WebSocket (Recommended for v1)

- Daemon connects outbound: `wss://bastion.example.com/api/ws/device?token=<token>`
- Works through any NAT, firewall, or corporate proxy
- Bidirectional: backend pushes invocations, daemon pushes results
- Heartbeat every 30s to detect stale connections
- Auto-reconnect with exponential backoff (1s → 2s → 4s → ... → 60s max)

### Future: gRPC Bidirectional Streaming

If typed contracts become important, upgrade to a gRPC bidirectional stream:
- Daemon opens outbound gRPC channel to a new `DeviceProxyService`
- Opens a `DeviceStream` bidirectional RPC
- Backend sends `DeviceInvocation` messages, daemon sends `DeviceResult` messages
- Same outbound-connection-through-NAT pattern, but with proto typing

---

## Device Token Management

Devices authenticate with a device token, not user JWT (JWTs expire too frequently for a long-lived daemon).

**Token lifecycle:**
1. User generates a device token in Settings UI: `POST /api/settings/device-tokens` → returns `{token, device_id, created_at}`
2. Token is stored in `device_tokens` table (linked to `user_id`)
3. User puts token in `~/.bastion/local-proxy.yml` or passes via CLI
4. Backend validates token on WebSocket connect, resolves to `user_id`
5. User can revoke tokens in Settings UI: `DELETE /api/settings/device-tokens/{token_id}`

**Database:**

```sql
CREATE TABLE device_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    token_hash TEXT NOT NULL,          -- bcrypt hash of the token
    device_name TEXT NOT NULL,         -- user-friendly name
    last_connected_at TIMESTAMPTZ,
    last_ip TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    revoked_at TIMESTAMPTZ
);
```

---

## Implementation Phases

### Phase 1: Foundation (MVP)

**Goal:** Agent can read files and get system info from user's machine.

| Component | Work | Estimated Lines |
|-----------|------|----------------|
| Daemon core (WebSocket client, config, CLI) | New standalone | ~300 |
| Capabilities: `read_file`, `list_directory`, `system_info` | New in daemon | ~150 |
| Policy engine (allowlist paths) | New in daemon | ~100 |
| Backend `/api/ws/device` endpoint | New `device_proxy_api.py` | ~100 |
| WebSocket manager device registry | Modify existing | ~100 |
| gRPC `InvokeDeviceTool` handler | Modify `grpc_tool_service.py` | ~50 |
| Proto additions | Modify `tool_service.proto` | ~30 |
| Orchestrator tool wrappers (3 tools) | New `local_proxy_tools.py` | ~100 |
| Device token endpoint | New or extend `settings_api.py` | ~60 |

**Total Phase 1:** ~990 lines across ~5 new files + ~4 modified files

### Phase 2: Shell & Clipboard

Add `shell_execute` (with allowlist), `clipboard_read`, `clipboard_write`, `desktop_notify`.

- Desktop confirmation dialog for high-risk operations
- Shell command allowlist/denylist enforcement
- ~300 additional lines in daemon capabilities + ~100 in orchestrator wrappers

### Phase 3: Screen & Automation

Add `screenshot`, `open_url`, `open_application`, `list_processes`.

- Screenshot compression and transmission (base64 or upload to Bastion storage)
- Process management capabilities
- ~300 additional lines

### Phase 4: Advanced Automation

Add `keyboard_type`, `mouse_click`, window management.

- Full desktop automation capabilities
- Requires robust confirmation mode
- ~400 additional lines

### Phase 5: Multi-Device & UI

- Settings UI for device management (list devices, capabilities, revoke tokens)
- Multi-device support (agent specifies which device, or broadcasts)
- Device status dashboard in Bastion frontend
- ~500 lines frontend + ~200 backend

---

## Open Questions

1. **Screenshot transport:** Base64 in WebSocket message, or upload to Bastion file storage and return a URL? Base64 is simpler but large screenshots could be 2-5MB.

2. **Streaming output:** For long-running shell commands, should the daemon stream stdout/stderr incrementally? Adds complexity but prevents timeouts on `find / -name "*.log"`.

3. **Multi-device:** If user has a workstation and a laptop both running the daemon, how does the agent pick which one? Default device? User specifies in chat?

4. **File transfer:** Should the daemon support uploading local files to Bastion document storage? This makes `read_file` → `create_document` a single operation for the agent.

5. **Daemon auto-update:** Should the daemon check for version compatibility with the Bastion instance and auto-update?

6. **Windows/macOS support:** Phase 1 can target Linux only (matching the server OS). How early should cross-platform support land?

7. **Agent awareness:** Should agents know the device OS and tailor commands accordingly (e.g., `ls` vs `dir`)? System info could provide this context automatically.
