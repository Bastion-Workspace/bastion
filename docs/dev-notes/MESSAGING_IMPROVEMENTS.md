# Inter-User Messaging: Audit and Improvement Plan

Assessment of the room-based inter-user messaging system (`chat_rooms` / `chat_messages`), gaps relative to best-of-breed products, and a roadmap for agent-in-chat integration.

## Current State

The room-based messaging system is a functional v1 with solid fundamentals.

### What works well

| Feature | Implementation | Notes |
|---------|---------------|-------|
| Room creation (DM + group) | `messaging_api.py`, `messaging_service.py` | `room_type_enum`: `direct`, `group` |
| Real-time delivery | Dual WebSocket paths: per-room + per-user | `websocket_manager.py` `connect_to_room` / `broadcast_to_room` |
| Presence | DB (`user_presence`), REST, WS fanout | Online / offline / away; `PresenceIndicator` component |
| File attachments | `messaging_attachment_service.py` | Images, audio, general files; MIME allowlist, size limits |
| Unread counts | `room_participants.last_read_at` | Per-user per-room read cursor |
| At-rest encryption | `encryption_service.py` | Optional Fernet; `MESSAGE_ENCRYPTION_AT_REST` flag |
| Soft delete | `chat_messages.deleted_at` | Sender-only soft delete |
| Mute / notification settings | `room_participants.notification_settings` JSONB | Per-room mute |
| Team-linked rooms | `chat_rooms.team_id` FK → `teams` | Rooms associated with agent teams |
| Agent watch bridge | `agent_conversation_watches` + `dispatch_conversation_reaction` | Celery task triggers agent on new room message |

### Architecture overview

```
Frontend                    Backend                         Orchestrator
─────────────────────────   ────────────────────────────    ──────────────
MessagingDrawer.js          messaging_api.py (REST + WS)
  └─ RoomChat.js            messaging_service.py
MessagingContext.js          messaging_attachment_service.py
messagingService.js          encryption_service.py
                            websocket_manager.py
                            ─── agent bridge ───
                            agent_conversation_watches      dispatch_conversation_reaction
                            (watch_type='chat_room')        → _call_grpc_orchestrator_custom_agent
```

### Key tables

- `chat_rooms` — room_id (UUID), room_name, room_type, created_by, team_id, last_message_at
- `room_participants` — (room_id, user_id) PK, last_read_at, notification_settings JSONB
- `chat_messages` — message_id (UUID), room_id, sender_id, message_content, message_type enum, metadata JSONB, deleted_at, encryption_version
- `message_reactions` — reaction_id, message_id, user_id, emoji; UNIQUE(message_id, user_id, emoji)
- `message_attachments` — file metadata + path, dimensions, is_animated
- `room_encryption_keys` — placeholder for E2EE
- `user_presence` — status enum, last_seen_at

Separate from AI conversations (`conversations` / `conversation_messages`), which have their own branching, citations, and orchestrator integration.

---

## Gaps vs Best-of-Breed

### P0 — Dead code to activate (small effort, high impact)

#### 1. Reactions UI not wired

**Status:** Backend complete, frontend half-done, UI not rendered.

- `message_reactions` table exists with full CRUD API
- `MessagingContext.js` exposes `addReaction` / `removeReaction`
- `RoomChat.js` **never renders reactions** — no emoji picker, no click handler, no reaction display beneath messages

**Fix:** Add emoji picker (e.g. `emoji-mart`) and reaction display to `RoomChat.js`. The context and API layer are ready.

#### 2. Typing indicators broken

**Status:** Server works, client-side delivery drops the event.

- `messagingService.js` has `sendTypingIndicator()` — sends `{ type: 'typing', is_typing }` over room WS
- Server re-broadcasts typing frames to other room members
- `messagingService.js` dispatches incoming typing via `this.messageHandlers.get(roomId)` — but **`messageHandlers` is never populated** during `connectToRoom`
- No "X is typing..." UI component exists in `RoomChat.js`

**Fix:** Register a typing handler in `connectToRoom`, expose typing state via `MessagingContext`, render a typing indicator in `RoomChat.js`.

---

### P1 — Core missing features (medium effort, high impact)

#### 3. @-mentions

No mention parsing, no mention-specific notifications, no `@user` autocomplete anywhere in the messaging stack.

**Why critical:**
- In group rooms, no way to direct attention to a specific person
- Blocks the most natural UX for invoking agents in chat (`@AgentName summarize this`)
- Standard expectation in any modern chat product

**Implementation sketch:**
- Parse `@username` or `@agent_name` patterns server-side in `send_message`
- Store mentions in `chat_messages.metadata` (e.g. `{"mentions": ["user_id_1", "agent_profile_id_2"]}`)
- Trigger push notifications for mentioned users
- For agent mentions: trigger `dispatch_conversation_reaction` with mention context
- Frontend: autocomplete component in message input, highlight mentions in rendered messages

#### 4. Reply-to / quoting (threading lite)

`chat_messages` has **no `parent_message_id`** column. No way to reply to or quote a specific message. Group conversations with multiple topics become confusing.

**Implementation sketch:**
- Add `parent_message_id UUID REFERENCES chat_messages(message_id)` to `chat_messages`
- API: accept optional `reply_to` in send message request
- Frontend: "Reply" action on message hover, quoted message preview above input, indented or linked display in message list
- Full Slack-style threads (separate pane) is a larger lift; inline reply-to is a good first step

#### 5. Agent write-back to rooms (close the orchestrator loop)

The agent watch system triggers agents when messages arrive in a room, but there's no clear path for the agent's response to be posted back as a `chat_message` in that room.

Currently `dispatch_conversation_reaction` calls `_call_grpc_orchestrator_custom_agent`, which stores results in `conversation_messages` (AI chat), **not** `chat_messages` (rooms).

**Implementation sketch:**
- After orchestrator completes, extract the agent response text
- Call `messaging_service.send_message` with `sender_id` mapped from `agent_profile_id` (or a virtual user)
- Set `metadata.from_agent_profile_id` to prevent re-trigger (loop guard already exists in `messaging_api.py`)
- Frontend: render agent messages with agent avatar/name from profile metadata

---

### P2 — Important polish (medium effort, medium-high impact)

#### 6. Message search

No endpoint to search message content within rooms. As rooms accumulate history, finding past information requires scrolling.

**Implementation sketch:**
- Add `GET /api/messaging/rooms/{room_id}/messages/search?q=` endpoint
- PostgreSQL `tsvector` / `ts_query` on `message_content`, or `ILIKE` for simplicity
- Frontend: search bar in room header, highlight matches in results

#### 7. Room history context for agent triggers

When an agent is triggered via watch, it receives only the single triggering message (`message_content`). No room history, no conversation context.

**Fix:** In `dispatch_conversation_reaction`, load last N messages from the room and include them in `extra_context` so the agent can respond coherently. Small change with large quality impact.

#### 8. Markdown rendering in room messages

`RoomChat.js` renders messages as plain text. The AI chat (`ChatMessage.js`) has full Markdown / code block / citation rendering.

**Fix:** Reuse the Markdown rendering pipeline from AI chat. Consider a lighter variant (no citations/charts) for room messages.

#### 9. Message editing

Messages can be deleted but not edited. No `is_edited` flag, no edit history.

**Implementation sketch:**
- Add `is_edited BOOLEAN DEFAULT FALSE` and `edited_at TIMESTAMPTZ` to `chat_messages`
- `PUT /api/messaging/rooms/{room_id}/messages/{message_id}` endpoint (sender only)
- Broadcast `message_edited` WS event
- Frontend: edit action on own messages, "(edited)" indicator

---

### P3 — Nice to have (lower priority)

#### 10. Link previews / rich embeds

No URL unfurling or OpenGraph preview cards. Messages with links render as plain text (except the special `TeamInvitationMessage` type).

#### 11. Pinned messages

No ability to pin important messages for room-level reference. Schema and API additions needed.

#### 12. Per-message read receipts

Currently `last_read_at` tracks room-level read state. No per-message "seen by X, Y, Z" indicators. Lower priority since room-level unread counts cover the primary use case.

#### 13. Agent "thinking" indicator in rooms

When an agent is processing a response, the room should show a typing/thinking indicator. Could reuse the existing typing indicator infrastructure once it's fixed.

#### 14. "Delete for me" fix

API accepts `delete_for=me|everyone` but the service treats both identically (sender-only soft delete). True "delete for me" needs a per-user message visibility table or a `hidden_by` JSONB array.

---

## Agent-in-Chat Integration: Deep Dive

### What exists today

```
User sends message in room
  → messaging_api.py: send_message
    → broadcast via WebSocket
    → check agent_conversation_watches for room_id
      → for each active watch:
        dispatch_conversation_reaction.delay(
          agent_profile_id, user_id, message_content,
          message_sender, watch_type="chat_room", room_id
        )
          → Redis cooldown lock
          → _call_grpc_orchestrator_custom_agent(...)
          → [response stored in AI conversation, NOT the room] ← GAP
```

Loop guard: messages with `metadata.from_agent_profile_id` skip the watch dispatch, preventing infinite agent loops.

Agent Factory UI: `ConversationWatchSection.js` lets users configure which rooms an agent watches.

### What's needed for production-grade agent-in-chat

| Requirement | Status | Work needed |
|-------------|--------|-------------|
| Agent triggered by room messages | Done | `agent_conversation_watches` + Celery |
| Agent response posted back to room | Missing | Write-back step in `dispatch_conversation_reaction` |
| Agent identity (avatar, name) in room | Missing | Virtual participant or metadata-based rendering |
| @-mention triggering (selective) | Missing | Mention parsing + selective agent dispatch |
| Room history as agent context | Missing | Load last N messages into `extra_context` |
| Agent thinking/typing indicator | Missing | WS typing event while agent processes |
| Multi-agent coordination | Missing | Priority/routing when multiple agents watch same room |
| Streaming agent responses in room | Missing | Chunk-by-chunk delivery via room WS (currently batch) |

### Recommended implementation order

1. **Agent write-back** — close the loop so agent responses appear in the room
2. **Agent identity rendering** — agents need their own avatar and name in the message list
3. **@-mention system** — selective triggering instead of every-message firing
4. **Room history context** — agents need conversation context to respond coherently
5. **Thinking indicator** — UX polish while agent processes
6. **Streaming** — optional; batch responses are acceptable for v1

---

## Priority Summary

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| P0 | Wire up reactions UI | Small | High |
| P0 | Fix typing indicator delivery | Small | High |
| P1 | @-mentions (users + agents) | Medium | Very High |
| P1 | Reply-to / quoting | Medium | High |
| P1 | Agent write-back to rooms | Medium | Very High |
| P2 | Message search | Medium | High |
| P2 | Room history context for agents | Small | High |
| P2 | Markdown rendering in rooms | Small | Medium |
| P2 | Message editing | Medium | Medium |
| P3 | Link previews | Medium | Medium |
| P3 | Pinned messages | Small | Medium |
| P3 | Per-message read receipts | Medium | Low-Medium |
| P3 | Agent thinking indicator | Small | Medium |
| P3 | Fix "delete for me" | Small | Low |

### Quick wins (< 1 day each)

- Reactions UI activation (backend is done)
- Typing indicator fix (90% of the code exists)
- Room history in agent context (small addition to Celery task)
- Markdown rendering (reuse AI chat renderer)

### Key files to modify

| Area | Files |
|------|-------|
| Room chat UI | `frontend/src/components/messaging/RoomChat.js` |
| Messaging context | `frontend/src/contexts/MessagingContext.js` |
| WS client | `frontend/src/services/messagingService.js` |
| Message API | `backend/api/messaging_api.py` |
| Message service | `backend/services/messaging/messaging_service.py` |
| Agent watch task | `backend/services/celery_tasks/agent_tasks.py` |
| DB schema | `backend/sql/migrations/` (new migration for parent_message_id, is_edited, etc.) |
