# Chat Channels Roadmap and Deployment Plan

Parity target: [OpenClaw Chat Channels](https://docs.openclaw.ai/channels). This document covers which platforms we aim to support, implementation approach, and a phased deployment plan.

## Current State

| Platform   | Status   | Notes                                              |
|-----------|----------|----------------------------------------------------|
| Telegram  | Supported| Bot API via python-telegram-bot; typing indicator. |
| Discord   | Supported| discord.py; servers, channels, DMs; typing.        |
| Microsoft | Separate | Microsoft Graph (email), not a chat channel.        |

All chat providers implement `BaseMessagingProvider` in `connections-service/providers/`, are registered in `service/provider_router.py`, and are driven by `service/channel_listener_manager.py` (inbound → backend → reply, typing indicator, /newchat, /model).

## Target Platforms (Parity with OpenClaw)

### Priority: Slack, Signal, Mattermost

| Platform   | OpenClaw stack     | Our approach (proposed)     | Complexity |
|-----------|--------------------|-----------------------------|------------|
| **Slack** | Bolt SDK           | Slack Bolt for Python (Bolt SDK) or slack_sdk; workspace apps, events API, socket mode or HTTP. | Medium     |
| **Signal**| signal-cli         | signal-cli REST/DBUS or a small bridge service; no official bot API.                          | Medium–High|
| **Mattermost** | Bot API + WebSocket | Mattermost Driver (e.g. mattermostdriver) or REST + WebSocket; bot account + token.       | Medium     |

### Stretch / Later: Matrix

| Platform | OpenClaw stack | Our approach (proposed) | Complexity |
|----------|----------------|-------------------------|------------|
| **Matrix** | Matrix protocol (plugin) | matrix-nio (async); rooms, DMs, end-to-end optional; protocol and terminology differ from “chat_id”. | High      |

Matrix is documented here for roadmap context; implementation may be deferred due to protocol complexity and E2E/identity handling.

## Implementation Pattern (Per Platform)

Each new channel should:

1. **Implement `BaseMessagingProvider`**  
   - `name`, `start()`, `stop()`, `send_message()`, `get_bot_info()`, `send_typing_indicator()` (or no-op).
2. **Normalize to `InboundMessage`**  
   - `sender_id`, `sender_name`, `chat_id`, `text`, `images`, `platform`, `connection_id`.
3. **Register in `provider_router.py`**  
   - Add to `get_messaging_providers()` dict (e.g. `"slack": SlackProvider`).
4. **No change to `channel_listener_manager`**  
   - Same flow: commands (/newchat, /model) vs chat → backend `send_external_message` → reply; typing loop already calls `send_typing_indicator(chat_id)` for all providers.
5. **Backend / DB**  
   - External connections table and UI already support multiple connection types; add new `provider` (or equivalent) value and any provider-specific config (tokens, webhook URLs, etc.) as needed.
6. **Formatting**  
   - Reuse or extend `providers/formatting.py` (e.g. Markdown → Slack mrkdwn, Mattermost Markdown, etc.) so agent output is safe and readable per platform.

Platform-specific details (auth, webhooks vs polling, rate limits, attachments) live in the provider module and config.

## Deployment Plan

### Phase 1: Slack

- **Goal:** Slack as a first-class channel (workspace app or bot).
- **Dependencies:** Slack app (Bot Token, optional Socket Mode or Events API + HTTP endpoint), `slack_sdk` or Bolt in `connections-service/requirements.txt`.
- **Config:** Bot token (and optionally signing secret, app token for socket mode) stored per connection.
- **Steps:**
  1. Add `SlackProvider` implementing `BaseMessagingProvider`; map Slack channel ID to `chat_id`, user to `sender_id`/`sender_name`.
  2. Handle Slack’s markdown variant (mrkdwn) in formatting.
  3. Register in provider router; add “Slack” in backend/UI as a connection type.
  4. Test in a dev workspace (DMs and channels as appropriate).
- **Typing:** Slack has typing indicators; implement `send_typing_indicator` via Slack API if available for the chosen SDK.

### Phase 2: Mattermost

- **Goal:** Mattermost servers (channels, DMs) as a channel.
- **Dependencies:** Mattermost server URL + bot/personal access token; `mattermostdriver` or REST + WebSocket.
- **Config:** Server URL, token (and optionally team ID), stored per connection.
- **Steps:**
  1. Add `MattermostProvider`; map channel/team to `chat_id`, user to sender fields.
  2. Use Mattermost’s Markdown (and attachment format if needed).
  3. Register in provider router; add “Mattermost” in backend/UI.
  4. Test against a Mattermost instance (self-hosted or cloud).
- **Typing:** Implement if Mattermost API supports it; otherwise no-op.

### Phase 3: Signal

- **Goal:** Signal as a channel (DMs / possibly groups).
- **Dependencies:** signal-cli (or equivalent) with REST/DBUS interface; linking or registration flow for the “bot” number.
- **Config:** signal-cli endpoint, auth (e.g. registered number/link), stored per connection.
- **Steps:**
  1. Add `SignalProvider`; if using signal-cli REST, map recipient ID to `chat_id`, sender to `sender_id`/`sender_name`.
  2. Text and optional image handling; Signal has no rich markdown, so plain text or minimal formatting.
  3. Register in provider router; add “Signal” in backend/UI.
  4. Document setup (signal-cli install, registration, linking).
- **Typing:** Signal may not expose a typing API for bots; no-op acceptable.
- **Note:** signal-cli is often run as a separate process; connections-service may call it via HTTP or DBUS. Consider a small adapter if needed.

### Phase 4 (Optional / Later): Matrix

- **Goal:** Matrix rooms and DMs as a channel.
- **Dependencies:** `matrix-nio`; homeserver URL; bot or user access (token or login).
- **Config:** Homeserver, user ID or bot token, room allowlist or auto-join rules.
- **Steps:**
  1. Add `MatrixProvider`; map room ID to `chat_id`, event sender to `sender_id`/`sender_name`.
  2. Format messages for Matrix (Markdown supported).
  3. Handle sync loop, join rules, and optionally E2E (complexity).
  4. Register in provider router; add “Matrix” in backend/UI.
- **Typing:** Matrix supports typing; implement if we add Matrix.
- **Deferral:** Document only until we commit to the extra protocol and ops burden.

## Documentation and References

- OpenClaw channels index: https://docs.openclaw.ai/channels  
- OpenClaw model providers: https://docs.openclaw.ai/providers/models  
- Slack: Bolt for Python / Slack API  
- Mattermost: REST API, WebSocket, Bot accounts  
- Signal: signal-cli (e.g. GitHub rest-api or DBUS)  
- Matrix: matrix-nio, Matrix spec  

## Summary

| Phase | Platform   | Priority | Complexity |
|-------|------------|----------|------------|
| 1     | Slack      | High     | Medium     |
| 2     | Mattermost | High     | Medium     |
| 3     | Signal     | High     | Medium–High|
| 4     | Matrix     | Stretch  | High       |

Implementation stays consistent: one provider class per platform, same listener and backend contract. New platforms require provider code, router registration, backend/UI connection type, and optional formatting/typing; no change to core orchestration logic.
