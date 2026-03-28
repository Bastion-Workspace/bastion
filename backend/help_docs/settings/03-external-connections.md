---
title: External connections
order: 3
---

# External connections

The **Connections** tab in Settings is where you configure **external connections**: email (Microsoft OAuth or IMAP/SMTP), messaging bots (Telegram, Discord, Slack, Twilio SMS), system email for notifications, and device tokens for push. These connections let agents and the system send and receive email and messages. This page describes each connection type and how to set them up. For **Agent Factory connectors** (custom API definitions for playbooks), see the Agent Factory docs — those are different from the connections here.

---

## Email

- **Microsoft OAuth (Office 365)** — Connect a Microsoft account so Bastion can read and send email via Microsoft Graph. Click **Connect** (or similar) and complete the OAuth flow in the browser. Once connected, agents and tools can access mailboxes and send mail as that account. Use **Disconnect** to remove the link.
- **IMAP / SMTP** — Configure incoming (IMAP) and outgoing (SMTP) server details: host, port, username, password, and TLS. Used for non-Microsoft mailboxes. After saving, the system can fetch email and send mail via SMTP. Useful for Gmail (with app password), self-hosted mail, or other providers.

You typically use one primary email connection (OAuth or IMAP/SMTP) for agent-driven email. The UI may show which account is connected and offer to refresh or re-authenticate.

---

## Messaging bots

Connect **messaging platforms** so agents can send and receive messages on your behalf. Each platform has its own setup (token, webhook, or OAuth). Common options:

- **Telegram** — Create a bot via @BotFather, get the bot token, and enter it in Settings. Connect the bot; agents can then send and receive Telegram messages in configured chats or channels.
- **Discord** — Create an application and bot in the Discord Developer Portal, get the bot token, and add it here. After inviting the bot to a server, agents can post and read messages.
- **Slack** — Use a Slack app and bot token (and optionally OAuth) to connect a workspace. Agents can post to channels and respond to messages depending on configuration.
- **Twilio SMS** — Add Twilio credentials so the system can send (and optionally receive) SMS. Used for notifications or agent-driven SMS.
After a bot is connected, it appears in the connections list. You can **Disconnect** or **Refresh** to re-check the link. Agents and notification flows use these connections when sending to the corresponding channel (e.g. “send to Telegram” in a playbook step).

---

## System email (SMTP)

**System email** is used for **notifications** sent by the system (e.g. digest emails, alerts), not necessarily for agent-composed email. Configure an SMTP server (host, port, user, password, TLS, from address/name). You may be able to **Send test email** to verify the settings. Some instances allow choosing an **existing connection** (e.g. the same SMTP used for agent email) as the system email source; others use a dedicated SMTP block. Admins may restrict who can change this.

---

## Device tokens

**Device tokens** store push notification tokens for devices (e.g. browsers or mobile). When you enable notifications in a browser, the token is registered here. The system uses these tokens to send push notifications (e.g. for alerts or mentions). You can view and remove tokens if your instance exposes a Device Tokens or Browser Tokens section in Connections or a related settings page.

---

## How connections are used

- **Agents and playbooks** — Tools like “send email” or “send channel message” use the connections you configure. For email, the connected mailbox is used to send and receive. For messaging, the connected bot is used to post and read in the linked channel or chat.
- **Notifications** — Scheduled or event-driven notifications (e.g. “daily briefing”, “mention in team”) can be delivered by email (system SMTP) or by messaging (Telegram, Discord, etc.) using these connections.
- **Security** — Tokens and passwords are stored securely. Only connect accounts and bots you trust. Use **Disconnect** when you no longer need a connection.

---

## Summary

- **Connections** in Settings configures email (Microsoft OAuth or IMAP/SMTP), messaging bots (Telegram, Discord, Slack, Twilio SMS), system SMTP for notifications, and device tokens for push.
- **Email** and **messaging** connections are used by agents and tools to send and receive; **system email** is for system notifications.
- **Agent Factory connectors** are separate: they define external APIs for playbook steps and are configured in Agent Factory, not in Settings > Connections.

---

## Related

- **Settings overview** — All settings tabs.
- **Models and providers** — Chat and image models (no connection setup).
- **Data Connectors overview** — Agent Factory connectors for playbooks.
- **Creating and configuring connectors** — Defining API connectors in Agent Factory.
