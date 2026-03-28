---
title: Chat and agents overview
order: 1
---

# Chat and agents

Bastion includes a chat interface and specialized agents for research, writing, coding, and general conversation. You open the chat from the right-hand sidebar, pick or create a conversation, and send messages; agents respond with streaming text and can use tools (search, file edits, and more). This page covers how to open chat, use conversations, work with agents, and what the chat input offers.

---

## Opening chat

- **Sidebar** — The chat lives in a **sidebar** on the right. When it is visible, you see the conversation list and the current chat. Use the **chevron** button on the right edge to open or collapse the sidebar; when collapsed, a floating **Open Chat** button appears so you can bring it back.
- **Resizing** — Drag the left edge of the chat sidebar to make it wider or narrower. The width is remembered for your next visit.
- **Full width** — On the Documents, Media, or Agent Factory page you can expand the chat to full width so it overlays the main content; toggle back to normal width when done.

---

## Conversations

The chat sidebar lists **conversations**. Each conversation has its own history and context.

- **Create** — Use the **New conversation** action (e.g. plus icon or menu) to start a new conversation. It appears in the list and becomes the active chat.
- **Switch** — Click a conversation in the list to switch to it. The messages for that conversation load and you can continue where you left off.
- **Rename** — Rename the current conversation from the sidebar (e.g. via a menu or edit control) so you can tell them apart.
- **Pin** — Pin important conversations so they stay at the top of the list.
- **Delete** — Remove a conversation from the list when you no longer need it. Deletion is permanent.

Conversation history is stored so you can close the sidebar or navigate away and return later. For more detail, see **Conversations**.

---

## How agents work

When you send a message, Bastion routes it to an agent. Built-in agents include:

- **Research** — Searches your documents and optionally the web, then synthesizes answers with sources.
- **Chat** — General conversation, follow-ups, and lightweight tasks using your knowledge base and tools.
- **Writing** — Helps with drafting, editing, and longer-form content; can work with the open document.
- **Coding** — Assists with code generation, analysis, and technical questions.

The system chooses an agent based on your message, or you can influence the behavior via context (e.g. having a document open). **Agent Factory** agents you create have an **@handle**; type **@** in the chat input and select an agent to run its playbook for that turn.

Responses **stream** in as they are generated. Agents may call **tools** (e.g. search documents, read or edit a file, list todos); you see progress and results in the thread. If an agent needs your approval (e.g. before running a tool), you get a prompt to approve or reject.

---

## Chat input area

At the bottom of the chat sidebar you have:

- **Text field** — Type your message. You can mention an Agent Factory agent by typing **@** and choosing from the list.
- **Send** — Submit the message. Depending on the agent and settings, you may have a choice of model or mode (e.g. fast vs full).
- **Attachments** — Attach files or reference the **open document** so the agent can see its content. The open editor context (current file, selection) is often sent automatically when relevant.
- **Model selector** — If available, choose which model the next reply should use (e.g. from Settings > Models).

The agent’s reply appears above the input. You can keep asking follow-up questions in the same conversation; the agent uses the thread history as context.

---

## Where to go next

- **Conversations** — Creating, renaming, pinning, deleting, and switching conversations in detail.
- **Org Quick Capture** — Quick Capture with **Ctrl+Shift+C** for notes, TODOs, and more.
- **Agent tools** — What tools agents can use (search, files, todos, etc.).
- **Agent Factory overview** — Building custom agents and @mentioning them in chat.
