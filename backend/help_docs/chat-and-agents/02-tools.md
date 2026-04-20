---
title: Agent tools
order: 2
---

# Agent tools

In chat, **tools** are concrete **actions** the system runs on your behalf while an agent works: searching your library, fetching a document, calling the web, sending a message, and so on. The model decides *when* to call a tool (for agents that use tools); the platform runs the tool and returns a result the model can use in its reply.

---

## How tool use shows up in the thread

- You may see **progress** or intermediate states while a tool runs (e.g. searching, loading).
- **Tool results** often appear as expandable blocks or summaries in the thread so you can inspect what the agent retrieved without rereading the whole answer.
- If something needs your **confirmation** before running (see below), the thread pauses until you approve or reject.

---

## Built-in agents and typical tool access

Routing picks a **built-in route** (research, chat, writing-focused editor routes, etc.). Each route can expose a different **tool set**. Examples:

- **Research-oriented routing** — When the **research** route is selected, the agent can use tools such as **search documents**, **web search**, **query enhancement**, **search segments across documents**, **crawl web content**, **get document content**, **get document metadata**, and **search images** (and similar combinations on related research routes).
- **General chat** — The default **chat** route is oriented toward conversation; it does not ship the full research tool bundle on the route definition itself. Context still comes from your conversation, attachments, and open document when the app sends them. For heavy lookup, the system may route to **research** instead.
- **Writing / editor routes** — Manuscript and editor-style routes often include **document search**, **get document content**, and **segment search** so the agent can ground answers in your files while you work in the editor.

Exact bindings can evolve with product defaults; **Agent Factory → Tools reference** describes the full catalog and categories.

---

## Tool approval

**Custom agents** (Agent Factory) can include **approval** steps in their playbooks: the run pauses until you confirm sensitive actions (e.g. sending email, applying edits). Built-in chat routing may also surface permission-style flows depending on configuration.

For **scheduled or background** runs, an agent profile can use a **default approval policy** so approval gates do not block unattended jobs (see **Agent Factory → Schedules and monitors** and **Profile settings**).

---

## Where to read more

- **Agent Factory → Tools reference** — Full tool catalog, categories, and how tools return `formatted` plus typed fields for playbook wiring.
- **Agent Factory → Playbook steps and flow control** — How **tool**, **LLM agent**, and **deep agent** steps bind tools and pass outputs between steps.

---

## Summary

Tools are the agent’s **hands**: search, read, act. You see their **effects** in the thread. Research-style routes expose a rich search-and-gather set; chat is conversational; editor routes emphasize documents. Use Agent Factory docs for the complete tool list and playbook wiring.
