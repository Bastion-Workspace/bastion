---
title: Data Connectors overview
order: 9
---

# Data Connectors overview

**Data connectors** let your custom agents call **external APIs** and services as part of a playbook. Instead of coding an integration, you define a connector: a **base URL**, **authentication**, and one or more **endpoints** (paths and how to read the response). Once a connector is attached to an agent profile, its endpoints show up as tools in the Workflow Composer so you can add “call this API” steps to your playbooks. This page explains what connectors are, when to use them, and how they fit into Agent Factory.

---

## What a connector is

A connector is a **reusable definition** of an external service:

- **Identity** — Name and description so you can recognize it in the UI.
- **Connection** — Base URL (e.g. `https://api.example.com`) and **authentication** (none, API key, bearer token, or OAuth via an External Connection).
- **Endpoints** — A list of operations you want to expose. Each endpoint has a path (e.g. `/v1/users`), HTTP method (GET or POST), optional parameters, and a **response list path** (a dot-notation path into the JSON response so the system can extract a list for downstream steps).

Connectors are stored in Agent Factory. You can **create your own** (e.g. for a private or third-party API) or **add from a template** if your instance offers templates for common services. Once a connector exists, you **attach it to an agent profile** in the profile’s Data Connections section. After that, the connector’s endpoints appear as tools when you edit that profile’s playbook — so you can add a step that “calls” an endpoint and wire its output to the next step.

---

## When to use connectors

Use a **connector** when your playbook needs to:

- Call a **REST API** (your own backend, a SaaS, or a public API) to fetch or send data.
- Reuse the same API across **multiple agents** — define the connector once, attach it to any profile.
- Keep **credentials and URLs** in one place — you configure auth and base URL on the connector (or via an External Connection for OAuth); playbook steps only reference the endpoint and pass parameters.

If the action is already a **built-in tool** (e.g. search documents, send email, run SQL), use that tool instead — you don’t need a connector for it. Connectors are for **custom** or **external** APIs that aren’t built into Bastion by default.

---

## How connectors become playbook tools

When you attach a connector to an agent profile:

1. The system registers the connector’s **endpoints** as tools available to that profile’s playbook.
2. In the Workflow Composer, when you add a step and choose an action, you’ll see options like **connector:MyAPI:endpoint_id** (or similar, depending on how your UI labels them). Selecting one means “call this endpoint when this step runs.”
3. You wire **inputs** to the step — e.g. parameters the endpoint expects, often from earlier steps or from runtime variables like `{query}`.
4. The step’s **output** is what the endpoint returns (typically a `formatted` summary plus structured fields). You can wire that output to later steps.

So connectors don’t require code in the playbook — you define the API once, then use it like any other tool in your workflow.

---

## Credentials and security

- **API key / Bearer** — In the **Connection** section you configure **how** the key is sent (credentials key name, header vs query parameter, header or param name) and enter the **key value** in the same place. That value is used when you run endpoint tests in the builder. For playbook runs, you set the key when you add the connector to an agent (data source binding). For **API key** you can choose to send it in a **header** (default, e.g. `X-API-Key`) or as a **query parameter** in the URL (e.g. `?apikey=...`). Some APIs, such as Alpha Vantage, require the key as a query parameter; in that case set "Send key in" to **Query parameter** and the **Query parameter name** (e.g. `apikey`). Keep connector definitions and credentials restricted to users who should have access.
- **OAuth (External Connection)** — The connector is configured to use "OAuth (External Connection)". In the Connection section you choose which **External Connection** to use when testing endpoints. At runtime, the profile's data source binding selects the connection. Secrets live in the connection store, not in the connector definition.

When you test an endpoint (see **Creating and configuring connectors**), credentials come from the Connection section — the same API key or token, or the same External Connection, you already set there.

---

## Summary

- **Connectors** define external APIs: base URL, auth, and endpoints. They’re reusable and attached to agent profiles.
- **Endpoints** become **playbook tools** so you can add “call this API” steps and wire results to the next step.
- Use connectors when you need to call custom or third-party REST APIs; use built-in tools when the action is already provided by Bastion.
- Auth is configured on the connector (or via External Connection for OAuth). See **Creating and configuring connectors** for step-by-step setup and testing.
