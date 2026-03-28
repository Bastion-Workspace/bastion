---
title: Creating and configuring connectors
order: 7
---

# Creating and configuring connectors

This page walks you through **creating a data connector** and **configuring its endpoints** so your playbooks can call an external API. You’ll set the base URL and authentication, add one or more endpoints (path, method, response mapping), and use the built-in **test** to confirm the API responds as expected. The exact fields may vary slightly depending on your UI; this guide covers the main concepts.

---

## Where to create a connector

In Agent Factory you can usually:

- **Create a new connector** from the Data Connections area (e.g. a “New connector” or “Create connector” action) and then attach it to a profile, or
- **Add from a template** when editing a profile’s Data Connections — if your instance provides templates for common APIs, you pick one and a connector is created and attached in one step.

Either way, once the connector exists you’ll open it to edit **Identity**, **Connection**, and **Endpoints**. The steps below assume you’re editing a connector definition (e.g. in a Connector Builder or similar screen).

---

## Step 1: Identity

Give the connector a **name** and, optionally, a **description**. The name should be something you’ll recognize when you see it in the list of data sources (e.g. “Acme CRM API” or “Internal Reporting Service”). The description can explain what the API does or which playbooks use it.

---

## Step 2: Connection (base URL and auth)

**Base URL**  
Enter the API's root URL, e.g. `https://api.example.com`. Do not include a trailing slash. Every endpoint you add will be relative to this base (e.g. path `/v1/users` becomes `https://api.example.com/v1/users`).

**Auth type**  
Choose how the API is authenticated:

| Auth type | What you configure |
|-----------|--------------------|
| **None** | No credentials; use for public APIs that don't require auth. |
| **API Key** | Configure **how** the key is sent: **Credentials key name** (e.g. `api_key`), **Send key in** (Header or Query parameter), and the header or query parameter name. Enter the **API key or token** in the same Connection section — this value is used when you run endpoint tests below. For playbook runs, set the key when you add the connector to a profile (data source binding). |
| **Bearer Token** | Configure the **credentials key name** and enter the **token** in the same Connection section. It is used for endpoint tests; for playbook runs, set the token in the profile's data source binding. |
| **OAuth (External Connection)** | In the same Connection section, select an **External Connection** (for testing). At runtime, the profile's data source binding selects which connection to use. |

Save the connection section before adding endpoints.

---

## Step 3: Endpoints

Each **endpoint** is one callable operation: a path, method, and (optionally) parameters and response mapping.

**Add an endpoint**  
Click “Add endpoint” (or equivalent). You’ll get a new endpoint to configure.

**Endpoint ID (slug)**  
A stable identifier for this endpoint (e.g. `get_users` or `fetch_orders`). This is what appears in the Composer when you choose “call this endpoint.” Use letters, numbers, and underscores; avoid spaces.

**Path**  
The path relative to the base URL, e.g. `/v1/users` or `/v2/orders`. You can use **placeholders** like `{id}` if your API supports them (e.g. `/v1/users/{id}`); when the step runs, you pass a value for `id` in the step’s inputs.

**Method**  
Usually **GET** (read) or **POST** (submit data). Pick what the API expects.

**Parameters**  
If the API expects query parameters or body fields, you define them here (or in the step inputs when you use the endpoint in a playbook). The UI may let you list parameter names and types; at run time you wire values from previous steps or runtime variables (e.g. `{query}` or `{step.result.id}`).

**Response list path**  
Many APIs return JSON like `{ "data": [ ... ] }` or `{ "items": [ ... ] }`. To use the list in later steps (e.g. “for each item”), you tell the system where the array lives using **dot notation**, e.g. `data` or `data.items`. Enter the path to the array; leave blank or use `.` if the response is already a top-level array. The system uses this to expose a structured list for wiring.

**Description**  
Optional short description of what this endpoint does (e.g. “List all users” or “Fetch order by ID”). Helpful when you have many endpoints.

You can add multiple endpoints to one connector. Each one becomes a separate “tool” in the playbook.

---

## Step 4: Testing an endpoint

Before saving, use **Test** to confirm the API responds correctly.

1. Select the **endpoint** you want to test.
2. **Credentials** are taken from the **Connection** section — the API key or token, or the External Connection, you already set there. You do not enter credentials again in the test form.
3. **Params** — Enter only the parameters the caller supplies (e.g. `user_id`, `symbol`, `limit`) as JSON in the test params field, e.g. `{"symbol": "AAPL", "limit": 10}`. Any fixed or default parameters defined on the endpoint are sent automatically.
4. Run **Test** (or "Run test"). The UI will show the **raw response** and any error message.

If the response is JSON, you can often **click a key** in the response viewer to set (or confirm) the **response list path** for that endpoint — so the system knows which part of the JSON to treat as the list for downstream steps.

Fix any auth or path issues, then save the connector. After that, attach the connector to an agent profile (if not already attached) and add a playbook step that uses this endpoint; wire its output to the next step as you would for any other tool.

---

## Summary

1. **Identity** — Name and description.
2. **Connection** — Base URL and auth type (none, API key, bearer, or OAuth via External Connection).
3. **Endpoints** — For each operation: endpoint ID, path, method, optional params, response list path (dot notation), and description.
4. **Test** — Run a test request with the chosen auth and params; use the raw response to set or verify the response list path.

Once the connector is saved and attached to a profile, its endpoints appear as tools in the Workflow Composer for that profile’s playbook. For an overview of how connectors fit into Agent Factory, see **Data Connectors overview**.
