---
title: Artifacts
order: 6
---

# Artifacts

**Artifacts** are rich visual outputs an agent can attach to a reply: HTML pages, charts, diagrams, SVG graphics, or live React components. They open in a **sidebar drawer** next to the chat so you can preview, read the source, copy it, save it to your library, or view it full screen.

---

## Artifact lifecycle

Artifacts start as **chat previews** in a sidebar drawer. From there you can:

- **Save to your artifact library** — keeps the artifact for reuse beyond the chat session.
- **Embed on the home dashboard** — add it as a widget card (resizable, always visible when you visit the dashboard).
- **Embed as a control pane** — pin it in the status bar as a compact popover that can run in the background.
- **Share via public link** — anyone with the link can view it (but live data queries require a login session, so the artifact should handle that gracefully).

Because the same artifact can appear in multiple places at different sizes, agents are instructed to keep layouts responsive and use cooperative state for anything stateful.

---

## Artifact types

| Type | Best for | Notes |
|------|----------|--------|
| **html** | Static or lightly interactive pages, custom layouts | Self-contained HTML (inline CSS/JS or CDN scripts). Renders in a sandboxed preview. |
| **chart** | Numeric or time-series visuals (e.g. Plotly) | Same as HTML: usually a snippet that loads a chart library from a CDN and draws the chart. |
| **mermaid** | Flowcharts, sequence diagrams, ERDs, Gantt charts | Pass **diagram source only** (no HTML wrapper). Valid Mermaid syntax. |
| **svg** | Vector graphics, icons, simple illustrations | SVG markup; displayed inline after sanitization. |
| **react** | Interactive UI demos, small components | JSX with a root **`App`** component or **`export default`**. No `import`; use **`React.useState`** and globals **`React`** / **`ReactDOM`**. Preview uses React 18 from a CDN inside a sandbox. |

---

## Live data in HTML, chart, and React artifacts

When you are signed in, the preview iframe includes **`window.bastion.query(path, queryParams)`**: it asks the main app to run an **allowlisted read-only GET** with your session and return JSON. Use this instead of **`fetch()`** to this product’s **`/api/...`** endpoints (the sandbox blocks direct API access).

- **Paths** are fixed to a small set (for example: todos, org agenda/tags/search, calendar, RSS feeds and articles, folder tree, document pins, folder contents, status bar data). Anything else is rejected.
- **Limits** apply (request rate and concurrency); prefer occasional refresh, not tight loops.
- **Shared links** to an artifact do not carry your login for that bridge, so live queries fail there—artifacts should handle errors or show a static fallback.

Agents with the **Artifact React** skill get the exact path list and coding patterns in their instructions.

---

## Cooperative state (multi-instance sync)

When an artifact is **saved to your library** and embedded in more than one place (for example a **home dashboard widget** and a **control pane** in the status bar), the preview can keep **shared state** across those instances.

- **`bastion.getState(key)`** — read a value from the shared map (JSON-serializable).
- **`bastion.setState(key, value)`** — write a value; other instances of the same saved artifact receive the update.
- **`bastion.onStateChange(callback)`** — register a listener; it receives `{ key, value, state }` when shared state changes (including from another instance).

Use these for timers, counters, toggles, or any mutable UI state that should stay aligned when the same artifact appears twice.

- **`bastion.notify({ badge: true, text: '...' })`** — optional hint for a badge on the control pane icon (implementation may show `text` when `badge` is true).

In a **one-off chat preview** (before save), cooperative APIs are effectively no-ops because there is no stable `artifact_id` — it is still safe for generated code to call them so the artifact is ready after save.

Agents with **Artifact Generation** or **Artifact React Components** skills are instructed to use cooperative state when building stateful artifacts.

---

## How to get an artifact

Ask the agent in natural language—for example: “Show this as a Mermaid diagram,” “Plot this data as a chart,” or “Build a small React counter demo.” Agents that have the **create_artifact** tool (often via skills like **Artifact Generation**, **Artifact Charts**, **Artifact Diagrams**, or **Artifact React Components**) will call the tool and you will see an **artifact card** in the thread.

---

## Drawer actions

When you click **Open** on an artifact card:

- **Preview** — Renders the artifact (iframe for HTML/chart/React, diagram for Mermaid, inline SVG).
- **Code** — Toggle syntax-highlighted source.
- **Full screen** — Expand the preview (not the code view).
- **Copy** — Copies the artifact source to the clipboard (you get a short confirmation when it succeeds).
- **Save** — Saves to your document library as HTML or as Markdown with a fenced code block.
- **Close** — Closes the drawer.

If the agent sends a **new** artifact while the drawer is already open, the UI can keep a short **version history** in that session so you can step back to a previous revision from the toolbar (session only; reloading the page clears it).

---

## Multiple artifacts in one reply

If the agent creates more than one artifact in a single turn, you may see **multiple cards** in the message—one per artifact. Open any card to inspect that artifact in the drawer.

---

## Tips

- Keep HTML and chart snippets **self-contained**; external scripts are typically limited to well-known CDNs.
- **Mermaid**: use raw diagram text (e.g. lines starting with `flowchart`, `sequenceDiagram`, …).
- **React**: prefer `function App() { ... }` or `const App = () => ...` or `export default function App()`; use `React.createElement` or JSX; avoid `import` because the preview has no bundler.
- **Live data**: use `bastion.query` for supported `/api` reads while signed in; do not rely on it on public share pages.
- Very large payloads may be rejected by the server; avoid huge embedded assets.

---

## Where to read more

- **Agent Factory** — Tools and skills catalog; agents need **create_artifact** (and optional artifact skills) to emit artifacts.
- **Agent tools** — How tool use appears in the chat thread.

---

## Summary

Artifacts add charts, diagrams, HTML, SVG, and React previews beside the chat. Open a card to preview, edit-view source, copy, save, or go full screen. Ask the agent clearly when you want a visual artifact; the right skills help it choose the correct **artifact_type**.
