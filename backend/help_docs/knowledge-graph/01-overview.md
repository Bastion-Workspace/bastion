---
title: Knowledge Graph overview
order: 1
---

# Knowledge Graph overview

The **Knowledge Graph** stores **entities** and **relationships** (e.g. people, places, topics, and links between them). Data is stored in Neo4j and is used by search and agents. You can explore the graph with two views: **Entity Relationship Graph** (entities and their relations) and **File Link Graph** (links between documents). This page describes what the graph is, how to open the views, and how agents populate and use it.

---

## What the Knowledge Graph is

The graph has **nodes** (entities or documents) and **edges** (relationships or links). Examples:

- **Entity graph** — Nodes are entities (person, organization, location, concept, etc.) extracted from your content or added by agents. Edges are relationships (e.g. “works at”, “located in”, “mentions”). Agents can add entities and relationships when processing documents or answering questions.
- **File link graph** — Nodes are **documents** (files in the Document Library). Edges are **links** between them (e.g. explicit markdown/org links or references). Useful to see how files connect.

The graph powers **semantic search** and **agent context**: when you or an agent searches, the system can use entity and relationship data to improve results. Agents can also query the graph (e.g. “find all entities of type X” or “what is related to Y”) and write new entities/relationships from their outputs.

---

## Opening the graph views

- **From the Document Library** — When viewing a document or the library, you can open a **tab** or **view** for the graph. Typical options:
  - **Entity Relationship Graph** — Opens a graph visualization (nodes and edges). You can **search** for an entity, **zoom** and **pan**, and click a node to see **details** (type, properties, related entities). **Refresh** reloads the layout and data. The view may offer **fullscreen** and filters (e.g. by entity type or relationship type).
  - **File Link Graph** — Same idea but nodes are **documents** and edges are **links** between them. Useful to see which files reference which.

- **Where the tabs live** — The exact entry point depends on the UI: it may be a tab type in the **TabbedContentManager** (e.g. “Entity graph”, “File graph”), a sidebar item, or a button from a document. Look for “Entity Relationship Graph”, “File Link Graph”, “Knowledge Graph”, or “File links”.

---

## Browsing entities and details

- **Entity view** — In the Entity Relationship Graph, click a **node** to select it. A **detail** panel or dialog may show the entity’s **type**, **name**, **properties**, and **relationships** (incoming and outgoing edges). You can follow a relationship to another entity and expand the graph from there.
- **Search** — Use the **search** box to find an entity by name or type. Selecting a result may center that node in the graph and show its details. **Refresh** reloads data from the server so new entities added by agents appear.

---

## How agents populate and use the graph

- **Population** — Agents (e.g. research, writing, or custom Agent Factory agents) can call tools that **add entities** and **relationships** to the Knowledge Graph. For example, after analyzing a document, an agent might extract “Person: Jane”, “Organization: Acme”, and “Relationship: Jane works at Acme”. That data is written to Neo4j and shows up in the Entity Relationship Graph.
- **Search** — Vector and keyword search can be augmented with graph data: e.g. “find documents related to entity X” or “expand from this entity to related documents”. So the graph improves discovery and context for both you and the agents.
- **File link graph** — Links between documents may be updated when you or an agent add links in content (e.g. markdown `[text](file.md)`). The File Link Graph view reflects those links so you can navigate and understand document structure.

---

## Summary

- The **Knowledge Graph** stores **entities** and **relationships** (Neo4j). **Entity Relationship Graph** shows entities and relations; **File Link Graph** shows **document** nodes and **links** between files.
- Open the views from the Document Library (tabs or sidebar). **Search** and **click** nodes to browse; use **Refresh** to reload. **Agents** populate the graph via tools and search uses it for better context and discovery.

---

## Related

- **Document Library overview** — Tabs and sidebar.
- **Agent tools** — Tools that read and write the graph.
- **Agent Factory overview** — Custom agents that can use graph tools.
