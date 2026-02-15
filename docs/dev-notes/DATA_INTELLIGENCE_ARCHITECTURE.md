# Data Intelligence Architecture: The "Trust Buster" Platform

> **"The more we know about the complex web of interactions, the better we can ensure a Square Deal for all!"**

## 1. Strategic Vision

We are building a **High-Volume Data Intelligence Platform** capable of ingesting disparate external data sources (government contracts, business registries, news feeds), correlating them into a unified view, and allowing users to visualize complex financial and relationship flows.

This architecture separates **Application State** (Users, Settings, Chat History) from **Intelligence Data** (Millions of transactions, crawled entities).

---

## 2. The Gathering Strategy (The Campaign)

We employ a **Hybrid "Big Stick" & "Rough Rider"** approach to data ingestion.

### A. The Registry (The Scout)
A central JSON-based registry (managed by Administrators) defines all known valid data sources. It acts as the traffic cop.

```json
{
  "sources": [
    {
      "id": "fed_contracts",
      "name": "Federal Contracts",
      "type": "API",
      "endpoint": "usaspending.gov",
      "update_frequency": "daily"
    },
    {
      "id": "fl_sunbiz",
      "name": "Florida Business Registry",
      "type": "CRAWL",
      "tool": "crawl_web_content",
      "update_frequency": "weekly"
    }
  ]
}
```

### B. The Agents
1.  **The Scout (Routing):** Receives a query ("Find construction contracts in Miami"). Checks the Registry. Dispatches the appropriate tool.
2.  **The Forager (Ingestion):**
    *   **Big Stick (APIs):** Uses typed connectors for structured APIs (e.g., `fetch_federal_data`). Fast, reliable.
    *   **Rough Riders (Crawlers):** Uses `Crawl4AI` for unstructured sites. Resilient, flexible.
3.  **The Cartographer (Resolution):**
    *   Normalizes raw data into standard entities (`Person`, `Organization`, `Transaction`).
    *   Performs **Entity Resolution**: "Is 'Acme Corp, Inc.' the same as 'Acme Corporation'?"

---

## 3. Storage Architecture (The Triad)

We strictly separate Intelligence Data from the main backend database to ensure performance and scalability.

### A. PostgreSQL: The Ledger (Data Service)
*   **Location:** `data-service` microservice database.
*   **Role:** The authoritative store for raw records and tabular data.
*   **Schema Strategy:**
    *   **Global Tables:** `external_entities`, `gov_contracts`, `public_filings`. (Managed by Admin, Read-Only for Users).
    *   **Workspace Tables:** `workspace_imports`, `user_annotations`. (Scoped to specific User Workspaces).
*   **Security:** Row-Level Security (RLS) ensures users only see Global data + their Workspace data.

### B. Neo4j: The Map Room (Knowledge Graph)
*   **Role:** Topology and Connections.
*   **Data Stored:**
    *   **Nodes:** Entities (Company, Person, Agency).
    *   **Edges:** Relationships (`DIRECTOR_OF`, `AWARDED_CONTRACT`, `TRANSFERRED_FUNDS`).
    *   **Aggregates:** Edge properties store summarized flows (e.g., `total_ytd_amount: 50M`) rather than millions of individual transaction edges.
*   **Partitioning:** Nodes are labeled `:Global` or `:Workspace_{UUID}` to segregate private intelligence.

### C. Vector Store: The Context (Semantic Search)
*   **Role:** Unstructured text search.
*   **Data Stored:** Embeddings of crawled news articles, press releases, and attached PDF documents.
*   **Use Case:** "Find all companies mentioned in articles about 'fraud' or 'embezzlement' in 2025."

---

## 4. Data Interpretation & Visualization (The War Room)

The Frontend provides a "Command Center" view, distinct from the standard Chat interface.

### A. Network Graph (The Map)
*   **Library:** `react-force-graph` or similar.
*   **Function:** Visualizes the Neo4j graph.
*   **Features:**
    *   Color-coded nodes (Blue=Corp, Red=Person, Green=Money).
    *   Shortest Path highlighting ("How is Governor X connected to Vendor Y?").

### B. Sankey Diagrams (The Flow)
*   **Function:** Visualizes the magnitude and direction of financial flows.
*   **Use Case:** Showing how a $50M grant disperses through five different subcontractors.

### C. Master Timeline (The Chronology)
*   **Function:** Correlates financial events with narrative events.
*   **Dual-Axis View:**
    *   **Top:** Bar chart of transaction volumes.
    *   **Bottom:** Point events for news articles/press releases.
*   **Insight:** Spotting spikes in funding that precede or follow major news events.

### D. Data Grid (The Ledger)
*   **Function:** High-performance tabular view of the raw Postgres data.
*   **Features:** Sorting, filtering, exporting. "Show me all transactions > $1M."

---

## 5. Implementation Roadmap

1.  **Phase 1: The Foundation**
    *   Define `data-service` schemas for `external_data_sources` and `intelligence_records`.
    *   Implement the Source Registry in the Admin panel.

2.  **Phase 2: The Connectors**
    *   Build the first API connector (`USASpending`).
    *   Build the generic `Crawl4AI` connector for the Forager agent.

3.  **Phase 3: The Graph**
    *   Enhance `KnowledgeGraphService` to support Financial Flow edges (hyperedges/aggregates).

4.  **Phase 4: The War Room**
    *   Build the Frontend Visualization components.

**"Speak softly and carry a big stick; you will go far."** - In data terms: Keep your architecture clean, but your processing power massive!
