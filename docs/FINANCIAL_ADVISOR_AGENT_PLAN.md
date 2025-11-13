## Financial Advising Agent — Planning Document

### Overview
This document outlines the design for a Financial Advising Agent and its companion Background Tracker. The agent answers finance questions with sourced, structured analyses, while the background process continuously gathers and refreshes company data (SEC filings, prices, financial statements, news) to keep context current and response latency low.

### Goals
- Provide fast, well‑sourced answers to finance questions (tickers, filings, actions, fundamentals, valuation, scenarios).
- Maintain continuously updated context via background collection of authoritative data.
- Use structured outputs (Pydantic) for type‑safe inter‑agent communication and response rendering.
- Fit cleanly into the existing LangGraph orchestrator, Celery background jobs, and Docker‑Compose runtime.

### Alignment with Current Architecture
- Orchestration: Extend `LangGraphOfficialOrchestrator` routing to recognize finance intents and delegate to a new `FinancialAgent`.
- Agents: Add `FinancialAgent` to `backend/services/langgraph_agents/` alongside `ResearchAgent`, `ChatAgent`, `DirectAgent`.
- Tools: Add finance‑specific tool modules (SEC filings, market data, fundamentals, corporate actions, news). Keep modules <500 lines each.
- Background processing: Implement Celery tasks for periodic refresh of tracked tickers; reuse existing queues, progress reporting, and Postgres persistence.
- Structured outputs: Add finance‑specific Pydantic response models in `backend/models/agent_response_models.py` (or a focused module) and parse/validate everywhere.

### Agents
#### Interactive: FinancialAgent
- Responsibilities: Entity resolution (ticker→company/CIK), Q&A on filings, corporate actions (buybacks/dividends/splits), fundamentals (TTM, QoQ/YoY), market data, peer benchmarking, event studies, and scenario analysis.
- Local‑first strategy: Prefer local DB/vector store/knowledge graph; request web/premium API permission only when needed.
- Outputs: Typed `FinancialAnalysisResponse` with fields for task_status, answer, assumptions, key_metrics, sources, confidence, and data timestamps.

#### Background: Finance Tracker
- Watchlists: User‑defined tickers tracked continuously.
- Schedules: Periodic tasks (Celery beat) to refresh SEC filings, prices, corporate actions, fundamentals, earnings calendars, and finance news.
- Processing: Parse and chunk filings; embed with metadata; compute deltas vs. prior filings; update Knowledge Graph; cache derived metrics for low‑latency reads.
- Storage: Postgres tables for filings/statements/prices/actions; vector store for chunks; knowledge graph edges for relationships.

### Core Capabilities
- SEC filings ingestion: 10‑K/10‑Q/8‑K, exhibits; extraction of MD&A, risk factors, guidance, buybacks; change detection.
- Corporate actions: Buybacks, dividends, splits; summarize magnitudes, cadence, and trends.
- Fundamentals: Income, balance sheet, cash flow; TTM/YoY/QoQ; ratios (margins, ROIC, leverage, FCF yield).
- Market data: OHLCV, liquidity, volatility, moving averages, drawdowns; event windows (pre/post earnings).
- News/research: Summaries with citations, deduped across reputable sources; sentiment and topic clustering.
- Scenarios: Tariffs/FX/commodity/rates shock analysis with simple sensitivities; explicit assumptions.
- Peer/sector benchmarking: Relative valuation and factor tilts; comps tables.
- Holdings/insiders: 13F and Form 4 summaries and trends.

### Tools (New Modules)
- `backend/services/langgraph_tools/sec_filings_tools.py`
  - Search by CIK/ticker; fetch latest 10‑K/10‑Q/8‑K; section extraction (MD&A, risk factors, repurchases); citation‑ready outputs.
- `backend/services/langgraph_tools/market_data_tools.py`
  - OHLCV, indicators, volatility metrics, drawdowns, event windows.
- `backend/services/langgraph_tools/fundamentals_tools.py`
  - Statements, standardized line‑items, TTM calculations, ratios.
- `backend/services/langgraph_tools/corp_actions_tools.py`
  - Buybacks/dividends/splits; period aggregation and summaries.
- `backend/services/langgraph_tools/news_finance_tools.py`
  - Finance‑oriented news search, source prioritization, dedupe, sentiment; integrates with existing web search tools when permitted.

All tools return structured, citation‑ready objects and respect permission gating for web/premium APIs.

### Structured Outputs
- Add a `FinancialAnalysisResponse` Pydantic model, for example:
  - `task_status: Literal["complete","incomplete","permission_required","error"]`
  - `answer: str` (concise narrative)
  - `assumptions: List[str]`
  - `key_metrics: Dict[str, Any]` (e.g., TTM revenue, margin, buyback spend)
  - `sources: List[Source]` (title, url, type, snippet, timestamp)
  - `confidence: float` (0.0–1.0) and `confidence_notes: str`
  - `data_timestamp: str` (ISO) per domain (filings/prices/news)
  - `next_steps: Optional[List[str]]`

### Background Scheduling
- Periodic jobs (per watchlist/ticker):
  - SEC check (every 4–6 hours)
  - Prices/indicators (intraday or daily depending on provider limits)
  - Corporate actions (daily)
  - Fundamentals (post‑quarterly cadence)
  - Earnings calendars/estimates (daily)
  - News (hourly)
- Each job writes:
  - Raw payload snapshot (JSON or normalized tables)
  - Parsed/normalized records in Postgres
  - Vector embeddings for filings/news chunks with metadata
  - Knowledge Graph updates (company→filing→section, company→supplier/customer, etc.)

### Data Sources (Configurable via env)
- SEC EDGAR (official; RSS/CIK programmatic fetch)
- Market data/fundamentals/corporate actions: provider abstraction (e.g., Polygon.io, Alpha Vantage, Finnhub, Yahoo Finance)
- Earnings calendars and estimates
- News and reputable finance media; SearXNG + crawl for citations

### Free APIs and Environment Variables
This system should prefer free or free‑tier providers and fall back gracefully. Configure via `.env` and reference in `docker-compose.yml` under the backend service.

#### SEC Filings (No API key required)
- Provider: SEC EDGAR
- Base: `https://data.sec.gov/`
- Endpoints:
  - Company submissions: `/submissions/CIK##########.json`
  - Company facts (XBRL fundamentals): `/api/xbrl/companyfacts/CIK##########.json`
  - Filings RSS per CIK
- Headers: The SEC requires a descriptive User‑Agent and contact email.
- Env:
  - `SEC_BASE_URL=https://data.sec.gov`
  - `SEC_USER_AGENT="YourAppName (your_email@example.com)"`
  - `SEC_CONTACT_EMAIL=your_email@example.com`

#### Web Search (No key when self‑hosted)
- Provider: SearXNG (self‑hosted meta‑search)
- Env:
  - `SEARXNG_URL=http://searxng:8080` (already used by existing tools)

#### Market Data (Free tiers; API key required unless noted)
- Alpha Vantage (free tier)
  - Base: `https://www.alphavantage.co/query`
  - Env: `ALPHAVANTAGE_API_KEY=...`
  - Notes: Time series, fundamentals (overview, earnings); subject to rate limits.
- Finnhub (free tier)
  - Base: `https://finnhub.io/api/v1`
  - Env: `FINNHUB_API_KEY=...`
  - Notes: Quotes, company profiles, news; free plan limits apply.
- Financial Modeling Prep (free tier)
  - Base: `https://financialmodelingprep.com/api/v3`
  - Env: `FMP_API_KEY=...`
  - Notes: Fundamentals, calendars, and more on free tier with constraints.
- Twelve Data (free tier)
  - Base: `https://api.twelvedata.com`
  - Env: `TWELVEDATA_API_KEY=...`
  - Notes: Intraday/daily data; limited symbols/rate.
- Stooq (no key; daily prices)
  - Base: `https://stooq.com/q/d/l/`
  - Example: `?s=AAPL&i=d`
  - Env: `STOOQ_BASE_URL=https://stooq.com/q/d/l/`
  - Notes: Daily bars; symbol mapping required.

#### Macroeconomic Data (Free tier)
- FRED (Federal Reserve)
  - Base: `https://api.stlouisfed.org/fred`
  - Env: `FRED_API_KEY=...`
  - Notes: Rates, macro series; useful for scenario analysis.

#### News (Free sources)
- SearXNG + crawler (no key; already integrated via web tools)
  - Uses `SEARXNG_URL` and internal crawl tools for citations.
- GDELT (no key for many endpoints)
  - Base: `https://api.gdeltproject.org/api/v2/`
  - Env: `GDELT_BASE_URL=https://api.gdeltproject.org/api/v2`
  - Notes: Global news metadata; filter required for finance relevance.

#### Optional/Experimental
- Yahoo Finance (unofficial endpoints; no key)
  - Env toggles: `YF_ENABLE_UNOFFICIAL=false`, `YF_USER_AGENT="..."`
  - Notes: Subject to change/ToS; prefer official/free‑tier providers above.

#### Provider Selection and Fallbacks
- Configure preferences:
  - `MARKET_DATA_PROVIDER=alpha_vantage` (options: `alpha_vantage|stooq|finnhub|fmp|twelve_data`)
  - `NEWS_PROVIDER=searxng` (options: `searxng|gdelt`)
- The FinancialAgent and background tasks should attempt providers in configured order and degrade gracefully when rate‑limited or unavailable.

#### Example .env entries
```dotenv
# SEC EDGAR
SEC_BASE_URL=https://data.sec.gov
SEC_USER_AGENT=PlatoApp (admin@example.com)
SEC_CONTACT_EMAIL=admin@example.com

# Search / News
SEARXNG_URL=http://searxng:8080
GDELT_BASE_URL=https://api.gdeltproject.org/api/v2

# Market Data Providers (free tiers)
ALPHAVANTAGE_API_KEY=
FINNHUB_API_KEY=
FMP_API_KEY=
TWELVEDATA_API_KEY=
STOOQ_BASE_URL=https://stooq.com/q/d/l/
FRED_API_KEY=

# Provider preferences
MARKET_DATA_PROVIDER=alpha_vantage
NEWS_PROVIDER=searxng

# Optional Yahoo Finance (unofficial)
YF_ENABLE_UNOFFICIAL=false
YF_USER_AGENT=PlatoApp/1.0
```

#### docker-compose.yml mapping (backend service)
```yaml
services:
  backend:
    environment:
      - SEC_BASE_URL
      - SEC_USER_AGENT
      - SEC_CONTACT_EMAIL
      - SEARXNG_URL
      - GDELT_BASE_URL
      - ALPHAVANTAGE_API_KEY
      - FINNHUB_API_KEY
      - FMP_API_KEY
      - TWELVEDATA_API_KEY
      - STOOQ_BASE_URL
      - FRED_API_KEY
      - MARKET_DATA_PROVIDER
      - NEWS_PROVIDER
      - YF_ENABLE_UNOFFICIAL
      - YF_USER_AGENT
```

Notes on rate limits: Free tiers often impose per‑minute and daily quotas. Implement respectful rate limiting, retries with backoff, and caching in background tasks. Always consult current provider documentation before production use.

### Example Queries and Approach
- Impact of tariffs on Ford: Identify exposure from filings; gather recent tariff news; compute historical price reactions; produce assumptions, sensitivity, and confidence; cite sources.
- Rundown on Apple buybacks: Aggregate authorization and repurchase activity from 10‑K/10‑Q and 8‑K; compute trend and shares reduced; relate to FCF; cite.
- Should I buy Tesla: Balanced synopsis across fundamentals, valuation, news flow, technical context; include risks, “not investment advice” note, data recency, and confidence.

### Additional Feature Ideas (Top 10)
1) Portfolio what‑if shocks (tariffs/rates/oil/FX) per ticker and aggregate.
2) Earnings preview briefs (consensus vs. trend and guidance; alt signals if available).
3) Event study toolkit (post‑earnings drift, buyback announcements, product launches).
4) Insider/13F heatmaps (alignment/divergence with institutional flows).
5) Supply‑chain graph (key suppliers/customers) with shock propagation.
6) Factor exposure readout (value/momentum/quality/size; rolling betas).
7) Valuation sandbox (quick DCF/comps with adjustable levers; scenario tables).
8) Material‑changes detector (risk factor diffs across filings).
9) Regulatory radar (dockets tied to tickers with status and citations).
10) Smart morning brief for watchlists (filings, price moves, key news, alerts).

### Phased Delivery
- Phase 1: FinancialAgent + SEC filings + prices/corporate actions + permission‑aware web citations + background watchlist refresh.
- Phase 2: Fundamentals platform + buyback/dividend analytics + peer comps + alerting.
- Phase 3: Scenario analysis, factor exposures, supply‑chain graph, valuation sandbox.

### Compliance & Safety
- Always include “not investment advice,” explicit assumptions, and data timestamps.
- Maintain permission gates for web/premium APIs; rate limits/backoff.
- Provide confidence scores and precise citations.

### Financial Advisor Sub‑Agents (Interactive and Background)
The Financial Advisor may coordinate multiple specialized sub‑agents. Several operate as background cavalry beyond simple “tracked companies” monitoring.

- Market Event Sentinel (background)
  - Purpose: Track earnings, 8‑K, guidance updates, notable finance news; emit material‑change alerts into context.
  - Runs: Scheduled checks (intra‑day for events/news; daily summary). Writes alerts to Postgres and updates vector store.

- SEC Diff Analyzer (background)
  - Purpose: Redline MD&A/risk factors/repurchases across new vs. prior 10‑K/10‑Q; quantify language deltas and highlight material changes.
  - Runs: On filing arrival; produces structured diff artifacts, citations, and embeddings per section.

- Valuation Sandbox Agent (interactive)
  - Purpose: Quick comps and DCF with user‑adjustable levers; outputs scenario tables, assumptions, and citations.
  - Invocation: User queries or orchestrator chaining from FA.

- Scenario Simulator (interactive)
  - Purpose: Tariff/FX/rate/commodity shocks; sensitivity readouts with explicit assumptions and confidence.
  - Invocation: User scenario questions or orchestrator follow‑ups.

- Portfolio Risk Scout (interactive/background)
  - Purpose: Factor exposures, correlation clusters, drawdowns, VaR‑style summaries; regime/volatility shift alerts.
  - Runs: Daily/weekly background scans; interactive deep‑dives on demand.

- Insider & 13F Tracker (background)
  - Purpose: Aggregate Form 4 and 13F flows; flag unusual activity and alignment/divergence with theses.
  - Runs: Daily for Form 4; quarterly for 13F refresh with diffs.

- Supply‑Chain Mapper (background)
  - Purpose: Build/refresh supplier–customer graph from filings/news; enable shock propagation queries for scenarios.
  - Runs: Weekly or on material discovery; maintains KG edges and provenance.

- Finance Briefing Herald (background)
  - Purpose: Morning/weekly briefs for watchlists: filings, price moves, key news, alerts.
  - Runs: Scheduled digests; posts summaries with links and timestamps.

- Visualization Cartographer (interactive)
  - Purpose: Compose charts/tables/event studies from stored metrics and analysis outputs; export‑ready artifacts.
  - Invocation: On request or orchestrator post‑analysis step.

- Source Verification Provost (finance‑tuned; interactive/background)
  - Purpose: Cross‑check facts, validate citations, de‑duplicate sources, score reliability for finance content.
  - Runs: Inline during interactive responses and as a background audit pass for key artifacts.

Integration notes
- Each sub‑agent exposes Pydantic‑validated outputs and can be invoked as LangGraph nodes or Celery background tasks.
- Background agents are configured via env and scheduled in Celery beat; results feed Postgres, vector store, and Knowledge Graph.
- FA routes user intents to the proper sub‑agent and synthesizes the final, cited response.


