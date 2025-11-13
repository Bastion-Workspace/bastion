# SearXNG Configuration

This directory contains the configuration for SearXNG, a privacy-focused metasearch engine that aggregates results from multiple search engines.

## What is SearXNG?

SearXNG is a free internet metasearch engine which aggregates results from various search engines and databases. It's designed to be self-hosted and privacy-focused.

## Configuration

- `settings.yml` - Main SearXNG configuration file
- Configured to use multiple search engines: **Google, Bing, DuckDuckGo, and Startpage**
- **Google autocomplete** enabled for better search suggestions
- Optimized for API usage with rate limiting disabled
- Minimal logging for production use

## Web Search Integration

The Codex Knowledge Base uses SearXNG for all web search functionality via the MCP (Model Context Protocol) web search tool. This provides:

- **Multiple search engines**: Google, Bing, DuckDuckGo, and Startpage for comprehensive coverage
- **Better search quality**: Google's advanced algorithms combined with privacy-focused alternatives
- **Enhanced suggestions**: Google autocomplete for better query completion
- **Privacy**: No tracking or data collection
- **Reliability**: Self-hosted, no external API dependencies
- **Performance**: Local deployment, fast response times

## Service Details

- **URL**: http://localhost:8080 (internal: http://searxng:8080)
- **API Format**: JSON responses for programmatic access
- **Health Check**: Automatic monitoring via Docker Compose
- **Engines**: Google, Bing, DuckDuckGo, Startpage (weighted for relevance)
- **Autocomplete**: Google-powered search suggestions

## Docker Integration

SearXNG runs as a containerized service in the Docker Compose stack and is automatically configured when you run:

```bash
docker compose up --build
```

The web search functionality will automatically use SearXNG with all enabled search engines for comprehensive results. 