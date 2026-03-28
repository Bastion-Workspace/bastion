"""
Research-oriented route definitions (dispatch via CustomAgentRunner).

Routes: research (deep), knowledge_builder, security_analysis, site_crawl,
website_crawler.
"""

from orchestrator.routes.route_schema import EngineType, Route

RESEARCH_ROUTES = [
    Route(
        name="research",
        description=(
            "Factual and exploratory questions using Agent Factory tools: local document search, "
            "web search, query enhancement, conversation cache, segments, crawl, and image search. "
            "Use for how-to and what-is questions, and for 'find me' / 'show me' queries about stored content."
        ),
        engine=EngineType.CUSTOM_AGENT,
        domains=["general", "research", "information", "management"],
        actions=["query", "analysis"],
        keywords=[
            "research", "find information", "tell me about", "investigate",
            "how do i", "how do I", "how to", "how can i", "how can I", "what is", "what are",
            "anticipate", "predict", "forecast", "effects", "impact", "consequences",
            "would be", "will be", "might be", "could be", "likely to",
            "analyze", "analysis", "explain", "describe",
            "economic", "policy", "legislation", "regulation", "tariff", "tax",
            "do some research", "do research", "can you research", "please research",
            "find me a picture", "find me a pic", "find me an image", "find me a photo",
            "show me a picture", "show me a pic", "show me an image", "show me a photo",
            "do we have", "in our collection", "in my collection", "comic", "comics",
            "photo", "photos", "picture", "pictures", "image", "images",
        ],
        priority=80,
        override_continuity=True,
        tools=[
            "search_documents_tool",
            "search_web_tool",
            "enhance_query_tool",
            "search_conversation_cache_tool",
            "search_segments_across_documents_tool",
            "crawl_web_content_tool",
            "get_document_content_tool",
            "get_document_metadata_tool",
            "search_images_tool",
        ],
        tool_io_map={
            "search_documents_tool": "search_documents",
            "search_web_tool": "search_web",
            "enhance_query_tool": "enhance_query",
            "search_conversation_cache_tool": "search_conversation_cache",
            "search_segments_across_documents_tool": "search_segments_across_documents",
            "crawl_web_content_tool": "crawl_web_content",
            "get_document_content_tool": "get_document_content",
            "get_document_metadata_tool": "get_document_metadata",
            "search_images_tool": "search_images",
        },
    ),
    Route(
        name="knowledge_builder",
        description=(
            "Fact-check, verify claims, distill knowledge, and build research documents using "
            "document and web search tools (not general open-ended lookup; use the research route for that)."
        ),
        engine=EngineType.CUSTOM_AGENT,
        domains=["research", "knowledge", "information", "truth"],
        actions=["query", "analysis"],
        keywords=[
            "distill", "distill knowledge", "build knowledge",
            "compile knowledge", "research document", "investigate",
            "find the truth", "verify claims", "fact check",
            "cross-reference", "truth investigation",
        ],
        priority=75,
        tools=["search_documents_tool", "search_web_tool", "enhance_query_tool"],
        tool_io_map={
            "search_documents_tool": "search_documents",
            "search_web_tool": "search_web",
            "enhance_query_tool": "enhance_query",
        },
    ),
    Route(
        name="security_analysis",
        description="Security and vulnerability scanning for URLs and websites.",
        engine=EngineType.CUSTOM_AGENT,
        domains=["general", "security", "web"],
        actions=["analysis", "query"],
        keywords=[
            "security scan", "vulnerability scan", "security analysis", "check for vulnerabilities",
            "security audit", "pen test", "security assessment", "exposed files", "security headers",
            "scan for vulnerabilities", "security check", "website security",
        ],
        priority=90,
        tools=["search_web_tool", "crawl_web_content_tool"],
        tool_io_map={"search_web_tool": "search_web", "crawl_web_content_tool": "crawl_web_content"},
    ),
    Route(
        name="site_crawl",
        description="Crawl a single site or domain and extract content (one-off).",
        engine=EngineType.CUSTOM_AGENT,
        domains=["research", "web", "information"],
        actions=["query"],
        keywords=["crawl site", "crawl website", "site crawl", "domain crawl", "crawl domain"],
        priority=90,
        tools=["crawl_web_content_tool"],
        tool_io_map={"crawl_web_content_tool": "crawl_web_content"},
    ),
    Route(
        name="website_crawler",
        description="Ingest a URL or whole website with recursive crawl (save/process content).",
        engine=EngineType.CUSTOM_AGENT,
        domains=["general", "web", "management"],
        actions=["management", "query"],
        keywords=["crawl", "capture", "ingest", "scrape", "download website", "url"],
        priority=92,
        tools=["crawl_web_content_tool", "search_web_tool"],
        tool_io_map={"crawl_web_content_tool": "crawl_web_content", "search_web_tool": "search_web"},
    ),
]
