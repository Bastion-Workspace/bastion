"""
Research engine skill definitions.

Skills: research (deep), knowledge_builder, security_analysis, site_crawl,
website_crawler.
"""

from orchestrator.skills.skill_schema import EngineType, Skill

RESEARCH_SKILLS = [
    Skill(
        name="research",
        description=(
            "Multi-round research with web and local search, gap analysis, and synthesis. "
            "Use for factual and how-to questions: 'how do I', 'how to', 'what is', 'how can I'. "
            "Searches the user's local document and image collections (comics, photos, artwork). "
            "Use for 'do we have', 'find me', 'show me' queries about stored content."
        ),
        engine=EngineType.RESEARCH,
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
        tools=["search_documents_tool", "search_web_tool", "expand_query_tool", "search_conversation_cache_tool"],
        subgraphs=["research_workflow", "gap_analysis", "web_research", "assessment", "data_formatting", "visualization"],
    ),
    Skill(
        name="knowledge_builder",
        description="Fact-check, verify claims, distill knowledge, and build research documents (not general web lookup; use research for that).",
        engine=EngineType.RESEARCH,
        domains=["research", "knowledge", "information", "truth"],
        actions=["query", "analysis"],
        keywords=[
            "distill", "distill knowledge", "build knowledge",
            "compile knowledge", "research document", "investigate",
            "find the truth", "verify claims", "fact check",
            "cross-reference", "truth investigation",
        ],
        priority=75,
        tools=["search_documents_tool", "search_web_tool", "expand_query_tool"],
        subgraphs=["fact_verification", "knowledge_document"],
    ),
    Skill(
        name="security_analysis",
        description="Security and vulnerability scanning for URLs and websites.",
        engine=EngineType.RESEARCH,
        domains=["general", "security", "web"],
        actions=["analysis", "query"],
        keywords=[
            "security scan", "vulnerability scan", "security analysis", "check for vulnerabilities",
            "security audit", "pen test", "security assessment", "exposed files", "security headers",
            "scan for vulnerabilities", "security check", "website security",
        ],
        priority=90,
        tools=["search_web_tool", "crawl_web_content_tool"],
        subgraphs=[],
    ),
    Skill(
        name="site_crawl",
        description="Crawl a single site or domain and extract content (one-off).",
        engine=EngineType.RESEARCH,
        domains=["research", "web", "information"],
        actions=["query"],
        keywords=["crawl site", "crawl website", "site crawl", "domain crawl", "crawl domain"],
        priority=90,
        tools=["crawl_web_content_tool"],
        subgraphs=[],
    ),
    Skill(
        name="website_crawler",
        description="Ingest a URL or whole website with recursive crawl (save/process content).",
        engine=EngineType.RESEARCH,
        domains=["general", "web", "management"],
        actions=["management", "query"],
        keywords=["crawl", "capture", "ingest", "scrape", "download website", "url"],
        priority=92,
        tools=["crawl_web_content_tool", "search_web_tool"],
        subgraphs=[],
    ),
]
