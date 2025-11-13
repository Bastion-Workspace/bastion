"""
Substack Research Helper - Roosevelt's Article Enhancement Research Engine

Handles research question generation, local/web search, and findings synthesis
for the Substack Agent's research-augmented writing workflow.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from models.substack_models import ArticleResearchPlan, ResearchQuestion, ResearchFinding

logger = logging.getLogger(__name__)


class SubstackResearchHelper:
    """Helper for research-augmented article writing."""

    def __init__(self, chat_service):
        self.chat_service = chat_service

    async def generate_research_plan(
        self,
        user_query: str,
        article_texts: List[str],
        tweets_text: Optional[str],
        background_text: Optional[str],
        model_name: str
    ) -> ArticleResearchPlan:
        """
        Analyze provided content and user query to generate a research plan.
        Returns plan with needs_research=False if no questions needed.
        """
        try:
            logger.info("🔍 RESEARCH PLAN: Analyzing content to determine research needs...")

            # Build context summary
            content_summary = self._build_content_summary(
                article_texts, tweets_text, background_text
            )

            # System prompt for research planning
            system_prompt = (
                "You are a research planning assistant for article writing. "
                "Analyze the provided content and user request to determine if additional research is needed.\n\n"
                "CRITICAL DECISION RULES:\n"
                "- If source materials are comprehensive and sufficient, set needs_research=FALSE with empty questions list\n"
                "- Only generate questions if there are genuine knowledge gaps\n"
                "- Maximum 5 questions, prioritize most important\n"
                "- Prefer 'local' search (our document database) for historical/established facts\n"
                "- Use 'web' search only for current events, statistics, or very recent information\n"
                "- Use 'both' when comprehensive coverage is needed\n\n"
                "EXAMPLES OF WHEN TO SKIP RESEARCH:\n"
                "- Opinion pieces based on provided tweets/articles (user's analysis doesn't need more sources)\n"
                "- Commentary on specific provided articles (content is self-contained)\n"
                "- Creative writing or narrative pieces (research not relevant)\n"
                "- User explicitly says 'just write from these sources'\n\n"
                "EXAMPLES OF WHEN TO RESEARCH:\n"
                "- User asks for 'comprehensive analysis' or 'deep dive'\n"
                "- Topics mention needing 'context' or 'background'\n"
                "- References to statistics, data, or 'latest information'\n"
                "- Comparative analysis needing additional cases/examples\n"
            )

            task_prompt = (
                f"=== USER REQUEST ===\n{user_query}\n\n"
                f"=== PROVIDED CONTENT ===\n{content_summary}\n\n"
                "=== YOUR TASK ===\n"
                "Analyze whether additional research would enhance this article.\n\n"
                "RESPOND WITH VALID JSON ONLY:\n"
                "{\n"
                '  "needs_research": true/false,\n'
                '  "research_questions": [],  // Empty if needs_research=false\n'
                '  "estimated_depth": "none"|"light"|"moderate"|"deep",\n'
                '  "web_search_needed": true/false,\n'
                '  "rationale": "Brief explanation of decision"\n'
                "}\n\n"
                "If needs_research=true, include questions array:\n"
                "{\n"
                '  "question": "What specific information do we need?",\n'
                '  "priority": "high"|"medium"|"low",\n'
                '  "search_type": "local"|"web"|"both",\n'
                '  "rationale": "Why this matters"\n'
                "}"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt}
            ]

            response = await self.chat_service.openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
            )

            content = response.choices[0].message.content or "{}"
            logger.info(f"🔍 RESEARCH PLAN: LLM response: {content[:300]}...")

            # Parse JSON
            text = content.strip()
            if '```json' in text:
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            elif '```' in text:
                text = text.replace('```', '').strip()

            data = json.loads(text)
            plan = ArticleResearchPlan(**data)

            logger.info(
                f"✅ RESEARCH PLAN: needs_research={plan.needs_research}, "
                f"questions={len(plan.research_questions)}, depth={plan.estimated_depth}"
            )

            return plan

        except Exception as e:
            logger.error(f"❌ Research plan generation failed: {e}")
            # Return safe default: no research needed
            return ArticleResearchPlan(
                needs_research=False,
                research_questions=[],
                estimated_depth="none",
                web_search_needed=False,
                rationale="Error during planning, proceeding without research"
            )

    def _build_content_summary(
        self,
        article_texts: List[str],
        tweets_text: Optional[str],
        background_text: Optional[str]
    ) -> str:
        """Build a compact summary of provided content."""
        parts = []

        if article_texts:
            for i, art in enumerate(article_texts, 1):
                # Take first 500 chars of each article
                preview = art[:500] if art else ""
                parts.append(f"ARTICLE {i} (preview): {preview}...")

        if tweets_text:
            parts.append(f"TWEETS: {tweets_text[:300]}...")

        if background_text:
            parts.append(f"BACKGROUND: {background_text[:300]}...")

        return "\n\n".join(parts) if parts else "No source content provided."

    async def execute_research(
        self,
        research_plan: ArticleResearchPlan,
        has_web_permission: bool
    ) -> List[ResearchFinding]:
        """
        Execute research based on the plan.
        Searches local documents, and web if permission granted.
        """
        findings = []

        for question_obj in research_plan.research_questions:
            finding = await self._research_question(
                question_obj,
                has_web_permission
            )
            findings.append(finding)

        return findings

    async def _research_question(
        self,
        question_obj: ResearchQuestion,
        has_web_permission: bool
    ) -> ResearchFinding:
        """Research a single question using available search methods."""
        try:
            logger.info(f"🔍 RESEARCHING: {question_obj.question}")

            # Always try local search first (if type is local or both)
            local_results = []
            if question_obj.search_type in ["local", "both"]:
                local_results = await self._search_local(question_obj.question)

            # Try web search if needed and permitted
            web_results = []
            if question_obj.search_type in ["web", "both"] and has_web_permission:
                web_results = await self._search_web(question_obj.question)

            # Combine and format results
            all_results = local_results + web_results
            summary = self._format_findings(all_results, question_obj.question)

            # Determine which search type was actually used
            search_type_used = "local"
            if web_results and not local_results:
                search_type_used = "web"
            elif web_results and local_results:
                search_type_used = "both"

            finding = ResearchFinding(
                question=question_obj.question,
                sources_found=len(all_results),
                summary=summary,
                citations=[r.get("source", "Unknown") for r in all_results[:5]],
                search_type_used=search_type_used
            )

            logger.info(
                f"✅ RESEARCH COMPLETE: Found {len(all_results)} sources for: {question_obj.question[:50]}..."
            )

            return finding

        except Exception as e:
            logger.error(f"❌ Research failed for question '{question_obj.question}': {e}")
            return ResearchFinding(
                question=question_obj.question,
                sources_found=0,
                summary=f"Research error: {str(e)}",
                citations=[],
                search_type_used="local"
            )

    async def _search_local(self, query: str) -> List[Dict[str, Any]]:
        """Search local document database."""
        try:
            # Import and use search tool directly (simpler than going through registry)
            from services.langgraph_tools.unified_search_tools import search_local
            
            # Execute search
            result = await search_local(query=query, limit=5)

            # Parse results (format depends on tool implementation)
            if isinstance(result, str):
                # Simple string result
                return [{"content": result, "source": "Local Documents"}]
            elif isinstance(result, dict):
                # Structured result
                return [result]
            elif isinstance(result, list):
                # List of results
                return result
            else:
                return []

        except Exception as e:
            logger.warning(f"Local search failed: {e}")
            return []

    async def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """Search web for information."""
        try:
            # Import and use search tool directly
            from services.langgraph_tools.web_content_tools import search_web
            
            # Execute search
            result = await search_web(query=query, max_results=3)

            # Parse results
            if isinstance(result, str):
                return [{"content": result, "source": "Web Search"}]
            elif isinstance(result, dict):
                return [result]
            elif isinstance(result, list):
                return result
            else:
                return []

        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []

    def _format_findings(self, results: List[Dict[str, Any]], question: str) -> str:
        """Format research findings into a readable summary."""
        if not results:
            return f"No specific information found for: {question}"

        # Extract content from results
        content_pieces = []
        for r in results[:5]:  # Limit to top 5
            if isinstance(r, dict):
                content = r.get("content") or r.get("text") or r.get("summary") or ""
                if content:
                    # Take first 300 chars
                    content_pieces.append(content[:300])

        if not content_pieces:
            return "Sources found but content unavailable"

        # Combine into summary
        summary = " | ".join(content_pieces)
        return summary[:1000]  # Limit total length

    def format_research_for_article_prompt(
        self,
        findings: List[ResearchFinding]
    ) -> str:
        """Format research findings for inclusion in article generation prompt."""
        if not findings:
            return ""

        parts = ["=== RESEARCH FINDINGS ===\n"]

        for i, finding in enumerate(findings, 1):
            parts.append(f"QUESTION {i}: {finding.question}")
            parts.append(f"SOURCES FOUND: {finding.sources_found}")
            parts.append(f"KEY FINDINGS: {finding.summary}")
            if finding.citations:
                parts.append(f"CITATIONS: {', '.join(finding.citations[:3])}")
            parts.append("")  # Blank line

        return "\n".join(parts)

