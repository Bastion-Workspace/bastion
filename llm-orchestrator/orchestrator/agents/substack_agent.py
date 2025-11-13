"""
Substack Agent
LangGraph agent for long-form article and tweet generation
Generates publication-ready content with research integration
"""

import logging
import re
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_core.messages import AIMessage

from orchestrator.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SubstackAgent(BaseAgent):
    """
    Substack Agent for long-form article and tweet generation
    
    Synthesizes multiple sources into cohesive, engaging content
    with optional research augmentation
    """
    
    def __init__(self):
        super().__init__("substack_agent")
        self._grpc_client = None
    
    async def _get_grpc_client(self):
        """Get or create gRPC client for backend tools"""
        if self._grpc_client is None:
            from orchestrator.clients.backend_tool_client import get_backend_tool_client
            self._grpc_client = await get_backend_tool_client()
        return self._grpc_client
    
    def _build_system_prompt(self, persona: Optional[Dict[str, Any]] = None) -> str:
        """Build article writing system prompt"""
        base = (
            "You are a professional long-form article writer specializing in blog posts and Substack publications. "
            "Your task is to synthesize multiple source materials into a cohesive, engaging article.\n\n"
            "=== ARTICLE WRITING PRINCIPLES ===\n"
            "STRUCTURE:\n"
            "- Compelling title that captures the essence\n"
            "- Strong opening hook that draws readers in\n"
            "- Clear thesis or central argument\n"
            "- Well-organized body sections with logical flow\n"
            "- Smooth transitions between ideas\n"
            "- Satisfying conclusion that reinforces main points\n\n"
            "STYLE:\n"
            "- Active voice and strong verbs\n"
            "- Varied sentence structure for rhythm\n"
            "- Concrete examples and specific details\n"
            "- Direct quotes from source material when impactful\n"
            "- Clear, accessible language (avoid jargon unless necessary)\n"
            "- Natural conversational tone while maintaining professionalism\n\n"
            "CONTENT INTEGRATION:\n"
            "- Weave multiple sources together seamlessly\n"
            "- Compare and contrast different perspectives\n"
            "- Provide context and analysis, not just summary\n"
            "- Cite sources naturally in text (e.g., 'According to...', 'As reported in...')\n"
            "- Reference tweet authors and content explicitly when using social media sources\n"
            "- Build original arguments on top of source material\n\n"
            "MARKDOWN FORMATTING:\n"
            "- Use ## for main section headers\n"
            "- Use ### for subsection headers\n"
            "- **Bold** for emphasis on key points\n"
            "- *Italics* for subtle emphasis or titles\n"
            "- > Blockquotes for direct quotations from sources\n"
            "- Bullet points or numbered lists for clarity when appropriate\n"
            "- Em-dashes (—) for parenthetical thoughts\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "- Target length: 2000-5000 words (adjustable per request)\n"
            "- Ground content in provided sources—cite specific names, quotes, statistics\n"
            "- Maintain consistent voice and perspective throughout\n"
            "- Provide original insight and analysis, not just summarization\n"
            "- Make it publishable—polished, professional, engaging\n"
        )
        
        if persona:
            persona_name = persona.get("ai_name", "")
            persona_style = persona.get("persona_style", "")
            if persona_name or persona_style:
                base += f"\n\nPERSONA: Write as {persona_name or 'assistant'} with {persona_style or 'professional'} style."
        
        return base
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process article generation request"""
        try:
            logger.info("📝 Substack Agent: Starting article generation")
            
            # Extract state components
            messages = state.get("messages", [])
            user_id = state.get("user_id")
            shared_memory = state.get("shared_memory", {})
            active_editor = shared_memory.get("active_editor", {})
            
            # Check if editor is substack/blog type
            frontmatter = active_editor.get("frontmatter", {})
            doc_type = str(frontmatter.get("type", "")).strip().lower()
            
            if doc_type not in ["substack", "blog"]:
                logger.info(f"📝 Substack agent skipping: editor type is '{doc_type}', not 'substack' or 'blog'")
                return self._create_response(
                    success=True,
                    response="Active editor is not a substack/blog document. Substack agent requires editor with type='substack' or type='blog'.",
                    skipped=True
                )
            
            # Get user message and editor content
            latest_message = messages[-1] if messages else None
            user_message = latest_message.content if hasattr(latest_message, 'content') else ""
            editor_content = active_editor.get("content", "")
            
            # Detect tweet mode
            user_message_lower = user_message.lower()
            tweet_mode = any(keyword in user_message_lower for keyword in [
                'tweet', 'twitter', 'x post', 'short post', 'social media post',
                'tweet-sized', 'tweet size', 'short-form', 'brief', 'concise',
                '280 character', 'social post'
            ])
            
            if tweet_mode:
                logger.info("🐦 Tweet mode activated")
            
            # Extract request parameters
            target_length = int(frontmatter.get("target_length_words", 2500))
            tone = str(frontmatter.get("tone", "conversational")).lower()
            style = str(frontmatter.get("style", "commentary")).lower()
            
            # Extract structured sections
            persona_text, background_text, articles, tweets_text = self._extract_sections(
                editor_content, user_message
            )
            
            # Fetch URL if provided
            fetched_content = await self._fetch_url_content(user_message)
            if fetched_content:
                articles.append(fetched_content)
            
            # Build content block
            content_block = self._build_content_block(
                persona_text, background_text, articles, tweets_text
            )
            
            # Build system prompt
            persona = state.get("persona")
            system_prompt = self._build_system_prompt(persona)
            
            # Build task instructions
            task_block = self._build_task_block(
                user_message, target_length, tone, style, tweet_mode, frontmatter
            )
            
            # Execute LLM
            chat_service = await self._get_chat_service()
            
            messages_to_send = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Current Date/Time: {datetime.now().isoformat()}"},
            ]
            
            if content_block:
                messages_to_send.append({"role": "user", "content": content_block})
            messages_to_send.append({"role": "user", "content": task_block})
            
            logger.info(f"📝 Generating article with {len(messages_to_send)} messages")
            
            response = await chat_service.openai_client.chat.completions.create(
                model=chat_service.model,
                messages=messages_to_send,
                temperature=0.4
            )
            
            content = response.choices[0].message.content or "{}"
            
            # Parse structured response
            article_text, metadata = self._parse_response(content, tweet_mode)
            
            logger.info("✅ Substack Agent: Article generation complete")
            
            return self._create_response(
                success=True,
                response=article_text,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"❌ Substack Agent ERROR: {e}")
            return self._create_error_result(f"Article generation failed: {str(e)}")
    
    def _extract_sections(
        self, 
        editor_content: str, 
        user_message: str
    ) -> tuple:
        """Extract structured sections from content"""
        try:
            def extract_section(title: str, text: str) -> Optional[str]:
                """Extract ## Title or Title: sections"""
                lines = text.split('\n')
                start = None
                
                # Try markdown heading: ## Title
                pattern_md = rf"^##\s*{title}\s*$"
                for idx, line in enumerate(lines):
                    if re.match(pattern_md, line.strip(), flags=re.IGNORECASE):
                        start = idx + 1
                        break
                
                # Try simple heading: Title:
                if start is None:
                    pattern_simple = rf"^{title}\s*:\s*$"
                    for idx, line in enumerate(lines):
                        if re.match(pattern_simple, line.strip(), flags=re.IGNORECASE):
                            start = idx + 1
                            break
                
                if start is None:
                    return None
                
                # Capture until next heading
                collected = []
                for line in lines[start:]:
                    stripped = line.strip()
                    if stripped.startswith('## ') or re.match(r'^(background|article|persona|tweet)\s*(\d+)?:', stripped, re.IGNORECASE):
                        break
                    collected.append(line)
                
                body = '\n'.join(collected).strip()
                return body or None
            
            # Extract from editor first, then user message
            persona_text = extract_section('Persona', editor_content) or extract_section('Persona', user_message)
            background_text = extract_section('Background', editor_content) or extract_section('Background', user_message)
            tweets_text = extract_section('Tweet', editor_content) or extract_section('Tweets', editor_content) or \
                         extract_section('Tweet', user_message) or extract_section('Tweets', user_message)
            
            # Extract articles
            articles = []
            for i in range(1, 4):
                article = extract_section(f'Article {i}', editor_content) or extract_section(f'Article {i}', user_message)
                if article:
                    articles.append(article)
            
            # Also try generic "Article"
            generic_article = extract_section('Article', editor_content) or extract_section('Article', user_message)
            if generic_article and generic_article not in articles:
                articles.insert(0, generic_article)
            
            return persona_text, background_text, articles, tweets_text
            
        except Exception as e:
            logger.warning(f"Section extraction failed: {e}")
            return None, None, [], None
    
    async def _fetch_url_content(self, user_message: str) -> Optional[str]:
        """Fetch content from URL if present in message"""
        try:
            url_match = re.search(r'https?://[^\s)>\"]+', user_message)
            if not url_match:
                return None
            
            url = url_match.group(0)
            logger.info(f"📝 Fetching URL: {url}")
            
            grpc_client = await self._get_grpc_client()
            
            # Use web search to get content (we could also add a crawl RPC if needed)
            result = await grpc_client.search_web(query=url, max_results=1)
            
            if result.get("success") and result.get("results"):
                # Return first result's content
                return result["results"][0].get("snippet", "")
            
            return None
            
        except Exception as e:
            logger.warning(f"URL fetch failed: {e}")
            return None
    
    def _build_content_block(
        self,
        persona_text: Optional[str],
        background_text: Optional[str],
        articles: List[str],
        tweets_text: Optional[str]
    ) -> str:
        """Build content block for LLM"""
        sections = []
        
        if persona_text:
            sections.append("=== PERSONA ===\n" + persona_text)
        if background_text:
            sections.append("=== BACKGROUND ===\n" + background_text)
        
        for i, article in enumerate(articles, 1):
            sections.append(f"=== ARTICLE {i} ===\n" + article)
        
        if tweets_text:
            sections.append("=== TWEETS ===\n" + tweets_text)
        
        return "\n\n".join(sections)
    
    def _build_task_block(
        self,
        user_message: str,
        target_length: int,
        tone: str,
        style: str,
        tweet_mode: bool,
        frontmatter: Dict[str, Any]
    ) -> str:
        """Build task instruction block"""
        if tweet_mode:
            return (
                "=== TWEET-SIZED CONTENT INSTRUCTIONS ===\n\n"
                "Generate SHORT-FORM social media content (tweet/X post).\n\n"
                "CRITICAL REQUIREMENTS:\n"
                "- Maximum 280 characters (strict limit for single tweet)\n"
                "- Can generate multiple tweets if needed (thread format)\n"
                "- Punchy, attention-grabbing opening\n"
                "- One clear message or insight per tweet\n\n"
                "RESPOND WITH JSON:\n"
                "{\n"
                '  "task_status": "complete",\n'
                '  "article_text": "Your tweet text here",\n'
                '  "metadata": {"word_count": 280, "reading_time_minutes": 0, "section_count": 1}\n'
                "}\n\n"
                f"User instruction: {user_message}\n"
            )
        else:
            return (
                "=== ARTICLE WRITING INSTRUCTIONS ===\n\n"
                "Generate a long-form article synthesizing provided sources.\n\n"
                f"Target length: {target_length} words\n"
                f"Tone: {tone}\n"
                f"Style: {style}\n\n"
                "FORMAT: Start with # Title, use ## for sections, ### for subsections\n"
                "CITE SOURCES: Reference articles/tweets naturally in text\n\n"
                "RESPOND WITH JSON:\n"
                "{\n"
                '  "task_status": "complete",\n'
                '  "article_text": "# Title\\n\\n## Section...\\n\\nYour article markdown here",\n'
                '  "metadata": {"word_count": 2500, "reading_time_minutes": 10, "section_count": 5}\n'
                "}\n\n"
                f"User instruction: {user_message}\n"
            )
    
    def _parse_response(self, content: str, tweet_mode: bool) -> tuple:
        """Parse LLM JSON response"""
        try:
            # Strip code fences
            text = content.strip()
            if '```json' in text:
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            elif '```' in text:
                text = text.replace('```', '').strip()
            
            data = json.loads(text)
            
            article_text = data.get("article_text", "")
            metadata = data.get("metadata", {})
            
            # Ensure metadata has required fields
            if not isinstance(metadata, dict):
                metadata = {}
            metadata.setdefault("word_count", len(article_text.split()))
            metadata.setdefault("reading_time_minutes", max(1, metadata.get("word_count", 0) // 200))
            metadata.setdefault("section_count", article_text.count("##"))
            
            return article_text, metadata
            
        except Exception as e:
            logger.warning(f"JSON parse failed, using fallback: {e}")
            
            # Fallback: treat as raw text
            if tweet_mode and content.strip() and not content.strip().startswith('{'):
                return content.strip(), {
                    "word_count": len(content),
                    "reading_time_minutes": 0,
                    "section_count": 1
                }
            else:
                error_text = f"Failed to parse response: {e}\n\nRaw content: {content[:500]}..."
                return error_text, {
                    "word_count": 0,
                    "reading_time_minutes": 0,
                    "section_count": 0
                }
    
    def _create_response(
        self,
        success: bool,
        response: str,
        metadata: Dict[str, Any] = None,
        skipped: bool = False
    ) -> Dict[str, Any]:
        """Create standardized response"""
        return {
            "messages": [AIMessage(content=response)],
            "agent_results": {
                "agent_type": "substack_agent",
                "success": success,
                "metadata": metadata or {},
                "skipped": skipped,
                "is_complete": True
            },
            "is_complete": True
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        logger.error(f"❌ Substack Agent error: {error_message}")
        return {
            "messages": [AIMessage(content=f"Article generation failed: {error_message}")],
            "agent_results": {
                "agent_type": "substack_agent",
                "success": False,
                "error": error_message,
                "is_complete": True
            },
            "is_complete": True
        }


# Singleton instance
_substack_agent_instance = None


def get_substack_agent() -> SubstackAgent:
    """Get global substack agent instance"""
    global _substack_agent_instance
    if _substack_agent_instance is None:
        _substack_agent_instance = SubstackAgent()
    return _substack_agent_instance

