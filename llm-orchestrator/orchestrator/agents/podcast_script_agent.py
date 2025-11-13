"""
Podcast Script Agent
LangGraph agent for ElevenLabs TTS podcast script generation
Generates engaging, emotionally dynamic scripts with audio cues
"""

import logging
import re
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_core.messages import AIMessage

from orchestrator.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class PodcastScriptAgent(BaseAgent):
    """
    Podcast Script Agent for ElevenLabs TTS script generation
    
    Generates dynamic, emotionally engaging podcast scripts with
    inline bracket cues for text-to-speech systems
    """
    
    def __init__(self):
        super().__init__("podcast_script_agent")
        self._grpc_client = None
    
    async def _get_grpc_client(self):
        """Get or create gRPC client for backend tools"""
        if self._grpc_client is None:
            from orchestrator.clients.backend_tool_client import get_backend_tool_client
            self._grpc_client = await get_backend_tool_client()
        return self._grpc_client
    
    def _build_system_prompt(self, persona: Optional[Dict[str, Any]] = None) -> str:
        """Build podcast script system prompt"""
        base = (
            "You are a professional podcast scriptwriter. "
            "Produce a single-narrator plain-text script suitable for ElevenLabs TTS, with inline bracket cues.\n\n"
            "=== INLINE CUE LEXICON (ElevenLabs v3 Audio Tags) ===\n"
            "TIMING & RHYTHM CONTROL:\n"
            "- [pause] - Brief pause for effect\n"
            "- [breathes] - Natural breathing moment\n"
            "- [continues after a beat] - Thoughtful pause before continuing\n"
            "- [rushed] - Fast-paced, urgent delivery\n"
            "- [slows down] - Deliberate, measured speech\n"
            "- [deliberate] - Intentionally careful pacing\n"
            "- [rapid-fire] - Very fast, machine-gun delivery\n"
            "- [drawn out] - Extended, stretched pronunciation\n\n"
            "EMPHASIS & STRESS:\n"
            "- [stress on next word] - Emphasizes the following word\n"
            "- [emphasized] - General emphasis on phrase\n"
            "- [understated] - Downplayed, subtle delivery\n"
            "- ALL CAPS for additional emphasis and urgency\n\n"
            "HESITATION & RHYTHM:\n"
            "- [stammers] - Verbal stumbling\n"
            "- [repeats] - Repetition of words for effect\n"
            "- [timidly] - Uncertain, hesitant delivery\n"
            "- [suspicious tone] - Questioning, doubtful\n"
            "- [questioning] - Inquiry or doubt\n\n"
            "TONE & EMOTION:\n"
            "- [flatly] - Monotone, emotionless\n"
            "- [warmly] - Friendly, welcoming\n"
            "- [whisper] / [whispering] - Quiet, conspiratorial\n"
            "- [excited] - Enthusiastic energy\n"
            "- [quietly] - Subdued volume\n"
            "- [hesitant] - Uncertain, cautious\n"
            "- [nervous] - Anxious, worried\n"
            "- [angrily] - Angry tone\n"
            "- [fed up] - Exasperated, at limit\n"
            "- [mocking] - Sarcastic, derisive\n"
            "- [exasperated] / [exasperated sigh] - Frustrated, weary\n"
            "- [disgusted] - Revulsion, contempt\n"
            "- [outraged] / [indignant] - Moral objection\n"
            "- [shouting] / [frustrated shouting] / [enraged] / [furious] - Intense anger\n"
            "- [annoyed] / [building anger] - Escalating irritation\n"
            "- [incensed] / [ranting] - Passionate tirades\n"
            "- [dramatically] - Theatrical emphasis\n\n"
            "STAGE DIRECTIONS & REACTIONS:\n"
            "- [laughs] / [laughing] / [chuckles] / [giggle] / [big laugh] - Laughter variations\n"
            "- [clears throat] - Throat clearing\n"
            "- [sighs] - Audible sigh\n"
            "- [gasp] / [shudder] - Shock and revulsion\n"
            "- [gulps] - Nervous swallowing\n\n"
            "NATURAL STAMMERING & VERBAL STUMBLES (NO TAGS - use creative spelling):\n"
            "- When excited or angry, include natural verbal stumbles for authenticity.\n"
            "- Examples: 'F...folks', 'I...I just can't believe', 'This is—this is OUTRAGEOUS', 'ugh', 'gah', 'argh'\n"
            "- Use ellipses (...) and em-dashes (—) to show hesitation, interruption, or passionate stammering.\n"
            "- 'They're trying to—to TELL US that...', 'This is just—ugh—DISGUSTING'\n\n"
            "CRITICAL: Plain text ONLY. No markdown, no code fences, no SSML.\n"
            "CRITICAL: Keep short paragraphs for breath; tasteful em-dashes.\n"
            "CRITICAL: Use bracket cues FREQUENTLY and ANIMATEDLY for dynamic delivery.\n"
            "CRITICAL: USE EMOTIONAL CUES LIBERALLY.\n"
            "CRITICAL: DON'T BE AFRAID TO SHOUT for emphasis - this adds passion and engagement!\n"
            "CRITICAL: Include NATURAL STAMMERING when excited/angry.\n"
            "CRITICAL: STRICT LENGTH LIMIT - Maximum 3,000 characters total. Be punchy and impactful, not verbose!\n"
        )
        
        if persona:
            persona_name = persona.get("ai_name", "")
            persona_style = persona.get("persona_style", "")
            if persona_name or persona_style:
                base += f"\n\nPERSONA: Write as {persona_name or 'host'} with {persona_style or 'warm'} style."
        
        return base
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process podcast script generation request"""
        try:
            logger.info("🎙️ Podcast Script Agent: Starting script generation")
            
            # Extract state components
            messages = state.get("messages", [])
            user_id = state.get("user_id")
            shared_memory = state.get("shared_memory", {})
            active_editor = shared_memory.get("active_editor", {})
            
            # Check if editor is podcast type
            frontmatter = active_editor.get("frontmatter", {})
            doc_type = str(frontmatter.get("type", "")).strip().lower()
            
            if doc_type != "podcast":
                logger.info(f"🎙️ Podcast agent skipping: editor type is '{doc_type}', not 'podcast'")
                return self._create_response(
                    success=True,
                    response="Active editor is not a podcast document. Podcast agent requires editor with type='podcast'.",
                    skipped=True
                )
            
            # Get user message and editor content
            latest_message = messages[-1] if messages else None
            user_message = latest_message.content if hasattr(latest_message, 'content') else ""
            editor_content = active_editor.get("content", "")
            
            # Extract request parameters
            target_length = int(frontmatter.get("target_length_words", 900))
            tone = str(frontmatter.get("tone", "warm")).lower()
            pacing = str(frontmatter.get("pacing", "moderate")).lower()
            include_music = bool(frontmatter.get("include_music_cues", False))
            include_sfx = bool(frontmatter.get("include_sfx_cues", False))
            
            # Extract structured sections
            persona_text, background_text, articles, tweets_text = self._extract_sections(
                editor_content, user_message
            )
            
            # Fetch URL if provided
            fetched_content = await self._fetch_url_content(user_message)
            if fetched_content:
                articles.insert(0, fetched_content)
            
            # Build content block
            content_block = self._build_content_block(
                persona_text, background_text, articles, tweets_text
            )
            
            # Build system prompt
            persona = state.get("persona")
            system_prompt = self._build_system_prompt(persona)
            
            # Build task instructions
            task_block = self._build_task_block(
                user_message, target_length, tone, pacing, include_music, include_sfx
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
            
            logger.info(f"🎙️ Generating podcast script with {len(messages_to_send)} messages")
            
            response = await chat_service.openai_client.chat.completions.create(
                model=chat_service.model,
                messages=messages_to_send,
                temperature=0.3
            )
            
            content = response.choices[0].message.content or "{}"
            
            # Parse structured response
            script_text, metadata = self._parse_response(content)
            
            logger.info("✅ Podcast Script Agent: Script generation complete")
            
            return self._create_response(
                success=True,
                response=script_text,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"❌ Podcast Script Agent ERROR: {e}")
            return self._create_error_result(f"Script generation failed: {str(e)}")
    
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
            background_text = extract_section('Background', user_message)
            tweets_text = extract_section('Tweet', user_message) or extract_section('Tweets', user_message)
            
            # Extract articles
            articles = []
            for i in range(1, 4):
                article = extract_section(f'Article {i}', user_message)
                if article:
                    articles.append(article)
            
            # Also try generic "Article"
            generic_article = extract_section('Article', user_message)
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
            logger.info(f"🎙️ Fetching URL: {url}")
            
            grpc_client = await self._get_grpc_client()
            
            # Use web search to get content
            result = await grpc_client.search_web(query=url, max_results=1)
            
            if result.get("success") and result.get("results"):
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
            sections.append("=== PERSONA DEFINITIONS ===\n" + persona_text)
        if background_text:
            sections.append("=== BACKGROUND ===\n" + background_text)
        
        for i, article in enumerate(articles, 1):
            sections.append(f"=== ARTICLE {i if len(articles) > 1 else ''} ===\n" + article)
        
        if tweets_text:
            sections.append("=== TWEETS ===\n" + tweets_text)
        
        return "\n\n".join(sections)
    
    def _build_task_block(
        self,
        user_message: str,
        target_length: int,
        tone: str,
        pacing: str,
        include_music: bool,
        include_sfx: bool
    ) -> str:
        """Build task instruction block"""
        return (
            "=== REQUEST ===\n"
            f"User instruction: {user_message.strip()}\n\n"
            f"Target length: {target_length} words\n"
            f"Tone: {tone}\n"
            f"Pacing: {pacing}\n"
            f"Include music cues: {include_music}\n"
            f"Include SFX cues: {include_sfx}\n\n"
            "FORMAT GUIDELINES:\n"
            "- Read user request carefully for format (monologue vs dialogue)\n"
            "- If 'monologue' or 'commentary' → single narrator WITHOUT speaker labels\n"
            "- If 'dialogue', 'debate', 'conversation' → multi-speaker WITH labels (HOST:, CALLER:, etc.)\n"
            "- If unclear, default to MONOLOGUE\n"
            "- Ground content in provided sources - cite names, quotes, specifics\n"
            "- Use bracket cues FREQUENTLY: [excited], [mocking], [shouting], [pause], [breathes]\n"
            "- Include natural stammering: 'F...folks', 'This is—ugh—OUTRAGEOUS'\n"
            "- MAXIMUM 3,000 characters total (strict limit)\n\n"
            "RESPOND WITH JSON:\n"
            "{\n"
            '  "task_status": "complete",\n'
            '  "script_text": "Your podcast script here with [bracket cues]",\n'
            '  "metadata": {"words": 900, "estimated_duration_sec": 180, "tag_counts": {}}\n'
            "}\n"
        )
    
    def _parse_response(self, content: str) -> tuple:
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
            
            script_text = data.get("script_text", "")
            metadata = data.get("metadata", {})
            
            # Strip disallowed tags
            script_text = re.sub(r"\[pause:[^\]]+\]", "", script_text, flags=re.IGNORECASE)
            script_text = re.sub(r"\[(?:beat|breath)\]", "", script_text, flags=re.IGNORECASE)
            
            # Ensure metadata has required fields
            if not isinstance(metadata, dict):
                metadata = {}
            metadata.setdefault("words", len(script_text.split()))
            metadata.setdefault("estimated_duration_sec", max(60, metadata.get("words", 0) // 3))
            metadata.setdefault("tag_counts", {})
            
            return script_text, metadata
            
        except Exception as e:
            logger.warning(f"JSON parse failed, using fallback: {e}")
            
            # Fallback: treat as raw text
            error_text = f"Failed to parse response: {e}\n\nRaw content: {content[:500]}..."
            return error_text, {
                "words": 0,
                "estimated_duration_sec": 0,
                "tag_counts": {}
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
                "agent_type": "podcast_script_agent",
                "success": success,
                "metadata": metadata or {},
                "skipped": skipped,
                "is_complete": True
            },
            "is_complete": True
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        logger.error(f"❌ Podcast Script Agent error: {error_message}")
        return {
            "messages": [AIMessage(content=f"Podcast script generation failed: {error_message}")],
            "agent_results": {
                "agent_type": "podcast_script_agent",
                "success": False,
                "error": error_message,
                "is_complete": True
            },
            "is_complete": True
        }


# Singleton instance
_podcast_script_agent_instance = None


def get_podcast_script_agent() -> PodcastScriptAgent:
    """Get global podcast script agent instance"""
    global _podcast_script_agent_instance
    if _podcast_script_agent_instance is None:
        _podcast_script_agent_instance = PodcastScriptAgent()
    return _podcast_script_agent_instance

