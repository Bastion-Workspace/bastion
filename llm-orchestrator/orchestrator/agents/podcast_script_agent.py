"""
Podcast Script Agent
LangGraph agent for ElevenLabs TTS podcast script generation
Generates engaging, emotionally dynamic scripts with audio cues
"""

import logging
import re
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict, Tuple
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END
from orchestrator.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class PodcastScriptState(TypedDict):
    """State for podcast script agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    persona: Optional[Dict[str, Any]]
    user_message: str
    editor_content: str
    frontmatter: Dict[str, Any]
    editing_mode: bool  # True if editor has existing content
    structured_edit: Optional[Dict[str, Any]]  # LLM-generated edit plan
    editor_operations: List[Dict[str, Any]]  # Resolved operations
    content_block: str
    system_prompt: str
    task_block: str
    script_text: str
    metadata_result: Dict[str, Any]
    response: Dict[str, Any]
    task_status: str
    error: str


class PodcastScriptAgent(BaseAgent):
    """
    Podcast Script Agent for ElevenLabs TTS script generation
    
    Generates dynamic, emotionally engaging podcast scripts with
    inline bracket cues for text-to-speech systems
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("podcast_script_agent")
        self._grpc_client = None
        logger.info("🎙️ Podcast Script Agent ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for podcast script agent"""
        workflow = StateGraph(PodcastScriptState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("extract_content", self._extract_content_node)
        workflow.add_node("generate_script", self._generate_script_node)
        workflow.add_node("resolve_operations", self._resolve_operations_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Conditional routing based on editing mode
        workflow.add_edge("prepare_context", "extract_content")
        workflow.add_edge("extract_content", "generate_script")
        
        # Route based on editing mode
        workflow.add_conditional_edges(
            "generate_script",
            lambda state: "resolve_operations" if state.get("editing_mode") else "format_response",
            {
                "resolve_operations": "resolve_operations",
                "format_response": "format_response"
            }
        )
        
        workflow.add_edge("resolve_operations", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    async def _get_grpc_client(self):
        """Get or create gRPC client for backend tools"""
        if self._grpc_client is None:
            from orchestrator.clients.backend_tool_client import get_backend_tool_client
            self._grpc_client = await get_backend_tool_client()
        return self._grpc_client
    
    def _build_system_prompt(self, persona: Optional[Dict[str, Any]] = None, editing_mode: bool = False) -> str:
        """Build podcast script system prompt"""
        base = (
            "You are a professional podcast scriptwriter. "
            "Produce a single-narrator plain-text script suitable for ElevenLabs TTS, with inline bracket cues.\n\n"
        )
        
        # Add editing mode instructions if in editing mode
        if editing_mode:
            base += (
                "=== EDITING MODE - STRUCTURED OPERATIONS ===\n"
                "The editor contains existing content. You must generate targeted edit operations as JSON:\n\n"
                "{\n"
                '  "summary": "Brief description of changes",\n'
                '  "operations": [\n'
                "    {\n"
                '      "op_type": "replace_range",\n'
                '      "start": 0,\n'
                '      "end": 50,\n'
                '      "text": "New script content with [brackets]",\n'
                '      "original_text": "Exact text from document (20-40 words)",\n'
                '      "occurrence_index": 0\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                "Operation types:\n"
                "- replace_range: Replace existing text with new text\n"
                "- delete_range: Remove existing text\n"
                "- insert_after_heading: Add content after a specific heading\n\n"
                "CRITICAL: Provide \"original_text\" with EXACT verbatim text from the document for replace/delete operations.\n"
                "CRITICAL: For insert_after_heading, provide \"anchor_text\" with the exact heading line.\n\n"
            )
        
        base += (
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
            persona_style = persona.get("persona_style", "professional")
            
            # Use centralized style instruction from BaseAgent
            style_instruction = self._get_style_instruction(persona_style)
            
            if persona_name:
                base += f"\n\nYou are {persona_name}."
            base += f"\n\n{style_instruction}"
        
        return base
    
    async def _prepare_context_node(self, state: PodcastScriptState) -> Dict[str, Any]:
        """Prepare context: extract message and check editor type"""
        try:
            logger.info("📋 Preparing context for podcast script generation...")
            
            messages = state.get("messages", [])
            shared_memory = state.get("shared_memory", {})
            active_editor = shared_memory.get("active_editor", {})
            
            # Check if editor is podcast type
            frontmatter = active_editor.get("frontmatter", {})
            doc_type = str(frontmatter.get("type", "")).strip().lower()
            
            if doc_type != "podcast":
                logger.info(f"🎙️ Podcast agent skipping: editor type is '{doc_type}', not 'podcast'")
                return {
                    "response": self._create_response(
                    success=True,
                    response="Active editor is not a podcast document. Podcast agent requires editor with type='podcast'.",
                    skipped=True
                    ),
                    "task_status": "skipped"
                }
            
            # Get user message and editor content
            latest_message = messages[-1] if messages else None
            user_message = latest_message.content if hasattr(latest_message, 'content') else ""
            editor_content = active_editor.get("content", "")
            
            # Detect editing mode: if editor has content (after frontmatter), use editing mode
            editing_mode = False
            if editor_content:
                # Strip frontmatter to check if there's actual content
                import re
                frontmatter_match = re.match(r'^---\s*\n[\s\S]*?\n---\s*\n', editor_content)
                if frontmatter_match:
                    body_content = editor_content[frontmatter_match.end():].strip()
                    editing_mode = len(body_content) > 0
                else:
                    editing_mode = len(editor_content.strip()) > 0
            
            logger.info(f"🎙️ Podcast agent mode: {'EDITING' if editing_mode else 'GENERATION'}")
            
            return {
                "user_message": user_message,
                "editor_content": editor_content,
                "frontmatter": frontmatter,
                "editing_mode": editing_mode,
                "editor_operations": [],
                "structured_edit": None
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare context: {e}")
            return {
                "user_message": "",
                "editor_content": "",
                "frontmatter": {},
                "error": str(e),
                "task_status": "error"
            }
    
    async def _extract_content_node(self, state: PodcastScriptState) -> Dict[str, Any]:
        """Extract content sections and build content/task blocks"""
        try:
            logger.info("📝 Extracting content sections...")
            
            user_message = state.get("user_message", "")
            editor_content = state.get("editor_content", "")
            frontmatter = state.get("frontmatter", {})
            persona = state.get("persona")
            
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
            
            # Build system prompt (check editing mode from state)
            editing_mode = state.get("editing_mode", False)
            system_prompt = self._build_system_prompt(persona, editing_mode=editing_mode)
            
            # Build task instructions (check editing mode from state)
            editing_mode = state.get("editing_mode", False)
            task_block = self._build_task_block(
                user_message, target_length, tone, pacing, include_music, include_sfx, editing_mode=editing_mode
            )
            
            return {
                "content_block": content_block,
                "system_prompt": system_prompt,
                "task_block": task_block
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to extract content: {e}")
            return {
                "content_block": "",
                "system_prompt": "",
                "task_block": "",
                "error": str(e)
            }
    
    async def _generate_script_node(self, state: PodcastScriptState) -> Dict[str, Any]:
        """Generate podcast script using LLM"""
        try:
            editing_mode = state.get("editing_mode", False)
            editor_content = state.get("editor_content", "")
            
            if editing_mode:
                logger.info("🎙️ Generating podcast script edits (editing mode)...")
            else:
                logger.info("🎙️ Generating podcast script (generation mode)...")
            
            content_block = state.get("content_block", "")
            system_prompt = state.get("system_prompt", "")
            task_block = state.get("task_block", "")
            
            # Use centralized LLM access from BaseAgent
            llm = self._get_llm(temperature=0.3, state=state)
            
            # Build LangChain messages
            messages = [
                SystemMessage(content=system_prompt),
                SystemMessage(content=f"Current Date/Time: {datetime.now().isoformat()}")
            ]
            
            if content_block:
                messages.append(HumanMessage(content=content_block))
            
            # In editing mode, include current editor content
            if editing_mode and editor_content:
                # Strip frontmatter for editing context
                import re
                frontmatter_match = re.match(r'^---\s*\n[\s\S]*?\n---\s*\n', editor_content)
                if frontmatter_match:
                    body_content = editor_content[frontmatter_match.end():]
                else:
                    body_content = editor_content
                
                messages.append(HumanMessage(
                    content=f"=== CURRENT EDITOR CONTENT ===\n{body_content}\n\n=== END CURRENT CONTENT ===\n\n{task_block}"
                ))
            else:
                messages.append(HumanMessage(content=task_block))
            
            logger.info(f"🎙️ Generating podcast script with {len(messages)} messages")
            
            # Use LangChain interface
            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            if editing_mode:
                # Parse structured edit operations
                structured_edit = self._parse_editing_response(content)
                logger.info("✅ Podcast Script Agent: Edit plan generation complete")
                return {
                    "structured_edit": structured_edit,
                    "task_status": "complete"
                }
            else:
                # Parse structured response (generation mode)
                script_text, metadata = self._parse_response(content)
                logger.info("✅ Podcast Script Agent: Script generation complete")
                
                result = self._create_response(
                    success=True,
                    response=script_text,
                    metadata=metadata
                )
                
                return {
                    "response": result,
                    "script_text": script_text,
                    "metadata_result": metadata,
                    "task_status": "complete"
                }
            
        except Exception as e:
            logger.error(f"❌ Script generation failed: {e}")
            return {
                "response": self._create_error_result(f"Script generation failed: {str(e)}"),
                "task_status": "error",
                "error": str(e)
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """
        Process podcast script generation request using LangGraph workflow
        
        Args:
            query: User query string
            metadata: Optional metadata dictionary (persona, editor context, etc.)
            messages: Optional conversation history
            
        Returns:
            Dictionary with podcast script response and metadata
        """
        try:
            logger.info(f"🎙️ Podcast Script Agent: Starting script generation: {query[:100]}...")
            
            # Extract user_id and shared_memory from metadata
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")
            shared_memory = metadata.get("shared_memory", {}) or {}
            
            # Prepare new messages (current query)
            new_messages = self._prepare_messages_with_query(messages, query)
            
            # Get workflow to access checkpoint
            workflow = await self._get_workflow()
            config = self._get_checkpoint_config(metadata)
            
            # Load and merge checkpointed messages to preserve conversation history
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages
            )
            
            # Load shared_memory from checkpoint if available
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            
            # Merge shared_memory: start with checkpoint, then update with NEW data (so new active_editor overwrites old)
            shared_memory_merged = existing_shared_memory.copy()
            shared_memory_merged.update(shared_memory)  # New data (including updated active_editor) takes precedence
            
            # Build initial state for LangGraph workflow
            initial_state: PodcastScriptState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
                "persona": metadata.get("persona"),
                "user_message": "",
                "editor_content": "",
                "frontmatter": {},
                "editing_mode": False,
                "structured_edit": None,
                "editor_operations": [],
                "content_block": "",
                "system_prompt": "",
                "task_block": "",
                "script_text": "",
                "metadata_result": {},
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Invoke LangGraph workflow with checkpointing
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract response
            response = final_state.get("response", {
                "messages": [AIMessage(content="Podcast script generation failed")],
                "agent_results": {
                    "agent_type": self.agent_type,
                    "is_complete": False
                },
                "is_complete": False
            })
            
            # Add editor operations at top level for compatibility (if in editing mode)
            if final_state.get("editing_mode") and final_state.get("editor_operations"):
                response["editor_operations"] = final_state.get("editor_operations", [])
                if final_state.get("response", {}).get("agent_results", {}).get("manuscript_edit"):
                    response["manuscript_edit"] = final_state["response"]["agent_results"]["manuscript_edit"]
            
            return response
            
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
        include_sfx: bool,
        editing_mode: bool = False
    ) -> str:
        """Build task instruction block"""
        base = (
            "=== REQUEST ===\n"
            f"User instruction: {user_message.strip()}\n\n"
            f"Target length: {target_length} words\n"
            f"Tone: {tone}\n"
            f"Pacing: {pacing}\n"
            f"Include music cues: {include_music}\n"
            f"Include SFX cues: {include_sfx}\n\n"
        )
        
        if editing_mode:
            base += (
                "EDITING MODE: Generate targeted edit operations.\n"
                "Review the current editor content and create operations to modify it.\n"
                "Provide EXACT original_text from the document for replace/delete operations.\n\n"
                "RESPOND WITH JSON:\n"
                "{\n"
                '  "summary": "Brief description of changes",\n'
                '  "operations": [\n'
                "    {\n"
                '      "op_type": "replace_range",\n'
                '      "start": 0,\n'
                '      "end": 50,\n'
                '      "text": "New script content with [brackets]",\n'
                '      "original_text": "Exact text from document (20-40 words)",\n'
                '      "occurrence_index": 0\n'
                "    }\n"
                "  ]\n"
                "}\n"
            )
        else:
            base += (
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
        
        return base
    
    def _parse_response(self, content: str) -> tuple:
        """Parse LLM JSON response (generation mode)"""
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
    
    def _parse_editing_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM JSON response (editing mode)"""
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
            
            # Validate structure
            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object")
            
            operations = data.get("operations", [])
            if not isinstance(operations, list):
                operations = []
            
            return {
                "summary": data.get("summary", "Edit plan ready"),
                "operations": operations
            }
            
        except Exception as e:
            logger.warning(f"JSON parse failed for editing response: {e}")
            return {
                "summary": "Failed to parse edit plan",
                "operations": []
            }
    
    def _resolve_operation_simple(
        self,
        content: str,
        op_dict: Dict[str, Any],
        frontmatter_end: int = 0
    ) -> Tuple[int, int, str, float]:
        """Resolve operation to exact positions using progressive search"""
        
        op_type = op_dict.get("op_type", "replace_range")
        original_text = op_dict.get("original_text")
        anchor_text = op_dict.get("anchor_text")
        occurrence_index = op_dict.get("occurrence_index", 0)
        text = op_dict.get("text", "")
        
        # Strategy 1: Exact match with original_text
        if original_text and op_type in ("replace_range", "delete_range"):
            count = 0
            search_from = 0
            while True:
                pos = content.find(original_text, search_from)
                if pos == -1:
                    break
                if count == occurrence_index:
                    end_pos = pos + len(original_text)
                    # Guard frontmatter
                    pos = max(pos, frontmatter_end)
                    end_pos = max(end_pos, pos)
                    return pos, end_pos, text, 1.0
                count += 1
                search_from = pos + 1
        
        # Strategy 2: Anchor text for insert_after_heading
        if anchor_text and op_type == "insert_after_heading":
            pos = content.find(anchor_text)
            if pos != -1:
                # Find end of line after anchor
                line_end = content.find("\n", pos)
                if line_end == -1:
                    line_end = len(content)
                insert_pos = line_end + 1
                # Guard frontmatter
                insert_pos = max(insert_pos, frontmatter_end)
                return insert_pos, insert_pos, text, 0.9
        
        # Fallback: use approximate positions
        start = op_dict.get("start", frontmatter_end)
        end = op_dict.get("end", start)
        start = max(start, frontmatter_end)
        end = max(end, start)
        return start, end, text, 0.5
    
    def _get_frontmatter_end(self, content: str) -> int:
        """Find frontmatter end position"""
        import re
        match = re.match(r'^---\s*\n[\s\S]*?\n---\s*\n', content)
        return match.end() if match else 0
    
    async def _resolve_operations_node(self, state: PodcastScriptState) -> Dict[str, Any]:
        """Resolve editor operations with progressive search"""
        try:
            logger.info("🎙️ Resolving podcast script operations...")
            
            editor_content = state.get("editor_content", "")
            structured_edit = state.get("structured_edit")
            
            if not structured_edit or not isinstance(structured_edit.get("operations"), list):
                return {
                    "editor_operations": [],
                    "error": "No operations to resolve",
                    "task_status": "error"
                }
            
            fm_end_idx = self._get_frontmatter_end(editor_content)
            editor_operations = []
            operations = structured_edit.get("operations", [])
            
            for op in operations:
                try:
                    resolved_start, resolved_end, resolved_text, resolved_confidence = self._resolve_operation_simple(
                        editor_content,
                        op,
                        frontmatter_end=fm_end_idx
                    )
                    
                    logger.info(f"Resolved {op.get('op_type')} [{resolved_start}:{resolved_end}] confidence={resolved_confidence:.2f}")
                    
                    # Build operation dict
                    resolved_op = {
                        "op_type": op.get("op_type", "replace_range"),
                        "start": resolved_start,
                        "end": resolved_end,
                        "text": resolved_text,
                        "original_text": op.get("original_text"),
                        "anchor_text": op.get("anchor_text"),
                        "occurrence_index": op.get("occurrence_index", 0),
                        "confidence": resolved_confidence
                    }
                    
                    editor_operations.append(resolved_op)
                    
                except Exception as e:
                    logger.warning(f"Operation resolution failed: {e}")
                    continue
            
            return {
                "editor_operations": editor_operations
            }
            
        except Exception as e:
            logger.error(f"Failed to resolve operations: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "editor_operations": [],
                "error": str(e),
                "task_status": "error"
            }
    
    async def _format_response_node(self, state: PodcastScriptState) -> Dict[str, Any]:
        """Format final response with editor operations"""
        try:
            editing_mode = state.get("editing_mode", False)
            editor_operations = state.get("editor_operations", [])
            structured_edit = state.get("structured_edit", {})
            script_text = state.get("script_text", "")
            metadata_result = state.get("metadata_result", {})
            
            if editing_mode:
                # Editing mode: return operations
                preview = "\n\n".join([
                    op.get("text", "").strip()
                    for op in editor_operations
                    if op.get("text", "").strip()
                ]).strip()
                response_text = preview if preview else (structured_edit.get("summary", "Edit plan ready."))
                
                result = {
                    "response": response_text,
                    "task_status": "complete",
                    "agent_type": "podcast_script_agent"
                }
                
                # Add editor operations
                if editor_operations:
                    result["editor_operations"] = editor_operations
                    result["manuscript_edit"] = {
                        **structured_edit,
                        "operations": editor_operations
                    }
                
                return {
                    "response": {
                        "messages": [AIMessage(content=response_text)],
                        "agent_results": {
                            "agent_type": "podcast_script_agent",
                            "is_complete": True,
                            "editor_operations": editor_operations,
                            "manuscript_edit": result.get("manuscript_edit")
                        },
                        "is_complete": True
                    },
                    "task_status": "complete"
                }
            else:
                # Generation mode: return script text
                result = self._create_response(
                    success=True,
                    response=script_text,
                    metadata=metadata_result
                )
                
                return {
                    "response": result,
                    "task_status": "complete"
                }
            
        except Exception as e:
            logger.error(f"❌ Format response failed: {e}")
            return {
                "response": self._create_error_result(f"Response formatting failed: {str(e)}"),
                "task_status": "error",
                "error": str(e)
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

