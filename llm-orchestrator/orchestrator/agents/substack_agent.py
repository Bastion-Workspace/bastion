"""
Substack Agent
LangGraph agent for long-form article and tweet generation
Generates publication-ready content with research integration
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


class SubstackState(TypedDict):
    """State for substack agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    persona: Optional[Dict[str, Any]]
    user_message: str
    editor_content: str
    frontmatter: Dict[str, Any]
    tweet_mode: bool
    editing_mode: bool  # True if editor has existing content
    structured_edit: Optional[Dict[str, Any]]  # LLM-generated edit plan
    editor_operations: List[Dict[str, Any]]  # Resolved operations
    content_block: str
    system_prompt: str
    task_block: str
    article_text: str
    metadata_result: Dict[str, Any]
    response: Dict[str, Any]
    task_status: str
    error: str


class SubstackAgent(BaseAgent):
    """
    Substack Agent for long-form article and tweet generation
    
    Synthesizes multiple sources into cohesive, engaging content
    with optional research augmentation
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("substack_agent")
        self._grpc_client = None
        logger.info("📝 Substack Agent ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for substack agent"""
        workflow = StateGraph(SubstackState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("extract_content", self._extract_content_node)
        workflow.add_node("generate_article", self._generate_article_node)
        workflow.add_node("resolve_operations", self._resolve_operations_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Conditional routing based on editing mode
        workflow.add_edge("prepare_context", "extract_content")
        workflow.add_edge("extract_content", "generate_article")
        
        # Route based on editing mode
        workflow.add_conditional_edges(
            "generate_article",
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
        """Build article writing system prompt"""
        base = (
            "You are a professional long-form article writer specializing in blog posts and Substack publications. "
            "Your task is to synthesize multiple source materials into a cohesive, engaging article.\n\n"
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
                '      "text": "New article content",\n'
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
            persona_style = persona.get("persona_style", "professional")
            
            # Use centralized style instruction from BaseAgent
            style_instruction = self._get_style_instruction(persona_style)
            
            if persona_name:
                base += f"\n\nYou are {persona_name}."
            base += f"\n\n{style_instruction}"
        
        return base
    
    async def _prepare_context_node(self, state: SubstackState) -> Dict[str, Any]:
        """Prepare context: extract message and check editor type"""
        try:
            logger.info("📋 Preparing context for article generation...")
            
            messages = state.get("messages", [])
            shared_memory = state.get("shared_memory", {})
            active_editor = shared_memory.get("active_editor", {})
            
            # Check if editor is substack/blog type
            frontmatter = active_editor.get("frontmatter", {})
            doc_type = str(frontmatter.get("type", "")).strip().lower()
            
            if doc_type not in ["substack", "blog"]:
                logger.info(f"📝 Substack agent skipping: editor type is '{doc_type}', not 'substack' or 'blog'")
                return {
                    "response": self._create_response(
                    success=True,
                    response="Active editor is not a substack/blog document. Substack agent requires editor with type='substack' or type='blog'.",
                    skipped=True
                    ),
                    "task_status": "skipped"
                }
            
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
            
            logger.info(f"📝 Substack agent mode: {'EDITING' if editing_mode else 'GENERATION'}")
            
            return {
                "user_message": user_message,
                "editor_content": editor_content,
                "frontmatter": frontmatter,
                "tweet_mode": tweet_mode,
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
                "tweet_mode": False,
                "error": str(e),
                "task_status": "error"
            }
    
    async def _extract_content_node(self, state: SubstackState) -> Dict[str, Any]:
        """Extract content sections and build content/task blocks"""
        try:
            logger.info("📝 Extracting content sections...")
            
            user_message = state.get("user_message", "")
            editor_content = state.get("editor_content", "")
            frontmatter = state.get("frontmatter", {})
            tweet_mode = state.get("tweet_mode", False)
            persona = state.get("persona")
            
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
            
            # Build system prompt (check editing mode from state)
            editing_mode = state.get("editing_mode", False)
            system_prompt = self._build_system_prompt(persona, editing_mode=editing_mode)
            
            # Build task instructions (check editing mode from state)
            editing_mode = state.get("editing_mode", False)
            task_block = self._build_task_block(
                user_message, target_length, tone, style, tweet_mode, frontmatter, editing_mode=editing_mode
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
    
    async def _generate_article_node(self, state: SubstackState) -> Dict[str, Any]:
        """Generate article using LLM"""
        try:
            editing_mode = state.get("editing_mode", False)
            editor_content = state.get("editor_content", "")
            tweet_mode = state.get("tweet_mode", False)
            
            if editing_mode:
                logger.info("📝 Generating article edits (editing mode)...")
            else:
                logger.info("📝 Generating article (generation mode)...")
            
            content_block = state.get("content_block", "")
            system_prompt = state.get("system_prompt", "")
            task_block = state.get("task_block", "")
            
            # Use centralized LLM access from BaseAgent
            llm = self._get_llm(temperature=0.4, state=state)
            
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
            
            logger.info(f"📝 Generating article with {len(messages)} messages")
            
            # Use LangChain interface
            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            if editing_mode:
                # Parse structured edit operations
                structured_edit = self._parse_editing_response(content)
                logger.info("✅ Substack Agent: Edit plan generation complete")
                return {
                    "structured_edit": structured_edit,
                    "task_status": "complete"
                }
            else:
                # Parse structured response (generation mode)
                article_text, metadata = self._parse_response(content, tweet_mode)
                logger.info("✅ Substack Agent: Article generation complete")
                
                result = self._create_response(
                    success=True,
                    response=article_text,
                    metadata=metadata
                )
                
                return {
                    "response": result,
                    "article_text": article_text,
                    "metadata_result": metadata,
                    "task_status": "complete"
                }
            
        except Exception as e:
            logger.error(f"❌ Article generation failed: {e}")
            return {
                "response": self._create_error_result(f"Article generation failed: {str(e)}"),
                "task_status": "error",
                "error": str(e)
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """
        Process article generation request using LangGraph workflow
        
        Args:
            query: User query string
            metadata: Optional metadata dictionary (persona, editor context, etc.)
            messages: Optional conversation history
            
        Returns:
            Dictionary with article response and metadata
        """
        try:
            logger.info(f"📝 Substack Agent: Starting article generation: {query[:100]}...")
            
            # Extract user_id and shared_memory from metadata
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")
            shared_memory = metadata.get("shared_memory", {}) or {}
            persona = metadata.get("persona")
            
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
            initial_state: SubstackState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
                "persona": persona,
                "user_message": "",
                "editor_content": "",
                "frontmatter": {},
                "tweet_mode": False,
                "editing_mode": False,
                "structured_edit": None,
                "editor_operations": [],
                "content_block": "",
                "system_prompt": "",
                "task_block": "",
                "article_text": "",
                "metadata_result": {},
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Invoke LangGraph workflow with checkpointing
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract response
            response = final_state.get("response", {
                "messages": [AIMessage(content="Article generation failed")],
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
        frontmatter: Dict[str, Any],
        editing_mode: bool = False
    ) -> str:
        """Build task instruction block"""
        if editing_mode:
            base = (
                "=== EDITING MODE ===\n"
                "Generate targeted edit operations.\n"
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
                '      "text": "New article content",\n'
                '      "original_text": "Exact text from document (20-40 words)",\n'
                '      "occurrence_index": 0\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                f"User instruction: {user_message}\n"
            )
            return base
        elif tweet_mode:
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
    
    async def _resolve_operations_node(self, state: SubstackState) -> Dict[str, Any]:
        """Resolve editor operations with progressive search"""
        try:
            logger.info("📝 Resolving substack article operations...")
            
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
    
    async def _format_response_node(self, state: SubstackState) -> Dict[str, Any]:
        """Format final response with editor operations"""
        try:
            editing_mode = state.get("editing_mode", False)
            editor_operations = state.get("editor_operations", [])
            structured_edit = state.get("structured_edit", {})
            article_text = state.get("article_text", "")
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
                    "agent_type": "substack_agent"
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
                            "agent_type": "substack_agent",
                            "is_complete": True,
                            "editor_operations": editor_operations,
                            "manuscript_edit": result.get("manuscript_edit")
                        },
                        "is_complete": True
                    },
                    "task_status": "complete"
                }
            else:
                # Generation mode: return article text
                result = self._create_response(
                    success=True,
                    response=article_text,
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

