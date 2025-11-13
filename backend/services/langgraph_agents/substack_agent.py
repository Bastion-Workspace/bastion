"""
Substack Agent - Roosevelt's Long-Form Article Writer

Generates long-form markdown articles suitable for blog posts and Substack publications
by synthesizing multiple source articles, tweets, and background information through
a defined persona voice.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict

from .base_agent import BaseAgent, TaskStatus
from models.substack_models import SubstackArticleRequest, SubstackArticleResponse
from services.substack_research_helper import SubstackResearchHelper


logger = logging.getLogger(__name__)


class SubstackAgent(BaseAgent):
    def __init__(self):
        super().__init__("substack_agent")
        logger.info("📝 BULLY! Substack Agent mounted and ready to write!")

    def _build_system_prompt(self, persona: Dict[str, Any] | None) -> str:
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
        try:
            persona_prompt = self._build_persona_prompt(persona)
        except Exception:
            persona_prompt = ""
        return base + persona_prompt

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"📝 SUBSTACK AGENT: Starting process with state keys: {list(state.keys())}")
            shared_memory = state.get("shared_memory", {}) or {}
            active_editor = shared_memory.get("active_editor", {}) or {}
            logger.info(f"📝 SUBSTACK AGENT: Active editor keys: {list(active_editor.keys())}")

            # Gating: require editor open with frontmatter.type == 'substack' or 'blog'
            frontmatter = active_editor.get("frontmatter", {}) or {}
            doc_type = str(frontmatter.get("type", "")).strip().lower()
            logger.info(f"📝 SUBSTACK AGENT: Document type: '{doc_type}'")
            if doc_type not in ["substack", "blog"]:
                logger.info("📝 Substack agent skipped: active editor type is not 'substack' or 'blog'.")
                return self._create_success_result(
                    response="Active editor is not a substack/blog doc; substack agent skipping.",
                    tools_used=[],
                    processing_time=0.0,
                    shared_memory_updates={},
                    additional_data={"skipped": True},
                )

            # Extract request parameters from user message and frontmatter
            user_message = state.get("current_query", "") or ""
            editor_content = active_editor.get("content", "") or ""
            logger.info(f"📝 SUBSTACK AGENT: Editor content length: {len(editor_content)}")
            logger.info(f"📝 SUBSTACK AGENT: User message length: {len(user_message)}")

            # Detect if user wants tweet-sized/short-form content instead of long-form article
            user_message_lower = user_message.lower()
            tweet_mode = any(keyword in user_message_lower for keyword in [
                'tweet', 'twitter', 'x post', 'short post', 'social media post',
                'tweet-sized', 'tweet size', 'short-form', 'brief', 'concise',
                '280 character', 'social post'
            ])
            
            if tweet_mode:
                logger.info("🐦 TWEET MODE: User requested short-form content")

            # Determine topic from user message or frontmatter
            user_topic = None
            if user_message.strip():
                # Look for explicit topic requests
                topic_match = re.search(r'(?:topic|subject|about|title):\s*([^\n]+)', user_message, re.IGNORECASE)
                if topic_match:
                    user_topic = topic_match.group(1).strip()
                else:
                    # Use first line of user message as topic if it looks like a request
                    first_line = user_message.split('\n')[0].strip()
                    if len(first_line) > 10 and not first_line.startswith('##'):
                        user_topic = first_line

            # Build request object
            req = SubstackArticleRequest(
                topic=user_topic or frontmatter.get("title") or frontmatter.get("topic") or "Article",
                target_length_words=int(frontmatter.get("target_length_words") or 2500),
                tone=str(frontmatter.get("tone") or "conversational").lower() if str(frontmatter.get("tone") or "").strip() else "conversational",
                style=str(frontmatter.get("style") or "commentary").lower() if str(frontmatter.get("style") or "").strip() else "commentary",
                include_citations=bool(frontmatter.get("include_citations", True)),
                include_conclusion=bool(frontmatter.get("include_conclusion", True)),
            )

            persona = state.get("persona")
            system_prompt = self._build_system_prompt(persona)
            now_line = f"Current Date/Time: {datetime.now().isoformat()}"

            # Build structured output requirement
            schema_hint = (
                "STRUCTURED OUTPUT REQUIRED: Respond ONLY with valid JSON for SubstackArticleResponse. Fields: "
                "task_status (complete|incomplete|error), article_text (string with markdown), "
                "metadata (object with word_count, reading_time_minutes, section_count)."
            )

            # Content policy - different for tweet mode vs article mode
            if tweet_mode:
                policy = (
                    "=== TWEET-SIZED CONTENT INSTRUCTIONS ===\n\n"
                    "You are generating SHORT-FORM social media content (tweet/X post).\n\n"
                    "CRITICAL REQUIREMENTS:\n"
                    "- Maximum 280 characters (strict limit for single tweet)\n"
                    "- Can generate multiple tweets if needed (thread format)\n"
                    "- Punchy, attention-grabbing opening\n"
                    "- One clear message or insight per tweet\n"
                    "- Use the persona's voice and perspective\n\n"
                    "CONTENT APPROACH:\n"
                    "- Extract the most compelling angle from provided sources\n"
                    "- Sharp, memorable phrasing\n"
                    "- Can include hashtags if appropriate\n"
                    "- Can tag relevant accounts with @username format\n"
                    "- If generating thread: number tweets (1/3, 2/3, 3/3)\n\n"
                    "TWEET FORMAT:\n"
                    "SINGLE TWEET: Just the text (max 280 chars)\n"
                    "THREAD: Separate tweets with blank lines, number them (e.g., 1/4, 2/4, 3/4, 4/4)\n\n"
                    "EXAMPLES:\n"
                    "Single: 'The latest GDP figures tell a story the headlines miss: growth is uneven, and the middle class is still struggling. We need policy, not just platitudes.'\n\n"
                    "Thread:\n"
                    "1/3 The latest GDP figures tell a story the headlines miss. While topline growth looks solid, the distribution is deeply uneven.\n\n"
                    "2/3 Middle-class wage growth remains stagnant even as corporate profits hit record highs. This isn't just inequality—it's policy failure.\n\n"
                    "3/3 We need concrete action: progressive taxation, higher minimum wage, and investment in public goods. Platitudes won't cut it anymore.\n\n"
                    "GROUND IN SOURCES:\n"
                    "- Reference key facts from provided articles\n"
                    "- React to specific tweets if provided\n"
                    "- Channel the persona's perspective\n"
                    "- Make it punchy and memorable\n\n"
                    "⚠️ CRITICAL JSON OUTPUT REQUIREMENT ⚠️\n"
                    "You MUST return the tweet content wrapped in the SubstackArticleResponse JSON structure:\n"
                    "- Put your tweet(s) in the 'article_text' field\n"
                    "- Set task_status to 'complete'\n"
                    "- Set metadata.word_count to approximate character count\n"
                    "DO NOT return raw tweet text - it MUST be in the JSON structure!\n\n"
                    "JSON EXAMPLE FOR TWEET:\n"
                    "{\n"
                    '  "task_status": "complete",\n'
                    '  "article_text": "1/3 Your tweet text here...\\n\\n2/3 Second tweet...\\n\\n3/3 Final tweet...",\n'
                    '  "metadata": {"word_count": 420, "reading_time_minutes": 0, "section_count": 1}\n'
                    "}\n\n"
                )
            else:
                policy = (
                    "=== ARTICLE WRITING INSTRUCTIONS ===\n\n"
                "READ CAREFULLY: Review all provided source materials before writing.\n\n"
                "IF MULTIPLE ARTICLES PROVIDED:\n"
                "- Identify common themes and divergent perspectives\n"
                "- Synthesize information from all sources\n"
                "- Compare and contrast different viewpoints\n"
                "- Create a unified narrative that acknowledges multiple angles\n"
                "- Cite each source explicitly when referencing its content\n\n"
                "IF TWEETS PROVIDED:\n"
                "- Reference tweet authors by @username\n"
                "- Quote tweet content directly when impactful\n"
                "- Provide context for social media reactions\n"
                "- Analyze what the tweets reveal about public discourse\n\n"
                "IF BACKGROUND PROVIDED:\n"
                "- Use background information to provide context\n"
                "- Connect current discussion to broader themes\n"
                "- Don't just rehash background—build on it\n\n"
                "IF URL PROVIDED:\n"
                "- Fetched content will be included as a source\n"
                "- Treat it like any other article source\n"
                "- Cite the publication/author when available\n\n"
                "PERSONA GUIDANCE:\n"
                "- If PERSONA section provided, write in that voice consistently\n"
                "- Match the persona's perspective, tone, and style\n"
                "- Let persona inform what aspects you emphasize\n"
                "- Don't just mention the persona—embody it throughout\n\n"
                "WRITING QUALITY:\n"
                "- This should be publication-ready\n"
                "- Strong opening that hooks the reader\n"
                "- Clear thesis or central argument\n"
                "- Well-organized sections with descriptive headers\n"
                "- Original analysis and insights, not just summary\n"
                "- Compelling conclusion that reinforces main points\n\n"
                "FORMAT:\n"
                "- Start with a compelling title (# Title)\n"
                "- Use ## for main sections, ### for subsections\n"
                "- **Bold** key points, *italics* for subtle emphasis\n"
                "- > Blockquotes for direct quotations\n"
                "- Natural paragraph breaks for readability\n\n"
                "LENGTH:\n"
                f"- Target: {req.target_length_words} words\n"
                "- This is a LONG-FORM article, not a summary\n"
                "- Develop ideas fully with supporting evidence\n"
                "- Don't rush—give each point proper treatment\n\n"
                + schema_hint
            )

            # Detect URL in user message for optional content fetching
            user_has_url = False
            url = None
            try:
                url_match = re.search(r'https?://[^\s)>\"]+', user_message)
                if url_match:
                    url = url_match.group(0)
                    user_has_url = True
                    logger.info(f"📝 SUBSTACK AGENT: Detected URL: {url}")
            except Exception as e:
                logger.warning(f"📝 SUBSTACK AGENT: URL detection failed: {e}")

            # Extract structured sections from editor content and user message
            # Persona comes from editor first, other sections from user message
            try:
                # Normalize newlines
                um = user_message.replace('\r\n', '\n')
                ec = editor_content.replace('\r\n', '\n')

                def extract_section(title: str, text: str) -> str | None:
                    """Extract ## Title or Title: sections from text"""
                    lines = text.split('\n')
                    start = None

                    # Try markdown heading first: ## Title
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

                    # Capture until next heading or end
                    collected = []
                    for line in lines[start:]:
                        stripped = line.strip()
                        # Stop at next markdown heading or simple heading
                        if stripped.startswith('## ') or re.match(r'^(background|article|persona|tweet)\s*(\d+)?:', stripped, re.IGNORECASE):
                            break
                        collected.append(line)
                    body = '\n'.join(collected).strip()
                    return body or None

                # Extract sections from BOTH editor content and user message
                # Check editor first, then fall back to message
                persona_text = extract_section('Persona', ec) or extract_section('Persona', um)
                background_text = extract_section('Background', ec) or extract_section('Background', um)
                article_1_text = extract_section('Article 1', ec) or extract_section('Article', ec) or extract_section('Article 1', um) or extract_section('Article', um)
                article_2_text = extract_section('Article 2', ec) or extract_section('Article 2', um)
                article_3_text = extract_section('Article 3', ec) or extract_section('Article 3', um)
                tweets_text = extract_section('Tweet', ec) or extract_section('Tweets', ec) or extract_section('Tweet', um) or extract_section('Tweets', um)

                logger.info(f"📝 SUBSTACK AGENT: Extracted persona length: {len(persona_text or '')} (from {'editor' if extract_section('Persona', ec) else 'message'})")
                logger.info(f"📝 SUBSTACK AGENT: Extracted background length: {len(background_text or '')}")
                logger.info(f"📝 SUBSTACK AGENT: Extracted article 1 length: {len(article_1_text or '')}")
                logger.info(f"📝 SUBSTACK AGENT: Extracted article 2 length: {len(article_2_text or '')}")
                logger.info(f"📝 SUBSTACK AGENT: Extracted article 3 length: {len(article_3_text or '')}")
                logger.info(f"📝 SUBSTACK AGENT: Extracted tweets length: {len(tweets_text or '')}")

            except Exception as e:
                logger.warning(f"📝 SUBSTACK AGENT: Section extraction failed: {e}")
                persona_text = None
                background_text = None
                article_1_text = None
                article_2_text = None
                article_3_text = None
                tweets_text = None

            # Fetch URL content if provided
            fetched_content = None
            if user_has_url and url:
                try:
                    logger.info(f"📝 SUBSTACK AGENT: Fetching content from URL: {url}")
                    try:
                        from services.langgraph_tools.langgraph_tools import crawl_web_content as crawl4ai_tool
                        logger.info(f"📝 SUBSTACK AGENT: Using Crawl4AI for article extraction")
                        fetched = await crawl4ai_tool(url=url)
                        logger.info(f"📝 SUBSTACK AGENT: Crawl4AI fetch result type: {type(fetched)}")

                        if isinstance(fetched, dict):
                            fetched_content = fetched.get("content") or fetched.get("text")
                            logger.info(f"📝 SUBSTACK AGENT: Crawl4AI content length: {len(fetched_content or '')}")
                            if fetched_content:
                                logger.info(f"📝 SUBSTACK AGENT: Content preview (first 300 chars): {fetched_content[:300]}")
                        elif isinstance(fetched, str):
                            fetched_content = fetched
                            logger.info(f"📝 SUBSTACK AGENT: String content length: {len(fetched_content)}")
                    except Exception as crawl4ai_error:
                        logger.warning(f"📝 SUBSTACK AGENT: Crawl4AI failed, falling back to basic fetch: {crawl4ai_error}")
                        crawl_tool = await self._get_tool_function("crawl_web_content")
                        if crawl_tool:
                            fetched = await crawl_tool(url=url)
                            if isinstance(fetched, dict):
                                fetched_content = fetched.get("content") or fetched.get("text")
                except Exception as e:
                    logger.error(f"📝 SUBSTACK AGENT: Failed to fetch URL {url}: {e}")
                    fetched_content = None

            # Build content sections for LLM
            content_sections = []
            if persona_text:
                content_sections.append("=== PERSONA ===\n" + persona_text)
            if fetched_content:
                content_sections.append("=== FETCHED ARTICLE ===\n" + fetched_content)
            if background_text:
                content_sections.append("=== BACKGROUND ===\n" + background_text)
            if article_1_text:
                content_sections.append("=== ARTICLE 1 ===\n" + article_1_text)
            if article_2_text:
                content_sections.append("=== ARTICLE 2 ===\n" + article_2_text)
            if article_3_text:
                content_sections.append("=== ARTICLE 3 ===\n" + article_3_text)
            if tweets_text:
                content_sections.append("=== TWEETS ===\n" + tweets_text)

            content_block = "\n\n".join(content_sections) + ("\n\n" if content_sections else "")

            # Add URL fetch failure note if applicable
            if user_has_url and url and not fetched_content:
                content_block += f"=== URL FETCH FAILED ===\n"
                content_block += f"Could not fetch content from: {url}\n"
                content_block += f"Please verify the URL is accessible and try again.\n\n"

            # === RESEARCH PHASE ===
            # Generate research plan and execute if needed (skip for tweet mode)
            chat_service = await self._get_chat_service()
            try:
                from services.settings_service import settings_service
                tc_model = await settings_service.get_text_completion_model()
                model_name = tc_model or await self._get_model_name()
            except Exception:
                model_name = await self._get_model_name()

            research_findings = []
            
            # Skip research phase for tweet-sized content (sources already provided)
            if not tweet_mode:
                research_helper = SubstackResearchHelper(chat_service)
                
                # Collect article texts for research planning
                article_texts = [t for t in [article_1_text, article_2_text, article_3_text, fetched_content] if t]
                
                research_plan = await research_helper.generate_research_plan(
                user_query=user_message,
                article_texts=article_texts,
                tweets_text=tweets_text,
                background_text=background_text,
                model_name=model_name
                )
                
                logger.info(f"📝 RESEARCH PLAN: needs_research={research_plan.needs_research}, questions={len(research_plan.research_questions)}")
                
                # Execute research if plan indicates it's needed
                if research_plan.needs_research and research_plan.research_questions:
                    logger.info(f"🔍 EXECUTING RESEARCH: {len(research_plan.research_questions)} questions")
                    
                    # Substack agent has blanket permission to search web (no HITL needed)
                    # This is a writing/research agent, web search is expected behavior
                    has_web_permission = True
                    
                    # Execute research freely
                    research_findings = await research_helper.execute_research(
                        research_plan,
                        has_web_permission
                    )
                    
                    logger.info(f"✅ RESEARCH COMPLETE: {len(research_findings)} findings")
                    
                    # Add research findings to content block
                    research_block = research_helper.format_research_for_article_prompt(research_findings)
                    if research_block:
                        content_block += research_block + "\n\n"

            # Compute topic line
            if tweet_mode:
                topic_line = "Task: Generate tweet-sized social media content from provided sources"
            else:
                topic_line = f"Topic: {req.topic}"
                if fetched_content or background_text or article_1_text or article_2_text or article_3_text or tweets_text:
                    topic_line = "Topic: Long-form article synthesizing provided source materials"
                if research_findings:
                    topic_line += " (enhanced with research)"

            # Build task block
            task_block = (
                "=== REQUEST ===\n"
                f"{topic_line}\n"
                f"User instruction: {user_message.strip()}\n\n"
                f"Target length: {req.target_length_words} words\n"
                f"Tone: {req.tone}\n"
                f"Style: {req.style}\n"
                f"Include citations: {req.include_citations}\n"
                f"Include conclusion: {req.include_conclusion}\n\n"
                + (f"Source URL: {url}\n\n" if user_has_url and url else "")
                + policy
            )

            # Construct messages: content first, then task
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": now_line},
            ]
            if content_block:
                messages.append({"role": "user", "content": content_block})
            messages.append({"role": "user", "content": task_block})

            # Execute LLM (chat_service and model_name already initialized in research phase)
            logger.info(f"📝 SUBSTACK AGENT: Using model: {model_name}")
            logger.info(f"📝 SUBSTACK AGENT: Message count: {len(messages)}")
            logger.info(f"📝 SUBSTACK AGENT: Content block length: {len(content_block)}")
            logger.info(f"📝 SUBSTACK AGENT: Task block length: {len(task_block)}")

            response = await chat_service.openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.4,  # Slightly higher than podcast for more creative writing
            )

            content = response.choices[0].message.content or "{}"
            logger.info(f"📝 SUBSTACK AGENT: LLM response length: {len(content)}")
            logger.info(f"📝 SUBSTACK AGENT: LLM response preview: {content[:200]}...")

            # Parse structured response
            try:
                import json
                text = content.strip()
                if '```json' in text:
                    m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                    if m:
                        text = m.group(1).strip()
                elif '```' in text:
                    text = text.replace('```', '').strip()
                data = json.loads(text)

                # Fix metadata if needed
                if "metadata" in data and isinstance(data["metadata"], dict):
                    fixed_metadata = {}
                    for key, value in data["metadata"].items():
                        if isinstance(value, str):
                            try:
                                if '.' in value:
                                    fixed_metadata[key] = float(value)
                                else:
                                    fixed_metadata[key] = int(value)
                            except ValueError:
                                fixed_metadata[key] = value
                        else:
                            fixed_metadata[key] = value
                    data["metadata"] = fixed_metadata

                structured = SubstackArticleResponse(**data)

            except Exception as e:
                logger.error(f"❌ Substack JSON parse failed: {e}")
                
                # FALLBACK: If in tweet mode and LLM returned raw text, wrap it
                if tweet_mode and content.strip() and not content.strip().startswith('{'):
                    logger.info(f"🔧 FALLBACK: Wrapping raw tweet content in JSON structure")
                    char_count = len(content.strip())
                    structured = SubstackArticleResponse(
                        task_status="complete",
                        article_text=content.strip(),
                        metadata={
                            "word_count": char_count,
                            "reading_time_minutes": 0,
                            "section_count": 1,
                        }
                    )
                else:
                    # True error - return error response
                    error_text = f"SUBSTACK AGENT ERROR:\n\nFailed to parse LLM response: {e}\n\nRaw response: {content[:500]}...\n\nPlease check the agent logs for details."
                    structured = SubstackArticleResponse(
                        task_status="error",
                        article_text=error_text,
                        metadata={
                            "word_count": len(error_text.split()),
                            "reading_time_minutes": 0,
                            "section_count": 0,
                        },
                    )

            # Prepare agent results
            agent_results = {
                "structured_response": structured.dict(),
                "timestamp": datetime.now().isoformat(),
            }

            state["agent_results"] = agent_results
            state["latest_response"] = structured.article_text
            state["is_complete"] = structured.task_status == "complete"
            return state

        except Exception as e:
            logger.error(f"❌ Substack Agent failed: {e}")
            return self._create_error_result(str(e))

