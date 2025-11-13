"""
Podcast Script Agent - Roosevelt's Single-Narrator ElevenLabs Scribe

Generates a plain-text podcast script with inline bracket cues suitable
for pasting directly into ElevenLabs TTS.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from .base_agent import BaseAgent, TaskStatus
from models.podcast_models import PodcastScriptRequest, PodcastScriptResponse
import re


logger = logging.getLogger(__name__)


class PodcastScriptAgent(BaseAgent):
    def __init__(self):
        super().__init__("podcast_script_agent")
        logger.info("🎙️ BULLY! Podcast Script Agent mounted and ready to draft!")

    def _build_system_prompt(self, persona: Dict[str, Any] | None) -> str:
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
            "- [drawn out] - Extended, stretched pronunciation (e.g., 'Sooooo...')\n\n"
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
            "- 'They're trying to—to TELL US that...', 'This is just—ugh—DISGUSTING'\n"
            "- Be creative: 'Wh...what?!', 'You've GOT to be—', 'I mean, come ON!'\n\n"
            "SOUND EFFECTS (optional):\n"
            "- [sfx:door-knock], [sfx:city-ambience-low] when allowed.\n\n"
            "ALL CAPS WORDS provide additional emphasis and urgency when needed.\n\n"
            "=== STRUCTURE TEMPLATE ===\n"
            "Single-segment commentary (no intro/outro structure needed):\n\n"
            "CRITICAL: Plain text ONLY. No markdown, no code fences, no SSML.\n"
            "CRITICAL: Keep short paragraphs for breath; tasteful em-dashes.\n"
            "CRITICAL: Use bracket cues FREQUENTLY and ANIMATEDLY for dynamic delivery.\n"
            "CRITICAL: Encourage [emphasis], [speed up], [hurried], ALL CAPS for impact.\n"
            "CRITICAL: USE EMOTIONAL CUES LIBERALLY - [exasperated], [disgusted], [outraged], [shouting], [furious], [mocking], [incensed].\n"
            "CRITICAL: DON'T BE AFRAID TO SHOUT for emphasis - this adds passion and engagement!\n"
            "CRITICAL: Include NATURAL STAMMERING when excited/angry: 'F...folks', 'I—I can't believe', 'ugh', 'This is just—'. Be creative!\n"
            "CRITICAL: STRICT LENGTH LIMIT - Maximum 3,000 characters total. Be punchy and impactful, not verbose!\n"
        )
        try:
            persona_prompt = self._build_persona_prompt(persona)
        except Exception:
            persona_prompt = ""
        return base + persona_prompt

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"🎙️ PODCAST AGENT: Starting process with state keys: {list(state.keys())}")
            shared_memory = state.get("shared_memory", {}) or {}
            active_editor = shared_memory.get("active_editor", {}) or {}
            logger.info(f"🎙️ PODCAST AGENT: Active editor keys: {list(active_editor.keys())}")

            # Gating: require editor open with frontmatter.type == 'podcast'
            frontmatter = active_editor.get("frontmatter", {}) or {}
            doc_type = str(frontmatter.get("type", "")).strip().lower()
            logger.info(f"🎙️ PODCAST AGENT: Document type: '{doc_type}'")
            if doc_type != "podcast":
                logger.info("🎙️ Podcast agent skipped: active editor type is not 'podcast'.")
                return self._create_success_result(
                    response="Active editor is not a podcast doc; podcast agent skipping.",
                    tools_used=[],
                    processing_time=0.0,
                    shared_memory_updates={},
                    additional_data={"skipped": True},
                )

            # Extract request hints from latest user message if available
            user_message = state.get("current_query", "") or ""
            # Also get editor content for persona extraction
            editor_content = active_editor.get("content", "") or ""
            logger.info(f"🎙️ PODCAST AGENT: Editor content length: {len(editor_content)}")
            # Defaults; prioritize user query over file title
            # Extract topic from user message first, fall back to frontmatter
            user_topic = None
            if user_message.strip():
                # Look for explicit topic requests
                topic_match = re.search(r'(?:topic|subject|about):\s*([^\n]+)', user_message, re.IGNORECASE)
                if topic_match:
                    user_topic = topic_match.group(1).strip()
                else:
                    # Use first line of user message as topic if it looks like a request
                    first_line = user_message.split('\n')[0].strip()
                    if len(first_line) > 10 and not first_line.startswith('##'):
                        user_topic = first_line
            
            req = PodcastScriptRequest(
                topic=user_topic or frontmatter.get("title") or frontmatter.get("topic") or "Podcast Episode",
                target_length_words=int(frontmatter.get("target_length_words") or 900),
                tone=str(frontmatter.get("tone") or "warm").lower() if str(frontmatter.get("tone") or "").strip() else "warm",
                pacing=str(frontmatter.get("pacing") or "moderate").lower() if str(frontmatter.get("pacing") or "").strip() else "moderate",
                include_music_cues=bool(frontmatter.get("include_music_cues", False)),
                include_sfx_cues=bool(frontmatter.get("include_sfx_cues", False)),
            )

            persona = state.get("persona")
            system_prompt = self._build_system_prompt(persona)
            now_line = f"Current Date/Time: {datetime.now().isoformat()}"

            # Build deterministic instructions and JSON schema requirement
            schema_hint = (
                "STRUCTURED OUTPUT REQUIRED: Respond ONLY with valid JSON for PodcastScriptResponse. Fields: "
                "task_status (complete|incomplete|error), script_text (string), metadata (object with words, "
                "estimated_duration_sec, tag_counts map)."
            )

            # Content policy - UNIFIED for both monologue and dialogue
            # Trust the LLM to choose the format based on user's natural language instructions!
            policy = (
                "FORMAT GUIDELINES (The LLM decides: monologue vs dialogue based on user request):\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "- Read the user's query/request carefully for format and behavioral instructions.\n"
                "- If they request 'monologue' or 'commentary' → produce single-narrator script WITHOUT speaker labels\n"
                "- If they request 'dialogue', 'debate', 'conversation', 'interview', or mention multiple speakers → produce multi-speaker script WITH speaker labels (HOST:, CALLER:, GUEST:, etc.)\n"
                "- If unclear, default to MONOLOGUE format.\n"
                "- If PERSONA DEFINITIONS section is provided, use those character descriptions to inform speech patterns, perspectives, personalities, and conversational dynamics.\n"
                "- Follow ANY stylistic or tonal instructions from the user precisely.\n\n"
                
                "MULTI-SPEAKER DIALOGUE FORMAT (only when user requests dialogue/conversation):\n"
                "- Use clear speaker labels: HOST:, CALLER:, GUEST:, CO-HOST: (or custom names like KIRK:, CARSON: if provided in persona)\n"
                "- MULTI-CHARACTER INTERACTION TAGS (ElevenLabs v3):\n"
                "  * [interrupting] - Speaker cuts in while other is talking\n"
                "  * [overlapping] - Both speakers talking simultaneously\n"
                "  * [cuts in] - Sharp interruption\n"
                "  * [interjecting] - Quick insertion during other's speech\n"
                "  * [starting to speak] - Beginning before being cut off\n"
                "- Let PERSONA DEFINITIONS and USER INSTRUCTIONS guide conversational style:\n"
                "  * Combative/firebrand personalities → frequent [interrupting], [overlapping], [cuts in]\n"
                "  * Respectful/academic types → complete thoughts, turn-taking, thoughtful responses\n"
                "  * Interview format → interviewer asks questions, interviewee provides full answers\n"
                "- Use em-dashes (—) for interrupted speech, ellipses (...) for trailing off\n"
                "- Both speakers use emotional bracket cues: [mocking], [defensive], [thoughtful], [agreeing], etc.\n"
                "- Character-specific voice cues (optional): [childlike tone], [deep voice], [pirate voice], [robotic tone]\n\n"
                
                "SINGLE-NARRATOR MONOLOGUE FORMAT (default unless user requests dialogue):\n"
                "- Single narrator voice; NO character names or speaker labels\n"
                "- Speak directly to the listener as the narrator\n"
                "- Do NOT self-identify by name in the script\n\n"
                
                "EMOTIONAL DELIVERY & BRACKET CUES (for both formats):\n"
                "- Use bracket cues FREQUENTLY and ANIMATEDLY for dynamic delivery\n"
                "- EMOTIONAL CUES: [exasperated], [disgusted], [outraged], [shouting], [furious], [mocking], [incensed], [ranting], [laughing], [sighs]\n"
                "- PACING/EMPHASIS: [emphasis], [speed up], [hurried], [rapid-fire], [stressed], [deliberate], [rushed], [slows down]\n"
                "- ADD NATURAL STAMMERING when excited/angry: 'F...folks', 'I—I can't believe', 'ugh', 'gah', 'This is just—DISGUSTING'\n"
                "- Use ellipses (...) and em-dashes (—) creatively for hesitation and interruption\n"
                "- ALL CAPS for emphasis and passionate moments\n"
                "- ABSOLUTELY DO NOT use [music:...] or [pause:...] or [beat] or [breath] tags\n"
                "- If include_sfx_cues is false, omit [sfx:...] entirely\n\n"
                
                "CONTENT GROUNDING & STRUCTURE:\n"
                "- Ignore the file title/filename entirely when BACKGROUND/ARTICLE/TWEETS/FETCHED CONTENT is present\n"
                "- Ground every segment in provided content: reference at least 3 concrete specifics (names, quotes, claims, numbers, @usernames)\n"
                "- Quote key lines and react to them with commentary\n"
                "- Do NOT discuss podcasting craft; produce topical commentary on the provided material\n"
                "- If MULTIPLE ARTICLES are present (Article, Article 2, Article 3), weave them together naturally\n"
                "- If TWEETS section is present, cite specific @usernames and quote exact tweet text\n"
                "- If both BACKGROUND and ARTICLE(s)/TWEETS are present, integrate all sources seamlessly\n"
                "- Produce SINGLE-SEGMENT commentary only (no intro/outro structure)\n"
                "- STRICT LENGTH LIMIT: Maximum 3,000 characters total (including all cues and labels). Keep it punchy!\n"
                )

            # Initialize optional content holders
            fetched_content = None
            background_text = None
            article_text = None

            # Detect URL in user message to optionally fetch/analyze content
            user_has_url = False
            url = None
            try:
                url_match = re.search(r'https?://[^\s)>\"]+', user_message)
                if url_match:
                    url = url_match.group(0)
                    user_has_url = True
                    logger.info(f"🎙️ PODCAST AGENT: Detected URL: {url}")
            except Exception as e:
                logger.warning(f"🎙️ PODCAST AGENT: URL detection failed: {e}")
                pass

            # Detect explicit request for music cues in user message
            user_requests_music = False
            try:
                if "[music:" in user_message.lower() or "music cue" in user_message.lower():
                    user_requests_music = True
            except Exception:
                pass

            # Detect style/perspective hints (keep these for backwards compatibility)
            requested_style = None
            requested_perspective = None
            try:
                m = re.search(r"style of\s+([^\n,]+)", user_message, flags=re.IGNORECASE)
                if m:
                    requested_style = m.group(1).strip()
                # Simple perspective keywords
                lower_msg = user_message.lower()
                for key in ["conservative", "liberal", "progressive", "libertarian", "populist", "centrist"]:
                    if key in lower_msg:
                        requested_perspective = key
                        break
            except Exception:
                pass

            # Detect pasted Background/Article/Persona sections (including multiple articles)
            # Check BOTH editor content and user message, prioritize editor content for Persona
            try:
                # Normalize newlines
                um = user_message.replace('\r\n', '\n')
                ec = editor_content.replace('\r\n', '\n')
                def extract_section(title: str, text: str) -> str | None:
                    # Try both formats: "## Title" (markdown) and "Title:" (simple)
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
                    
                    # capture until next heading starting with ## or Title: or end
                    collected = []
                    for line in lines[start:]:
                        stripped = line.strip()
                        # Stop at next markdown heading or simple heading
                        if stripped.startswith('## ') or re.match(r'^(background|article|##)\s*:', stripped, re.IGNORECASE):
                            break
                        collected.append(line)
                    body = '\n'.join(collected).strip()
                    return body or None

                # Extract persona from EDITOR CONTENT first (preferred), fallback to user message
                persona_text = extract_section('Persona', ec) or extract_section('Persona', um)
                # Extract other sections from user message (these are typically pasted in chat)
                background_text = extract_section('Background', um)
                article_text = extract_section('Article', um)
                article_2_text = extract_section('Article 2', um)
                article_3_text = extract_section('Article 3', um)
                tweets_text = extract_section('Tweets', um)
                logger.info(f"🎙️ PODCAST AGENT: Extracted persona length: {len(persona_text or '')} (from {'editor' if extract_section('Persona', ec) else 'message'})")
                logger.info(f"🎙️ PODCAST AGENT: Extracted background length: {len(background_text or '')}")
                logger.info(f"🎙️ PODCAST AGENT: Extracted article length: {len(article_text or '')}")
                logger.info(f"🎙️ PODCAST AGENT: Extracted article 2 length: {len(article_2_text or '')}")
                logger.info(f"🎙️ PODCAST AGENT: Extracted article 3 length: {len(article_3_text or '')}")
                logger.info(f"🎙️ PODCAST AGENT: Extracted tweets length: {len(tweets_text or '')}")
            except Exception as e:
                logger.warning(f"🎙️ PODCAST AGENT: Section extraction failed: {e}")
                persona_text = None
                background_text = background_text or None
                article_text = article_text or None
                article_2_text = None
                article_3_text = None
                tweets_text = None

            # Compose messages
            # Compute topic line; override if we have source content
            topic_line = f"Topic: {req.topic}"
            if fetched_content or background_text or article_text or article_2_text or article_3_text or tweets_text:
                topic_line = "Topic: Commentary on provided Background/Article/Tweets/Source content"

            task_block = (
                "=== REQUEST ===\n"
                f"{topic_line}\n"
                f"User instruction: {user_message.strip()}\n\n"
                f"Target length (words): {req.target_length_words}\n"
                f"Tone: {req.tone}\n"
                f"Pacing: {req.pacing}\n"
                f"Include music cues: {req.include_music_cues or user_requests_music}\n"
                f"Include SFX cues: {req.include_sfx_cues}\n"
                "MAXIMUM OUTPUT LENGTH: 3,000 characters (including bracket cues). Keep it punchy and focused!\n\n"
                + (f"Source URL: {url}\n\n" if user_has_url and url else "")
                + (f"Requested style: {requested_style}\n" if requested_style else "")
                + (f"Perspective: {requested_perspective}\n\n" if requested_perspective else "\n")
                + policy + "\n\n"
                + "=== FORMAT EXAMPLES ===\n\n"
                "MONOLOGUE EXAMPLE (single narrator, no speaker labels):\n"
                "Tweets: @SenSchumer: 'Even Trump voters agree!' @ConservativeVoice: 'This poll is total garbage.'\n"
                "Output:\n"
                "[excited] Okay folks, [emphasis] TWEETS OF THE DAY! @SenSchumer tweets: 'Even Trump voters agree!' [mocking] Oh REALLY, Chuck? @ConservativeVoice nails it: 'This poll is total GARBAGE.' [shouting] EXACTLY! The reality? It's pressure politics dressed up as inevitability.\n\n"
                "DIALOGUE EXAMPLE (multi-speaker with labels and v3 tags):\n"
                "Tweets: @SenSchumer: 'Even Trump voters agree!' @ConservativeVoice: 'This poll is garbage.'\n"
                "Output:\n"
                "HOST: [excited] Alright folks, we've got caller Carson on the line! Carson, you see this Schumer tweet?\n\n"
                "CALLER: [starting to speak] Well, actually, the poll methodology—\n\n"
                "HOST: [interrupting] NO NO NO! [mocking] 'The methodology'—listen to this! @ConservativeVoice already DEMOLISHED this garbage poll!\n\n"
                "CALLER: [interjecting] But if you'd let me—\n\n"
                "HOST: [overlapping] [laughing] Folks, you hear this? [shouting] The poll is RIGGED and he wants to talk methodology!\n\n"
                + schema_hint
            )

            # NOTE: messages will be constructed AFTER any fetched/background/article content is appended

            # If a URL is present, try to fetch content via centralized tools
            fetched_content = None
            if user_has_url and url:
                try:
                    # Try Crawl4AI first for better article extraction
                    logger.info(f"🎙️ PODCAST AGENT: Fetching content from URL: {url}")
                    try:
                        from services.langgraph_tools.langgraph_tools import crawl_web_content as crawl4ai_tool
                        logger.info(f"🎙️ PODCAST AGENT: Using Crawl4AI for article extraction")
                        fetched = await crawl4ai_tool(url=url)
                        logger.info(f"🎙️ PODCAST AGENT: Crawl4AI fetch result type: {type(fetched)}")
                        
                        if isinstance(fetched, dict):
                            # Crawl4AI returns dict with 'content' key containing extracted text
                            fetched_content = fetched.get("content") or fetched.get("text")
                            logger.info(f"🎙️ PODCAST AGENT: Crawl4AI fetch result keys: {list(fetched.keys())}")
                            logger.info(f"🎙️ PODCAST AGENT: Crawl4AI content length: {len(fetched_content or '')}")
                            if fetched_content:
                                logger.info(f"🎙️ PODCAST AGENT: Content preview (first 500 chars): {fetched_content[:500]}")
                        elif isinstance(fetched, str):
                            fetched_content = fetched
                            logger.info(f"🎙️ PODCAST AGENT: String content length: {len(fetched_content)}")
                    except Exception as crawl4ai_error:
                        logger.warning(f"🎙️ PODCAST AGENT: Crawl4AI failed, falling back to basic fetch: {crawl4ai_error}")
                        # Fallback to basic crawl_web_content
                        crawl_tool = await self._get_tool_function("crawl_web_content")
                        if crawl_tool:
                            fetched = await crawl_tool(url=url)
                            logger.info(f"🎙️ PODCAST AGENT: Basic fetch result type: {type(fetched)}")
                            if isinstance(fetched, dict):
                                fetched_content = fetched.get("content") or fetched.get("text")
                                logger.info(f"🎙️ PODCAST AGENT: Basic fetch result keys: {list(fetched.keys())}")
                                logger.info(f"🎙️ PODCAST AGENT: Basic content length: {len(fetched_content or '')}")
                                if fetched_content:
                                    logger.info(f"🎙️ PODCAST AGENT: Content preview (first 500 chars): {fetched_content[:500]}")
                except Exception as e:
                    logger.error(f"🎙️ PODCAST AGENT: Failed to fetch URL {url}: {e}")
                    fetched_content = None

            # Note: fetched_content is already handled in content_sections below

            # Build content-first block to maximize grounding
            content_sections = []
            if persona_text:
                content_sections.append("=== PERSONA DEFINITIONS ===\n" + persona_text)
            if fetched_content:
                content_sections.append("=== FETCHED CONTENT ===\n" + fetched_content)
            if background_text:
                content_sections.append("=== BACKGROUND ===\n" + background_text)
            if article_text:
                content_sections.append("=== ARTICLE ===\n" + article_text)
            if article_2_text:
                content_sections.append("=== ARTICLE 2 ===\n" + article_2_text)
            if article_3_text:
                content_sections.append("=== ARTICLE 3 ===\n" + article_3_text)
            if tweets_text:
                content_sections.append("=== TWEETS ===\n" + tweets_text)

            content_block = "\n\n".join(content_sections) + ("\n\n" if content_sections else "")
            
            # If URL fetch failed but URL was provided, add a note to the user
            if user_has_url and url and not fetched_content:
                content_block += f"=== URL FETCH FAILED ===\n"
                content_block += f"Could not fetch content from: {url}\n"
                content_block += f"This may be due to:\n"
                content_block += f"- URL accessibility issues\n"
                content_block += f"- Anti-bot protection\n"
                content_block += f"- Network connectivity problems\n"
                content_block += f"Please verify the URL is accessible and try again.\n\n"

            # Now construct messages: content first, then strict task
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": now_line},
            ]
            if content_block:
                messages.append({"role": "user", "content": content_block})
            messages.append({"role": "user", "content": task_block})

            # Execute LLM
            chat_service = await self._get_chat_service()
            try:
                from services.settings_service import settings_service
                tc_model = await settings_service.get_text_completion_model()
                model_name = tc_model or await self._get_model_name()
            except Exception:
                model_name = await self._get_model_name()

            logger.info(f"🎙️ PODCAST AGENT: Using model: {model_name}")
            logger.info(f"🎙️ PODCAST AGENT: Message count: {len(messages)}")
            logger.info(f"🎙️ PODCAST AGENT: Content block length: {len(content_block)}")
            logger.info(f"🎙️ PODCAST AGENT: Task block length: {len(task_block)}")

            response = await chat_service.openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
            )

            content = response.choices[0].message.content or "{}"
            logger.info(f"🎙️ PODCAST AGENT: LLM response length: {len(content)}")
            logger.info(f"🎙️ PODCAST AGENT: LLM response preview: {content[:200]}...")

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
                
                # Fix metadata if it contains error strings instead of proper types
                if "metadata" in data and isinstance(data["metadata"], dict):
                    fixed_metadata = {}
                    for key, value in data["metadata"].items():
                        if isinstance(value, str):
                            # Try to convert string numbers, otherwise skip
                            try:
                                if '.' in value:
                                    fixed_metadata[key] = float(value)
                                else:
                                    fixed_metadata[key] = int(value)
                            except ValueError:
                                # Skip non-numeric strings
                                continue
                        else:
                            fixed_metadata[key] = value
                    data["metadata"] = fixed_metadata
                
                structured = PodcastScriptResponse(**data)
                # Post-process to strip disallowed tags ([pause:*], [beat], [breath])
                try:
                    script = structured.script_text or ""
                    script = re.sub(r"\[pause:[^\]]+\]", "", script, flags=re.IGNORECASE)
                    script = re.sub(r"\[(?:beat|breath)\]", "", script, flags=re.IGNORECASE)
                    structured.script_text = script
                except Exception:
                    pass

                # Heuristic grounding check: ensure anchor terms from content appear; if not, retry with stricter constraints
                try:
                    anchors: list[str] = []
                    source_blob = "\n".join([s for s in [fetched_content, background_text, article_text, article_2_text, article_3_text, tweets_text] if s])
                    if source_blob:
                        # Extract candidate anchors (capitalized tokens and key nouns)
                        cand = re.findall(r"\b[A-Z][A-Za-z\-]{2,}(?:\s+[A-Z][A-Za-z\-]{2,})?\b", source_blob)
                        # De-duplicate and filter common words
                        stop = set(["The","This","That","And","But","For","With","From","Into","About","After","Before","During","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday","January","February","March","April","May","June","July","August","September","October","November","December"]) 
                        uniq = []
                        for w in cand:
                            w2 = w.strip()
                            if w2 not in stop and w2 not in uniq and len(uniq) < 8:
                                uniq.append(w2)
                        anchors = uniq[:5]
                    missing = 0
                    if anchors:
                        script_l = (structured.script_text or "").lower()
                        for a in anchors:
                            if a.lower() not in script_l:
                                missing += 1
                    if anchors and missing >= max(1, len(anchors)//2):
                        # Retry with MUST-USE anchors and include 1-2 direct quotes if available
                        quotes = re.findall(r"“([^”]{20,200})”|\"([^\"]{20,200})\"", source_blob)
                        quote_lines = []
                        for q in quotes:
                            qtext = q[0] or q[1]
                            if qtext:
                                quote_lines.append(qtext.strip())
                            if len(quote_lines) >= 2:
                                break
                        must_block = (
                            "\n=== MANDATORY ANCHORS ===\n"
                            + "\n".join(f"- USE BY NAME: {a}" for a in anchors)
                            + ("\n\n=== QUOTES TO INCORPORATE ===\n" + "\n".join(f"- \"{q}\"" for q in quote_lines) if quote_lines else "")
                            + "\n\nSTRICT REQUIREMENT: Weave these anchors and quotes directly; cite names explicitly. Do NOT discuss podcast craft."
                        )
                        retry_messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "system", "content": now_line},
                            {"role": "user", "content": must_block + "\n\n" + user_block},
                        ]
                        retry_response = await chat_service.openai_client.chat.completions.create(
                            model=model_name,
                            messages=retry_messages,
                            temperature=0.2,
                        )
                        retry_content = retry_response.choices[0].message.content or "{}"
                        try:
                            t = retry_content.strip()
                            if '```json' in t:
                                m2 = re.search(r'```json\s*\n([\s\S]*?)\n```', t)
                                if m2:
                                    t = m2.group(1).strip()
                            elif '```' in t:
                                t = t.replace('```', '').strip()
                            d2 = json.loads(t)
                            sr2 = PodcastScriptResponse(**d2)
                            scr2 = sr2.script_text or ""
                            scr2 = re.sub(r"\[pause:[^\]]+\]", "", scr2, flags=re.IGNORECASE)
                            scr2 = re.sub(r"\[(?:beat|breath)\]", "", scr2, flags=re.IGNORECASE)
                            structured = sr2.copy(update={"script_text": scr2})
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"❌ Podcast JSON parse failed: {e}")
                # Return error information instead of fake content
                error_text = f"PODCAST AGENT ERROR:\n\nFailed to parse LLM response: {e}\n\nRaw response: {content[:500]}...\n\nPlease check the agent logs for details."
                structured = PodcastScriptResponse(
                    task_status="error",
                    script_text=error_text,
                    metadata={
                        "words": len(error_text.split()),
                        "estimated_duration_sec": 0,
                        "tag_counts": {},
                    },
                )

            # Prepare agent results
            agent_results = {
                "structured_response": structured.dict(),
                "timestamp": datetime.now().isoformat(),
            }

            state["agent_results"] = agent_results
            state["latest_response"] = structured.script_text
            state["is_complete"] = structured.task_status == "complete"
            return state

        except Exception as e:
            logger.error(f"❌ Podcast Script Agent failed: {e}")
            return self._create_error_result(str(e))


