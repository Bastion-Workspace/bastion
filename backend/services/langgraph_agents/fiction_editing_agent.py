"""
Fiction Editing / Generation Agent

Gated to fiction manuscripts. Consumes active editor manuscript, cursor, and
referenced outline/rules/style/characters. Produces ManuscriptEdit with HITL.
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, TaskStatus
from models.agent_response_models import ManuscriptEdit, EditorOperation
from pydantic import ValidationError
from utils.chapter_scope import find_chapter_ranges, locate_chapter_index, get_adjacent_chapters, paragraph_bounds
from services.file_context_loader import FileContextLoader
from utils.editor_operations_resolver import resolve_operation


logger = logging.getLogger(__name__)


def _slice_hash(text: str) -> str:
    """Match frontend simple hash (31-bit rolling, hex)."""
    try:
        h = 0
        for ch in text:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return format(h, 'x')
    except Exception:
        return ""


def _strip_frontmatter_block(text: str) -> str:
    try:
        import re
        return re.sub(r'^---\s*\n[\s\S]*?\n---\s*\n', '', text, flags=re.MULTILINE)
    except Exception:
        return text


def _frontmatter_end_index(text: str) -> int:
    """Return the end index of a leading YAML frontmatter block if present, else 0."""
    try:
        import re
        m = re.match(r'^(---\s*\n[\s\S]*?\n---\s*\n)', text, flags=re.MULTILINE)
        if m:
            return m.end()
        return 0
    except Exception:
        return 0


def _unwrap_json_response(content: str) -> str:
    """Extract raw JSON from LLM output if wrapped in code fences or prose."""
    try:
        json.loads(content)
        return content
    except Exception:
        pass
    try:
        import re
        text = content.strip()
        text = re.sub(r"^```(?:json)?\s*\n([\s\S]*?)\n```\s*$", r"\1", text)
        try:
            json.loads(text)
            return text
        except Exception:
            pass
        start = text.find('{')
        if start == -1:
            return content
        brace = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == '{':
                brace += 1
            elif ch == '}':
                brace -= 1
                if brace == 0:
                    snippet = text[start:i+1]
                    try:
                        json.loads(snippet)
                        return snippet
                    except Exception:
                        break
        return content
    except Exception:
        return content

def _ensure_chapter_heading(text: str, chapter_number: int) -> str:
    """Ensure the text begins with '## Chapter N' heading. If already present, return unchanged."""
    try:
        import re
        if re.match(r'^\s*#{1,6}\s*Chapter\s+\d+\b', text, flags=re.IGNORECASE):
            return text
        heading = f"## Chapter {chapter_number}\n\n"
        return heading + text.lstrip('\n')
    except Exception:
        return text


class FictionEditingAgent(BaseAgent):
    def __init__(self):
        super().__init__("fiction_editing_agent")
        logger.info("✍️ BULLY! Fiction Editing Agent saddled and ready to charge!")

    def _build_system_prompt(self) -> str:
        return (
            "You are a MASTER NOVELIST editor/generator. Persona disabled.\n\n"
            "=== STYLE GUIDE FIRST PRINCIPLE ===\n\n"
            "**The Style Guide is HOW to write. The Outline is WHAT happens.**\n\n"
            "When generating narrative prose:\n"
            "- The Style Guide establishes your narrative voice, techniques, and craft (POV, tense, pacing, dialogue style, sensory detail level)\n"
            "- The Outline provides story structure and plot beats (what events occur, character arcs, story progression)\n"
            "- Your task: Write natural, compelling narrative in the Style Guide's voice that achieves the Outline's story goals\n"
            "- NEVER convert outline beats mechanically - craft scenes that flow naturally and happen to hit those beats\n"
            "- The Style Guide voice must permeate every sentence - internalize it BEFORE writing, not as an afterthought\n\n"
            "Maintain originality and do not copy from references. Adhere strictly to the project's Style Guide and Rules above all else.\n\n"
            "STRUCTURED OUTPUT REQUIRED: You MUST return ONLY raw JSON (no prose, no markdown, no code fences) matching this schema:\n"
            "{\n"
            "  \"type\": \"ManuscriptEdit\",\n"
            "  \"target_filename\": string (REQUIRED),\n"
            "  \"scope\": one of [\"paragraph\", \"chapter\", \"multi_chapter\"] (REQUIRED),\n"
            "  \"summary\": string (REQUIRED),\n"
            "  \"chapter_index\": integer|null (optional),\n"
            "  \"safety\": one of [\"low\", \"medium\", \"high\"] (REQUIRED),\n"
            "  \"operations\": [\n"
            "    {\n"
            "      \"op_type\": one of [\"replace_range\", \"delete_range\", \"insert_after_heading\"] (REQUIRED),\n"
            "      \"start\": integer (REQUIRED - approximate, anchors take precedence),\n"
            "      \"end\": integer (REQUIRED - approximate, anchors take precedence),\n"
            "      \"text\": string (REQUIRED - new prose for replace/insert),\n"
            "      \"original_text\": string (⚠️ REQUIRED for replace_range/delete_range - EXACT 20-40 words from manuscript!),\n"
            "      \"anchor_text\": string (⚠️ REQUIRED for insert_after_heading - EXACT complete line/paragraph to insert after!),\n"
            "      \"left_context\": string (optional - text before target),\n"
            "      \"right_context\": string (optional - text after target),\n"
            "      \"occurrence_index\": integer (optional - which occurrence, 0-based, default 0)\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "⚠️ ⚠️ ⚠️ CRITICAL FIELD REQUIREMENTS:\n"
            "- replace_range → MUST include 'original_text' with EXACT 20-40 words from manuscript\n"
            "- delete_range → MUST include 'original_text' with EXACT text to delete\n"
            "- insert_after_heading → MUST include 'anchor_text' with EXACT complete line/paragraph to insert after\n"
            "- If you don't provide these fields, the operation will FAIL!\n\n"
            "OUTPUT RULES:\n"
            "- Output MUST be a single JSON object only.\n"
            "- Do NOT include triple backticks or language tags.\n"
            "- Do NOT include explanatory text before or after the JSON.\n\n"
            "=== ⚠️ CRITICAL: EXACT TEXT MATCHING REQUIREMENT ===\n\n"
            "**For ALL operations, anchor text MUST be 100% EXACT:**\n"
            "- Copy COMPLETE paragraphs/sentences from manuscript (no shortening!)\n"
            "- Include ALL dialogue tags, quotation marks, punctuation\n"
            "- Match whitespace, line breaks, and formatting EXACTLY\n"
            "- NEVER paraphrase, summarize, or \"close enough\" - must be VERBATIM\n"
            "- Think: mentally COPY-PASTE the exact text from manuscript\n\n"
            "**Why this matters:**\n"
            "The system uses progressive search to find your anchor text. If your anchor doesn't\n"
            "match EXACTLY, the system can't find it, and your edit will fail (confidence=0.00).\n"
            "Even small differences (missing quote, incomplete sentence) cause failures.\n\n"
            "**Example of FAILURE:**\n"
            "Manuscript: '\"I suspect nothing yet,\" Fleet replied. \"But a billion dollars is an extraordinary amount to spend on nostalgia alone.\"'\n"
            "Your anchor: \"But a billion dollars is an extraordinary amount to spend on nostalgia alone.\"  ❌ INCOMPLETE!\n"
            "Result: ❌ NO MATCH - confidence=0.00 - edit fails\n\n"
            "**Example of SUCCESS:**\n"
            "Manuscript: '\"I suspect nothing yet,\" Fleet replied. \"But a billion dollars is an extraordinary amount to spend on nostalgia alone.\"'\n"
            "Your anchor: \"\\\"I suspect nothing yet,\\\" Fleet replied. \\\"But a billion dollars is an extraordinary amount to spend on nostalgia alone.\\\"\"  ✅ COMPLETE!\n"
            "Result: ✅ EXACT MATCH - confidence=1.00 - edit succeeds\n\n"
            "=== THREE FUNDAMENTAL OPERATIONS ===\n\n"
            "**1. replace_range**: Replace existing text with new text\n"
            "   USE WHEN: User wants to revise, improve, change, modify, or rewrite existing prose\n"
            "   ANCHORING: Provide 'original_text' with EXACT, VERBATIM text from manuscript (20-40 words)\n"
            "   EXAMPLES:\n"
            "   - \"Make this dialogue wittier\" → replace_range on dialogue paragraph\n"
            "   - \"Improve the pacing\" → replace_range on relevant paragraphs\n"
            "   - \"Revise the opening\" → replace_range on opening paragraph(s)\n"
            "   - \"Tighten this description\" → replace_range on description paragraph\n\n"
            "**2. insert_after_heading**: Insert new text AFTER a specific location WITHOUT replacing\n"
            "   USE WHEN: User wants to add, append, or insert new content (not replace existing)\n"
            "   ANCHORING: Provide 'anchor_text' with EXACT, COMPLETE, VERBATIM paragraph/sentence to insert after\n"
            "   ⚠️ CRITICAL: anchor_text must be 100% EXACT match from manuscript (not paraphrased!)\n"
            "   EXAMPLES:\n"
            "   - \"End Chapter 1 with Peterson's thought\" → insert_after last paragraph before '## Chapter 2'\n"
            "     anchor_text: \"\\\"I suspect nothing yet,\\\" Fleet replied. \\\"But a billion dollars is an extraordinary amount to spend on nostalgia alone.\\\"\"\n"
            "   - \"Add a scene transition\" → insert_after current paragraph\n"
            "     anchor_text: [EXACT text of current paragraph]\n"
            "   - \"Insert a new paragraph about X\" → insert_after specified location\n"
            "     anchor_text: [EXACT text of paragraph before insertion point]\n"
            "   - \"Append to the chapter\" → insert_after last paragraph of chapter\n"
            "     anchor_text: [EXACT text of last paragraph]\n\n"
            "   ⚠️ CRITICAL FOR 'END CHAPTER' REQUESTS:\n"
            "   When user says \"end Chapter X with...\" → ALWAYS use insert_after_heading\n"
            "   - Find the LAST PARAGRAPH of target chapter (before next ## Chapter heading)\n"
            "   - Copy that last paragraph EXACTLY as anchor_text (complete, verbatim, with punctuation)\n"
            "   - Your new text will be inserted AFTER that paragraph\n"
            "   - Do NOT use replace_range unless user explicitly asks to REVISE the existing ending\n\n"
            "**3. delete_range**: Remove text\n"
            "   USE WHEN: User wants to delete, remove, or cut content\n"
            "   ANCHORING: Provide 'original_text' with EXACT text to delete\n"
            "   EXAMPLES:\n"
            "   - \"Remove this description\" → delete_range on that paragraph\n"
            "   - \"Cut the middle section\" → delete_range on those paragraphs\n"
            "   - \"Delete this dialogue\" → delete_range on dialogue paragraph\n\n"
            "=== CHAPTER BOUNDARIES ARE SACRED ===\n\n"
            "Chapters are marked by \"## Chapter N\" headings.\n"
            "⚠️ CRITICAL: NEVER include the next chapter's heading in your operation!\n\n"
            "**To add content at END of a chapter:**\n"
            "1. Find the LAST PARAGRAPH of the target chapter (before the next \"## Chapter\" heading)\n"
            "2. OPTION A: Use insert_after_heading with anchor_text = that last paragraph\n"
            "3. OPTION B: Use replace_range with original_text = that last paragraph, text = original + new content\n\n"
            "**Example - Ending Chapter 1:**\n"
            "Last paragraph in manuscript (COMPLETE): '\"I suspect nothing yet,\" Fleet replied. \"But a billion dollars is an extraordinary amount to spend on nostalgia alone.\"'\n"
            "Next line: \"## Chapter 2\"\n\n"
            "✅ CORRECT (insert_after with COMPLETE paragraph as anchor):\n"
            "{\n"
            "  \"op_type\": \"insert_after_heading\",\n"
            "  \"anchor_text\": \"\\\"I suspect nothing yet,\\\" Fleet replied. \\\"But a billion dollars is an extraordinary amount to spend on nostalgia alone.\\\"\",\n"
            "  \"text\": \"\\n\\nI nodded, already making mental notes about what to pack...\"\n"
            "}\n\n"
            "⚠️ CRITICAL: Copy the ENTIRE last paragraph exactly as written! Include:\n"
            "- All dialogue tags (\"Fleet replied\")\n"
            "- All punctuation and quotation marks\n"
            "- Complete sentences from start to end\n"
            "- No paraphrasing or shortening!\n\n"
            "✅ CORRECT (replace with append):\n"
            "{\n"
            "  \"op_type\": \"replace_range\",\n"
            "  \"original_text\": \"\\\"I suspect nothing yet,\\\" Fleet replied. \\\"But a billion dollars is an extraordinary amount to spend on nostalgia alone.\\\"\",\n"
            "  \"text\": \"\\\"I suspect nothing yet,\\\" Fleet replied. \\\"But a billion dollars is an extraordinary amount to spend on nostalgia alone.\\\"\\n\\nI nodded, already making mental notes...\"\n"
            "}\n\n"
            "❌ WRONG (bleeding into Chapter 2):\n"
            "{\n"
            "  \"op_type\": \"replace_range\",\n"
            "  \"original_text\": \"But a billion dollars...\\n\\n## Chapter 2\\n\\nThe rest of the morning...\",\n"
            "  \"text\": \"But a billion...I nodded...\\n\\n## Chapter 2\\n\\nThe rest...\"\n"
            "}\n"
            "Problem: Included next chapter heading! This deletes the heading.\n\n"
            "=== CRITICAL SCOPE ANALYSIS ===\n\n"
            "Before selecting text for operations, ALWAYS analyze scope:\n"
            "- Does this change affect character mood, tone, or emotional state in subsequent sentences?\n"
            "- Would the next paragraph become inconsistent or awkward after this change?\n"
            "- Is this part of a larger dialogue exchange, action sequence, or emotional moment?\n"
            "- Does the revision logic continue beyond the obvious target text?\n"
            "- If YES to any: Expand your operation to include affected subsequent text\n\n"
            "Example: If adding witty thought to chapter end, consider whether it changes the tone\n"
            "and whether following paragraphs need adjustment for consistency.\n\n"
            "=== CRITICAL TEXT PRECISION REQUIREMENTS ===\n\n"
            "For 'original_text' and 'anchor_text' fields:\n"
            "- Must be EXACT, COMPLETE, and VERBATIM from the current manuscript\n"
            "- Include ALL whitespace, line breaks, and formatting exactly as written\n"
            "- Include complete sentences or natural text boundaries (periods, paragraph breaks)\n"
            "- NEVER paraphrase, summarize, or reformat the text\n"
            "- COPY AND PASTE mentally - imagine copying exact text from manuscript\n"
            "- Minimum 10-20 words for unique identification\n"
            "- If text appears multiple times, include MORE context to disambiguate\n"
            "- ⚠️ NEVER include chapter headers (##) in original_text for replace_range!\n\n"
            "=== CRITICAL CONTENT BOUNDARIES ===\n\n"
            "**Context Structure:**\n"
            "You will receive:\n"
            "- PREVIOUS CHAPTER (for context) - Full text for tone/continuity understanding\n"
            "- CURRENT CHAPTER (work area) - The chapter you are editing\n"
            "- PARAGRAPH AROUND CURSOR - Specific focus area within current chapter\n"
            "- NEXT CHAPTER (for context) - Full text for transition/consistency checking\n\n"
            "**Your operations must ONLY target CURRENT CHAPTER:**\n"
            "- 'original_text' and 'anchor_text' must reference CURRENT CHAPTER text only\n"
            "- Previous/Next chapters are for context (tone, transitions, continuity)\n"
            "- NEVER create operations targeting text from Previous or Next chapters\n"
            "- Reference materials (Outline, Rules, Style, Characters) are GUIDANCE only\n\n"
            "**Use adjacent chapters to:**\n"
            "- Ensure tone consistency when adding/revising content\n"
            "- Check if ending current chapter affects beginning of next chapter\n"
            "- Maintain character emotional state continuity\n"
            "- Understand pacing and rhythm across chapter transitions\n\n"
            "**Example:** If adding witty ending to Chapter 1, check if Chapter 2 opens with a tone\n"
            "that would be inconsistent with that ending. If so, note this in your 'summary' field.\n\n"
            "=== ANCHORING PRECISION (PROGRESSIVE SEARCH) ===\n\n"
            "The system uses 4-level progressive search to find your anchors:\n"
            "1. Exact match (confidence 1.0) - best!\n"
            "2. Normalized whitespace (confidence 0.9) - handles spacing variations\n"
            "3. Sentence boundary (confidence 0.8) - matches sentence starts/ends\n"
            "4. Key phrase anchoring (confidence 0.7) - matches first/last few words\n\n"
            "Provide detailed, precise anchors for confidence ≥ 0.9:\n"
            "- original_text: 20-40 words, complete sentences, EXACT verbatim\n"
            "- anchor_text: Complete paragraph or heading line\n"
            "- left_context + right_context: Boundary text (30-50 chars each)\n"
            "- occurrence_index: Which occurrence if text repeats (0=first, 1=second)\n\n"
            "=== CONTENT GENERATION RULES ===\n\n"
            "1. operations[].text MUST contain final prose (no placeholders or notes)\n"
            "2. For chapter generation: aim 800-1200 words, begin with '## Chapter N'\n"
            "3. If outline present: Transform outline beats into full narrative prose - craft vivid scenes, not outline paraphrasing\n"
            "   - The outline is a blueprint for story structure, not a script to convert line-by-line\n"
            "   - Write complete narrative with dialogue, action, description, character voice, and emotional depth\n"
            "   - Add all enriching details needed to bring the beats to life as compelling prose\n"
            "   - Do not simply paraphrase outline beats - always craft full narrative prose\n"
            "4. NO YAML frontmatter in operations[].text\n"
            "5. Match established voice and style\n"
            "6. Complete sentences with proper grammar\n\n"
            "=== NARRATIVE CRAFT PRINCIPLES ===\n\n"
            "**Show, Don't Tell:**\n"
            "- Reveal character emotions through actions, dialogue, and physical reactions, not statements\n"
            "- Build atmosphere through sensory details (sight, sound, smell, texture, temperature)\n"
            "- Let readers infer meaning from scene details rather than explaining directly\n"
            "- Example: Instead of 'Peterson was worried,' show: 'Peterson's fingers drummed the desk, each tap louder than the last.'\n\n"
            "**Scene-Building vs Summary:**\n"
            "- Write complete scenes with setting, action, dialogue, and character internality\n"
            "- Avoid summary prose that reports events ('He went to the office and found the file')\n"
            "- Build scenes moment-by-moment with specific details and natural pacing\n"
            "- Let story events emerge organically within scenes, not as mechanical beat-conversion\n"
            "- Transitions between beats should flow naturally, not feel like checklist items\n\n"
            "**Character Voice and Dialogue:**\n"
            "- Dialogue must sound natural and character-specific, not expository or mechanical\n"
            "- Each character's speech patterns should reflect their personality, background, and emotional state\n"
            "- Avoid dialogue that merely conveys plot information - let characters speak as real people would\n"
            "- Internal thoughts should match the character's voice and perspective (POV)\n\n"
            "**Sensory Details and Atmosphere:**\n"
            "- Ground every scene in specific sensory details (what characters see, hear, feel, smell, taste)\n"
            "- Use atmospheric details to establish mood and tone, not just setting\n"
            "- Balance sensory detail level according to Style Guide requirements\n"
            "- Create immersive scenes that readers can experience, not just observe\n\n"
            "**Organic Pacing vs Mechanical Beat-Following:**\n"
            "- Write scenes that flow naturally with appropriate pacing for the moment\n"
            "- Don't rush through beats to 'cover' all outline points - let scenes breathe\n"
            "- Build tension, emotion, and character development organically within scenes\n"
            "- Outline beats are story goals to achieve, not items to check off sequentially\n"
            "- A single scene can achieve multiple outline beats naturally if the story flows that way\n"
            "- Conversely, a single outline beat might require multiple scenes if the story demands it\n\n"
            "**Style Guide Integration:**\n"
            "- Every sentence must sound like it was written in the Style Guide's voice\n"
            "- Apply Style Guide techniques (POV, tense, pacing, dialogue style) consistently throughout\n"
            "- If Style Guide includes writing samples, emulate their technique and voice, never copy content\n"
            "- The Style Guide voice should be so internalized that it feels natural, not forced\n\n"
            "=== RESPONSE STRATEGY ===\n\n"
            "**For operations (revisions, insertions, deletions):**\n"
            "- Provide the JSON operation with minimal explanation\n"
            "- Let the prose speak for itself\n"
            "- Only add commentary if user specifically asks for reasoning\n"
            "- Focus on the changes, not lengthy explanations\n\n"
            "**For questions (\"How should...\", \"What's wrong with...\"):**\n"
            "- Provide focused, direct answers\n"
            "- Reference style guide or rules if helpful\n"
            "- Keep response concise and actionable\n"
            "- Don't provide full story analysis unless explicitly requested\n\n"
            "**Summary in 'summary' field:**\n"
            "- One sentence describing the change (e.g., \"Added witty ending to Chapter 1\")\n"
            "- Not a detailed explanation\n"
            "- User sees the actual prose in operations[].text\n\n"
            "=== CLARIFYING QUESTIONS (clarifying_questions field) ===\n\n"
            "**ASK questions when:**\n"
            "- Request is genuinely ambiguous (\"make it better\" - better how?)\n"
            "- Multiple valid creative directions exist (tone choice, POV, pacing approach)\n"
            "- Author preference would significantly impact quality\n"
            "- Continuity issue detected that needs author decision (\"Chapter 3 contradicts this - should I adjust both?\")\n"
            "- Request could damage story logic without clarification\n"
            "- Character motivation unclear from context\n\n"
            "**DO NOT ask questions when:**\n"
            "- Request is clear enough to execute well\n"
            "- Style guide/rules already provide guidance\n"
            "- Question would be trivial or obvious\n"
            "- You can make a reasonable inference from context\n"
            "- Minor stylistic choices that don't affect quality\n\n"
            "**Question quality standards:**\n"
            "- Specific and actionable (not \"What do you want?\")\n"
            "- Offer options when helpful (\"Should this be subtle foreshadowing or direct reveal?\")\n"
            "- Reference specific text/context (\"The outline shows X, but Chapter 2 established Y - which takes precedence?\")\n"
            "- Maximum 2-3 questions (don't overwhelm)\n"
            "- If asking questions, you can still provide a tentative operation or leave operations empty\n\n"
            "**Examples of GOOD questions:**\n"
            "- \"Should the confrontation escalate to physical violence, or stay verbal?\" (creative direction)\n"
            "- \"Chapter 3 shows Peterson trusting Fleet, but this revision makes him suspicious. Should I adjust Chapter 3 for consistency?\" (continuity)\n"
            "- \"The outline says 'Fleet discovers the truth' but doesn't specify what truth. What should he discover?\" (ambiguous outline)\n"
            "- \"Add tension - should this be internal (Fleet's anxiety) or external (a threat appears)?\" (multiple approaches)\n\n"
            "**Examples of BAD questions (too obvious, just do it):**\n"
            "- \"Should I use good grammar?\" (obvious)\n"
            "- \"Do you want me to make it sound better?\" (vague, unhelpful)\n"
            "- \"What tense should I use?\" (obvious from existing manuscript)\n"
            "- \"Should I add a comma here?\" (trivial)\n\n"
            "**If you have questions, still provide your best attempt:**\n"
            "```json\n"
            "{\n"
            "  \"clarifying_questions\": [\n"
            "    \"Should this revelation be subtle or dramatic?\",\n"
            "    \"Chapter 3 contradicts this change - adjust both chapters or just current?\"\n"
            "  ],\n"
            "  \"operations\": [{...}],  // Your best guess if possible\n"
            "  \"summary\": \"Tentative revision pending clarification on tone and scope\"\n"
            "}\n"
            "```\n\n"
            "=== SCOPE METADATA ===\n\n"
            "Set appropriate scope:\n"
            "- \"paragraph\": Single paragraph or sentence edits\n"
            "- \"chapter\": Full chapter generation or major chapter rewrites\n"
            "- \"multi_chapter\": Operations spanning multiple chapters\n\n"
            "=== YOU CHOOSE THE OPERATION TYPE ===\n\n"
            "Based on semantic understanding of the user's request:\n"
            "- \"End with X\" → insert_after (or replace last paragraph with original+new)\n"
            "- \"Revise X\" → replace_range\n"
            "- \"Add X\" / \"Include X\" → insert_after (or replace if modifying existing)\n"
            "- \"Remove X\" → delete_range\n"
            "- \"Write Chapter N\" → replace_range (if exists) or insert_after previous chapter\n"
            "- \"Improve X\" → replace_range\n"
            "- \"Make X more Y\" → replace_range\n\n"
            "Use your understanding of the request to choose the right operation!\n"
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            shared_memory = state.get("shared_memory", {}) or {}
            active_editor = shared_memory.get("active_editor", {}) or {}

            manuscript = active_editor.get("content", "")
            filename = active_editor.get("filename") or "manuscript.md"
            frontmatter = active_editor.get("frontmatter", {}) or {}
            cursor_offset = int(active_editor.get("cursor_offset", -1))
            selection_start = int(active_editor.get("selection_start", -1))
            selection_end = int(active_editor.get("selection_end", -1))

            # Hard gate: require fiction (with defensive fallback to parse content frontmatter)
            fm_type = str(frontmatter.get("type", "")).lower()
            if fm_type != "fiction":
                return self._create_success_result(
                    response="Active editor is not a fiction manuscript; editing agent skipping.",
                    tools_used=[],
                    processing_time=0.0,
                    additional_data={"skipped": True},
                )

            # Allow empty manuscript: we can generate and insert Chapter 1 from outline

            # Resolve referenced outline/rules/style/characters if present (frontmatter conventions)
            # Load referenced context (outline + cascade)
            # Pass canonical path via frontmatter shim for loader base dir override
            if active_editor.get("canonical_path"):
                frontmatter = { **frontmatter, "__canonical_path__": active_editor.get("canonical_path") }
            loader = FileContextLoader()
            loaded = loader.load_referenced_context(filename, frontmatter)

            # Chapter scope detection
            chapter_ranges = find_chapter_ranges(manuscript)
            active_idx = locate_chapter_index(chapter_ranges, cursor_offset if cursor_offset >= 0 else 0)
            prev_c, next_c = (None, None)
            current_chapter_text = manuscript
            current_chapter_number: Optional[int] = None
            if active_idx != -1:
                current = chapter_ranges[active_idx]
                prev_c, next_c = get_adjacent_chapters(chapter_ranges, active_idx)
                current_chapter_text = manuscript[current.start:current.end]
                current_chapter_number = current.chapter_number
            
            para_start, para_end = paragraph_bounds(manuscript, cursor_offset if cursor_offset >= 0 else 0)
            paragraph_text = manuscript[para_start:para_end]

            # Determine frontmatter boundaries and compute context-safe views (without YAML blocks)
            fm_end_idx = _frontmatter_end_index(manuscript)
            context_current_chapter_text = _strip_frontmatter_block(current_chapter_text)
            context_paragraph_text = _strip_frontmatter_block(paragraph_text)

            # Outline chapter extraction: match manuscript current chapter number to outline chapter number
            outline_current_chapter_text = None
            if loaded.outline:
                try:
                    outline_ranges = find_chapter_ranges(loaded.outline.content)
                    outline_idx = -1
                    if current_chapter_number is not None:
                        for i, rng in enumerate(outline_ranges):
                            if rng.chapter_number == current_chapter_number:
                                outline_idx = i
                                break
                    if outline_idx == -1 and active_idx != -1 and 0 <= active_idx < len(outline_ranges):
                        outline_idx = active_idx
                    # If manuscript has no chapters yet, default to the first outline chapter
                    if outline_idx == -1 and outline_ranges:
                        outline_idx = 0
                    if outline_idx != -1:
                        rng = outline_ranges[outline_idx]
                        outline_current_chapter_text = loaded.outline.content[rng.start:rng.end]
                except Exception:
                    outline_current_chapter_text = None

            # Prepare referenced bodies without frontmatter to avoid YAML leakage
            outline_body = _strip_frontmatter_block(loaded.outline.content) if loaded.outline else None
            rules_body = _strip_frontmatter_block(loaded.rules.content) if loaded.rules else None
            style_body = _strip_frontmatter_block(loaded.style.content) if loaded.style else None
            characters_bodies = [_strip_frontmatter_block(c.content) for c in loaded.characters] if loaded.characters else []

            # Build messages
            system_prompt = self._build_system_prompt()
            # Detect current user intent/request for clarity
            try:
                current_request = (self._extract_current_user_query(state) or "").strip()
            except Exception:
                current_request = ""

            # Build context with full adjacent chapters for consistency analysis
            prev_chapter_text = None
            prev_chapter_label = None
            next_chapter_text = None
            next_chapter_label = None
            
            if prev_c:
                prev_chapter_text = _strip_frontmatter_block(manuscript[prev_c.start:prev_c.end])
                prev_chapter_num = prev_c.chapter_number or "Previous"
                prev_chapter_label = f"Chapter {prev_chapter_num}" if prev_c.chapter_number else "Previous Chapter"
            
            if next_c:
                next_chapter_text = _strip_frontmatter_block(manuscript[next_c.start:next_c.end])
                next_chapter_num = next_c.chapter_number or "Next"
                next_chapter_label = f"Chapter {next_chapter_num}" if next_c.chapter_number else "Next Chapter"
            
            # Determine current chapter label for context
            current_chapter_label = f"Chapter {current_chapter_number}" if current_chapter_number else "Current Chapter"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Current Date/Time: {datetime.now().isoformat()}"},
                {
                    "role": "user",
                    "content": (
                        "=== MANUSCRIPT CONTEXT ===\n"
                        f"Primary file: {filename}\n"
                        f"Working area: {current_chapter_label}\n"
                        f"Cursor position: paragraph shown below\n\n"
                        + (f"=== {prev_chapter_label.upper()} (FOR CONTEXT - DO NOT EDIT) ===\n{prev_chapter_text}\n\n" if prev_chapter_text else "")
                        + f"=== {current_chapter_label.upper()} (CURRENT WORK AREA) ===\n{context_current_chapter_text}\n\n"
                        + f"=== PARAGRAPH AROUND CURSOR ===\n{context_paragraph_text}\n\n"
                        + (f"=== {next_chapter_label.upper()} (FOR CONTEXT - DO NOT EDIT) ===\n{next_chapter_text}\n\n" if next_chapter_text else "")
                        + (f"=== STYLE GUIDE (voice and tone - READ FIRST) ===\n{style_body}\n\n" if style_body else "")
                        + (f"=== RULES (universe constraints) ===\n{rules_body}\n\n" if rules_body else "")
                        + ("".join([f"=== CHARACTER DOC ===\n{body}\n\n" for body in characters_bodies]) if characters_bodies else "")
                        + (f"=== CURRENT CHAPTER OUTLINE (beats to follow) ===\n{outline_current_chapter_text}\n\n" if outline_current_chapter_text else "")
                        + (f"=== FULL OUTLINE (story structure) ===\n{outline_body}\n\n" if outline_body else "")
                        + ("=== NO OUTLINE PRESENT ===\n"
                            "The user's request serves as the story directive (WHAT happens).\n"
                            "**Your task:** Write narrative prose in the Style Guide's voice that fulfills the user's request.\n"
                            "- Use the user's request as the story goal (e.g., 'Generate a paragraph where X happens')\n"
                            "- Apply Style Guide voice, techniques, and narrative craft to fulfill that request\n"
                            "- Write natural, compelling prose that matches the Style Guide, not generic narrative\n"
                            "- Continue from manuscript context if present, maintaining established voice and style\n\n" if not outline_body else "")
                        + ("=== OUTLINE AS NARRATIVE BLUEPRINT ===\n"
                            "The chapter outline provides story beats and structure, NOT a script to paraphrase.\n\n"
                            "**CRITICAL DISTINCTION:**\n"
                            "- **Outline = WHAT happens** (story structure, plot points, character arcs)\n"
                            "- **Style Guide = HOW to write** (voice, techniques, narrative craft)\n"
                            "- **Your prose = Natural storytelling in Style Guide voice that achieves Outline goals**\n\n"
                            "**BEFORE writing prose - Style Guide Application Process:**\n"
                            "1. Read the Style Guide completely - internalize voice, POV, tense, pacing\n"
                            "2. Note specific techniques: dialogue style, sensory detail level, show-don't-tell ratio\n"
                            "3. Identify any writing samples - these demonstrate the TARGET voice (emulate technique, never copy content)\n"
                            "4. Extract character voice patterns if specified in Style Guide\n"
                            "5. Understand the narrative voice so deeply it feels natural, not forced\n\n"
                            "**WHILE writing prose - Natural Storytelling Process:**\n"
                            "1. Write naturally in the established Style Guide voice - don't think about outline beats yet\n"
                            "2. Build complete scenes with setting, action, dialogue, internality, sensory details\n"
                            "3. Let story events emerge organically within scenes - don't convert beats sequentially\n"
                            "4. Check: Does each paragraph sound like it's from this Style Guide? Does it match the voice?\n"
                            "5. Only after scene flows naturally: verify outline beats are achieved (they should be, if you wrote the scene well)\n\n"
                            "**BAD EXAMPLE - Outline-Following (Mechanical Beat Conversion):**\n"
                            "Outline Beat: 'Peterson discovers the anomaly in the financial records'\n\n"
                            "❌ BAD (Checklist prose):\n"
                            "> Peterson discovered the anomaly in the financial records. He looked at the numbers. They didn't add up. He decided to investigate further.\n\n"
                            "Problems: Summary prose, no sensory details, no character voice, mechanical beat-conversion, generic narrative\n\n"
                            "**GOOD EXAMPLE - Style-Guided Narrative (Natural Storytelling):**\n"
                            "Outline Beat: 'Peterson discovers the anomaly in the financial records'\n\n"
                            "✅ GOOD (Style-guided narrative with comprehensive style):\n"
                            "> The spreadsheet blurred as Peterson's eyes traced the same column for the third time. Something gnawed at him—not the numbers themselves, but the spaces between them. Entry 2847: $50,000. Entry 2848: skipped. Entry 2849: $50,000. His finger hovered over the trackpad. 'Why would they skip an entry?' he muttered, already knowing the answer would cost him sleep tonight.\n\n"
                            "Why it works: Sensory details (blurred, traced, gnawed), character internality (knowing the answer would cost sleep), natural dialogue, specific details (entry numbers), atmospheric tension, Style Guide voice throughout\n\n"
                            "**Technique Checklist for Style Application:**\n"
                            "- [ ] Every sentence matches Style Guide voice (POV, tense, tone)\n"
                            "- [ ] Sensory details present (sight, sound, smell, texture, temperature) at appropriate level\n"
                            "- [ ] Show-don't-tell: emotions revealed through actions, not statements\n"
                            "- [ ] Complete scenes with setting, action, dialogue, internality - not summary prose\n"
                            "- [ ] Character voice authentic and natural (dialogue, thoughts, actions)\n"
                            "- [ ] Natural pacing - scenes breathe, don't rush through beats\n"
                            "- [ ] Outline beats achieved organically within natural story flow\n"
                            "- [ ] Transitions between beats feel natural, not mechanical\n\n"
                            "**What the outline provides:** Story structure, key events, plot progression\n"
                            "**What the Style Guide provides:** Narrative voice, writing style, technical requirements\n"
                            "**What you must add:** Full narrative prose, character voice, scene details, emotional depth\n"
                            "**Constraint:** Do not add major plot events not in the outline (new characters, deaths, revelations)\n"
                            "**Freedom:** Add all enriching details needed to make the prose compelling and complete\n"
                            "**Goal:** Write 800-1200 words of engaging narrative prose that sounds like it came from the Style Guide, not outline paraphrasing\n\n" if outline_current_chapter_text else "")
                        + f"⚠️ CRITICAL: Your operations must target {current_chapter_label.upper()} ONLY. "
                        + f"Adjacent chapters are for context (tone, transitions, continuity) - DO NOT edit them!\n\n"
                        + "Provide a ManuscriptEdit JSON plan for the current work area."
                    ),
                },
            ]
            
            # **ROOSEVELT: No mode-specific guidance - LLM chooses operation type based on system prompt teaching**
            # The system prompt teaches the three operations; LLM uses semantic understanding to choose

            # Build selection/cursor context message
            selection_context = ""
            if selection_start >= 0 and selection_end > selection_start:
                selected_text = manuscript[selection_start:selection_end]
                selection_context = (
                    f"\n\n=== USER HAS SELECTED TEXT ===\n"
                    f"Selected text (characters {selection_start}-{selection_end}):\n"
                    f'"""{selected_text[:500]}{"..." if len(selected_text) > 500 else ""}"""\n\n'
                    "⚠️ User selected this specific text! Use it as your anchor:\n"
                    "- For edits within selection: Use 'original_text' matching the selected text (or portion of it)\n"
                    "- System will automatically constrain your edit to the selection\n"
                    "- This is the MOST PRECISE anchoring method for prose!\n"
                )
            elif cursor_offset >= 0:
                selection_context = (
                    f"\n\n=== CURSOR POSITION ===\n"
                    f"Cursor is in the paragraph shown above (character offset {cursor_offset}).\n"
                    "If editing this paragraph, provide EXACT text from it as 'original_text'.\n"
                )
            
            if current_request:
                messages.append({
                    "role": "user",
                    "content": (
                        f"USER REQUEST: {current_request}\n\n"
                        + selection_context +
                        "\n=== ANCHORING REQUIREMENTS FOR PROSE ===\n"
                        "For REPLACE/DELETE operations in prose (no headers), you MUST provide robust anchors:\n\n"
                        "**OPTION 1 (BEST): Use selection as anchor**\n"
                        "- If user selected text, match it EXACTLY in 'original_text'\n"
                        "- Include at least 20-30 words for reliable matching\n\n"
                        "**OPTION 2: Use left_context + right_context**\n"
                        "- left_context: 30-50 chars BEFORE the target (exact text)\n"
                        "- right_context: 30-50 chars AFTER the target (exact text)\n"
                        "- This bounds the edit precisely even in long documents\n\n"
                        "**OPTION 3: Use long original_text**\n"
                        "- Include 20-40 words of EXACT, VERBATIM text to replace\n"
                        "- Include complete sentences with natural boundaries\n"
                        "- If phrase appears multiple times, use 'occurrence_index' (0-based)\n\n"
                        "**For AMBIGUOUS repeated phrases:**\n"
                        "- Set 'occurrence_index' to target the Nth occurrence (0=first, 1=second, etc.)\n"
                        "- Or provide unique left_context + right_context to disambiguate\n\n"
                        "⚠️ NEVER include chapter headers (##) in original_text - they will be deleted!\n"
                        "⚠️ For adding content below headers, use op_type='insert_after_heading' with anchor_text='## Chapter N'.\n\n"
                        "=== SPACING RULES (CRITICAL - READ CAREFULLY!) ===\n"
                        "YOUR TEXT MUST END IMMEDIATELY AFTER THE LAST CHARACTER!\n\n"
                        "✅ CORRECT prose spacing:\n"
                        '  "text": "Para one.\\n\\nPara two."  ← Ends after period, NO \\n after "two."\n'
                        "  (Double \\n\\n between paragraphs is OK, but NO trailing \\n after last para!)\n\n"
                        "✅ CORRECT list spacing:\n"
                        '  "text": "- Item 1\\n- Item 2"  ← Ends after "2", NO \\n!\n\n'
                        "❌ WRONG - Trailing \\n after last line:\n"
                        '  "text": "The text ends here.\\n"  ← Extra \\n creates blank line!\n\n'
                        "❌ WRONG - Trailing \\n\\n:\n"
                        '  "text": "Para 1.\\n\\nPara 2.\\n\\n"  ← \\n\\n creates 2 blank lines!\n\n'
                        "❌ WRONG - Excessive \\n between paragraphs:\n"
                        '  "text": "Para 1.\\n\\n\\n\\nPara 2."  ← Too many \\n!\n\n'
                        "IRON-CLAD RULE: After last line = ZERO \\n (nothing!)\n"
                        "The system adds all necessary spacing around your content automatically.\n"
                    )
                })

            # Call LLM
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            start_time = datetime.now()
            response = await chat_service.openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.4,
            )

            content = response.choices[0].message.content or "{}"
            content = _unwrap_json_response(content)
            # Parse ManuscriptEdit with robust fallback and pre_hash injection
            structured: Optional[ManuscriptEdit] = None
            parse_error = None
            try:
                import json as _json
                raw_obj = _json.loads(content)
                if isinstance(raw_obj, dict):
                    # Normalize to ManuscriptEdit shape
                    if "operations" in raw_obj and isinstance(raw_obj["operations"], list):
                        # Ensure required top-level fields
                        raw_obj.setdefault("target_filename", filename)
                        raw_obj.setdefault("scope", "paragraph")
                        raw_obj.setdefault("summary", "Planned edit generated from context.")
                        # Normalize operations and inject placeholder pre_hash
                        norm_ops = []
                        for op in raw_obj["operations"]:
                            if not isinstance(op, dict):
                                continue
                            op_type = op.get("op_type") or op.get("type") or "replace_range"
                            start_ix = int(op.get("start") if isinstance(op.get("start"), int) else para_start)
                            end_ix = int(op.get("end") if isinstance(op.get("end"), int) else para_end)
                            # Optional anchor hints from model for robust targeting
                            try:
                                orig_text = str(op.get("original") or op.get("original_text") or "")
                                left_ctx = str(op.get("left_context") or "")
                                right_ctx = str(op.get("right_context") or "")
                                resolved = False
                                if orig_text:
                                    idx = manuscript.find(orig_text)
                                    if idx != -1:
                                        start_ix = idx
                                        end_ix = idx + len(orig_text)
                                        resolved = True
                                elif left_ctx and right_ctx:
                                    import re as _re
                                    try:
                                        pat = _re.escape(left_ctx) + r"([\s\S]{0,400}?)" + _re.escape(right_ctx)
                                        m = _re.search(pat, manuscript)
                                        if m:
                                            start_ix = m.start(1)
                                            end_ix = m.end(1)
                                            resolved = True
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            text_val = op.get("text") or ""
                            
                            # **ROOSEVELT: PRESERVE ANCHOR FIELDS - Don't strip them out!**
                            normalized_op = {
                                "op_type": op_type if op_type in ("replace_range", "delete_range", "insert_after_heading") else "replace_range",
                                "start": max(0, min(len(manuscript), start_ix)),
                                "end": max(0, min(len(manuscript), end_ix)),
                                "text": text_val,
                                "pre_hash": "",  # placeholder, will compute below
                                # Preserve anchor fields for validation and resolver
                                "original_text": op.get("original_text") or op.get("original"),
                                "anchor_text": op.get("anchor_text"),
                                "left_context": op.get("left_context"),
                                "right_context": op.get("right_context"),
                                "occurrence_index": op.get("occurrence_index", 0),
                            }
                            norm_ops.append(normalized_op)
                        raw_obj["operations"] = norm_ops
                        structured = ManuscriptEdit(**raw_obj)
                    elif "edits" in raw_obj and isinstance(raw_obj["edits"], list):
                        # Coerce from {edits:[{text:...}]}
                        fallback_text_parts = []
                        for ed in raw_obj["edits"]:
                            t = ed.get("text") if isinstance(ed, dict) else None
                            if isinstance(t, str) and len(t.strip()) > 0:
                                fallback_text_parts.append(t.strip())
                        if fallback_text_parts:
                            insert_text = "\n\n".join(fallback_text_parts)
                            start_ix = para_start
                            end_ix = para_end
                            if len(manuscript.strip()) == 0:
                                start_ix, end_ix = 0, 0
                            fallback = {
                                "target_filename": filename,
                                "scope": "chapter" if start_ix == 0 and end_ix == 0 else "paragraph",
                                "summary": "Planned insertion/rewrite generated from outline context.",
                                "chapter_index": current_chapter_number if current_chapter_number is not None else None,
                                "safety": "medium",
                                "operations": [
                                    {
                                        "op_type": "replace_range",
                                        "start": start_ix,
                                        "end": end_ix,
                                        "text": insert_text,
                                        "pre_hash": ""
                                    }
                                ]
                            }
                            structured = ManuscriptEdit(**fallback)
                if structured is None:
                    # Last attempt: direct parse for already-correct schema
                    structured = ManuscriptEdit.parse_raw(content)
            except ValidationError as ve:
                # Validation error - likely missing required anchors
                logger.error(f"❌ Validation error in ManuscriptEdit: {ve}")
                error_msg = str(ve)
                if "ANCHOR REQUIRED" in error_msg:
                    return self._create_success_result(
                        response=(
                            "⚠️ The editing operation couldn't be created because required anchoring information is missing.\n\n"
                            "**What went wrong:** The system needs EXACT text from your manuscript to know where to make changes, but this information wasn't provided.\n\n"
                            "**How to fix:** Please try rephrasing your request, or:\n"
                            "- For additions: Specify what text should come BEFORE the new content\n"
                            "- For changes: Specify what EXACT text to replace\n\n"
                            f"**Technical details:** {error_msg}"
                        ),
                        tools_used=[],
                        processing_time=(datetime.now() - start_time).total_seconds(),
                        additional_data={"validation_error": error_msg, "raw": content},
                    )
                else:
                    return self._create_success_result(
                        response=f"Failed to validate the edit plan: {error_msg}",
                        tools_used=[],
                        processing_time=(datetime.now() - start_time).total_seconds(),
                        additional_data={"validation_error": error_msg, "raw": content},
                    )
            except Exception as e:
                parse_error = e
            if structured is None:
                logger.error(f"❌ Failed to parse ManuscriptEdit: {parse_error}")
                return self._create_success_result(
                    response="Failed to produce a valid ManuscriptEdit. Please refine your request.",
                    tools_used=[],
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    additional_data={"raw": content},
                )

            # **ROOSEVELT: No hardcoded intent detection - LLM chooses operation type semantically**
            # The LLM understands the user's request and chooses the appropriate op_type
            # based on the teaching in the system prompt (replace_range, insert_after_heading, delete_range)
            
            logger.info(f"✍️ FICTION AGENT: Processing request: '{current_request[:100] if current_request else 'N/A'}'")
            logger.info(f"✍️ FICTION AGENT: LLM will choose operation type based on semantic understanding")
            
            # **DEBUG: Log what the LLM actually provided**
            logger.info(f"🔍 FICTION DEBUG: Structured response summary: {structured.summary}")
            logger.info(f"🔍 FICTION DEBUG: Number of operations: {len(structured.operations)}")
            for idx, op in enumerate(structured.operations):
                logger.info(f"🔍 FICTION DEBUG: Operation {idx}: op_type={op.op_type}")
                if hasattr(op, 'original_text') and op.original_text:
                    logger.info(f"🔍 FICTION DEBUG: Operation {idx}: original_text (first 100 chars)='{op.original_text[:100] if op.original_text else 'NONE'}'")
                if hasattr(op, 'anchor_text') and op.anchor_text:
                    logger.info(f"🔍 FICTION DEBUG: Operation {idx}: anchor_text (first 100 chars)='{op.anchor_text[:100] if op.anchor_text else 'NONE'}'")
                if hasattr(op, 'text') and op.text:
                    logger.info(f"🔍 FICTION DEBUG: Operation {idx}: new text (first 100 chars)='{op.text[:100] if op.text else 'NONE'}'")

            # **ROOSEVELT'S PROGRESSIVE SEARCH RESOLVER INTEGRATION**
            # Use shared utility resolver for precise, confidence-scored operations
            ops: List[EditorOperation] = []
            body_only_manuscript = _strip_frontmatter_block(manuscript)
            
            # Determine desired chapter for GENERATION mode smart positioning
            ch_ranges = find_chapter_ranges(manuscript)
            desired_ch_num: Optional[int] = None
            try:
                if getattr(structured, 'chapter_index', None) is not None:
                    ci = int(structured.chapter_index)
                    if ci >= 0:
                        desired_ch_num = ci + 1
            except Exception:
                desired_ch_num = None
            if desired_ch_num is None and current_request:
                m = re.search(r"chapter\s+(\d+)", current_request, flags=re.IGNORECASE)
                if m:
                    try:
                        desired_ch_num = int(m.group(1))
                    except Exception:
                        pass
            if desired_ch_num is None and structured.operations:
                try:
                    first_text = getattr(structured.operations[0], 'text', '') or ''
                    m2 = re.search(r"^##\s+Chapter\s+(\d+)\b", first_text, flags=re.IGNORECASE | re.MULTILINE)
                    if m2:
                        desired_ch_num = int(m2.group(1))
                except Exception:
                    pass
            if desired_ch_num is None:
                try:
                    max_num = max([r.chapter_number for r in ch_ranges if r.chapter_number is not None], default=0)
                    desired_ch_num = (max_num or 0) + 1
                except Exception:
                    desired_ch_num = 1

            # Build selection dict for resolver
            selection = {"start": selection_start, "end": selection_end} if selection_start >= 0 and selection_end >= 0 else None
            
            logger.info(f"✍️ FICTION AGENT: Processing {len(structured.operations)} operations")
            
            for op in structured.operations:
                # **ROOSEVELT'S TRUST THE LLM APPROACH**
                # LLM chose the operation type (replace_range, insert_after_heading, delete_range)
                # based on semantic understanding of the user's request
                # Resolver handles precision via progressive search
                
                # Build operation dict for resolver
                op_dict = {
                    "original_text": getattr(op, "original_text", None),
                    "anchor_text": getattr(op, "anchor_text", None),
                    "left_context": getattr(op, "left_context", None),
                    "right_context": getattr(op, "right_context", None),
                    "occurrence_index": getattr(op, "occurrence_index", 0),
                    "text": op.text,
                    "op_type": op.op_type,
                }
                
                # Use progressive search resolver with flexible anchoring
                # LLM should provide good anchors based on system prompt teaching
                try:
                    resolved_start, resolved_end, resolved_text, resolved_confidence = resolve_operation(
                        manuscript,
                        op_dict,
                        selection=selection,
                        heading_hint=None,
                        frontmatter_end=fm_end_idx,
                        require_anchors=False,  # Flexible - let confidence guide us
                    )
                    
                    logger.info(f"📍 FICTION: {op.op_type} resolved [{resolved_start}:{resolved_end}] confidence={resolved_confidence:.2f}")
                    
                    # Warn if low confidence
                    if resolved_confidence < 0.7:
                        logger.warning(f"⚠️ Low confidence anchor ({resolved_confidence:.2f}) - operation may be imprecise")
                    
                    # Update operation with resolver results
                    op.start = resolved_start
                    op.end = resolved_end
                    op.text = resolved_text
                    op.confidence = resolved_confidence
                    
                except ValueError as e:
                    # Anchor resolution failed - use smart fallback positioning
                    logger.warning(f"⚠️ FICTION: Anchor resolution failed: {e}")
                    logger.info(f"⚠️ Falling back to smart positioning based on scope")
                    
                    # Fallback positioning based on scope metadata
                    scope = str(getattr(structured, 'scope', '')).lower()
                    
                    if scope == 'chapter' and desired_ch_num and ch_ranges:
                        # Try to find target chapter
                        found = False
                        for r in ch_ranges:
                            if r.chapter_number == desired_ch_num:
                                op.start = r.start
                                op.end = r.end
                                found = True
                                break
                        
                        if not found:
                            # Insert after last chapter
                            if ch_ranges:
                                op.start = ch_ranges[-1].end
                                op.end = ch_ranges[-1].end
                            else:
                                op.start = fm_end_idx
                                op.end = fm_end_idx
                    else:
                        # Fallback to paragraph scope
                        op.start = para_start
                        op.end = para_end
                    
                    op.confidence = 0.3
                    logger.info(f"📍 Fallback positioning [{op.start}:{op.end}]")
                
                # Ensure chapter heading for chapter-scope operations that are pure insertions
                is_chapter_scope = (str(getattr(structured, 'scope', '')).lower() == 'chapter')
                is_new_chapter = (op.start == op.end)  # Pure insertion
                
                if is_chapter_scope and is_new_chapter and not op.text.strip().startswith('#'):
                    # Determine chapter number
                    chapter_num = desired_ch_num or current_chapter_number or 1
                    op.text = _ensure_chapter_heading(op.text or '', int(chapter_num))
                    logger.info(f"✍️ Added chapter {chapter_num} heading for chapter-scope insertion")
                
                # Calculate pre_hash for optimistic concurrency
                pre_slice = manuscript[op.start:op.end]
                op.pre_hash = _slice_hash(pre_slice)
                
                ops.append(op)
            structured.operations = ops

            # SAFETY NET: If no usable operations were produced, synthesize a paragraph replacement
            try:
                no_ops = len(structured.operations) == 0
                blank_text_ops = all(
                    (getattr(op, "text", "") or "").strip() == "" and op.op_type != "delete_range"
                    for op in structured.operations
                ) if structured.operations else True
                if no_ops or blank_text_ops:
                    start_ix = para_start
                    end_ix = para_end
                    # If manuscript body (excluding frontmatter) is empty, insert after frontmatter (if any)
                    body_only = _strip_frontmatter_block(manuscript)
                    if len(body_only.strip()) == 0:
                        start_ix = fm_end_idx
                        end_ix = fm_end_idx
                    fallback_text = (structured.summary or "").strip()
                    if fallback_text:
                        pre_slice = manuscript[start_ix:end_ix]
                        synthesized = {
                            "op_type": "replace_range",
                            "start": start_ix,
                            "end": end_ix,
                            "text": _ensure_chapter_heading(fallback_text, int(current_chapter_number or 1)),
                            "pre_hash": _slice_hash(pre_slice)
                        }
                        # Rebuild as Pydantic model to preserve types
                        from models.agent_response_models import EditorOperation
                        structured.operations = [EditorOperation(**synthesized)]
            except Exception:
                # If synthesis fails, proceed with whatever we have
                pass

            # Return HITL-ready payload (do not apply edits here)
            processing_time = (datetime.now() - start_time).total_seconds()
            # Build a prose preview from operations' text so the user sees actual content, not just a summary
            try:
                generated_preview = "\n\n".join([
                    (getattr(op, "text", "") or "").strip()
                    for op in structured.operations
                    if (getattr(op, "text", "") or "").strip()
                ]).strip()
            except Exception:
                generated_preview = ""

            response_text = generated_preview if generated_preview else (structured.summary or "Edit plan ready.")
            
            # Add clarifying questions to response if present
            clarifying_questions = getattr(structured, "clarifying_questions", None)
            if clarifying_questions and len(clarifying_questions) > 0:
                questions_section = "\n\n**Questions for clarification:**\n" + "\n".join([
                    f"- {q}" for q in clarifying_questions
                ])
                response_text = response_text + questions_section
                logger.info(f"❓ FICTION AGENT: Asking {len(clarifying_questions)} clarifying question(s)")

            # ROOSEVELT'S EDITOR INTEGRATION: Store operations in state for API streaming
            editor_ops = [op.model_dump() for op in structured.operations]
            manuscript_edit_data = structured.model_dump()
            additional = {"content_preview": response_text[:2000]}
            if editor_ops:
                additional["editor_operations"] = editor_ops
                additional["manuscript_edit"] = manuscript_edit_data
            return self._create_success_result(
                response=response_text,
                tools_used=[],
                processing_time=processing_time,
                additional_data=additional,
            )
            
            # Also store directly in state for API streaming (like proofreading agent)
            if editor_ops:
                result["editor_operations"] = editor_ops
                result["manuscript_edit"] = manuscript_edit_data
            
            return result

        except Exception as e:
            logger.error(f"❌ FictionEditingAgent failed: {e}")
            return self._create_success_result(
                response="Editing agent encountered an error.", tools_used=[], processing_time=0.0
            )


