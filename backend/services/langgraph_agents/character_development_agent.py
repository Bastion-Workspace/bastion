"""
Character Development Agent

Gated to character documents. Consumes active editor buffer, cursor/selection,
and cascades outline → rules/style/characters where available. Produces
EditorOperations for Prefer Editor HITL application.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent
from models.agent_response_models import ManuscriptEdit, EditorOperation
from services.file_context_loader import FileContextLoader

from utils.chapter_scope import paragraph_bounds
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

class CharacterDevelopmentAgent(BaseAgent):
	def __init__(self):
		super().__init__("character_development_agent")
		logger.info("🧭 BULLY! Character Development Agent ready to shape the cast!")

	def _build_system_prompt(self) -> str:
		return (
			"You are a Character Development Assistant for type: character files. Persona disabled."
			" Preserve frontmatter; write clean Markdown in body.\n\n"
			"STRUCTURED OUTPUT REQUIRED: Return ONLY raw JSON (no prose, no markdown, no code fences) matching this schema:\n"
			"{\n"
			"  \"type\": \"ManuscriptEdit\",\n"
			"  \"target_filename\": string,\n"
			"  \"scope\": one of [\"paragraph\", \"chapter\", \"multi_chapter\"],\n"
			"  \"summary\": string,\n"
			"  \"chapter_index\": integer|null,\n"
			"  \"safety\": one of [\"low\", \"medium\", \"high\"],\n"
			"  \"operations\": [\n"
			"    {\n"
			"      \"op_type\": one of [\"replace_range\", \"delete_range\", \"insert_after_heading\"],\n"
			"      \"start\": integer (approximate),\n"
			"      \"end\": integer (approximate),\n"
			"      \"text\": string,\n"
			"      \"original_text\": string (REQUIRED for replace/delete, optional for insert - EXACT verbatim text from file),\n"
			"      \"anchor_text\": string (optional - for inserts, exact line to insert after),\n"
			"      \"left_context\": string (optional - text before target),\n"
			"      \"right_context\": string (optional - text after target),\n"
			"      \"occurrence_index\": integer (optional, default 0 if text appears multiple times)\n"
			"    }\n"
			"  ]\n"
			"}\n\n"
			"OUTPUT RULES:\n"
			"- Output MUST be a single JSON object only.\n"
			"- Do NOT include triple backticks or language tags.\n"
			"- Do NOT include explanatory text before or after the JSON.\n\n"
			"FORMATTING CONTRACT (CHARACTER FILES):\n"
			"- Never emit YAML frontmatter in operations[].text; preserve existing YAML.\n"
			"- Use Markdown headings and bullet lists for sections.\n"
			"- Preferred major-character scaffold: Basic Information, Personality (traits/strengths/flaws), Dialogue Patterns, Internal Monologue, Relationships, Character Arc.\n"
			"- Supporting cast: concise entries (Role, Traits, Speech, Relation to MC, Notes).\n"
			"- Relationships doc: pairs with Relationship Type, Dynamics, Conflict Sources, Interaction Patterns, Evolution.\n\n"
			"EDIT RULES:\n"
			"1) Make surgical edits near cursor/selection unless re-organization is requested.\n"
			"2) Maintain existing structure; update in place; avoid duplicate headings.\n"
			"3) Enforce universe consistency against Rules and outline-provided character network.\n"
			"4) NO PLACEHOLDER FILLERS: If a requested section has no content yet, create the heading only and leave the body blank. Do NOT insert placeholders like '[To be developed]' or 'TBD'.\n"
			"ANCHOR REQUIREMENTS (CRITICAL):\n"
			"For EVERY operation, you MUST provide precise anchors:\n\n"
			"REVISE/DELETE Operations:\n"
			"- ALWAYS include 'original_text' with EXACT, VERBATIM text from the file\n"
			"- Minimum 10-20 words, include complete sentences with natural boundaries\n"
			"- Copy and paste directly - do NOT retype or modify\n"
			"- ⚠️ NEVER include header lines (###, ##) in original_text!\n"
			"- OR provide both 'left_context' and 'right_context' (exact surrounding text)\n\n"
			"INSERT Operations (PREFERRED for adding content below headers!):\n"
			"- **PRIMARY METHOD**: Use op_type='insert_after_heading' with anchor_text='### Header' when adding content below ANY header\n"
			"- Provide 'anchor_text' with EXACT, COMPLETE header line to insert after (verbatim from file)\n"
			"- This is the SAFEST method - it NEVER deletes headers, always inserts AFTER them\n"
			"- Use this even when the section has placeholder text - the resolver will position correctly\n"
			"- ALTERNATIVE: Provide 'original_text' with text to insert after\n"
			"- FALLBACK: Provide 'left_context' with text before insertion point (minimum 10-20 words)\n\n"
			"Additional Options:\n"
			"- 'occurrence_index' if text appears multiple times (0-based, default 0)\n"
			"- Start/end indices are approximate; anchors take precedence\n\n"
			"=== OPERATION TYPE EXAMPLES ===\n\n"
			"Example 1: Adding content below a header - ALWAYS USE insert_after_heading!\n"
			"✅ CORRECT (PREFERRED METHOD):\n"
			"{\n"
			"  \"op_type\": \"insert_after_heading\",\n"
			"  \"anchor_text\": \"### Traits\",\n"
			"  \"text\": \"- Analytical thinker\\\\n- Selfish in most matters\",\n"
			"  \"note\": \"Adding character traits\"\n"
			"}\n\n"
			"❌ WRONG - Don't use replace_range with header included:\n"
			"{\n"
			"  \"op_type\": \"replace_range\",\n"
			"  \"original_text\": \"### Traits\\\\n- [To be developed]\",  // ❌ Header will be deleted!\n"
			"}\n\n"
			"Example 2: Replacing placeholder text - USE replace_range to replace ONLY the placeholder!\n"
			"✅ CORRECT - Text ends right after last word:\n"
			"{\n"
			"  \"op_type\": \"replace_range\",\n"
			"  \"original_text\": \"- [To be developed based on story needs]\",\n"
			'  "text": "- Analytical thinker\\n- Selfish in most matters"  ← Ends after "matters", NO \\n!\n'
			"}\n"
			"Result: Clean replacement with proper spacing.\n\n"
			"❌ WRONG - Trailing \\n after last line:\n"
			"{\n"
			"  \"op_type\": \"replace_range\",\n"
			"  \"original_text\": \"- [To be developed]\",\n"
			'  "text": "- Analytical thinker\\n- Selfish\\n"  ← Extra \\n creates blank line!\n'
			"}\n\n"
			"❌ WRONG - Trailing \\n\\n after last line:\n"
			"{\n"
			"  \"op_type\": \"replace_range\",\n"
			"  \"original_text\": \"- [To be developed]\",\n"
			'  "text": "- Analytical thinker\\n- Selfish\\n\\n"  ← \\n\\n creates 2 blank lines!\n'
			"}\n\n"
			"❌ WRONG - Don't use insert_after_heading when placeholder exists:\n"
			"{\n"
			"  \"op_type\": \"insert_after_heading\",\n"
			"  \"anchor_text\": \"### Traits\",\n"
			"  \"text\": \"- Analytical thinker\",\n"
			"  \"note\": \"This will KEEP the placeholder text below!\"\n"
			"}\n\n"
			"Example 3: Deleting placeholder (USE delete_range)\n"
			"✅ CORRECT:\n"
			"{\n"
			"  \"op_type\": \"delete_range\",\n"
			"  \"original_text\": \"- [Connections to other characters to be established]\",\n"
			"  \"text\": \"\",\n"
			"  \"note\": \"Removing placeholder\"\n"
			"}\n\n"
			"=== DECISION TREE FOR OPERATION TYPE ===\n"
			"1. Section is COMPLETELY EMPTY below a header? → insert_after_heading with anchor_text\n"
			"2. Section has PLACEHOLDER text like '[To be developed]'? → replace_range to replace ONLY the placeholder (NO header!)\n"
			"3. Section has EXISTING content you want to UPDATE? → replace_range with original_text (NO headers!)\n"
			"4. Want to DELETE specific content? → delete_range with original_text (NO headers!)\n\n"
			"KEY RULES:\n"
			"- If placeholder text exists, REPLACE it with replace_range - don't insert above it!\n\n"
			"=== SPACING RULES (CRITICAL - READ CAREFULLY!) ===\n"
			"\n"
			"YOUR TEXT MUST END IMMEDIATELY AFTER THE LAST CHARACTER!\n"
			"\n"
			"✅ CORRECT - Text ends right after last word:\n"
			'  "text": "- Analytical thinker\\n- Selfish in most matters"  ← Ends after "matters" with NO \\n\n'
			"\n"
			"BREAKDOWN: What the \\n characters mean:\n"
			'  "- Analytical thinker   ← Line 1\n'
			'   \\n                      ← Newline separator (connects line 1 to line 2)\n'
			'   - Selfish in most matters"  ← Line 2 ends HERE, no \\n after it!\n'
			"\n"
			"✅ CORRECT three-line example:\n"
			'  "text": "- Item 1\\n- Item 2\\n- Item 3"  ← Last char is "3", NO \\n after it\n'
			"\n"
			"❌ WRONG - Adding \\n after last line:\n"
			'  "text": "- Analytical thinker\\n- Selfish\\n"  ← Extra \\n after last line = WRONG!\n'
			"\n"
			"❌ WRONG - Adding \\n\\n after last line:\n"
			'  "text": "- Item 1\\n- Item 2\\n\\n"  ← Creates 2 blank lines after!\n'
			"\n"
			"❌ WRONG - Double \\n\\n between items:\n"
			'  "text": "- Item 1\\n\\n- Item 2"  ← Creates blank line between items!\n'
			"\n"
			"IRON-CLAD RULE: Count your \\n characters carefully!\n"
			"- Between lines: ONE \\n\n"
			"- After last line: ZERO \\n (nothing!)\n"
			"\n"
			"The system adds all necessary spacing around your content automatically.\n\n"
			"NO PLACEHOLDER TEXT: Leave empty sections blank, do NOT insert '[To be developed]' or 'TBD'.\n"
		)

	async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
		try:
			shared_memory = state.get("shared_memory", {}) or {}
			active_editor = shared_memory.get("active_editor", {}) or {}

			text = active_editor.get("content", "") or ""
			filename = active_editor.get("filename") or "character.md"
			frontmatter = active_editor.get("frontmatter", {}) or {}
			cursor_offset = int(active_editor.get("cursor_offset", -1))
			selection_start = int(active_editor.get("selection_start", -1))
			selection_end = int(active_editor.get("selection_end", -1))

			# Gate by type: character (strict)
			doc_type = str(frontmatter.get("type", "")).strip().lower()
			if doc_type != "character":
				return self._create_success_result(
					response="Active editor is not a character file; skipping.", tools_used=[], processing_time=0.0, additional_data={"skipped": True}
				)

			# Cascade: outline → rules/style/characters (via FileContextLoader)
			if active_editor.get("canonical_path"):
				frontmatter = { **frontmatter, "__canonical_path__": active_editor.get("canonical_path") }
			loader = FileContextLoader()
			loaded = loader.load_referenced_context(filename, frontmatter)
			style_text = _strip_frontmatter_block(loaded.style.content) if loaded.style else None
			outline_body = _strip_frontmatter_block(loaded.outline.content) if loaded.outline else None
			rules_body = _strip_frontmatter_block(loaded.rules.content) if loaded.rules else None
			character_bodies = [_strip_frontmatter_block(c.content) for c in (loaded.characters or [])]

			# Scope: prefer selection; else paragraph around cursor
			para_start, para_end = paragraph_bounds(text, cursor_offset if cursor_offset >= 0 else 0)
			if selection_start >= 0 and selection_end > selection_start:
				para_start, para_end = selection_start, selection_end

			# Build messages
			system_prompt = self._build_system_prompt()
			try:
				current_request = (self._extract_current_user_query(state) or "").strip()
			except Exception:
				current_request = ""

			messages = [
				{"role": "system", "content": system_prompt},
				{"role": "system", "content": f"Current Date/Time: {datetime.now().isoformat()}"},
				{
					"role": "user",
					"content": (
						"=== CHARACTER CONTEXT ===\n"
						f"File: {filename}\n\n"
						"Current Character (frontmatter stripped):\n" + _strip_frontmatter_block(text) + "\n\n"
						+ ("=== OUTLINE (if present) ===\n" + outline_body + "\n\n" if outline_body else "")
						+ ("=== RULES (if present) ===\n" + rules_body + "\n\n" if rules_body else "")
						+ ("=== STYLE GUIDE (if present) ===\n" + style_text + "\n\n" if style_text else "")
						+ ("".join(["=== RELATED CHARACTER DOC ===\n" + b + "\n\n" for b in character_bodies]) if character_bodies else "")
						+ ("REVISION MODE: Apply minimal targeted edits; use paragraph-level replace_range ops.\n\n" if current_request and any(k in current_request.lower() for k in ["revise", "revision", "tweak", "adjust", "polish", "tighten"]) else "")
						+ "Provide a ManuscriptEdit JSON plan strictly within scope."
					),
				},
			]
			if current_request:
				messages.append({
					"role": "user",
					"content": (
						f"USER REQUEST: {current_request}\n\n"
						"CRITICAL ANCHORING INSTRUCTIONS:\n"
						"- For REVISE/DELETE: Provide 'original_text' with EXACT, VERBATIM text from file (10-20+ words, complete sentences)\n"
						"- For INSERT: Provide 'anchor_text' or 'original_text' with exact line to insert after, OR 'left_context' with text before insertion\n"
						"- Copy text directly from the file - do NOT retype or paraphrase\n"
						"- Without precise anchors, the operation WILL FAIL"
					)
				})

			# Call model
			chat_service = await self._get_chat_service()
			model_name = await self._get_model_name()
			start_time = datetime.now()
			response = await chat_service.openai_client.chat.completions.create(
				model=model_name,
				messages=messages,
				temperature=0.35,
			)

			content = response.choices[0].message.content or "{}"
			content = _unwrap_json_response(content)
			structured: Optional[ManuscriptEdit] = None
			try:
				raw = json.loads(content)
				if isinstance(raw, dict) and isinstance(raw.get("operations"), list):
					raw.setdefault("target_filename", filename)
					raw.setdefault("scope", "paragraph")
					raw.setdefault("summary", "Planned character edit generated from context.")
					ops: List[Dict[str, Any]] = []
					for op in raw["operations"]:
						if not isinstance(op, dict):
							continue
						start_ix = int(op.get("start", para_start))
						end_ix = int(op.get("end", para_end))
						op_type = op.get("op_type")
						if op_type not in ("replace_range", "delete_range", "insert_after_heading"):
							op_type = "replace_range"
						# PRESERVE ANCHOR FIELDS from LLM response for resolver
						ops.append({
							"op_type": op_type,
							"start": max(0, min(len(text), start_ix)),
							"end": max(0, min(len(text), end_ix)),
							"text": op.get("text", ""),
							"original_text": op.get("original_text"),
							"anchor_text": op.get("anchor_text"),
							"left_context": op.get("left_context"),
							"right_context": op.get("right_context"),
							"occurrence_index": op.get("occurrence_index", 0),
							"pre_hash": ""
						})
					raw["operations"] = ops
					structured = ManuscriptEdit(**raw)
				else:
					structured = ManuscriptEdit.parse_raw(content)
			except Exception:
				structured = None

			if structured is None:
				return self._create_success_result(
					response="Failed to produce a valid Character edit plan. Ensure ONLY raw JSON ManuscriptEdit with operations is returned (no code fences or prose).", tools_used=[], processing_time=(datetime.now() - start_time).total_seconds(), additional_data={"raw": content}
				)

			# Inject pre_hash and clamp when in revision-like requests
			lower_req = (current_request or "").lower()
			revision_mode = any(k in lower_req for k in ["revise", "revision", "tweak", "adjust", "polish", "tighten", "edit only"]) if lower_req else False

			ed_ops: List[EditorOperation] = []
			# Calculate frontmatter end to convert body-relative indices to absolute
			try:
				body_only_text = _strip_frontmatter_block(text)
				fm_end_idx = len(text) - len(body_only_text)
			except Exception:
				fm_end_idx = 0
			# Build helpers to anchor edits within '## Basic Information'
			def _find_section_bounds(doc: str, heading: str) -> (int, int):
				try:
					import re as _re
					pat = _re.compile(rf"^##\s+{_re.escape(heading)}\s*$", _re.MULTILINE | _re.IGNORECASE)
					m = pat.search(doc)
					if not m:
						return -1, -1
					start = m.end() + 1
					# Next top-level or same-level heading
					m2 = _re.search(r"^##\s+.+$", doc[start:], _re.MULTILINE)
					end = start + m2.start() if m2 else len(doc)
					return start, end
				except Exception:
					return -1, -1

			def _replace_or_insert_field(doc: str, sec_start: int, sec_end: int, label: str, value: str) -> (int, int, str):
				"""Find '- **Label**:' line in section and replace it; else insert at end of section.
				Return (start, end, new_text) as a replace_range op payload.
				"""
				try:
					import re as _re
					section = doc[sec_start:sec_end]
					pat = _re.compile(rf"(^\s*-\s*\*\*{_re.escape(label)}\*\*\s*:\s*).*$", _re.MULTILINE)
					m = pat.search(section)
					newline_val = f"- **{label}**: {value}"
					if m:
						line_start = sec_start + m.start(0)
						line_end = sec_start + m.end(0)
						return line_start, line_end, newline_val
					# No match: insert at end of section, ensure spacing
					insert_at = sec_end
					left_tail = doc[max(0, insert_at-2):insert_at]
					if left_tail.endswith("\n\n"):
						prefix = ""
					elif left_tail.endswith("\n"):
						prefix = "\n"
					else:
						prefix = "\n\n"
					return insert_at, insert_at, f"{prefix}{newline_val}\n"
				except Exception:
					# On error, append to end of document
					insert_at = len(doc)
					return insert_at, insert_at, f"\n\n- **{label}**: {value}\n"

			def _anchor_heading_section(doc: str, proposed_text: str) -> Optional[tuple]:
				"""If proposed_text begins with a heading (## or ###), find matching heading in doc and
				return (start, end, normalized_text) to replace the entire section from the heading line
				to the next heading of same or higher level.
				"""
				try:
					import re as _re
					m = _re.match(r"^(#{2,6})\s+(.+?)\s*$", proposed_text.strip().split('\n', 1)[0])
					if not m:
						return None
					hevel = len(m.group(1))
					title = m.group(2).strip()
					# Find matching heading (same level, case-insensitive, normalized title)
					pattern = rf"^{'#'*hevel}\s+{_re.escape(title)}\s*$"
					mh = _re.search(pattern, doc, _re.MULTILINE | _re.IGNORECASE)
					if not mh:
						return None
					sec_start = mh.start()
					# End at next heading with level <= current
					next_pat = rf"^#{{1,{hevel}}}\s+.+$"
					mn = _re.search(next_pat, doc[mh.end():], _re.MULTILINE)
					sec_end = mh.end() + (mn.start() if mn else len(doc) - mh.end())
					# Normalize proposed_text: ensure it includes a single heading line
					pt = proposed_text
					return (sec_start, sec_end, pt)
				except Exception:
					return None

			def _infer_heading_hint_from_text(proposed_text: str) -> Optional[Dict[str, Any]]:
				"""Infer a reasonable heading hint (level/title) based on the content of proposed_text."""
				try:
					pt = (proposed_text or "").lower()
					# Dialogue patterns cues
					if any(k in pt for k in ["dialogue", "speech", "exclamation", "phrases", "expressions", "vocabulary", "accent"]):
						return {"level": 2, "title": "Dialogue Patterns"}
					# Relationships cues
					if any(k in pt for k in ["relationship", "sister", "brother", "mother", "father", "friend", "jill"]):
						return {"level": 2, "title": "Relationships"}
					# Strengths cues
					if any(k in pt for k in ["strength", "strong", "analytical", "problem-solving", "loyal", "protective", "work ethic"]):
						return {"level": 3, "title": "Strengths"}
					# Flaws cues
					if any(k in pt for k in ["flaw", "selfish", "lacks empathy", "manipulative", "weakness", "self-centered"]):
						return {"level": 3, "title": "Flaws"}
					# Traits default
					if any(k in pt for k in ["trait", "methodical", "animated", "pragmatic", "precise", "direct"]):
						return {"level": 3, "title": "Traits"}
					return None
				except Exception:
					return None

			# Detect target section from the user's request for section-aware anchoring when no heading is provided
			anchor_start, anchor_end = -1, -1
			try:
				lr = (current_request or "").lower()
				# Map keywords to headings
				keyword_sections = [
					(["dialogue", "speech", "patterns", "phrases", "expressions", "vocabulary"], r"^##\s+Dialogue\s+Patterns\s*$"),
					(["internal monologue", "thought", "internal voice"], r"^##\s+Internal\s+Monologue\s*$"),
					(["relationship", "connections"], r"^##\s+Relationships\s*$"),
					(["character arc", "arc"], r"^##\s+Character\s+Arc\s*$"),
					(["traits"], r"^###\s+Traits\s*$"),
					(["strengths"], r"^###\s+Strengths\s*$"),
					(["flaws", "weaknesses"], r"^###\s+Flaws\s*$"),
				]
				import re as _re
				for keys, hx in keyword_sections:
					if any(k in lr for k in keys):
						m = _re.search(hx, text, flags=_re.IGNORECASE | _re.MULTILINE)
						if m:
							ss = m.end() + 1
							m2 = _re.search(r"^#\s+.+$|^##\s+.+$|^###\s+.+$", text[ss:], flags=_re.MULTILINE)
							se = ss + (m2.start() if m2 else len(text) - ss)
							anchor_start, anchor_end = ss, se
							break
			except Exception:
				anchor_start, anchor_end = -1, -1

			for op in structured.operations:
				# Use absolute indices from normalization; do not re-apply fm_end_idx
				start = max(0, min(len(text), op.start))
				end = max(0, min(len(text), op.end))
				# If not revision mode and we detected a target section, anchor default inserts to end of section
				if not revision_mode and anchor_start != -1 and anchor_end != -1:
					# Only override when the model didn't specify a real range (start==end or near 0)
					if start == end or start < fm_end_idx + 2:
						start = anchor_end
						end = anchor_end
				# Protect YAML frontmatter from edits: never modify or insert before frontmatter end
				if start < fm_end_idx:
					# Skip any deletion targeting frontmatter
					if op.op_type == "delete_range":
						continue
					# If fully within frontmatter, convert to insert at body start
					if end <= fm_end_idx:
						start = fm_end_idx
						end = fm_end_idx
					else:
						# Overlap: clamp start to body start
						start = fm_end_idx
				# Clamp for revision-like
				if revision_mode and op.op_type != "delete_range":
					if selection_start >= 0 and selection_end > selection_start:
						start = max(selection_start, start)
						end = min(selection_end, end)
				# sanitize insert text and, for Basic Information, anchor Age/Occupation updates
				try:
					if isinstance(op.text, str):
						op.text = _strip_frontmatter_block(op.text)
						# Strip placeholder filler like '[To be developed]' and 'TBD'
						try:
							import re as _re
							op.text = _re.sub(r"^\s*-\s*\[To be.*?\]|^\s*\[To be.*?\]|^\s*TBD\s*$", "", op.text, flags=_re.IGNORECASE | _re.MULTILINE)
							op.text = _re.sub(r"\n{3,}", "\n\n", op.text)
						except Exception:
							pass
						# If anchored section is known and this is a pure insertion, prefer replacing placeholder bullets in that section
						if not revision_mode and anchor_start != -1 and anchor_end != -1 and start == end:
							sec_text = text[anchor_start:anchor_end]
						# De-duplicate bullets already present in section
						try:
							import re as _re
							prop_lines = [ln for ln in (op.text or '').splitlines() if ln.strip().startswith('-')]
							filtered = []
							for bl in prop_lines:
								if bl.strip() and bl.strip() not in sec_text:
									filtered.append(bl)
							# If section has placeholder bullets, replace them instead of appending
							ph_rx = _re.compile(r"^(\s*-\s*\[[^\n]*\]\s*)$", _re.MULTILINE)
							ph_matches = list(ph_rx.finditer(sec_text))
							if ph_matches and filtered:
								# Compute contiguous placeholder block bounds
								block_start = ph_matches[0].start()
								block_end = ph_matches[-1].end()
								# Map to absolute positions
								start = anchor_start + block_start
								end = anchor_start + block_end
								op.text = "\n".join(filtered) + "\n"
								# Ensure not to merge with next heading
								right_head = text[end:end+2]
								if not right_head.startswith("\n") and not op.text.endswith("\n\n"):
									op.text = f"{op.text}\n"
							elif filtered:
								# No placeholders; keep original anchored append content using filtered bullets
								op.text = ("\n\n" if not sec_text.endswith("\n\n") else "") + "\n".join(filtered) + "\n"
						except Exception:
							pass
						# If proposed text begins with a heading, anchor to that section
						anch = _anchor_heading_section(text, op.text)
						if anch:
							start, end, op.text = anch
						# Ensure a clean separation when inserting new sections (start == end)
						if start == end:
							left_tail = text[max(0, start-2):start]
							if left_tail.endswith("\n\n"):
								needed_prefix = ""
							elif left_tail.endswith("\n"):
								needed_prefix = "\n"
							else:
								needed_prefix = "\n\n"
							try:
								leading_stripped = re.sub(r'^\n+', '', op.text)
								op.text = f"{needed_prefix}{leading_stripped}"
							except Exception:
								op.text = f"{needed_prefix}{op.text}"
							# Add trailing spacing if the next char is not a newline
							right_head = text[start:start+2]
							if right_head.startswith("\n\n"):
								pass
							elif right_head.startswith("\n"):
								if not op.text.endswith("\n"):
									op.text = f"{op.text}\n"
							else:
								if op.text.endswith("\n\n"):
									pass
								elif op.text.endswith("\n"):
									op.text = f"{op.text}\n"
								else:
									op.text = f"{op.text}\n\n"
						# Field-aware anchoring for Basic Information bullets
						sec_start, sec_end = _find_section_bounds(text, "Basic Information")
						if sec_start != -1 and sec_end != -1:
							# Attempt to extract Age/Occupation from op.text when it looks like a bullet update
							try:
								import re as _re
								m_age = _re.search(r"\*\*Age\*\*\s*:\s*([^\n]+)", op.text, _re.IGNORECASE)
								m_occ = _re.search(r"\*\*Occupation\*\*\s*:\s*([^\n]+)", op.text, _re.IGNORECASE)
								if m_age:
									val = m_age.group(1).strip()
									rs, re_, new_line = _replace_or_insert_field(text, sec_start, sec_end, "Age", val)
									start, end, op.text = rs, re_, new_line
								elif m_occ:
									val = m_occ.group(1).strip()
									rs, re_, new_line = _replace_or_insert_field(text, sec_start, sec_end, "Occupation", val)
									start, end, op.text = rs, re_, new_line
							except Exception:
								pass
				except Exception:
					pass
				# Final resolution with shared resolver (uses original/context/heading_hint/selection)
				selection = {"start": selection_start, "end": selection_end} if selection_start >= 0 and selection_end >= 0 else None
				# If op lacks anchors and is a pure insert, infer heading hint per op text
				per_op_heading_hint = None
				try:
					no_anchors = not any([getattr(op, "original", None), getattr(op, "original_text", None), getattr(op, "left_context", None), getattr(op, "right_context", None)])
					if no_anchors and start == end:
						per_op_heading_hint = _infer_heading_hint_from_text(op.text)
				except Exception:
					per_op_heading_hint = None
				# Build operation dict for resolver with all available anchor fields
				op_dict = {
					"original": getattr(op, "original", None),
					"original_text": getattr(op, "original_text", None),
					"anchor_text": getattr(op, "anchor_text", None),
					"left_context": getattr(op, "left_context", None),
					"right_context": getattr(op, "right_context", None),
					"occurrence_index": getattr(op, "occurrence_index", 0),
					"text": op.text,
					"op_type": op.op_type,
				}
				logger.info(f"🔍 Calling resolver with anchors: original_text={bool(op_dict.get('original_text'))}, "
					f"anchor_text={bool(op_dict.get('anchor_text'))}, "
					f"left_context={bool(op_dict.get('left_context'))}, "
					f"right_context={bool(op_dict.get('right_context'))}")
				
				# CRITICAL: If original_text starts with a heading, ensure replacement text includes it too
				original_text_val = op_dict.get('original_text')
				if original_text_val and op.text and op.op_type == "replace_range":
					try:
						# Check if original starts with a heading
						orig_first_line = original_text_val.strip().split('\n')[0]
						if orig_first_line.startswith('#'):
							# Check if replacement text starts with the same heading
							replacement_first_line = op.text.strip().split('\n')[0] if op.text.strip() else ""
							if not replacement_first_line.startswith('#'):
								# Prepend the heading to the replacement text
								logger.info(f"🔧 HEADING PRESERVATION: Prepending '{orig_first_line}' to replacement text")
								op.text = f"{orig_first_line}\n{op.text}"
					except Exception as e:
						logger.warning(f"⚠️ Heading preservation check failed: {e}")
				
				resolved_start, resolved_end, resolved_text, resolved_confidence = resolve_operation(
					text,
					op_dict,
					selection=selection,
					heading_hint=per_op_heading_hint,
					frontmatter_end=fm_end_idx,
					require_anchors=True,
				)
				op.start = resolved_start
				op.end = resolved_end
				op.text = resolved_text
				op.confidence = resolved_confidence
				logger.info(f"📍 Operation resolved: [{resolved_start}:{resolved_end}] confidence={resolved_confidence:.2f}")
				pre_slice = text[resolved_start:resolved_end]
				op.pre_hash = _slice_hash(pre_slice)
				ed_ops.append(op)
			structured.operations = ed_ops

			processing_time = (datetime.now() - start_time).total_seconds()
			preview = "\n\n".join([(getattr(op, "text", "") or "").strip() for op in structured.operations if (getattr(op, "text", "") or "").strip()])
			response_text = preview if preview else (structured.summary or "Edit plan ready.")

			ops_dump = [op.model_dump() for op in structured.operations]
			additional = {"content_preview": response_text[:2000]}
			if ops_dump:
				additional["editor_operations"] = ops_dump
				additional["manuscript_edit"] = structured.model_dump()
			return self._create_success_result(
				response=response_text,
				tools_used=[],
				processing_time=processing_time,
				additional_data=additional,
			)
		except Exception as e:
			logger.error(f"❌ CharacterDevelopmentAgent failed: {e}")
			return self._create_success_result(
				response="Character agent encountered an error.", tools_used=[], processing_time=0.0
			)
