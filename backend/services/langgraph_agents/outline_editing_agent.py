"""
Outline Editing / Development Agent

Gated to outline documents. Consumes full outline body (frontmatter stripped),
loads Style, Rules, and Character references directly from this file's
frontmatter, and emits editor operations for Prefer Editor HITL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent
from models.agent_response_models import ManuscriptEdit, EditorOperation, OutlineClarificationRequest
from services.file_context_loader import FileContextLoader
from utils.editor_operations_resolver import resolve_operation
from config import settings
from pathlib import Path

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
	"""Extract a raw JSON object from LLM output if wrapped in code fences or prose."""
	try:
		# If already valid JSON, return as-is
		json.loads(content)
		return content
	except Exception:
		pass

	try:
		import re
		text = content.strip()
		# Strip fenced code blocks ```json ... ``` or ``` ... ```
		text = re.sub(r"^```(?:json)?\s*\n([\s\S]*?)\n```\s*$", r"\1", text)
		# Try direct parse
		try:
			json.loads(text)
			return text
		except Exception:
			pass
		# Heuristic: take first balanced {...}
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


class OutlineEditingAgent(BaseAgent):
	def __init__(self):
		super().__init__("outline_editing_agent")
		logger.info("🗺️ BULLY! Outline Editing Agent ready to structure the campaign!")

	def _build_system_prompt(self) -> str:
		return (
			"You are an Outline Development Assistant for type: outline files. Persona disabled."
			" Preserve frontmatter; operate on Markdown body only.\n\n"
			"ROOSEVELT'S \"BEST EFFORT\" DOCTRINE:\n"
			"ALWAYS provide outline content - never leave the user empty-handed!\n\n"
			"OPERATIONAL STRATEGY:\n"
			"1. **ALWAYS create content** based on available information\n"
			"2. **Make reasonable inferences** from existing context\n"
			"3. **ONLY skip clarification** when you're highly confident (>0.85)\n"
			"4. **Ask questions SPARINGLY** - only for truly critical gaps\n\n"
			"WHEN TO REQUEST CLARIFICATION (ONLY CRITICAL GAPS):\n"
			"✅ Ask when:\n"
			"- User request is genuinely ambiguous with multiple valid interpretations\n"
			"- A critical plot element contradicts established rules/characters\n"
			"- Creating the wrong content would be worse than asking\n\n"
			"❌ DO NOT ask when:\n"
			"- You can make reasonable inferences from context\n"
			"- Existing chapters/rules/characters provide guidance\n"
			"- User request is clear enough for basic implementation\n"
			"- You can create placeholder structure that's useful\n\n"
			"STRUCTURED OUTPUT OPTIONS:\n\n"
			"OPTION 1 - OutlineClarificationRequest (RARE - only for critical ambiguity):\n"
			"{\n"
			"  \"task_status\": \"incomplete\",\n"
			"  \"clarification_needed\": true,\n"
			"  \"questions\": [\"Critical question that blocks progress?\"],\n"
			"  \"context\": \"Why this blocks content creation\",\n"
			"  \"missing_elements\": [\"critical_element\"],\n"
			"  \"suggested_direction\": \"Optional suggestion\",\n"
			"  \"section_affected\": \"Chapter 3\",\n"
			"  \"confidence_without_clarification\": 0.3\n"
			"}\n"
			"⚠️ USE THIS SPARINGLY - Only when creating content would be genuinely harmful!\n\n"
			"OPTION 2 - ManuscriptEdit (DEFAULT - use >90% of the time):\n"
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
			"STRICT OUTLINE SCAFFOLD:\n"
			"# Overall Synopsis\n"
			"# Notes\n"
			"# Characters\n"
			"- Protagonists\n"
			"- Antagonists\n"
			"- Supporting Characters\n"
			"# Outline\n"
			"## Chapter 1\n"
			"## Chapter 2\n"
			"... (continue numerically)\n\n"
			"CHARACTER LIST FORMAT:\n"
			"- Protagonists\n  - Name - Brief role\n"
			"- Antagonists\n  - Name - Brief role\n"
			"- Supporting Characters\n  - Name - Brief role\n\n"
			"CHAPTER CONTENT RULES:\n"
			"- Start with a 3-5 sentence summary paragraph.\n"
			"- Follow with main beats as '-' bullets; each may have up to two '  -' sub-bullets.\n"
			"- Max 8-10 main beats per chapter.\n"
			"- Focus on plot events, actions, reveals, conflicts; avoid prose/dialogue.\n"
			"- Use exact '## Chapter N' headings; never titles.\n\n"
			"EDIT RULES:\n"
			"1) Make surgical edits near cursor/selection unless re-organization is requested.\n"
			"2) Maintain scaffold; if missing, create only requested sections.\n"
			"3) Enforce universe consistency against directly referenced Rules and Characters; match Style.\n"
			"4) NO PLACEHOLDER FILLERS: If a requested section has no content yet, create the heading only and leave the body blank. Do NOT insert placeholders like '[To be developed]' or 'TBD'.\n\n"
			"ANCHOR REQUIREMENTS (CRITICAL):\n"
			"For EVERY operation, you MUST provide precise anchors:\n\n"
			"REVISE/DELETE Operations:\n"
			"- ALWAYS include 'original_text' with EXACT, VERBATIM text from the file\n"
			"- Minimum 10-20 words, include complete sentences with natural boundaries\n"
			"- Copy and paste directly - do NOT retype or modify\n"
			"- ⚠️ NEVER include header lines (###, ##, #) in original_text!\n"
			"- OR provide both 'left_context' and 'right_context' (exact surrounding text)\n\n"
			"INSERT Operations (PREFERRED for adding content below headers!):\n"
			"- **PRIMARY METHOD**: Use op_type='insert_after_heading' with anchor_text='## Chapter N' when adding content below ANY header\n"
			"- Provide 'anchor_text' with EXACT, COMPLETE header line to insert after (verbatim from file)\n"
			"- This is the SAFEST method - it NEVER deletes headers, always inserts AFTER them\n"
			"- Use this for adding chapter summaries, beats, or any content below headers\n"
			"- ALTERNATIVE: Provide 'original_text' with text to insert after\n"
			"- FALLBACK: Provide 'left_context' with text before insertion point (minimum 10-20 words)\n\n"
			"Additional Options:\n"
			"- 'occurrence_index' if text appears multiple times (0-based, default 0)\n"
			"- Start/end indices are approximate; anchors take precedence\n\n"
			"=== DECISION TREE FOR OPERATION TYPE ===\n"
			"1. Chapter/section is COMPLETELY EMPTY? → insert_after_heading with anchor_text=\"## Chapter N\"\n"
			"2. Chapter has PLACEHOLDER or existing content to replace? → replace_range (NO headers in original_text!)\n"
			"3. Deleting SPECIFIC content? → delete_range with original_text (NO headers!)\n\n"
			"⚠️ CRITICAL: When replacing placeholder content, use 'replace_range' on ONLY the placeholder!\n"
			"⚠️ NEVER include headers in 'original_text' for replace_range - headers will be deleted!\n"
			"✅ Correct: {\"op_type\": \"replace_range\", \"original_text\": \"[Placeholder beats]\", \"text\": \"- Beat 1\\n- Beat 2\"}\n"
			"❌ Wrong: {\"op_type\": \"insert_after_heading\", \"anchor_text\": \"## Chapter 1\", \"text\": \"...\"} when placeholder exists!\n\n"
			"=== SPACING RULES (CRITICAL - READ CAREFULLY!) ===\n"
			"YOUR TEXT MUST END IMMEDIATELY AFTER THE LAST CHARACTER!\n\n"
			'✅ CORRECT: "- Beat 1\\n- Beat 2\\n- Beat 3"  ← Ends after "3" with NO \\n\n'
			'❌ WRONG: "- Beat 1\\n- Beat 2\\n"  ← Extra \\n after last line creates blank line!\n'
			'❌ WRONG: "- Beat 1\\n- Beat 2\\n\\n"  ← \\n\\n creates 2 blank lines!\n'
			'❌ WRONG: "- Beat 1\\n\\n- Beat 2"  ← Double \\n\\n between items creates blank line!\n\n'
			"IRON-CLAD RULE: After last line = ZERO \\n (nothing!)\n"
		)

	async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
		try:
			shared_memory = state.get("shared_memory", {}) or {}
			active_editor = shared_memory.get("active_editor", {}) or {}

			text = active_editor.get("content", "") or ""
			filename = active_editor.get("filename") or "outline.md"
			frontmatter = active_editor.get("frontmatter", {}) or {}
			cursor_offset = int(active_editor.get("cursor_offset", -1))
			selection_start = int(active_editor.get("selection_start", -1))
			selection_end = int(active_editor.get("selection_end", -1))

			# Gate by type: outline (strict)
			doc_type = str(frontmatter.get("type", "")).strip().lower()
			if doc_type != "outline":
				return self._create_success_result(
					response="Active editor is not an outline file; skipping.", tools_used=[], processing_time=0.0, additional_data={"skipped": True}
				)
			
			# ROOSEVELT'S CLARIFICATION CONTEXT: Check if responding to previous clarification request
			previous_clarification = shared_memory.get("pending_outline_clarification")
			clarification_context = ""
			if previous_clarification:
				logger.info("📝 OUTLINE AGENT: Detected previous clarification request, including context")
				clarification_context = (
					"\n\n=== PREVIOUS CLARIFICATION REQUEST ===\n"
					f"Context: {previous_clarification.get('context', '')}\n"
					f"Questions Asked:\n"
				)
				for i, q in enumerate(previous_clarification.get('questions', []), 1):
					clarification_context += f"{i}. {q}\n"
				clarification_context += "\nThe user's response is in their latest message. Use this context to proceed with the outline development.\n"

			# Resolve base directory for direct reference loading
			try:
				canonical = active_editor.get("canonical_path")
				if canonical:
					base_dir = Path(canonical).resolve().parent
				else:
					base_dir = (Path(settings.UPLOAD_DIR)).resolve()
			except Exception:
				base_dir = (Path(settings.UPLOAD_DIR)).resolve()

			# Directly load references from THIS outline's frontmatter (no cascading)
			loader = FileContextLoader()
			style_text = None
			rules_text = None
			character_bodies: List[str] = []

			def _resolve_read_strip(ref: str) -> Optional[str]:
				try:
					p = loader._resolve_path(base_dir, ref)
					if p:
						data = Path(p).read_text(encoding="utf-8")
						return _strip_frontmatter_block(data)
				except Exception:
					return None
				return None

			# Style (only 'style')
			if frontmatter.get("style"):
				style_text = _resolve_read_strip(str(frontmatter.get("style")))
			# Rules (only 'rules')
			if frontmatter.get("rules"):
				rules_text = _resolve_read_strip(str(frontmatter.get("rules")))
			# Characters list and per-key
			# Prefer multiple entries via character_<name>: path
			for k, v in list(frontmatter.items()):
				kl = str(k).lower().strip()
				if kl.startswith("character_") and v:
					body = _resolve_read_strip(str(v))
					if body:
						character_bodies.append(body)
			# Optionally support YAML list under 'characters' (not comma-separated)
			chars_list = frontmatter.get("characters")
			if isinstance(chars_list, list):
				for ref in chars_list:
					try:
						ref_str = str(ref).strip()
						if ref_str:
							body = _resolve_read_strip(ref_str)
							if body:
								character_bodies.append(body)
					except Exception:
						pass

			# Full-file context; selection guides surgical ops
			body_only = _strip_frontmatter_block(text)
			from utils.chapter_scope import paragraph_bounds
			para_start, para_end = paragraph_bounds(text, cursor_offset if cursor_offset >= 0 else 0)
			if selection_start >= 0 and selection_end > selection_start:
				para_start, para_end = selection_start, selection_end

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
						"=== OUTLINE CONTEXT ===\n"
						f"File: {filename}\n\n"
						"Current Outline (frontmatter stripped):\n" + body_only + "\n\n"
						+ ("=== RULES ===\n" + rules_text + "\n\n" if rules_text else "")
						+ ("=== STYLE GUIDE ===\n" + style_text + "\n\n" if style_text else "")
						+ ("".join(["=== CHARACTER DOC ===\n" + b + "\n\n" for b in character_bodies]) if character_bodies else "")
						+ clarification_context
						+ ("REVISION MODE: Apply minimal targeted edits; prefer bullet/paragraph-level replace_range ops.\n\n" if current_request and any(k in current_request.lower() for k in ["revise", "revision", "tweak", "adjust", "polish", "tighten"]) else "")
						+ "Provide a ManuscriptEdit JSON plan strictly within scope (or OutlineClarificationRequest if you need more information)."
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

			chat_service = await self._get_chat_service()
			model_name = await self._get_model_name()
			start_time = datetime.now()
			response = await chat_service.openai_client.chat.completions.create(
				model=model_name,
				messages=messages,
				temperature=0.3,
			)

			content = response.choices[0].message.content or "{}"
			content = _unwrap_json_response(content)
			
			# ROOSEVELT'S "BEST EFFORT" RESPONSE: Try parsing as clarification request first, but this should be RARE
			clarification_request: Optional[OutlineClarificationRequest] = None
			structured: Optional[ManuscriptEdit] = None
			
			try:
				# First, check if this is a rare clarification-only request
				raw = json.loads(content)
				if isinstance(raw, dict) and raw.get("clarification_needed") is True:
					# Validate this is truly necessary (confidence should be very low)
					confidence = raw.get("confidence_without_clarification", 0.5)
					if confidence < 0.4:
						clarification_request = OutlineClarificationRequest(**raw)
						logger.warning(f"🤔 OUTLINE AGENT: Requesting clarification (confidence={confidence:.2f}) - RARE PATH")
					else:
						logger.info(f"⚠️ OUTLINE AGENT: Agent wanted clarification but confidence={confidence:.2f} is too high - expecting content instead")
			except Exception:
				pass
			
			# If it's a genuine clarification request (confidence < 0.4), return early
			if clarification_request:
				processing_time = (datetime.now() - start_time).total_seconds()
				questions_formatted = "\n".join([f"{i+1}. {q}" for i, q in enumerate(clarification_request.questions)])
				response_text = (
					f"**⚠️ Critical Ambiguity Detected**\n\n"
					f"{clarification_request.context}\n\n"
					f"I cannot proceed without clarification on:\n\n"
					f"{questions_formatted}\n\n"
					f"(Confidence without clarification: {clarification_request.confidence_without_clarification:.0%})"
				)
				if clarification_request.suggested_direction:
					response_text += f"\n\n**Suggestion:** {clarification_request.suggested_direction}\n"
				
				# Store clarification request in shared_memory for next turn
				shared_memory_out = shared_memory.copy()
				shared_memory_out["pending_outline_clarification"] = clarification_request.model_dump()
				
				logger.info(f"🛑 OUTLINE AGENT: Blocking on critical clarification - no content generated")
				return self._create_success_result(
					response=response_text,
					tools_used=[],
					processing_time=processing_time,
					additional_data={
						"clarification_request": clarification_request.model_dump(),
						"requires_user_input": True,
						"task_status": "incomplete",
						"shared_memory": shared_memory_out
					},
				)
			
			# Otherwise, parse as ManuscriptEdit
			try:
				raw = json.loads(content)
				if isinstance(raw, dict) and isinstance(raw.get("operations"), list):
					raw.setdefault("target_filename", filename)
					raw.setdefault("scope", "paragraph")
					raw.setdefault("summary", "Planned outline edit generated from context.")
					ops: List[Dict[str, Any]] = []
					for op in raw["operations"]:
						if not isinstance(op, dict):
							continue
						start_ix = int(op.get("start", para_start))
						end_ix = int(op.get("end", para_end))
						# Optional anchor hints from model for robust targeting in outline
						try:
							orig_text = str(op.get("original") or op.get("original_text") or "")
							left_ctx = str(op.get("left_context") or "")
							right_ctx = str(op.get("right_context") or "")
							resolved = False
							if orig_text:
								idx = text.find(orig_text)
								if idx != -1:
									start_ix = idx
									end_ix = idx + len(orig_text)
									resolved = True
							elif left_ctx and right_ctx:
								import re as _re
								m = _re.search(_re.escape(left_ctx) + r"([\s\S]{0,400}?)" + _re.escape(right_ctx), text)
								if m:
									start_ix = m.start(1)
									end_ix = m.end(1)
									resolved = True
						except Exception:
							pass
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
					response="Failed to produce a valid Outline edit plan. Ensure ONLY raw JSON ManuscriptEdit with operations is returned (no code fences or prose).",
					tools_used=[],
					processing_time=(datetime.now() - start_time).total_seconds(),
					additional_data={"raw": content}
				)

			# Determine desired chapter placement and adjust operation anchors robustly
			from utils.chapter_scope import find_chapter_ranges
			ranges = find_chapter_ranges(text)
			# Infer desired chapter number: chapter_index (0-based) > from request > from op text > append after last
			desired_ch_num: Optional[int] = None
			try:
				if getattr(structured, "chapter_index", None) is not None:
					ci = int(structured.chapter_index)
					if ci >= 0:
						desired_ch_num = ci + 1
			except Exception:
				pass
			if desired_ch_num is None and current_request:
				m = re.search(r"chapter\s+(\d+)", current_request, flags=re.IGNORECASE)
				if m:
					try:
						desired_ch_num = int(m.group(1))
					except Exception:
						pass
			# As last hint, look inside first op text for a heading
			if desired_ch_num is None and structured.operations:
				try:
					first_text = getattr(structured.operations[0], "text", "") or ""
					m2 = re.search(r"^##\s+Chapter\s+(\d+)\b", first_text, flags=re.IGNORECASE | re.MULTILINE)
					if m2:
						desired_ch_num = int(m2.group(1))
				except Exception:
					pass
			# Default: append after last existing chapter
			if desired_ch_num is None:
				try:
					max_num = max([r.chapter_number for r in ranges if r.chapter_number is not None], default=0)
					desired_ch_num = (max_num or 0) + 1
				except Exception:
					desired_ch_num = 1

			# Compute insertion window (chapter-aware); if not a chapter request, anchor to outline meta sections
			insert_start = None
			insert_end = None
			# If chapter exists, replace its entire block
			for r in ranges:
				if r.chapter_number == desired_ch_num:
					insert_start, insert_end = r.start, r.end
					break
			# Else insert after the previous chapter by number
			if insert_start is None:
				prev = None
				for r in ranges:
					if r.chapter_number is not None and r.chapter_number < desired_ch_num:
						if (prev is None) or (r.chapter_number > prev.chapter_number):
							prev = r
				if prev is not None:
					insert_start = prev.end
					insert_end = prev.end
			# Else if no chapters, insert right after '# Outline' heading if present
			if insert_start is None:
				m_outline = re.search(r"^#\s+Outline\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
				if m_outline:
					# Insert after the heading line and any following blank line
					line_end = text.find("\n", m_outline.end())
					if line_end == -1:
						line_end = m_outline.end()
					insert_start = line_end + 1
					insert_end = insert_start
				else:
					insert_start = len(text)
					insert_end = insert_start

			# If the request is clearly about meta sections (not chapters), re-anchor within that section
			if current_request and insert_start is not None and insert_end is not None:
				try:
					lr = current_request.lower()
					def _section_bounds(hx: str):
						m = re.search(hx, text, flags=re.IGNORECASE | re.MULTILINE)
						if not m:
							return None
						ss = m.end() + 1
						m2 = re.search(r"^#\s+.+$|^##\s+Chapter\s+\d+\b", text[ss:], flags=re.MULTILINE)
						se = ss + (m2.start() if m2 else len(text) - ss)
						return ss, se
					if 'overall synopsis' in lr:
						b = _section_bounds(r"^#\s+Overall\s+Synopsis\s*$")
						if b:
							insert_start, insert_end = b
					elif 'notes' in lr:
						b = _section_bounds(r"^#\s+Notes\s*$")
						if b:
							insert_start, insert_end = b
					elif 'characters' in lr:
						b = _section_bounds(r"^#\s+Characters\s*$")
						if b:
							insert_start, insert_end = b
				except Exception:
					pass

			lower_req = (current_request or "").lower()
			revision_mode = any(k in lower_req for k in ["revise", "revision", "tweak", "adjust", "polish", "tighten", "edit only"]) if lower_req else False

			# Calculate frontmatter end
			try:
				body_only_text = _strip_frontmatter_block(text)
				fm_end_idx = len(text) - len(body_only_text)
			except Exception:
				fm_end_idx = 0
			
			ed_ops: List[EditorOperation] = []
			for op in structured.operations:
				# Sanitize op text
				try:
					if isinstance(op.text, str):
						op.text = _strip_frontmatter_block(op.text)
						# Strip placeholder filler lines
						try:
							op.text = re.sub(r"^\s*-\s*\[To be.*?\]|^\s*\[To be.*?\]|^\s*TBD\s*$", "", op.text, flags=re.IGNORECASE | re.MULTILINE)
							op.text = re.sub(r"\n{3,}", "\n\n", op.text)
						except Exception:
							pass
				except Exception:
					pass
				
				# Build selection dict for resolver
				selection = {"start": selection_start, "end": selection_end} if selection_start >= 0 and selection_end >= 0 else None
				
				# Use progressive search resolver
				resolved_start, resolved_end, resolved_text, resolved_confidence = resolve_operation(
					text,
					{
						"original": getattr(op, "original", None),
						"original_text": getattr(op, "original_text", None),
						"anchor_text": getattr(op, "anchor_text", None),
						"left_context": getattr(op, "left_context", None),
						"right_context": getattr(op, "right_context", None),
						"occurrence_index": getattr(op, "occurrence_index", 0),
						"text": op.text,
						"op_type": op.op_type,
					},
					selection=selection,
					heading_hint=None,  # Outline agent relies on explicit anchors or whole-doc context
					frontmatter_end=fm_end_idx,
					require_anchors=True,
				)
				op.start = resolved_start
				op.end = resolved_end
				op.text = resolved_text
				op.confidence = resolved_confidence
				logger.info(f"📍 Outline operation resolved: [{resolved_start}:{resolved_end}] confidence={resolved_confidence:.2f}")
				
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
			
			# Clear any pending clarification since we're completing successfully
			shared_memory_out = shared_memory.copy()
			shared_memory_out.pop("pending_outline_clarification", None)
			additional["shared_memory"] = shared_memory_out
			
			return self._create_success_result(
				response=response_text,
				tools_used=[],
				processing_time=processing_time,
				additional_data=additional,
			)
		except Exception as e:
			logger.error(f"❌ OutlineEditingAgent failed: {e}")
			return self._create_success_result(
				response="Outline agent encountered an error.", tools_used=[], processing_time=0.0
			)
