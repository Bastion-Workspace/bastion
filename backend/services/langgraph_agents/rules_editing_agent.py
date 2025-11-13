"""
Rules Editing / Generation Agent

Gated to rules documents. Consumes active editor buffer, cursor/selection, and
referenced style/characters from frontmatter. Produces EditorOperations suitable
for Prefer Editor HITL application.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent
from models.agent_response_models import ManuscriptEdit, EditorOperation
from services.file_context_loader import FileContextLoader
from utils.editor_operations_resolver import resolve_operation

logger = logging.getLogger(__name__)


def _simple_hash(text: str) -> str:
	"""Match frontend sliceHash: 32-bit rolling hash to hex string."""
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

def _frontmatter_end_index(text: str) -> int:
	"""Return end index of a leading YAML frontmatter block if present, else 0."""
	try:
		import re
		m = re.match(r'^(---\s*\n[\s\S]*?\n---\s*\n)', text, flags=re.MULTILINE)
		if m:
			return m.end()
		return 0
	except Exception:
		return 0


class RulesEditingAgent(BaseAgent):
	def __init__(self):
		super().__init__("rules_editing_agent")
		logger.info("📜 BULLY! Rules Editing Agent mounted and ready to lay down the big stick!")

	def _build_system_prompt(self) -> str:
		return (
			"You are a MASTER UNIVERSE ARCHITECT for RULES documents (worldbuilding, series continuity). "
			"Persona disabled. Adhere strictly to frontmatter, project Rules, and Style.\n\n"
			"STRUCTURED OUTPUT REQUIRED: Return ONLY raw JSON (no prose, no markdown, no code fences) matching this schema:\n"
			"{\n"
			"  \"type\": \"ManuscriptEdit\",\n"
			"  \"target_filename\": string,\n"
			"  \"scope\": one of [\"paragraph\", \"chapter\", \"multi_chapter\"],\n"
			"  \"summary\": string,\n"
			"  \"chapter_index\": integer|null,\n"
			"  \"safety\": one of [\"low\", \"medium\", \"high\"],\n"
			"  \"operations\": [ { \"op_type\": one of [\"replace_range\", \"delete_range\", \"insert_after_heading\"], \"start\": integer, \"end\": integer, \"text\": string } ]\n"
			"}\n\n"
			"OUTPUT RULES:\n"
			"- Output MUST be a single JSON object only.\n"
			"- Do NOT include triple backticks or language tags.\n"
			"- Do NOT include explanatory text before or after the JSON.\n\n"
			"FORMATTING CONTRACT (RULES DOCUMENTS):\n"
			"- Never emit YAML frontmatter in operations[].text. Preserve existing frontmatter as-is.\n"
			"- Use Markdown headings and lists for the body.\n"
			"- When creating or normalizing structure, prefer this scaffold (top-level headings):\n"
			"  ## Background\n"
			"  ## Universe Constraints (physical/magical/technological laws)\n"
			"  ## Systems\n"
			"  ### Magic or Technology Systems\n"
			"  ### Resource & Economy Constraints\n"
			"  ## Social Structures & Culture\n"
			"  ### Institutions & Power Dynamics\n"
			"  ## Geography & Environment\n"
			"  ## Religion & Philosophy\n"
			"  ## Timeline & Continuity\n"
			"  ### Chronology (canonical)\n"
			"  ### Continuity Rules (no-retcon constraints)\n"
			"  ## Series Synopsis\n"
			"  ### Book 1\n"
			"  ### Book 2\n"
			"  ... (as needed)\n"
			"  ## Character References\n"
			"  ### Cast Integration & Constraints\n"
			"  ## Change Log (Rule Evolution)\n\n"
			"RULES FOR EDITS:\n"
			"1) Make focused, surgical edits near the cursor/selection unless the user requests re-organization.\n"
			"2) Maintain the scaffold above; if missing, create only the minimal sections the user asked for.\n"
			"3) Prefer paragraph/sentence-level replacements; avoid large-span rewrites unless asked.\n"
			"4) Enforce consistency: cross-check constraints against Series Synopsis and Characters.\n\n"
			"ANCHOR REQUIREMENTS (CRITICAL):\n"
			"For EVERY operation, you MUST provide precise anchors:\n\n"
			"REVISE/DELETE Operations:\n"
			"- ALWAYS include 'original_text' with EXACT, VERBATIM text from the file\n"
			"- Minimum 10-20 words, include complete sentences with natural boundaries\n"
			"- Copy and paste directly - do NOT retype or modify\n"
			"- ⚠️ NEVER include header lines (###, ##, #) in original_text!\n"
			"- OR provide both 'left_context' and 'right_context' (exact surrounding text)\n\n"
			"INSERT Operations (PREFERRED for adding content below headers!):\n"
			"- **PRIMARY METHOD**: Use op_type='insert_after_heading' with anchor_text='## Section' when adding content below ANY header\n"
			"- Provide 'anchor_text' with EXACT, COMPLETE header line to insert after (verbatim from file)\n"
			"- This is the SAFEST method - it NEVER deletes headers, always inserts AFTER them\n"
			"- Use this for adding rules, constraints, systems, or any worldbuilding content below headers\n"
			"- ALTERNATIVE: Provide 'original_text' with text to insert after\n"
			"- FALLBACK: Provide 'left_context' with text before insertion point (minimum 10-20 words)\n\n"
			"Additional Options:\n"
			"- 'occurrence_index' if text appears multiple times (0-based, default 0)\n"
			"- Start/end indices are approximate; anchors take precedence\n\n"
			"=== DECISION TREE FOR OPERATION TYPE ===\n"
			"1. Section is COMPLETELY EMPTY? → insert_after_heading with anchor_text=\"## Section\"\n"
			"2. Section has PLACEHOLDER or existing rules to replace? → replace_range (NO headers in original_text!)\n"
			"3. Deleting SPECIFIC content? → delete_range with original_text (NO headers!)\n\n"
			"⚠️ CRITICAL: When replacing placeholder content, use 'replace_range' on ONLY the placeholder!\n"
			"⚠️ NEVER include headers in 'original_text' for replace_range - headers will be deleted!\n"
			"✅ Correct: {\"op_type\": \"replace_range\", \"original_text\": \"[Placeholder rules]\", \"text\": \"- Rule 1\\n- Rule 2\"}\n"
			"❌ Wrong: {\"op_type\": \"insert_after_heading\"} when placeholder text exists - it will keep the placeholder!\n\n"
			"=== SPACING RULES (CRITICAL - READ CAREFULLY!) ===\n"
			"YOUR TEXT MUST END IMMEDIATELY AFTER THE LAST CHARACTER!\n\n"
			'✅ CORRECT: "- Rule 1\\n- Rule 2\\n- Rule 3"  ← Ends after "3" with NO \\n\n'
			'❌ WRONG: "- Rule 1\\n- Rule 2\\n"  ← Extra \\n after last line creates blank line!\n'
			'❌ WRONG: "- Rule 1\\n- Rule 2\\n\\n"  ← \\n\\n creates 2 blank lines!\n'
			'❌ WRONG: "- Rule 1\\n\\n- Rule 2"  ← Double \\n\\n between items creates blank line!\n\n'
			"IRON-CLAD RULE: After last line = ZERO \\n (nothing!)\n"
			"5) Headings must be clear; do not duplicate sections. If an equivalent heading exists, update it in place.\n"
			"6) When adding Timeline & Continuity entries, keep a chronological order and explicit constraints (MUST/MUST NOT).\n"
			"7) When adding Series Synopsis entries, keep book-by-book bullets with continuity notes.\n"
			"8) NO PLACEHOLDER FILLERS: If a requested section has no content yet, create the heading only and leave the body blank. Do NOT insert placeholders like '[To be developed]' or 'TBD'.\n"
		)

	async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
		try:
			shared_memory = state.get("shared_memory", {}) or {}
			active_editor = shared_memory.get("active_editor", {}) or {}

			buffer_text = active_editor.get("content", "") or ""
			normalized_text = buffer_text.replace("\r\n", "\n")
			filename = active_editor.get("filename") or "rules.md"
			frontmatter = active_editor.get("frontmatter", {}) or {}
			cursor_offset = int(active_editor.get("cursor_offset", -1))
			selection_start = int(active_editor.get("selection_start", -1))
			selection_end = int(active_editor.get("selection_end", -1))

			try:
				logger.info("🧭 RULES AGENT FRONTMATTER: %s", json.dumps(frontmatter) if isinstance(frontmatter, dict) else str(frontmatter))
			except Exception:
				pass

			# STRICT GATE: require explicit frontmatter.type == 'rules' (no fallbacks)
			fm_type = ""
			if isinstance(frontmatter, dict):
				fm_type = str(frontmatter.get("type") or "").strip().lower()
			if fm_type != "rules":
				logger.info("📜 Rules Agent Gate: Detected type='%s' (expected 'rules'); skipping.", fm_type)
				return self._create_success_result(
					response="Active editor is not a Rules document; rules agent skipping.",
					tools_used=[],
					processing_time=0.0,
					additional_data={"skipped": True},
				)

			# Resolve referenced context: style + character docs from frontmatter
			if active_editor.get("canonical_path"):
				frontmatter = { **frontmatter, "__canonical_path__": active_editor.get("canonical_path") }
			loader = FileContextLoader()
			loaded = loader.load_referenced_context(filename, frontmatter)

			style_guide_text = _strip_frontmatter_block(loaded.style.content) if loaded.style else None

			# Character references: support keys like characters: a.md, b.md and any key starting with character/ref_character
			character_paths: List[str] = []
			try:
				for k, v in (frontmatter or {}).items():
					kl = str(k).lower().strip()
					if kl == "characters" and v:
						character_paths.extend([p.strip() for p in str(v).split(',') if p.strip()])
					elif kl.startswith("character") or kl.startswith("ref_character"):
						if v:
							character_paths.append(str(v).strip())
			except Exception:
				character_paths = []

			character_texts: List[str] = []
			if character_paths:
				base_dir = Path(active_editor.get("canonical_path") or filename).parent
				project_root = Path(loader.project_root)
				for p in character_paths:
					ref = Path(p)
					candidate = (base_dir / (ref if ref.suffix else Path(str(ref) + ".md"))).resolve()
					try:
						candidate.relative_to(project_root)
					except Exception:
						continue
					if candidate.exists() and candidate.is_file():
						try:
							text = candidate.read_text(encoding="utf-8")
							character_texts.append(_strip_frontmatter_block(text))
						except Exception:
							pass

			# Build LLM messages
			system_prompt = self._build_system_prompt()
			try:
				current_request = (self._extract_current_user_query(state) or "").strip()
			except Exception:
				current_request = ""

			fm_end_idx = _frontmatter_end_index(normalized_text)
			body_only = _strip_frontmatter_block(normalized_text)

			# Scope: prefer selection, else paragraph around cursor
			from utils.chapter_scope import paragraph_bounds
			para_start, para_end = paragraph_bounds(normalized_text, cursor_offset if cursor_offset >= 0 else 0)
			if selection_start >= 0 and selection_end > selection_start:
				para_start, para_end = selection_start, selection_end

			messages = [
				{"role": "system", "content": system_prompt},
				{"role": "system", "content": f"Current Date/Time: {datetime.now().isoformat()}"},
				{
					"role": "user",
					"content": (
						"=== RULES CONTEXT ===\n"
						f"File: {filename}\n\n"
						"Current Buffer (frontmatter stripped):\n" + body_only + "\n\n"
						+ ("=== STYLE GUIDE ===\n" + style_guide_text + "\n\n" if style_guide_text else "")
						+ ("".join(["=== CHARACTER DOC ===\n" + t + "\n\n" for t in character_texts]) if character_texts else "")
						+ ("REVISION MODE: Apply minimal targeted edits per the user's rubric. Prefer paragraph-level replace_range ops.\n\n" if current_request and any(k in current_request.lower() for k in ["revise", "revision", "tweak", "adjust", "polish", "tighten"]) else "")
						+ "Maintain the FORMATTING CONTRACT sections where present; if creating new content, adhere to the scaffold. Provide a ManuscriptEdit JSON plan strictly within scope."
					),
				},
			]
			if current_request:
				messages.append({
					"role": "user",
					"content": (
						f"USER REQUEST: {current_request}\n\n"
						"ANCHORING: For replace/delete, include 'original_text' or both 'left_context' and 'right_context' (<=60 chars each)."
					)
				})

			# Call model
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
			structured: Optional[ManuscriptEdit] = None
			parse_error = None
			try:
				raw = json.loads(content)
				if isinstance(raw, dict) and isinstance(raw.get("operations"), list):
					# Normalize to ManuscriptEdit shape
					raw.setdefault("target_filename", filename)
					raw.setdefault("scope", "paragraph")
					raw.setdefault("summary", "Planned rules edit generated from context.")
					# Bounds-safe operations and pre_hash placeholder with optional anchor resolution
					norm_ops: List[Dict[str, Any]] = []
					for op in raw["operations"]:
						if not isinstance(op, dict):
							continue
						# Optional anchors from the model (backward compatible)
						orig_text = str(op.get("original") or op.get("original_text") or "")
						left_ctx = str(op.get("left_context") or "")
						right_ctx = str(op.get("right_context") or "")

						# Start/end from LLM are relative to body_only; convert to absolute buffer indices by default
						start_ix = int(op.get("start", para_start)) + fm_end_idx
						end_ix = int(op.get("end", para_end)) + fm_end_idx

						# If anchors provided, try to resolve a more reliable span in the full normalized_text
						resolved = False
						if orig_text:
							idx = normalized_text.find(orig_text)
							if idx != -1:
								start_ix = idx
								end_ix = idx + len(orig_text)
								resolved = True
						elif left_ctx and right_ctx:
							try:
								pattern = re.escape(left_ctx) + r"([\s\S]{0,200}?)" + re.escape(right_ctx)
								m = re.search(pattern, normalized_text)
								if m:
									start_ix = m.start(1)
									end_ix = m.end(1)
									resolved = True
							except Exception:
								pass

					# Build normalized op (with optional robust anchors)
						op_type = op.get("op_type")
						if op_type not in ("replace_range", "delete_range", "insert_after_heading"):
							op_type = "replace_range"
					# Optional anchor hints from model
					try:
						orig_text = str(op.get("original") or op.get("original_text") or orig_text)
						left_ctx = str(op.get("left_context") or left_ctx)
						right_ctx = str(op.get("right_context") or right_ctx)
						resolved2 = resolved
						if (not resolved2) and orig_text:
							idx2 = normalized_text.find(orig_text)
							if idx2 != -1:
								start_ix = idx2
								end_ix = idx2 + len(orig_text)
								resolved2 = True
						elif (not resolved2) and left_ctx and right_ctx:
							m2 = re.search(re.escape(left_ctx) + r"([\s\S]{0,400}?)" + re.escape(right_ctx), normalized_text)
							if m2:
								start_ix = m2.start(1)
								end_ix = m2.end(1)
								resolved2 = True
					except Exception:
						pass
					start_ix = max(0, min(len(normalized_text), start_ix))
					end_ix = max(start_ix, min(len(normalized_text), end_ix))
					insert_text = op.get("text", "")
					# Ensure separation when inserting new sections
					if start_ix == end_ix:
						left_tail = normalized_text[max(0, start_ix-2):start_ix]
						if left_tail.endswith("\n\n"):
							needed_prefix = ""
						elif left_tail.endswith("\n"):
							needed_prefix = "\n"
						else:
							needed_prefix = "\n\n"
						try:
							leading_stripped = re.sub(r'^\n+', '', insert_text)
							insert_text = f"{needed_prefix}{leading_stripped}"
						except Exception:
							insert_text = f"{needed_prefix}{insert_text}"
					norm_ops.append({
						"op_type": op_type,
						"start": start_ix,
						"end": end_ix,
						"text": insert_text,
						"pre_hash": "",
						"__orig": orig_text,
						"__lctx": left_ctx,
						"__rctx": right_ctx,
						"__resolved": resolved,
					})
					raw["operations"] = norm_ops
					structured = ManuscriptEdit(**{k: v for k, v in raw.items() if k != "operations"}, operations=[
						EditorOperation(**{k: v for k, v in op.items() if not k.startswith("__")}) for op in norm_ops
					])
				else:
					structured = ManuscriptEdit.parse_raw(content)
			except Exception as e:
				parse_error = e
			if structured is None:
				logger.error(f"❌ Failed to parse Rules ManuscriptEdit: {parse_error}")
				return self._create_success_result(
					response="Failed to produce a valid Rules edit plan. Ensure ONLY raw JSON ManuscriptEdit with operations is returned (no code fences or prose).",
					tools_used=[],
					processing_time=(datetime.now() - start_time).total_seconds(),
					additional_data={"raw": content},
				)

			# If the user's request does not express edit intent, suppress operations (answer analytically only)
			edit_intent = False
			try:
				lr = (current_request or "").lower()
				edit_intent = any(k in lr for k in [
					"edit", "change", "replace", "update", "revise", "rewrite",
					"insert", "add ", "delete", "adjust", "fix ", "fix:", "modify"
				])
			except Exception:
				edit_intent = False
			if not edit_intent:
				structured.operations = []

			# Inject pre_hash and clamp to selection/paragraph when revision-like
			revision_mode = False
			try:
				lr = (current_request or "").lower()
				revision_mode = any(k in lr for k in ["revise", "revision", "tweak", "adjust", "polish", "tighten", "edit only"])
			except Exception:
				revision_mode = False

			# Section anchoring helpers for targeted placement inside Rules scaffold
			def _find_section_bounds(doc: str, heading_regex: str) -> (int, int):
				try:
					pat = re.compile(heading_regex, re.IGNORECASE | re.MULTILINE)
					m = pat.search(doc)
					if not m:
						return -1, -1
					sec_start = m.end() + 1
					m2 = re.search(r"^##\s+.+$", doc[sec_start:], re.MULTILINE)
					sec_end = sec_start + (m2.start() if m2 else len(doc) - sec_start)
					return sec_start, sec_end
				except Exception:
					return -1, -1

			def _anchor_heading_section(doc: str, proposed_text: str) -> Optional[tuple]:
				# If proposal starts with a heading (## or ###), target that entire section
				try:
					m = re.match(r"^(#{2,6})\s+(.+?)\s*$", proposed_text.strip().split('\n', 1)[0])
					if not m:
						return None
					hevel = len(m.group(1))
					title = m.group(2).strip()
					pattern = rf"^{'#'*hevel}\s+{re.escape(title)}\s*$"
					mh = re.search(pattern, doc, re.MULTILINE | re.IGNORECASE)
					if not mh:
						return None
					sec_start = mh.start()
					next_pat = rf"^#{{1,{hevel}}}\s+.+$"
					mn = re.search(next_pat, doc[mh.end():], re.MULTILINE)
					sec_end = mh.end() + (mn.start() if mn else len(doc) - mh.end())
					return (sec_start, sec_end, proposed_text)
				except Exception:
					return None

			# Infer target section from request keywords
			anchor_start, anchor_end = -1, -1
			try:
				lr = (current_request or "").lower()
				section_map = [
					("background", r"^##\s+Background\s*$"),
					("universe", r"^##\s+Universe\s+Constraints[\s\S]*$"),
					("systems", r"^##\s+Systems\s*$"),
					("timeline", r"^##\s+Timeline\s*&?\s*Continuity\s*$"),
					("chronology", r"^###\s+Chronology\s*\(canonical\)\s*$"),
					("series", r"^##\s+Series\s+Synopsis\s*$"),
					("characters", r"^##\s+Character\s+References\s*$"),
					("changelog", r"^##\s+Change\s+Log\s*\(Rule\s+Evolution\)\s*$"),
				]
				for key, rx in section_map:
					if key in lr:
						anchor_start, anchor_end = _find_section_bounds(normalized_text, rx)
						if anchor_start != -1:
							break
			except Exception:
				anchor_start, anchor_end = -1, -1

			ed_ops: List[EditorOperation] = []
			for op in structured.operations:
				start = max(0, min(len(normalized_text), op.start))
				end = max(0, min(len(normalized_text), op.end))
				# If the op text begins with a heading that exists, anchor to that full section
				if isinstance(getattr(op, "text", None), str):
					anch = _anchor_heading_section(normalized_text, op.text)
					if anch:
						start, end, op.text = anch
				# If not in revision mode and we detected a target section, anchor to that section window
				if not revision_mode and anchor_start != -1:
					start = anchor_start
					end = anchor_end
				# Protect YAML frontmatter: compute its end once
				fm_end_idx = _frontmatter_end_index(normalized_text)
				if start < fm_end_idx:
					if op.op_type == "delete_range":
						# Skip deletions in frontmatter
						continue
					if end <= fm_end_idx:
						# Fully within frontmatter -> convert to insert at body start
						start = fm_end_idx
						end = fm_end_idx
					else:
						# Overlap -> clamp to body start
						start = fm_end_idx
				if revision_mode and op.op_type != "delete_range":
					# Prefer explicit selection; else clamp to paragraph around cursor
					if selection_start >= 0 and selection_end > selection_start:
						start = max(selection_start, start)
						end = min(selection_end, end)
					else:
						start = max(para_start, start)
						end = min(para_end, end)
					start = max(0, min(len(normalized_text), start))
					end = max(start, min(len(normalized_text), end))

				# Guardrail: drop obviously spurious micro-edits unless anchored to age/years context
				span_len = max(0, end - start)
				proposed_text = (getattr(op, "text", "") or "")
				if span_len <= 2 and len(proposed_text) <= 4:
					window_start = max(0, start - 64)
					window_end = min(len(normalized_text), end + 64)
					window = normalized_text[window_start:window_end].lower()
					age_ctx = any(token in window for token in [
						"year-old", "years old", " yo", " age ", "aged ", " age:", " old)"
					])
					if not age_ctx:
						# Skip micro edit; not contextually safe
						continue

					# Always strip any YAML frontmatter from inserted text; strip placeholder filler; ensure trailing spacing on insert
				try:
					if isinstance(op.text, str):
						op.text = _strip_frontmatter_block(op.text)
						try:
							op.text = re.sub(r"^\s*-\s*\[To be.*?\]|^\s*\[To be.*?\]|^\s*TBD\s*$", "", op.text, flags=re.IGNORECASE | re.MULTILINE)
							op.text = re.sub(r"\n{3,}", "\n\n", op.text)
						except Exception:
							pass
						if start == end:
							right_head = normalized_text[start:start+2]
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
				except Exception:
					pass
				pre_slice = normalized_text[start:end]
				op.pre_hash = _simple_hash(pre_slice)
				ed_ops.append(op)
			structured.operations = ed_ops

			processing_time = (datetime.now() - start_time).total_seconds()
			preview = "\n\n".join([(getattr(op, "text", "") or "").strip() for op in structured.operations if (getattr(op, "text", "") or "").strip()])
			response_text = preview if preview else (structured.summary or "Edit plan ready.")

			editor_ops = [op.model_dump() for op in structured.operations]
			additional: Dict[str, Any] = {"content_preview": response_text[:2000]}
			if editor_ops:
				additional["editor_operations"] = editor_ops
				additional["manuscript_edit"] = structured.model_dump()
			return self._create_success_result(
				response=response_text,
				tools_used=[],
				processing_time=processing_time,
				additional_data=additional,
			)
		except Exception as e:
			logger.error(f"❌ RulesEditingAgent failed: {e}")
			return self._create_success_result(
				response="Rules agent encountered an error.", tools_used=[], processing_time=0.0
			)
