"""
SysML Model Generation / Editing Agent

Gated to SysML documents. Generates standards-compliant SysML models in XMI format.
Eclipse Papyrus compatible. Consumes active editor content and produces structured model edits.
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


def _extract_xmi_from_markdown(text: str) -> str:
    """
    Extract XMI from markdown code fences if present.
    
    LLMs often wrap XMI in ```xml...``` fences even when told to put it in JSON.
    """
    if not text or not isinstance(text, str):
        return text
    
    import re
    
    # Try to extract from ```xml ... ``` fence
    xml_fence_match = re.search(r'```xml\s*\n([\s\S]*?)\n```', text, re.MULTILINE)
    if xml_fence_match:
        logger.info("🔧 EXTRACT: Found XMI in ```xml``` fence, extracting...")
        return xml_fence_match.group(1).strip()
    
    # Try to extract from generic ``` ... ``` fence
    generic_fence_match = re.search(r'```\s*\n([\s\S]*?)\n```', text, re.MULTILINE)
    if generic_fence_match:
        content = generic_fence_match.group(1).strip()
        # Only return if it looks like XML
        if content.startswith('<'):
            logger.info("🔧 EXTRACT: Found XMI in generic ``` fence, extracting...")
            return content
    
    # No fence found, return as-is
    return text.strip()


def _auto_fix_xmi(xmi_code: str) -> str:
    """
    Auto-fix common XMI issues that LLMs forget.
    
    Returns:
        Fixed XMI code
    """
    if not xmi_code or not isinstance(xmi_code, str):
        return xmi_code
    
    # First, extract from markdown fences if present
    code = _extract_xmi_from_markdown(xmi_code)
    
    # Auto-prepend XML declaration if missing
    # LLMs frequently forget this despite explicit instructions!
    if not code.startswith('<?xml'):
        logger.info("🔧 AUTO-FIX: Prepending XML declaration to XMI")
        code = '<?xml version="1.0" encoding="UTF-8"?>\n' + code
    
    return code


def _validate_xmi_syntax(xmi_code: str) -> tuple[bool, Optional[str]]:
    """
    Basic XMI/XML syntax validation for SysML models.
    
    Returns:
        (is_valid, error_message)
    """
    if not xmi_code or not isinstance(xmi_code, str):
        return False, "Empty or invalid XMI code"
    
    code = xmi_code.strip()
    
    # Check for XML declaration (should be present after auto-fix)
    if not code.startswith('<?xml'):
        return False, "Missing XML declaration - XMI must start with <?xml version=\"1.0\"?>"
    
    # Check for required XMI namespaces
    required_namespaces = ['xmlns:xmi=', 'xmlns:uml=', 'xmlns:SysML=']
    for ns in required_namespaces:
        if ns not in code:
            return False, f"Missing required namespace: {ns}"
    
    # Check for xmi:XMI root element
    if '<xmi:XMI' not in code:
        return False, "Missing <xmi:XMI> root element"
    
    if '</xmi:XMI>' not in code:
        return False, "Missing </xmi:XMI> closing tag"
    
    # Check for uml:Model
    if '<uml:Model' not in code:
        return False, "Missing <uml:Model> element - XMI must contain a UML model"
    
    # Warn about pathmap usage (not portable)
    if 'pathmap://' in code:
        logger.warning("⚠️ XMI contains pathmap:// references which are not portable - should use inline type definitions with href=\"#TypeName\"")
    
    # Check for inline primitive type definitions (recommended for portability)
    has_inline_types = 'xmi:type="uml:PrimitiveType"' in code and 'xmi:id="Integer"' in code
    if not has_inline_types and ('href=' in code):
        logger.info("💡 TIP: Consider defining primitive types inline for better portability")
    
    # Basic XML well-formedness checks
    try:
        import xml.etree.ElementTree as ET
        ET.fromstring(code)
    except ET.ParseError as e:
        return False, f"XML parsing error: {str(e)}"
    except Exception as e:
        return False, f"XML validation error: {str(e)}"
    
    # Check for balanced angle brackets (basic check)
    open_count = code.count('<')
    close_count = code.count('>')
    if open_count != close_count:
        return False, f"Unbalanced XML tags: {open_count} opening '<' vs {close_count} closing '>'"
    
    return True, None


class SysMLAgent(BaseAgent):
    def __init__(self):
        super().__init__("sysml_agent")
        logger.info("🏗️ BULLY! SysML Agent saddled and ready to design systems!")

    def _build_system_prompt(self) -> str:
        return (
            "You are a SYSTEMS ENGINEER and SysML expert. Generate standards-compliant SysML models in XMI format.\n"
            "Follow OMG SysML 1.6 specifications and Eclipse UML2 metamodel for Papyrus compatibility.\n\n"
            "STRUCTURED OUTPUT REQUIRED: You MUST return ONLY raw JSON (no prose, no markdown, no code fences) matching this schema:\n"
            "{\n"
            "  \"type\": \"ManuscriptEdit\",\n"
            "  \"target_filename\": string (REQUIRED),\n"
            "  \"scope\": one of [\"paragraph\", \"chapter\", \"multi_chapter\"] (REQUIRED - use 'paragraph' for single diagram, 'chapter' for full document),\n"
            "  \"summary\": string (REQUIRED - brief description of diagram changes),\n"
            "  \"chapter_index\": null,\n"
            "  \"safety\": one of [\"low\", \"medium\", \"high\"] (REQUIRED - usually 'low' for diagrams),\n"
            "  \"operations\": [\n"
            "    {\n"
            "      \"op_type\": one of [\"replace_range\", \"delete_range\", \"insert_after_heading\"] (REQUIRED),\n"
            "      \"start\": integer (REQUIRED - character offset),\n"
            "      \"end\": integer (REQUIRED - character offset),\n"
            "      \"text\": string (REQUIRED - XMI/UML model code),\n"
            "      \"original_text\": string (REQUIRED for replace_range/delete_range - EXACT text from document),\n"
            "      \"anchor_text\": string (REQUIRED for insert_after_heading - EXACT heading to insert after),\n"
            "      \"left_context\": string (optional),\n"
            "      \"right_context\": string (optional)\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "OUTPUT RULES:\n"
            "- Output MUST be a single JSON object only\n"
            "- Do NOT include triple backticks or language tags in the JSON\n"
            "- Do NOT include explanatory text before or after JSON\n"
            "- XMI/UML model code goes in the 'text' field of operations\n"
            "- XMI must be valid XML with proper namespaces\n"
            "- XMI must be compatible with Eclipse Papyrus\n"
            "- Wrap the XMI content in ```xml markdown code fence INSIDE the text field\n\n"
            "=== XMI FORMAT REQUIREMENTS ===\n\n"
            "Generate Eclipse UML2 XMI with these exact namespaces:\n"
            "```xml\n"
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<xmi:XMI xmlns:xmi=\"http://schema.omg.org/spec/XMI/2.1\"\n"
            "         xmlns:uml=\"http://www.eclipse.org/uml2/5.0.0/UML\"\n"
            "         xmlns:SysML=\"http://www.omg.org/spec/SysML/1.6/SysML\">\n"
            "  <uml:Model xmi:id=\"model_name\" name=\"Model Name\">\n"
            "    <!-- Your model content here -->\n"
            "  </uml:Model>\n"
            "</xmi:XMI>\n"
            "```\n\n"
            "=== SysML BUILDING BLOCKS ===\n\n"
            "**CRITICAL: Define Primitive Types Inline for Portability**\n"
            "ALWAYS include these type definitions at the start of your model (right after <uml:Model>):\n"
            "```xml\n"
            "<packagedElement xmi:type=\"uml:PrimitiveType\" xmi:id=\"Integer\" name=\"Integer\"/>\n"
            "<packagedElement xmi:type=\"uml:PrimitiveType\" xmi:id=\"Real\" name=\"Real\"/>\n"
            "<packagedElement xmi:type=\"uml:PrimitiveType\" xmi:id=\"String\" name=\"String\"/>\n"
            "<packagedElement xmi:type=\"uml:PrimitiveType\" xmi:id=\"Boolean\" name=\"Boolean\"/>\n"
            "```\n\n"
            "**Blocks:** Components as `<packagedElement xmi:type=\"uml:Class\">` with `<SysML:Block>` stereotype\n"
            "**Properties:** Use `<ownedAttribute>` with type href referencing inline types (e.g., `href=\"#Real\"`)\n"
            "**Units:** Add `<ownedComment><body>Unit: kg</body></ownedComment>` to properties\n"
            "**Composition:** Use `aggregation=\"composite\"` attribute for part-whole relationships\n"
            "**Requirements:** Class with `<SysML:Requirement>` stereotype\n"
            "**Packages:** Use `<packagedElement xmi:type=\"uml:Package\">` for subsystems\n\n"
            "**Key Pattern - Stereotype Application:**\n"
            "1. Define UML element with xmi:id (e.g., `<packagedElement xmi:id=\"console_block\" ...>`)\n"
            "2. Apply SysML stereotype referencing it (e.g., `<SysML:Block base_Class=\"console_block\"/>`)\n\n"
            "**Type References (Use Inline Types):**\n"
            "- Real: `<type xmi:type=\"uml:PrimitiveType\" href=\"#Real\"/>`\n"
            "- Integer: `<type xmi:type=\"uml:PrimitiveType\" href=\"#Integer\"/>`\n"
            "- Boolean: `<type xmi:type=\"uml:PrimitiveType\" href=\"#Boolean\"/>`\n"
            "- String: `<type xmi:type=\"uml:PrimitiveType\" href=\"#String\"/>`\n\n"
            "**ID Naming:** Use unique, descriptive snake_case IDs (e.g., `heart_block`, `pump_voltage_prop`)\n\n"
            "=== COMPLETE EXAMPLE ===\n\n"
            "```xml\n"
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<xmi:XMI xmi:version=\"2.1\"\n"
            "         xmlns:xmi=\"http://schema.omg.org/spec/XMI/2.1\"\n"
            "         xmlns:uml=\"http://www.eclipse.org/uml2/5.0.0/UML\"\n"
            "         xmlns:SysML=\"http://www.omg.org/spec/SysML/1.6/SysML\">\n"
            "  <uml:Model xmi:id=\"vehicle_model\" name=\"Vehicle System\">\n"
            "    \n"
            "    <!-- ALWAYS Define Primitive Types First -->\n"
            "    <packagedElement xmi:type=\"uml:PrimitiveType\" xmi:id=\"Integer\" name=\"Integer\"/>\n"
            "    <packagedElement xmi:type=\"uml:PrimitiveType\" xmi:id=\"Real\" name=\"Real\"/>\n"
            "    <packagedElement xmi:type=\"uml:PrimitiveType\" xmi:id=\"String\" name=\"String\"/>\n"
            "    <packagedElement xmi:type=\"uml:PrimitiveType\" xmi:id=\"Boolean\" name=\"Boolean\"/>\n"
            "    \n"
            "    <!-- System Blocks -->\n"
            "    <packagedElement xmi:type=\"uml:Class\" xmi:id=\"vehicle_block\" name=\"Vehicle\">\n"
            "      <ownedAttribute xmi:id=\"vehicle_mass\" name=\"mass\">\n"
            "        <type xmi:type=\"uml:PrimitiveType\" href=\"#Real\"/>\n"
            "        <ownedComment xmi:id=\"mass_unit\"><body>Unit: kg</body></ownedComment>\n"
            "      </ownedAttribute>\n"
            "      <ownedAttribute xmi:id=\"vehicle_engine\" name=\"engine\" type=\"engine_block\" aggregation=\"composite\"/>\n"
            "    </packagedElement>\n"
            "    <SysML:Block xmi:id=\"vehicle_stereotype\" base_Class=\"vehicle_block\"/>\n"
            "    \n"
            "    <packagedElement xmi:type=\"uml:Class\" xmi:id=\"engine_block\" name=\"Engine\">\n"
            "      <ownedAttribute xmi:id=\"engine_power\" name=\"power\">\n"
            "        <type xmi:type=\"uml:PrimitiveType\" href=\"#Real\"/>\n"
            "        <ownedComment xmi:id=\"power_unit\"><body>Unit: W</body></ownedComment>\n"
            "      </ownedAttribute>\n"
            "    </packagedElement>\n"
            "    <SysML:Block xmi:id=\"engine_stereotype\" base_Class=\"engine_block\"/>\n"
            "    \n"
            "  </uml:Model>\n"
            "</xmi:XMI>\n"
            "```\n\n"
            "=== OPERATIONS ===\n\n"
            "**For empty documents:** Use `insert_after_heading` with `anchor_text=\"---\"` (frontmatter end)\n"
            "**For adding content:** Use `insert_after_heading` with exact heading text as anchor\n"
            "**For modifying:** Use `replace_range` with exact `original_text` to replace\n"
            "**For deleting:** Use `delete_range` with exact `original_text`\n\n"
            "Wrap XMI in ```xml markdown fence in the `text` field.\n\n"
            "⚠️ CRITICAL REMINDERS:\n"
            "1. ALWAYS define primitive types (Integer, Real, String, Boolean) inline at start of model\n"
            "2. Use href=\"#TypeName\" to reference inline types, NOT pathmap:// URIs\n"
            "3. This ensures XMI works in ANY tool without plugins\n\n"
            "Remember: Generate portable, standards-compliant SysML in XMI format.\n"
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            shared_memory = state.get("shared_memory", {}) or {}
            active_editor = shared_memory.get("active_editor", {}) or {}

            document_content = active_editor.get("content", "")
            filename = active_editor.get("filename") or "system_design.md"
            frontmatter = active_editor.get("frontmatter", {}) or {}
            cursor_offset = int(active_editor.get("cursor_offset", -1))
            selection_start = int(active_editor.get("selection_start", -1))
            selection_end = int(active_editor.get("selection_end", -1))

            # Hard gate: require sysml type
            fm_type = str(frontmatter.get("type", "")).lower()
            if fm_type != "sysml":
                return self._create_success_result(
                    response="Active editor is not a SysML document; agent skipping.",
                    tools_used=[],
                    processing_time=0.0,
                    additional_data={"skipped": True},
                )

            # Extract current user query
            try:
                current_request = (self._extract_current_user_query(state) or "").strip()
            except Exception:
                current_request = ""

            # Build messages for LLM
            system_prompt = self._build_system_prompt()
            
            # Determine frontmatter boundaries
            fm_end_idx = _frontmatter_end_index(document_content)
            context_document = _strip_frontmatter_block(document_content)
            
            # Build selection context if user selected text
            selection_context = ""
            if selection_start >= 0 and selection_end > selection_start:
                selected_text = document_content[selection_start:selection_end]
                selection_context = (
                    f"\n\n=== USER HAS SELECTED TEXT ===\n"
                    f"Selected text (characters {selection_start}-{selection_end}):\n"
                    f'"""{selected_text[:500]}{"..." if len(selected_text) > 500 else ""}"""\n\n'
                    "The user selected this specific text for editing.\n"
                )
            
            # Determine if document is empty/minimal
            content_length = len(context_document.strip())
            is_empty_or_minimal = content_length < 50
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Current Date/Time: {datetime.now().isoformat()}"},
                {
                    "role": "user",
                    "content": (
                        "=== SYSML DOCUMENT CONTEXT ===\n"
                        f"Primary file: {filename}\n"
                        f"Document type: SysML Model (XMI format)\n"
                        f"Document content length: {content_length} characters {'(EMPTY/MINIMAL - use insert_after_heading!)' if is_empty_or_minimal else '(has content - can use replace_range if modifying existing)'}\n\n"
                        f"=== CURRENT DOCUMENT CONTENT ===\n{context_document if context_document.strip() else '(empty document - start fresh)'}\n\n"
                        + (f"USER REQUEST: {current_request}\n\n" if current_request else "")
                        + selection_context +
                        "\n=== YOUR TASK ===\n"
                        "Analyze the user's request and generate appropriate SysML model in XMI format.\n"
                        "Create blocks, properties, and relationships as needed for their system design.\n"
                        + ("⚠️ DOCUMENT IS EMPTY/MINIMAL: Use insert_after_heading with anchor_text='---' (or any existing heading)!\n" if is_empty_or_minimal else "")
                        + "Provide a ManuscriptEdit JSON with the XMI model code (wrapped in ```xml) in the operations[].text field.\n"
                    )
                },
            ]

            # Call LLM
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            start_time = datetime.now()
            
            response = await chat_service.openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,  # Lower temperature for consistent diagram generation
            )

            content = response.choices[0].message.content or "{}"
            logger.info(f"📝 RAW LLM RESPONSE (first 1000 chars): {content[:1000]}")
            content = _unwrap_json_response(content)
            
            # Parse ManuscriptEdit response
            structured: Optional[ManuscriptEdit] = None
            parse_error = None
            
            try:
                import json as _json
                raw_obj = _json.loads(content)
                
                if isinstance(raw_obj, dict):
                    # Normalize to ManuscriptEdit shape
                    if "operations" in raw_obj and isinstance(raw_obj["operations"], list):
                        raw_obj.setdefault("target_filename", filename)
                        raw_obj.setdefault("scope", "paragraph")
                        raw_obj.setdefault("summary", "SysML diagram generated")
                        
                        # Normalize operations
                        norm_ops = []
                        for op in raw_obj["operations"]:
                            if not isinstance(op, dict):
                                continue
                            
                            op_type = op.get("op_type") or op.get("type") or "replace_range"
                            start_ix = int(op.get("start", fm_end_idx))
                            end_ix = int(op.get("end", fm_end_idx))
                            text_val = op.get("text") or ""
                            
                            # Smart anchor_text fallback for empty documents
                            anchor_text_val = op.get("anchor_text")
                            if op_type == "insert_after_heading":
                                if not anchor_text_val or (isinstance(anchor_text_val, str) and len(anchor_text_val.strip()) < 3):
                                    # Empty anchor_text on insert_after_heading - provide smart default
                                    if is_empty_or_minimal:
                                        # Empty document: insert after frontmatter
                                        if "---" in context_document:
                                            anchor_text_val = "---"
                                            logger.info(f"🔧 SMART FALLBACK: Empty anchor_text, using '---' for frontmatter insertion")
                                        else:
                                            # No frontmatter marker, use beginning of document
                                            anchor_text_val = context_document.split('\n')[0] if context_document.strip() else "# System Diagram"
                                            logger.info(f"🔧 SMART FALLBACK: Empty anchor_text, using first line: '{anchor_text_val[:50]}'")
                                    else:
                                        # Document has content - try to find first heading
                                        lines = context_document.split('\n')
                                        for line in lines:
                                            if line.strip().startswith('#'):
                                                anchor_text_val = line.strip()
                                                logger.info(f"🔧 SMART FALLBACK: Empty anchor_text, using heading: '{anchor_text_val[:50]}'")
                                                break
                                        if not anchor_text_val:
                                            anchor_text_val = "---"  # Last resort
                            
                            # Auto-fix and validate XMI syntax in the text field
                            if text_val and ('<xmi:XMI' in text_val or '<uml:Model' in text_val):
                                logger.info(f"📝 XMI BEFORE AUTO-FIX (first 500 chars): {text_val[:500]}")
                                
                                # Apply auto-fixes for common LLM mistakes
                                text_val = _auto_fix_xmi(text_val)
                                
                                logger.info(f"📝 XMI AFTER AUTO-FIX (first 500 chars): {text_val[:500]}")
                                
                                # Validate the fixed XMI
                                is_valid, validation_error = _validate_xmi_syntax(text_val)
                                if not is_valid:
                                    logger.error(f"❌ XMI syntax validation failed: {validation_error}")
                                    return self._create_success_result(
                                        response=f"XMI syntax error: {validation_error}\n\nPlease ensure your XMI:\n- Starts with <?xml version=\"1.0\"?>\n- Has proper XMI namespaces\n- Contains <uml:Model> element\n- Is well-formed XML\n- Is compatible with Eclipse Papyrus",
                                        tools_used=[],
                                        processing_time=(datetime.now() - start_time).total_seconds(),
                                        additional_data={"validation_error": validation_error, "invalid_code": text_val[:500]},
                                    )
                                logger.info(f"✅ XMI syntax validation passed for operation")
                            
                            normalized_op = {
                                "op_type": op_type if op_type in ("replace_range", "delete_range", "insert_after_heading") else "replace_range",
                                "start": max(0, min(len(document_content), start_ix)),
                                "end": max(0, min(len(document_content), end_ix)),
                                "text": text_val,  # Use the auto-fixed text
                                "pre_hash": "",  # Will compute below
                                "original_text": op.get("original_text") or op.get("original"),
                                "anchor_text": anchor_text_val,
                                "left_context": op.get("left_context"),
                                "right_context": op.get("right_context"),
                                "occurrence_index": op.get("occurrence_index", 0),
                            }
                            norm_ops.append(normalized_op)
                        
                        raw_obj["operations"] = norm_ops
                        structured = ManuscriptEdit(**raw_obj)
                        
            except ValidationError as ve:
                logger.error(f"❌ Validation error in SysML operation: {ve}")
                error_msg = str(ve)
                return self._create_success_result(
                    response=f"Failed to validate the SysML operation: {error_msg}",
                    tools_used=[],
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    additional_data={"validation_error": error_msg, "raw": content},
                )
            except Exception as e:
                parse_error = e
            
            if structured is None:
                logger.error(f"❌ Failed to parse SysML response: {parse_error}")
                return self._create_success_result(
                    response="Failed to produce a valid SysML operation. Please refine your request.",
                    tools_used=[],
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    additional_data={"raw": content},
                )

            # Resolve operations with progressive search
            ops: List[EditorOperation] = []
            selection = {"start": selection_start, "end": selection_end} if selection_start >= 0 and selection_end >= 0 else None
            
            for op in structured.operations:
                op_dict = {
                    "original_text": getattr(op, "original_text", None),
                    "anchor_text": getattr(op, "anchor_text", None),
                    "left_context": getattr(op, "left_context", None),
                    "right_context": getattr(op, "right_context", None),
                    "occurrence_index": getattr(op, "occurrence_index", 0),
                    "text": op.text,
                    "op_type": op.op_type,
                }
                
                try:
                    resolved_start, resolved_end, resolved_text, resolved_confidence = resolve_operation(
                        document_content,
                        op_dict,
                        selection=selection,
                        heading_hint=None,
                        frontmatter_end=fm_end_idx,
                        require_anchors=False,
                    )
                    
                    logger.info(f"📍 SYSML: {op.op_type} resolved [{resolved_start}:{resolved_end}] confidence={resolved_confidence:.2f}")
                    
                    op.start = resolved_start
                    op.end = resolved_end
                    op.text = resolved_text
                    op.confidence = resolved_confidence
                    
                except ValueError as e:
                    logger.warning(f"⚠️ SYSML: Anchor resolution failed: {e}")
                    # Fallback to insertion after frontmatter
                    op.start = fm_end_idx
                    op.end = fm_end_idx
                    op.confidence = 0.3
                
                # Calculate pre_hash
                pre_slice = document_content[op.start:op.end]
                op.pre_hash = _slice_hash(pre_slice)
                
                ops.append(op)
            
            structured.operations = ops

            # Build response
            processing_time = (datetime.now() - start_time).total_seconds()
            
            try:
                generated_preview = "\n\n".join([
                    (getattr(op, "text", "") or "").strip()
                    for op in structured.operations
                    if (getattr(op, "text", "") or "").strip()
                ]).strip()
            except Exception:
                generated_preview = ""

            response_text = generated_preview if generated_preview else (structured.summary or "SysML diagram ready.")
            
            # Add clarifying questions if present
            clarifying_questions = getattr(structured, "clarifying_questions", None)
            if clarifying_questions and len(clarifying_questions) > 0:
                questions_section = "\n\n**Questions for clarification:**\n" + "\n".join([
                    f"- {q}" for q in clarifying_questions
                ])
                response_text = response_text + questions_section

            # Store operations for API streaming
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

        except Exception as e:
            logger.error(f"❌ SysMLAgent failed: {e}")
            return self._create_success_result(
                response="SysML agent encountered an error.", tools_used=[], processing_time=0.0
            )
