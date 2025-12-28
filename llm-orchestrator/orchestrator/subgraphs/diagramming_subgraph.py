"""
Diagramming Subgraph

Reusable subgraph for generating technical diagrams (Mermaid, ASCII art, circuit markup).
Can be used by:
- Electronics Agent (circuit diagrams, state machines, pin tables)
- General Project Agent (workflows, Gantt charts, decision trees)
- Any agent needing diagram generation

Inputs:
- query: The diagram request or original query
- messages: Conversation history for context
- metadata: Optional metadata (user_id, etc.)
- project_context: Optional project context (referenced files, active editor, etc.)

Outputs:
- diagram_result: Diagram generation result with success status
- diagram_type: Type of diagram generated
- diagram_content: Diagram content (Mermaid syntax, ASCII art, markdown table)
- diagram_format: Output format (mermaid, ascii, markdown_table)
"""

import logging
import json
import re
from typing import Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from orchestrator.agents.base_agent import BaseAgent
from orchestrator.models.diagramming_models import DiagramAnalysis, DiagramResult

logger = logging.getLogger(__name__)


# Use Dict[str, Any] for compatibility with any agent state
DiagrammingSubgraphState = Dict[str, Any]


def _build_diagram_analysis_prompt(user_request: str, conversation_context: str, project_context: Dict[str, Any] = None) -> str:
    """Build prompt for LLM to analyze diagramming needs"""
    
    project_info = ""
    if project_context:
        active_editor = project_context.get("active_editor", {})
        referenced_files = project_context.get("referenced_files", {})
        if active_editor:
            project_info += f"\n**ACTIVE DOCUMENT**: {active_editor.get('filename', 'Unknown')}\n"
        if referenced_files:
            project_info += f"\n**REFERENCED FILES**: {len(referenced_files)} files available for context\n"
    
    return f"""You are a Technical Diagram Specialist - an expert at creating visual representations of technical concepts, circuits, workflows, and system architectures.

**MISSION**: Analyze the user's request and determine the best diagram type and generate appropriate diagram content.

**USER REQUEST**: {user_request}

**CONVERSATION CONTEXT**:
{conversation_context}
{project_info}

**STRUCTURED OUTPUT REQUIRED**:

You MUST respond with valid JSON matching this schema:
{{
    "diagram_type": "mermaid_flowchart|mermaid_sequence|mermaid_state|mermaid_gantt|mermaid_class|mermaid_er|circuit_ascii|pin_table|block_diagram",
    "title": "Diagram title that summarizes what the diagram shows",
    "description": "Description of what the diagram represents",
    "diagram_content": "The actual diagram code (Mermaid syntax, ASCII art, or markdown table)",
    "metadata": {{"key": "value"}},
    "confidence": 0.9,
    "reasoning": "Explanation of why this diagram type and content were chosen"
}}

**DIAGRAM TYPE SELECTION GUIDE**:

**MERMAID DIAGRAMS**:
- **mermaid_flowchart**: Process flows, decision trees, workflows, circuit block diagrams
- **mermaid_sequence**: Communication protocols, timing diagrams, interaction flows
- **mermaid_state**: State machines, firmware state transitions, system states
- **mermaid_gantt**: Project timelines, task schedules, milestones
- **mermaid_class**: Software architecture, class hierarchies, component relationships
- **mermaid_er**: Entity-relationship diagrams, database schemas, system architecture

**CIRCUIT/TECHNICAL DIAGRAMS**:
- **circuit_ascii**: Simple circuit diagrams in ASCII art format
- **pin_table**: Pin connection tables for microcontrollers (markdown table format)
- **block_diagram**: System block diagrams (can use Mermaid flowchart or ASCII)

**MERMAID SYNTAX EXAMPLES**:

**Flowchart Example**:
```mermaid
flowchart TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Action 1]
    B -->|No| D[Action 2]
    C --> E[End]
    D --> E
```

**Sequence Diagram Example**:
```mermaid
sequenceDiagram
    participant A as Component A
    participant B as Component B
    A->>B: Request
    B-->>A: Response
```

**State Diagram Example**:
```mermaid
stateDiagram-v2
    [*] --> State1
    State1 --> State2: Event
    State2 --> [*]
```

**Gantt Chart Example**:
```mermaid
gantt
    title Project Timeline
    dateFormat YYYY-MM-DD
    section Phase 1
    Task 1 :a1, 2024-01-01, 30d
    Task 2 :a2, after a1, 20d
```

**PIN TABLE FORMAT**:
Use markdown table format:
```markdown
| Pin | Function | Connected To | Notes |
|-----|----------|-------------|-------|
| 1   | VCC      | 3.3V        | Power |
| 2   | GND      | Ground      | Ground |
| 3   | GPIO     | LED         | Output |
```

**ASCII CIRCUIT EXAMPLE**:
```
     +3.3V
      |
      R1 (10k)
      |
      +--- GPIO Pin
      |
     GND
```

**GENERATION RULES**:
1. **Use appropriate diagram type** - Match the user's request and context
2. **Complete diagrams** - Include all mentioned components, connections, states, or steps
3. **Valid syntax** - Ensure Mermaid syntax is correct, ASCII is readable, tables are properly formatted
4. **Clear labels** - Use descriptive names for all elements
5. **Context-aware** - Reference project context when available (component names, pin numbers, etc.)

**CRITICAL**:
1. **STRUCTURED JSON ONLY** - No plain text responses!
2. **Valid diagram_type** - Must match one of the supported types
3. **Complete diagram_content** - Must be valid Mermaid syntax, readable ASCII, or proper markdown table
4. **Use project context** - Reference actual components, pins, or project details when available

**JSON RESPONSE EXAMPLE**:
```json
{{
    "diagram_type": "mermaid_state",
    "title": "Firmware State Machine",
    "description": "State transitions for the main firmware loop",
    "diagram_content": "stateDiagram-v2\\n    [*] --> Idle\\n    Idle --> Processing: Data Ready\\n    Processing --> Idle: Complete",
    "metadata": {{"states": ["Idle", "Processing"], "transitions": 2}},
    "confidence": 0.95,
    "reasoning": "State diagram best represents firmware state transitions"
}}
```
"""


async def analyze_diagram_request_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze diagram request and determine diagram type"""
    try:
        logger.info("Analyzing diagram request...")
        
        query = state.get("query", "")
        messages = state.get("messages", [])
        project_context = state.get("project_context", {})
        
        # Extract conversation context
        if not messages:
            conversation_context = "No previous conversation context available."
        else:
            recent_messages = messages[-5:] if len(messages) > 5 else messages
            context_parts = []
            for i, msg in enumerate(recent_messages):
                if hasattr(msg, 'content'):
                    role = "ASSISTANT" if hasattr(msg, 'type') and msg.type == "ai" else "USER"
                    content = msg.content
                    # Truncate very long messages
                    if len(content) > 500:
                        content = content[:500] + "..."
                    context_parts.append(f"{i+1}. {role}: {content}")
            conversation_context = "\n".join(context_parts)
        
        # Build analysis prompt
        analysis_prompt = _build_diagram_analysis_prompt(query, conversation_context, project_context)
        
        # Get LLM for analysis (low temperature for consistent analysis)
        base_agent = BaseAgent("diagramming_subgraph")
        llm = base_agent._get_llm(temperature=0.2, state=state)
        datetime_context = base_agent._get_datetime_context()
        
        # Build messages
        analysis_messages = [
            SystemMessage(content="You are a technical diagram specialist. Always respond with valid JSON."),
            SystemMessage(content=datetime_context)
        ]
        
        if messages:
            analysis_messages.extend(messages)
        
        analysis_messages.append(HumanMessage(content=analysis_prompt))
        
        # Call LLM
        response = await llm.ainvoke(analysis_messages)
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Extract JSON from response
        text = response_content.strip()
        
        # Extract JSON from markdown code blocks
        if '```json' in text:
            match = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
            if match:
                text = match.group(1).strip()
        elif '```' in text:
            match = re.search(r'```\s*\n([\s\S]*?)\n```', text)
            if match:
                text = match.group(1).strip()
        
        # Extract JSON object
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            text = json_match.group(0)
        
        # Parse JSON
        try:
            analysis_dict = json.loads(text)
            # Validate with Pydantic
            diagram_analysis = DiagramAnalysis(**analysis_dict)
            
            logger.info(f"Diagram analysis complete: {diagram_analysis.diagram_type}, confidence: {diagram_analysis.confidence}")
            
            return {
                "diagram_analysis": diagram_analysis.dict(),
                "diagram_type": diagram_analysis.diagram_type,
                "diagram_title": diagram_analysis.title,
                "diagram_description": diagram_analysis.description,
                "diagram_content": diagram_analysis.diagram_content,
                "diagram_metadata": diagram_analysis.metadata,
                # Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            logger.warning(f"Failed to parse diagram analysis: {e}")
            return {
                "diagram_analysis": None,
                "diagram_error": f"Failed to parse diagram analysis: {str(e)}",
                # Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
    except Exception as e:
        logger.error(f"Diagram analysis failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "diagram_analysis": None,
            "diagram_error": str(e),
            # Preserve critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def generate_diagram_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate diagram content using LLM"""
    try:
        logger.info("Generating diagram content...")
        
        diagram_analysis = state.get("diagram_analysis")
        
        if not diagram_analysis:
            error_msg = state.get("diagram_error", "No diagram analysis available")
            logger.warning(f"Cannot generate diagram: {error_msg}")
            return {
                "diagram_result": {
                    "success": False,
                    "error": error_msg
                },
                "diagram_type": None,
                "diagram_content": None,
                # Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        # Extract diagram parameters
        diagram_type = diagram_analysis.get("diagram_type")
        title = diagram_analysis.get("title", "")
        description = diagram_analysis.get("description", "")
        initial_content = diagram_analysis.get("diagram_content", "")
        metadata = diagram_analysis.get("metadata", {})
        
        # Determine format based on diagram type
        if diagram_type.startswith("mermaid_"):
            diagram_format = "mermaid"
        elif diagram_type == "pin_table":
            diagram_format = "markdown_table"
        else:
            diagram_format = "ascii"
        
        # If initial content is already good, use it; otherwise refine with LLM
        if initial_content and len(initial_content) > 20:
            diagram_content = initial_content
            logger.info(f"Using initial diagram content from analysis ({len(diagram_content)} chars)")
        else:
            # Refine diagram with LLM (medium temperature for creativity)
            base_agent = BaseAgent("diagramming_subgraph")
            llm = base_agent._get_llm(temperature=0.5, state=state)
            
            refinement_prompt = f"""Generate a complete {diagram_type} diagram with the following requirements:

**Title**: {title}
**Description**: {description}
**Metadata**: {json.dumps(metadata, indent=2)}

**Requirements**:
1. Generate complete, valid diagram content
2. For Mermaid diagrams, use proper syntax
3. For ASCII circuits, make them readable and clear
4. For pin tables, use proper markdown table format
5. Include all relevant elements from the metadata

**Output only the diagram content** (no explanations, no markdown code blocks, just the raw diagram code).
"""
            
            refinement_messages = [
                SystemMessage(content="You are a technical diagram generator. Output only the diagram content."),
                HumanMessage(content=refinement_prompt)
            ]
            
            response = await llm.ainvoke(refinement_messages)
            diagram_content = response.content if hasattr(response, 'content') else str(response)
            diagram_content = diagram_content.strip()
            
            # Remove markdown code blocks if present
            if '```mermaid' in diagram_content:
                match = re.search(r'```mermaid\s*\n([\s\S]*?)\n```', diagram_content)
                if match:
                    diagram_content = match.group(1).strip()
            elif '```' in diagram_content:
                match = re.search(r'```\s*\n([\s\S]*?)\n```', diagram_content)
                if match:
                    diagram_content = match.group(1).strip()
        
        logger.info(f"Diagram content generated: {diagram_type}, format: {diagram_format}, length: {len(diagram_content)}")
        
        return {
            "diagram_result": {
                "success": True,
                "diagram_type": diagram_type,
                "title": title,
                "diagram_format": diagram_format,
                "diagram_content": diagram_content,
                "validation_errors": None,
                "error": None
            },
            "diagram_type": diagram_type,
            "diagram_title": title,
            "diagram_format": diagram_format,
            "diagram_content": diagram_content,
            # Preserve critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
        
    except Exception as e:
        logger.error(f"Diagram generation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "diagram_result": {
                "success": False,
                "diagram_type": state.get("diagram_type", "unknown"),
                "title": "",
                "diagram_format": "unknown",
                "diagram_content": None,
                "validation_errors": None,
                "error": str(e)
            },
            "diagram_type": None,
            "diagram_content": None,
            # Preserve critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def validate_diagram_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate diagram syntax and completeness"""
    try:
        logger.info("Validating diagram...")
        
        diagram_result = state.get("diagram_result", {})
        
        if not diagram_result or not diagram_result.get("success"):
            error_msg = diagram_result.get("error", "No diagram result available")
            logger.warning(f"Cannot validate diagram: {error_msg}")
            return {
                "diagram_result": diagram_result,
                # Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        diagram_type = diagram_result.get("diagram_type", "")
        diagram_format = diagram_result.get("diagram_format", "")
        diagram_content = diagram_result.get("diagram_content", "")
        validation_errors = []
        
        # Validate based on format
        if diagram_format == "mermaid":
            # Basic Mermaid syntax validation
            if not diagram_content.strip():
                validation_errors.append("Mermaid diagram content is empty")
            elif not any(keyword in diagram_content.lower() for keyword in ["flowchart", "sequence", "state", "gantt", "class", "erDiagram"]):
                validation_errors.append("Mermaid diagram missing required diagram type keyword")
            # Check for balanced brackets/parentheses (basic check)
            open_brackets = diagram_content.count('[') + diagram_content.count('(') + diagram_content.count('{')
            close_brackets = diagram_content.count(']') + diagram_content.count(')') + diagram_content.count('}')
            if open_brackets != close_brackets:
                validation_errors.append(f"Unbalanced brackets in Mermaid diagram (open: {open_brackets}, close: {close_brackets})")
        
        elif diagram_format == "markdown_table":
            # Validate markdown table format
            if not diagram_content.strip():
                validation_errors.append("Pin table content is empty")
            elif "|" not in diagram_content:
                validation_errors.append("Pin table missing table separators (|)")
            elif not re.search(r'\|[-\s|:]+\|', diagram_content):
                validation_errors.append("Pin table missing separator row")
        
        elif diagram_format == "ascii":
            # Basic ASCII validation
            if not diagram_content.strip():
                validation_errors.append("ASCII diagram content is empty")
            elif len(diagram_content) < 10:
                validation_errors.append("ASCII diagram content too short")
        
        # Update diagram result with validation
        if validation_errors:
            logger.warning(f"Diagram validation found {len(validation_errors)} issues: {validation_errors}")
            diagram_result["validation_errors"] = validation_errors
        else:
            logger.info("Diagram validation passed")
            diagram_result["validation_errors"] = None
        
        return {
            "diagram_result": diagram_result,
            # Preserve critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
        
    except Exception as e:
        logger.error(f"Diagram validation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return existing result even if validation fails
        return {
            "diagram_result": state.get("diagram_result", {
                "success": False,
                "error": f"Validation error: {str(e)}"
            }),
            # Preserve critical state keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


def build_diagramming_subgraph(checkpointer) -> StateGraph:
    """
    Build diagramming subgraph for generating technical diagrams
    
    This subgraph analyzes diagram requests, generates appropriate diagrams (Mermaid, ASCII, tables),
    and validates the output.
    
    Expected state inputs:
    - query: str - The diagram request or original query
    - messages: List (optional) - Conversation history for context
    - metadata: Dict[str, Any] (optional) - Metadata for checkpointing and user model selection
    - project_context: Dict[str, Any] (optional) - Project context (referenced files, active editor)
    
    Returns state with:
    - diagram_result: Dict[str, Any] - Diagram generation result with success status
    - diagram_type: str - Type of diagram generated
    - diagram_content: str - Diagram content (Mermaid syntax, ASCII art, or markdown table)
    - diagram_format: str - Output format (mermaid, ascii, markdown_table)
    """
    subgraph = StateGraph(Dict[str, Any])
    
    # Add nodes
    subgraph.add_node("analyze_diagram_request", analyze_diagram_request_node)
    subgraph.add_node("generate_diagram", generate_diagram_node)
    subgraph.add_node("validate_diagram", validate_diagram_node)
    
    # Set entry point
    subgraph.set_entry_point("analyze_diagram_request")
    
    # Linear flow: analyze -> generate -> validate -> END
    subgraph.add_edge("analyze_diagram_request", "generate_diagram")
    subgraph.add_edge("generate_diagram", "validate_diagram")
    subgraph.add_edge("validate_diagram", END)
    
    return subgraph.compile(checkpointer=checkpointer)
