"""
General Project Agent - Decision Nodes Module
Handles decision extraction and documentation verification
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any

from orchestrator.models.general_project_models import (
    GeneralProjectDecision, GeneralProjectDocumentationInconsistency,
    GeneralProjectDocumentationVerificationResult
)

logger = logging.getLogger(__name__)


class GeneralProjectDecisionNodes:
    """Decision-related nodes for general project agent"""
    
    def __init__(self, agent):
        """
        Initialize with reference to main agent for access to helper methods
        
        Args:
            agent: GeneralProjectAgent instance (for _get_llm, _get_fast_model, etc.)
        """
        self.agent = agent
    
    async def extract_decisions_node(self, state) -> Dict[str, Any]:
        """
        Extract project decisions from conversation and response.
        
        Uses LLM to identify decisions made during the conversation:
        - Requirement selections
        - Design choices
        - Specification decisions
        - Approach decisions
        - Tradeoff decisions
        """
        try:
            query = state.get("query", "")
            response = state.get("response", {})
            response_text = response.get("response", "") if isinstance(response, dict) else str(response)
            messages = state.get("messages", [])
            user_id = state.get("user_id", "")
            metadata = state.get("metadata", {})
            
            # Get conversation context (last 5 messages)
            conversation_context = ""
            if messages:
                recent_messages = messages[-5:]
                for msg in recent_messages:
                    if hasattr(msg, 'content') and isinstance(msg.content, str):
                        role = "User" if hasattr(msg, 'type') and msg.type == "human" else "Agent"
                        conversation_context += f"{role}: {msg.content}\n\n"
            
            # Get existing decisions for context
            existing_decisions = state.get("project_decisions", [])
            
            # Build prompt for LLM to extract decisions
            fast_model = self.agent._get_fast_model(state)
            llm = self.agent._get_llm(temperature=0.1, model=fast_model, state=state)
            
            prompt = f"""You are a decision extraction expert. Analyze the conversation and identify project decisions made.

**QUERY**: {query}

**CONVERSATION CONTEXT**:
{conversation_context[:2000]}

**CURRENT RESPONSE**:
{response_text[:2000]}

**EXISTING DECISIONS** (for context):
{json.dumps(existing_decisions[-3:], indent=2) if existing_decisions else "None"}

**TASK**: Identify all project decisions made in this conversation. A decision is:
- A requirement selection (e.g., "We'll use X approach")
- A design choice (e.g., "Use Y method")
- A specification decision (e.g., "Budget of $Z")
- An approach decision (e.g., "Phase-based implementation")
- A tradeoff decision (e.g., "Prioritize cost over speed")
- Other project decisions

**CRITICAL - DECISION_TYPE VALUES**: You MUST use EXACTLY one of these values (no variations):
- "requirement" - for requirement choices
- "design_choice" - for design methodology choices
- "specification" - for specification decisions
- "approach" - for approach or methodology decisions
- "tradeoff" - for tradeoff decisions
- "other" - for any other type of decision

**INSTRUCTIONS**:
1. Extract decisions with clear details
2. Identify what was replaced (if any)
3. Note the reason for the decision (if mentioned)
4. List alternatives considered (if any)
5. Determine which files should document this decision
6. Link to superseded decisions (if this replaces a previous decision)
7. Use EXACT decision_type values listed above

**OUTPUT FORMAT**: Return ONLY valid JSON:
{{
  "decisions": [
    {{
      "decision_id": "dec_001",
      "timestamp": "2024-01-15T10:30:00",
      "decision_type": "design_choice",
      "decision_summary": "Use phased approach for HVAC installation",
      "details": {{
        "approach": "Phased installation",
        "phases": ["Planning", "Installation", "Testing"]
      }},
      "replaced_item": "Single-phase approach",
      "reason": "Reduced disruption and better cost management",
      "alternatives_considered": ["Single-phase", "Full replacement"],
      "documented_in": ["./design_docs.md"],
      "supersedes": []
    }}
  ]
}}

Return ONLY the JSON object, no markdown, no code blocks."""
            
            try:
                response_obj = await llm.ainvoke([{"role": "user", "content": prompt}])
                content = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
                decisions_dict = self.agent._extract_json_from_response(content) or {}
            except Exception as e:
                logger.warning(f"Decision extraction failed: {e}")
                decisions_dict = {}
            
            decisions_list = decisions_dict.get("decisions", [])
            
            # Generate decision IDs and timestamps if not provided
            existing_decisions = state.get("project_decisions", [])
            next_id = len(existing_decisions) + 1
            
            validated_decisions = []
            for i, decision in enumerate(decisions_list):
                if not decision.get("decision_id"):
                    decision["decision_id"] = f"dec_{next_id + i:03d}"
                if not decision.get("timestamp"):
                    decision["timestamp"] = datetime.now().isoformat()
                
                # Normalize decision_type
                decision_type = decision.get("decision_type", "").lower()
                if decision_type == "specification_decision":
                    decision["decision_type"] = "specification"
                elif decision_type == "approach_decision":
                    decision["decision_type"] = "approach"
                elif decision_type == "design_choice_decision":
                    decision["decision_type"] = "design_choice"
                elif decision_type == "requirement_decision":
                    decision["decision_type"] = "requirement"
                elif decision_type == "tradeoff_decision":
                    decision["decision_type"] = "tradeoff"
                
                try:
                    validated_decision = GeneralProjectDecision(**decision)
                    validated_decisions.append(validated_decision.dict() if hasattr(validated_decision, 'dict') else validated_decision.model_dump())
                except Exception as e:
                    logger.warning(f"Failed to validate decision: {e}")
                    if "decision_type" in str(e):
                        original_type = decision.get("decision_type", "unknown")
                        decision["decision_type"] = "other"
                        try:
                            validated_decision = GeneralProjectDecision(**decision)
                            logger.info(f"Fixed decision_type '{original_type}' -> 'other'")
                            validated_decisions.append(validated_decision.dict() if hasattr(validated_decision, 'dict') else validated_decision.model_dump())
                        except Exception as e2:
                            logger.warning(f"Failed to validate even with 'other' type: {e2}")
                            validated_decisions.append(decision)
                    else:
                        validated_decisions.append(decision)
            
            if validated_decisions:
                logger.info(f"Extracted {len(validated_decisions)} decision(s)")
                all_decisions = existing_decisions + validated_decisions
                return {
                    "project_decisions": all_decisions,
                    # ✅ CRITICAL: Preserve critical state keys
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "referenced_context": state.get("referenced_context", {}),
                    "response": state.get("response", {})
                }
            
            return {
                # ✅ CRITICAL: Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {}),
                "response": state.get("response", {})
            }
            
        except Exception as e:
            logger.error(f"Decision extraction failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                # ✅ CRITICAL: Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {}),
                "response": state.get("response", {})
            }
    
    async def verify_documentation_node(self, state) -> Dict[str, Any]:
        """
        Verify that documentation accurately reflects project decisions.
        
        Uses LLM to compare:
        - Recent decisions vs. documented state
        - Requirement selections vs. requirement documentation
        - Design decisions vs. design documentation
        - Specifications vs. actual implementation
        """
        try:
            project_decisions = state.get("project_decisions", [])
            referenced_context = state.get("referenced_context", {})
            user_id = state.get("user_id", "")
            metadata = state.get("metadata", {})
            
            if not project_decisions:
                logger.info("No decisions to verify - skipping verification")
                return {
                    "documentation_verification_result": {
                        "verification_status": "consistent",
                        "inconsistencies": [],
                        "missing_documentation": [],
                        "outdated_sections": [],
                        "completeness_score": 1.0,
                        "consistency_score": 1.0,
                        "reasoning": "No decisions to verify"
                    },
                    # ✅ CRITICAL: Preserve critical state keys
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "referenced_context": state.get("referenced_context", {}),
                    "project_decisions": state.get("project_decisions", [])
                }
            
            # Get recent decisions (last 5)
            recent_decisions = project_decisions[-5:]
            
            # Get documentation content from referenced files
            documentation_content = {}
            for category, file_list in referenced_context.items():
                if isinstance(file_list, list):
                    for file_doc in file_list:
                        if isinstance(file_doc, dict):
                            content = file_doc.get("content", "")
                            filename = file_doc.get("filename", "")
                            if content:
                                documentation_content[filename] = content[:5000]
            
            # Build prompt for LLM to verify consistency
            fast_model = self.agent._get_fast_model(state)
            llm = self.agent._get_llm(temperature=0.1, model=fast_model, state=state)
            
            prompt = f"""You are a documentation verification expert. Compare project decisions with documentation and identify inconsistencies.

**RECENT DECISIONS**:
{json.dumps(recent_decisions, indent=2)}

**DOCUMENTATION CONTENT**:
{json.dumps({k: v[:2000] for k, v in list(documentation_content.items())[:5]}, indent=2) if documentation_content else "No documentation available"}

**TASK**: Verify that documentation accurately reflects the decisions made. Look for:
1. **Mismatches**: Documentation mentions old approach but decision selected new one
2. **Specification conflicts**: Documentation has different values than decisions
3. **Outdated information**: Documentation hasn't been updated to reflect decisions
4. **Missing documentation**: Decisions not documented in appropriate files
5. **Contradictions**: Documentation contradicts decisions

**INSTRUCTIONS**:
1. Compare each decision with relevant documentation sections
2. Identify inconsistencies with high confidence (>0.8)
3. Suggest specific fixes for each inconsistency
4. Calculate consistency and completeness scores
5. Return structured verification results

**OUTPUT FORMAT**: Return ONLY valid JSON:
{{
  "verification_status": "consistent|inconsistent|needs_review",
  "inconsistencies": [
    {{
      "file": "./design_docs.md",
      "section": "Installation Approach",
      "issue_type": "mismatch",
      "description": "Documentation mentions single-phase but decision was phased approach",
      "severity": "high",
      "documented_value": "Single-phase installation",
      "actual_value": "Phased installation",
      "related_decision_id": "dec_001",
      "suggested_fix": "Update to phased approach or remove obsolete single-phase section"
    }}
  ],
  "missing_documentation": [],
  "outdated_sections": [],
  "completeness_score": 0.85,
  "consistency_score": 0.70,
  "reasoning": "Explanation of verification results"
}}

Return ONLY the JSON object, no markdown, no code blocks."""
            
            try:
                response_obj = await llm.ainvoke([{"role": "user", "content": prompt}])
                content = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
                verification_dict = self.agent._extract_json_from_response(content) or {}
            except Exception as e:
                logger.warning(f"Verification failed: {e}")
                verification_dict = {
                    "verification_status": "needs_review",
                    "inconsistencies": [],
                    "missing_documentation": [],
                    "outdated_sections": [],
                    "completeness_score": 0.5,
                    "consistency_score": 0.5,
                    "reasoning": "Verification failed"
                }
            
            # Validate inconsistencies with Pydantic
            inconsistencies = verification_dict.get("inconsistencies", [])
            validated_inconsistencies = []
            for inconsistency in inconsistencies:
                try:
                    validated = GeneralProjectDocumentationInconsistency(**inconsistency)
                    validated_inconsistencies.append(validated.dict() if hasattr(validated, 'dict') else validated.model_dump())
                except Exception as e:
                    logger.warning(f"Failed to validate inconsistency: {e}")
                    validated_inconsistencies.append(inconsistency)
            
            verification_dict["inconsistencies"] = validated_inconsistencies
            
            # Pre-process missing_documentation and outdated_sections to convert lists to strings
            for item in verification_dict.get("missing_documentation", []):
                if isinstance(item, dict):
                    for key, value in item.items():
                        if isinstance(value, list):
                            item[key] = ", ".join(str(v) for v in value)
            
            for item in verification_dict.get("outdated_sections", []):
                if isinstance(item, dict):
                    for key, value in item.items():
                        if isinstance(value, list):
                            item[key] = ", ".join(str(v) for v in value)
            
            # Validate with Pydantic model
            try:
                verification_result = GeneralProjectDocumentationVerificationResult(**verification_dict)
                verification_dict = verification_result.dict() if hasattr(verification_result, 'dict') else verification_result.model_dump()
                logger.info(f"Verification complete: {verification_dict.get('verification_status')} ({len(validated_inconsistencies)} inconsistencies)")
            except Exception as e:
                logger.warning(f"Failed to validate verification result: {e}")
                # Continue with unvalidated dict - model change to Any should help, but if validation still fails, use raw dict
            
            return {
                "documentation_verification_result": verification_dict,
                # ✅ CRITICAL: Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {}),
                "project_decisions": state.get("project_decisions", [])
            }
            
        except Exception as e:
            logger.error(f"Documentation verification failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "documentation_verification_result": {
                    "verification_status": "needs_review",
                    "inconsistencies": [],
                    "missing_documentation": [],
                    "outdated_sections": [],
                    "completeness_score": 0.0,
                    "consistency_score": 0.0,
                    "reasoning": "Verification failed due to error"
                },
                # ✅ CRITICAL: Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {}),
                "project_decisions": state.get("project_decisions", [])
            }


