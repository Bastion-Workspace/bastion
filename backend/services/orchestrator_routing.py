"""
Orchestrator Routing Helpers - Roosevelt's Decisive Dispatch
Pure routing functions extracted for clarity and reuse.
"""

import logging
from typing import Dict, Any

class IntentType:
    RESEARCH = "research"
    DATA_FORMATTING = "data_formatting"
    WEATHER = "weather"
    CHAT = "chat"
    DIRECT = "direct"
    ORG_INBOX = "org_inbox"
    IMAGE_GENERATION = "image_generation"
    WARGAMING = "wargaming"
    PROOFREADING = "proofreading"
    ARTICLE_ANALYSIS = "article_analysis"
    ENTERTAINMENT = "entertainment"


logger = logging.getLogger(__name__)


def route_from_intent(state: Dict[str, Any]) -> str:
    """Route based on capability-based intent classification with multi-step workflow support."""
    try:
        # ROOSEVELT'S GRAPH DEPTH MONITORING: Track traversal depth for recursion limit awareness
        graph_depth = state.get("_graph_depth", 0) + 1
        state["_graph_depth"] = graph_depth
        logger.info(f"📊 GRAPH DEPTH: {graph_depth}/50 (recursion_limit)")
        
        if graph_depth > 40:  # 80% warning threshold
            logger.warning(f"⚠️ APPROACHING RECURSION LIMIT: {graph_depth}/50")
        
        # **BULLY! MESSAGING OVERRIDE**: Route messaging requests to messaging_agent
        # Roosevelt's precise pattern matching - no false positives!
        try:
            user_msg = ""
            msgs = state.get("messages", []) or []
            for msg in reversed(msgs):
                if hasattr(msg, "type") and msg.type == "human":
                    user_msg = str(msg.content or "")
                    break
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_msg = str(msg.get("content") or "")
                    break
            
            user_msg_lower = user_msg.lower()
            
            # **ROOSEVELT'S PRECISE MESSAGING PATTERNS**: Exclude innocent conversational phrases
            # Match only explicit messaging commands, not "tell me about" or "send me info"
            import re
            
            # Positive patterns: explicit messaging commands
            messaging_patterns = [
                r"send\s+(a\s+)?message\s+to\s+\w+",  # "send message to John"
                r"send\s+to\s+\w+",  # "send to John" (but not "send to me")
                r"dm\s+\w+",  # "dm John"
                r"direct\s+message\s+\w+",  # "direct message John"
                r"tell\s+\w+\s+(that|about|to)",  # "tell John that..." (but not "tell me")
            ]
            
            # Negative patterns: exclude conversational phrases
            exclude_patterns = [
                r"tell\s+me\s+",  # "tell me about"
                r"tell\s+us\s+",  # "tell us about"
                r"send\s+me\s+",  # "send me info"
                r"send\s+us\s+",  # "send us data"
            ]
            
            # Check if message matches exclusions first
            is_excluded = any(re.search(pattern, user_msg_lower) for pattern in exclude_patterns)
            
            # Only route to messaging if matches patterns AND not excluded
            if not is_excluded:
                is_messaging = any(re.search(pattern, user_msg_lower) for pattern in messaging_patterns)
                if is_messaging:
                    logger.info("💬 BULLY! MESSAGING OVERRIDE: Routing to messaging_agent")
                    return "messaging_agent"
        except Exception as e:
            logger.error(f"❌ Messaging pattern check failed: {e}")
        
        # URL-triggered site crawl: if research intent and message contains a URL, route to site_crawl_agent
        try:
            user_msg = ""
            msgs = state.get("messages", []) or []
            for msg in reversed(msgs):
                if hasattr(msg, "type") and msg.type == "human":
                    user_msg = str(msg.content or "")
                    break
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_msg = str(msg.get("content") or "")
                    break
            import re
            has_url = bool(re.search(r'https?://[^\s)>\"]+', user_msg))
            if has_url:
                intent_result = state.get("intent_classification", {}) or {}
                if (intent_result.get("target_agent") == "research_agent") or (intent_result.get("intent_type") == "research"):
                    return "site_crawl_agent"
        except Exception:
            pass
        # Project capture in progress? Always route to org_project_agent until resolved
        shared_memory = state.get("shared_memory", {}) or {}
        pending_proj = shared_memory.get("pending_project_capture") or {}
        if isinstance(pending_proj, dict) and (pending_proj.get("awaiting_confirmation") or pending_proj.get("missing_fields")):
            logger.info("🎯 ROUTING OVERRIDE: Pending project capture detected → org_project_agent")
            return "org_project_agent"

        # === CONVERSATION-LEVEL AGENT LOCK OVERRIDE ===
        # Honor a persistent agent lock set in state/shared_memory to force routing to a specific agent
        locked_agent = state.get("locked_agent") or shared_memory.get("locked_agent") or shared_memory.get("agent_lock")
        if isinstance(locked_agent, str) and locked_agent.strip():
            locked_agent_normalized = locked_agent.strip().lower()
            # Normalize common inputs to routing destinations
            # Accept values like "wargaming", "wargaming_agent", "WARGAMING"
            lock_map = {
                "chat": "chat",
                "chat_agent": "chat",
                "research": "research",
                "research_agent": "research",
                "weather": "weather",
                "weather_agent": "weather",
                "data_formatting": "data_formatting",
                "data_formatting_agent": "data_formatting",
                "rss": "rss",
                "rss_agent": "rss",
                "org_inbox": "org_inbox",
                "org_inbox_agent": "org_inbox",
                "content_analysis": "content_analysis",
                "content_analysis_agent": "content_analysis",
                "fiction_editing": "fiction_editing",
                "fiction_editing_agent": "fiction_editing",
                "rules_editing": "rules_editing",
                "rules_editing_agent": "rules_editing",
                "sysml": "sysml",
                "sysml_agent": "sysml",
                "character_development": "character_development",
                "character_development_agent": "character_development",
                "outline_editing": "outline_editing",
                "outline_editing_agent": "outline_editing",
                "image_generation": "image_generation",
                "image_generation_agent": "image_generation",
                "wargaming": "wargaming",
                "wargaming_agent": "wargaming",
            }
            locked_route = lock_map.get(locked_agent_normalized)
            if locked_route:
                logger.info(f"🎯 AGENT LOCK OVERRIDE: Forcing routing to '{locked_route}' (locked_agent='{locked_agent}')")
                return locked_route

        intent_result = state.get("intent_classification", {})
        target_agent = intent_result.get("target_agent")

        # If simple classifier provided a direct agent, route decisively
        if isinstance(target_agent, str) and target_agent.strip():
            agent_to_routing_map = {
                "chat_agent": "chat_agent",
                "research_agent": "research_agent", 
                "weather_agent": "weather_agent",
                "data_formatting_agent": "data_formatting_agent",
                "rss_agent": "rss_agent",
                "content_analysis_agent": "content_analysis_agent",
                "story_analysis_agent": "story_analysis_agent",
                "fiction_editing_agent": "fiction_editing_agent",
                "sysml_agent": "sysml_agent",
                "fact_checking_agent": "fact_checking_agent",
                "podcast_script_agent": "podcast_script_agent",
                "substack_agent": "substack_agent",
                "image_generation_agent": "image_generation_agent",
                "wargaming_agent": "wargaming_agent",
                "combined_proofread_and_analyze": "combined_proofread_and_analyze",
                "rules_editing_agent": "rules_editing_agent",
                "character_development_agent": "character_development_agent",
                "outline_editing_agent": "outline_editing_agent",
                "org_inbox_agent": "org_inbox_agent",
                "org_project_agent": "org_project_agent",
                "messaging_agent": "messaging_agent",  # BULLY! Messaging cavalry!
                "entertainment_agent": "entertainment_agent",  # BULLY! Entertainment cavalry!
            }
            dest = agent_to_routing_map.get(target_agent.strip(), "chat_agent")
            # HITL: Permission routing applies only to research path
            if intent_result.get("permission_required") is True and dest == "research_agent":
                logger.info("🛑 PERMISSION REQUIRED: Routing to web_search_permission node (research only)")
                return "web_search_permission"
            logger.info(f"🎯 DIRECT ROUTING: {target_agent} → {dest}")
            return dest

        logger.info("🎯 CAPABILITY ROUTING FROM INTENT: No direct target_agent; checking capability_result")

        # **ROOSEVELT'S CAPABILITY-BASED ROUTING**: Use enhanced routing decision
        capability_result = intent_result.get("capability_result")
        if capability_result:
            routing_decision = capability_result.get("routing_decision", {})
            primary_agent = routing_decision.get("primary_agent")
            primary_confidence = routing_decision.get("primary_confidence", 0.0)
            
            logger.info(f"🎯 CAPABILITY ROUTING: {primary_agent} (confidence: {primary_confidence:.2f})")
            
            # Check for multi-step workflow requirements (research only)
            permission_req = routing_decision.get("permission_requirement", {})
            if permission_req.get("required", False) and not permission_req.get("auto_grant_eligible", False) and primary_agent == "research_agent":
                logger.info("🛑 PERMISSION REQUIRED: Routing to web_search_permission node (research only)")
                return "web_search_permission"
            
            # Convert agent names to routing destinations (use full node names)
            agent_to_routing_map = {
                "chat_agent": "chat_agent",
                "research_agent": "research_agent", 
                "weather_agent": "weather_agent",
                "data_formatting_agent": "data_formatting_agent",
                "rss_agent": "rss_agent",
                "coding_agent": "chat_agent",  # Coding routes to chat agent for now
                "calculate_agent": "chat_agent",  # Calculate routes to chat agent
                "content_analysis_agent": "content_analysis_agent",
                "story_analysis_agent": "story_analysis_agent",
                "fiction_editing_agent": "fiction_editing_agent",
                "sysml_agent": "sysml_agent",
                "fact_checking_agent": "fact_checking_agent",
                "podcast_script_agent": "podcast_script_agent",
                "substack_agent": "substack_agent",
                "image_generation_agent": "image_generation_agent",
                "wargaming_agent": "wargaming_agent",
                "combined_proofread_and_analyze": "combined_proofread_and_analyze",
                "rules_editing_agent": "rules_editing_agent",
                "character_development_agent": "character_development_agent",
                "outline_editing_agent": "outline_editing_agent",
                "messaging_agent": "messaging_agent",  # BULLY! Messaging cavalry!
            }
            
            routing_destination = agent_to_routing_map.get(primary_agent, "chat_agent")
            logger.info(f"🎯 CAPABILITY ROUTING: {primary_agent} → {routing_destination}")
            return routing_destination

        # Final fallback: chat_agent
        logger.info("🎯 SIMPLE FALLBACK ROUTING: → chat_agent")
        return "chat_agent"

    except Exception as e:
        logger.error(f"❌ Capability routing error: {e}")
        return "chat_agent"


def route_from_research(state: Dict[str, Any]) -> str:
    """Route based on research agent results."""
    try:
        agent_results = state.get("agent_results", {})

        status = agent_results.get("status", "unknown")

        structured_response = agent_results.get("structured_response", {})
        if structured_response:
            structured_status = structured_response.get("task_status")
            if hasattr(structured_status, "value"):
                structured_status = structured_status.value
            elif structured_status:
                structured_status = str(structured_status).lower()
            if structured_status:
                status = structured_status

        template_requested = state.get("template_id") is not None
        awaiting_confirmation = agent_results.get("awaiting_confirmation", False)
        suggested_template_id = agent_results.get("suggested_template_id")
        template_suggested = awaiting_confirmation and suggested_template_id is not None

        has_research_findings = "research_findings" in state.get("shared_memory", {})
        research_complete_for_template = template_requested and has_research_findings

        routing_recommendation = agent_results.get("routing_recommendation")

        logger.info(
            f"🎯 ROUTING FROM RESEARCH: status={status}, "
            f"template_requested={template_requested}, template_suggested={template_suggested}, "
            f"routing_recommendation={routing_recommendation}"
        )

        # REMOVED: Permission routing - research agent now does comprehensive search directly
        if template_suggested:
            logger.info("📋 Template suggestion detected - returning to user for confirmation")
            return "final_response"
        elif research_complete_for_template:
            logger.info("📋 Routing to Report Agent for template formatting")
            return "template_report"
        elif routing_recommendation == "data_formatting" and status == "complete":
            logger.info(
                "📊 SMART ROUTING: Research complete with formatting request - routing to Data Formatting Agent"
            )
            return "data_formatting_agent"
        # ROOSEVELT'S NATURAL COLLABORATION: Let the LLM handle collaboration decisions
        # No special routing needed - let natural conversation flow handle it
        elif status == "complete":
            return "final_response"
        else:
            return "final_response"

    except Exception as e:
        logger.error(f"❌ Research routing error: {e}")
        return "end"


def route_from_chat(state: Dict[str, Any]) -> str:
    """Route from chat agent - simplified to natural conversation flow"""
    try:
        logger.info("💬 CHAT COMPLETE: Routing to final response for natural conversation flow")
        return "final_response"
        
    except Exception as e:
        logger.error(f"❌ Chat routing error: {e}")
        return "final_response"


def route_from_permission(state: Dict[str, Any]) -> str:
    """Route based on permission response."""
    try:
        permission_granted = state.get("permission_granted", False)
        is_complete = state.get("is_complete", False)
        agent_results = state.get("agent_results", {})
        status = agent_results.get("status", "unknown")
        shared_memory = state.get("shared_memory", {})
        web_permission = shared_memory.get("web_search_permission")

        agent_permission_granted = agent_results.get("permission_granted", False)

        logger.info(
            f"🎯 ROUTING FROM PERMISSION: granted={permission_granted}, complete={is_complete}, status={status}, web_permission={web_permission}"
        )
        logger.info(f"🎯 ROUTING FROM PERMISSION: agent_permission_granted={agent_permission_granted}")
        logger.info(
            f"🎯 ROUTING FROM PERMISSION: agent_results keys: {list(agent_results.keys()) if agent_results else 'None'}"
        )

        if (
            permission_granted
            or agent_permission_granted
            or status == "permission_granted"
            or web_permission is True
            or web_permission == "granted"
        ):
            logger.info("✅ PERMISSION DETECTED: Routing back to research_agent for web search")
            return "research_agent"
        elif is_complete:
            return "final_response"
        else:
            logger.info("❌ NO PERMISSION: Routing to final_response")
            return "final_response"

    except Exception as e:
        logger.error(f"❌ Permission routing error: {e}")
        return "end"


def route_from_rss(state: Dict[str, Any]) -> str:
    """Route based on RSS agent results."""
    try:
        agent_results = state.get("agent_results", {})
        rss_operations = agent_results.get("rss_operations", [])

        has_metadata_required = any(
            op.get("status") == "metadata_required" for op in rss_operations
        )

        all_complete = all(
            op.get("status") in ["success", "error"] for op in rss_operations
        )

        logger.info(
            f"🎯 ROUTING FROM RSS: has_metadata_required={has_metadata_required}, all_complete={all_complete}"
        )

        if has_metadata_required:
            return "metadata_request"
        elif all_complete:
            return "final_response"
        else:
            return "final_response"

    except Exception as e:
        logger.error(f"❌ RSS routing error: {e}")
        return "end"


def route_from_outline(state: Dict[str, Any]) -> str:
    """Route from outline editing agent - check if clarification needed.
    
    ROOSEVELT'S CLARIFICATION FLOW:
    - If agent requests clarification → conversation continues, waits for user input
    - If agent completed edit → final_response
    """
    try:
        agent_results = state.get("agent_results", {})
        
        # Check if clarification is requested
        requires_user_input = agent_results.get("requires_user_input", False)
        task_status = agent_results.get("task_status", "")
        clarification_request = agent_results.get("clarification_request")
        
        logger.info(
            f"🎯 ROUTING FROM OUTLINE: requires_user_input={requires_user_input}, "
            f"task_status={task_status}, has_clarification={clarification_request is not None}"
        )
        
        # If agent requests clarification, conversation continues
        # LangGraph will wait for next user message which will re-enter at intent_classifier
        if requires_user_input or task_status == "incomplete" or clarification_request:
            logger.info("🤔 OUTLINE AGENT: Clarification requested, ending turn to await user response")
            # End this turn - next user message will start fresh workflow
            return "final_response"
        
        # Otherwise complete normally
        return "final_response"
    
    except Exception as e:
        logger.error(f"❌ Outline routing error: {e}")
        return "final_response"


# ROOSEVELT'S NATURAL COLLABORATION: Removed collaboration routing function
# Let agents handle collaboration decisions with full conversation context


