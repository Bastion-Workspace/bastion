"""
Orchestrator Node Implementations - Roosevelt's Modular Command
Extracted node logic with stable signatures for delegation.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


async def intent_classifier_node(ctx, state: Dict[str, Any]) -> Dict[str, Any]:
	"""
	**ROOSEVELT'S CAPABILITY-BASED INTENT CLASSIFICATION NODE**
	
	Enhanced with dynamic agent discovery and structured routing decisions
	"""
	try:
		logger.info("🎯 CAPABILITY-BASED INTENT CLASSIFICATION: Processing with live agent discovery...")

		# Process with the enhanced intent classification agent
		# **ROOSEVELT'S "TRUST THE LLM" DOCTRINE - BYPASS BRITTLE PATTERN MATCHING**
		# BYPASSED: Hardcoded fiction editor override (too brittle, misroutes queries)
		# The intent classifier LLM handles fiction editing vs analysis routing intelligently
		# based on action_intent (modification/generation → fiction_editing_agent, analysis → content_analysis_agent)
		# 
		# TO REVERT: Uncomment the section below and remove this bypass comment
		# 
		# WARGAMING OVERRIDE: Still active (specific enough to warrant pattern matching)
		try:
			sm = state.get("shared_memory", {}) or {}
			# WARGAMING OVERRIDE: If a wargame is active, route outcome/damage queries to wargaming_agent
			try:
				if isinstance(sm.get("wargaming_state"), dict):
					user_query = ""
					msgs = state.get("messages", []) or []
					for msg in reversed(msgs):
						if hasattr(msg, "type") and msg.type == "human":
							user_query = str(msg.content or "")
							break
						if isinstance(msg, dict) and msg.get("role") == "user":
							user_query = str(msg.get("content") or "")
							break
					uq = user_query.lower()
					wargame_outcome_triggers = [
						"outcome", "result", "effects", "effect",
						"damage", "casualties", "bda", "battle damage",
						"assessment", "impact", "what happened",
						"after the attack", "how many died", "how many casualties",
						"how much damage", "estimate"
					]
					if any(t in uq for t in wargame_outcome_triggers):
						logger.info("🪖 WARGAMING OVERRIDE: Routing outcome/damage query to wargaming_agent")
						updated_state = { **state }
						updated_state["intent_classification"] = {
							"target_agent": "wargaming_agent",
							"confidence": 0.98,
							"permission_required": False
						}
						return updated_state
			except Exception:
				pass
		except Exception:
			pass
		
		# BYPASSED FICTION OVERRIDE CODE (preserved for easy revert):
		# Uncomment below to restore hardcoded fiction routing
		"""
		try:
			sm = state.get("shared_memory", {}) or {}
			pref = (sm.get("editor_preference") or '').lower()
			if pref == 'ignore':
				raise Exception('Editor preference = ignore; skipping editor override')
			ae = sm.get("active_editor", {}) or {}
			if ae.get("is_editable") and isinstance(ae.get("filename"), str):
				if ae["filename"].lower().endswith('.md') and ((ae.get("frontmatter", {}) or {}).get("type", "").lower() == 'fiction'):
					# Decide between story analysis vs editing based on simple keyword triggers (can be improved)
					user_query = ""
					msgs = state.get("messages", []) or []
					for msg in reversed(msgs):
						if hasattr(msg, "type") and msg.type == "human":
							user_query = str(msg.content or "")
							break
						if isinstance(msg, dict) and msg.get("role") == "user":
							user_query = str(msg.get("content") or "")
							break

					edit_triggers = [
						# Core editing verbs
						"revise", "rewrite", "edit ", "tighten", "improve prose", "polish",
						# Generation verbs
						"generate", "write", "draft", "compose", "create",
						# Modification verbs (ROOSEVELT: Catch "Can we end..." type queries)
						"add", "insert", "include", "put in", "place",
						"change", "modify", "update", "alter", "adjust",
						"end with", "start with", "begin with", "finish with", "conclude with",
						"end chapter", "start chapter", "open chapter", "close chapter",
						# Expansion verbs
						"expand", "extend", "lengthen", "flesh out", "beef up",
						"more detail", "more dialogue", "more description", "more tension",
						# Scene/structure operations
						"continue scene", "new chapter", "expand scene", "insert paragraph",
						"write chapter", "new scene", "add scene"
					]
					if any(t in user_query.lower() for t in edit_triggers):
						logger.info("✍️ FICTION OVERRIDE: Routing to fiction_editing_agent for manuscript operations")
						updated_state = { **state }
						updated_state["intent_classification"] = {
							"target_agent": "fiction_editing_agent",
							"confidence": 0.95,
							"permission_required": False
						}
						return updated_state
					else:
						# Check if this is a combined proofread+analyze request
						user_message = ""
						messages = state.get("messages", [])
						for msg in reversed(messages):
							if hasattr(msg, "type") and msg.type == "human":
								user_message = str(msg.content).lower()
								break
							
						# Only override to story_analysis if it's NOT a combined request
						if "proofread" in user_message and "analyze" in user_message:
							logger.info("📚 COMBINED REQUEST: Fiction markdown with proofread+analyze - letting intent classifier decide")
							# Let the intent classifier handle the combined request
						else:
							logger.info("📚 STORY OVERRIDE: Active editor is fiction markdown - routing to story_analysis")
							updated_state = { **state }
							updated_state["intent_classification"] = {
								"target_agent": "content_analysis_agent",
								"confidence": 0.99,
								"permission_required": False
							}
							return updated_state
		except Exception:
			pass
		"""

		updated_state = await ctx.intent_classification_agent.process(state)
		
		# Extract simple intent classification results
		intent_classification = updated_state.get("intent_classification", {})
		
		# Generate conversation title if needed
		latest_message = ctx._get_latest_user_message(state) if hasattr(ctx, "_get_latest_user_message") else ""
		if not state.get("conversation_title") and latest_message and len(latest_message.strip()) > 0:
			try:
				title = await ctx._generate_conversation_title(latest_message)
				updated_state["conversation_title"] = title
				logger.info(f"✅ Generated conversation title: '{title}'")
			except Exception as e:
				logger.warning(f"⚠️ Title generation failed: {e}")
				fallback_title = latest_message[:60] + "..." if len(latest_message) > 60 else latest_message
				updated_state["conversation_title"] = fallback_title

		# Extract routing information from simple classification
		target_agent = intent_classification.get("target_agent", "chat_agent")
		confidence = intent_classification.get("confidence", 0.0)
		
		logger.info(f"🎯 SIMPLE CLASSIFICATION: → {target_agent}")
		logger.info(f"🎯 Routing confidence: {confidence:.2f}")
		logger.info("🎯 ROOSEVELT'S LEAN COMMAND CENTER: Simple classification complete")

		# Ensure intent_classification is properly set for routing
		updated_state["intent_classification"] = intent_classification
		
		return updated_state
	
	except Exception as e:
		logger.error(f"❌ Intent classification agent error: {e}")
		# Return error state
		return {
			**state,
			"intent_classification": {
				"target_agent": "chat_agent",
				"confidence": 0.3,
				"permission_required": False
			}
		}


async def agent_node(ctx, state: Dict[str, Any], agent_attr: str, response_field: str) -> Dict[str, Any]:
	"""Generic agent node runner with conversation intelligence management."""
	try:
		logger.info(f"🤖 AGENT NODE: Processing with {agent_attr}...")
		
		# Convert state for agent processing
		agent_state = ctx._convert_to_agent_state(state, agent_attr)
		agent = getattr(ctx, agent_attr)
		
		# Execute agent processing
		result_state = await agent.process(agent_state) if hasattr(agent, "process") else await agent._process_request(agent_state)
		
		# Extract results
		agent_results = result_state.get("agent_results", {})
		response = result_state.get("latest_response", "") or agent_results.get("response", "")
		
		# ROOSEVELT'S CONVERSATION INTELLIGENCE: Update intelligence with agent output
		if response and len(response.strip()) > 0:
			try:
				from services.conversation_intelligence_service import get_conversation_intelligence_service
				intel_service = await get_conversation_intelligence_service()
				
				# Update conversation intelligence with this agent's output
				updated_state = await intel_service.analyze_and_update_intelligence(
					state=state,
					agent_type=agent_attr,
					agent_output=response,
					agent_results=agent_results
				)
				
				# Merge intelligence back into result
				result_state["conversation_intelligence"] = updated_state.get("conversation_intelligence", {})
				logger.info(f"✅ INTELLIGENCE UPDATED: {agent_attr} output added to conversation cache")
				
			except Exception as intel_error:
				logger.warning(f"⚠️ Intelligence update failed for {agent_attr}: {intel_error}")
				# Don't fail agent execution for intelligence issues
		
		# === UNIVERSAL AGENT LOCK HANDLING ===
		# Allow agents to request conversation-level routing lock/unlock
		shared_memory_out = result_state.get("shared_memory", {}) or {}
		try:
			# Direct signal: agent_results.locked_agent = "wargaming" | None to clear
			locked_agent_signal = agent_results.get("locked_agent") if isinstance(agent_results, dict) else None
			# Structured signal: agent_results.conversation_controls.locked_agent
			conversation_controls = agent_results.get("conversation_controls", {}) if isinstance(agent_results, dict) else {}
			locked_agent_struct = conversation_controls.get("locked_agent") if isinstance(conversation_controls, dict) else None
			# Unlock flag: agent_results.unlock_agent = True clears lock
			unlock_flag = bool(agent_results.get("unlock_agent")) if isinstance(agent_results, dict) else False

			# Choose precedence: explicit unlock > explicit struct > direct
			if unlock_flag:
				if "locked_agent" in shared_memory_out:
					try:
						del shared_memory_out["locked_agent"]
					except Exception:
						shared_memory_out["locked_agent"] = None
				logger.info("🔓 AGENT LOCK: Unlock requested by agent; clearing locked_agent from shared_memory")
			else:
				desired_lock = locked_agent_struct if (isinstance(locked_agent_struct, str) and locked_agent_struct.strip()) else (
					locked_agent_signal if (isinstance(locked_agent_signal, str) and locked_agent_signal.strip()) else None
				)
				if desired_lock:
					shared_memory_out["locked_agent"] = desired_lock.strip()
					logger.info(f"🔒 AGENT LOCK: Agent requested lock to '{desired_lock.strip()}'")
		except Exception as _lock_err:
			# Never fail the node for lock handling issues
			pass

		return {
			"agent_results": agent_results,
			response_field: response,
			"latest_response": response,
			"shared_memory": shared_memory_out,
			"conversation_intelligence": result_state.get("conversation_intelligence", {}),
		}
		
	except Exception as e:
		logger.error(f"❌ Agent node error ({agent_attr}): {e}")
		return {"agent_results": {"status": "error", "response": f"{agent_attr} error: {str(e)}"}}


async def weather_agent_node(ctx, state: Dict[str, Any]) -> Dict[str, Any]:
	try:
		logger.info("🌤️ WEATHER AGENT: Starting meteorological intelligence operation...")
		if ctx.weather_agent is None:
			from services.langgraph_agents.weather_agent import WeatherAgent
			ctx.weather_agent = WeatherAgent()
		agent_state = ctx._convert_to_agent_state(state, "weather_agent")
		result_state = await ctx.weather_agent._process_request(agent_state)
		agent_results = result_state.get("agent_results", {})
		response = result_state.get("response", "")
		return {
			"agent_results": agent_results,
			"weather_response": response,
			"latest_response": response,
			"shared_memory": result_state.get("shared_memory", {}),
		}
	except Exception as e:
		logger.error(f"❌ Weather agent error: {e}")
		return {"agent_results": {"task_status": "error", "response": f"Weather error: {str(e)}"}}


async def rss_metadata_request_node(ctx, state: Dict[str, Any]) -> Dict[str, Any]:
	try:
		logger.info(f"🛑 RSS METADATA REQUEST: Processing metadata request...")
		messages = state.get("messages", [])
		latest_user_message = None
		for msg in reversed(messages):
			if hasattr(msg, 'content') and getattr(msg, 'type', '') == 'human':
				latest_user_message = msg.content
				break
		agent_results = state.get("agent_results", {})
		rss_operations = agent_results.get("rss_operations", [])
		metadata_operations = [op for op in rss_operations if op.get("status") == "metadata_required"]
		if not metadata_operations:
			return {"metadata_provided": True}
		if latest_user_message:
			metadata_provided = await ctx._parse_rss_metadata_response(latest_user_message, metadata_operations)
			if metadata_provided:
				return {"metadata_provided": True}
		request_message = ctx._build_rss_metadata_request(metadata_operations)
		shared_memory = state.get("shared_memory", {})
		shared_memory["rss_metadata_request"] = {
			"pending_operations": metadata_operations,
			"request_message": request_message,
		}
		return {"metadata_provided": False, "shared_memory": shared_memory, "latest_response": request_message}
	except Exception as e:
		logger.error(f"❌ RSS metadata request error: {e}")
		return {"metadata_provided": False}


async def web_search_permission_node(ctx, state: Dict[str, Any]) -> Dict[str, Any]:
	try:
		logger.info("🛑 WEB SEARCH PERMISSION NODE: Processing user response (resumption)...")
		messages = state.get("messages", [])
		shared_memory = state.get("shared_memory", {})
		latest_user_message = None
		for msg in reversed(messages):
			if hasattr(msg, 'type') and msg.type == 'human':
				latest_user_message = msg.content.lower().strip()
				break
		approval_keywords = ["yes", "y", "ok", "proceed", "approved", "continue", "go ahead"]
		permission_granted = (latest_user_message and any(k in latest_user_message for k in approval_keywords))
		if permission_granted:
			shared_memory["web_search_permission"] = True
			return {"shared_memory": shared_memory, "agent_results": {"permission_granted": True}}
		else:
			from langchain_core.messages import AIMessage
			return {
				"messages": messages + [AIMessage(content="I understand. I'll work with the available local information.")],
				"shared_memory": shared_memory,
				"is_complete": True,
			}
	except Exception as e:
		logger.error(f"❌ WEB SEARCH PERMISSION ERROR: {e}")
		return state


# ROOSEVELT'S NATURAL COLLABORATION: Removed collaboration suggestion and handler nodes
# Let agents handle collaboration decisions with full conversation context


async def final_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
	try:
		logger.info("📋 FINAL RESPONSE: Assembling final response...")
		latest_response = state.get("latest_response", "") or state.get("chat_agent_response", "")
		agent_results = state.get("agent_results", {})
		logger.info(f"📋 FINAL RESPONSE: Agent results keys: {list(agent_results.keys()) if isinstance(agent_results, dict) else 'Not a dict'}")
		
		# **ROOSEVELT: PRESERVE EDITOR OPERATIONS FOR HITL EDITING**
		editor_operations = None
		manuscript_edit = None
		if isinstance(agent_results, dict):
			editor_operations = agent_results.get("editor_operations")
			manuscript_edit = agent_results.get("manuscript_edit")
			if editor_operations:
				logger.info(f"📋 FINAL RESPONSE: Preserving {len(editor_operations)} editor operations")
		
		# **ROOSEVELT'S CITATION CAVALRY: Extract citations from agent results!**
		citations = []
		if isinstance(agent_results, dict):
			citations = agent_results.get("citations", [])
			if citations:
				logger.info(f"📋 FINAL RESPONSE: Found {len(citations)} citations from agent")
		
		agent_response = ""
		if isinstance(agent_results, dict):
			structured_response = agent_results.get("structured_response", {})
			logger.info(f"📋 FINAL RESPONSE: Structured response keys: {list(structured_response.keys()) if isinstance(structured_response, dict) else 'Not a dict'}")
			if isinstance(structured_response, dict):
				agent_response = structured_response.get("findings", "") or structured_response.get("response", "") or structured_response.get("script_text", "")
				logger.info(f"📋 FINAL RESPONSE: Extracted agent response length: {len(agent_response)}")
				
				# **ROOSEVELT'S DOUBLE-CHECK**: Also check structured_response for citations (some agents put them here)
				if not citations:
					structured_citations = structured_response.get("citations", [])
					if structured_citations:
						citations = structured_citations
						logger.info(f"📋 FINAL RESPONSE: Found {len(citations)} citations from structured_response")
		
		response_content = agent_response or latest_response
		
		if response_content:
			from langchain_core.messages import AIMessage
			msgs = state.get("messages", [])
			
			# **ROOSEVELT'S CITATION INCLUSION: Add citations to AIMessage metadata!**
			additional_kwargs = {}
			if citations:
				additional_kwargs["citations"] = citations
				logger.info(f"✅ FINAL RESPONSE: Including {len(citations)} citations in AIMessage metadata")
			
			result = {
				"messages": msgs + [AIMessage(
					content=response_content,
					additional_kwargs=additional_kwargs
				)], 
				"latest_response": response_content, 
				"is_complete": True,
				# **ROOSEVELT'S AGENT RESULTS PRESERVATION: Keep agent_results for streaming endpoint!**
				"agent_results": agent_results if isinstance(agent_results, dict) else {}
			}
			# **ROOSEVELT: ADD EDITOR OPERATIONS TO STATE FOR STREAMING**
			if editor_operations:
				result["editor_operations"] = editor_operations
				result["manuscript_edit"] = manuscript_edit
				logger.info(f"📋 FINAL RESPONSE: Added editor_operations to final state")
			
			# **ROOSEVELT'S CITATION DEBUGGING**: Log what we're returning
			if citations:
				logger.info(f"📋 FINAL RESPONSE: Returning agent_results with {len(citations)} citations")
			
			return result
		else:
			from langchain_core.messages import AIMessage
			fallback_msg = "I apologize, but I couldn't generate a proper response."
			return {"messages": state.get("messages", []) + [AIMessage(content=fallback_msg)], "latest_response": fallback_msg, "is_complete": True}
	except Exception as e:
		logger.error(f"❌ Final response error: {e}")
		from langchain_core.messages import AIMessage
		error_msg = f"Final response error: {str(e)}"
		return {"messages": state.get("messages", []) + [AIMessage(content=error_msg)], "latest_response": error_msg, "is_complete": True}


async def update_metadata_node(ctx, state: Dict[str, Any]) -> Dict[str, Any]:
	try:
		updates = {}
		messages = state.get("messages", [])
		user_message = None
		for msg in messages:
			if hasattr(msg, 'type') and msg.type == "human":
				user_message = msg.content
				break
		updates["conversation_updated_at"] = datetime.now().isoformat()
		if not state.get("conversation_title") and user_message and len(user_message.strip()) > 0:
			title = await ctx._generate_conversation_title(user_message)
			updates["conversation_title"] = title
			if not state.get("conversation_created_at"):
				updates["conversation_created_at"] = datetime.now().isoformat()
			if not state.get("conversation_tags"):
				updates["conversation_tags"] = []
			if not state.get("conversation_description"):
				updates["conversation_description"] = None
			if not state.get("is_pinned"):
				updates["is_pinned"] = False
			if not state.get("is_archived"):
				updates["is_archived"] = False
		if user_message and len(user_message.strip()) > 0:
			topic = user_message[:50] + "..." if len(user_message) > 50 else user_message
			updates["conversation_topic"] = topic.strip()
		return updates
	except Exception as e:
		logger.warning(f"⚠️ METADATA NODE: Failed to update conversation metadata: {e}")
		return {}


