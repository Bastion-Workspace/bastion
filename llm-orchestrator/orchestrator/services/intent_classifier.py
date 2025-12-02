"""
Simple Intent Classification Service

Ported 1:1 from backend/services/simple_intent_service.py for consistent routing.

Classifies user intent (WHAT they want) and action intent (HOW they want to interact),
then routes to the appropriate agent with context awareness.
"""

import json
import logging
import re
import os
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.utils.openrouter_client import get_openrouter_client
from orchestrator.models.intent_models import SimpleIntentResult, IntentClassificationState
from orchestrator.services.agent_capabilities import (
    detect_domain,
    route_within_domain,
    find_best_agent_match
)

logger = logging.getLogger(__name__)


class IntentClassifier:
	"""
	Intent Classification Service using LangGraph workflow
	
	Two-stage classification with capability-based routing:
	- Stage 1: Detect domain (LLM node)
	- Stage 2: Classify action intent (LLM node)
	- Stage 3: Route using capability matching (deterministic node)
	"""
	
	def __init__(self):
		"""Initialize intent classifier with LangGraph workflow"""
		self._openai_client = None
		self._classification_model = None
		self.workflow = self._build_workflow()
	
	async def _get_openai_client(self):
		"""Get or create OpenRouter client with automatic reasoning support"""
		if self._openai_client is None:
			api_key = os.getenv("OPENROUTER_API_KEY")
			if not api_key:
				raise ValueError("OPENROUTER_API_KEY environment variable not set")
			
			# Use OpenRouterClient wrapper for automatic reasoning support
			self._openai_client = get_openrouter_client(api_key=api_key)
		return self._openai_client
	
	def _get_classification_model(self) -> str:
		"""
		Get classification model from environment
		
		Falls back to fast model if not specified.
		Matches backend's settings_service.get_classification_model()
		"""
		if self._classification_model is None:
			# Try environment variable first
			self._classification_model = os.getenv(
				"CLASSIFICATION_MODEL",
				"anthropic/claude-haiku-4.5"  # Fast model default
			)
			logger.info(f"Classification model: {self._classification_model}")
		return self._classification_model
	
	def _build_workflow(self) -> StateGraph:
		"""Build LangGraph workflow for intent classification"""
		workflow = StateGraph(IntentClassificationState)
		
		# Add nodes
		workflow.add_node("detect_domain", self._detect_domain_node)
		workflow.add_node("classify_action_intent", self._classify_action_intent_node)
		workflow.add_node("route_agent", self._route_agent_node)
		
		# Entry point
		workflow.set_entry_point("detect_domain")
		
		# Flow: detect_domain -> classify_action_intent -> route_agent -> END
		workflow.add_edge("detect_domain", "classify_action_intent")
		workflow.add_edge("classify_action_intent", "route_agent")
		workflow.add_edge("route_agent", END)
		
		return workflow.compile()
	
	async def classify_intent(
		self, 
		user_message: str, 
		conversation_context: Optional[Dict[str, Any]] = None
	) -> SimpleIntentResult:
		"""
		Classify intent using LangGraph workflow
		
		Args:
			user_message: The user's query/message
			conversation_context: Context dict with keys:
				- messages: List of conversation messages
				- shared_memory: Dict with active_editor, active_pipeline_id, etc.
				- conversation_intelligence: Dict with agent_outputs, etc.
		
		Returns:
			SimpleIntentResult with target_agent, action_intent, permission, confidence
		"""
		try:
			logger.info(f"🎯 INTENT CLASSIFICATION: Processing message: {user_message[:100]}...")
			
			# Log conversation context for debugging
			context = conversation_context or {}
			shared_memory = context.get('shared_memory', {}) or {}
			primary_agent = shared_memory.get('primary_agent_selected')
			last_agent = shared_memory.get('last_agent')
			
			if primary_agent:
				logger.info(f"📋 CONVERSATION CONTEXT: primary_agent_selected = '{primary_agent}'")
			else:
				logger.info(f"📋 CONVERSATION CONTEXT: primary_agent_selected = None (new conversation or no previous agent)")
			
			if last_agent and last_agent != primary_agent:
				logger.info(f"📋 CONVERSATION CONTEXT: last_agent = '{last_agent}' (different from primary_agent)")
			
			# Initialize state
			initial_state: IntentClassificationState = {
				"user_message": user_message,
				"conversation_context": context,
				"domain": "",
				"action_intent": "",
				"target_agent": "",
				"confidence": 0.0,
				"reasoning": "",
				"permission_required": False,
				"result": None
			}
			
			# Run workflow
			final_state = await self.workflow.ainvoke(initial_state)
			
			# Return result
			return final_state["result"]
			
		except Exception as e:
			logger.error(f"❌ Intent classification failed: {e}")
			# Simple fallback
			return self._create_simple_fallback(user_message, conversation_context)
	
	async def _detect_domain_node(self, state: IntentClassificationState) -> IntentClassificationState:
		"""
		Stage 1: Detect domain using LLM
		
		Uses LLM for semantic understanding of domain, not just keyword matching.
		"""
		try:
			user_message = state["user_message"]
			context = state["conversation_context"]
			
			# Extract editor context
			shared_memory = context.get('shared_memory', {})
			active_editor = shared_memory.get('active_editor', {}) or {}
			editor_context = {}
			if active_editor:
				fm = active_editor.get('frontmatter', {}) or {}
				editor_type = fm.get('type', '').strip().lower()
				if editor_type:
					editor_context = {'type': editor_type}
			
			# Log primary agent for domain detection
			primary_agent = shared_memory.get('primary_agent_selected')
			if primary_agent:
				logger.info(f"🔍 DOMAIN DETECTION: Analyzing with primary_agent = '{primary_agent}' (will consider for continuity)")
			
			# Build domain detection prompt
			prompt = self._build_domain_detection_prompt(user_message, editor_context, context)
			
			# Get LLM classification
			openai_client = await self._get_openai_client()
			classification_model = self._get_classification_model()
			
			response = await openai_client.chat.completions.create(
				messages=[
					{"role": "system", "content": "You are a Domain Classifier. Classify ONLY the domain. Respond with JSON ONLY (no prose, no markdown fences)."},
					{"role": "user", "content": prompt}
				],
				model=classification_model,
				temperature=0.1,
				max_tokens=150
			)
			
			# Parse response
			response_content = response.choices[0].message.content.strip()
			if "```json" in response_content:
				match = re.search(r'```json\s*\n(.*?)\n```', response_content, re.DOTALL)
				if match:
					response_content = match.group(1).strip()
			elif "```" in response_content:
				response_content = response_content.replace("```", "").strip()
			
			data = json.loads(response_content)
			domain = data.get('domain', 'general').lower()
			
			# Validate domain
			valid_domains = ['electronics', 'fiction', 'weather', 'general', 'content', 'management']
			if domain not in valid_domains:
				logger.warning(f"⚠️ Invalid domain '{domain}', defaulting to 'general'")
				domain = 'general'
			
			logger.info(f"📊 STAGE 1 - DOMAIN: {domain}")
			
			state["domain"] = domain
			return state
			
		except Exception as e:
			logger.error(f"❌ Domain detection failed: {e}")
			# Fallback to deterministic detection
			domain = detect_domain(
				query=state["user_message"],
				editor_context=editor_context if editor_context.get('type') else None,
				conversation_history={
					'last_agent': shared_memory.get('last_agent'),
					'primary_agent_selected': shared_memory.get('primary_agent_selected')
				}
			)
			state["domain"] = domain
			return state
	
	async def _classify_action_intent_node(self, state: IntentClassificationState) -> IntentClassificationState:
		"""
		Stage 2: Classify action intent using LLM
		"""
		try:
			user_message = state["user_message"]
			context = state["conversation_context"]
			
			# Build action intent prompt
			prompt = self._build_action_intent_prompt(user_message, context)
			
			# Get LLM classification
			openai_client = await self._get_openai_client()
			classification_model = self._get_classification_model()
			
			response = await openai_client.chat.completions.create(
				messages=[
					{"role": "system", "content": "You are an Action Intent Classifier. Classify ONLY the action intent. Respond with JSON ONLY (no prose, no markdown fences)."},
					{"role": "user", "content": prompt}
				],
				model=classification_model,
				temperature=0.1,
				max_tokens=200
			)
			
			# Parse response
			response_content = response.choices[0].message.content.strip()
			if "```json" in response_content:
				match = re.search(r'```json\s*\n(.*?)\n```', response_content, re.DOTALL)
				if match:
					response_content = match.group(1).strip()
			elif "```" in response_content:
				response_content = response_content.replace("```", "").strip()
			
			data = json.loads(response_content)
			action_intent = data.get('action_intent', 'query').lower()
			
			# Validate
			valid_intents = ['observation', 'generation', 'modification', 'analysis', 'query', 'management']
			if action_intent not in valid_intents:
				logger.warning(f"⚠️ Invalid action_intent '{action_intent}', defaulting to 'query'")
				action_intent = 'query'
			
			logger.info(f"📊 STAGE 2 - ACTION INTENT: {action_intent}")
			
			state["action_intent"] = action_intent
			return state
			
		except Exception as e:
			logger.error(f"❌ Action intent classification failed: {e}")
			state["action_intent"] = 'query'
			return state
	
	async def _route_agent_node(self, state: IntentClassificationState) -> IntentClassificationState:
		"""
		Stage 3: Route to agent using capability matching
		"""
		try:
			domain = state["domain"]
			action_intent = state["action_intent"]
			user_message = state["user_message"]
			context = state["conversation_context"]
			
			# Extract editor context
			shared_memory = context.get('shared_memory', {})
			active_editor = shared_memory.get('active_editor', {}) or {}
			editor_context = {}
			if active_editor:
				fm = active_editor.get('frontmatter', {}) or {}
				editor_type = fm.get('type', '').strip().lower()
				if editor_type:
					editor_context = {'type': editor_type}
			
			# Extract conversation history
			primary_agent = shared_memory.get('primary_agent_selected')
			last_agent = shared_memory.get('last_agent')
			conversation_history = {
				'last_agent': last_agent,
				'primary_agent_selected': primary_agent
			}
			
			# Log primary agent for routing
			if primary_agent:
				logger.info(f"🔄 AGENT ROUTING: Using primary_agent = '{primary_agent}' for capability matching (continuity boost)")
			else:
				logger.info(f"🔄 AGENT ROUTING: No primary_agent (new conversation)")
			
			# Route using capability matching
			target_agent, confidence = find_best_agent_match(
				domain=domain,
				action_intent=action_intent,
				query=user_message,
				editor_context=editor_context if editor_context.get('type') else None,
				conversation_history=conversation_history
			)
			
			logger.info(f"📊 STAGE 3 - ROUTING: {target_agent} (confidence: {confidence:.2f})")
			
			# Check if agent needs web search permission
			agents_requiring_web_search = {
				'research_agent',  # Always needs web search for research
				'site_crawl_agent',  # Always needs web search for crawling
				'substack_agent',  # Has built-in research capability
			}
			
			# Electronics agent may need web search for component research
			# Check if query suggests web search is needed
			electronics_needs_web = (
				target_agent == 'electronics_agent' and
				any(kw in user_message.lower() for kw in [
					'research', 'find', 'look up', 'search for', 'component', 'datasheet',
					'specification', 'price', 'availability', 'where to buy'
				])
			)
			
			# Determine if permission is required
			needs_web_search = (
				target_agent in agents_requiring_web_search or
				electronics_needs_web
			)
			
			# Check if permission already exists in shared_memory
			web_permission_granted = shared_memory.get('web_search_permission', False)
			
			# Set permission_required flag
			# If agent needs web search but permission not granted, require it
			permission_required = needs_web_search and not web_permission_granted
			
			if permission_required:
				logger.info(f"🛡️ PERMISSION CHECK: {target_agent} needs web search, permission not granted → permission_required=True")
			elif needs_web_search and web_permission_granted:
				logger.info(f"✅ PERMISSION CHECK: {target_agent} needs web search, permission already granted")
			
			# Build reasoning
			reasoning = f"Domain: {domain}, Action: {action_intent}, Editor: {editor_context.get('type', 'none')} → {target_agent}"
			if permission_required:
				reasoning += " (web search permission required)"
			
			# Create result
			result = SimpleIntentResult(
				target_agent=target_agent,
				action_intent=action_intent,
				permission_required=permission_required,
				confidence=confidence,
				reasoning=reasoning
			)
			
			state["target_agent"] = target_agent
			state["confidence"] = confidence
			state["reasoning"] = reasoning
			state["result"] = result
			
			return state
			
		except Exception as e:
			logger.error(f"❌ Agent routing failed: {e}")
			# Fallback - check permissions even in error case
			shared_memory = context.get('shared_memory', {}) if 'context' in locals() else {}
			web_permission_granted = shared_memory.get('web_search_permission', False)
			fallback_agent = "chat_agent"
			permission_required = False
			
			# If fallback is to research_agent, check permissions
			if fallback_agent == 'research_agent' and not web_permission_granted:
				permission_required = True
			
			result = SimpleIntentResult(
				target_agent=fallback_agent,
				action_intent=state.get("action_intent", "query"),
				permission_required=permission_required,
				confidence=0.5,
				reasoning=f"Fallback due to error: {str(e)}"
			)
			state["result"] = result
			return state
	
	def _build_domain_detection_prompt(self, user_message: str, editor_context: Dict[str, Any], conversation_context: Dict[str, Any]) -> str:
		"""
		Build prompt for domain detection
		"""
		editor_hint = ""
		if editor_context.get('type'):
			editor_hint = f"\n\n**EDITOR CONTEXT**: User has a '{editor_context['type']}' editor open. This is a STRONG signal for domain detection."
		
		shared_memory = conversation_context.get('shared_memory', {})
		primary_agent = shared_memory.get('primary_agent_selected')
		continuity_hint = ""
		if primary_agent:
			continuity_hint = f"\n\n**CONVERSATION CONTEXT**: Previous agent was '{primary_agent}'. Consider if this is a continuation."
		
		return f"""Classify the DOMAIN of this user message:

**USER MESSAGE**: "{user_message}"{editor_hint}{continuity_hint}

**DOMAIN OPTIONS** (choose ONE):

1. **electronics** - Electronics, circuits, embedded systems, Arduino, ESP32, microcontrollers, sensors, components
2. **fiction** - Fiction writing, stories, manuscripts, chapters, characters, plots, worldbuilding
3. **weather** - Weather conditions, forecasts, temperature, climate
4. **content** - Content creation (articles, podcasts, blog posts)
5. **management** - Task management, project organization, system configuration
6. **general** - General queries, research, information gathering, or unclear domain

**CRITICAL RULES**:
- Editor context is PRIMARY signal - if editor type matches a domain, use that domain
- "our electronics project" or "electronics project" → electronics
- "chapter", "scene", "manuscript" → fiction
- "weather", "temperature", "forecast" → weather
- If unclear, default to "general"

**OUTPUT FORMAT** (JSON ONLY):
{{
  "domain": "electronics|fiction|weather|content|management|general",
  "reasoning": "Brief explanation of domain choice"
}}"""
	
	def _build_action_intent_prompt(self, user_message: str, conversation_context: Dict[str, Any]) -> str:
		"""
		Build focused prompt for action intent classification only
		
		Much simpler than full routing prompt - just classifies HOW user wants to interact.
		"""
		# Get conversation context hints
		shared = conversation_context.get('shared_memory', {}) or {}
		primary_agent = shared.get('primary_agent_selected')
		last_response = shared.get('last_response')
		
		continuity_hint = ""
		if primary_agent and last_response:
			response_preview = last_response[:400] + "..." if len(last_response) > 400 else last_response
			continuity_hint = f"\n\n**CONTEXT**: Previous agent '{primary_agent}' responded: \"{response_preview}\"\nIf this is a continuation/response to that, consider that context."
		
		return f"""Classify the ACTION INTENT of this user message:

**USER MESSAGE**: "{user_message}"{continuity_hint}

**ACTION INTENT OPTIONS** (choose ONE):

1. **observation** - User wants to see/check/review/confirm existing content
   - Examples: "Do you see...", "Show me...", "What's in...", "How is...", "Is there..."
   - Intent: View/confirm what exists, NOT create/modify

2. **generation** - User wants to CREATE/WRITE/DRAFT NEW content
   - Examples: "Write...", "Create...", "Draft...", "Generate...", "Compose..."
   - Intent: Create something new

3. **modification** - User wants to CHANGE/EDIT/REVISE EXISTING content
   - Examples: "Edit...", "Revise...", "Change...", "Improve...", "Update...", "Fix..."
   - Intent: Alter existing content

4. **analysis** - User wants CRITIQUE/FEEDBACK/ASSESSMENT/COMPARISON/SUMMARIZATION
   - Examples: "Analyze...", "Critique...", "Review...", "Compare...", "Summarize...", "Find differences..."
   - Intent: Get feedback, assessment, or summary

5. **query** - User seeks EXTERNAL INFORMATION or FACTS (NOT document analysis)
   - Examples: "Tell me about...", "What is...", "Explain...", "Research...", "Find information about..."
   - Intent: Get information about general topics (NOT specific documents)

6. **management** - User wants to ORGANIZE/CONFIGURE/MANAGE system/tasks/project files
   - Examples: "Add TODO...", "Save...", "Update project files...", "Mark as done...", "Crawl website..."
   - Intent: System, task, or project file management

**CRITICAL RULES**:
- Document-specific queries (mentions specific files/documents) → **analysis** (NOT query)
- "How is X looking?" or "How is X going?" → **observation** (checking status)
- "Save what we discussed" → **management** (project file operation)
- Comparison/contrast queries → **analysis** (NOT query)

**OUTPUT FORMAT** (JSON ONLY):
{{
  "action_intent": "observation|generation|modification|analysis|query|management",
  "reasoning": "Brief explanation of why this action intent was chosen"
}}"""
	
	def _build_simple_prompt(self, user_message: str, conversation_context: Dict[str, Any]) -> str:
		"""
		Build lean, focused prompt for intent classification with task-aware biasing
		
		Identical to backend implementation for consistent routing.
		"""
		
		# **CONTEXT-AWARE AGENT FILTERING**: Only show available agents based on page context
		has_active_pipeline = conversation_context.get("shared_memory", {}).get("active_pipeline_id") is not None
		recent_messages = len(conversation_context.get("messages", []))
		last_agent = conversation_context.get("shared_memory", {}).get("last_agent")

		context_hint = ""
		if has_active_pipeline:
			context_hint = "\n**CONTEXT**: User has an active pipeline in conversation."
		elif recent_messages > 1:
			context_hint = "\n**CONTEXT**: Ongoing conversation with previous messages."
			if last_agent:
				context_hint += f" Last agent used: {last_agent}."

		# **PRIMARY SIGNAL: CONVERSATION CONTINUITY** - This is the MOST IMPORTANT signal for routing
		# Use primary_agent_selected (not last_agent) to avoid utility agents like data_formatter
		# If user is clearly continuing the previous conversation, route back to the same primary agent
		primary_agent = conversation_context.get("shared_memory", {}).get("primary_agent_selected")
		last_response = conversation_context.get("shared_memory", {}).get("last_response")
		
		conversation_continuity_hint = ""
		if primary_agent and last_response and recent_messages > 1:
			# Truncate response for prompt (keep enough context to understand what agent said)
			response_preview = last_response[:800] + "..." if len(last_response) > 800 else last_response
			conversation_continuity_hint = f"""
**🎯 PRIMARY ROUTING SIGNAL: CONVERSATION CONTINUITY**

The '{primary_agent}' agent previously responded with:
"{response_preview}"

**CRITICAL ROUTING RULES**:
1. **CONTINUATION DETECTION**: If the user's message is clearly continuing or responding to the above agent message, route to '{primary_agent}'.
   - Examples of continuations: "Yes, please", "That sounds good", "Can you show me more?", "What about X?", "Save that", "Update it", "How does that work?", "Tell me more", "What's next?"
   - These are responses to the agent's previous message, not new topics.

2. **TOPIC CHANGE DETECTION**: If the user's message is clearly a NEW topic unrelated to the above conversation, route to the appropriate agent for that new topic.
   - Examples of topic changes: "What's the weather?" (after electronics discussion) → weather_agent
   - "Tell me about Napoleon" (after project discussion) → research_agent
   - The user is explicitly changing topics, not continuing the conversation.

3. **SEMANTIC UNDERSTANDING**: Use semantic analysis, not keyword matching:
   - "Save what we discussed" after electronics_agent response → electronics_agent (continuation)
   - "What's the weather?" after electronics_agent response → weather_agent (topic change)
   - "Update the project" after electronics_agent response → electronics_agent (continuation)

**BALANCED ROUTING PRIORITY**:
1. **SEMANTIC INTENT FIRST**: If the query clearly requires a specialized agent (research, weather, etc.), route to that agent even if continuing a conversation.
2. **CONTINUITY SECOND**: Only maintain continuity if the query is clearly continuing the previous topic.
3. **RESEARCH QUERIES**: Future predictions, economic analysis, policy effects, "what would be", "anticipate effects" → research_agent (even if continuing chat conversation)
4. **TOPIC CHANGE**: If clearly a new, unrelated topic → route to appropriate agent

**EXAMPLES**:
- "What do we anticipate the effects of tariff checks would be?" → research_agent (research query, even if continuing chat)
- "What's the weather?" → weather_agent (clear topic change)
- "Yes, please continue" → {primary_agent} (clear continuation)
- "Tell me more about that" → {primary_agent} (clear continuation)
- "What are the implications of X?" → research_agent (research query, switch even if continuing)"""
		elif primary_agent and recent_messages > 1:
			conversation_continuity_hint = f"\n**CONTINUITY**: The primary agent used was '{primary_agent}'. If this appears to be a continuation of the previous conversation, route to '{primary_agent}'."
		
		# **CONVERSATION HISTORY**: Include recent conversation messages for additional context
		conversation_history_hint = ""
		messages = conversation_context.get("messages", [])
		if messages and len(messages) > 1:
			# Include last 3-4 message pairs (6-8 messages) for context
			recent_messages = messages[-8:] if len(messages) > 8 else messages
			conversation_lines = []
			for msg in recent_messages:
				role = msg.get("role", "unknown")
				content = msg.get("content", "")
				# Truncate very long messages to avoid overwhelming the prompt
				if len(content) > 300:
					content = content[:300] + "..."
				if role == "user":
					conversation_lines.append(f"User: {content}")
				elif role == "assistant":
					conversation_lines.append(f"Assistant: {content}")
			
			if conversation_lines:
				conversation_history_hint = f"\n\n**ADDITIONAL CONTEXT - RECENT CONVERSATION HISTORY**:\n" + "\n".join(conversation_lines) + "\n\n(Use this for additional context, but conversation continuity signal above takes priority)"
		
		# Bias toward ORG_INBOX if org agent has been active in this conversation
		org_bias_hint = ""
		try:
			ci = conversation_context.get("conversation_intelligence", {}) or {}
			ao = (ci.get("agent_outputs", {}) or {})
			if isinstance(ao, dict) and ao.get("org_inbox_agent"):
				org_bias_hint = "\n**CONTEXT**: The user was recently working with org-mode tasks; prefer ORG_INBOX for task-like statements."
		except Exception:
			pass
		
		# **CONTEXTUAL SIGNAL: EDITOR CONTEXT** - Provides helpful context but does NOT override conversation continuity
		# User could have editor open but ask about weather - editor is contextual, not primary
		editor_hint = ""
		try:
			shared = conversation_context.get("shared_memory", {}) or {}
			active_editor = shared.get("active_editor", {}) or {}
			fm = (active_editor.get("frontmatter") or {})
			doc_type = str((fm.get('type') or '')).strip().lower()
			if doc_type == 'fiction':
				editor_hint = "\n\n**CONTEXTUAL INFO**: Active editor contains FICTION manuscript. This is helpful context, but conversation continuity (above) takes priority."
			elif doc_type == 'rules':
				editor_hint = "\n\n**CONTEXTUAL INFO**: Active editor contains RULES document. This is helpful context, but conversation continuity (above) takes priority."
			elif doc_type == 'character':
				editor_hint = "\n\n**CONTEXTUAL INFO**: Active editor contains CHARACTER profile. This is helpful context, but conversation continuity (above) takes priority."
			elif doc_type == 'outline':
				editor_hint = "\n\n**CONTEXTUAL INFO**: Active editor contains OUTLINE. This is helpful context, but conversation continuity (above) takes priority."
			elif doc_type == 'style':
				editor_hint = "\n\n**CONTEXTUAL INFO**: Active editor contains STYLE guide. This is helpful context, but conversation continuity (above) takes priority."
			elif doc_type == 'sysml':
				editor_hint = "\n\n**CONTEXTUAL INFO**: Active editor contains SYSML diagram document. This is helpful context, but conversation continuity (above) takes priority."
			elif doc_type == 'podcast':
				editor_hint = "\n\n**CONTEXTUAL INFO**: Active editor contains PODCAST document. This is helpful context, but conversation continuity (above) takes priority."
			elif doc_type in ['substack', 'blog']:
				editor_hint = "\n\n**CONTEXTUAL INFO**: Active editor contains SUBSTACK/BLOG document. This is helpful context, but conversation continuity (above) takes priority."
			elif doc_type == 'electronics':
				editor_hint = "\n\n**CONTEXTUAL INFO**: Active editor contains ELECTRONICS project. This is helpful context for electronics-related queries, but conversation continuity (above) takes priority. If user asks about weather or other unrelated topics, ignore editor context."
		except Exception:
			pass
		
		# **DYNAMIC AGENT LIST**: Build available agents based on context
		# Pipeline agent is ONLY available when on pipelines page
		pipeline_agent_section = ""
		pipeline_agent_enum = ""
		data_formatting_note = ""
		data_formatting_avoid = ""
		
		if has_active_pipeline:
			pipeline_agent_section = """
**DATA PIPELINE AGENTS**:
- **pipeline_agent**
  - ACTION INTENTS: generation, modification
  - USE FOR: Design, create, and modify AWS data pipelines (S3, Glue, Lambda, Redshift, etc.)
  - TRIGGERS: "create pipeline", "AWS pipeline", "S3 pipeline", "ETL pipeline", "data pipeline", 
             "Glue job", "Lambda function", "transform data", "source to sink", 
             "bucket to bucket", "process CSV", "ingest data", "data flow"
  - EXAMPLES:
    * "Create a pipeline from S3 bucket A to bucket B"
    * "Add a transform to convert epoch time"
    * "Build an ETL pipeline for processing CSV files"
    * "Set up a Glue job to transform data"
  - AVOID: Simple data formatting (use data_formatting_agent), table creation without AWS context
"""
			pipeline_agent_enum = "pipeline_agent|"
			data_formatting_note = " (NOT AWS pipelines!)"
			data_formatting_avoid = "\n  - AVOID: AWS pipeline creation (use pipeline_agent)"
		
		return f"""You are Roosevelt's Simple Intent Classifier - quick and decisive routing!

**MISSION**: Classify user intent (WHAT they want) and action intent (HOW they want to interact), then route to the right agent.

**USER MESSAGE**: "{user_message}"{conversation_continuity_hint}{conversation_history_hint}{org_bias_hint}{editor_hint}

**ACTION INTENT CLASSIFICATION (CRITICAL - CLASSIFY FIRST!):**

Every query has a PRIMARY action intent that determines routing behavior:

**1. OBSERVATION** - User wants to see/check/review/confirm existing content
   - Language: "Do you see...", "Show me...", "What's in...", "Does this have...", "Can you see...", "Is there..."
   - Intent: Confirm or view what exists, NOT create/modify
   - Examples:
     * "Do you see our outline" → observation (checking content)
     * "Show me what's in the manuscript" → observation (viewing)
     * "Does this chapter have dialogue?" → observation (checking)

**2. GENERATION** - User wants to CREATE/WRITE/DRAFT NEW content
   - Language: "Write...", "Create...", "Draft...", "Generate...", "Compose...", "Produce..."
   - Intent: Create something new that doesn't exist yet
   - Examples:
     * "Write chapter 3" → generation (creating prose)
     * "Create an outline for this story" → generation (new structure)
     * "Generate a character profile" → generation (new content)

**3. MODIFICATION** - User wants to CHANGE/EDIT/REVISE EXISTING content
   - Language: "Edit...", "Revise...", "Change...", "Improve...", "Rewrite...", "Fix...", "Update..."
   - Intent: Alter existing content
   - Examples:
     * "Edit this scene to add tension" → modification (changing prose)
     * "Revise the outline" → modification (updating structure)
     * "Improve this dialogue" → modification (enhancing existing)

**4. ANALYSIS** - User wants CRITIQUE/FEEDBACK/ASSESSMENT/EVALUATION/COMPARISON/SUMMARIZATION
   - Language: "Analyze...", "Critique...", "Review...", "Evaluate...", "Compare...", "Contrast...", "Summarize...", "Find differences...", "Find similarities...", "What conflicts...", "What contradictions...", "Is this good...", "What do you think..."
   - Intent: Get feedback, assessment, comparison, or summary of specific content
   - **CRITICAL**: Comparison/contrast queries are ALWAYS analysis, NOT query
   - **CRITICAL**: Document-specific summarization (e.g., "summarize file X", "summarize our document Y") is ANALYSIS, NOT query
   - Examples:
     * "Analyze the plot structure" → analysis (critique)
     * "Compare our worldcom documents" → analysis (document comparison)
     * "Summarize our file called ebberss504" → analysis (document-specific summarization)
     * "Summarize document named X" → analysis (specific document summary)
     * "Find conflicts in company policies" → analysis (multi-document comparison)
     * "What are the differences between these rules?" → analysis (contrast)
     * "Is this character compelling?" → analysis (assessment)
     * "Review this chapter" → analysis (evaluation)

**5. QUERY** - User seeks EXTERNAL INFORMATION or FACTS (NOT document analysis)
   - Language: "Tell me about...", "What is...", "Explain...", "Search for...", "Find information about...", "Research..."
   - Intent: Get information or facts from knowledge base/web (GENERAL topics, NOT specific documents)
   - **CRITICAL**: Query is for INFORMATION GATHERING about general topics, not document comparison/analysis
   - **CRITICAL**: If query mentions specific document/file names, it's ANALYSIS (use content_analysis_agent), NOT query
   - Examples:
     * "Tell me about the Enron scandal" → query (external information, general topic)
     * "What is three-act structure?" → query (definition, general concept)
     * "Research quantum computing" → query (information gathering, broad topic)
     * "Find information about Napoleon" → query (fact finding, historical figure)
   - COUNTER-EXAMPLES (these are ANALYSIS, NOT query):
     * "Summarize our file called ebberss504" → analysis (specific document)
     * "Tell me about document X in our database" → analysis (specific document)
     * "What's in the worldcom files" → analysis (specific document set)

**6. MANAGEMENT** - User wants to ORGANIZE/CONFIGURE/MANAGE system/tasks/project files
   - Language: "Add TODO...", "Mark as done...", "Crawl website...", "Set up...", "Configure...", "Save...", "Save what we discussed...", "Update project files..."
   - Intent: System, task, or project file management
   - **CRITICAL DISTINCTION**:
     * **Org-mode task management**: Managing inbox.org tasks (TODO lists) → org_inbox_agent or org_project_agent
     * **Project file management**: Saving/updating project files (electronics, fiction, etc.) → editor-specific agent (electronics_agent, fiction_editing_agent, etc.)
     * **System management**: Website crawling, configuration → website_crawler_agent or appropriate system agent
   - Examples:
     * "Add TODO: Finish chapter 5" → management (org-mode task) → org_inbox_agent
     * "Save what we discussed on this electronics project" → management (project file) → electronics_agent (if electronics editor)
     * "Update project files" → management (project file) → editor agent matching editor type
     * "Crawl this website" → management (system/data ingestion) → website_crawler_agent
     * "Mark task as complete" → management (org-mode task) → org_inbox_agent

**CRITICAL ROUTING RULES USING ACTION INTENT**:

1. **SEMANTIC ROUTING**: Don't rely on pattern matching! Understand the action intent FIRST, then route based on agent capabilities.

2. **DOCUMENT-SPECIFIC = ANALYSIS (NEVER QUERY)**:
   - "Compare documents" → analysis intent → content_analysis_agent
   - "Summarize our file called X" → analysis intent → content_analysis_agent
   - "Summarize document Y" → analysis intent → content_analysis_agent
   - "Find conflicts" → analysis intent → content_analysis_agent
   - "What are the differences" → analysis intent → content_analysis_agent
   - "Similarities between X and Y" → analysis intent → content_analysis_agent
   - **ANY query referencing specific documents/files by name** → analysis intent → content_analysis_agent
   - **NEVER route document-specific queries to research_agent!**

3. **EDITOR CONTEXT + ACTION INTENT**: Editor type provides CONTEXT, but ACTION INTENT determines routing:
   - observation + fiction editor → fiction_editing_agent (agent reads editor and responds)
   - generation + fiction editor → fiction_editing_agent (creating prose)
   - modification + fiction editor → fiction_editing_agent (editing prose)
   - analysis + fiction editor → story_analysis_agent or content_analysis_agent (critique)
   - query + ANY editor → research_agent OR chat_agent (semantic: external info vs document questions)

**ALL AVAILABLE AGENTS WITH ACTION INTENT COMPATIBILITY**:

**FICTION WRITING AGENTS** (type: fiction):
- **fiction_editing_agent**
  - ACTION INTENTS: observation, generation, modification
  - USE FOR: Creating/editing fiction prose, viewing/checking editor content, writing chapters, drafting scenes
  - TRIGGERS: "write chapter", "edit scene", "do you see", "show me this", "draft opening", "revise dialogue"
  - AVOID: pure analysis requests (use story_analysis_agent)

- **story_analysis_agent**
  - ACTION INTENTS: analysis
  - USE FOR: Critiquing fiction manuscripts, analyzing plot/character/pacing/themes
  - TRIGGERS: "analyze plot", "critique character", "review pacing", "is this compelling"
  - AVOID: generation requests, observation queries

- **content_analysis_agent**
  - ACTION INTENTS: analysis
  - USE FOR: Content critique, document comparison, document-specific summarization, finding conflicts/differences/similarities
  - **CRITICAL**: Document-specific queries (mentions specific file/document names) should route here for analysis
  - TRIGGERS: 
    * "analyze content", "compare documents", "find conflicts", "find differences", "contrast these", "similarities between"
    * "summarize our file called X", "summarize document Y", "summarize ebberss504", "analyze file named Z"
    * "our file called...", "document named...", "the document...", "file X..."
  - AVOID: generation requests, observation queries, general information queries (use research_agent)

- **proofreading_agent**
  - ACTION INTENTS: modification (minimal corrections only)
  - USE FOR: Grammar/spelling/style corrections aligned to style guide
  - TRIGGERS: "proofread", "check grammar", "fix typos", "style corrections"
  - AVOID: plot changes, generation, analysis

**CREATIVE DEVELOPMENT AGENTS**:
- **outline_editing_agent**
  - ACTION INTENTS: observation, generation, modification
  - USE FOR: Create/refine story outlines, view outline content
  - TRIGGERS: "create outline", "expand outline", "do you see", "show me outline", "refine structure"
  - AVOID: analysis requests (use content_analysis_agent)

- **rules_editing_agent**
  - ACTION INTENTS: observation, generation, modification
  - USE FOR: Create/expand world-building rules, view rules content, detect contradictions
  - TRIGGERS: "create rules", "do you see rules", "define magic system", "establish canon", "expand rules"
  - AVOID: analysis requests (use content_analysis_agent)

- **character_development_agent**
  - ACTION INTENTS: observation, generation, modification
  - USE FOR: Create/develop character profiles, view character content, backstory, motivations, arcs
  - TRIGGERS: "create character", "show me character", "develop protagonist", "character profile"
  - AVOID: analysis requests (use content_analysis_agent)

**CONTENT GENERATION AGENTS**:
- **podcast_script_agent**
  - ACTION INTENTS: observation, generation, modification
  - USE FOR: Generate TTS-ready podcast scripts, view podcast content, audio cues (type: podcast editors)
  - TRIGGERS: "create podcast", "generate script", "do you see script", "write podcast episode"
  - AVOID: general podcast questions without editor context

- **substack_agent**
  - ACTION INTENTS: observation, generation, modification
  - USE FOR: Generate long-form articles OR tweet-sized posts, view article content (type: substack/blog editors)
  - HAS BUILT-IN RESEARCH: Can automatically research topics and incorporate findings
  - TRIGGERS: "write article", "create post", "do you see article", "generate tweet", "article about X"
  - AVOID: pure research queries without article generation context

**RESEARCH & INFORMATION AGENTS**:
- **research_agent**
  - ACTION INTENTS: query (information gathering)
  - USE FOR: Standalone research and information gathering about GENERAL TOPICS (NOT specific documents)
  - HAS org-mode search tools for searching ALL user's .org files
  - TRIGGERS: "research X" (general topic), "find information about Y" (general), "tell me about Z" (general concept), "search for facts", "what is X" (definition)
  - **CRITICAL**: Use for BROAD/GENERAL topics, NOT specific document analysis
  - AVOID: 
    * Document-specific queries (use content_analysis_agent): "summarize file X", "our document called Y"
    * Document comparison (use content_analysis_agent): "compare documents", "find conflicts"
    * Article generation (use substack_agent)
    * Observation queries (use chat_agent)

- **chat_agent**
  - ACTION INTENTS: observation, query
  - USE FOR: Conversational queries, observation checks, quick lookups, general questions
  - CAN READ: org-mode TODOs (list_org_todos), search by tag (search_org_by_tag)
  - TRIGGERS: "do you see", "show me", "what's in", "explain"
  - USE AS FALLBACK: When no specialized agent fits

- **help_agent**
  - ACTION INTENTS: query
  - USE FOR: Application help, UI navigation instructions, feature documentation, system capabilities overview
  - TRIGGERS: "how do I", "how can I", "help with", "what is [feature]", "how does [agent] work", 
             "show me how to", "guide for", "instructions for", "what can I do", 
             "what agents are available", "available features", "how to use", "getting started"
  - EXAMPLES:
    * "How do I send a message to another user?" → help_agent (UI navigation)
    * "How does the research agent work?" → help_agent (agent documentation)
    * "What can this application do?" → help_agent (feature discovery)
    * "Help with RSS feeds" → help_agent (feature guidance)
    * "What agents are available?" → help_agent (system capabilities)
    * "How do I create a TODO?" → help_agent (feature instructions)
  - AVOID: General knowledge questions (use chat_agent), research queries (use research_agent)

**MANAGEMENT AGENTS** (ORG-MODE TASK MANAGEMENT ONLY):
- **org_inbox_agent**
  - ACTION INTENTS: management (org-mode task modifications ONLY)
  - USE FOR: MANAGING inbox.org (add TODO, toggle done, update task, change TODO state)
  - TRIGGERS: "add TODO", "mark done", "toggle", "update task", "change state"
  - **CRITICAL**: ONLY for org-mode task management (inbox.org), NOT for project file management
  - **AVOID**: 
    * Project file operations (use editor agents: electronics_agent, fiction_editing_agent, etc.)
    * Saving/updating project files (use editor agents)
    * Technical project planning (use electronics_agent, etc.)
    * Reading/searching org files (use chat_agent or research_agent)

- **org_project_agent**
  - ACTION INTENTS: management (org-mode project entry creation ONLY)
  - USE FOR: Creating new org-mode project entries for task management (NOT technical project design/planning)
  - TRIGGERS: "start project", "create project", "new initiative", "launch campaign" (ONLY for org-mode task/project management, NOT technical design)
  - **CRITICAL**: This is for ORG-MODE PROJECT MANAGEMENT (creating org-mode entries), not technical project planning/design
  - **AVOID**: 
    * Technical project planning (use electronics_agent, research_agent, or chat_agent): "design a circuit project", "plan an electronic project", "formulate a new electronic project"
    * Software project planning (use chat_agent or research_agent): "create a software project", "plan a coding project"
    * Project file operations (use editor agents: electronics_agent, etc.)
    * Saving/updating project files (use editor agents)
    * Simple tasks (use org_inbox_agent)

- **website_crawler_agent**
  - ACTION INTENTS: management (data ingestion)
  - USE FOR: Recursively crawl and ingest entire websites for future reference
  - TRIGGERS: "crawl website", "crawl this site", "capture website", "ingest site"
  - AVOID: single-page views (use research_agent)
{pipeline_agent_section}
**SPECIALIZED AGENTS**:
- **email_agent**
  - ACTION INTENTS: generation (email composition)
  - USE FOR: Drafting and sending emails with user approval, context-aware email composition
  - TRIGGERS: "send an email to", "email to", "compose email", "draft email", "send email", "email the research to", "send the findings to"
  - EXAMPLES:
    * "Send an email to john@example.com about the meeting" → email_agent
    * "Email the research on X to bob@company.com" → email_agent
    * "Compose an email to sarah@example.com" → email_agent
    * "Send the research findings to my manager" → email_agent
  - AVOID: General email questions (use chat_agent), email configuration (use help_agent)

- **data_formatting_agent**
  - ACTION INTENTS: modification (data transformation)
  - USE FOR: Format data into tables, CSV, JSON for display/output{data_formatting_note}
  - TRIGGERS: "format as table", "convert to CSV", "create data structure", "show as JSON"{data_formatting_avoid}

- **image_generation_agent**
  - ACTION INTENTS: generation
  - USE FOR: Generate images using DALL-E or other image models
  - TRIGGERS: "create image", "generate picture", "draw", "visualize"

# WargamingAgent removed - not fully fleshed out

- **entertainment_agent**
  - ACTION INTENTS: query, analysis
  - USE FOR: Movie/TV information, recommendations, entertainment comparisons
  - TRIGGERS: "tell me about [movie]", "recommend movies like", "find TV shows about",
              "compare Breaking Bad and The Wire", "what should I watch", "movies similar to"
  - AVOID: non-entertainment queries

- **electronics_agent**
  - ACTION INTENTS: generation, query, modification, analysis, **management** (project file operations)
  - USE FOR: Circuit design, embedded programming, component selection, electronics troubleshooting, electronics project planning/design, **saving/updating electronics project files**
  - TRIGGERS: "design circuit", "Arduino code", "ESP32 firmware", "component selection",
              "voltage divider", "resistor calculator", "PCB layout", "microcontroller programming",
              "embedded system", "circuit analysis", "calculate resistor values", "motor driver",
              "sensor integration", "power supply design", "electronic project", "electronics project",
              "plan electronic", "formulate electronic project", "design electronic system", "electro-magnetic control",
              **"save project", "save what we discussed", "update project files", "save the system", "save the design"**
  - **CRITICAL**: Use for technical electronics project planning/design AND project file management, NOT org-mode project management
  - **MANAGEMENT ACTIONS**: When user wants to save/update electronics project files → electronics_agent (NOT org_inbox_agent or org_project_agent)
  - AVOID: non-electronics technical queries, org-mode task management (use org_inbox_agent/org_project_agent)

**ORG-MODE ROUTING RULES (CRITICAL - READ CAREFULLY!)**:
- **org_inbox_agent**: ONLY for MODIFICATION actions on inbox.org
  - "Add TODO: Buy milk" → org_inbox_agent
  - "Mark task 3 as done" → org_inbox_agent
  - "Toggle TODO state" → org_inbox_agent
  - "Update my task about X" → org_inbox_agent
- **chat_agent**: For READING/LISTING org data (quick casual queries)
  - "What's on my TODO list?" → chat_agent (uses list_org_todos)
  - "What's tagged @work?" → chat_agent (uses search_org_by_tag)
  - "Show me my tasks" → chat_agent
- **research_agent**: For SEARCHING org files for reference/research
  - "Find my notes about quarterly report" → research_agent (uses search_org_files)
  - "Search my org files for project X" → research_agent
  - "What did I write about Y in my org files?" → research_agent
  - "Show me all WAITING tasks related to budget" → research_agent (complex filtered search)

ROUTING HINTS FOR PROJECT CAPTURE:
- **ORG-MODE PROJECT MANAGEMENT**: If the user requests creating a project entry for task management (phrases like 'start project', 'create project', 'new initiative', 'launch campaign') with NO technical context, route to org_project_agent.
- **TECHNICAL PROJECT PLANNING**: If the user mentions "project" in a technical context (electronics, software, engineering, etc.), route to the appropriate specialized agent:
  * "electronic project", "electronics project", "plan electronic", "formulate electronic project" → electronics_agent
  * "software project", "coding project", "programming project" → chat_agent or research_agent
  * "engineering project" → research_agent or chat_agent
- **KEY DISTINCTION**: org_project_agent is for TASK/PROJECT MANAGEMENT (creating org-mode entries), NOT for technical project design/planning.

**STRICT OUTPUT FORMAT - JSON ONLY (NO MARKDOWN, NO EXPLANATION):**
You MUST respond with a single JSON object matching this schema:
{{
  "target_agent": "research_agent|chat_agent|help_agent|fiction_editing_agent|rules_editing_agent|outline_editing_agent|character_development_agent|data_formatting_agent|{pipeline_agent_enum}rss_agent|image_generation_agent|proofreading_agent|content_analysis_agent|story_analysis_agent|combined_proofread_and_analyze|org_inbox_agent|org_project_agent|website_crawler_agent|podcast_script_agent|substack_agent|entertainment_agent|electronics_agent",
  "action_intent": "observation|generation|modification|analysis|query|management",
  "permission_required": false,
  "confidence": 0.0,
  "reasoning": "Brief explanation of routing decision based on action intent"
}}

**CRITICAL REQUIREMENTS**:
1. **CLASSIFY ACTION INTENT FIRST**: Determine observation/generation/modification/analysis/query/management
2. **ROUTE BASED ON ACTION INTENT + EDITOR CONTEXT**: 
   - observation + fiction editor → fiction_editing_agent (agent reads editor and responds)
   - generation + fiction editor → fiction_editing_agent (creating prose)
   - modification + fiction editor → fiction_editing_agent (editing prose)
   - analysis + fiction editor → story_analysis_agent (critique)
   - query + fiction editor → research_agent OR chat_agent (semantic choice: external info vs document questions)
   - (Similar logic for outline/rules/character/podcast/substack editors)
3. **RESPECT ACTION INTENT WITH EDITOR AWARENESS**: 
   - "Do you see our outline" + fiction editor → fiction_editing_agent (observation intent, uses editor context)
   - "Write chapter 3" + fiction editor → fiction_editing_agent (generation intent)
   - "Compare our worldcom documents" → content_analysis_agent (analysis intent, document comparison)
   - "Tell me about Enron" + fiction editor → research_agent (query intent, external information)
   - "Find conflicts in policies" → content_analysis_agent (analysis intent, comparison)
4. **DOCUMENT-SPECIFIC = ANALYSIS**: Any query referencing specific documents/files by name is ANALYSIS intent → content_analysis_agent
   - "Summarize our file called X" → analysis
   - "Compare these documents" → analysis  
   - "What's in document Y" → analysis
   - "Analyze file Z" → analysis
5. **GENERAL TOPICS = QUERY**: Broad information requests about concepts/people/events (NOT specific documents) are QUERY intent → research_agent
   - "Tell me about the Enron scandal" → query (general topic)
   - "What is quantum computing" → query (concept definition)
   - "Research Napoleon Bonaparte" → query (historical figure)
   - "What do we anticipate the effects of X would be?" → query (future prediction/research)
   - "What are the implications of Y?" → query (analysis/research)
   - "How will Z affect the economy?" → query (economic analysis/research)
   - "What would happen if..." → query (hypothetical analysis/research)
6. **NO BRITTLE PATTERN MATCHING**: Use semantic understanding, not keyword matching
7. **ALWAYS INCLUDE reasoning**: Explain why this action_intent was chosen
8. **Return JSON ONLY**: No code fences, no markdown, no extra text
"""
	
	async def _parse_simple_response(self, response: str, user_message: str, conversation_context: Optional[Dict[str, Any]]) -> SimpleIntentResult:
		"""
		Parse simple JSON response with robust error handling
		
		Identical to backend implementation for consistent routing.
		"""
		try:
			# Clean response
			response_content = response.strip()
			if "```json" in response_content:
				match = re.search(r'```json\s*\n(.*?)\n```', response_content, re.DOTALL)
				if match:
					response_content = match.group(1).strip()
			elif "```" in response_content:
				response_content = response_content.replace("```", "").strip()
			
			# Parse JSON
			data = json.loads(response_content)
			
			# Extract action_intent with fallback
			action_intent = data.get('action_intent', 'query').lower()
			
			# Validate action_intent
			valid_action_intents = ['observation', 'generation', 'modification', 'analysis', 'query', 'management']
			if action_intent not in valid_action_intents:
				logger.warning(f"⚠️ Invalid action_intent '{action_intent}', defaulting to 'query'")
				action_intent = 'query'
				data['action_intent'] = 'query'
			
			logger.info(f"🎯 ACTION INTENT: {action_intent}")
			
			# Derive defaults and enforce ACTION INTENT + EDITOR routing
			try:
				shared = (conversation_context or {}).get('shared_memory', {}) if conversation_context else {}
			except Exception:
				shared = {}
			try:
				active_editor = shared.get('active_editor', {}) or {}
				fm = (active_editor.get('frontmatter') or {})
				doc_type = str((fm.get('type') or '')).strip().lower()
				
				# Get primary_agent for conversation continuity check
				primary_agent = shared.get('primary_agent_selected')
				
				# **EDITOR CONTEXT OVERRIDE**: Editor type is PRIMARY signal for domain-specific routing
				# If user has a specialized editor open, route to that agent (with exceptions for analysis)
				# This fixes cases like "How is our electronics project looking?" → should go to electronics_agent
				current_agent = data.get('target_agent', 'chat_agent')
				
				if doc_type == 'electronics':
					# STRICT GATING: Only route to electronics_agent if project_plan.md is open
					filename = active_editor.get('filename', '').lower()
					
					# Check if this is actually a project_plan file
					if 'project_plan' in filename or filename == 'project_plan.md':
						# Valid electronics project plan open → route electronics queries
						query_lower = user_message.lower()
						electronics_keywords = ['electronics', 'circuit', 'arduino', 'esp32', 'project', 'component', 'sensor', 'microcontroller', 'firmware', 'embedded', 'voltage', 'resistor', 'pcb', 'schematic', 'design', 'our', 'the project']
						non_electronics_keywords = ['weather', 'temperature', 'forecast', 'rain', 'snow']
						
						# Check if query is clearly NOT electronics-related
						if any(kw in query_lower for kw in non_electronics_keywords):
							# Non-electronics query, don't override routing
							logger.info(f"📝 Electronics project open, but query is non-electronics")
						elif any(kw in query_lower for kw in electronics_keywords):
							# Electronics query with project open → electronics_agent
							if current_agent != 'electronics_agent':
								logger.info(f"🔄 EDITOR GATING: {current_agent} → electronics_agent (project_plan.md open)")
								data['target_agent'] = 'electronics_agent'
								data['reasoning'] = f"Electronics project plan open. Routing to electronics_agent."
						else:
							# Ambiguous query with electronics project open → electronics_agent
							logger.info(f"🔄 EDITOR GATING: Ambiguous query with electronics project → electronics_agent")
							data['target_agent'] = 'electronics_agent'
							data['reasoning'] = f"Electronics project plan open. Routing to electronics_agent."
					else:
						# Electronics document open, but NOT project_plan → don't route to electronics_agent
						logger.info(f"📝 Electronics doc open ({filename}), but not project_plan - no electronics_agent override")
						
						# Check if informational query → research_agent
						from orchestrator.services.agent_capabilities import is_information_lookup_query
						if is_information_lookup_query(user_message):
							logger.info(f"🔍 Informational query detected → research_agent")
							data['target_agent'] = 'research_agent'
							data['reasoning'] = f"Electronics document open but not project_plan. Informational query → research_agent."
						# Otherwise let natural domain classification decide (likely chat_agent)
				
				elif doc_type == 'fiction':
					# Fiction queries go to fiction_editing_agent (unless analysis)
					if action_intent == 'analysis':
						if current_agent not in ['story_analysis_agent', 'content_analysis_agent']:
							logger.info(f"🔄 EDITOR OVERRIDE: {current_agent} → story_analysis_agent (editor type: fiction, action: analysis)")
							data['target_agent'] = 'story_analysis_agent'
							data['reasoning'] = f"Editor context (fiction) + analysis intent → story_analysis_agent"
					else:
						# Non-analysis fiction queries → fiction_editing_agent
						if current_agent not in ['fiction_editing_agent', 'outline_editing_agent', 'character_development_agent', 'rules_editing_agent']:
							logger.info(f"🔄 EDITOR OVERRIDE: {current_agent} → fiction_editing_agent (editor type: fiction, action: {action_intent})")
							data['target_agent'] = 'fiction_editing_agent'
							data['reasoning'] = f"Editor context (fiction) + {action_intent} intent → fiction_editing_agent"
				
				elif doc_type in ['outline', 'character', 'rules']:
					# Specialized fiction editors → their specific agents
					editor_agent_map = {
						'outline': 'outline_editing_agent',
						'character': 'character_development_agent',
						'rules': 'rules_editing_agent'
					}
					target_agent = editor_agent_map.get(doc_type)
					if target_agent and current_agent != target_agent and action_intent != 'analysis':
						logger.info(f"🔄 EDITOR OVERRIDE: {current_agent} → {target_agent} (editor type: {doc_type})")
						data['target_agent'] = target_agent
						data['reasoning'] = f"Editor context ({doc_type}) → {target_agent}"
				
				if doc_type:
					logger.info(f"📝 EDITOR CONTEXT: type={doc_type}, action_intent={action_intent}, primary_agent={primary_agent}, final_agent={data.get('target_agent')}")
				else:
					logger.info(f"✅ NO EDITOR CONTEXT: → {data.get('target_agent')}")
					
					
			except Exception as e:
				logger.error(f"❌ Action intent routing failed: {e}")
				pass
			
			# **PIPELINE AGENT ISOLATION**: Strict page-based access control
			try:
				active_pipeline_id = shared.get('active_pipeline_id')
				pipeline_preference = shared.get('pipeline_preference', 'prefer').lower()
				current_agent = data.get('target_agent', 'chat_agent')
				
				# **CRITICAL**: Pipeline agent is ONLY accessible from pipelines page
				# If NOT on pipelines page (no active_pipeline_id), BLOCK pipeline agent
				if not active_pipeline_id and current_agent == 'pipeline_agent':
					logger.info(f"🛑 ROOSEVELT: Blocking pipeline_agent - not on pipelines page (no active_pipeline_id)")
					logger.info(f"🔄 ROOSEVELT: Redirecting pipeline_agent → research_agent (general query handling)")
					data['target_agent'] = 'research_agent'  # Redirect to research agent for general queries
					current_agent = 'research_agent'
				
				# If pipeline preference is enabled and there's an active pipeline
				if pipeline_preference == 'prefer' and active_pipeline_id:
					# Check if query is pipeline-related
					query_lower = user_message.lower()
					pipeline_keywords = [
						'pipeline', 'node', 'nodes', 'version', 'versions',
						'compile', 'execute', 'run', 'flow', 'etl', 'elt',
						'lambda', 'glue', 's3', 'bucket', 'redshift', 'athena',
						'transform', 'source', 'sink', 'kinesis', 'emr',
						'batch', 'streaming', 'ingestion', 'processing',
						'this', 'it', 'what', 'how', 'upgrade', 'update',
						'deprecated', 'status', 'data', 'add', 'modify',
						'change', 'convert', 'epoch', 'csv', 'json', 'parquet'
					]
					
					has_pipeline_context = any(kw in query_lower for kw in pipeline_keywords)
					
					# If query could be pipeline-related, prefer pipeline agent
					if has_pipeline_context:
						# But don't override clear intents for other things
						non_pipeline_agents = [
							'weather_agent', 'research_agent', 'rss_agent',
							'image_generation_agent',  # WargamingAgent removed - not fully fleshed out
							# FactCheckingAgent removed - not actively used
							'org_inbox_agent', 'org_project_agent',
							'fiction_editing_agent', 'story_analysis_agent',
							'content_analysis_agent', 'proofreading_agent'
						]
						
						# Only prefer pipeline agent if not clearly something else
						if current_agent not in non_pipeline_agents:
							logger.info(f"🔧 PIPELINE PREFERENCE: {current_agent} → pipeline_agent (active_pipeline: {active_pipeline_id})")
							data['target_agent'] = 'pipeline_agent'
			except Exception as e:
				logger.error(f"Pipeline preference logic failed: {e}")
				pass
			
			# Check permissions after parsing (override LLM's permission_required if needed)
			shared = (conversation_context or {}).get('shared_memory', {}) if conversation_context else {}
			target_agent = data.get('target_agent', 'chat_agent')
			
			# Agents that always need web search
			agents_requiring_web_search = {
				'research_agent',
				'site_crawl_agent',
				'substack_agent',
			}
			
			# Electronics agent may need web search
			electronics_needs_web = (
				target_agent == 'electronics_agent' and
				any(kw in user_message.lower() for kw in [
					'research', 'find', 'look up', 'search for', 'component', 'datasheet',
					'specification', 'price', 'availability', 'where to buy'
				])
			)
			
			needs_web_search = (
				target_agent in agents_requiring_web_search or
				electronics_needs_web
			)
			
			web_permission_granted = shared.get('web_search_permission', False)
			
			# Override permission_required based on actual permission state
			if needs_web_search:
				data['permission_required'] = not web_permission_granted
				if data['permission_required']:
					logger.info(f"🛡️ PERMISSION OVERRIDE: {target_agent} needs web search, permission not granted")
			
			# Create SimpleIntentResult with validation
			result = SimpleIntentResult(**data)
			
			logger.info(f"✅ SIMPLE CLASSIFICATION: → {result.target_agent} (confidence: {result.confidence}, permission_required: {result.permission_required})")
			return result
			
		except json.JSONDecodeError as e:
			logger.error(f"❌ JSON parsing failed: {e}")
			logger.error(f"❌ Raw response: {response_content}")
			return self._create_simple_fallback(user_message, conversation_context)
		except Exception as e:
			logger.error(f"❌ Simple parsing failed: {e}")
			return self._create_simple_fallback(user_message, conversation_context)
	
	def _create_simple_fallback(self, user_message: str, conversation_context: Optional[Dict[str, Any]] = None) -> SimpleIntentResult:
		"""
		Create fallback classification using two-stage approach
		
		Uses deterministic domain detection and simple keyword-based action intent.
		"""
		try:
			context = conversation_context or {}
			shared_memory = context.get('shared_memory', {})
			active_editor = shared_memory.get('active_editor', {}) or {}
			editor_context = {}
			if active_editor:
				fm = active_editor.get('frontmatter', {}) or {}
				editor_context = {'type': fm.get('type', '').strip().lower()}
			
			conversation_history = {
				'last_agent': shared_memory.get('last_agent'),
				'primary_agent_selected': shared_memory.get('primary_agent_selected')
			}
			
			# Stage 1: Detect domain
			domain = detect_domain(
				query=user_message,
				editor_context=editor_context if editor_context.get('type') else None,
				conversation_history=conversation_history
			)
			
			# Stage 2: Simple keyword-based action intent
			message_lower = user_message.lower()
			if any(kw in message_lower for kw in ['write', 'create', 'draft', 'generate', 'compose']):
				action_intent = 'generation'
			elif any(kw in message_lower for kw in ['edit', 'revise', 'change', 'improve', 'update', 'fix']):
				action_intent = 'modification'
			elif any(kw in message_lower for kw in ['analyze', 'critique', 'review', 'compare', 'summarize']):
				action_intent = 'analysis'
			elif any(kw in message_lower for kw in ['add todo', 'save', 'mark', 'crawl', 'update project']):
				action_intent = 'management'
			elif any(kw in message_lower for kw in ['do you see', 'show me', "what's in", 'how is', 'is there']):
				action_intent = 'observation'
			else:
				action_intent = 'query'
			
			# Stage 3: Route
			target_agent, confidence = find_best_agent_match(
				domain=domain,
				action_intent=action_intent,
				query=user_message,
				editor_context=editor_context if editor_context.get('type') else None,
				conversation_history=conversation_history
			)
			
			# Check if agent needs web search permission (same logic as main routing)
			agents_requiring_web_search = {
				'research_agent',
				'site_crawl_agent',
				'substack_agent',
			}
			
			electronics_needs_web = (
				target_agent == 'electronics_agent' and
				any(kw in user_message.lower() for kw in [
					'research', 'find', 'look up', 'search for', 'component', 'datasheet',
					'specification', 'price', 'availability', 'where to buy'
				])
			)
			
			needs_web_search = (
				target_agent in agents_requiring_web_search or
				electronics_needs_web
			)
			
			web_permission_granted = shared_memory.get('web_search_permission', False)
			permission_required = needs_web_search and not web_permission_granted
			
			logger.info(f"🎯 FALLBACK CLASSIFICATION: → {target_agent} (domain: {domain}, action: {action_intent})")
			if permission_required:
				logger.info(f"🛡️ FALLBACK PERMISSION CHECK: {target_agent} needs web search, permission not granted")
			
			return SimpleIntentResult(
				target_agent=target_agent,
				action_intent=action_intent,
				permission_required=permission_required,
				confidence=max(confidence, 0.5),  # Minimum 0.5 for fallback
				reasoning=f"Fallback classification: Domain={domain}, Action={action_intent}"
			)
		except Exception as e:
			logger.error(f"❌ Fallback classification failed: {e}")
			# Ultimate fallback
			return SimpleIntentResult(
				target_agent="chat_agent",
				action_intent="query",
				permission_required=False,
				confidence=0.3,
				reasoning="Ultimate fallback due to error"
			)


# Singleton instance for easy access
_intent_classifier_instance: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
	"""Get or create singleton intent classifier instance"""
	global _intent_classifier_instance
	if _intent_classifier_instance is None:
		_intent_classifier_instance = IntentClassifier()
	return _intent_classifier_instance

