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

from openai import AsyncOpenAI
from orchestrator.models.intent_models import SimpleIntentResult

logger = logging.getLogger(__name__)


class IntentClassifier:
	"""
	Simple Intent Classification Service
	
	Provides semantic routing to 24+ specialized agents based on:
	- Action intent (observation, generation, modification, analysis, query, management)
	- Editor context (fiction, rules, outline, character, podcast, substack, sysml)
	- Pipeline context (active_pipeline_id presence)
	- Conversation intelligence (recent agent activity)
	"""
	
	def __init__(self):
		"""Initialize intent classifier with LLM client"""
		self._openai_client = None
		self._classification_model = None
	
	async def _get_openai_client(self) -> AsyncOpenAI:
		"""Get or create OpenAI client"""
		if self._openai_client is None:
			api_key = os.getenv("OPENROUTER_API_KEY")
			if not api_key:
				raise ValueError("OPENROUTER_API_KEY environment variable not set")
			
			self._openai_client = AsyncOpenAI(
				api_key=api_key,
				base_url="https://openrouter.ai/api/v1"
			)
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
	
	async def classify_intent(
		self, 
		user_message: str, 
		conversation_context: Optional[Dict[str, Any]] = None
	) -> SimpleIntentResult:
		"""
		Classify user intent and route to appropriate agent
		
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
			logger.info(f"🎯 SIMPLE CLASSIFICATION: Processing message: {user_message[:100]}...")
			
			# Build simple prompt
			prompt = self._build_simple_prompt(user_message, conversation_context or {})
			
			# Get LLM classification
			openai_client = await self._get_openai_client()
			classification_model = self._get_classification_model()
			
			response = await openai_client.chat.completions.create(
				messages=[
					{"role": "system", "content": "You are Roosevelt's Simple Intent Classifier. Respond with JSON ONLY (no prose, no markdown fences), matching the required schema strictly."},
					{"role": "user", "content": prompt}
				],
				model=classification_model,
				temperature=0.1,
				max_tokens=500  # Simple responses don't need many tokens
			)
			
			# Extract response content
			response_content = response.choices[0].message.content
			
			# Parse simple JSON response
			return await self._parse_simple_response(response_content, user_message, conversation_context)
			
		except Exception as e:
			logger.error(f"❌ Simple intent classification failed: {e}")
			# Simple fallback
			return self._create_simple_fallback(user_message, conversation_context)
	
	def _build_simple_prompt(self, user_message: str, conversation_context: Dict[str, Any]) -> str:
		"""
		Build lean, focused prompt for intent classification with task-aware biasing
		
		Identical to backend implementation for consistent routing.
		"""
		
		# **CONTEXT-AWARE AGENT FILTERING**: Only show available agents based on page context
		has_active_pipeline = conversation_context.get("shared_memory", {}).get("active_pipeline_id") is not None
		recent_messages = len(conversation_context.get("messages", []))
		
		context_hint = ""
		if has_active_pipeline:
			context_hint = "\n**CONTEXT**: User has an active pipeline in conversation."
		elif recent_messages > 1:
			context_hint = "\n**CONTEXT**: Ongoing conversation with previous messages."
		
		# Bias toward ORG_INBOX if org agent has been active in this conversation
		org_bias_hint = ""
		try:
			ci = conversation_context.get("conversation_intelligence", {}) or {}
			ao = (ci.get("agent_outputs", {}) or {})
			if isinstance(ao, dict) and ao.get("org_inbox_agent"):
				org_bias_hint = "\n**CONTEXT**: The user was recently working with org-mode tasks; prefer ORG_INBOX for task-like statements."
		except Exception:
			pass
		
		# Detect active editor context type
		editor_hint = ""
		try:
			shared = conversation_context.get("shared_memory", {}) or {}
			active_editor = shared.get("active_editor", {}) or {}
			fm = (active_editor.get("frontmatter") or {})
			doc_type = str((fm.get('type') or '')).strip().lower()
			if doc_type == 'fiction':
				editor_hint = "\n\nEDITOR CONTEXT: Active editor contains FICTION manuscript. Prefer fiction_editing_agent."
			elif doc_type == 'rules':
				editor_hint = "\n\nEDITOR CONTEXT: Active editor contains RULES document. Prefer rules_editing_agent. (Do NOT use ARTICLE_ANALYSIS.)"
			elif doc_type == 'character':
				editor_hint = "\n\nEDITOR CONTEXT: Active editor contains CHARACTER profile. Prefer character_development_agent."
			elif doc_type == 'outline':
				editor_hint = "\n\nEDITOR CONTEXT: Active editor contains OUTLINE. Prefer outline_editing_agent. (Do NOT use ARTICLE_ANALYSIS.)"
			elif doc_type == 'style':
				editor_hint = "\n\nEDITOR CONTEXT: Active editor contains STYLE guide. Prefer chat_agent for general edits; DO NOT use ARTICLE_ANALYSIS."
			elif doc_type == 'sysml':
				editor_hint = "\n\nEDITOR CONTEXT: Active editor contains SYSML diagram document. Prefer sysml_agent for system design, UML, and SysML diagram generation/modification."
			elif doc_type == 'podcast':
				editor_hint = "\n\nEDITOR CONTEXT: Active editor contains PODCAST document. Prefer podcast_script_agent when the user requests a podcast script/commentary."
			elif doc_type in ['substack', 'blog']:
				editor_hint = "\n\nEDITOR CONTEXT: Active editor contains SUBSTACK/BLOG document. Prefer substack_agent for article/tweet generation requests, EVEN IF they mention research/historical context (substack_agent has built-in research). Use research_agent ONLY for pure research queries without article generation."
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

**USER MESSAGE**: "{user_message}"{context_hint}{org_bias_hint}{editor_hint}

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

**6. MANAGEMENT** - User wants to ORGANIZE/CONFIGURE/MANAGE system/tasks
   - Language: "Add TODO...", "Mark as done...", "Crawl website...", "Set up...", "Configure..."
   - Intent: System or task management
   - Examples:
     * "Add TODO: Finish chapter 5" → management (task creation)
     * "Crawl this website" → management (data ingestion)
     * "Mark task as complete" → management (status update)

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

**MANAGEMENT AGENTS**:
- **org_inbox_agent**
  - ACTION INTENTS: management (task modifications ONLY)
  - USE FOR: MANAGING inbox.org (add TODO, toggle done, update task, change TODO state)
  - TRIGGERS: "add TODO", "mark done", "toggle", "update task", "change state"
  - AVOID: reading/searching org files (use chat_agent or research_agent)

- **org_project_agent**
  - ACTION INTENTS: management (project creation)
  - USE FOR: Creating new projects with structured metadata
  - TRIGGERS: "start project", "create project", "new initiative", "launch campaign"
  - AVOID: simple tasks (use org_inbox_agent)

- **website_crawler_agent**
  - ACTION INTENTS: management (data ingestion)
  - USE FOR: Recursively crawl and ingest entire websites for future reference
  - TRIGGERS: "crawl website", "crawl this site", "capture website", "ingest site"
  - AVOID: single-page views (use research_agent)
{pipeline_agent_section}
**SPECIALIZED AGENTS**:
- **data_formatting_agent**
  - ACTION INTENTS: modification (data transformation)
  - USE FOR: Format data into tables, CSV, JSON for display/output{data_formatting_note}
  - TRIGGERS: "format as table", "convert to CSV", "create data structure", "show as JSON"{data_formatting_avoid}

- **image_generation_agent**
  - ACTION INTENTS: generation
  - USE FOR: Generate images using DALL-E or other image models
  - TRIGGERS: "create image", "generate picture", "draw", "visualize"

- **wargaming_agent**
  - ACTION INTENTS: analysis, query
  - USE FOR: Military scenario analysis and outcome assessment
  - TRIGGERS: "wargame scenario", "battle outcome", "military analysis"

- **entertainment_agent**
  - ACTION INTENTS: query, analysis
  - USE FOR: Movie/TV information, recommendations, entertainment comparisons
  - TRIGGERS: "tell me about [movie]", "recommend movies like", "find TV shows about", 
              "compare Breaking Bad and The Wire", "what should I watch", "movies similar to"
  - AVOID: non-entertainment queries

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
- If the user requests a project (phrases like 'start project', 'create project', 'new project', 'initiative', 'campaign', 'launch') with no active editor context, route to org_project_agent.

**STRICT OUTPUT FORMAT - JSON ONLY (NO MARKDOWN, NO EXPLANATION):**
You MUST respond with a single JSON object matching this schema:
{{
  "target_agent": "research_agent|chat_agent|help_agent|fiction_editing_agent|rules_editing_agent|outline_editing_agent|character_development_agent|data_formatting_agent|{pipeline_agent_enum}rss_agent|image_generation_agent|wargaming_agent|proofreading_agent|content_analysis_agent|story_analysis_agent|combined_proofread_and_analyze|org_inbox_agent|org_project_agent|website_crawler_agent|podcast_script_agent|substack_agent|entertainment_agent",
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
				
				# **ACTION INTENT ROUTING**: Action intent determines agent, not just editor type
				if doc_type:
					logger.info(f"📝 EDITOR CONTEXT: type={doc_type}, action_intent={action_intent}")
					
					# OBSERVATION intent → editor-specific agent (they can read and respond about content)
					if action_intent == 'observation':
						editor_agent_map = {
							'fiction': 'fiction_editing_agent',
							'rules': 'rules_editing_agent',
							'outline': 'outline_editing_agent',
							'character': 'character_development_agent',
							'sysml': 'sysml_agent',
							'podcast': 'podcast_script_agent',
							'substack': 'substack_agent',
							'blog': 'substack_agent',
						}
						preferred_agent = editor_agent_map.get(doc_type, 'chat_agent')
						data['target_agent'] = preferred_agent
						logger.info(f"👁️ OBSERVATION INTENT: {doc_type} → {preferred_agent} (editor agent can read and respond)")
					
					# ANALYSIS intent → analysis agents
					elif action_intent == 'analysis':
						if doc_type == 'fiction':
							data['target_agent'] = 'story_analysis_agent'
							logger.info(f"📊 ANALYSIS INTENT: Fiction → story_analysis_agent")
						else:
							data['target_agent'] = 'content_analysis_agent'
							logger.info(f"📊 ANALYSIS INTENT: {doc_type} → content_analysis_agent")
					
					# GENERATION/MODIFICATION intent → editor agents
					elif action_intent in ['generation', 'modification']:
						editor_agent_map = {
							'fiction': 'fiction_editing_agent',
							'rules': 'rules_editing_agent',
							'outline': 'outline_editing_agent',
							'character': 'character_development_agent',
							'sysml': 'sysml_agent',
							'podcast': 'podcast_script_agent',
							'substack': 'substack_agent',
							'blog': 'substack_agent',
						}
						preferred_agent = editor_agent_map.get(doc_type)
						if preferred_agent:
							data['target_agent'] = preferred_agent
							logger.info(f"✍️ {action_intent.upper()} INTENT: {doc_type} → {preferred_agent}")
						else:
							# Trust LLM's decision for unknown editor types
							logger.info(f"✍️ {action_intent.upper()} INTENT: Unknown doc_type '{doc_type}', trusting LLM")
					
					# QUERY intent → TRUST LLM DECISION (research_agent vs chat_agent)
					elif action_intent == 'query':
						# **"TRUST THE LLM" DOCTRINE**: 
						# LLM distinguishes between external info queries (research_agent) 
						# and document-context queries (chat_agent)
						logger.info(f"❓ QUERY INTENT: Trusting LLM decision: {data.get('target_agent')} (distinguishes 'Tell me about X' vs 'What's in this doc?')")
						# Don't override - LLM chose correctly based on query semantics
					
					# MANAGEMENT intent → keep LLM's decision (org_inbox, website_crawler, etc.)
					elif action_intent == 'management':
						logger.info(f"⚙️ MANAGEMENT INTENT: Keeping LLM decision: {data.get('target_agent')}")
						# Don't override - LLM chose the right management agent
					
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
							'image_generation_agent', 'wargaming_agent',
							'fact_checking_agent', 'org_inbox_agent', 'org_project_agent',
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
			
			# Create SimpleIntentResult with validation
			result = SimpleIntentResult(**data)
			
			logger.info(f"✅ SIMPLE CLASSIFICATION: → {result.target_agent} (confidence: {result.confidence})")
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
		Create simple fallback classification based on keywords
		
		Identical to backend implementation for consistent routing.
		"""
		
		message_lower = user_message.lower()
		
		# Minimal fallback - only route to wargaming_agent for clear triggers; else prefer editor agent, else chat
		if any(phrase in message_lower for phrase in [
			"i am the us", "i am russia", "i am china", "i am iran", "i am the uk",
			"diplomats", "embassy", "sanctions", "naval", "submarine", "mobilize",
			"airspace", "no-fly", "border incursion", "expelled", "ejected your diplomats",
			"carrier group", "patrols off your coast", "wargame", "war game", "escalate"
		]):
			target_agent = "wargaming_agent"
		else:
			target_agent = "chat_agent"
			try:
				shared = (conversation_context or {}).get('shared_memory', {}) or {}
				ae = shared.get('active_editor', {}) or {}
				fm = (ae.get('frontmatter') or {})
				doc_type = str((fm.get('type') or '')).strip().lower()
				editor_agent_map = {
					'rules': 'rules_editing_agent',
					'outline': 'outline_editing_agent',
					'character': 'character_development_agent',
					'fiction': 'fiction_editing_agent',
				}
				target_agent = editor_agent_map.get(doc_type, target_agent)
			except Exception:
				pass
		
		logger.info(f"🎯 SIMPLE FALLBACK: → {target_agent}")
		
		return SimpleIntentResult(
			target_agent=target_agent,
			action_intent="query",  # Default to query for fallback
			permission_required=False,
			confidence=0.6,
			reasoning="Fallback classification due to parsing error"
		)


# Singleton instance for easy access
_intent_classifier_instance: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
	"""Get or create singleton intent classifier instance"""
	global _intent_classifier_instance
	if _intent_classifier_instance is None:
		_intent_classifier_instance = IntentClassifier()
	return _intent_classifier_instance

