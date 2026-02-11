"""
Agent Capabilities and Domain Detection

Provides domain detection and capability-based routing for intent classification.
Keeps routing logic separate from LLM classification to maintain lean architecture.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


# Agent capability declarations
AGENT_CAPABILITIES = {
    'electronics_agent': {
        'domains': ['electronics', 'circuit', 'embedded', 'arduino', 'esp32', 'microcontroller'],
        'actions': ['observation', 'generation', 'modification', 'analysis', 'query', 'management'],
        'editor_types': ['electronics'],
        'keywords': ['electronics', 'circuit', 'arduino', 'esp32', 'raspberry pi', 'microcontroller', 
                     'sensor', 'resistor', 'voltage', 'pcb', 'schematic', 'firmware', 'embedded'],
        'context_boost': 20  # Strong preference when editor matches
        # Note: Can handle queries without editor, but route_within_domain enforces editor for editing actions
    },
    # fiction_editing_agent removed; fiction only via writing_assistant_agent → fiction_editing_subgraph
    'story_analysis_agent': {
        'domains': ['fiction', 'writing', 'story'],
        'actions': ['analysis'],
        'editor_types': ['fiction'],
        'keywords': ['analyze', 'critique', 'review', 'pacing', 'structure', 'themes'],
        'context_boost': 15,
        'requires_explicit_keywords': True  # Must have explicit analysis keywords to route here
    },
    'writing_assistant_agent': {
        'domains': ['fiction', 'writing', 'outline', 'character', 'rules', 'style', 'series', 'content'],
        'actions': ['observation', 'generation', 'modification'],
        'editor_types': ['outline', 'rules', 'style', 'character', 'nfoutline', 'article', 'substack', 'blog', 'fiction'],  # outline + article/substack/blog via subgraphs
        'keywords': ['outline', 'structure', 'act', 'plot points', 'character', 'rules', 'style', 'series', 'build an outline', 'non-fiction outline', 'article', 'blog', 'substack'],
        'context_boost': 20,
        'requires_editor': True  # Must have active editor - no keyword bypass for editing agents
    },
    # Standalone agents removed; only via writing_assistant_agent → subgraphs: outline, character, rules, style, article
    'series_editing_agent': {
        'domains': ['fiction', 'writing', 'series'],
        'actions': ['observation', 'generation', 'modification'],
        'editor_types': ['series'],
        'keywords': ['series', 'synopsis', 'book status', 'series continuity', 'future books'],
        'context_boost': 20,
        'requires_editor': True  # Must have active editor - no keyword bypass for editing agents
    },
    'weather_agent': {
        'domains': ['weather', 'forecast', 'climate'],
        'actions': ['query', 'observation'],
        'editor_types': [],
        'keywords': ['weather', 'temperature', 'forecast', 'rain', 'snow', 'sunny', 'cloudy'],
        'context_boost': 0
    },
    'email_agent': {
        'domains': ['email', 'inbox', 'mail'],
        'actions': ['query', 'observation', 'generation', 'modification'],
        'editor_types': [],
        'keywords': [
            'email', 'inbox', 'emails', 'mail', 'message', 'messages',
            'send email', 'reply to', 'read my email', 'check email', 'search email',
            'unread', 'draft', 'compose', 'forward'
        ],
        'context_boost': 0
    },
    'navigation_agent': {
        'domains': ['navigation', 'locations', 'routes', 'maps'],
        'actions': ['observation', 'query', 'management'],
        'editor_types': [],
        'keywords': [
            'create location', 'save location', 'add location', 'new location at',
            'list locations', 'show my locations', 'saved locations', 'my locations',
            'delete location', 'remove location',
            'route from', 'route to', 'directions', 'how do i get', 'navigate',
            'drive from', 'walk from', 'map', 'turn by turn', 'save route'
        ],
        'context_boost': 0
    },
    'research_agent': {
        'domains': ['general', 'research', 'information', 'management'],
        'actions': ['query', 'analysis'],
        'editor_types': [],
        'keywords': [
            'research', 'find information', 'tell me about', 'what is',
            'anticipate', 'predict', 'forecast', 'effects', 'impact', 'consequences',
            'would be', 'will be', 'might be', 'could be', 'likely to',
            'analyze', 'analysis', 'explain', 'describe', 'investigate',
            'what are', 'what were', 'how will', 'how did', 'why did',
            'economic', 'policy', 'legislation', 'regulation', 'tariff', 'tax',
            # Information lookup patterns
            'how can i', 'how do i', 'how to', 'how would i',
            'what is the procedure', 'what are the steps', 'what is the process',
            'where can i find', 'where do i find', 'where is',
            'tell me how', 'show me how', 'explain how',
            'instructions for', 'manual for', 'guide for', 'tutorial for',
            'change the', 'set the', 'adjust the', 'configure the',
            # Image/picture search patterns
            'find me a picture', 'find me a pic', 'find me an image', 'find me a photo',
            'find me pictures', 'find me images', 'find me photos',
            'show me a picture', 'show me a pic', 'show me an image', 'show me a photo',
            'show me pictures', 'show me images', 'show me photos',
            'get me a picture', 'get me an image', 'get me a photo',
            'do some research', 'do research', 'can you research', 'please research',
            # Aggregate-from-my-docs: local search + stitch (journals, entries, graph my X)
            'from the journals', 'from my journals', 'from my documents', 'from the documents',
            'graph my', 'chart my', 'from my entries', 'journal entries', 'my journals',
            'from my notes', 'from the last month', 'from the last week', 'weight loss', 'track my'
        ],
        'context_boost': 0,
        'override_continuity': True  # Explicit research requests override conversation continuity
    },
    'content_analysis_agent': {
        'domains': ['general', 'analysis', 'documents'],
        'actions': ['analysis'],
        'editor_types': ['article', 'blog', 'substack', 'nfoutline', 'reference', 'document'],
        'keywords': ['compare', 'summarize', 'analyze', 'find differences', 'find conflicts'],
        'context_boost': 0
    },
    'chat_agent': {
        'domains': ['general'],
        'actions': ['observation', 'query'],
        'editor_types': [],
        'keywords': [],
        'context_boost': 0
    },
    'image_generation_agent': {
        'domains': ['general', 'image', 'visual', 'art'],
        'actions': ['generation'],
        'editor_types': [],
        'keywords': [
            'create image', 'generate image', 'generate picture', 'draw', 'visualize',
            'image', 'picture', 'photo', 'photography', 'create a picture',
            'make an image', 'generate a photo', 'create a photo', 'draw a picture',
            'create an image', 'make a picture', 'generate an image', 'create picture'
        ],
        'context_boost': 0
    },
    'image_description_agent': {
        'domains': ['general', 'image', 'vision'],
        'actions': ['observation', 'query'],
        'editor_types': [],
        'keywords': [
            'describe this image', 'describe the image', 'what is in this image',
            'what\'s in this image', 'what does this image show', 'describe this picture',
            'what do you see', 'analyze this image', 'what is this image', 'caption this',
            'describe this photo', 'what\'s in the image', 'tell me about this image'
        ],
        'context_boost': 0,
        'requires_image_context': True
    },
    'site_crawl_agent': {
        'domains': ['research', 'web', 'information'],
        'actions': ['query'],
        'editor_types': [],
        'keywords': ['crawl site', 'crawl website', 'site crawl', 'domain crawl', 'crawl domain'],
        'context_boost': 0
    },
    'security_analysis_agent': {
        'domains': ['general', 'security', 'web'],
        'actions': ['analysis', 'query'],
        'editor_types': [],
        'keywords': [
            'security scan', 'vulnerability scan', 'security analysis', 'check for vulnerabilities',
            'security audit', 'pen test', 'security assessment', 'exposed files', 'security headers',
            'scan for vulnerabilities', 'security check', 'website security'
        ],
        'context_boost': 0
    },
    'general_project_agent': {
        'domains': ['general', 'management'],
        'actions': ['observation', 'generation', 'modification', 'analysis', 'management'],
        'editor_types': ['project'],
        'keywords': ['project plan', 'scope', 'timeline', 'requirements', 'tasks', 'project', 'planning', 'design', 'specification'],
        'context_boost': 15,  # Moderate boost when project editor is active
        'requires_editor': True  # Only route when project editor is active (unless explicit keywords)
    },
    'help_agent': {
        'domains': ['general', 'help', 'documentation'],
        'actions': ['query'],
        'editor_types': [],
        'keywords': [
            'how do i', 'how can i', 'help with', 'what is', 'how does', 'how to',
            'show me how', 'guide for', 'instructions for', 'what can i do',
            'what agents are available', 'available features', 'getting started',
            'help', 'documentation', 'tutorial', 'user guide', 'feature guide'
        ],
        'context_boost': 0
    },
    'reference_agent': {
        'domains': ['general', 'reference', 'journal', 'log'],
        'actions': ['query', 'analysis', 'observation', 'generation'],  # generation for visualization requests
        'editor_types': ['reference'],
        'keywords': [
            'journal', 'log', 'record', 'tracking', 'diary', 'food log', 'weight log', 'mood log', 
            'graph', 'chart', 'visualize',
            # Calculation keywords
            'calculate', 'calculation', 'compute', 'math', 'formula', 'btu', 'heat loss', 'heat losses', 
            'manual j', 'hvac', 'electrical', 'ohms law', 'convert units', 'unit conversion'
        ],
        'context_boost': 20  # Strong preference when editor type matches
    },
    'knowledge_builder_agent': {
        'domains': ['research', 'knowledge', 'information', 'truth'],
        'actions': ['query', 'analysis'],
        'editor_types': [],
        'keywords': [
            'distill', 'distill knowledge', 'build knowledge',
            'compile knowledge', 'research document', 'investigate',
            'find the truth', 'verify claims', 'fact check',
            'cross-reference', 'truth investigation'
        ],
        'context_boost': 0,
        'requires_explicit_keywords': False
    },
    'org_content_agent': {
        'domains': ['general'],
        'actions': ['query', 'observation'],  # READ-ONLY: no modification/management
        'editor_types': ['org'],
        'keywords': [
            'org', 'org-mode', 'orgmode', 'todo', 'task', 
            'project', 'what', 'show', 'list', 'find',
            'tagged', 'tag', 'org file'
        ],
        'context_boost': 15,  # Higher boost for org file context
        'requires_editor': False  # Can answer without editor if query references org files
    },
    'org_capture_agent': {
        'domains': ['general', 'management'],
        'actions': ['generation', 'management'],
        'editor_types': [],
        'keywords': [
            'capture', 'inbox', 'capture to inbox', 'for my inbox', 'add to inbox', 'quick capture'
        ],
        'context_boost': 0,
        'requires_editor': False
    },
    # technical_hyperspace_agent removed - implementation plan in dev-notes/TECHNICAL_HYPERSPACE_IMPLEMENTATION.md
    # article/substack/blog only via writing_assistant_agent → article_writing_subgraph
    'podcast_script_agent': {
        'domains': ['content', 'writing', 'podcast'],
        'actions': ['observation', 'generation', 'modification'],
        'editor_types': ['podcast'],
        'keywords': ['podcast', 'script', 'episode', 'show notes'],
        'context_boost': 20,
        'requires_editor': True
    }
}


def detect_domain(
    query: str, 
    editor_context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[Dict[str, Any]] = None
) -> str:
    """
    Stage 1: Detect primary domain from query and context
    
    Priority:
    1. Editor type (strongest signal)
    2. Query keywords
    3. Conversation history (last agent)
    4. Explicit domain mentions
    
    Returns: Domain string ('electronics', 'fiction', 'weather', 'general', etc.)
    """
    query_lower = query.lower()
    
    # 1. Editor type is PRIMARY signal
    if editor_context:
        editor_type = editor_context.get('type', '').strip().lower()
        if editor_type:
            # Map editor types to domains
            editor_domain_map = {
                'electronics': 'electronics',
                'fiction': 'fiction',
                'outline': 'fiction',
                'character': 'fiction',
                'rules': 'fiction',
                'style': 'fiction',
                'nfoutline': 'content',
                'article': 'content',
                'podcast': 'content',
                'substack': 'content',
                'blog': 'content',
                'project': 'general',
                'reference': 'general',
                'system': 'systems',
                'systems': 'systems'
            }
            domain = editor_domain_map.get(editor_type)
            if domain:
                logger.info(f"🔍 DOMAIN DETECTION: Editor type '{editor_type}' → domain '{domain}'")
                return domain
    
    # 2. Query keywords (check against agent capabilities)
    domain_scores = {}
    for agent, capabilities in AGENT_CAPABILITIES.items():
        for keyword in capabilities['keywords']:
            if keyword in query_lower:
                for domain in capabilities['domains']:
                    domain_scores[domain] = domain_scores.get(domain, 0) + 1
    
    if domain_scores:
        best_domain = max(domain_scores, key=domain_scores.get)
        logger.info(f"🔍 DOMAIN DETECTION: Query keywords → domain '{best_domain}' (score: {domain_scores[best_domain]})")
        return best_domain
    
    # 3. Conversation history
    if conversation_history:
        last_agent = conversation_history.get('last_agent') or conversation_history.get('primary_agent_selected')
        if last_agent:
            # Map agent to domain
            agent_domain_map = {
                'electronics_agent': 'electronics',
                'fiction_editing_agent': 'fiction',
                'story_analysis_agent': 'fiction',
                'writing_assistant_agent': 'fiction',
                'series_editing_agent': 'fiction',
                'weather_agent': 'weather',
                'email_agent': 'email',
                'research_agent': 'general',
                'site_crawl_agent': 'general',
                'security_analysis_agent': 'general',
                'reference_agent': 'general',
                'image_generation_agent': 'general',
                'org_content_agent': 'general',
                'org_capture_agent': 'general',
                'podcast_script_agent': 'content'
            }
            domain = agent_domain_map.get(last_agent)
            if domain:
                logger.info(f"🔍 DOMAIN DETECTION: Conversation history → domain '{domain}'")
                return domain
    
    # 4. Default to general
    logger.info(f"🔍 DOMAIN DETECTION: No strong signal → domain 'general'")
    return 'general'


def is_information_lookup_query(query: str) -> bool:
    """
    Detect if a query is an information lookup (how-to, instructions, documentation)
    vs technical understanding (design, analysis, project management)
    
    Returns: True if query is information lookup (should go to research_agent)
    """
    query_lower = query.lower()
    
    # Information lookup patterns - these should go to research_agent
    information_lookup_patterns = [
        'how can i', 'how do i', 'how to', 'how would i',
        'what is the procedure', 'what are the steps', 'what is the process',
        'where can i find', 'where do i find', 'where is',
        'tell me how', 'show me how', 'explain how',
        'instructions for', 'manual for', 'guide for', 'tutorial for',
        'how does one', 'how should i', 'how might i',
        'what is the way to', 'what is the method to',
        'can you tell me how', 'can you explain how',
        'change the', 'set the', 'adjust the', 'configure the',
        'what are the settings', 'what settings', 'what configuration',
        'more detail', 'more details', 'more information', 'tell me more',
        'elaborate on', 'explain more', 'need more'
    ]
    
    # Check for information lookup patterns
    for pattern in information_lookup_patterns:
        if pattern in query_lower:
            return True
    
    return False


def route_within_domain(
    domain: str,
    action_intent: str,
    query: str,
    editor_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Stage 2: Route within domain based on action intent and capabilities
    
    Returns: Target agent name
    """
    query_lower = query.lower()
    editor_type = editor_context.get('type', '').strip().lower() if editor_context else ''
    
    # If no type in editor_context, detect from filename or language
    if not editor_type and editor_context:
        filename = editor_context.get('filename', '').lower()
        language = editor_context.get('language', '').lower()
        
        # Detect org files by extension or language
        if filename.endswith('.org') or language == 'org':
            editor_type = 'org'
    
    # Domain-specific routing rules
    if domain == 'electronics':
        # Check if this is an information lookup query (how-to, instructions, documentation)
        # These should go to research_agent even if in electronics domain
        if is_information_lookup_query(query):
            logger.info(f"🔍 INFORMATION LOOKUP: Electronics domain query detected as information lookup → research_agent")
            return 'research_agent'
        
        # For editing operations (generation, modification), require active editor
        if action_intent in ['generation', 'modification']:
            if editor_type == 'electronics':
                return 'electronics_agent'
            else:
                # No electronics editor active - route to chat_agent for general discussion
                logger.info(f"🎯 ELECTRONICS DOMAIN: Editing action but no active editor - routing to chat_agent")
                return 'chat_agent'
        
        # For query/analysis actions, electronics_agent can handle without editor
        return 'electronics_agent'
    
    elif domain == 'fiction':
        # Fiction domain has multiple agents - route by action intent
        if action_intent == 'analysis':
            return 'story_analysis_agent'
        elif editor_type == 'outline':
            return 'writing_assistant_agent'  # Phase 1: Writing Assistant handles outline
        elif editor_type == 'character':
            return 'writing_assistant_agent'  # character_development_subgraph
        elif editor_type == 'rules':
            return 'writing_assistant_agent'  # Phase 2: Writing Assistant handles rules
        elif editor_type == 'series':
            return 'series_editing_agent'
        elif editor_type == 'style':
            return 'writing_assistant_agent'  # Phase 3: Writing Assistant handles style
        elif editor_type == 'fiction':
            return 'writing_assistant_agent'
        else:
            # No editor active - route to chat_agent for general fiction discussion
            logger.info(f"🎯 FICTION DOMAIN: No active editor - routing to chat_agent instead of fiction_editing_agent")
            return 'chat_agent'
    
    elif domain == 'weather':
        return 'weather_agent'

    elif domain == 'email':
        return 'email_agent'
    
    elif domain == 'navigation':
        return 'navigation_agent'
    
    elif domain == 'content':
        if editor_type in ['article', 'substack', 'blog']:
            return 'writing_assistant_agent'  # Article subgraph handles article/substack/blog
        if editor_type == 'nfoutline':
            return 'writing_assistant_agent'
        if editor_type == 'podcast':
            return 'podcast_script_agent'
        logger.info(f"🎯 CONTENT DOMAIN: No matching editor (editor_type='{editor_type}') - routing to chat_agent")
        return 'chat_agent'
    
    elif domain == 'general':
        # General domain routing
        # Capture-to-inbox: route first (no editor required)
        capture_patterns = ['capture to', 'for my inbox', 'add to inbox', 'to my inbox']
        if any(p in query_lower for p in capture_patterns):
            return 'org_capture_agent'
        # Check if reference editor is active - prefer reference_agent
        if editor_type == 'reference':
            return 'reference_agent'
        
        # Check if project editor is active - prefer general_project_agent
        if editor_type == 'project':
            return 'general_project_agent'
        
        if action_intent == 'query':
            # Check for org-related content queries first
            # BUT require active org editor - org_content_agent is editor-gated
            org_keywords = ['org', 'org-mode', 'todo', 'project', 'task']
            if any(kw in query_lower for kw in org_keywords):
                # Verify active org editor exists
                if editor_context:
                    editor_type = editor_context.get('type', '').strip().lower()
                    if not editor_type:
                        # Try to detect from filename or language
                        filename = editor_context.get('filename', '').lower()
                        language = editor_context.get('language', '').lower()
                        if filename.endswith('.org') or language == 'org':
                            editor_type = 'org'
                    if editor_type == 'org':
                        return 'org_content_agent'
                # No active org editor - route to chat_agent instead
                logger.info(f"🎯 ORG CONTENT QUERY: No active org editor - routing to chat_agent instead of org_content_agent")
                return 'chat_agent'
            # Check for help/documentation queries
            help_keywords = ['how do i', 'how can i', 'help with', 'what is [feature]', 'how does [agent] work',
                           'show me how to', 'guide for', 'instructions for', 'what can i do',
                           'what agents are available', 'available features', 'how to use', 'getting started',
                           'help', 'documentation', 'tutorial', 'user guide', 'feature guide']
            if any(kw in query_lower for kw in help_keywords):
                return 'help_agent'
            # Check for site crawl requests
            elif any(kw in query_lower for kw in ['crawl site', 'crawl website', 'site crawl', 'domain crawl', 'crawl domain']):
                return 'site_crawl_agent'
            # Check if it's document-specific (analysis) vs general research
            elif any(kw in query_lower for kw in ['compare', 'summarize', 'analyze', 'find differences', 'find conflicts']):
                return 'content_analysis_agent'
            else:
                return 'research_agent'
        elif action_intent == 'analysis':
            return 'content_analysis_agent'
        elif action_intent == 'observation':
            # Check for org-related content queries
            # BUT require active org editor - org_content_agent is editor-gated
            org_keywords = ['org', 'org-mode', 'todo', 'project', 'task']
            if any(kw in query_lower for kw in org_keywords):
                # Verify active org editor exists
                if editor_context:
                    editor_type = editor_context.get('type', '').strip().lower()
                    if not editor_type:
                        # Try to detect from filename or language
                        filename = editor_context.get('filename', '').lower()
                        language = editor_context.get('language', '').lower()
                        if filename.endswith('.org') or language == 'org':
                            editor_type = 'org'
                    if editor_type == 'org':
                        return 'org_content_agent'
                # No active org editor - route to chat_agent instead
                logger.info(f"🎯 ORG CONTENT OBSERVATION: No active org editor - routing to chat_agent instead of org_content_agent")
                return 'chat_agent'
            # Conversational statements, greetings, checking status → chat_agent
            return 'chat_agent'
        elif action_intent == 'generation':
            # Check for org capture (add to inbox) before image generation
            if any(p in query_lower for p in ['capture to', 'for my inbox', 'add to inbox', 'to my inbox']):
                return 'org_capture_agent'
            # Check for image generation keywords first
            image_keywords = ['create image', 'generate image', 'generate picture', 'draw', 'visualize',
                            'image', 'picture', 'photo', 'photography', 'create a picture',
                            'make an image', 'generate a photo', 'create a photo', 'draw a picture',
                            'create an image', 'make a picture', 'generate an image', 'create picture']
            if any(kw in query_lower for kw in image_keywords):
                return 'image_generation_agent'
            # For other generation without editor context, default to chat_agent
            # general_project_agent should only be selected via capability matching when editor is active
            return 'chat_agent'
        elif action_intent == 'modification':
            # Modification without specific editor context → chat_agent
            return 'chat_agent'
        elif action_intent == 'management':
            # Org capture (add to inbox) → org_capture_agent
            if any(p in query_lower for p in ['capture to', 'for my inbox', 'add to inbox', 'to my inbox']):
                return 'org_capture_agent'
            # Other management operations → chat_agent (org_content_agent is read-only)
            return 'chat_agent'
        else:
            # Default fallback
            return 'chat_agent'
    
    # Fallback
    return 'chat_agent'


def score_agent_capabilities(
    agent: str,
    domain: str,
    action_intent: str,
    query: str,
    editor_context: Optional[Dict[str, Any]] = None,
    last_agent: Optional[str] = None,
    editor_preference: str = 'prefer'
) -> float:
    """
    Score how well an agent matches the required capabilities
    
    Args:
        agent: Agent name
        domain: Detected domain
        action_intent: Action intent
        query: User query
        editor_context: Editor context (if available)
        last_agent: Last agent used (for continuity)
        editor_preference: Editor preference setting ('prefer' or 'ignore')
    
    Returns: Score (higher = better match)
    """
    if agent not in AGENT_CAPABILITIES:
        return 0.0
    
    capabilities = AGENT_CAPABILITIES[agent]
    score = 0.0
    
    # Check if agent requires editor but none is provided
    requires_editor = capabilities.get('requires_editor', False)
    has_editor_types = bool(capabilities.get('editor_types', []))
    editing_actions = ['generation', 'modification']
    
    # HARD GATE: Editor-gated agents (agents with editor_types) require an active editor when editor_preference == 'prefer'
    # This ensures that editor-gated agents like org_content_agent only route when an editor is actually open
    if has_editor_types and editor_preference == 'prefer':
        if not editor_context:
            # Editor-gated agent but no active editor - block routing
            logger.debug(f"  🚫 HARD GATE: Blocking {agent} - editor_preference is 'prefer' but no active editor, agent has editor_types: {capabilities.get('editor_types')}")
            return 0.0
        
        # Verify editor type matches agent's editor_types
        editor_type = editor_context.get('type', '').strip().lower()
        if not editor_type:
            # Try to detect from filename or language
            filename = editor_context.get('filename', '').lower()
            language = editor_context.get('language', '').lower()
            if filename.endswith('.org') or language == 'org':
                editor_type = 'org'
        
        if editor_type not in capabilities.get('editor_types', []):
            # Editor type doesn't match - block routing
            logger.debug(f"  🚫 HARD GATE: Blocking {agent} - editor type '{editor_type}' doesn't match agent editor_types: {capabilities.get('editor_types')}")
            return 0.0
    
    # HARD GATE: Editor-gated agents (agents with editor_types) can ONLY route when editor_preference == 'prefer'
    # This ensures that "prefer editor" MUST be checked for editor-gated agents like reference_agent
    if has_editor_types and editor_preference != 'prefer':
        # Editor-gated agent but "prefer editor" is not checked - block routing
        logger.info(f"🚫 HARD GATE: Blocking {agent} - editor_preference is '{editor_preference}' (not 'prefer'), agent has editor_types: {capabilities.get('editor_types')}")
        return 0.0
    
    # Editing agents (agents with editor_types) require an editor for editing actions
    # Even if requires_editor is not explicitly set, if agent has editor_types and action is editing, require editor
    if has_editor_types and action_intent in editing_actions:
        if not editor_context:
            # Editing agent with editing action but no editor - block routing
            logger.debug(f"  Agent {agent} is an editing agent with editing action '{action_intent}' but no active editor - blocking")
            return 0.0
    
    # Explicit requires_editor flag check
    if requires_editor and not editor_context:
        # Editing agents (agents with editor_types) should ALWAYS require an editor
        # No keyword bypass allowed - they need an active document to edit
        if has_editor_types:
            # Strict requirement: editing agents must have active editor
            logger.debug(f"  Agent {agent} is an editing agent and requires active editor - no keyword bypass allowed")
            return 0.0
        
        # For non-editing agents with requires_editor, allow keyword bypass
        # Check if query has explicit keywords - if so, allow routing
        query_lower = query.lower()
        keyword_matches = sum(1 for kw in capabilities['keywords'] if kw in query_lower)
        if keyword_matches == 0:
            # No editor and no keywords - don't route to this agent
            logger.debug(f"  Agent {agent} requires editor but none provided and no keywords matched")
            return 0.0
    
    # Domain match
    domain_match = domain in capabilities['domains']
    if domain_match:
        score += 10.0
    else:
        # Penalize domain mismatch to prevent continuity from overriding semantic intent
        # Only apply penalty if agent has specific domain requirements (not general agents)
        if capabilities['domains'] and 'general' not in capabilities['domains']:
            score -= 15.0  # Strong penalty for domain mismatch
            logger.debug(f"  -15.0 penalty: {agent} domain mismatch (detected: '{domain}', agent domains: {capabilities['domains']})")
    
    # Editor context match (strong boost)
    if editor_context:
        editor_type = editor_context.get('type', '').strip().lower()
        if editor_type in capabilities['editor_types']:
            score += capabilities['context_boost']
    
    # Keyword match (check before action intent to use in action intent logic)
    query_lower = query.lower()
    keyword_matches = sum(1 for kw in capabilities['keywords'] if kw in query_lower)
    score += keyword_matches * 2.0
    
    # Action intent match (CRITICAL - must match for routing)
    if action_intent in capabilities['actions']:
        score += 5.0
        
        # Special handling for story_analysis_agent: requires explicit analysis keywords
        # This prevents basic questions from routing to analysis agent
        if agent == 'story_analysis_agent':
            # Must have explicit analysis keywords (analyze, critique, review, etc.)
            explicit_analysis_keywords = ['analyze', 'critique', 'review', 'assess', 'evaluate', 'examine', 'study']
            has_explicit_keyword = any(kw in query_lower for kw in explicit_analysis_keywords)
            if not has_explicit_keyword:
                # No explicit analysis keyword - heavily penalize (prefer fiction_editing_agent for questions)
                score -= 15.0
                logger.debug(f"  -15.0 penalty: {agent} requires explicit analysis keywords (analyze/critique/review)")
    else:
        # Agent doesn't support this action intent - heavily penalize
        # Only allow if it's a very strong keyword match or editor context match
        if keyword_matches == 0 and not editor_context:
            logger.debug(f"  Agent {agent} doesn't support action '{action_intent}' and no keywords/editor match - excluding")
            return 0.0  # Exclude entirely if no supporting context
        # Allow with heavy penalty if there's editor/keyword context
        score -= 10.0
        logger.debug(f"  -10.0 penalty: {agent} doesn't support action '{action_intent}'")
    
    # Special boost for research_agent on information lookup queries
    # BUT only if action intent is 'query' (research doesn't handle 'observation')
    # AND only if NOT asking about current document (document questions should go to editor agents)
    is_document_question = any(word in query.lower() for word in ['this', 'the document', 'the file', 'current', 'here', 'in this'])
    if agent == 'research_agent' and action_intent == 'query' and is_information_lookup_query(query) and not is_document_question:
        score += 15.0  # Strong boost to override electronics domain routing
        logger.debug(f"  +15.0 information lookup boost for research_agent")
    
    # Strong boost for explicit research keywords (overrides continuity)
    explicit_research_keywords = [
        'do some research', 'do research', 'can you research', 'please research',
        'find me a picture', 'find me a pic', 'find me an image', 'find me a photo',
        'find me pictures', 'find me images', 'find me photos',
        'show me a picture', 'show me a pic', 'show me an image', 'show me a photo',
        'show me pictures', 'show me images', 'show me photos'
    ]
    if agent == 'research_agent' and any(kw in query_lower for kw in explicit_research_keywords):
        score += 20.0  # Very strong boost to override continuity
        logger.debug(f"  +20.0 explicit research keyword boost for research_agent (overrides continuity)")
    
    # Special boost for editor-matched agents when question is about current document
    if editor_context and is_document_question and editor_type in capabilities['editor_types']:
        score += 10.0  # Strong boost for document-specific questions
        logger.debug(f"  +10.0 document question boost for {agent} (editor matches)")
    
    # Conversation continuity (reduced boost to avoid overriding semantic intent)
    # Only apply continuity boost if there's at least some semantic match (domain OR keywords)
    # This prevents continuity from routing queries to completely irrelevant agents
    if last_agent == agent:
        has_semantic_match = domain_match or keyword_matches > 0
        if has_semantic_match:
            score += 2.0  # Reduced from 3.0 to allow semantic queries to override
            logger.debug(f"  +2.0 continuity boost for {agent} (last_agent match, semantic relevance confirmed)")
        else:
            logger.debug(f"  No continuity boost for {agent} - domain mismatch and no keyword matches (preventing irrelevant routing)")
    
    return score


def find_best_agent_match(
    domain: str,
    action_intent: str,
    query: str,
    editor_context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[Dict[str, Any]] = None,
    editor_preference: str = 'prefer'
) -> Tuple[str, float]:
    """
    Find best agent match using capability scoring
    
    Args:
        domain: Detected domain
        action_intent: Action intent
        query: User query
        editor_context: Editor context (if available)
        conversation_history: Conversation history for continuity
        editor_preference: Editor preference setting ('prefer' or 'ignore')
    
    Returns: (agent_name, confidence_score)
    """
    # EDITOR TYPE OVERRIDE: When editor_preference is 'prefer' and editor type matches,
    # route directly to that agent (editor type is PRIMARY signal)
    if editor_preference == 'prefer' and editor_context:
        editor_type = editor_context.get('type', '').strip().lower()
        
        # If no type in editor_context, detect from filename or language
        if not editor_type:
            filename = editor_context.get('filename', '').lower()
            language = editor_context.get('language', '').lower()
            
            # Detect org files by extension or language
            if filename.endswith('.org') or language == 'org':
                editor_type = 'org'
        
        if editor_type:
            # Find agents that support this editor type
            matching_agents = []
            for agent, capabilities in AGENT_CAPABILITIES.items():
                # Skip internal-only agents
                if capabilities.get('internal_only', False):
                    continue
                if editor_type in capabilities.get('editor_types', []):
                    matching_agents.append(agent)
            
            if matching_agents:
                if len(matching_agents) == 1:
                    # Single match - route directly (editor type is PRIMARY signal)
                    best_agent = matching_agents[0]
                    logger.info(f"🎯 EDITOR TYPE OVERRIDE: editor_type='{editor_type}' → {best_agent} (editor type is PRIMARY signal)")
                    return best_agent, 0.95  # High confidence for editor type match
                else:
                    # Multiple agents match - use capability scoring to choose
                    logger.info(f"🎯 EDITOR TYPE MATCH: editor_type='{editor_type}' matches {len(matching_agents)} agents, using capability scoring")
                    # Continue to capability scoring below, but these agents will get context_boost
    
    # Extract last_agent and primary_agent_selected for continuity
    last_agent = None
    primary_agent_selected = None
    if conversation_history:
        last_agent = conversation_history.get('last_agent')
        primary_agent_selected = conversation_history.get('primary_agent_selected')
    
    # Use primary_agent_selected if last_agent is not set (for continuity in ongoing conversations)
    continuity_agent = last_agent or primary_agent_selected
    
    scores = {}
    for agent in AGENT_CAPABILITIES.keys():
        # Skip internal-only agents from direct routing (they should be called by other agents)
        capabilities = AGENT_CAPABILITIES.get(agent, {})
        if capabilities.get('internal_only', False):
            # Only allow if explicitly requested via keywords or if there's research data in shared_memory
            # For now, exclude from direct routing entirely
            continue
        
        score = score_agent_capabilities(
            agent=agent,
            domain=domain,
            action_intent=action_intent,
            query=query,
            editor_context=editor_context,
            last_agent=continuity_agent,  # Use continuity_agent (last_agent or primary_agent_selected)
            editor_preference=editor_preference
        )
        scores[agent] = score
    
    if not scores or max(scores.values()) == 0:
        # No good match, use domain-based routing
        best_agent = route_within_domain(domain, action_intent, query, editor_context)
        logger.info(f"🎯 CAPABILITY MATCHING: No strong match, using domain routing → {best_agent}")
        return best_agent, 0.5
    
    best_agent = max(scores, key=scores.get)
    best_score = scores[best_agent]
    
    # Log top 3 scores for debugging
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_scores[:3]
    logger.info(f"🎯 CAPABILITY MATCHING: Top agents - {', '.join([f'{agent} ({score:.1f})' for agent, score in top_3])}")
    
    # INTELLIGENT AGENT SWITCHING: Only switch if new agent scores significantly higher
    # This prevents unnecessary switches for marginal cases while allowing clear topic changes
    # Higher threshold when switching FROM chat_agent (since it handles general conversation)
    # CRITICAL: Use continuity_agent (last_agent or primary_agent_selected) for continuity checks
    MIN_SCORE_DIFFERENCE_FOR_SWITCH = 3.0  # Default minimum score difference
    MIN_SCORE_DIFFERENCE_FROM_CHAT = 1.5  # Lower threshold so research/local-lookup can switch from chat
    
    # OVERRIDE: Explicit research keywords bypass continuity (so we don't stay in chat)
    query_lower = query.lower()
    explicit_research_keywords = [
        'do some research', 'do research', 'can you research', 'please research',
        'research and ', 'research and see', 'research and find', 'research and check',
        'see if we have', 'check if we have', 'do we have any', 'find any ',
        'find me a picture', 'find me a pic', 'find me an image', 'find me a photo',
        'find me pictures', 'find me images', 'find me photos',
        'show me a picture', 'show me a pic', 'show me an image', 'show me a photo',
        'show me pictures', 'show me images', 'show me photos',
    ]
    has_explicit_research = any(kw in query_lower for kw in explicit_research_keywords)
    
    if continuity_agent and continuity_agent != best_agent:
        continuity_agent_score = scores.get(continuity_agent, 0.0)
        score_difference = best_score - continuity_agent_score
        
        # Use higher threshold when switching from chat_agent
        if continuity_agent == 'chat_agent':
            threshold = MIN_SCORE_DIFFERENCE_FROM_CHAT
        else:
            threshold = MIN_SCORE_DIFFERENCE_FOR_SWITCH
        
        # OVERRIDE: Explicit research keywords bypass continuity check
        if has_explicit_research and best_agent == 'research_agent':
            # User explicitly asked for research - override continuity
            logger.info(f"🔍 RESEARCH OVERRIDE: Explicit research keyword detected - routing to research_agent (bypassing continuity)")
            logger.info(f"   → {continuity_agent}: {continuity_agent_score:.1f} vs {best_agent}: {best_score:.1f}")
        elif score_difference < threshold:
            # Score difference is too small - maintain continuity
            logger.info(f"🔄 CONTINUITY: Keeping {continuity_agent} (score difference {score_difference:.1f} < {threshold} threshold)")
            logger.info(f"   → {continuity_agent}: {continuity_agent_score:.1f} vs {best_agent}: {best_score:.1f}")
            best_agent = continuity_agent
            best_score = continuity_agent_score
        else:
            # Score difference is significant - switch agents
            logger.info(f"🔄 TOPIC CHANGE: Switching to {best_agent} (score difference {score_difference:.1f} >= {threshold} threshold)")
            logger.info(f"   → {continuity_agent}: {continuity_agent_score:.1f} vs {best_agent}: {best_score:.1f}")
    elif continuity_agent and continuity_agent == best_agent:
        logger.info(f"✅ CONTINUITY: Routed to {best_agent} (matches continuity_agent, conversation continuity maintained)")
    
    # Normalize score to 0-1 confidence
    max_possible_score = 50.0  # Approximate max
    confidence = min(best_score / max_possible_score, 1.0)
    
    logger.info(f"🎯 CAPABILITY MATCHING: Selected {best_agent} (score: {best_score:.1f}, confidence: {confidence:.2f})")
    
    return best_agent, confidence
