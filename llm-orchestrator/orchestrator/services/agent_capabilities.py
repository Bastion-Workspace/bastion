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
    },
    'fiction_editing_agent': {
        'domains': ['fiction', 'writing', 'story', 'manuscript'],
        'actions': ['observation', 'generation', 'modification'],
        'editor_types': ['fiction'],
        'keywords': ['chapter', 'scene', 'dialogue', 'character', 'plot', 'manuscript', 'story', 'prose'],
        'context_boost': 20
    },
    'story_analysis_agent': {
        'domains': ['fiction', 'writing', 'story'],
        'actions': ['analysis'],
        'editor_types': ['fiction'],
        'keywords': ['analyze', 'critique', 'review', 'pacing', 'structure', 'themes'],
        'context_boost': 15
    },
    'outline_editing_agent': {
        'domains': ['fiction', 'writing', 'outline'],
        'actions': ['observation', 'generation', 'modification'],
        'editor_types': ['outline'],
        'keywords': ['outline', 'structure', 'act', 'plot points'],
        'context_boost': 20
    },
    'character_development_agent': {
        'domains': ['fiction', 'writing', 'character'],
        'actions': ['observation', 'generation', 'modification'],
        'editor_types': ['character'],
        'keywords': ['character', 'protagonist', 'antagonist', 'backstory', 'motivation'],
        'context_boost': 20
    },
    'rules_editing_agent': {
        'domains': ['fiction', 'writing', 'worldbuilding'],
        'actions': ['observation', 'generation', 'modification'],
        'editor_types': ['rules'],
        'keywords': ['rules', 'worldbuilding', 'canon', 'magic system', 'lore'],
        'context_boost': 20
    },
    'weather_agent': {
        'domains': ['weather', 'forecast', 'climate'],
        'actions': ['query', 'observation'],
        'editor_types': [],
        'keywords': ['weather', 'temperature', 'forecast', 'rain', 'snow', 'sunny', 'cloudy'],
        'context_boost': 0
    },
    'research_agent': {
        'domains': ['general', 'research', 'information'],
        'actions': ['query'],
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
            'change the', 'set the', 'adjust the', 'configure the'
        ],
        'context_boost': 0
    },
    'content_analysis_agent': {
        'domains': ['general', 'analysis', 'documents'],
        'actions': ['analysis'],
        'editor_types': [],
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
    'site_crawl_agent': {
        'domains': ['research', 'web', 'information'],
        'actions': ['query'],
        'editor_types': [],
        'keywords': ['crawl site', 'crawl website', 'site crawl', 'domain crawl', 'crawl domain'],
        'context_boost': 0
    },
    'proofreading_agent': {
        'domains': ['fiction', 'writing'],
        'actions': ['modification'],
        'editor_types': ['fiction'],
        'keywords': ['proofread', 'check grammar', 'fix typos', 'style corrections', 'grammar check', 'spell check'],
        'context_boost': 20
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
                'podcast': 'content',
                'substack': 'content',
                'blog': 'content',
                'project': 'general'
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
                'outline_editing_agent': 'fiction',
                'character_development_agent': 'fiction',
                'rules_editing_agent': 'fiction',
                'proofreading_agent': 'fiction',
                'weather_agent': 'weather',
                'research_agent': 'general',
                'site_crawl_agent': 'general'
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
        'what are the settings', 'what settings', 'what configuration'
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
    
    # Domain-specific routing rules
    if domain == 'electronics':
        # Check if this is an information lookup query (how-to, instructions, documentation)
        # These should go to research_agent even if in electronics domain
        if is_information_lookup_query(query):
            logger.info(f"🔍 INFORMATION LOOKUP: Electronics domain query detected as information lookup → research_agent")
            return 'research_agent'
        
        # Technical understanding, design, analysis, project management → electronics_agent
        return 'electronics_agent'
    
    elif domain == 'fiction':
        # Fiction domain has multiple agents - route by action intent
        if action_intent == 'analysis':
            return 'story_analysis_agent'
        elif editor_type == 'outline':
            return 'outline_editing_agent'
        elif editor_type == 'character':
            return 'character_development_agent'
        elif editor_type == 'rules':
            return 'rules_editing_agent'
        else:
            return 'fiction_editing_agent'
    
    elif domain == 'weather':
        return 'weather_agent'
    
    elif domain == 'general':
        # General domain routing
        # Check if project editor is active - prefer general_project_agent
        if editor_type == 'project':
            return 'general_project_agent'
        
        if action_intent == 'query':
            # Check for help/documentation queries first
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
            # Conversational statements, greetings, checking status → chat_agent
            return 'chat_agent'
        elif action_intent == 'generation':
            # For generation without editor context, default to chat_agent
            # general_project_agent should only be selected via capability matching when editor is active
            return 'chat_agent'
        elif action_intent == 'modification':
            # Modification without specific editor context → chat_agent
            return 'chat_agent'
        elif action_intent == 'management':
            # Management operations → check for specific keywords or default to chat_agent
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
    last_agent: Optional[str] = None
) -> float:
    """
    Score how well an agent matches the required capabilities
    
    Returns: Score (higher = better match)
    """
    if agent not in AGENT_CAPABILITIES:
        return 0.0
    
    capabilities = AGENT_CAPABILITIES[agent]
    score = 0.0
    
    # Check if agent requires editor but none is provided
    requires_editor = capabilities.get('requires_editor', False)
    if requires_editor and not editor_context:
        # Check if query has explicit keywords - if so, allow routing
        query_lower = query.lower()
        keyword_matches = sum(1 for kw in capabilities['keywords'] if kw in query_lower)
        if keyword_matches == 0:
            # No editor and no keywords - don't route to this agent
            logger.debug(f"  Agent {agent} requires editor but none provided and no keywords matched")
            return 0.0
    
    # Domain match
    if domain in capabilities['domains']:
        score += 10.0
    
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
    if agent == 'research_agent' and action_intent == 'query' and is_information_lookup_query(query):
        score += 15.0  # Strong boost to override electronics domain routing
        logger.debug(f"  +15.0 information lookup boost for research_agent")
    
    # Conversation continuity (reduced boost to avoid overriding semantic intent)
    if last_agent == agent:
        score += 2.0  # Reduced from 3.0 to allow semantic queries to override
        logger.debug(f"  +2.0 continuity boost for {agent} (last_agent match)")
    
    return score


def find_best_agent_match(
    domain: str,
    action_intent: str,
    query: str,
    editor_context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[Dict[str, Any]] = None
) -> Tuple[str, float]:
    """
    Find best agent match using capability scoring
    
    Returns: (agent_name, confidence_score)
    """
    last_agent = None
    if conversation_history:
        last_agent = conversation_history.get('last_agent') or conversation_history.get('primary_agent_selected')
    
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
            last_agent=last_agent
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
    MIN_SCORE_DIFFERENCE_FOR_SWITCH = 3.0  # Default minimum score difference
    MIN_SCORE_DIFFERENCE_FROM_CHAT = 8.0  # Higher threshold when switching from chat_agent
    
    if last_agent and last_agent != best_agent:
        last_agent_score = scores.get(last_agent, 0.0)
        score_difference = best_score - last_agent_score
        
        # Use higher threshold when switching from chat_agent (handles general conversation)
        threshold = MIN_SCORE_DIFFERENCE_FROM_CHAT if last_agent == 'chat_agent' else MIN_SCORE_DIFFERENCE_FOR_SWITCH
        
        if score_difference < threshold:
            # Score difference is too small - maintain continuity
            logger.info(f"🔄 CONTINUITY: Keeping {last_agent} (score difference {score_difference:.1f} < {threshold} threshold)")
            logger.info(f"   → {last_agent}: {last_agent_score:.1f} vs {best_agent}: {best_score:.1f}")
            best_agent = last_agent
            best_score = last_agent_score
        else:
            # Score difference is significant - switch agents
            logger.info(f"🔄 TOPIC CHANGE: Switching to {best_agent} (score difference {score_difference:.1f} >= {threshold} threshold)")
            logger.info(f"   → {last_agent}: {last_agent_score:.1f} vs {best_agent}: {best_score:.1f}")
    elif last_agent and last_agent == best_agent:
        logger.info(f"✅ CONTINUITY: Routed to {best_agent} (matches primary_agent, conversation continuity maintained)")
    
    # Normalize score to 0-1 confidence
    max_possible_score = 50.0  # Approximate max
    confidence = min(best_score / max_possible_score, 1.0)
    
    logger.info(f"🎯 CAPABILITY MATCHING: Selected {best_agent} (score: {best_score:.1f}, confidence: {confidence:.2f})")
    
    return best_agent, confidence

