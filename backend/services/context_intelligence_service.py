"""
Context Intelligence Service
Extensible system for analyzing conversation context and building agent intelligence
Based on LangGraph best practices for flexible, pluggable context analysis
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Protocol, TypedDict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ContextAnalyzer(Protocol):
    """Protocol for pluggable context analyzers"""
    
    def analyze(self, messages: List[Dict], state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversation context and return insights"""
        ...
    
    def get_analyzer_name(self) -> str:
        """Return the name of this analyzer"""
        ...


class ConversationIntelligence(TypedDict):
    """Structured conversation intelligence data"""
    topics_discovered: List[str]
    research_questions: List[Dict[str, Any]]
    user_intents: List[Dict[str, Any]]
    conversation_flow: List[Dict[str, Any]]
    context_changes: List[Dict[str, Any]]
    information_needs: List[Dict[str, Any]]


class TopicAnalyzer:
    """Analyzer for discovering conversation topics and themes"""
    
    def get_analyzer_name(self) -> str:
        return "topic_analyzer"
    
    def analyze(self, messages: List[Dict], state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversation for topics and themes"""
        topics = []
        keywords_found = {}
        
        # Extract all text content
        all_text = " ".join([
            msg.get("content", "") for msg in messages 
            if msg.get("content")
        ]).lower()
        
        # Topic detection patterns
        topic_patterns = {
            "fish_and_aquaculture": ["fish", "orfe", "pond", "aquarium", "koi", "goldfish", "aquatic"],
            "british_culture": ["british", "uk", "england", "scotland", "wales", "london"],
            "famous_people": ["famous", "celebrity", "notable", "well-known", "renowned"],
            "ownership": ["owner", "owners", "owns", "owned", "possession", "keeper"],
            "technology": ["software", "code", "programming", "computer", "system"],
            "research": ["research", "study", "analysis", "investigation", "examine"],
            "science": ["scientific", "biology", "chemistry", "physics", "nature"],
            "business": ["business", "company", "corporate", "enterprise", "commercial"],
            "history": ["history", "historical", "past", "ancient", "heritage"],
            "geography": ["location", "place", "region", "country", "city", "area"]
        }
        
        for topic, keywords in topic_patterns.items():
            matches = [kw for kw in keywords if kw in all_text]
            if matches:
                topics.append(topic)
                keywords_found[topic] = matches
        
        return {
            "topics_discovered": topics,
            "keywords_by_topic": keywords_found,
            "analysis_timestamp": datetime.now().isoformat()
        }


class ResearchQuestionAnalyzer:
    """Analyzer for identifying research questions and information requests"""
    
    def get_analyzer_name(self) -> str:
        return "research_question_analyzer"
    
    def analyze(self, messages: List[Dict], state: Dict[str, Any]) -> Dict[str, Any]:
        """Identify research questions and information needs"""
        research_questions = []
        
        for i, message in enumerate(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                
                # Question patterns
                question_indicators = [
                    r'\bwho\b.*\?',
                    r'\bwhat\b.*\?',
                    r'\bwhere\b.*\?',
                    r'\bwhen\b.*\?',
                    r'\bwhy\b.*\?',
                    r'\bhow\b.*\?',
                    r'\?',
                    r'\btell me about\b',
                    r'\bfind.*information\b',
                    r'\bsearch for\b',
                    r'\blook up\b'
                ]
                
                for pattern in question_indicators:
                    if re.search(pattern, content, re.IGNORECASE):
                        research_questions.append({
                            "question": content,
                            "message_index": i,
                            "timestamp": datetime.now().isoformat(),
                            "question_type": self._classify_question_type(content),
                            "complexity": self._assess_question_complexity(content),
                            "domains": self._identify_knowledge_domains(content)
                        })
                        break
        
        return {
            "research_questions": research_questions,
            "total_questions": len(research_questions),
            "question_complexity_distribution": self._get_complexity_distribution(research_questions)
        }
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of research question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["who", "person", "people", "individual"]):
            return "person_identification"
        elif any(word in question_lower for word in ["what", "define", "explain", "describe"]):
            return "factual_information"
        elif any(word in question_lower for word in ["where", "location", "place"]):
            return "geographical"
        elif any(word in question_lower for word in ["when", "time", "date", "year"]):
            return "temporal"
        elif any(word in question_lower for word in ["why", "reason", "cause"]):
            return "causal_explanation"
        elif any(word in question_lower for word in ["how", "method", "process"]):
            return "procedural"
        else:
            return "general_inquiry"
    
    def _assess_question_complexity(self, question: str) -> str:
        """Assess the complexity of a research question"""
        complexity_indicators = {
            "simple": ["is", "are", "can", "do", "does"],
            "medium": ["how", "why", "what", "where", "when"],
            "complex": ["relationship", "impact", "influence", "analyze", "compare", "evaluate"]
        }
        
        question_lower = question.lower()
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                return complexity
        
        return "medium"
    
    def _identify_knowledge_domains(self, question: str) -> List[str]:
        """Identify knowledge domains relevant to the question"""
        domains = []
        question_lower = question.lower()
        
        domain_patterns = {
            "biology": ["fish", "animal", "species", "biology", "nature"],
            "culture": ["british", "culture", "tradition", "cultural"],
            "celebrity": ["famous", "celebrity", "notable", "well-known"],
            "commerce": ["business", "owner", "commercial", "industry"],
            "geography": ["location", "place", "region", "country"],
            "history": ["history", "historical", "past", "heritage"],
            "technology": ["software", "computer", "digital", "technology"]
        }
        
        for domain, keywords in domain_patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                domains.append(domain)
        
        return domains or ["general"]
    
    def _get_complexity_distribution(self, questions: List[Dict]) -> Dict[str, int]:
        """Get distribution of question complexity levels"""
        distribution = {"simple": 0, "medium": 0, "complex": 0}
        
        for q in questions:
            complexity = q.get("complexity", "medium")
            distribution[complexity] = distribution.get(complexity, 0) + 1
        
        return distribution


class PermissionAnalyzer:
    """Analyzer for tracking permissions and authorizations"""
    
    def get_analyzer_name(self) -> str:
        return "permission_analyzer"
    
    def analyze(self, messages: List[Dict], state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversation for permission patterns"""
        permission_events = []
        current_permission_state = {
            "web_search": False,
            "tool_usage": False,
            "data_access": False
        }
        
        for i, message in enumerate(messages):
            content = message.get("content", "").lower()
            role = message.get("role")
            
            # System permission requests
            if role == "assistant":
                if "permission" in content and "web search" in content:
                    permission_events.append({
                        "type": "permission_request",
                        "permission": "web_search",
                        "message_index": i,
                        "timestamp": datetime.now().isoformat()
                    })
                elif "would you like" in content and "search" in content:
                    permission_events.append({
                        "type": "permission_offer",
                        "permission": "web_search",
                        "message_index": i,
                        "timestamp": datetime.now().isoformat()
                    })
            
            # User permission responses
            elif role == "user":
                if content.strip() in ["yes", "y", "sure", "proceed", "go ahead", "ok", "okay"]:
                    # Check if this follows a permission request
                    if (i > 0 and 
                        "search" in messages[i-1].get("content", "").lower() and
                        messages[i-1].get("role") == "assistant"):
                        permission_events.append({
                            "type": "permission_granted",
                            "permission": "web_search",
                            "message_index": i,
                            "timestamp": datetime.now().isoformat()
                        })
                        current_permission_state["web_search"] = True
                
                elif content.strip() in ["no", "n", "don't", "stop", "cancel"]:
                    permission_events.append({
                        "type": "permission_denied",
                        "permission": "web_search",
                        "message_index": i,
                        "timestamp": datetime.now().isoformat()
                    })
        
        return {
            "permission_events": permission_events,
            "current_permissions": current_permission_state,
            "web_search_granted": current_permission_state["web_search"],
            "latest_permission_timestamp": permission_events[-1]["timestamp"] if permission_events else None
        }


class ContextIntelligenceService:
    """Service for comprehensive conversation context analysis"""
    
    def __init__(self):
        self.analyzers = {
            "topic_analyzer": TopicAnalyzer(),
            "research_question_analyzer": ResearchQuestionAnalyzer(),
            "permission_analyzer": PermissionAnalyzer()
        }
        logger.info("✅ Context Intelligence Service initialized with built-in analyzers")
    
    def register_analyzer(self, name: str, analyzer: ContextAnalyzer):
        """Register a custom context analyzer"""
        self.analyzers[name] = analyzer
        logger.info(f"📝 Registered custom analyzer: {name}")
    
    def analyze_conversation(self, messages: List[Dict], state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive conversation analysis"""
        logger.info("🧠 Starting comprehensive conversation analysis...")
        
        analysis_results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "message_count": len(messages),
            "analyzer_results": {}
        }
        
        # Run all analyzers
        for name, analyzer in self.analyzers.items():
            try:
                logger.debug(f"🔍 Running analyzer: {name}")
                result = analyzer.analyze(messages, state)
                analysis_results["analyzer_results"][name] = result
                logger.debug(f"✅ Analyzer {name} completed successfully")
            except Exception as e:
                logger.error(f"❌ Analyzer {name} failed: {e}")
                analysis_results["analyzer_results"][name] = {"error": str(e)}
        
        # Synthesize results into conversation intelligence
        conversation_intelligence = self._synthesize_intelligence(analysis_results)
        analysis_results["conversation_intelligence"] = conversation_intelligence
        
        logger.info(f"✅ Conversation analysis complete with {len(self.analyzers)} analyzers")
        return analysis_results
    
    def _synthesize_intelligence(self, analysis_results: Dict[str, Any]) -> ConversationIntelligence:
        """Synthesize analyzer results into structured conversation intelligence"""
        analyzer_results = analysis_results.get("analyzer_results", {})
        
        # Extract topics
        topics = analyzer_results.get("topic_analyzer", {}).get("topics_discovered", [])
        
        # Extract research questions
        research_questions = analyzer_results.get("research_question_analyzer", {}).get("research_questions", [])
        
        # Extract permission state
        permission_data = analyzer_results.get("permission_analyzer", {})
        
        # Build user intents from various analyzers
        user_intents = []
        if research_questions:
            user_intents.append({
                "intent": "information_seeking",
                "confidence": 0.9,
                "evidence": f"{len(research_questions)} research questions identified"
            })
        
        if permission_data.get("web_search_granted"):
            user_intents.append({
                "intent": "web_research_authorized",
                "confidence": 1.0,
                "evidence": "User granted web search permission"
            })
        
        # Build conversation flow
        conversation_flow = []
        if research_questions:
            for i, q in enumerate(research_questions):
                conversation_flow.append({
                    "step": i + 1,
                    "type": "research_question",
                    "content": q["question"],
                    "timestamp": q["timestamp"]
                })
        
        # Identify information needs
        information_needs = []
        for question in research_questions:
            information_needs.append({
                "need": question["question"],
                "domains": question["domains"],
                "complexity": question["complexity"],
                "priority": "high" if question["complexity"] == "complex" else "medium"
            })
        
        return {
            "topics_discovered": topics,
            "research_questions": research_questions,
            "user_intents": user_intents,
            "conversation_flow": conversation_flow,
            "context_changes": [],  # TODO: Implement context change detection
            "information_needs": information_needs
        }
    
    def update_state_with_intelligence(self, state: Dict[str, Any], messages: List[Dict]) -> Dict[str, Any]:
        """Update agent state with comprehensive conversation intelligence"""
        try:
            # Analyze conversation
            analysis = self.analyze_conversation(messages, state)
            
            # Update conversation intelligence
            state["conversation_intelligence"] = analysis["conversation_intelligence"]
            
            # Update context registry
            state["context_registry"]["active_analyzers"] = list(self.analyzers.keys())
            state["context_registry"]["analyzer_results"] = analysis["analyzer_results"]
            
            # Update permission state from analysis
            permission_analysis = analysis["analyzer_results"].get("permission_analyzer", {})
            if permission_analysis:
                state["permission_state"]["web_search_permissions"]["granted"] = permission_analysis.get("web_search_granted", False)
                if permission_analysis.get("latest_permission_timestamp"):
                    state["permission_state"]["web_search_permissions"]["granted_timestamp"] = permission_analysis["latest_permission_timestamp"]
            
            # Update research context
            research_questions = analysis["conversation_intelligence"]["research_questions"]
            if research_questions:
                state["research_context"]["domain_expertise_needed"] = list(set([
                    domain for q in research_questions for domain in q.get("domains", [])
                ]))
                
                complexity_levels = [q.get("complexity", "medium") for q in research_questions]
                if "complex" in complexity_levels:
                    state["research_context"]["research_depth_required"] = "deep"
                elif "medium" in complexity_levels:
                    state["research_context"]["research_depth_required"] = "moderate"
            
            logger.info("✅ State updated with conversation intelligence")
            return state
            
        except Exception as e:
            logger.error(f"❌ Failed to update state with intelligence: {e}")
            return state


# Global service instance
context_intelligence_service = ContextIntelligenceService()
