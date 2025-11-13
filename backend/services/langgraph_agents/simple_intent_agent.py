"""
Simple Intent Classification Agent - Roosevelt's "Lean Routing Command"

**BULLY!** Quick, decisive intent classification without complex JSON cavalry charges!
"""

import logging
from typing import Dict, Any

from services.langgraph_agents.base_agent import BaseAgent
from services.simple_intent_service import SimpleIntentService
from models.simple_intent_models import SimpleIntentResult

logger = logging.getLogger(__name__)


class SimpleIntentAgent(BaseAgent):
	"""
	Roosevelt's Simple Intent Classification Agent
	
	**BULLY!** Fast, focused intent classification - no bloated analysis!
	Just intent → agent → permission → done!
	"""
	
	def __init__(self):
		super().__init__("simple_intent_agent")
		self.intent_service = SimpleIntentService()
		logger.info("🎯 Simple Intent Agent initialized - lean and mean routing specialist")
	
	async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
		"""
		**BULLY!** Process intent classification with simple, focused approach
		"""
		try:
			# Extract user message
			messages = state.get("messages", [])
			if not messages:
				return self._create_error_result("No messages found in state")
			
			user_message = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
			
			logger.info(f"🎯 SIMPLE INTENT: Processing message: {user_message[:100]}...")
			
			# Get simple intent classification
			intent_result = await self.intent_service.classify_intent(
				user_message=user_message,
				conversation_context=state
			)
			
			logger.info(f"✅ SIMPLE ROUTING: → {intent_result.target_agent}")
			
			# Lean result for orchestrator consumption
			legacy_result = {
				"target_agent": intent_result.target_agent,
				"confidence": intent_result.confidence,
				"permission_required": intent_result.permission_required,
			}
			
			return {
				"agent_results": {
					"agent_type": "simple_intent_agent",
					"intent_classification": legacy_result,
					"task_status": "complete",
					"processing_time": 0.1,
					"timestamp": self._get_timestamp()
				},
				"intent_classification": legacy_result,
				"latest_response": f"Intent classified: → {intent_result.target_agent}"
			}
			
		except Exception as e:
			logger.error(f"❌ Simple intent classification failed: {e}")
			return self._create_error_result(f"Intent classification error: {str(e)}")
	
	def _convert_to_legacy_format(self, intent_result: SimpleIntentResult) -> Dict[str, Any]:
		"""Deprecated - retained for interface, not used."""
		return {
			"target_agent": intent_result.target_agent,
			"confidence": intent_result.confidence,
			"permission_required": intent_result.permission_required,
		}
	
	def _create_error_result(self, error_message: str) -> Dict[str, Any]:
		"""Create error result in expected format"""
		return {
			"agent_results": {
				"agent_type": "simple_intent_agent",
				"task_status": "error",
				"error_message": error_message,
				"processing_time": 0.0,
				"timestamp": self._get_timestamp()
			},
			"intent_classification": {
				"target_agent": "chat_agent",
				"confidence": 0.5,
				"permission_required": False
			},
			"error_state": error_message
		}
	
	def _get_timestamp(self) -> str:
		"""Get current timestamp"""
		from datetime import datetime
		return datetime.now().isoformat()


# Async wrapper function for tool registry compatibility
async def classify_simple_intent(user_message: str, conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
	"""
	**BULLY!** Simple intent classification wrapper for tool registry
	"""
	agent = SimpleIntentAgent()
	state = {
		"messages": [type('Message', (), {'content': user_message})()],
		**(conversation_context or {})
	}
	return await agent._process_request(state)
