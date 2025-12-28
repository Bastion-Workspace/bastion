"""
Capability Workflow Engine - Roosevelt's "Multi-Step Campaign Coordinator"
Handles complex multi-step workflows with capability-based routing
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

from models.capability_routing_models import CapabilityBasedIntentResult, RoutingDecision
from services.agent_intelligence_network import get_agent_network, CollaborationPermission

logger = logging.getLogger(__name__)


class WorkflowStepType(str, Enum):
    """Types of workflow steps"""
    PERMISSION_CHECK = "permission_check"
    AGENT_EXECUTION = "agent_execution"
    COLLABORATION_OFFER = "collaboration_offer"
    DATA_FORMATTING = "data_formatting"
    RESULT_SYNTHESIS = "result_synthesis"


class WorkflowStep:
    """Individual step in a multi-step workflow"""
    
    def __init__(
        self,
        step_type: WorkflowStepType,
        agent_type: str,
        description: str,
        prerequisites: List[str] = None,
        auto_execute: bool = False
    ):
        self.step_type = step_type
        self.agent_type = agent_type
        self.description = description
        self.prerequisites = prerequisites or []
        self.auto_execute = auto_execute
        self.completed = False
        self.result = None


class CapabilityWorkflowEngine:
    """
    Roosevelt's Multi-Step Workflow Engine
    
    **BULLY!** Orchestrates complex workflows with capability-based routing!
    """
    
    def __init__(self):
        self._agent_network = get_agent_network()
        self._active_workflows: Dict[str, List[WorkflowStep]] = {}
    
    async def plan_workflow(
        self, 
        capability_result: CapabilityBasedIntentResult,
        conversation_context: Dict[str, Any]
    ) -> List[WorkflowStep]:
        """Plan multi-step workflow based on capability analysis"""
        try:
            workflow_steps = []
            
            # **STEP 1: PERMISSION CHECK** (if required)
            permission_req = capability_result.routing_decision.permission_requirement
            if permission_req.required and not permission_req.auto_grant_eligible:
                workflow_steps.append(WorkflowStep(
                    step_type=WorkflowStepType.PERMISSION_CHECK,
                    agent_type="permission_handler",
                    description=f"Request {permission_req.permission_type} permission",
                    auto_execute=False
                ))
            
            # **STEP 2: PRIMARY AGENT EXECUTION**
            primary_agent = capability_result.routing_decision.primary_agent
            workflow_steps.append(WorkflowStep(
                step_type=WorkflowStepType.AGENT_EXECUTION,
                agent_type=primary_agent,
                description=f"Execute primary task with {primary_agent}",
                prerequisites=["permission_check"] if permission_req.required else [],
                auto_execute=True
            ))
            
            # **STEP 3: COLLABORATION OPPORTUNITIES** (based on agent capabilities)
            collaboration_steps = await self._identify_collaboration_opportunities(
                capability_result, conversation_context
            )
            workflow_steps.extend(collaboration_steps)
            
            # **STEP 4: DATA FORMATTING** (if beneficial)
            # Note: Data formatting is now handled internally by research_agent via subgraph
            # No need to add explicit formatting steps here
            
            logger.info(f"🗂️ WORKFLOW PLANNED: {len(workflow_steps)} steps for {capability_result.intent_type}")
            return workflow_steps
            
        except Exception as e:
            logger.error(f"❌ Workflow planning failed: {e}")
            # Return simple single-step workflow
            return [WorkflowStep(
                step_type=WorkflowStepType.AGENT_EXECUTION,
                agent_type=capability_result.routing_decision.primary_agent,
                description="Execute primary task",
                auto_execute=True
            )]
    
    async def _identify_collaboration_opportunities(
        self,
        capability_result: CapabilityBasedIntentResult,
        conversation_context: Dict[str, Any]
    ) -> List[WorkflowStep]:
        """Identify potential collaboration steps based on agent capabilities"""
        collaboration_steps = []
        
        try:
            primary_agent = capability_result.routing_decision.primary_agent
            
            # **RESEARCH → WEATHER COLLABORATION**
            if primary_agent == "research_agent":
                # Check if research might benefit from weather information
                user_message = self._extract_user_message(conversation_context)
                if any(keyword in user_message.lower() for keyword in [
                    "travel", "trip", "vacation", "outdoor", "event", "venue", "location"
                ]):
                    weather_agent_info = self._agent_network.get_agent_info("weather_agent")
                    if weather_agent_info and weather_agent_info.collaboration_permission == CollaborationPermission.SUGGEST_ONLY:
                        collaboration_steps.append(WorkflowStep(
                            step_type=WorkflowStepType.COLLABORATION_OFFER,
                            agent_type="weather_agent",
                            description="Offer weather information for travel/location context",
                            prerequisites=["research_agent"],
                            auto_execute=False
                        ))
            
            # **WEATHER → RESEARCH COLLABORATION**
            elif primary_agent == "weather_agent":
                # Weather might benefit from location research
                user_message = self._extract_user_message(conversation_context)
                if any(keyword in user_message.lower() for keyword in [
                    "trip", "travel", "visit", "vacation", "activities", "things to do"
                ]):
                    collaboration_steps.append(WorkflowStep(
                        step_type=WorkflowStepType.COLLABORATION_OFFER,
                        agent_type="research_agent",
                        description="Offer location research and activity recommendations",
                        prerequisites=["weather_agent"],
                        auto_execute=False
                    ))
            
            # **ANY AGENT → DATA FORMATTING**
            # Note: Data formatting is now handled internally by research_agent via subgraph
            # No need to add explicit formatting steps here
            
            logger.info(f"🤝 COLLABORATION OPPORTUNITIES: {len(collaboration_steps)} steps identified")
            return collaboration_steps
            
        except Exception as e:
            logger.error(f"❌ Collaboration identification failed: {e}")
            return []
    
    def _extract_user_message(self, conversation_context: Dict[str, Any]) -> str:
        """Extract user message from conversation context"""
        try:
            messages = conversation_context.get("messages", [])
            if messages:
                latest_message = messages[-1]
                if hasattr(latest_message, 'content'):
                    return latest_message.content
                elif isinstance(latest_message, dict):
                    return latest_message.get("content", "")
            return ""
        except Exception as e:
            logger.error(f"❌ Failed to extract user message: {e}")
            return ""
    
    def _should_add_formatting_step(self, capability_result: CapabilityBasedIntentResult) -> bool:
        """Determine if data formatting step would be beneficial"""
        try:
            # Check if primary agent produces data that benefits from formatting
            primary_agent = capability_result.routing_decision.primary_agent
            
            # Research agent often produces data suitable for formatting
            if primary_agent == "research_agent":
                return True
            
            # Check if multiple agents are involved (complex data)
            if len(capability_result.capable_agents) > 2:
                return True
            
            # Check if intent suggests data organization
            if capability_result.intent_type in ["RESEARCH", "COMPUTATIONAL"]:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Formatting step assessment failed: {e}")
            return False
    
    async def execute_workflow_step(
        self,
        step: WorkflowStep,
        state: Dict[str, Any],
        workflow_id: str
    ) -> Dict[str, Any]:
        """Execute a single workflow step"""
        try:
            logger.info(f"🎯 EXECUTING WORKFLOW STEP: {step.step_type.value} → {step.agent_type}")
            
            if step.step_type == WorkflowStepType.PERMISSION_CHECK:
                return await self._execute_permission_check(step, state)
            
            elif step.step_type == WorkflowStepType.AGENT_EXECUTION:
                return await self._execute_agent_step(step, state)
            
            elif step.step_type == WorkflowStepType.COLLABORATION_OFFER:
                return await self._execute_collaboration_offer(step, state)
            
            elif step.step_type == WorkflowStepType.DATA_FORMATTING:
                return await self._execute_formatting_step(step, state)
            
            elif step.step_type == WorkflowStepType.RESULT_SYNTHESIS:
                return await self._execute_synthesis_step(step, state)
            
            else:
                logger.warning(f"⚠️ Unknown workflow step type: {step.step_type}")
                return state
                
        except Exception as e:
            logger.error(f"❌ Workflow step execution failed: {e}")
            step.completed = False
            return state
    
    async def _execute_permission_check(self, step: WorkflowStep, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute permission check step"""
        # This would integrate with HITL permission system
        logger.info("🛑 PERMISSION CHECK: Setting up permission request")
        
        # Mark step as requiring user input
        state["requires_user_input"] = True
        state["permission_step"] = {
            "step_description": step.description,
            "agent_type": step.agent_type
        }
        
        step.completed = True
        return state
    
    async def _execute_agent_step(self, step: WorkflowStep, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute primary agent step"""
        logger.info(f"🤖 AGENT EXECUTION: {step.agent_type}")
        
        # This would route to the actual agent
        # For now, mark as routing decision
        state["workflow_routing"] = step.agent_type
        step.completed = True
        return state
    
    async def _execute_collaboration_offer(self, step: WorkflowStep, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute collaboration offer step"""
        logger.info(f"🤝 COLLABORATION OFFER: {step.agent_type}")
        
        # Store collaboration suggestion in shared memory
        shared_memory = state.get("shared_memory", {})
        shared_memory["pending_collaboration"] = {
            "suggested_agent": step.agent_type,
            "suggestion": step.description,
            "auto_execute": step.auto_execute
        }
        state["shared_memory"] = shared_memory
        
        step.completed = True
        return state
    
    async def _execute_formatting_step(self, step: WorkflowStep, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data formatting step"""
        logger.info("📊 DATA FORMATTING: Auto-executing formatting")
        
        # Mark for auto-formatting
        state["auto_format_results"] = True
        step.completed = True
        return state
    
    async def _execute_synthesis_step(self, step: WorkflowStep, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute result synthesis step"""
        logger.info("🔄 RESULT SYNTHESIS: Combining multi-agent results")
        
        # This would synthesize results from multiple agents
        step.completed = True
        return state
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of active workflow"""
        workflow = self._active_workflows.get(workflow_id, [])
        
        total_steps = len(workflow)
        completed_steps = sum(1 for step in workflow if step.completed)
        
        return {
            "workflow_id": workflow_id,
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "progress_percentage": (completed_steps / total_steps * 100) if total_steps > 0 else 0,
            "current_step": next((step.description for step in workflow if not step.completed), None),
            "is_complete": completed_steps == total_steps
        }


# Global instance
_workflow_engine: Optional[CapabilityWorkflowEngine] = None


def get_workflow_engine() -> CapabilityWorkflowEngine:
    """Get the global workflow engine instance"""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = CapabilityWorkflowEngine()
    return _workflow_engine
