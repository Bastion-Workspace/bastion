"""
Agent Workflow Engine
Orchestrates complex multi-agent workflows with chaining, parallel execution, and coordination
Theodore Roosevelt's "Big Stick" for agent coordination!
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator
import json

from .agent_chain_memory import (
    AgentChainMemory, 
    get_agent_chain_memory,
    AgentChainStatus,
    DataHandoffType,
    AgentResult
)

logger = logging.getLogger(__name__)


class AgentWorkflowEngine:
    """Advanced workflow engine for coordinating multi-agent chains"""
    
    def __init__(self):
        self.chain_memory = get_agent_chain_memory()
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.agent_registry = {}
        self._initialize_agent_registry()
    
    def _initialize_agent_registry(self):
        """Initialize registry of available agents"""
        # NOTE: Agent chaining is deprecated - all agents have been migrated to llm-orchestrator
        # This registry is kept for backward compatibility with agent_chaining_api
        # but the referenced agents no longer exist in the backend
        self.agent_registry = {
            # All agents migrated to llm-orchestrator gRPC service
            # ResearchAgent and DirectAgent removed - use llm-orchestrator instead
        }
        logger.warning(f"⚠️ Agent registry is empty - all agents migrated to llm-orchestrator. Agent chaining API may not function properly.")
    
    async def execute_workflow(
        self,
        workflow_template: str,
        user_context: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a multi-agent workflow with real-time streaming"""
        
        if not workflow_id:
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            yield {
                "type": "workflow_started",
                "workflow_id": workflow_id,
                "template": workflow_template,
                "message": f"🚀 Starting {workflow_template} workflow...",
                "timestamp": datetime.now().isoformat()
            }
            
            # Create workflow from template
            workflow = self.chain_memory.create_workflow(
                workflow_id=workflow_id,
                template_name=workflow_template,
                user_context=user_context
            )
            
            self.active_executions[workflow_id] = {
                "start_time": datetime.now(),
                "status": AgentChainStatus.RUNNING
            }
            
            yield {
                "type": "workflow_planned",
                "workflow_id": workflow_id,
                "total_steps": len(workflow["steps"]),
                "message": f"📋 Workflow planned with {len(workflow['steps'])} steps",
                "timestamp": datetime.now().isoformat()
            }
            
            # Execute workflow steps
            async for update in self._execute_workflow_steps(workflow_id, workflow):
                yield update
            
            # Final workflow status
            final_status = self.chain_memory.get_workflow_status(workflow_id)
            
            yield {
                "type": "workflow_completed",
                "workflow_id": workflow_id,
                "final_status": final_status,
                "message": f"✅ Workflow completed: {final_status['status']}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Workflow execution failed for {workflow_id}: {e}")
            yield {
                "type": "workflow_error",
                "workflow_id": workflow_id,
                "error": str(e),
                "message": f"❌ Workflow failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        finally:
            # Cleanup execution tracking
            if workflow_id in self.active_executions:
                del self.active_executions[workflow_id]
    
    async def _execute_workflow_steps(
        self,
        workflow_id: str,
        workflow: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute workflow steps with proper dependency management"""
        
        try:
            max_iterations = 50  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                # Get next ready step
                next_step = self.chain_memory.get_next_ready_step(workflow_id)
                
                if not next_step:
                    # Check if we're done or stuck
                    workflow_status = self.chain_memory.get_workflow_status(workflow_id)
                    if workflow_status and workflow_status["status"] in [AgentChainStatus.COMPLETED, AgentChainStatus.FAILED]:
                        break
                    
                    # Check for deadlock
                    pending_steps = [s for s in workflow["steps"] if s["status"] == AgentChainStatus.PENDING]
                    if pending_steps:
                        logger.warning(f"⚠️ Possible deadlock in workflow {workflow_id}, {len(pending_steps)} pending steps")
                        # Try to execute a pending step anyway
                        next_step = pending_steps[0]
                    else:
                        break
                
                yield {
                    "type": "step_starting",
                    "workflow_id": workflow_id,
                    "step_id": next_step["step_id"],
                    "agent_type": next_step["agent_type"],
                    "message": f"🎯 Starting {next_step['agent_type']} agent: {next_step['task_description']}",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Mark step as running
                self.chain_memory.update_workflow_step_status(
                    workflow_id,
                    next_step["step_id"],
                    AgentChainStatus.RUNNING
                )
                
                # Execute the step
                async for step_update in self._execute_step(workflow_id, next_step):
                    yield step_update
            
            if iteration >= max_iterations:
                logger.error(f"❌ Workflow {workflow_id} exceeded maximum iterations")
                yield {
                    "type": "workflow_error",
                    "workflow_id": workflow_id,
                    "error": "Maximum iterations exceeded",
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"❌ Error executing workflow steps: {e}")
            yield {
                "type": "step_error",
                "workflow_id": workflow_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _execute_step(
        self,
        workflow_id: str,
        step: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a single workflow step"""
        
        step_start_time = datetime.now()
        
        try:
            # Prepare step input data
            step_input = await self._prepare_step_input(workflow_id, step)
            
            yield {
                "type": "step_prepared",
                "workflow_id": workflow_id,
                "step_id": step["step_id"],
                "input_keys": list(step_input.keys()),
                "message": f"📦 Prepared input for {step['agent_type']} agent",
                "timestamp": datetime.now().isoformat()
            }
            
            # Get and initialize agent
            agent = await self._get_agent_instance(step["agent_type"])
            
            # Create agent state
            agent_state = {
                "user_id": step_input.get("user_id", "workflow_engine"),
                "conversation_id": step_input.get("conversation_id"),
                "current_query": step_input.get("task_query", step["task_description"]),
                "messages": step_input.get("messages", []),
                "workflow_context": {
                    "workflow_id": workflow_id,
                    "step_id": step["step_id"],
                    "step_input": step_input,
                    "chain_data": step_input.get("chain_data", {})
                },
                "shared_memory": {},
                "persona": step_input.get("persona")
            }
            
            yield {
                "type": "step_executing",
                "workflow_id": workflow_id,
                "step_id": step["step_id"],
                "agent_type": step["agent_type"],
                "message": f"⚡ Executing {step['agent_type']} agent...",
                "timestamp": datetime.now().isoformat()
            }
            
            # Execute agent
            result_state = await agent.process(agent_state)
            
            # Extract results
            execution_time = (datetime.now() - step_start_time).total_seconds()
            response = result_state.get("agent_results", {}).get("response", "No response generated")
            tools_used = result_state.get("tools_used", [])
            
            # Create agent result
            agent_result = AgentResult(
                agent_type=step["agent_type"],
                agent_id=f"{step['agent_type']}_{step['step_id']}",
                execution_id=step["step_id"],
                status=AgentChainStatus.COMPLETED,
                response=response,
                data_outputs=self._extract_data_outputs(result_state),
                tools_used=tools_used,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                confidence_score=result_state.get("confidence_level")
            )
            
            # Store result in chain memory
            self.chain_memory.store_agent_result(
                agent_type=step["agent_type"],
                execution_id=step["step_id"],
                response=response,
                data_outputs=agent_result.data_outputs,
                tools_used=tools_used,
                execution_time=execution_time,
                confidence_score=agent_result.confidence_score
            )
            
            # Update step status
            self.chain_memory.update_workflow_step_status(
                workflow_id,
                step["step_id"],
                AgentChainStatus.COMPLETED,
                agent_result
            )
            
            # Create data handoffs for dependent steps
            await self._create_step_handoffs(workflow_id, step, agent_result)
            
            yield {
                "type": "step_completed",
                "workflow_id": workflow_id,
                "step_id": step["step_id"],
                "agent_type": step["agent_type"],
                "execution_time": execution_time,
                "tools_used": tools_used,
                "response_preview": response[:200] + "..." if len(response) > 200 else response,
                "message": f"✅ {step['agent_type']} agent completed in {execution_time:.2f}s",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Step execution failed: {e}")
            
            # Mark step as failed
            self.chain_memory.update_workflow_step_status(
                workflow_id,
                step["step_id"],
                AgentChainStatus.FAILED
            )
            
            yield {
                "type": "step_failed",
                "workflow_id": workflow_id,
                "step_id": step["step_id"],
                "error": str(e),
                "message": f"❌ Step failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _prepare_step_input(self, workflow_id: str, step: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input data for a workflow step"""
        
        try:
            workflow = self.chain_memory.active_workflows[workflow_id]
            user_context = workflow["user_context"]
            
            step_input = {
                "user_id": user_context.get("user_id"),
                "conversation_id": user_context.get("conversation_id"),
                "persona": user_context.get("persona"),
                "messages": user_context.get("messages", []),
                "task_query": step["task_description"],
                "chain_data": {}
            }
            
            # Collect data from dependent steps
            for dep_step_id in step["depends_on"]:
                dep_step = self.chain_memory._find_step_by_id(workflow, dep_step_id)
                if dep_step and dep_step.get("result"):
                    step_input["chain_data"][dep_step_id] = {
                        "agent_type": dep_step["agent_type"],
                        "response": dep_step["result"]["response"],
                        "data_outputs": dep_step["result"]["data_outputs"],
                        "tools_used": dep_step["result"]["tools_used"]
                    }
            
            # If this step has dependencies, modify the task query to include context
            if step["depends_on"] and step_input["chain_data"]:
                context_summary = self._build_chain_context_summary(step_input["chain_data"])
                step_input["task_query"] = f"""WORKFLOW STEP: {step['task_description']}

PREVIOUS STEP CONTEXT:
{context_summary}

USER ORIGINAL REQUEST: {user_context.get('original_query', 'Not specified')}

Please complete this workflow step using the context from previous steps."""
            
            return step_input
            
        except Exception as e:
            logger.error(f"❌ Error preparing step input: {e}")
            return {"task_query": step["task_description"]}
    
    def _build_chain_context_summary(self, chain_data: Dict[str, Any]) -> str:
        """Build a summary of previous step results for context"""
        
        context_parts = []
        
        for step_id, step_data in chain_data.items():
            agent_type = step_data["agent_type"]
            response_preview = step_data["response"][:300] + "..." if len(step_data["response"]) > 300 else step_data["response"]
            tools_used = step_data.get("tools_used", [])
            
            context_parts.append(f"""
STEP: {step_id} ({agent_type} agent)
TOOLS USED: {', '.join(tools_used) if tools_used else 'None'}
RESULT: {response_preview}
""")
        
        return "\n".join(context_parts)
    
    def _extract_data_outputs(self, result_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data outputs from agent result state"""
        
        data_outputs = {}
        
        # Extract from shared memory
        shared_memory = result_state.get("shared_memory", {})
        if shared_memory:
            data_outputs["shared_memory"] = shared_memory
        
        # Extract search results
        if "search_results" in shared_memory:
            data_outputs["search_results"] = shared_memory["search_results"]
        
        # Extract tool results
        if "search_history" in shared_memory:
            data_outputs["tools_executed"] = shared_memory["search_history"]
        
        # Extract confidence metrics
        if "confidence_level" in result_state:
            data_outputs["confidence_level"] = result_state["confidence_level"]
        
        # Extract any workflow-specific outputs
        workflow_context = result_state.get("workflow_context", {})
        if workflow_context:
            data_outputs["workflow_context"] = workflow_context
        
        return data_outputs
    
    async def _create_step_handoffs(
        self,
        workflow_id: str,
        completed_step: Dict[str, Any],
        agent_result: AgentResult
    ):
        """Create data handoffs for steps that depend on this completed step"""
        
        try:
            workflow = self.chain_memory.active_workflows[workflow_id]
            
            # Find dependent steps
            dependent_steps = [
                step for step in workflow["steps"]
                if completed_step["step_id"] in step["depends_on"]
            ]
            
            for dep_step in dependent_steps:
                # Determine handoff type
                handoff_type = self._determine_handoff_type(
                    completed_step["agent_type"],
                    dep_step["agent_type"]
                )
                
                # Create handoff
                handoff_id = self.chain_memory.create_data_handoff(
                    from_agent=agent_result.agent_id,
                    to_agent=f"{dep_step['agent_type']}_{dep_step['step_id']}",
                    handoff_type=handoff_type,
                    data_package={
                        "response": agent_result.response,
                        "data_outputs": agent_result.data_outputs,
                        "tools_used": agent_result.tools_used,
                        "execution_context": {
                            "workflow_id": workflow_id,
                            "step_id": completed_step["step_id"],
                            "task_description": completed_step["task_description"]
                        }
                    },
                    processing_instructions=f"Process results from {completed_step['agent_type']} step for {dep_step['task_description']}"
                )
                
                logger.info(f"🤝 Created handoff {handoff_id} for step dependency")
                
        except Exception as e:
            logger.error(f"❌ Error creating step handoffs: {e}")
    
    def _determine_handoff_type(self, from_agent: str, to_agent: str) -> DataHandoffType:
        """Determine the appropriate handoff type between agents"""
        
        handoff_map = {
            ("research", "chat"): DataHandoffType.RESEARCH_TO_ANALYSIS,
            ("research", "coding"): DataHandoffType.RESEARCH_TO_CODING,
            ("chat", "coding"): DataHandoffType.ANALYSIS_TO_CODING,
            ("coding", "chat"): DataHandoffType.CODING_TO_VALIDATION
        }
        
        return handoff_map.get((from_agent, to_agent), DataHandoffType.ITERATIVE_REFINEMENT)
    
    async def _get_agent_instance(self, agent_type: str):
        """Get an instance of the specified agent type"""
        
        try:
            if agent_type not in self.agent_registry:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            agent_info = self.agent_registry[agent_type]
            module_path = agent_info["module"]
            class_name = agent_info["class"]
            
            # Dynamic import
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            
            return agent_class()
            
        except Exception as e:
            logger.error(f"❌ Failed to get agent instance for {agent_type}: {e}")
            raise
    
    async def execute_simple_chain(
        self,
        agents: List[str],
        user_context: Dict[str, Any],
        chain_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a simple sequential chain of agents"""
        
        if not chain_id:
            chain_id = f"chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            yield {
                "type": "chain_started",
                "chain_id": chain_id,
                "agents": agents,
                "message": f"🔗 Starting agent chain: {' → '.join(agents)}",
                "timestamp": datetime.now().isoformat()
            }
            
            chain_results = []
            chain_data = {}
            
            for i, agent_type in enumerate(agents):
                yield {
                    "type": "chain_step",
                    "chain_id": chain_id,
                    "step": i + 1,
                    "total_steps": len(agents),
                    "agent_type": agent_type,
                    "message": f"🎯 Executing {agent_type} agent ({i+1}/{len(agents)})",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Prepare agent state
                agent_state = {
                    "user_id": user_context.get("user_id"),
                    "conversation_id": user_context.get("conversation_id"),
                    "current_query": user_context.get("query"),
                    "messages": user_context.get("messages", []),
                    "chain_context": {
                        "chain_id": chain_id,
                        "step": i + 1,
                        "previous_results": chain_results,
                        "chain_data": chain_data
                    },
                    "shared_memory": {},
                    "persona": user_context.get("persona")
                }
                
                # Modify query if this isn't the first agent
                if i > 0 and chain_results:
                    context_summary = self._build_simple_chain_context(chain_results)
                    agent_state["current_query"] = f"""AGENT CHAIN STEP {i+1}: {user_context.get('query')}

PREVIOUS AGENT RESULTS:
{context_summary}

Please build upon these previous results to complete your part of the chain."""
                
                # Execute agent
                agent = await self._get_agent_instance(agent_type)
                result_state = await agent.process(agent_state)
                
                # Store results
                response = result_state.get("agent_results", {}).get("response", "No response")
                tools_used = result_state.get("tools_used", [])
                
                result = {
                    "agent_type": agent_type,
                    "step": i + 1,
                    "response": response,
                    "tools_used": tools_used,
                    "data_outputs": self._extract_data_outputs(result_state)
                }
                
                chain_results.append(result)
                chain_data[f"step_{i+1}_{agent_type}"] = result
                
                yield {
                    "type": "chain_step_completed",
                    "chain_id": chain_id,
                    "step": i + 1,
                    "agent_type": agent_type,
                    "response_preview": response[:200] + "..." if len(response) > 200 else response,
                    "tools_used": tools_used,
                    "message": f"✅ {agent_type} completed step {i+1}",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Final synthesis
            final_response = chain_results[-1]["response"]  # Last agent's response
            
            yield {
                "type": "chain_completed",
                "chain_id": chain_id,
                "final_response": final_response,
                "total_steps": len(agents),
                "message": f"✅ Agent chain completed: {' → '.join(agents)}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Simple chain execution failed: {e}")
            yield {
                "type": "chain_error",
                "chain_id": chain_id,
                "error": str(e),
                "message": f"❌ Chain failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_simple_chain_context(self, chain_results: List[Dict[str, Any]]) -> str:
        """Build context summary for simple chain execution"""
        
        context_parts = []
        
        for result in chain_results:
            agent_type = result["agent_type"]
            step = result["step"]
            response_preview = result["response"][:200] + "..." if len(result["response"]) > 200 else result["response"]
            tools_used = result.get("tools_used", [])
            
            context_parts.append(f"""
STEP {step} ({agent_type} agent):
TOOLS: {', '.join(tools_used) if tools_used else 'None'}
RESULT: {response_preview}
""")
        
        return "\n".join(context_parts)
    
    def get_active_executions(self) -> Dict[str, Any]:
        """Get information about currently active executions"""
        
        return {
            "active_workflows": len(self.active_executions),
            "executions": {
                exec_id: {
                    "start_time": exec_info["start_time"].isoformat(),
                    "status": exec_info["status"],
                    "duration_seconds": (datetime.now() - exec_info["start_time"]).total_seconds()
                }
                for exec_id, exec_info in self.active_executions.items()
            }
        }


# Global workflow engine instance
_workflow_engine = None

def get_workflow_engine() -> AgentWorkflowEngine:
    """Get the global workflow engine instance"""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = AgentWorkflowEngine()
        logger.info("🏭 Initialized global agent workflow engine")
    return _workflow_engine
