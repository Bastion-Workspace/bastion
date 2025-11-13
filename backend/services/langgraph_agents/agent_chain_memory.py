"""
Agent Chain Memory System
Enhanced shared memory for agent-to-agent data passing and workflow coordination
The "Square Deal" for agent collaboration!
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class AgentChainStatus(str, Enum):
    """Status values for agent chain execution"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataHandoffType(str, Enum):
    """Types of data handoffs between agents"""
    RESEARCH_TO_ANALYSIS = "research_to_analysis"
    ANALYSIS_TO_CODING = "analysis_to_coding"
    RESEARCH_TO_CODING = "research_to_coding"
    CODING_TO_VALIDATION = "coding_to_validation"
    MULTI_RESEARCH_SYNTHESIS = "multi_research_synthesis"
    ITERATIVE_REFINEMENT = "iterative_refinement"


@dataclass
class AgentResult:
    """Structured result from an agent execution"""
    agent_type: str
    agent_id: str
    execution_id: str
    status: AgentChainStatus
    response: str
    data_outputs: Dict[str, Any]
    tools_used: List[str]
    execution_time: float
    timestamp: str
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class ChainWorkflowStep:
    """Individual step in an agent chain workflow"""
    step_id: str
    agent_type: str
    task_description: str
    input_requirements: List[str]
    output_specifications: List[str]
    depends_on: List[str]  # Previous step IDs this depends on
    status: AgentChainStatus
    result: Optional[AgentResult] = None
    retry_count: int = 0
    max_retries: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class DataHandoff:
    """Structured data handoff between agents"""
    handoff_id: str
    handoff_type: DataHandoffType
    from_agent: str
    to_agent: str
    data_package: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: str
    processing_instructions: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


class AgentChainMemory:
    """Enhanced shared memory system for agent chaining workflows"""
    
    def __init__(self):
        self.chain_history: List[Dict[str, Any]] = []
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.data_handoffs: Dict[str, DataHandoff] = {}
        self.agent_results: Dict[str, AgentResult] = {}
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
        # Initialize with common workflow templates
        self._initialize_workflow_templates()
    
    def _initialize_workflow_templates(self):
        """Initialize common workflow templates for agent chaining"""
        
        # Research → Analysis → Synthesis Pattern
        self.workflow_templates["research_analysis_synthesis"] = {
            "name": "Research Analysis Synthesis",
            "description": "Multi-stage research with analysis and synthesis",
            "steps": [
                {
                    "step_id": "research_phase",
                    "agent_type": "research",
                    "task_description": "Conduct comprehensive research on the topic",
                    "input_requirements": ["user_query", "research_scope"],
                    "output_specifications": ["research_findings", "source_citations", "confidence_metrics"],
                    "depends_on": []
                },
                {
                    "step_id": "analysis_phase", 
                    "agent_type": "chat",
                    "task_description": "Analyze research findings for patterns and insights",
                    "input_requirements": ["research_findings", "analysis_framework"],
                    "output_specifications": ["key_insights", "patterns_identified", "recommendations"],
                    "depends_on": ["research_phase"]
                },
                {
                    "step_id": "synthesis_phase",
                    "agent_type": "chat",
                    "task_description": "Synthesize analysis into comprehensive response",
                    "input_requirements": ["key_insights", "user_intent"],
                    "output_specifications": ["final_response", "supporting_evidence"],
                    "depends_on": ["analysis_phase"]
                }
            ]
        }
        
        # Research → Coding Implementation Pattern
        self.workflow_templates["research_coding_implementation"] = {
            "name": "Research to Coding Implementation", 
            "description": "Research technical solutions then implement code",
            "steps": [
                {
                    "step_id": "technical_research",
                    "agent_type": "research",
                    "task_description": "Research technical approaches and best practices",
                    "input_requirements": ["technical_requirements", "constraints"],
                    "output_specifications": ["technical_approaches", "code_examples", "best_practices"],
                    "depends_on": []
                },
                {
                    "step_id": "solution_design",
                    "agent_type": "coding",
                    "task_description": "Design solution architecture based on research",
                    "input_requirements": ["technical_approaches", "requirements"],
                    "output_specifications": ["solution_architecture", "implementation_plan"],
                    "depends_on": ["technical_research"]
                },
                {
                    "step_id": "code_implementation",
                    "agent_type": "coding", 
                    "task_description": "Implement the designed solution",
                    "input_requirements": ["solution_architecture", "implementation_plan"],
                    "output_specifications": ["implemented_code", "documentation", "usage_examples"],
                    "depends_on": ["solution_design"]
                }
            ]
        }
        
        # Parallel Research → Synthesis Pattern
        self.workflow_templates["parallel_research_synthesis"] = {
            "name": "Parallel Research Synthesis",
            "description": "Multiple research agents exploring different aspects",
            "steps": [
                {
                    "step_id": "primary_research",
                    "agent_type": "research",
                    "task_description": "Primary research on main topic",
                    "input_requirements": ["main_topic", "research_depth"],
                    "output_specifications": ["primary_findings", "core_sources"],
                    "depends_on": []
                },
                {
                    "step_id": "secondary_research", 
                    "agent_type": "research",
                    "task_description": "Secondary research on related aspects",
                    "input_requirements": ["related_topics", "research_scope"],
                    "output_specifications": ["secondary_findings", "supporting_sources"],
                    "depends_on": []
                },
                {
                    "step_id": "synthesis_coordination",
                    "agent_type": "chat",
                    "task_description": "Synthesize findings from multiple research streams",
                    "input_requirements": ["primary_findings", "secondary_findings"],
                    "output_specifications": ["comprehensive_analysis", "integrated_insights"],
                    "depends_on": ["primary_research", "secondary_research"]
                }
            ]
        }
        
        logger.info(f"✅ Initialized {len(self.workflow_templates)} workflow templates")
    
    def create_workflow(
        self, 
        workflow_id: str, 
        template_name: str, 
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new agent chain workflow from template"""
        
        try:
            if template_name not in self.workflow_templates:
                raise ValueError(f"Unknown workflow template: {template_name}")
            
            template = self.workflow_templates[template_name]
            
            # Create workflow instance
            workflow = {
                "workflow_id": workflow_id,
                "template_name": template_name,
                "status": AgentChainStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "user_context": user_context,
                "steps": [],
                "current_step": 0,
                "completed_steps": [],
                "failed_steps": [],
                "total_execution_time": 0.0
            }
            
            # Create workflow steps from template
            for step_template in template["steps"]:
                step = ChainWorkflowStep(
                    step_id=f"{workflow_id}_{step_template['step_id']}",
                    agent_type=step_template["agent_type"],
                    task_description=step_template["task_description"],
                    input_requirements=step_template["input_requirements"],
                    output_specifications=step_template["output_specifications"],
                    depends_on=[f"{workflow_id}_{dep}" for dep in step_template["depends_on"]],
                    status=AgentChainStatus.PENDING
                )
                workflow["steps"].append(step.to_dict())
            
            # Store active workflow
            self.active_workflows[workflow_id] = workflow
            
            logger.info(f"🔗 Created workflow {workflow_id} with {len(workflow['steps'])} steps")
            return workflow
            
        except Exception as e:
            logger.error(f"❌ Failed to create workflow {workflow_id}: {e}")
            raise
    
    def get_next_ready_step(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the next step that's ready to execute in the workflow"""
        
        try:
            if workflow_id not in self.active_workflows:
                return None
            
            workflow = self.active_workflows[workflow_id]
            
            for step in workflow["steps"]:
                # Skip if already completed or failed
                if step["status"] in [AgentChainStatus.COMPLETED, AgentChainStatus.FAILED]:
                    continue
                
                # Skip if already running
                if step["status"] == AgentChainStatus.RUNNING:
                    continue
                
                # Check if all dependencies are completed
                dependencies_met = True
                for dep_step_id in step["depends_on"]:
                    dep_step = self._find_step_by_id(workflow, dep_step_id)
                    if not dep_step or dep_step["status"] != AgentChainStatus.COMPLETED:
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    return step
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error finding next ready step for workflow {workflow_id}: {e}")
            return None
    
    def _find_step_by_id(self, workflow: Dict[str, Any], step_id: str) -> Optional[Dict[str, Any]]:
        """Find a step by ID in the workflow"""
        for step in workflow["steps"]:
            if step["step_id"] == step_id:
                return step
        return None
    
    def create_data_handoff(
        self,
        from_agent: str,
        to_agent: str,
        handoff_type: DataHandoffType,
        data_package: Dict[str, Any],
        processing_instructions: Optional[str] = None
    ) -> str:
        """Create a data handoff between agents"""
        
        try:
            handoff_id = f"handoff_{from_agent}_{to_agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            handoff = DataHandoff(
                handoff_id=handoff_id,
                handoff_type=handoff_type,
                from_agent=from_agent,
                to_agent=to_agent,
                data_package=data_package,
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "data_size": len(json.dumps(data_package)),
                    "keys_transferred": list(data_package.keys())
                },
                timestamp=datetime.now().isoformat(),
                processing_instructions=processing_instructions
            )
            
            self.data_handoffs[handoff_id] = handoff
            
            logger.info(f"🤝 Created data handoff {handoff_id}: {from_agent} → {to_agent}")
            return handoff_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create data handoff: {e}")
            raise
    
    def get_handoff_data(self, handoff_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from a handoff"""
        
        if handoff_id in self.data_handoffs:
            handoff = self.data_handoffs[handoff_id]
            return {
                "data_package": handoff.data_package,
                "metadata": handoff.metadata,
                "processing_instructions": handoff.processing_instructions,
                "from_agent": handoff.from_agent
            }
        return None
    
    def store_agent_result(
        self,
        agent_type: str,
        execution_id: str,
        response: str,
        data_outputs: Dict[str, Any],
        tools_used: List[str],
        execution_time: float,
        confidence_score: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> str:
        """Store results from an agent execution"""
        
        try:
            agent_id = f"{agent_type}_{execution_id}"
            
            result = AgentResult(
                agent_type=agent_type,
                agent_id=agent_id,
                execution_id=execution_id,
                status=AgentChainStatus.COMPLETED if not error_message else AgentChainStatus.FAILED,
                response=response,
                data_outputs=data_outputs,
                tools_used=tools_used,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                error_message=error_message,
                confidence_score=confidence_score
            )
            
            self.agent_results[agent_id] = result
            
            logger.info(f"💾 Stored agent result for {agent_id}")
            return agent_id
            
        except Exception as e:
            logger.error(f"❌ Failed to store agent result: {e}")
            raise
    
    def update_workflow_step_status(
        self,
        workflow_id: str,
        step_id: str,
        status: AgentChainStatus,
        result: Optional[AgentResult] = None
    ):
        """Update the status of a workflow step"""
        
        try:
            if workflow_id not in self.active_workflows:
                logger.warning(f"⚠️ Workflow {workflow_id} not found")
                return
            
            workflow = self.active_workflows[workflow_id]
            step = self._find_step_by_id(workflow, step_id)
            
            if not step:
                logger.warning(f"⚠️ Step {step_id} not found in workflow {workflow_id}")
                return
            
            step["status"] = status
            if result:
                step["result"] = result.to_dict()
            
            # Update workflow completion tracking
            if status == AgentChainStatus.COMPLETED:
                if step_id not in workflow["completed_steps"]:
                    workflow["completed_steps"].append(step_id)
            elif status == AgentChainStatus.FAILED:
                if step_id not in workflow["failed_steps"]:
                    workflow["failed_steps"].append(step_id)
            
            # Check if workflow is complete
            all_steps_done = all(
                step["status"] in [AgentChainStatus.COMPLETED, AgentChainStatus.FAILED]
                for step in workflow["steps"]
            )
            
            if all_steps_done:
                has_failures = any(step["status"] == AgentChainStatus.FAILED for step in workflow["steps"])
                workflow["status"] = AgentChainStatus.FAILED if has_failures else AgentChainStatus.COMPLETED
                workflow["completed_at"] = datetime.now().isoformat()
                
                logger.info(f"🏁 Workflow {workflow_id} completed with status: {workflow['status']}")
            
        except Exception as e:
            logger.error(f"❌ Error updating workflow step status: {e}")
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow"""
        
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            
            return {
                "workflow_id": workflow_id,
                "status": workflow["status"],
                "completed_steps": len(workflow["completed_steps"]),
                "total_steps": len(workflow["steps"]),
                "failed_steps": len(workflow["failed_steps"]),
                "current_step": workflow.get("current_step", 0),
                "execution_time": workflow.get("total_execution_time", 0.0)
            }
        
        return None
    
    def cleanup_completed_workflows(self, max_age_hours: int = 24):
        """Clean up old completed workflows to manage memory"""
        
        try:
            current_time = datetime.now()
            workflows_to_remove = []
            
            for workflow_id, workflow in self.active_workflows.items():
                if workflow["status"] in [AgentChainStatus.COMPLETED, AgentChainStatus.FAILED]:
                    completed_at = datetime.fromisoformat(workflow.get("completed_at", workflow["created_at"]))
                    age_hours = (current_time - completed_at).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        workflows_to_remove.append(workflow_id)
                        # Archive to chain history
                        self.chain_history.append(workflow)
            
            for workflow_id in workflows_to_remove:
                del self.active_workflows[workflow_id]
                logger.info(f"🧹 Archived completed workflow {workflow_id}")
            
            if workflows_to_remove:
                logger.info(f"🧹 Cleaned up {len(workflows_to_remove)} completed workflows")
                
        except Exception as e:
            logger.error(f"❌ Error cleaning up workflows: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for agent chaining"""
        
        try:
            total_workflows = len(self.active_workflows) + len(self.chain_history)
            active_workflows = len(self.active_workflows)
            completed_workflows = len([w for w in self.chain_history if w.get("status") == AgentChainStatus.COMPLETED])
            
            avg_execution_time = 0.0
            if self.chain_history:
                total_time = sum(w.get("total_execution_time", 0.0) for w in self.chain_history)
                avg_execution_time = total_time / len(self.chain_history)
            
            return {
                "total_workflows": total_workflows,
                "active_workflows": active_workflows, 
                "completed_workflows": completed_workflows,
                "success_rate": completed_workflows / max(total_workflows, 1),
                "average_execution_time": avg_execution_time,
                "total_handoffs": len(self.data_handoffs),
                "available_templates": list(self.workflow_templates.keys())
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating performance metrics: {e}")
            return {}


# Global instance for agent chain memory
_agent_chain_memory = None

def get_agent_chain_memory() -> AgentChainMemory:
    """Get the global agent chain memory instance"""
    global _agent_chain_memory
    if _agent_chain_memory is None:
        _agent_chain_memory = AgentChainMemory()
        logger.info("🧠 Initialized global agent chain memory")
    return _agent_chain_memory
