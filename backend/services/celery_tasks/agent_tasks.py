"""
Agent-Specific Celery Tasks
Background processing for individual agents
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

from services.celery_app import celery_app, update_task_progress, TaskStatus

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="agents.research_task")
def research_background_task(
    self,
    user_id: str,
    conversation_id: str,
    query: str,
    persona: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Background task specifically for research operations"""
    try:
        logger.info(f"🔬 ASYNC RESEARCH: Starting background research for user {user_id}")
        
        update_task_progress(self, 1, 4, "Initializing research agent...")
        
        # Run async research processing
        result = asyncio.run(_async_research_processing(
            self, user_id, conversation_id, query, persona
        ))
        
        logger.info(f"✅ ASYNC RESEARCH: Completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ ASYNC RESEARCH ERROR: {e}")
        
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "Research processing failed",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": False,
            "error": str(e),
            "message": "Background research processing failed"
        }


async def _async_research_processing(
    task,
    user_id: str,
    conversation_id: str,
    query: str,
    persona: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Internal async function for research processing"""
    try:
        from services.langgraph_agents import ResearchAgentHITL
        
        update_task_progress(task, 2, 4, "Loading conversation context...")
        
        # Load conversation history for context
        from services.conversation_service import get_conversation_service
        conv_service = await get_conversation_service()
        conversation_messages = await conv_service.get_messages(conversation_id, user_id)
        
        update_task_progress(task, 3, 4, "Executing research with full context...")
        
        # Create research agent and process
        research_agent = ResearchAgentHITL()
        agent_state = {
            "messages": conversation_messages,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "current_query": query,
            "persona": persona,
            "agent_insights": {"research_agent": {}},
            "shared_memory": {}
        }
        
        result_state = await research_agent.process(agent_state)
        
        update_task_progress(task, 4, 4, "Research completed!")
        
        task.update_state(
            state=TaskStatus.SUCCESS,
            meta={
                "result": result_state,
                "message": "Research completed successfully",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": True,
            "response": result_state.get("agent_results", {}).get("response", "Research completed"),
            "agent_type": "research",
            "task_id": task.request.id
        }
        
    except Exception as e:
        logger.error(f"❌ Async research processing error: {e}")
        
        task.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "Research processing error",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": False,
            "error": str(e),
            "message": "Research processing failed"
        }


@celery_app.task(bind=True, name="agents.coding_task")
def coding_background_task(
    self,
    user_id: str,
    conversation_id: str,
    query: str,
    persona: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Background task specifically for coding operations"""
    try:
        logger.info(f"💻 ASYNC CODING: Starting background coding for user {user_id}")
        
        update_task_progress(self, 1, 3, "Initializing coding agent...")
        
        result = asyncio.run(_async_coding_processing(
            self, user_id, conversation_id, query, persona
        ))
        
        logger.info(f"✅ ASYNC CODING: Completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ ASYNC CODING ERROR: {e}")
        
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "Coding processing failed",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": False,
            "error": str(e),
            "message": "Background coding processing failed"
        }


async def _async_coding_processing(
    task,
    user_id: str,
    conversation_id: str,
    query: str,
    persona: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Internal async function for coding processing"""
    try:
        from services.langgraph_agents import ChatAgent
        
        update_task_progress(task, 2, 3, "Executing task with chat agent (coding agent removed)...")
        
        # Create chat agent and process (coding agent removed for simplicity)
        chat_agent = ChatAgent()
        agent_state = {
            "messages": [],  # Simplified for now
            "user_id": user_id,
            "conversation_id": conversation_id,
            "current_query": query,
            "persona": persona,
            "agent_insights": {"coding_agent": {}},
            "shared_memory": {}
        }
        
        result_state = await chat_agent.process(agent_state)
        
        update_task_progress(task, 3, 3, "Coding completed!")
        
        task.update_state(
            state=TaskStatus.SUCCESS,
            meta={
                "result": result_state,
                "message": "Coding completed successfully",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": True,
            "response": result_state.get("agent_results", {}).get("response", "Coding completed"),
            "agent_type": "coding",
            "task_id": task.request.id
        }
        
    except Exception as e:
        logger.error(f"❌ Async coding processing error: {e}")
        
        task.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "Coding processing error",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": False,
            "error": str(e),
            "message": "Coding processing failed"
        }


@celery_app.task(bind=True, name="agents.batch_processing")
def batch_agent_processing(
    self,
    tasks: list,
    user_id: str,
    conversation_id: str
) -> Dict[str, Any]:
    """Process multiple agent tasks in sequence"""
    try:
        logger.info(f"📦 BATCH PROCESSING: {len(tasks)} tasks for user {user_id}")
        
        results = []
        total_tasks = len(tasks)
        
        for i, task_config in enumerate(tasks):
            update_task_progress(
                self, i + 1, total_tasks, 
                f"Processing task {i+1}/{total_tasks}: {task_config.get('agent_type', 'unknown')}"
            )
            
            # Process individual task based on type
            agent_type = task_config.get("agent_type")
            query = task_config.get("query")
            persona = task_config.get("persona")
            
            if agent_type == "research":
                result = asyncio.run(_async_research_processing(
                    self, user_id, conversation_id, query, persona
                ))
            elif agent_type == "coding":
                result = asyncio.run(_async_coding_processing(
                    self, user_id, conversation_id, query, persona
                ))
            else:
                result = {"success": False, "error": f"Unknown agent type: {agent_type}"}
            
            results.append({
                "task_index": i,
                "agent_type": agent_type,
                "result": result
            })
        
        self.update_state(
            state=TaskStatus.SUCCESS,
            meta={
                "results": results,
                "message": f"Batch processing completed: {len(tasks)} tasks",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": True,
            "results": results,
            "total_tasks": total_tasks,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"❌ BATCH PROCESSING ERROR: {e}")
        
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={
                "error": str(e),
                "message": "Batch processing failed",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": False,
            "error": str(e),
            "message": "Batch processing failed"
        }
