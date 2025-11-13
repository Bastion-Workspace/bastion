"""
Celery Utilities
Safe exception handling and serialization for Celery tasks
"""

import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


def safe_serialize_error(exception: Exception, context: str = "") -> Dict[str, Any]:
    """
    Safely serialize an exception for Celery storage
    Avoids the 'Exception information must include the exception type' error
    """
    try:
        error_data = {
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            "error_context": context,
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc() if logger.isEnabledFor(logging.DEBUG) else None
        }
        
        # Test serialization to ensure it's JSON-safe
        json.dumps(error_data)
        return error_data
        
    except Exception as e:
        logger.warning(f"Failed to serialize exception {exception}: {e}")
        # Fallback to basic error info
        return {
            "error_type": "SerializationError",
            "error_message": f"Failed to serialize original error: {str(exception)[:500]}",
            "error_context": context,
            "timestamp": datetime.now().isoformat(),
            "original_error_type": type(exception).__name__
        }


def safe_update_task_state(task, state: str, meta: Dict[str, Any]) -> bool:
    """
    Safely update Celery task state with proper error handling
    """
    try:
        # Ensure meta is JSON-serializable
        serializable_meta = make_json_safe(meta)
        
        task.update_state(
            state=state,
            meta=serializable_meta
        )
        return True
        
    except Exception as e:
        logger.error(f"Failed to update task state: {e}")
        
        # Fallback with minimal safe data
        try:
            safe_meta = {
                "error": "Failed to update task state",
                "timestamp": datetime.now().isoformat(),
                "original_state": state
            }
            task.update_state(state="FAILURE", meta=safe_meta)
        except Exception as e2:
            logger.error(f"Even fallback task state update failed: {e2}")
        
        return False


def make_json_safe(obj: Any, max_depth: int = 10) -> Any:
    """
    Recursively make an object JSON-serializable
    """
    if max_depth <= 0:
        return "<max_depth_exceeded>"
    
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(item, max_depth - 1) for item in obj]
    
    if isinstance(obj, dict):
        safe_dict = {}
        for key, value in obj.items():
            # Ensure key is a string
            safe_key = str(key)
            safe_dict[safe_key] = make_json_safe(value, max_depth - 1)
        return safe_dict
    
    if hasattr(obj, '__dict__'):
        return make_json_safe(obj.__dict__, max_depth - 1)
    
    # For other types, convert to string
    try:
        return str(obj)[:1000]  # Limit string length
    except:
        return "<unserializable_object>"


def safe_task_wrapper(task_func):
    """
    Decorator for Celery tasks to provide safe exception handling
    """
    def wrapper(self, *args, **kwargs):
        try:
            return task_func(self, *args, **kwargs)
        except Exception as e:
            logger.error(f"Task {task_func.__name__} failed: {e}")
            
            # Create safe error response
            error_data = safe_serialize_error(e, f"Task: {task_func.__name__}")
            
            # Update task state safely
            safe_update_task_state(
                self, 
                "FAILURE", 
                {
                    "error": error_data["error_message"],
                    "error_type": error_data["error_type"],
                    "timestamp": error_data["timestamp"],
                    "context": error_data["error_context"]
                }
            )
            
            # Return safe error response
            return {
                "success": False,
                "error": error_data["error_message"],
                "error_type": error_data["error_type"],
                "timestamp": error_data["timestamp"]
            }
    
    return wrapper


def clean_result_for_storage(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean a result dictionary for safe Celery storage
    """
    try:
        # Make the entire result JSON-safe
        cleaned_result = make_json_safe(result)
        
        # Ensure critical fields are present and safe
        if "success" not in cleaned_result:
            cleaned_result["success"] = False
        
        if "timestamp" not in cleaned_result:
            cleaned_result["timestamp"] = datetime.now().isoformat()
        
        # Limit large text fields
        if "response" in cleaned_result and isinstance(cleaned_result["response"], str):
            if len(cleaned_result["response"]) > 50000:  # 50KB limit
                cleaned_result["response"] = cleaned_result["response"][:50000] + "...[truncated]"
        
        return cleaned_result
        
    except Exception as e:
        logger.error(f"Failed to clean result for storage: {e}")
        return {
            "success": False,
            "error": "Failed to process result",
            "timestamp": datetime.now().isoformat()
        }


def create_progress_meta(current_step: int, total_steps: int, message: str) -> Dict[str, Any]:
    """
    Create safe progress metadata for Celery task updates
    """
    return {
        "current_step": current_step,
        "total_steps": total_steps,
        "percentage": int((current_step / total_steps) * 100) if total_steps > 0 else 0,
        "message": str(message)[:500],  # Limit message length
        "timestamp": datetime.now().isoformat()
    }
