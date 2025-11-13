"""
Priority Queue Fix for Document Processing
Ensures zip files and new uploads get priority over reprocessing
"""

import asyncio
import heapq
from typing import List, Tuple, Any
from dataclasses import dataclass, field

@dataclass
class PriorityJob:
    """Job with priority for processing queue"""
    priority: int
    timestamp: float
    job_data: Any = field(compare=False)
    
    def __lt__(self, other):
        # Lower priority number = higher priority
        if self.priority != other.priority:
            return self.priority < other.priority
        # If same priority, older jobs first
        return self.timestamp < other.timestamp

class PriorityProcessingQueue:
    """Priority queue for document processing jobs"""
    
    def __init__(self):
        self._queue = []
        self._index = 0
        self._lock = asyncio.Lock()
    
    async def put(self, job_data: Any, priority: int = 5) -> None:
        """Add job with priority (lower number = higher priority)"""
        import time
        
        async with self._lock:
            priority_job = PriorityJob(
                priority=priority,
                timestamp=time.time(),
                job_data=job_data
            )
            heapq.heappush(self._queue, priority_job)
    
    async def get(self) -> Any:
        """Get highest priority job"""
        while True:
            async with self._lock:
                if self._queue:
                    priority_job = heapq.heappop(self._queue)
                    return priority_job.job_data
            
            # Wait if queue is empty
            await asyncio.sleep(0.1)
    
    def qsize(self) -> int:
        """Get queue size"""
        return len(self._queue)
    
    async def task_done(self):
        """Mark task as done (for compatibility)"""
        pass
    
    async def join(self):
        """Wait for all tasks to complete"""
        while self.qsize() > 0:
            await asyncio.sleep(0.1)

# Priority levels
PRIORITY_LEVELS = {
    'zip_upload': 1,      # Highest priority - new zip files
    'new_upload': 2,      # High priority - new single files  
    'url_import': 3,      # Medium priority - URL imports
    'reprocess': 8,       # Low priority - reprocessing
    'background': 10      # Lowest priority - background tasks
}
