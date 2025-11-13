"""
Org-Mode Time Tracking Service - Roosevelt's "Productivity Meter"

**BULLY!** Track every minute of your productive cavalry charge!

This service handles:
- Clock in/out on tasks
- LOGBOOK drawer management
- Time duration calculations
- Active clock tracking per user
- Time reports and statistics

Org-mode clocking format:
```org
* TODO Write documentation
:LOGBOOK:
CLOCK: [2025-10-22 Tue 10:00]--[2025-10-22 Tue 11:30] =>  1:30
CLOCK: [2025-10-22 Tue 14:00]--[2025-10-22 Tue 15:15] =>  1:15
:END:
```
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class OrgClockingService:
    """Service for org-mode time tracking"""
    
    def __init__(self):
        # Track active clocks per user
        # Format: {user_id: {'file_path': str, 'line_number': int, 'start_time': datetime, 'heading': str}}
        self._active_clocks = {}
    
    async def clock_in(
        self,
        user_id: str,
        file_path: str,
        line_number: int,
        heading: str
    ) -> Dict[str, Any]:
        """
        Clock in to a task
        
        **ROOSEVELT TIME TRACKING!** Start the clock!
        
        Args:
            user_id: User ID
            file_path: File path (e.g., "OrgMode/tasks.org")
            line_number: Line number of task heading (1-based)
            heading: Task heading text (for display)
        
        Returns:
            Dict with success status and clock details
        """
        try:
            logger.info(f"⏰ CLOCK IN: User {user_id} at {file_path}:{line_number}")
            
            # Check if user already has an active clock
            if user_id in self._active_clocks:
                existing_clock = self._active_clocks[user_id]
                logger.warning(f"⚠️ User {user_id} already has active clock at {existing_clock['file_path']}")
                return {
                    "success": False,
                    "error": "already_clocked_in",
                    "message": f"Already clocked into: {existing_clock['heading']}",
                    "active_clock": existing_clock
                }
            
            # Record active clock
            now = datetime.now()
            clock_data = {
                'file_path': file_path,
                'line_number': line_number,
                'start_time': now,
                'heading': heading
            }
            self._active_clocks[user_id] = clock_data
            
            logger.info(f"✅ Clock started: {heading}")
            
            return {
                "success": True,
                "clock_id": user_id,  # Simple clock ID = user_id for now
                "start_time": now.isoformat(),
                "heading": heading,
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"❌ Clock in failed: {e}", exc_info=True)
            raise
    
    async def clock_out(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Clock out of current task
        
        Writes CLOCK entry to LOGBOOK drawer
        
        Args:
            user_id: User ID
        
        Returns:
            Dict with success status and duration
        """
        try:
            logger.info(f"⏰ CLOCK OUT: User {user_id}")
            
            # Check if user has active clock
            if user_id not in self._active_clocks:
                return {
                    "success": False,
                    "error": "not_clocked_in",
                    "message": "No active clock found"
                }
            
            clock_data = self._active_clocks[user_id]
            end_time = datetime.now()
            start_time = clock_data['start_time']
            duration = end_time - start_time
            
            # Resolve file path
            full_path = await self._resolve_file_path(user_id, clock_data['file_path'])
            
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {clock_data['file_path']}")
            
            # Read file
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find task heading
            task_line_idx = clock_data['line_number'] - 1
            
            if task_line_idx < 0 or task_line_idx >= len(lines):
                raise ValueError(f"Invalid line number: {clock_data['line_number']}")
            
            # Format CLOCK entry
            clock_entry = self._format_clock_entry(start_time, end_time, duration)
            
            # Add to LOGBOOK drawer
            new_lines = self._add_to_logbook(lines, task_line_idx, clock_entry)
            
            # Write updated file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            # Remove active clock
            del self._active_clocks[user_id]
            
            duration_str = self._format_duration(duration)
            logger.info(f"✅ Clock stopped: {clock_data['heading']} ({duration_str})")
            
            return {
                "success": True,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "duration_display": duration_str,
                "heading": clock_data['heading'],
                "file_path": clock_data['file_path']
            }
            
        except Exception as e:
            logger.error(f"❌ Clock out failed: {e}", exc_info=True)
            # Clean up active clock on error
            if user_id in self._active_clocks:
                del self._active_clocks[user_id]
            raise
    
    async def get_active_clock(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user's active clock if any
        
        Args:
            user_id: User ID
        
        Returns:
            Active clock data or None
        """
        if user_id not in self._active_clocks:
            return None
        
        clock_data = self._active_clocks[user_id].copy()
        # Calculate elapsed time
        elapsed = datetime.now() - clock_data['start_time']
        clock_data['elapsed_seconds'] = elapsed.total_seconds()
        clock_data['elapsed_display'] = self._format_duration(elapsed)
        clock_data['start_time'] = clock_data['start_time'].isoformat()
        
        return clock_data
    
    async def get_time_report(
        self,
        user_id: str,
        file_path: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate time report from CLOCK entries
        
        Args:
            user_id: User ID
            file_path: Optional specific file (otherwise all files)
            days: Days to include in report
        
        Returns:
            Time report with statistics
        """
        try:
            logger.info(f"📊 TIME REPORT: User {user_id}, days: {days}")
            
            # TODO: Implement LOGBOOK parsing and statistics
            # For now, return placeholder
            return {
                "success": True,
                "total_seconds": 0,
                "total_display": "0:00",
                "days_analyzed": days,
                "entries": []
            }
            
        except Exception as e:
            logger.error(f"❌ Time report failed: {e}", exc_info=True)
            raise
    
    async def _resolve_file_path(self, user_id: str, relative_path: str) -> Path:
        """Resolve relative org file path to absolute path"""
        from backend.config import settings
        from services.database_manager.database_helpers import fetch_one
        
        # Get username
        row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
        username = row['username'] if row else user_id
        
        # Construct file path
        upload_dir = Path(settings.UPLOAD_DIR)
        user_base_dir = upload_dir / "Users" / username
        file_path = user_base_dir / relative_path
        
        return file_path
    
    def _format_clock_entry(self, start: datetime, end: datetime, duration: timedelta) -> str:
        """Format CLOCK entry for LOGBOOK"""
        start_str = start.strftime("%Y-%m-%d %a %H:%M")
        end_str = end.strftime("%Y-%m-%d %a %H:%M")
        duration_str = self._format_duration(duration)
        
        return f"CLOCK: [{start_str}]--[{end_str}] => {duration_str}\n"
    
    def _format_duration(self, duration: timedelta) -> str:
        """Format duration as H:MM"""
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        
        return f"{hours:2d}:{minutes:02d}"
    
    def _add_to_logbook(self, lines: List[str], heading_idx: int, clock_entry: str) -> List[str]:
        """
        Add CLOCK entry to LOGBOOK drawer
        
        Creates LOGBOOK if it doesn't exist
        """
        new_lines = lines.copy()
        
        # Look for existing LOGBOOK after heading
        logbook_start = None
        logbook_end = None
        
        for i in range(heading_idx + 1, min(heading_idx + 20, len(lines))):
            line = lines[i].strip()
            
            if line == ':LOGBOOK:':
                logbook_start = i
            elif logbook_start is not None and line == ':END:':
                logbook_end = i
                break
            elif line.startswith('*'):  # Next heading - no LOGBOOK found
                break
        
        if logbook_start is not None and logbook_end is not None:
            # LOGBOOK exists - add entry before :END:
            new_lines.insert(logbook_end, clock_entry)
        else:
            # Create new LOGBOOK
            logbook_lines = [
                ":LOGBOOK:\n",
                clock_entry,
                ":END:\n"
            ]
            # Insert after heading (and SCHEDULED/DEADLINE if present)
            insert_idx = heading_idx + 1
            
            # Skip SCHEDULED/DEADLINE lines
            while insert_idx < len(lines) and (
                'SCHEDULED:' in lines[insert_idx] or 
                'DEADLINE:' in lines[insert_idx] or
                'CLOSED:' in lines[insert_idx]
            ):
                insert_idx += 1
            
            # Insert LOGBOOK
            for log_line in reversed(logbook_lines):
                new_lines.insert(insert_idx, log_line)
        
        return new_lines


# Singleton instance
_org_clocking_service_instance = None


async def get_org_clocking_service() -> OrgClockingService:
    """Get or create the OrgClockingService singleton"""
    global _org_clocking_service_instance
    if _org_clocking_service_instance is None:
        _org_clocking_service_instance = OrgClockingService()
    return _org_clocking_service_instance



