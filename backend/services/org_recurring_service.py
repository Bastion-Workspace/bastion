"""
Org-Mode Recurring Tasks Service - Roosevelt's "Habit Cavalry"

**BULLY!** Automatic task recurrence for building lasting habits!

This service handles:
- Parsing repeater syntax (+1w, .+1w, ++1w)
- Auto-creating next occurrence when task marked DONE
- Habit tracking and consistency monitoring
- Repeater interval calculations

Repeater Types:
- +1w  : Repeats from completion date (shifts if late)
- .+1w : Repeats from original schedule (doesn't shift)
- ++1w : Repeats from scheduled date (skips if late)
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class OrgRecurringService:
    """Service for handling recurring tasks and habits"""
    
    # Repeater regex: <2025-10-22 Tue +1w>
    REPEATER_PATTERN = r'<(\d{4}-\d{2}-\d{2})\s+\w+\s+([.+]+)(\d+)([dwmy])([^>]*)>'
    
    # Repeater types
    REPEATER_FROM_COMPLETION = '+'   # +1w - shift from completion
    REPEATER_FROM_TODAY = '.+'       # .+1w - from today/completion
    REPEATER_CATCH_UP = '++'         # ++1w - catch up to current
    
    def __init__(self):
        pass
    
    def parse_repeater(self, timestamp_str: str) -> Optional[Dict[str, Any]]:
        """
        Parse org-mode repeater timestamp
        
        Examples:
            <2025-10-22 Tue +1w> -> {'date': '2025-10-22', 'type': '+', 'count': 1, 'unit': 'w'}
            SCHEDULED: <2025-10-22 Tue .+2d> -> {'date': '2025-10-22', 'type': '.+', 'count': 2, 'unit': 'd'}
        
        Args:
            timestamp_str: Timestamp string with repeater
        
        Returns:
            Dict with repeater details or None if no repeater
        """
        match = re.search(self.REPEATER_PATTERN, timestamp_str)
        if not match:
            return None
        
        date_str, repeater_type, count_str, unit, extra = match.groups()
        
        return {
            'date': date_str,
            'type': repeater_type,
            'count': int(count_str),
            'unit': unit,
            'extra': extra.strip(),  # Could have time, etc.
            'full_match': match.group(0)
        }
    
    def calculate_next_occurrence(
        self,
        repeater: Dict[str, Any],
        completion_date: Optional[datetime] = None
    ) -> str:
        """
        Calculate next occurrence date based on repeater type
        
        Args:
            repeater: Parsed repeater dict from parse_repeater()
            completion_date: When task was completed (defaults to now)
        
        Returns:
            New timestamp string with updated date
        """
        if completion_date is None:
            completion_date = datetime.now()
        
        original_date = datetime.strptime(repeater['date'], '%Y-%m-%d')
        repeater_type = repeater['type']
        count = repeater['count']
        unit = repeater['unit']
        
        # Calculate next date based on repeater type
        if repeater_type == self.REPEATER_FROM_COMPLETION:
            # +1w: From completion date
            next_date = self._add_interval(completion_date, count, unit)
        
        elif repeater_type == self.REPEATER_FROM_TODAY:
            # .+1w: From today (or completion)
            next_date = self._add_interval(completion_date, count, unit)
        
        elif repeater_type == self.REPEATER_CATCH_UP:
            # ++1w: Catch up to current (skip missed occurrences)
            next_date = original_date
            today = datetime.now().date()
            
            # Keep adding interval until we're in the future
            while next_date.date() <= today:
                next_date = self._add_interval(next_date, count, unit)
        
        else:
            logger.warning(f"Unknown repeater type: {repeater_type}")
            next_date = self._add_interval(completion_date, count, unit)
        
        # Format new timestamp
        weekday = next_date.strftime('%a')
        new_date_str = next_date.strftime('%Y-%m-%d')
        
        # Rebuild timestamp with new date
        new_timestamp = f"<{new_date_str} {weekday} {repeater['type']}{count}{unit}"
        if repeater['extra']:
            new_timestamp += f" {repeater['extra']}"
        new_timestamp += ">"
        
        return new_timestamp
    
    def _add_interval(self, date: datetime, count: int, unit: str) -> datetime:
        """Add interval to date"""
        if unit == 'd':
            return date + timedelta(days=count)
        elif unit == 'w':
            return date + timedelta(weeks=count)
        elif unit == 'm':
            # Approximate month as 30 days
            return date + timedelta(days=count * 30)
        elif unit == 'y':
            # Approximate year as 365 days
            return date + timedelta(days=count * 365)
        else:
            logger.warning(f"Unknown interval unit: {unit}")
            return date
    
    async def handle_task_completion(
        self,
        user_id: str,
        file_path: str,
        line_number: int
    ) -> Dict[str, Any]:
        """
        Handle a recurring task being marked DONE
        
        1. Check if task has repeater
        2. If yes, reset to TODO and update timestamp
        3. Log completion for habit tracking
        
        Args:
            user_id: User ID
            file_path: File path (e.g., "OrgMode/tasks.org")
            line_number: Line number of task (1-based)
        
        Returns:
            Dict with success status and details
        """
        try:
            logger.info(f"🔁 RECURRING: Handling task completion at {file_path}:{line_number}")
            
            # Resolve file path
            from services.folder_service import FolderService
            from backend.config import settings
            from services.database_manager.database_helpers import fetch_one
            
            folder_service = FolderService()
            
            # Get username
            row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
            username = row['username'] if row else user_id
            
            # Construct file path
            upload_dir = Path(settings.UPLOAD_DIR)
            user_base_dir = upload_dir / "Users" / username
            full_path = user_base_dir / file_path
            
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read file
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if line_number < 1 or line_number > len(lines):
                raise ValueError(f"Invalid line number: {line_number}")
            
            # Get the task line
            task_line = lines[line_number - 1]
            
            # Check if it's a recurring task
            repeater = None
            for i in range(line_number - 1, min(line_number + 10, len(lines))):
                line = lines[i]
                if 'SCHEDULED:' in line or 'DEADLINE:' in line:
                    repeater = self.parse_repeater(line)
                    if repeater:
                        repeater['line_index'] = i
                        break
            
            if not repeater:
                return {
                    "success": True,
                    "is_recurring": False,
                    "message": "Task is not recurring"
                }
            
            logger.info(f"🔁 Found repeater: {repeater}")
            
            # Calculate next occurrence
            next_timestamp = self.calculate_next_occurrence(repeater)
            
            # Replace timestamp in file
            old_line = lines[repeater['line_index']]
            new_line = old_line.replace(repeater['full_match'], next_timestamp)
            lines[repeater['line_index']] = new_line
            
            # Reset DONE to TODO in task heading
            task_line_updated = re.sub(
                r'^\*+\s+(DONE|CANCELED|CANCELLED)\s+',
                lambda m: m.group(0).replace(m.group(1), 'TODO'),
                task_line
            )
            lines[line_number - 1] = task_line_updated
            
            # Write updated file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            logger.info(f"✅ Recurring task reset: {task_line.strip()}")
            logger.info(f"📅 Next occurrence: {next_timestamp}")
            
            return {
                "success": True,
                "is_recurring": True,
                "next_timestamp": next_timestamp,
                "message": f"Task reset to TODO with next occurrence: {next_timestamp}"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to handle recurring task: {e}", exc_info=True)
            raise
    
    async def get_habit_consistency(
        self,
        user_id: str,
        file_path: str,
        heading_text: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get habit consistency data for a recurring task
        
        Analyzes LOGBOOK or state changes to track completion consistency
        
        Args:
            user_id: User ID
            file_path: File path
            heading_text: Heading text to track
            days: Number of days to analyze
        
        Returns:
            Dict with consistency stats
        """
        # TODO: Implement consistency tracking from LOGBOOK
        # For now, return placeholder data
        return {
            "heading": heading_text,
            "total_completions": 0,
            "expected_completions": 0,
            "consistency_percentage": 0.0,
            "last_completion": None,
            "streak": 0
        }


# Singleton instance
_org_recurring_service_instance = None


async def get_org_recurring_service() -> OrgRecurringService:
    """Get or create the OrgRecurringService singleton"""
    global _org_recurring_service_instance
    if _org_recurring_service_instance is None:
        _org_recurring_service_instance = OrgRecurringService()
    return _org_recurring_service_instance



