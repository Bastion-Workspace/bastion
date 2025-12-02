"""
Email Rate Limiter - Prevents email abuse with conservative limits
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import asyncpg
from config import settings

logger = logging.getLogger(__name__)


class EmailRateLimiter:
    """Service for checking and enforcing email sending rate limits"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.hourly_limit = settings.EMAIL_HOURLY_LIMIT
        self.daily_limit = settings.EMAIL_DAILY_LIMIT
        self.enabled = settings.EMAIL_RATE_LIMITING_ENABLED
    
    async def check_rate_limit(self, user_id: str) -> Dict[str, Any]:
        """
        Check if user has exceeded rate limits
        
        Returns:
            Dict with 'allowed' boolean and limit information
        """
        if not self.enabled:
            return {
                "allowed": True,
                "hourly_remaining": self.hourly_limit,
                "daily_remaining": self.daily_limit
            }
        
        try:
            async with self.db_pool.acquire() as conn:
                now = datetime.utcnow()
                one_hour_ago = now - timedelta(hours=1)
                one_day_ago = now - timedelta(days=1)
                
                # Count emails sent in last hour
                hourly_count = await conn.fetchval("""
                    SELECT COUNT(*) 
                    FROM email_rate_limits
                    WHERE user_id = $1 AND sent_at > $2
                """, user_id, one_hour_ago)
                
                # Count emails sent in last 24 hours
                daily_count = await conn.fetchval("""
                    SELECT COUNT(*) 
                    FROM email_rate_limits
                    WHERE user_id = $1 AND sent_at > $2
                """, user_id, one_day_ago)
                
                # Check limits
                hourly_remaining = max(0, self.hourly_limit - hourly_count)
                daily_remaining = max(0, self.daily_limit - daily_count)
                
                if hourly_count >= self.hourly_limit:
                    # Calculate when hourly limit resets
                    oldest_in_hour = await conn.fetchval("""
                        SELECT MIN(sent_at)
                        FROM email_rate_limits
                        WHERE user_id = $1 AND sent_at > $2
                    """, user_id, one_hour_ago)
                    
                    if oldest_in_hour:
                        reset_time = oldest_in_hour + timedelta(hours=1)
                        minutes_until_reset = int((reset_time - now).total_seconds() / 60)
                    else:
                        minutes_until_reset = 60
                    
                    return {
                        "allowed": False,
                        "reason": "hourly_limit_exceeded",
                        "limit": self.hourly_limit,
                        "sent": hourly_count,
                        "reset_in_minutes": minutes_until_reset,
                        "hourly_remaining": 0,
                        "daily_remaining": daily_remaining
                    }
                
                if daily_count >= self.daily_limit:
                    # Calculate when daily limit resets
                    oldest_in_day = await conn.fetchval("""
                        SELECT MIN(sent_at)
                        FROM email_rate_limits
                        WHERE user_id = $1 AND sent_at > $2
                    """, user_id, one_day_ago)
                    
                    if oldest_in_day:
                        reset_time = oldest_in_day + timedelta(days=1)
                        hours_until_reset = int((reset_time - now).total_seconds() / 3600)
                    else:
                        hours_until_reset = 24
                    
                    return {
                        "allowed": False,
                        "reason": "daily_limit_exceeded",
                        "limit": self.daily_limit,
                        "sent": daily_count,
                        "reset_in_hours": hours_until_reset,
                        "hourly_remaining": hourly_remaining,
                        "daily_remaining": 0
                    }
                
                return {
                    "allowed": True,
                    "hourly_remaining": hourly_remaining,
                    "daily_remaining": daily_remaining,
                    "hourly_sent": hourly_count,
                    "daily_sent": daily_count
                }
                
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            # On error, allow the email (fail open, but log the error)
            return {
                "allowed": True,
                "error": str(e),
                "hourly_remaining": self.hourly_limit,
                "daily_remaining": self.daily_limit
            }
    
    async def record_sent_email(self, user_id: str, recipient: str):
        """Record a sent email for rate limiting"""
        if not self.enabled:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO email_rate_limits (user_id, sent_at, recipient_email)
                    VALUES ($1, $2, $3)
                """, user_id, datetime.utcnow(), recipient)
        except Exception as e:
            logger.error(f"Error recording sent email: {e}")

