"""
Email Agent - Drafts and sends emails with user approval and conversation context
"""

import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncpg

from services.langgraph_agents.base_agent import BaseAgent
from models.agent_response_models import EmailResponse, EmailDraft, TaskStatus
from services.email_service import email_service
from services.email_rate_limiter import EmailRateLimiter
from config import settings

logger = logging.getLogger(__name__)


class EmailAgent(BaseAgent):
    """
    Email Agent - Drafts and sends emails with HITL approval
    """
    
    def __init__(self):
        super().__init__("email_agent")
        self.rate_limiter = None
    
    async def _get_db_pool(self):
        """Get database connection pool"""
        from services.service_container import service_container
        if not service_container.is_initialized:
            await service_container.initialize()
        return service_container.db_pool
    
    async def _get_rate_limiter(self):
        """Get rate limiter instance"""
        if self.rate_limiter is None:
            db_pool = await self._get_db_pool()
            self.rate_limiter = EmailRateLimiter(db_pool)
        return self.rate_limiter
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process email requests with workflow nodes"""
        try:
            logger.info("📧 EMAIL AGENT: Starting email processing...")
            
            # Extract state
            user_id = state.get("user_id")
            query = self._extract_current_user_query(state)
            shared_memory = state.get("shared_memory", {})
            messages = state.get("messages", [])
            
            if not user_id:
                return self._create_error_response("User ID not found in state")
            
            # Node 1: Check verification
            verification_result = await self._check_verification_node(user_id)
            if not verification_result.get("email_verified", False):
                return {
                    "agent_results": {
                        "agent_type": self.agent_type,
                        "response": verification_result.get("message", "Email verification required"),
                        "task_status": TaskStatus.INCOMPLETE.value,
                        "send_status": "verification_required",
                        "user_email": verification_result.get("user_email")
                    },
                    "is_complete": False
                }
            
            # Node 2: Analyze request
            analysis_result = await self._analyze_request_node(query, state)
            if analysis_result.get("error"):
                return self._create_error_response(analysis_result["error"])
            
            # Node 3: Extract context
            context_result = await self._extract_context_node(state, analysis_result)
            
            # Node 4: Draft email
            draft_result = await self._draft_email_node(
                query, analysis_result, context_result, verification_result, state
            )
            
            if draft_result.get("error"):
                return self._create_error_response(draft_result["error"])
            
            # Node 5: Check rate limits
            rate_limit_result = await self._check_rate_limits_node(user_id)
            if not rate_limit_result.get("allowed", True):
                return {
                    "agent_results": {
                        "agent_type": self.agent_type,
                        "response": self._format_rate_limit_message(rate_limit_result),
                        "task_status": TaskStatus.INCOMPLETE.value,
                        "send_status": "rate_limited",
                        "rate_limit_info": rate_limit_result
                    },
                    "is_complete": False
                }
            
            # Node 6: Request approval (HITL)
            approval_result = await self._request_approval_node(draft_result, rate_limit_result)
            
            return approval_result
            
        except Exception as e:
            logger.error(f"❌ EMAIL AGENT: Processing failed: {e}")
            return self._create_error_response(f"Email agent error: {str(e)}")
    
    async def _check_verification_node(self, user_id: str) -> Dict[str, Any]:
        """Check if user's email is verified"""
        try:
            db_pool = await self._get_db_pool()
            async with db_pool.acquire() as conn:
                user = await conn.fetchrow("""
                    SELECT email, email_verified, display_name, username
                    FROM users
                    WHERE user_id = $1
                """, user_id)
                
                if not user:
                    return {
                        "email_verified": False,
                        "message": "User not found",
                        "error": "User not found"
                    }
                
                if not user["email_verified"]:
                    return {
                        "email_verified": False,
                        "user_email": user["email"],
                        "message": f"""
📧 **Email Verification Required**

To send emails through Bastion, you need to verify your email address first.

A verification link was sent to: {user['email']}

Please check your inbox and click the verification link. If you didn't receive it, 
you can request a new verification email from your profile settings.
"""
                    }
                
                return {
                    "email_verified": True,
                    "from_email": user["email"],
                    "from_name": user["display_name"] or user["username"]
                }
                
        except Exception as e:
            logger.error(f"Error checking verification: {e}")
            return {
                "email_verified": False,
                "message": f"Error checking verification: {str(e)}",
                "error": str(e)
            }
    
    async def _analyze_request_node(self, query: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse recipients and extract intent from query"""
        try:
            # Extract email addresses from query
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, query)
            
            if not emails:
                return {
                    "error": "No email addresses found in your request. Please include at least one recipient email address."
                }
            
            recipients = emails
            
            # Extract CC/BCC if mentioned
            cc = []
            bcc = []
            query_lower = query.lower()
            if "cc:" in query_lower or " cc " in query_lower:
                # Try to extract CC emails
                cc_match = re.search(r'cc[:\s]+([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', query_lower)
                if cc_match:
                    cc.append(cc_match.group(1))
            
            if "bcc:" in query_lower or " bcc " in query_lower:
                # Try to extract BCC emails
                bcc_match = re.search(r'bcc[:\s]+([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', query_lower)
                if bcc_match:
                    bcc.append(bcc_match.group(1))
            
            # Extract subject hints
            subject_hint = None
            if "subject:" in query_lower:
                subject_match = re.search(r'subject[:\s]+([^\n]+)', query_lower)
                if subject_match:
                    subject_hint = subject_match.group(1).strip()
            
            return {
                "recipients": recipients,
                "cc": cc,
                "bcc": bcc,
                "subject_hint": subject_hint,
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Error analyzing request: {e}")
            return {"error": f"Error analyzing request: {str(e)}"}
    
    async def _extract_context_node(self, state: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Smart extraction from conversation history"""
        try:
            shared_memory = state.get("shared_memory", {})
            agent_results = shared_memory.get("agent_results", {})
            messages = state.get("messages", [])
            
            referenced_content = {}
            context_sources = []
            
            # Check for research agent outputs
            if "research_agent" in str(agent_results):
                research_data = agent_results.get("research_agent", {})
                if research_data:
                    referenced_content["research"] = research_data
                    context_sources.append("research_agent")
            
            # Check for data formatting agent outputs
            if "data_formatting_agent" in str(agent_results):
                formatting_data = agent_results.get("data_formatting_agent", {})
                if formatting_data:
                    referenced_content["formatted_data"] = formatting_data
                    context_sources.append("data_formatting_agent")
            
            # Extract from conversation messages if user references "the research" or similar
            query = analysis.get("query", "").lower()
            if any(phrase in query for phrase in ["the research", "research on", "findings", "results"]):
                # Look for recent agent outputs in messages
                for msg in reversed(messages[-10:]):  # Check last 10 messages
                    if hasattr(msg, 'content'):
                        content = msg.content
                    elif isinstance(msg, dict):
                        content = msg.get("content", "")
                    else:
                        content = str(msg)
                    
                    if "research" in content.lower() or "findings" in content.lower():
                        referenced_content["conversation"] = content[:1000]  # Limit length
                        context_sources.append("conversation_history")
                        break
            
            # If ambiguous, we'll ask for confirmation in draft
            is_ambiguous = len(context_sources) == 0 and any(
                phrase in query for phrase in ["send", "email", "share"]
            )
            
            return {
                "referenced_content": referenced_content,
                "context_sources": context_sources,
                "is_ambiguous": is_ambiguous
            }
            
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return {
                "referenced_content": {},
                "context_sources": [],
                "is_ambiguous": True
            }
    
    async def _draft_email_node(
        self,
        query: str,
        analysis: Dict[str, Any],
        context: Dict[str, Any],
        verification: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to compose email"""
        try:
            from openai import AsyncOpenAI
            from config import settings
            
            client = AsyncOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1"
            )
            
            # Build context string
            context_text = ""
            if context.get("referenced_content"):
                if "research" in context["referenced_content"]:
                    research = context["referenced_content"]["research"]
                    if isinstance(research, dict):
                        context_text += f"\n\nResearch Findings:\n{research.get('response', str(research))}"
                    else:
                        context_text += f"\n\nResearch Findings:\n{research}"
                
                if "formatted_data" in context["referenced_content"]:
                    data = context["referenced_content"]["formatted_data"]
                    if isinstance(data, dict):
                        context_text += f"\n\nFormatted Data:\n{data.get('response', str(data))}"
                    else:
                        context_text += f"\n\nFormatted Data:\n{data}"
                
                if "conversation" in context["referenced_content"]:
                    context_text += f"\n\nRelevant Conversation:\n{context['referenced_content']['conversation']}"
            
            # Build prompt
            prompt = f"""You are an email composition specialist. Draft a professional email based on:

User Request: {query}
Recipients: {', '.join(analysis['recipients'])}
Context: {context_text if context_text else 'No specific context provided - compose based on user request'}

Create an appropriate email with:
- Clear, professional subject line
- Professional greeting
- Well-structured body that incorporates any context provided naturally
- Appropriate closing

If the user request mentions specific content to include (like research findings), incorporate that content in a natural, professional way.

Respond with JSON in this format:
{{
    "subject": "Email subject line",
    "body_text": "Plain text email body",
    "body_html": "<html><body><p>HTML email body</p></body></html>",
    "confidence": 0.85
}}"""
            
            response = await client.chat.completions.create(
                model=settings.DEFAULT_MODEL or "anthropic/claude-3.5-haiku",
                messages=[
                    {"role": "system", "content": "You are a professional email composition assistant. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            import json
            draft_data = json.loads(response.choices[0].message.content)
            
            # Use subject hint if provided
            subject = analysis.get("subject_hint") or draft_data.get("subject", "Email from Bastion")
            
            draft = EmailDraft(
                recipients=analysis["recipients"],
                cc=analysis.get("cc", []),
                bcc=analysis.get("bcc", []),
                subject=subject,
                body_text=draft_data.get("body_text", ""),
                body_html=draft_data.get("body_html"),
                from_email=verification["from_email"],
                from_name=verification["from_name"],
                confidence=draft_data.get("confidence", 0.8),
                context_sources=context.get("context_sources", [])
            )
            
            return {
                "draft": draft,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error drafting email: {e}")
            return {"error": f"Error drafting email: {str(e)}"}
    
    async def _check_rate_limits_node(self, user_id: str) -> Dict[str, Any]:
        """Check rate limits"""
        try:
            rate_limiter = await self._get_rate_limiter()
            return await rate_limiter.check_rate_limit(user_id)
        except Exception as e:
            logger.error(f"Error checking rate limits: {e}")
            return {"allowed": True, "error": str(e)}
    
    async def _request_approval_node(
        self,
        draft_result: Dict[str, Any],
        rate_limit_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Present draft to user for approval (HITL)"""
        try:
            draft = draft_result["draft"]
            
            # Format approval message
            approval_message = f"""📧 **Email Draft Ready for Review:**

**From:** {draft.from_email} ({draft.from_name}) ✅
**To:** {', '.join(draft.recipients)}
"""
            
            if draft.cc:
                approval_message += f"**CC:** {', '.join(draft.cc)}\n"
            if draft.bcc:
                approval_message += f"**BCC:** {', '.join(draft.bcc)}\n"
            
            approval_message += f"""
**Subject:** {draft.subject}

**Body:**
{draft.body_text}

---
✅ Reply "yes" or "send" to send this email
❌ Reply "no" or "cancel" to cancel
✏️ Reply with changes to revise the draft

Today's usage: {rate_limit_result.get('daily_sent', 0)}/{settings.EMAIL_DAILY_LIMIT} emails sent
"""
            
            return {
                "agent_results": {
                    "agent_type": self.agent_type,
                    "response": approval_message,
                    "task_status": TaskStatus.PERMISSION_REQUIRED.value,
                    "send_status": "draft",
                    "draft": draft.dict(),
                    "rate_limit_info": rate_limit_result
                },
                "is_complete": False,
                "requires_user_input": True
            }
            
        except Exception as e:
            logger.error(f"Error requesting approval: {e}")
            return self._create_error_response(f"Error requesting approval: {str(e)}")
    
    async def send_email_after_approval(
        self,
        draft: EmailDraft,
        user_id: str,
        conversation_id: str = None
    ) -> Dict[str, Any]:
        """Send email after user approval"""
        try:
            # Send email
            success = await email_service.send_email(
                to_email=draft.recipients[0],
                subject=draft.subject,
                body_text=draft.body_text,
                body_html=draft.body_html,
                cc=draft.cc if draft.cc else None,
                bcc=draft.bcc if draft.bcc else None,
                from_email=draft.from_email,
                from_name=draft.from_name
            )
            
            if not success:
                return {
                    "success": False,
                    "message": "Failed to send email",
                    "send_status": "failed"
                }
            
            # Record in rate limiter
            rate_limiter = await self._get_rate_limiter()
            for recipient in draft.recipients:
                await rate_limiter.record_sent_email(user_id, recipient)
            
            # Log to audit trail
            await self._log_email_node(user_id, draft, "sent", conversation_id)
            
            return {
                "success": True,
                "message": f"Email sent successfully to {', '.join(draft.recipients)}",
                "send_status": "sent"
            }
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            await self._log_email_node(user_id, draft, "failed", conversation_id, str(e))
            return {
                "success": False,
                "message": f"Error sending email: {str(e)}",
                "send_status": "failed"
            }
    
    async def _log_email_node(
        self,
        user_id: str,
        draft: EmailDraft,
        status: str,
        conversation_id: str = None,
        error_message: str = None
    ):
        """Log email to audit trail"""
        try:
            db_pool = await self._get_db_pool()
            async with db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO email_audit_log (
                        user_id, from_email, to_email, cc, bcc, subject,
                        body_preview, conversation_id, agent_type, send_status, error_message
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                    user_id,
                    draft.from_email,
                    ', '.join(draft.recipients),
                    ', '.join(draft.cc) if draft.cc else None,
                    ', '.join(draft.bcc) if draft.bcc else None,
                    draft.subject,
                    draft.body_text[:200],  # Preview
                    conversation_id,
                    self.agent_type,
                    status,
                    error_message
                )
        except Exception as e:
            logger.error(f"Error logging email: {e}")
    
    def _format_rate_limit_message(self, rate_limit_info: Dict[str, Any]) -> str:
        """Format rate limit error message"""
        if rate_limit_info.get("reason") == "hourly_limit_exceeded":
            return f"""📧 **Rate Limit Exceeded**

You've reached your hourly email limit of {rate_limit_info.get('limit', 5)} emails.

You can send more emails in {rate_limit_info.get('reset_in_minutes', 60)} minutes.

Daily usage: {rate_limit_info.get('daily_sent', 0)}/{settings.EMAIL_DAILY_LIMIT} emails sent today.
"""
        else:
            return f"""📧 **Daily Rate Limit Exceeded**

You've reached your daily email limit of {rate_limit_info.get('limit', 20)} emails.

You can send more emails in {rate_limit_info.get('reset_in_hours', 24)} hours.
"""

