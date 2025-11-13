"""
Fact-Checking Agent
Specialized agent for fact-checking content using web search and LLM analysis
"""

import logging
import json
import re
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from orchestrator.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class FactCheckingAgent(BaseAgent):
    """
    Fact-Checking Agent
    Verifies factual claims using web search and LLM analysis
    """
    
    def __init__(self):
        super().__init__("fact_checking_agent")
        self._openai_client = None
        self._grpc_client = None
    
    async def _get_openai_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client"""
        if self._openai_client is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            
            self._openai_client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        return self._openai_client
    
    async def _get_grpc_client(self):
        """Get or create gRPC client for backend tools"""
        if self._grpc_client is None:
            from orchestrator.clients.backend_tool_client import get_backend_tool_client
            self._grpc_client = await get_backend_tool_client()
        return self._grpc_client
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process fact-checking request"""
        try:
            logger.info("🔍 Fact-Checking Agent: Starting verification operation...")
            
            # Extract content from messages
            messages = state.get("messages", [])
            user_message = messages[-1].content if messages else ""
            user_id = state.get("user_id")
            
            # Extract content and claims from state or message
            content = state.get("content", user_message)
            claims = state.get("claims", [])
            
            if not content:
                return self._create_error_result("No content provided for fact-checking")
            
            # Perform fact-checking
            fact_check_result = await self.fact_check_content(content, claims, user_id)
            
            # Format results
            if fact_check_result["success"]:
                analysis = fact_check_result["analysis"]
                
                # Create formatted summary
                summary_lines = [
                    "## Fact-Checking Results",
                    f"**Overall Status:** {analysis['overall_status'].replace('_', ' ').title()}",
                    f"**Verification Rate:** {analysis['verification_rate']:.1%} ({analysis['verified_claims']}/{analysis['total_claims']} claims)",
                    f"**Overall Confidence:** {analysis['overall_confidence']:.1%}",
                    "",
                    "### Claim Verification Details:"
                ]
                
                for result in fact_check_result["results"][:5]:  # Top 5 results
                    verification = result["verification"]
                    status_emoji = "✅" if verification["verified"] else "❌" if verification["status"] == "disputed" else "⚠️"
                    
                    summary_lines.append(f"\n**{status_emoji} Claim {result['claim_id']}:** {result['claim'][:100]}...")
                    summary_lines.append(f"   Status: {verification['status']} (Confidence: {verification['confidence']:.1%})")
                    summary_lines.append(f"   Analysis: {verification['analysis']}")
                    
                    if verification.get("correct_facts"):
                        summary_lines.append(f"   ℹ️ Correct Facts: {verification['correct_facts']}")
                    
                    if verification["sources"]:
                        summary_lines.append(f"   Sources: {len(verification['sources'])} found")
                
                if len(fact_check_result["results"]) > 5:
                    summary_lines.append(f"\n... and {len(fact_check_result['results']) - 5} more claims")
                
                response_text = "\n".join(summary_lines)
                
                logger.info(f"✅ Fact-checking complete: {analysis['verified_claims']}/{analysis['total_claims']} claims verified")
                
                return {
                    "messages": [AIMessage(content=response_text)],
                    "agent_results": {
                        "agent_type": "fact_checking_agent",
                        "fact_checking_results": fact_check_result,
                        "formatted_summary": response_text,
                        "success": True,
                        "is_complete": True
                    },
                    "is_complete": True
                }
            else:
                error_msg = fact_check_result.get("error", "Fact-checking failed")
                return self._create_error_result(error_msg)
            
        except Exception as e:
            logger.error(f"❌ Fact-checking agent process failed: {e}")
            return self._create_error_result(str(e))
    
    async def fact_check_content(self, content: str, claims: List[str] = None, user_id: str = None) -> Dict[str, Any]:
        """
        Fact-check specific claims or extract and verify factual statements from content
        
        Args:
            content: The content to fact-check
            claims: Specific claims to verify (if None, will extract claims automatically)
            user_id: User ID for permission tracking
            
        Returns:
            Dict with fact-checking results
        """
        try:
            logger.info(f"🔍 Fact-checking content: {len(content)} characters")
            
            # Extract claims if not provided
            if not claims:
                claims = await self._extract_factual_claims(content)
            
            if not claims:
                return {
                    "success": True,
                    "claims_found": 0,
                    "results": [],
                    "analysis": {
                        "total_claims": 0,
                        "verified_claims": 0,
                        "disputed_claims": 0,
                        "unverified_claims": 0,
                        "verification_rate": 0.0,
                        "overall_confidence": 0.0,
                        "overall_status": "no_claims",
                        "summary": "No factual claims found to verify"
                    },
                    "message": "No factual claims found to verify"
                }
            
            logger.info(f"🔍 Found {len(claims)} claims to verify")
            
            # Verify each claim using web search
            verification_results = []
            for i, claim in enumerate(claims):
                logger.info(f"🔍 Verifying claim {i+1}/{len(claims)}: {claim[:100]}...")
                
                verification_result = await self._verify_claim(claim, user_id)
                verification_results.append({
                    "claim": claim,
                    "verification": verification_result,
                    "claim_id": i + 1
                })
            
            # Analyze overall fact-checking results
            analysis = self._analyze_verification_results(verification_results)
            
            return {
                "success": True,
                "claims_found": len(claims),
                "claims_verified": len([r for r in verification_results if r["verification"]["verified"]]),
                "results": verification_results,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Fact-checking failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "claims_found": 0,
                "results": [],
                "analysis": {
                    "total_claims": 0,
                    "verified_claims": 0,
                    "disputed_claims": 0,
                    "unverified_claims": 0,
                    "verification_rate": 0.0,
                    "overall_confidence": 0.0,
                    "overall_status": "error",
                    "summary": f"Error: {str(e)}"
                }
            }
    
    async def _extract_factual_claims(self, content: str) -> List[str]:
        """Extract factual claims from content that can be verified"""
        try:
            client = await self._get_openai_client()
            model = os.getenv("DEFAULT_MODEL", "anthropic/claude-3.5-sonnet")
            
            system_prompt = (
                "You are a fact-checking expert. Extract factual claims from the provided content that can be verified through web search. "
                "Focus on specific, verifiable statements like statistics, dates, names, events, and scientific facts. "
                "Be comprehensive - extract ALL verifiable claims, not just the most obvious ones.\n\n"
                "INCLUDE THESE TYPES OF CLAIMS:\n"
                "- Specific events, incidents, or occurrences\n"
                "- Statistics, numbers, percentages, or data\n"
                "- Dates, times, or historical facts\n"
                "- Names of people, places, organizations\n"
                "- Legal, policy, or regulatory statements\n"
                "- Scientific or medical claims\n"
                "- Economic or financial data\n\n"
                "Ignore opinions, subjective statements, and unverifiable claims.\n\n"
                "Return ONLY a JSON array of strings, each containing one factual claim. "
                "Do not include explanations or additional text.\n\n"
                "Example format: [\"Claim 1\", \"Claim 2\", \"Claim 3\"]"
            )
            
            user_prompt = f"Extract factual claims from this content:\n\n{content[:2000]}"
            
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            
            content_response = response.choices[0].message.content or "[]"
            
            # Clean and parse JSON
            text = content_response.strip()
            if '```json' in text:
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            elif '```' in text:
                m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            
            # Try to find JSON array
            json_match = re.search(r'\[[\s\S]*\]', text)
            if json_match:
                text = json_match.group(0)
            
            try:
                claims = json.loads(text)
                if isinstance(claims, list):
                    # Filter out empty or invalid claims
                    valid_claims = [claim for claim in claims if isinstance(claim, str) and len(claim.strip()) > 10]
                    logger.info(f"✅ Extracted {len(valid_claims)} valid claims")
                    return valid_claims
                else:
                    logger.warning("❌ Claims extraction returned non-list format")
                    return []
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse claims JSON: {e}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Claim extraction failed: {e}")
            return []
    
    async def _verify_claim(self, claim: str, user_id: str = None) -> Dict[str, Any]:
        """Verify a single claim using web search"""
        try:
            # Use backend gRPC tool for web search
            grpc_client = await self._get_grpc_client()
            
            # Search for the claim with fact-checking query
            search_query = f'"{claim}" fact check verification'
            
            logger.info(f"🔍 Searching web for: {search_query[:100]}...")
            
            search_result = await grpc_client.search_web(
                query=search_query,
                max_results=10,
                user_id=user_id or "system"
            )
            
            if not search_result.get("success"):
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "sources": [],
                    "status": "search_failed",
                    "analysis": "Web search failed",
                    "error": search_result.get("error", "Unknown search error")
                }
            
            # Analyze search results for verification
            verification_analysis = await self._analyze_search_results_for_verification(
                claim, 
                search_result.get("results", [])
            )
            
            return verification_analysis
            
        except Exception as e:
            logger.error(f"❌ Claim verification failed: {e}")
            return {
                "verified": False,
                "confidence": 0.0,
                "sources": [],
                "status": "verification_failed",
                "analysis": f"Verification failed: {str(e)}",
                "error": str(e)
            }
    
    async def _analyze_search_results_for_verification(self, claim: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze search results to determine if a claim is verified"""
        try:
            if not search_results:
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "sources": [],
                    "status": "no_sources",
                    "analysis": "No sources found to verify this claim"
                }
            
            # Use LLM to analyze search results for verification
            client = await self._get_openai_client()
            model = os.getenv("DEFAULT_MODEL", "anthropic/claude-3.5-sonnet")
            
            # Format search results for analysis
            results_text = ""
            for i, result in enumerate(search_results[:8], 1):  # Top 8 results
                results_text += f"\n{i}. {result.get('title', 'No title')}\n"
                results_text += f"   URL: {result.get('url', 'Unknown')}\n"
                results_text += f"   Snippet: {result.get('snippet', 'No content')}\n"
            
            system_prompt = (
                "You are a fact-checking expert. Analyze the provided search results to determine if the claim is verified, disputed, or unverified. "
                "Consider the credibility of sources and the consistency of information.\n\n"
                "CRITICAL ANALYSIS REQUIREMENTS:\n"
                "1. Examine ALL provided search results thoroughly\n"
                "2. Cross-reference information across multiple sources\n"
                "3. Consider source credibility (news outlets, official sources, fact-checking sites)\n"
                "4. Look for consensus or disagreement among sources\n"
                "5. Provide actual facts and corrections when claims are false or disputed\n"
                "6. If a claim is false, provide the correct information with specific sources\n"
                "7. Be specific about what evidence supports or contradicts the claim\n\n"
                "Return ONLY a JSON object with this exact structure:\n"
                "{\n"
                '  "verified": true/false,\n'
                '  "confidence": 0.0-1.0,\n'
                '  "status": "verified|disputed|unverified|insufficient_evidence",\n'
                '  "analysis": "Detailed explanation of verification status with specific evidence",\n'
                '  "correct_facts": "If claim is false/disputed, provide the correct facts here with sources",\n'
                '  "supporting_sources": ["source1", "source2"],\n'
                '  "contradicting_sources": ["source3"],\n'
                '  "recommendations": "Specific recommendations for improving the claim"\n'
                "}"
            )
            
            user_prompt = f"Claim to verify: {claim}\n\nSearch results:\n{results_text}"
            
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            
            content_response = response.choices[0].message.content or "{}"
            
            # Clean and parse JSON
            text = content_response.strip()
            if '```json' in text:
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            elif '```' in text:
                m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            
            # Try to find JSON object
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                text = json_match.group(0)
            
            try:
                analysis = json.loads(text)
                
                # Add source information
                sources = []
                for result in search_results[:8]:
                    sources.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("snippet", "")
                    })
                
                return {
                    "verified": analysis.get("verified", False),
                    "confidence": analysis.get("confidence", 0.0),
                    "status": analysis.get("status", "unverified"),
                    "analysis": analysis.get("analysis", "No analysis provided"),
                    "correct_facts": analysis.get("correct_facts", ""),
                    "recommendations": analysis.get("recommendations", ""),
                    "sources": sources,
                    "supporting_sources": analysis.get("supporting_sources", []),
                    "contradicting_sources": analysis.get("contradicting_sources", [])
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse verification analysis: {e}")
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "status": "analysis_failed",
                    "analysis": "Failed to analyze search results",
                    "sources": [],
                    "error": str(e)
                }
                
        except Exception as e:
            logger.error(f"❌ Search result analysis failed: {e}")
            return {
                "verified": False,
                "confidence": 0.0,
                "status": "analysis_failed",
                "analysis": f"Analysis failed: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def _analyze_verification_results(self, verification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall fact-checking results"""
        try:
            total_claims = len(verification_results)
            if total_claims == 0:
                return {
                    "total_claims": 0,
                    "verified_claims": 0,
                    "disputed_claims": 0,
                    "unverified_claims": 0,
                    "verification_rate": 0.0,
                    "overall_confidence": 0.0,
                    "overall_status": "no_claims",
                    "summary": "No claims to analyze"
                }
            
            verified_claims = len([r for r in verification_results if r["verification"]["verified"]])
            disputed_claims = len([r for r in verification_results if r["verification"]["status"] == "disputed"])
            unverified_claims = len([r for r in verification_results if r["verification"]["status"] == "unverified"])
            
            # Calculate overall confidence
            confidence_scores = [r["verification"]["confidence"] for r in verification_results if r["verification"]["confidence"] > 0]
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            # Determine overall status
            verification_rate = verified_claims / total_claims
            if verification_rate >= 0.8:
                overall_status = "highly_verified"
            elif verification_rate >= 0.6:
                overall_status = "mostly_verified"
            elif disputed_claims / total_claims >= 0.3:
                overall_status = "disputed"
            else:
                overall_status = "unverified"
            
            return {
                "total_claims": total_claims,
                "verified_claims": verified_claims,
                "disputed_claims": disputed_claims,
                "unverified_claims": unverified_claims,
                "verification_rate": verification_rate,
                "overall_confidence": overall_confidence,
                "overall_status": overall_status,
                "summary": f"Verified {verified_claims}/{total_claims} claims ({verification_rate*100:.1f}%)"
            }
            
        except Exception as e:
            logger.error(f"❌ Verification analysis failed: {e}")
            return {
                "total_claims": 0,
                "verified_claims": 0,
                "disputed_claims": 0,
                "unverified_claims": 0,
                "verification_rate": 0.0,
                "overall_confidence": 0.0,
                "overall_status": "analysis_failed",
                "summary": f"Analysis failed: {str(e)}",
                "error": str(e)
            }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        logger.error(f"❌ Fact-checking error: {error_message}")
        return {
            "messages": [AIMessage(content=f"Fact-checking failed: {error_message}")],
            "agent_results": {
                "agent_type": "fact_checking_agent",
                "success": False,
                "error": error_message,
                "is_complete": True
            },
            "is_complete": True
        }


# Singleton instance
_fact_checking_agent_instance = None


def get_fact_checking_agent() -> FactCheckingAgent:
    """Get global fact-checking agent instance"""
    global _fact_checking_agent_instance
    if _fact_checking_agent_instance is None:
        _fact_checking_agent_instance = FactCheckingAgent()
    return _fact_checking_agent_instance

