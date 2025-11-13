"""
Fact-Checking Agent - Roosevelt's "Trust but Verify" Intelligence Unit

Specialized agent for fact-checking non-fiction content using existing web search tools.
Called by content analysis agent when factual validation is needed.
"""

import logging
import json
import re
from datetime import datetime
from typing import Any, Dict, List

from .base_agent import BaseAgent
from models.article_analysis_models import ArticleAnalysisResult

logger = logging.getLogger(__name__)


class FactCheckingAgent(BaseAgent):
    def __init__(self):
        super().__init__("research_agent")  # Use research tools for web search
        logger.info("🔍 BULLY! Fact-Checking Agent ready to verify facts with a big stick!")

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
                "results": []
            }

    async def _extract_factual_claims(self, content: str) -> List[str]:
        """Extract factual claims from content that can be verified"""
        try:
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            
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
            
            response = await chat_service.openai_client.chat.completions.create(
                model=model_name,
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
            # Use the existing web search tools
            from services.langgraph_tools.centralized_tool_registry import get_tool_registry, AgentType
            tool_registry = await get_tool_registry()
            
            # Get the search_web function for research agent
            search_web_function = tool_registry.get_tool_function("search_web", AgentType.RESEARCH_AGENT)
            if not search_web_function:
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "sources": [],
                    "status": "tool_not_available",
                    "error": "search_web tool not available for fact-checking agent"
                }
            
            # Search for the claim with more comprehensive results
            search_query = f'"{claim}" fact check verification'
            search_result = await search_web_function(
                query=search_query,
                limit=10,  # BULLY! Increased from 5 to 10 for deeper analysis
                analyze_relevance=True,
                user_id=user_id
            )
            
            if not search_result.get("success"):
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "sources": [],
                    "status": "search_failed",
                    "error": search_result.get("error", "Unknown search error")
                }
            
            # Analyze search results for verification
            verification_analysis = await self._analyze_search_results_for_verification(claim, search_result.get("results", []))
            
            return verification_analysis
            
        except Exception as e:
            logger.error(f"❌ Claim verification failed: {e}")
            return {
                "verified": False,
                "confidence": 0.0,
                "sources": [],
                "status": "verification_failed",
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
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            
            # Format search results for analysis - BULLY! Use more results for deeper analysis
            results_text = ""
            for i, result in enumerate(search_results[:8], 1):  # Top 8 results for comprehensive analysis
                results_text += f"\n{i}. {result.get('title', 'No title')}\n"
                results_text += f"   Source: {result.get('source', 'Unknown')}\n"
                results_text += f"   Content: {result.get('snippet', 'No content')}\n"
            
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
            
            response = await chat_service.openai_client.chat.completions.create(
                model=model_name,
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
                
                # Add source information - BULLY! Include more sources for comprehensive analysis
                sources = []
                for result in search_results[:8]:  # Include top 8 sources
                    sources.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "source": result.get("source", ""),
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
                "analysis": "Failed to analyze search results",
                "sources": [],
                "error": str(e)
            }

    def _analyze_verification_results(self, verification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall fact-checking results"""
        try:
            total_claims = len(verification_results)
            verified_claims = len([r for r in verification_results if r["verification"]["verified"]])
            disputed_claims = len([r for r in verification_results if r["verification"]["status"] == "disputed"])
            unverified_claims = len([r for r in verification_results if r["verification"]["status"] == "unverified"])
            
            # Calculate overall confidence
            confidence_scores = [r["verification"]["confidence"] for r in verification_results if r["verification"]["confidence"] > 0]
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            # Determine overall status
            if verified_claims / total_claims >= 0.8:
                overall_status = "highly_verified"
            elif verified_claims / total_claims >= 0.6:
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
                "verification_rate": verified_claims / total_claims if total_claims > 0 else 0.0,
                "overall_confidence": overall_confidence,
                "overall_status": overall_status,
                "summary": f"Verified {verified_claims}/{total_claims} claims ({verified_claims/total_claims*100:.1f}%)"
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
                "summary": "Failed to analyze verification results",
                "error": str(e)
            }

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process fact-checking request from content analysis agent"""
        try:
            # Extract content and claims from state
            content = state.get("content", "")
            claims = state.get("claims", [])
            user_id = state.get("user_id")
            
            if not content:
                return {
                    "agent_results": {
                        "error": "No content provided for fact-checking",
                        "success": False
                    },
                    "is_complete": True
                }
            
            # Perform fact-checking
            fact_check_result = await self.fact_check_content(content, claims, user_id)
            
            # Format results for content analysis agent
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
                    
                    if verification["sources"]:
                        summary_lines.append(f"   Sources: {len(verification['sources'])} found")
                
                if len(fact_check_result["results"]) > 5:
                    summary_lines.append(f"\n... and {len(fact_check_result['results']) - 5} more claims")
                
                state["agent_results"] = {
                    "fact_checking_results": fact_check_result,
                    "formatted_summary": "\n".join(summary_lines),
                    "success": True,
                    "mode": "fact_checking"
                }
                state["latest_response"] = "\n".join(summary_lines)
            else:
                state["agent_results"] = {
                    "error": fact_check_result.get("error", "Fact-checking failed"),
                    "success": False,
                    "mode": "fact_checking"
                }
                state["latest_response"] = f"Fact-checking failed: {fact_check_result.get('error', 'Unknown error')}"
            
            state["is_complete"] = True
            return state
            
        except Exception as e:
            logger.error(f"❌ Fact-checking agent process failed: {e}")
            state["agent_results"] = {
                "error": str(e),
                "success": False,
                "mode": "fact_checking"
            }
            state["latest_response"] = f"Fact-checking failed: {str(e)}"
            state["is_complete"] = True
            return state


# Global instance for use by content analysis agent
_fact_checking_instance = None


async def get_fact_checking_agent() -> FactCheckingAgent:
    """Get global fact-checking agent instance"""
    global _fact_checking_instance
    if _fact_checking_instance is None:
        _fact_checking_instance = FactCheckingAgent()
    return _fact_checking_instance


async def fact_check_content(content: str, claims: List[str] = None, user_id: str = None) -> Dict[str, Any]:
    """LangGraph tool function: Fact-check content using web search"""
    agent = await get_fact_checking_agent()
    return await agent.fact_check_content(content, claims, user_id)
