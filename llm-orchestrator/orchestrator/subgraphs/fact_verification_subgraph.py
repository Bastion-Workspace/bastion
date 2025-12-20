"""
Fact Verification Subgraph

Reusable subgraph for truth-seeking and fact verification:
- Extract claims from research findings
- Cross-reference claims with independent sources
- Assess source credibility
- Detect contradictions
- Build consensus on verified claims

Used by knowledge_builder_agent for truth investigation.
"""

import logging
import json
import re
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.tools import search_web_tool, crawl_web_content_tool
from orchestrator.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


# Use Dict[str, Any] for compatibility with main agent state
VerificationSubgraphState = Dict[str, Any]


async def extract_claims_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract factual claims from research findings"""
    try:
        research_findings = state.get("research_findings", {})
        combined_results = research_findings.get("combined_results", "")
        
        if not combined_results:
            logger.warning("No research findings to extract claims from")
            return {
                "claims_identified": [],
                "verification_needed": False
            }
        
        logger.info("Extracting claims from research findings")
        
        extract_prompt = f"""Extract all factual claims from the following research findings. Focus on objective, verifiable statements.

RESEARCH FINDINGS:
{combined_results[:10000]}

Extract:
1. Specific factual claims (dates, numbers, names, relationships, events)
2. Categorize each claim as: "factual", "opinion", or "uncertain"
3. Note the source context for each claim

STRUCTURED OUTPUT REQUIRED - Respond with ONLY valid JSON matching this exact schema:
{{
    "claims": [
        {{
            "claim": "specific factual statement",
            "category": "factual" | "opinion" | "uncertain",
            "context": "source context or quote",
            "needs_verification": boolean
        }}
    ],
    "total_claims": number,
    "factual_claims_count": number
}}

Focus on extracting the most important factual claims that should be verified."""
        
        base_agent = BaseAgent("verification_subgraph")
        llm = base_agent._get_llm(temperature=0.3, state=state)
        datetime_context = base_agent._get_datetime_context()
        
        messages = [
            SystemMessage(content="You are a claim extraction specialist. Always respond with valid JSON."),
            SystemMessage(content=datetime_context),
            HumanMessage(content=extract_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        
        # Parse response
        try:
            text = response.content.strip()
            if '```json' in text:
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            elif '```' in text:
                m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                text = json_match.group(0)
            
            result = json.loads(text)
            claims = result.get("claims", [])
            factual_claims = [c for c in claims if c.get("category") == "factual" and c.get("needs_verification", True)]
            
            logger.info(f"Extracted {len(claims)} total claims, {len(factual_claims)} factual claims need verification")
            
            return {
                "claims_identified": claims,
                "factual_claims": factual_claims,
                "verification_needed": len(factual_claims) > 0
            }
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse claims extraction: {e}")
            return {
                "claims_identified": [],
                "factual_claims": [],
                "verification_needed": False
            }
        
    except Exception as e:
        logger.error(f"Extract claims error: {e}")
        return {
            "claims_identified": [],
            "factual_claims": [],
            "verification_needed": False
        }


async def cross_reference_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Cross-reference claims with independent sources"""
    try:
        factual_claims = state.get("factual_claims", [])
        
        if not factual_claims:
            logger.info("No factual claims to cross-reference")
            return {
                "verification_results": [],
                "independent_sources": []
            }
        
        logger.info(f"Cross-referencing {len(factual_claims)} factual claims")
        
        # Track tool usage
        shared_memory = state.get("shared_memory", {})
        previous_tools = shared_memory.get("previous_tools_used", [])
        if "search_web_tool" not in previous_tools:
            previous_tools.append("search_web_tool")
            previous_tools.append("crawl_web_content_tool")
            shared_memory["previous_tools_used"] = previous_tools
            state["shared_memory"] = shared_memory
        
        # For each major claim, search for independent verification
        verification_results = []
        independent_sources = []
        
        # Limit to top 5 claims to avoid too many searches
        for claim_obj in factual_claims[:5]:
            claim = claim_obj.get("claim", "")
            if not claim:
                continue
            
            try:
                # Search for independent verification
                search_query = f'"{claim}" verification fact check'
                search_result = await search_web_tool(query=search_query, max_results=5)
                logger.info(f"Tool used: search_web_tool (cross-reference: {claim[:50]}...)")
                
                # Extract URLs
                urls = re.findall(r'URL: (https?://[^\s]+)', search_result)
                
                # Crawl top 2 URLs for verification
                crawled_content = ""
                if urls:
                    top_urls = urls[:2]
                    crawl_result = await crawl_web_content_tool(urls=top_urls)
                    logger.info(f"Tool used: crawl_web_content_tool (verification crawl)")
                    crawled_content = crawl_result
                    
                    for url in top_urls:
                        independent_sources.append({
                            "url": url,
                            "type": "verification",
                            "claim": claim
                        })
                
                verification_results.append({
                    "claim": claim,
                    "verification_query": search_query,
                    "search_results": search_result,
                    "crawled_content": crawled_content,
                    "sources_found": len(urls),
                    "verified": len(urls) > 0
                })
                
            except Exception as e:
                logger.warning(f"Failed to cross-reference claim '{claim[:50]}...': {e}")
                verification_results.append({
                    "claim": claim,
                    "verified": False,
                    "error": str(e)
                })
        
        logger.info(f"Cross-referenced {len(verification_results)} claims, found {len(independent_sources)} independent sources")
        
        return {
            "verification_results": verification_results,
            "independent_sources": independent_sources
        }
        
    except Exception as e:
        logger.error(f"Cross-reference error: {e}")
        return {
            "verification_results": [],
            "independent_sources": []
        }


async def assess_credibility_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Assess source credibility and recency"""
    try:
        verification_results = state.get("verification_results", [])
        sources_found = state.get("sources_found", [])
        independent_sources = state.get("independent_sources", [])
        
        logger.info("Assessing source credibility")
        
        # Combine all sources
        all_sources = []
        for source in sources_found:
            all_sources.append({
                "type": source.get("type", "unknown"),
                "url": source.get("url", ""),
                "document_id": source.get("document_id", ""),
                "source": source.get("source", "unknown")
            })
        
        for source in independent_sources:
            all_sources.append({
                "type": "verification",
                "url": source.get("url", ""),
                "source": "independent_verification"
            })
        
        # Simple credibility assessment based on domain patterns
        credible_domains = [
            "edu", "gov", "org", "ac.uk", "edu.au",
            "wikipedia.org", "scholar.google", "pubmed", "arxiv"
        ]
        
        for source in all_sources:
            url = source.get("url", "")
            credibility_score = 0.5  # Default
            
            if url:
                # Check domain
                for domain in credible_domains:
                    if domain in url.lower():
                        credibility_score = 0.8
                        break
                
                # Wikipedia gets high score
                if "wikipedia.org" in url.lower():
                    credibility_score = 0.7
                
                # Academic sources get highest
                if any(x in url.lower() for x in ["edu", "ac.uk", "scholar", "pubmed", "arxiv"]):
                    credibility_score = 0.9
            
            source["credibility_score"] = credibility_score
        
        logger.info(f"Assessed credibility for {len(all_sources)} sources")
        
        return {
            "sources_with_credibility": all_sources,
            "high_credibility_sources": [s for s in all_sources if s.get("credibility_score", 0) >= 0.7]
        }
        
    except Exception as e:
        logger.error(f"Assess credibility error: {e}")
        return {
            "sources_with_credibility": [],
            "high_credibility_sources": []
        }


async def detect_contradictions_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Detect contradictions across sources"""
    try:
        claims_identified = state.get("claims_identified", [])
        verification_results = state.get("verification_results", [])
        research_findings = state.get("research_findings", {})
        
        logger.info("Detecting contradictions")
        
        # Use LLM to analyze for contradictions
        analysis_prompt = f"""Analyze the following claims and verification results for contradictions or conflicting information.

CLAIMS:
{json.dumps([c.get("claim", "") for c in claims_identified[:10]], indent=2)}

VERIFICATION RESULTS:
{json.dumps(verification_results[:5], indent=2)}

RESEARCH FINDINGS:
{research_findings.get("combined_results", "")[:5000]}

Identify:
1. Contradictory claims (same fact stated differently)
2. Conflicting sources (different sources say different things)
3. Uncertainty indicators (sources disagree or are unclear)

STRUCTURED OUTPUT REQUIRED - Respond with ONLY valid JSON matching this exact schema:
{{
    "contradictions": [
        {{
            "claim": "the disputed claim",
            "source_a": "what source A says",
            "source_b": "what source B says",
            "severity": "minor" | "moderate" | "major",
            "assessment": "which source is more credible and why"
        }}
    ],
    "uncertainties": ["list of uncertain claims"],
    "consensus_claims": ["list of claims with broad agreement"]
}}"""
        
        base_agent = BaseAgent("verification_subgraph")
        llm = base_agent._get_llm(temperature=0.3, state=state)
        datetime_context = base_agent._get_datetime_context()
        
        messages = [
            SystemMessage(content="You are a contradiction detection specialist. Always respond with valid JSON."),
            SystemMessage(content=datetime_context),
            HumanMessage(content=analysis_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        
        # Parse response
        try:
            text = response.content.strip()
            if '```json' in text:
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            elif '```' in text:
                m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                text = json_match.group(0)
            
            result = json.loads(text)
            contradictions = result.get("contradictions", [])
            uncertainties = result.get("uncertainties", [])
            consensus_claims = result.get("consensus_claims", [])
            
            logger.info(f"Detected {len(contradictions)} contradictions, {len(uncertainties)} uncertainties, {len(consensus_claims)} consensus claims")
            
            return {
                "contradictions": contradictions,
                "uncertainties": uncertainties,
                "consensus_claims": consensus_claims
            }
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse contradiction detection: {e}")
            return {
                "contradictions": [],
                "uncertainties": [],
                "consensus_claims": []
            }
        
    except Exception as e:
        logger.error(f"Detect contradictions error: {e}")
        return {
            "contradictions": [],
            "uncertainties": [],
            "consensus_claims": []
        }


async def build_consensus_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build consensus on verified claims"""
    try:
        claims_identified = state.get("claims_identified", [])
        verification_results = state.get("verification_results", [])
        contradictions = state.get("contradictions", [])
        consensus_claims = state.get("consensus_claims", [])
        sources_with_credibility = state.get("sources_with_credibility", [])
        
        logger.info("Building consensus on verified claims")
        
        # Build verified claims list
        verified_claims = []
        
        # Add consensus claims
        for claim_text in consensus_claims:
            # Find matching claim object
            matching_claim = None
            for claim_obj in claims_identified:
                if claim_text.lower() in claim_obj.get("claim", "").lower():
                    matching_claim = claim_obj
                    break
            
            if matching_claim:
                verified_claims.append({
                    "claim": matching_claim.get("claim", claim_text),
                    "status": "verified",
                    "confidence": "high",
                    "sources": len([s for s in sources_with_credibility if s.get("credibility_score", 0) >= 0.7]),
                    "category": matching_claim.get("category", "factual")
                })
        
        # Add verified claims from verification results
        for result in verification_results:
            if result.get("verified"):
                claim = result.get("claim", "")
                if claim and not any(vc.get("claim", "") == claim for vc in verified_claims):
                    verified_claims.append({
                        "claim": claim,
                        "status": "verified",
                        "confidence": "medium",
                        "sources": result.get("sources_found", 0),
                        "category": "factual"
                    })
        
        # Build consensus findings
        consensus_findings = {
            "verified_claims": verified_claims,
            "contradictions": contradictions,
            "uncertainties": state.get("uncertainties", []),
            "total_claims": len(claims_identified),
            "verified_count": len(verified_claims),
            "contradiction_count": len(contradictions)
        }
        
        logger.info(f"Built consensus: {len(verified_claims)} verified claims, {len(contradictions)} contradictions")
        
        # Determine if more research is needed
        needs_more_research = (
            len(contradictions) > 0 and len(verified_claims) < len(claims_identified) * 0.5
        )
        
        return {
            "verified_claims": verified_claims,
            "consensus_findings": consensus_findings,
            "needs_more_research": needs_more_research
        }
        
    except Exception as e:
        logger.error(f"Build consensus error: {e}")
        return {
            "verified_claims": [],
            "consensus_findings": {},
            "needs_more_research": False
        }


def build_fact_verification_subgraph(checkpointer) -> StateGraph:
    """Build fact verification subgraph"""
    subgraph = StateGraph(Dict[str, Any])
    
    # Add nodes
    subgraph.add_node("extract_claims", extract_claims_node)
    subgraph.add_node("cross_reference", cross_reference_node)
    subgraph.add_node("assess_credibility", assess_credibility_node)
    subgraph.add_node("detect_contradictions", detect_contradictions_node)
    subgraph.add_node("build_consensus", build_consensus_node)
    
    # Set entry point
    subgraph.set_entry_point("extract_claims")
    
    # Flow
    subgraph.add_conditional_edges(
        "extract_claims",
        lambda state: "cross_reference" if state.get("verification_needed") else "build_consensus",
        {
            "cross_reference": "cross_reference",
            "build_consensus": "build_consensus"
        }
    )
    
    subgraph.add_edge("cross_reference", "assess_credibility")
    subgraph.add_edge("assess_credibility", "detect_contradictions")
    subgraph.add_edge("detect_contradictions", "build_consensus")
    subgraph.add_edge("build_consensus", END)
    
    return subgraph.compile(checkpointer=checkpointer)







