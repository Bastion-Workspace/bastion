"""
Knowledge Document Synthesis Subgraph

Reusable subgraph for structuring verified knowledge into professional markdown documents:
- Organize findings hierarchically
- Generate document sections (Executive Summary, Core Findings, Evidence, Contradictions)
- Format citations with footnotes
- Create YAML frontmatter
- Assemble final markdown document

Used by knowledge_builder_agent for truth investigation output.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


# Use Dict[str, Any] for compatibility with main agent state
MarkdownSynthesisState = Dict[str, Any]


async def organize_findings_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Organize verified findings into hierarchical structure"""
    try:
        verified_claims = state.get("verified_claims", [])
        contradictions = state.get("contradictions", [])
        uncertainties = state.get("uncertainties", [])
        original_query = state.get("query", "")
        
        logger.info("Organizing findings into hierarchical structure")
        
        # Group claims by topic/theme using LLM
        organize_prompt = f"""Organize the following verified claims into logical topics/themes for a knowledge document.

ORIGINAL QUERY: {original_query}

VERIFIED CLAIMS:
{json.dumps([c.get("claim", "") for c in verified_claims[:20]], indent=2)}

CONTRADICTIONS:
{json.dumps(contradictions[:10], indent=2)}

UNCERTAINTIES:
{json.dumps(uncertainties[:10], indent=2)}

Organize into:
1. Main topics/themes (group related claims)
2. Sub-topics under each main topic
3. Priority order (most important findings first)

STRUCTURED OUTPUT REQUIRED - Respond with ONLY valid JSON matching this exact schema:
{{
    "topics": [
        {{
            "topic_name": "Main Topic Name",
            "subtopics": ["Sub-topic 1", "Sub-topic 2"],
            "claims": ["list of claim texts in this topic"],
            "priority": number (1-10, higher = more important)
        }}
    ],
    "executive_summary_points": ["key point 1", "key point 2", "key point 3"]
}}"""
        
        base_agent = BaseAgent("markdown_synthesis_subgraph")
        llm = base_agent._get_llm(temperature=0.5, state=state)
        datetime_context = base_agent._get_datetime_context()
        
        messages = [
            SystemMessage(content="You are a knowledge organization specialist. Always respond with valid JSON."),
            SystemMessage(content=datetime_context),
            HumanMessage(content=organize_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        
        # Parse response
        try:
            text = response.content.strip()
            if '```json' in text:
                import re
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            elif '```' in text:
                import re
                m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            
            import re
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                text = json_match.group(0)
            
            structure = json.loads(text)
            topics = structure.get("topics", [])
            executive_summary_points = structure.get("executive_summary_points", [])
            
            # Sort topics by priority
            topics.sort(key=lambda x: x.get("priority", 0), reverse=True)
            
            logger.info(f"Organized findings into {len(topics)} topics, {len(executive_summary_points)} summary points")
            
            return {
                "knowledge_structure": {
                    "topics": topics,
                    "executive_summary_points": executive_summary_points
                }
            }
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse organization: {e}")
            # Fallback: simple structure
            return {
                "knowledge_structure": {
                    "topics": [{
                        "topic_name": "Core Findings",
                        "subtopics": [],
                        "claims": [c.get("claim", "") for c in verified_claims[:10]],
                        "priority": 5
                    }],
                    "executive_summary_points": [c.get("claim", "") for c in verified_claims[:3]]
                }
            }
        
    except Exception as e:
        logger.error(f"Organize findings error: {e}")
        return {
            "knowledge_structure": {
                "topics": [],
                "executive_summary_points": []
            }
        }


async def generate_sections_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate markdown sections from organized findings"""
    try:
        knowledge_structure = state.get("knowledge_structure", {})
        verified_claims = state.get("verified_claims", [])
        contradictions = state.get("contradictions", [])
        uncertainties = state.get("uncertainties", [])
        original_query = state.get("query", "")
        sources_found = state.get("sources_found", [])
        
        logger.info("Generating markdown sections")
        
        # Build Executive Summary
        executive_summary_points = knowledge_structure.get("executive_summary_points", [])
        executive_summary = "## Executive Summary\n\n"
        if executive_summary_points:
            for i, point in enumerate(executive_summary_points[:5], 1):
                executive_summary += f"{i}. {point}\n\n"
        else:
            executive_summary += "This investigation compiled verified information from multiple sources.\n\n"
        
        # Build Core Findings
        core_findings = "## Core Findings\n\n"
        topics = knowledge_structure.get("topics", [])
        
        for topic in topics[:10]:  # Limit to top 10 topics
            topic_name = topic.get("topic_name", "Untitled Topic")
            claims = topic.get("claims", [])
            
            if claims:
                core_findings += f"### {topic_name}\n\n"
                
                for claim_text in claims[:5]:  # Limit to top 5 claims per topic
                    # Find matching verified claim
                    matching_claim = None
                    for vc in verified_claims:
                        if claim_text.lower() in vc.get("claim", "").lower():
                            matching_claim = vc
                            break
                    
                    status_icon = "✓" if matching_claim and matching_claim.get("status") == "verified" else "⚠"
                    confidence = matching_claim.get("confidence", "medium") if matching_claim else "medium"
                    
                    core_findings += f"**{status_icon} {claim_text}**\n"
                    core_findings += f"*Status: {confidence.capitalize()} confidence*\n\n"
        
        # Build Supporting Evidence
        supporting_evidence = "## Supporting Evidence\n\n"
        supporting_evidence += f"**Total Sources**: {len(sources_found)}\n\n"
        
        # Group sources by type
        local_sources = [s for s in sources_found if s.get("source") == "local"]
        web_sources = [s for s in sources_found if s.get("source") == "web"]
        
        if local_sources:
            supporting_evidence += "### Local Sources\n\n"
            for source in local_sources[:5]:
                doc_id = source.get("document_id", "unknown")
                supporting_evidence += f"- Document ID: {doc_id}\n"
        
        if web_sources:
            supporting_evidence += "\n### Web Sources\n\n"
            for source in web_sources[:10]:
                url = source.get("url", "")
                if url:
                    supporting_evidence += f"- [{url}]({url})\n"
        
        # Build Contradictions & Uncertainties
        contradictions_section = "## Contradictions & Uncertainties\n\n"
        
        if contradictions:
            contradictions_section += "### Conflicting Information\n\n"
            for i, contradiction in enumerate(contradictions[:5], 1):
                claim = contradiction.get("claim", "")
                source_a = contradiction.get("source_a", "")
                source_b = contradiction.get("source_b", "")
                assessment = contradiction.get("assessment", "")
                severity = contradiction.get("severity", "moderate")
                
                contradictions_section += f"**{i}. {claim}**\n"
                contradictions_section += f"- Source A: {source_a}\n"
                contradictions_section += f"- Source B: {source_b}\n"
                contradictions_section += f"- Severity: {severity.capitalize()}\n"
                contradictions_section += f"- Assessment: {assessment}\n\n"
        else:
            contradictions_section += "No major contradictions detected.\n\n"
        
        if uncertainties:
            contradictions_section += "### Unanswered Questions\n\n"
            for i, uncertainty in enumerate(uncertainties[:5], 1):
                contradictions_section += f"{i}. {uncertainty}\n"
            contradictions_section += "\n"
        
        # Build Citations section placeholder (will be filled in next node)
        citations_section = "## Citations\n\n"
        citations_section += "*Citations will be added in final assembly*\n\n"
        
        sections = {
            "executive_summary": executive_summary,
            "core_findings": core_findings,
            "supporting_evidence": supporting_evidence,
            "contradictions": contradictions_section,
            "citations": citations_section
        }
        
        logger.info(f"Generated {len(sections)} markdown sections")
        
        return {
            "markdown_sections": sections
        }
        
    except Exception as e:
        logger.error(f"Generate sections error: {e}")
        return {
            "markdown_sections": {}
        }


async def format_citations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Format citations with footnote references"""
    try:
        sources_found = state.get("sources_found", [])
        independent_sources = state.get("independent_sources", [])
        
        logger.info("Formatting citations")
        
        # Combine all sources
        all_sources = []
        seen_urls = set()
        seen_doc_ids = set()
        
        for source in sources_found:
            if source.get("type") == "document":
                doc_id = source.get("document_id", "")
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    all_sources.append({
                        "type": "document",
                        "document_id": doc_id,
                        "citation_number": len(all_sources) + 1
                    })
            elif source.get("type") == "web":
                url = source.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_sources.append({
                        "type": "web",
                        "url": url,
                        "citation_number": len(all_sources) + 1
                    })
        
        for source in independent_sources:
            url = source.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_sources.append({
                    "type": "verification",
                    "url": url,
                    "citation_number": len(all_sources) + 1
                })
        
        # Build citations list
        citations_list = []
        for source in all_sources:
            citation_num = source.get("citation_number", 0)
            if source.get("type") == "document":
                citations_list.append(f"[^{citation_num}]: Document ID: {source.get('document_id', 'unknown')}")
            elif source.get("type") == "web" or source.get("type") == "verification":
                url = source.get("url", "")
                citations_list.append(f"[^{citation_num}]: {url}")
        
        citations_text = "## Citations\n\n"
        citations_text += "\n".join(citations_list)
        citations_text += "\n\n"
        
        logger.info(f"Formatted {len(all_sources)} citations")
        
        return {
            "citations_formatted": citations_text,
            "citation_map": {s.get("citation_number"): s for s in all_sources}
        }
        
    except Exception as e:
        logger.error(f"Format citations error: {e}")
        return {
            "citations_formatted": "## Citations\n\n*Citations unavailable*\n\n",
            "citation_map": {}
        }


async def create_frontmatter_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate YAML frontmatter for document"""
    try:
        original_query = state.get("query", "")
        verified_claims = state.get("verified_claims", [])
        contradictions = state.get("contradictions", [])
        sources_found = state.get("sources_found", [])
        
        logger.info("Creating YAML frontmatter")
        
        # Extract topic from query
        topic = original_query[:100] if original_query else "Investigation"
        if len(original_query) > 100:
            topic = original_query[:97] + "..."
        
        # Determine verification status
        verification_status = "verified"
        if contradictions:
            verification_status = "partial"
        if len(verified_claims) == 0:
            verification_status = "unverified"
        
        # Build frontmatter
        frontmatter = {
            "title": f"Truth Investigation: {topic}",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "agent": "knowledge_builder",
            "query": original_query,
            "sources_count": len(sources_found),
            "verification_status": verification_status,
            "tags": ["truth-seeking", "research", verification_status],
            "confidence": "high" if len(verified_claims) > len(contradictions) * 2 else "medium"
        }
        
        frontmatter_text = "---\n"
        for key, value in frontmatter.items():
            if isinstance(value, list):
                frontmatter_text += f"{key}: {json.dumps(value)}\n"
            elif isinstance(value, str) and ("'" in value or '"' in value):
                frontmatter_text += f"{key}: {json.dumps(value)}\n"
            else:
                frontmatter_text += f"{key}: {value}\n"
        frontmatter_text += "---\n\n"
        
        logger.info("Created YAML frontmatter")
        
        return {
            "document_metadata": frontmatter,
            "frontmatter_text": frontmatter_text
        }
        
    except Exception as e:
        logger.error(f"Create frontmatter error: {e}")
        return {
            "document_metadata": {},
            "frontmatter_text": "---\ntitle: Investigation\ndate: " + datetime.now().strftime("%Y-%m-%d") + "\n---\n\n"
        }


async def assemble_markdown_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble final markdown document"""
    try:
        frontmatter_text = state.get("frontmatter_text", "")
        markdown_sections = state.get("markdown_sections", {})
        citations_formatted = state.get("citations_formatted", "")
        original_query = state.get("query", "")
        
        logger.info("Assembling final markdown document")
        
        # Build title from query
        title = original_query[:100] if original_query else "Investigation"
        if len(original_query) > 100:
            title = original_query[:97] + "..."
        
        # Assemble document
        markdown_content = frontmatter_text
        markdown_content += f"# {title}\n\n"
        
        # Add sections in order
        if markdown_sections.get("executive_summary"):
            markdown_content += markdown_sections["executive_summary"]
        
        if markdown_sections.get("core_findings"):
            markdown_content += markdown_sections["core_findings"]
        
        if markdown_sections.get("supporting_evidence"):
            markdown_content += markdown_sections["supporting_evidence"]
        
        if markdown_sections.get("contradictions"):
            markdown_content += markdown_sections["contradictions"]
        
        # Add citations
        markdown_content += citations_formatted
        
        # Add footer
        markdown_content += "---\n\n"
        markdown_content += f"*Generated by Knowledge Builder Agent on {datetime.now().strftime('%Y-%m-%d')}*\n"
        markdown_content += f"*Next review recommended: {(datetime.now().replace(month=datetime.now().month + 6) if datetime.now().month <= 6 else datetime.now().replace(year=datetime.now().year + 1, month=datetime.now().month - 6)).strftime('%Y-%m-%d')}*\n"
        
        logger.info(f"Assembled markdown document ({len(markdown_content)} characters)")
        
        return {
            "markdown_content": markdown_content
        }
        
    except Exception as e:
        logger.error(f"Assemble markdown error: {e}")
        return {
            "markdown_content": "# Investigation\n\n*Error generating document*\n"
        }


def build_knowledge_document_subgraph(checkpointer) -> StateGraph:
    """Build knowledge document synthesis subgraph"""
    subgraph = StateGraph(Dict[str, Any])
    
    # Add nodes
    subgraph.add_node("organize_findings", organize_findings_node)
    subgraph.add_node("generate_sections", generate_sections_node)
    subgraph.add_node("format_citations", format_citations_node)
    subgraph.add_node("create_frontmatter", create_frontmatter_node)
    subgraph.add_node("assemble_markdown", assemble_markdown_node)
    
    # Set entry point
    subgraph.set_entry_point("organize_findings")
    
    # Flow - parallel paths for efficiency
    subgraph.add_edge("organize_findings", "generate_sections")
    subgraph.add_edge("generate_sections", "format_citations")
    subgraph.add_edge("format_citations", "create_frontmatter")
    subgraph.add_edge("create_frontmatter", "assemble_markdown")
    subgraph.add_edge("assemble_markdown", END)
    
    return subgraph.compile(checkpointer=checkpointer)







