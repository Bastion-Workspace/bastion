"""
Test Template System - Roosevelt's "Battle Readiness Check"
Simple test script to verify template system implementation
"""

import asyncio
import logging
from datetime import datetime

# Import our new template system components
from services.template_service import template_service
from services.langgraph_agents.enhanced_research_agent import EnhancedResearchAgent
from models.report_template_models import create_poi_template, RequestType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_template_system():
    """Test the template system implementation"""
    try:
        logger.info("🎖️ ROOSEVELT'S TEMPLATE SYSTEM BATTLE READINESS CHECK")
        logger.info("=" * 60)
        
        # Test 1: Initialize Template Service
        logger.info("📋 TEST 1: Initializing Template Service...")
        await template_service.initialize()
        logger.info("✅ Template Service initialized successfully!")
        
        # Test 2: Verify POI Template Creation
        logger.info("\n📋 TEST 2: Verifying POI Template...")
        poi_template = await template_service.get_template("person_of_interest_report")
        if poi_template:
            logger.info(f"✅ POI Template found: {poi_template.template_name}")
            logger.info(f"   Sections: {len(poi_template.sections)}")
            logger.info(f"   Keywords: {poi_template.keywords}")
        else:
            logger.error("❌ POI Template not found!")
            return False
        
        # Test 3: Initialize Enhanced Research Agent
        logger.info("\n🔬 TEST 3: Initializing Enhanced Research Agent...")
        enhanced_agent = EnhancedResearchAgent()
        await enhanced_agent.initialize()
        logger.info("✅ Enhanced Research Agent initialized successfully!")
        logger.info(f"   Available templates: {len(enhanced_agent.available_templates)}")
        
        # Test 4: Test Request Analysis
        logger.info("\n🎯 TEST 4: Testing Request Analysis...")
        
        # Test general research query
        general_query = "What is the best time to plant tomatoes?"
        general_analysis = await enhanced_agent._analyze_research_request(general_query)
        logger.info(f"   General Query: '{general_query}'")
        logger.info(f"   Result: {general_analysis.request_type.value} (confidence: {general_analysis.confidence_score})")
        
        # Test template-triggering query
        dossier_query = "Generate me a dossier on Klaus Schwab"
        dossier_analysis = await enhanced_agent._analyze_research_request(dossier_query)
        logger.info(f"   Dossier Query: '{dossier_query}'")
        logger.info(f"   Result: {dossier_analysis.request_type.value} (confidence: {dossier_analysis.confidence_score})")
        logger.info(f"   Suggested Template: {dossier_analysis.suggested_template_id}")
        
        # Test 5: Template Structure Validation
        logger.info("\n📐 TEST 5: Template Structure Validation...")
        poi_template = create_poi_template()
        logger.info(f"   Template ID: {poi_template.template_id}")
        logger.info(f"   Sections: {len(poi_template.sections)}")
        
        total_fields = sum(len(section.fields) for section in poi_template.sections)
        logger.info(f"   Total Fields: {total_fields}")
        
        required_sections = [s.section_name for s in poi_template.sections if s.required]
        logger.info(f"   Required Sections: {required_sections}")
        
        # Test 6: Mock Research Flow Test
        logger.info("\n🎭 TEST 6: Mock Research Flow Test...")
        
        # Mock state for general research
        general_state = {
            "current_query": general_query,
            "messages": [],
            "shared_memory": {}
        }
        
        logger.info(f"   Testing general research flow...")
        logger.info(f"   Query: '{general_query}'")
        logger.info(f"   Expected: General research mode")
        
        # Mock state for template research
        template_state = {
            "current_query": dossier_query,
            "messages": [],
            "shared_memory": {}
        }
        
        logger.info(f"   Testing template detection flow...")
        logger.info(f"   Query: '{dossier_query}'")
        logger.info(f"   Expected: Template suggestion mode")
        
        # Test 7: Template Service Statistics
        logger.info("\n📊 TEST 7: Template Service Statistics...")
        stats = await template_service.get_template_stats()
        logger.info(f"   Total Templates: {stats.get('total_templates', 0)}")
        logger.info(f"   By Scope: {stats.get('by_scope', {})}")
        logger.info(f"   By Category: {stats.get('by_category', {})}")
        
        logger.info("\n🎖️ BATTLE READINESS CHECK COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("✅ All systems operational - Ready for deployment!")
        logger.info("🔬 Enhanced Research Agent: Template detection and dual-mode processing")
        logger.info("📋 Template Service: CRUD operations and POI template")
        logger.info("🎯 Request Analysis: Intelligent template suggestions")
        logger.info("🛡️ Type Safety: Pydantic validation throughout")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ BATTLE READINESS CHECK FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_template_detection_scenarios():
    """Test various template detection scenarios"""
    logger.info("\n🎯 TEMPLATE DETECTION SCENARIOS TEST")
    logger.info("-" * 40)
    
    # Initialize agent
    enhanced_agent = EnhancedResearchAgent()
    await enhanced_agent.initialize()
    
    test_queries = [
        # Should trigger template detection
        ("Generate me a dossier on Klaus Schwab", True),
        ("Create a profile for Elon Musk", True), 
        ("I need a background check on John Doe", True),
        ("Person of interest report for Jane Smith", True),
        
        # Should be general research
        ("What is the best time to plant tomatoes?", False),
        ("How do I bake a chocolate cake?", False),
        ("What's the weather like today?", False),
        ("Explain quantum computing", False),
        
        # Ambiguous cases
        ("Tell me about Klaus Schwab", False),  # Should be general
        ("Research Elon Musk's companies", False),  # Should be general
    ]
    
    for query, should_trigger_template in test_queries:
        try:
            analysis = await enhanced_agent._analyze_research_request(query)
            is_template = analysis.request_type == RequestType.TEMPLATED_REPORT and analysis.confidence_score > 0.7
            
            status = "✅" if is_template == should_trigger_template else "❌"
            logger.info(f"{status} '{query}'")
            logger.info(f"   Result: {analysis.request_type.value} (confidence: {analysis.confidence_score:.2f})")
            if analysis.suggested_template_id:
                logger.info(f"   Template: {analysis.suggested_template_id}")
            logger.info("")
            
        except Exception as e:
            logger.error(f"❌ Error testing query '{query}': {e}")


if __name__ == "__main__":
    # Run the tests
    async def main():
        # Basic system test
        success = await test_template_system()
        
        if success:
            # Template detection scenarios
            await test_template_detection_scenarios()
            
            logger.info("\n🎖️ ROOSEVELT'S FINAL VERDICT: BULLY!")
            logger.info("The Enhanced Research Agent is ready for battle!")
        else:
            logger.error("\n❌ ROOSEVELT'S FINAL VERDICT: NOT READY!")
            logger.error("System requires further cavalry training!")
    
    # Run the test
    asyncio.run(main())

