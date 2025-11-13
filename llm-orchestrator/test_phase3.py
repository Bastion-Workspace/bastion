"""
Test Phase 3 - Orchestrator to Backend Tool Service Communication
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator.backend_tool_client import get_backend_tool_client, close_backend_tool_client
from orchestrator.tools import search_documents_tool, get_document_content_tool


async def test_phase3():
    """Test orchestrator to backend communication"""
    
    print("🧪 Testing Phase 3 - Orchestrator → Backend Tool Service")
    print()
    
    try:
        # Test 1: Backend client connection
        print("📡 Test 1: Backend Client Connection")
        client = await get_backend_tool_client()
        print(f"   ✅ Connected to backend at {client.address}")
        print()
        
        # Test 2: Document search via client
        print("📡 Test 2: Document Search (Direct Client)")
        result = await client.search_documents(
            query="test",
            limit=3
        )
        print(f"   ✅ Search completed: {result['total_count']} results")
        if result['results']:
            for doc in result['results'][:2]:
                print(f"      - {doc['title']}")
        print()
        
        # Test 3: Document search via tool
        print("📡 Test 3: Document Search (Via Tool)")
        tool_result = await search_documents_tool(
            query="test",
            limit=3
        )
        print(f"   ✅ Tool result: {len(tool_result)} characters")
        print(f"   Preview: {tool_result[:200]}...")
        print()
        
        # Test 4: Weather data (placeholder)
        print("📡 Test 4: Weather Data (Placeholder)")
        weather = await client.get_weather(
            location="New York"
        )
        if weather:
            print(f"   ✅ Weather data received for {weather['location']}")
        else:
            print(f"   ⚠️  Weather service not yet implemented (expected)")
        print()
        
        # Test 5: Entity search (placeholder)
        print("📡 Test 5: Entity Search (Placeholder)")
        entities = await client.search_entities(
            query="test entity"
        )
        print(f"   ✅ Entity search completed: {len(entities)} results (placeholder)")
        print()
        
        print("✅ Phase 3 tests completed successfully!")
        print("   Orchestrator ↔ Backend communication is working")
        print()
        
    except Exception as e:
        print(f"❌ Phase 3 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        await close_backend_tool_client()
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_phase3())
    sys.exit(0 if success else 1)

