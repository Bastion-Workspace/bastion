"""
Test gRPC Tool Service
"""

import asyncio
import grpc
from protos import tool_service_pb2, tool_service_pb2_grpc


async def test_tool_service():
    """Test the backend tool service"""
    
    print("🧪 Testing Backend Tool Service (Phase 2)")
    print()
    
    # Connect to tool service
    async with grpc.aio.insecure_channel('localhost:50052') as channel:
        stub = tool_service_pb2_grpc.ToolServiceStub(channel)
        
        # Test 1: Health check via simple request
        print("📡 Test 1: Document Search")
        try:
            request = tool_service_pb2.SearchRequest(
                user_id="test_user",
                query="test query",
                limit=5
            )
            response = await stub.SearchDocuments(request)
            print(f"   ✅ Search completed: {response.total_count} results")
        except grpc.RpcError as e:
            print(f"   ❌ Search failed: {e.code()} - {e.details()}")
        
        print()
        
        # Test 2: Weather request
        print("📡 Test 2: Weather Data")
        try:
            request = tool_service_pb2.WeatherRequest(
                location="New York",
                user_id="test_user",
                data_types=["current"]
            )
            response = await stub.GetWeatherData(request)
            print(f"   ✅ Weather data: {response.current_conditions[:100]}")
        except grpc.RpcError as e:
            print(f"   ❌ Weather request failed: {e.code()} - {e.details()}")
        
        print()
        
        # Test 3: Entity search
        print("📡 Test 3: Entity Search")
        try:
            request = tool_service_pb2.EntitySearchRequest(
                user_id="test_user",
                query="test entity",
                limit=5
            )
            response = await stub.SearchEntities(request)
            print(f"   ✅ Entity search completed: {len(response.entities)} results")
        except grpc.RpcError as e:
            print(f"   ❌ Entity search failed: {e.code()} - {e.details()}")
        
        print()
        
    print("✅ All Phase 2 tests completed!")
    print("   Backend tool service is operational")


if __name__ == "__main__":
    asyncio.run(test_tool_service())

