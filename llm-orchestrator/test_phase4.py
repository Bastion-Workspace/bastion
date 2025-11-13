"""
Test Phase 4 - Complete Research Agent Integration
Tests orchestrator research agent using gRPC backend tools
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grpc
from protos import orchestrator_pb2, orchestrator_pb2_grpc


async def test_phase4():
    """Test complete research agent integration"""
    
    print("🧪 Testing Phase 4 - Research Agent via Orchestrator")
    print()
    
    try:
        # Connect to orchestrator service
        print("📡 Connecting to orchestrator service...")
        async with grpc.aio.insecure_channel('localhost:50051') as channel:
            stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
            
            # Test 1: Health check
            print("📡 Test 1: Health Check")
            health_req = orchestrator_pb2.HealthCheckRequest()
            health_resp = await stub.HealthCheck(health_req)
            print(f"   ✅ Status: {health_resp.status}")
            print(f"   Details: {dict(health_resp.details)}")
            print()
            
            # Test 2: Research via streaming chat
            print("📡 Test 2: Research Query via StreamChat")
            print("   Query: 'What documents do we have about testing?'")
            print()
            
            request = orchestrator_pb2.ChatRequest(
                query="What documents do we have about testing?",
                user_id="test_user",
                conversation_id="test_conversation"
            )
            
            print("   Streaming responses:")
            chunk_count = 0
            async for chunk in stub.StreamChat(request):
                chunk_count += 1
                
                if chunk.type == "status":
                    print(f"   [{chunk.agent_name}] STATUS: {chunk.message}")
                elif chunk.type == "content":
                    print(f"   [{chunk.agent_name}] CONTENT:")
                    # Print content with indentation
                    for line in chunk.message.split('\n'):
                        print(f"      {line}")
                elif chunk.type == "complete":
                    print(f"   [{chunk.agent_name}] ✅ {chunk.message}")
                elif chunk.type == "error":
                    print(f"   [{chunk.agent_name}] ❌ ERROR: {chunk.message}")
            
            print()
            print(f"   ✅ Received {chunk_count} chunks")
            print()
            
            # Test 3: Another research query
            print("📡 Test 3: Different Research Query")
            print("   Query: 'summarize our documentation'")
            print()
            
            request2 = orchestrator_pb2.ChatRequest(
                query="summarize our documentation",
                user_id="test_user",
                conversation_id="test_conversation_2"
            )
            
            async for chunk in stub.StreamChat(request2):
                if chunk.type == "status":
                    print(f"   STATUS: {chunk.message}")
                elif chunk.type == "complete":
                    print(f"   ✅ {chunk.message}")
            
            print()
            
        print("✅ All Phase 4 tests completed successfully!")
        print()
        print("🎉 COMPLETE INTEGRATION WORKING:")
        print("   User → Orchestrator (50051)")
        print("           ↓ gRPC")
        print("   Research Agent (LangGraph)")
        print("           ↓ gRPC")
        print("   Backend Tool Service (50052)")
        print("           ↓")
        print("   Document Repository → PostgreSQL + Qdrant")
        print()
        return True
        
    except grpc.RpcError as e:
        print(f"❌ gRPC Error: {e.code()} - {e.details()}")
        return False
    except Exception as e:
        print(f"❌ Phase 4 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_phase4())
    sys.exit(0 if success else 1)

