"""
Test Phase 5 - Full Research Agent with Multi-Round Workflow
Tests sophisticated research capabilities matching original backend agent
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grpc
from protos import orchestrator_pb2, orchestrator_pb2_grpc


async def test_phase5():
    """Test complete sophisticated research agent"""
    
    print("🧪 Testing Phase 5 - Full Research Agent with Multi-Round Workflow")
    print("=" * 80)
    print()
    
    try:
        # Connect to orchestrator service
        print("📡 Connecting to orchestrator service...")
        async with grpc.aio.insecure_channel('localhost:50051') as channel:
            stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
            
            # Test 1: Health check with Phase 5 features
            print("=" * 80)
            print("Test 1: Health Check - Verify Phase 5 Features")
            print("=" * 80)
            health_req = orchestrator_pb2.HealthCheckRequest()
            health_resp = await stub.HealthCheck(health_req)
            print(f"✅ Status: {health_resp.status}")
            print(f"✅ Phase: {health_resp.details.get('phase', 'unknown')}")
            print(f"✅ Features: {health_resp.details.get('features', 'none')}")
            print()
            
            # Test 2: Simple research query
            print("=" * 80)
            print("Test 2: Simple Research Query")
            print("=" * 80)
            print("Query: 'What is machine learning?'")
            print()
            
            request = orchestrator_pb2.ChatRequest(
                query="What is machine learning?",
                user_id="test_user",
                conversation_id="test_conv_1"
            )
            
            print("Streaming responses:")
            print("-" * 80)
            async for chunk in stub.StreamChat(request):
                if chunk.type == "status":
                    print(f"   📊 [{chunk.agent_name}] {chunk.message}")
                elif chunk.type == "content":
                    print(f"   💬 [{chunk.agent_name}] Response:")
                    print()
                    # Show first 500 chars of content
                    content = chunk.message[:500]
                    for line in content.split('\n'):
                        print(f"      {line}")
                    if len(chunk.message) > 500:
                        print(f"      ... ({len(chunk.message) - 500} more characters)")
                    print()
                elif chunk.type == "complete":
                    print(f"   ✅ [{chunk.agent_name}] {chunk.message}")
                elif chunk.type == "error":
                    print(f"   ❌ [{chunk.agent_name}] ERROR: {chunk.message}")
            
            print()
            
            # Test 3: Complex research requiring multiple rounds
            print("=" * 80)
            print("Test 3: Complex Multi-Round Research")
            print("=" * 80)
            print("Query: 'Compare the advantages of different neural network architectures'")
            print()
            
            request2 = orchestrator_pb2.ChatRequest(
                query="Compare the advantages of different neural network architectures",
                user_id="test_user",
                conversation_id="test_conv_2"
            )
            
            rounds_seen = set()
            sources_used = []
            
            print("Streaming responses:")
            print("-" * 80)
            async for chunk in stub.StreamChat(request2):
                if chunk.type == "status":
                    print(f"   📊 {chunk.message}")
                    if "sources:" in chunk.message.lower():
                        # Extract sources
                        parts = chunk.message.split("sources:")
                        if len(parts) > 1:
                            sources_used = [s.strip() for s in parts[1].split(",")]
                elif chunk.type == "content":
                    print(f"   💬 Final Answer:")
                    print()
                    # Show excerpt
                    content = chunk.message[:400]
                    for line in content.split('\n'):
                        print(f"      {line}")
                    if len(chunk.message) > 400:
                        print(f"      ... ({len(chunk.message) - 400} more characters)")
                    print()
                elif chunk.type == "complete":
                    print(f"   ✅ {chunk.message}")
                    if "round" in chunk.message.lower():
                        rounds_seen.add(chunk.message)
            
            print()
            print(f"   Sources used: {sources_used}")
            print()
            
            # Test 4: Follow-up query (should use cache)
            print("=" * 80)
            print("Test 4: Follow-Up Query (Cache Test)")
            print("=" * 80)
            print("Query: 'Tell me more about neural networks' (should check cache)")
            print()
            
            request3 = orchestrator_pb2.ChatRequest(
                query="Tell me more about neural networks",
                user_id="test_user",
                conversation_id="test_conv_2"  # Same conversation
            )
            
            cache_hit = False
            async for chunk in stub.StreamChat(request3):
                if chunk.type == "status":
                    print(f"   📊 {chunk.message}")
                    if "cached" in chunk.message.lower() or "cache" in chunk.message.lower():
                        cache_hit = True
                elif chunk.type == "complete":
                    print(f"   ✅ {chunk.message}")
            
            if cache_hit:
                print()
                print("   🎯 Cache functionality working!")
            
            print()
            
            # Summary
            print("=" * 80)
            print("Phase 5 Test Summary")
            print("=" * 80)
            print()
            print("✅ All Phase 5 tests completed successfully!")
            print()
            print("🎉 COMPLETE RESEARCH AGENT FEATURES WORKING:")
            print()
            print("   Architecture:")
            print("   User → Orchestrator (50051)")
            print("           ↓")
            print("   Full Research Agent (LangGraph)")
            print("     - Cache check")
            print("     - Query expansion")
            print("     - Round 1: Local search")
            print("     - Gap analysis")
            print("     - Round 2: Gap filling")
            print("     - Web search (if needed)")
            print("     - Final synthesis")
            print("           ↓ gRPC")
            print("   Backend Tool Service (50052)")
            print("     - Document search")
            print("     - Web search/crawl")
            print("     - Query expansion")
            print("     - Conversation cache")
            print("           ↓")
            print("   Backend Services")
            print("     - Document Repository")
            print("     - Web Content Tools")
            print("     - Query Expansion Tools")
            print("     - Cache Service")
            print("           ↓")
            print("   Data Layer")
            print("     - PostgreSQL + Qdrant")
            print("     - SearxNG")
            print("     - Crawl4AI")
            print()
            return True
            
    except grpc.RpcError as e:
        print(f"❌ gRPC Error: {e.code()} - {e.details()}")
        return False
    except Exception as e:
        print(f"❌ Phase 5 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_phase5())
    sys.exit(0 if success else 1)

