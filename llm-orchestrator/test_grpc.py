#!/usr/bin/env python3
"""
Simple gRPC test script for LLM Orchestrator Service
Tests basic connectivity and streaming
"""

import asyncio
import sys

import grpc
from protos import orchestrator_pb2, orchestrator_pb2_grpc


async def test_health_check():
    """Test health check endpoint"""
    try:
        async with grpc.aio.insecure_channel('localhost:50051') as channel:
            stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
            
            request = orchestrator_pb2.HealthCheckRequest()
            response = await stub.HealthCheck(request)
            
            print(f"✅ Health Check: {response.status}")
            print(f"   Details: {dict(response.details)}")
            return True
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


async def test_streaming_chat():
    """Test streaming chat endpoint"""
    try:
        async with grpc.aio.insecure_channel('localhost:50051') as channel:
            stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
            
            request = orchestrator_pb2.ChatRequest(
                user_id="test-user",
                conversation_id="test-conv-123",
                query="Hello from Phase 1 test",
                session_id="test-session"
            )
            
            print("\n📡 Testing streaming chat...")
            async for chunk in stub.StreamChat(request):
                print(f"   [{chunk.type}] {chunk.agent_name}: {chunk.message}")
            
            print("✅ Streaming chat completed successfully")
            return True
            
    except Exception as e:
        print(f"❌ Streaming chat failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🧪 Testing LLM Orchestrator gRPC Service (Phase 1)\n")
    
    health_ok = await test_health_check()
    if not health_ok:
        print("\n❌ Basic connectivity failed. Is the service running?")
        print("   Try: docker compose ps llm-orchestrator")
        sys.exit(1)
    
    stream_ok = await test_streaming_chat()
    
    if health_ok and stream_ok:
        print("\n✅ All Phase 1 tests passed!")
        print("   Service is ready for Phase 2 development")
    else:
        print("\n⚠️  Some tests failed")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())

