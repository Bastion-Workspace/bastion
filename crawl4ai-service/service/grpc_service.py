"""
gRPC Service Implementation - Crawl4AI Service
"""

import grpc
import logging

# Import generated proto files (will be generated during Docker build)
import sys
sys.path.insert(0, '/app')

from protos import crawl_service_pb2_grpc

from service.crawl_service import CrawlServiceImplementation
from service.browser_session_service import BrowserSessionServiceImplementation
from config.settings import settings

logger = logging.getLogger(__name__)


class CrawlServiceGRPCImplementation(crawl_service_pb2_grpc.CrawlServiceServicer):
    """Crawl service gRPC implementation"""
    
    def __init__(self):
        self.crawl_service = CrawlServiceImplementation()
        self._initialized = False
    
    async def initialize(self):
        """Initialize all components"""
        try:
            await self.crawl_service.initialize()
            self._initialized = True
            logger.info("Crawl Service gRPC implementation initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Crawl Service: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.crawl_service.cleanup()
    
    async def Crawl(self, request, context):
        """Single URL crawl"""
        try:
            if not self._initialized:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Service not initialized")
                from protos import crawl_service_pb2
                return crawl_service_pb2.CrawlResponse()
            
            return await self.crawl_service.crawl(request)
            
        except Exception as e:
            logger.error(f"Crawl failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            from protos import crawl_service_pb2
            return crawl_service_pb2.CrawlResponse(success=False, error=str(e))
    
    async def CrawlMany(self, request, context):
        """Parallel multi-URL crawl"""
        try:
            if not self._initialized:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Service not initialized")
                from protos import crawl_service_pb2
                return crawl_service_pb2.CrawlManyResponse()
            
            return await self.crawl_service.crawl_many(request)
            
        except Exception as e:
            logger.error(f"CrawlMany failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            from protos import crawl_service_pb2
            return crawl_service_pb2.CrawlManyResponse(success=False, error=str(e))
    
    async def AdaptiveCrawl(self, request, context):
        """Adaptive intelligent crawl"""
        try:
            if not self._initialized:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Service not initialized")
                from protos import crawl_service_pb2
                return crawl_service_pb2.CrawlResponse()
            
            return await self.crawl_service.adaptive_crawl(request)
            
        except Exception as e:
            logger.error(f"AdaptiveCrawl failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            from protos import crawl_service_pb2
            return crawl_service_pb2.CrawlResponse(success=False, error=str(e))
    
    async def HealthCheck(self, request, context):
        """Health check endpoint"""
        try:
            return await self.crawl_service.health_check(request)
        except Exception as e:
            logger.error(f"HealthCheck failed: {e}")
            from protos import crawl_service_pb2
            return crawl_service_pb2.HealthCheckResponse(
                status="unhealthy",
                crawl4ai_available=False,
                browser_available=False,
                service_version="0.7.2",
                details={"error": str(e)}
            )


class BrowserSessionServiceGRPCImplementation(crawl_service_pb2_grpc.BrowserSessionServiceServicer):
    """Browser session gRPC implementation (Playwright)."""

    def __init__(self):
        self.browser_session_service = BrowserSessionServiceImplementation()
        self._initialized = False

    async def initialize(self):
        """Initialize Playwright."""
        try:
            await self.browser_session_service.initialize()
            self._initialized = True
            logger.info("Browser Session Service gRPC implementation initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Browser Session Service: {e}")
            raise

    async def cleanup(self):
        """Cleanup all browser sessions and Playwright."""
        await self.browser_session_service.cleanup()

    async def CreateSession(self, request, context):
        try:
            if not self._initialized:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Service not initialized")
                from protos import crawl_service_pb2
                return crawl_service_pb2.BrowserSessionRef(session_id="")
            return await self.browser_session_service.create_session(request)
        except Exception as e:
            logger.error(f"CreateSession failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            from protos import crawl_service_pb2
            return crawl_service_pb2.BrowserSessionRef(session_id="")

    async def ExecuteAction(self, request, context):
        try:
            if not self._initialized:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Service not initialized")
                from protos import crawl_service_pb2
                return crawl_service_pb2.BrowserActionResult(success=False, error="Service not initialized")
            return await self.browser_session_service.execute_action(request)
        except Exception as e:
            logger.error(f"ExecuteAction failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            from protos import crawl_service_pb2
            return crawl_service_pb2.BrowserActionResult(success=False, error=str(e))

    async def DownloadFile(self, request, context):
        try:
            if not self._initialized:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Service not initialized")
                from protos import crawl_service_pb2
                return crawl_service_pb2.BrowserDownloadResult(success=False, error="Service not initialized")
            return await self.browser_session_service.download_file(request)
        except Exception as e:
            logger.error(f"DownloadFile failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            from protos import crawl_service_pb2
            return crawl_service_pb2.BrowserDownloadResult(success=False, error=str(e))

    async def DestroySession(self, request, context):
        try:
            if not self._initialized:
                from protos import crawl_service_pb2
                return crawl_service_pb2.BrowserEmpty()
            return await self.browser_session_service.destroy_session(request)
        except Exception as e:
            logger.error(f"DestroySession failed: {e}")
            from protos import crawl_service_pb2
            return crawl_service_pb2.BrowserEmpty()

    async def SaveSessionState(self, request, context):
        try:
            if not self._initialized:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Service not initialized")
                from protos import crawl_service_pb2
                return crawl_service_pb2.BrowserSessionStateResult(success=False, error="Service not initialized")
            return await self.browser_session_service.save_session_state(request)
        except Exception as e:
            logger.error(f"SaveSessionState failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            from protos import crawl_service_pb2
            return crawl_service_pb2.BrowserSessionStateResult(success=False, error=str(e))

    async def InspectPage(self, request, context):
        try:
            if not self._initialized:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Service not initialized")
                from protos import crawl_service_pb2
                return crawl_service_pb2.BrowserInspectResult(success=False, error="Service not initialized")
            return await self.browser_session_service.inspect_page(request)
        except Exception as e:
            logger.error(f"InspectPage failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            from protos import crawl_service_pb2
            return crawl_service_pb2.BrowserInspectResult(success=False, error=str(e))



