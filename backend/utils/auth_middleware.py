"""
Authentication Middleware - Protects endpoints and provides user context
"""

import os
import logging
from typing import Optional, Callable, Dict, Any
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

from services.auth_service import auth_service
from models.api_models import AuthenticatedUserResponse

logger = logging.getLogger(__name__)

# Development bypass setting
DEVELOPMENT_BYPASS_AUTH = os.getenv("DEVELOPMENT_BYPASS_AUTH", "false").lower() == "true"

# JWT Bearer token security scheme
security = HTTPBearer(auto_error=False)


def decode_jwt_token(token: str) -> Dict[str, Any]:
    """Decode and verify JWT token"""
    try:
        import jwt
        from config import settings
        
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired")
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        raise ValueError("Invalid token")
    except Exception as e:
        logger.error(f"JWT token decode error: {e}")
        raise ValueError("Token decode failed")


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware to handle authentication for protected routes"""
    
    def __init__(self, app, protected_paths: list = None):
        super().__init__(app)
        # Paths that require authentication
        self.protected_paths = protected_paths or [
            "/api/documents",
            "/api/query", 
            "/api/chat",
            "/api/settings",
            "/api/users",
            "/api/notes",
            "/api/graph",
            "/api/agents",
            "/api/models",
            "/api/admin",
            "/api/calibre",
            "/api/v2",
            "/api/resilient-embedding",
            "/api/research-plans",
            "/api/conversations",
            "/api/folders",
            "/api/categories",
            "/api/ocr",
            "/api/search",
            "/api/segmentation",
            "/api/pdf-text",
            "/api/migration"
        ]
        
        # Paths that don't require authentication
        self.public_paths = [
            "/health",
            "/api/auth/login",
            "/api/auth/logout",
            "/docs",
            "/openapi.json",
            "/api/files",  # Static file serving
            "/api/health",  # Health check endpoints
            "/api/models/available",  # Public model list
            "/api/models/current"  # Current model info
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process request with authentication check"""
        path = request.url.path
        
        # Skip authentication for public paths
        if any(path.startswith(public_path) for public_path in self.public_paths):
            response = await call_next(request)
            return response
        
        # Check if path requires authentication
        requires_auth = any(path.startswith(protected_path) for protected_path in self.protected_paths)
        
        if requires_auth:
            # Extract token from Authorization header
            authorization = request.headers.get("Authorization")
            if not authorization or not authorization.startswith("Bearer "):
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Authentication required"}
                )
            
            token = authorization.split(" ")[1]
            
            # Verify token and get user
            try:
                # Check if auth service is initialized
                if not hasattr(auth_service, 'db_pool') or auth_service.db_pool is None:
                    logger.error("❌ Auth service not initialized - db_pool is None")
                    from fastapi.responses import JSONResponse
                    return JSONResponse(
                        status_code=500,
                        content={"detail": "Authentication service not ready"}
                    )
                
                logger.info(f"🔍 Validating token for path: {path}")
                logger.info(f"🔍 Token received: {token[:50]}...")
                current_user = await auth_service.get_current_user(token)
                if not current_user:
                    logger.warning(f"❌ Invalid or expired token for path: {path}")
                    logger.info(f"🔍 Token validation failed - token: {token[:50]}...")
                    from fastapi.responses import JSONResponse
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or expired token"}
                    )
                
                # Add user to request state
                request.state.current_user = current_user
                logger.debug(f"✅ Authentication successful for user: {current_user.username} on path: {path}")
                
            except Exception as e:
                logger.error(f"❌ Authentication error for path {path}: {e}")
                import traceback
                logger.error(f"❌ Full traceback: {traceback.format_exc()}")
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Authentication error"}
                )
        
        response = await call_next(request)
        return response


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> AuthenticatedUserResponse:
    """Dependency to get current authenticated user"""
    
    # Development bypass - return a mock admin user
    if DEVELOPMENT_BYPASS_AUTH:
        logger.info("🔓 Development auth bypass active - using mock admin user")
        return AuthenticatedUserResponse(
            user_id="dev-admin-001",
            username="admin",
            email="admin@localhost",
            role="admin",
            is_active=True,
            created_at="2024-01-01T00:00:00Z",
            last_login="2024-01-01T00:00:00Z"
        )
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    current_user = await auth_service.get_current_user(credentials.credentials)
    if not current_user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return current_user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[AuthenticatedUserResponse]:
    """Dependency to get current user (optional, returns None if not authenticated)"""
    
    # Development bypass - return a mock admin user
    if DEVELOPMENT_BYPASS_AUTH:
        return AuthenticatedUserResponse(
            user_id="dev-admin-001",
            username="admin",
            email="admin@localhost",
            role="admin",
            is_active=True,
            created_at="2024-01-01T00:00:00Z",
            last_login="2024-01-01T00:00:00Z"
        )
    
    if not credentials:
        return None
    
    return await auth_service.get_current_user(credentials.credentials)


def require_role(required_role: str) -> Callable:
    """Decorator to require specific user role"""
    def role_checker(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
        if current_user.role != required_role and current_user.role != "admin":
            raise HTTPException(
                status_code=403, 
                detail=f"Insufficient permissions. {required_role} role required."
            )
        return current_user
    
    return role_checker


def require_admin():
    """Dependency to require admin role"""
    return require_role("admin")


async def get_current_user_id(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> str:
    """Dependency to get current user ID"""
    return current_user.user_id
