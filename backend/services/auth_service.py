"""
Authentication Service - Handles user authentication, sessions, and JWT tokens
"""

import asyncio
import hashlib
import json
import secrets
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List

import asyncpg
import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
import redis.asyncio as redis

from config import settings
from models.api_models import (
    LoginRequest, LoginResponse, UserCreateRequest, UserUpdateRequest, 
    PasswordChangeRequest, UserResponse, UsersListResponse, AuthenticatedUserResponse
)

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Global auth service instance
auth_service = None

class AuthenticationService:
    """Service for handling user authentication and session management"""
    
    def __init__(self):
        self.db_pool = None
        self.redis_client = None
        self.cache_ttl = 300  # 5 minutes cache TTL
        self._initialized = False
        # Emergency in-memory cache for session timing issues
        self._emergency_session_cache = {}  # token_hash -> (user_data, expiry_time)
    
    async def initialize(self, shared_db_pool=None):
        """Initialize the authentication service"""
        try:
            logger.info("🔧 Initializing authentication service...")
            
            # Use shared database connection pool if provided, otherwise create own
            if shared_db_pool:
                self.db_pool = shared_db_pool
                logger.info("✅ Authentication service using shared database pool")
            else:
                # Create database connection pool
                self.db_pool = await asyncpg.create_pool(
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    database=settings.POSTGRES_DB,
                    min_size=settings.DB_POOL_MIN_SIZE,
                    max_size=settings.DB_POOL_MAX_SIZE,
                    command_timeout=settings.DB_POOL_COMMAND_TIMEOUT,
                    max_queries=settings.DB_POOL_MAX_QUERIES,
                    max_inactive_connection_lifetime=settings.DB_POOL_MAX_INACTIVE_TIME
                )
                logger.info("✅ Authentication database pool established")
            
            # Test database connection and check if users table exists
            async with self.db_pool.acquire() as conn:
                # Check if users table exists
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'users'
                    )
                """)
                
                if table_exists:
                    logger.info("✅ Users table exists")
                else:
                    logger.error("❌ Users table does not exist!")
                    raise Exception("Users table not found - database may not be properly initialized")
            
            # Initialize Redis for caching
            if settings.REDIS_URL:
                self.redis_client = redis.from_url(settings.REDIS_URL)
                await self.redis_client.ping()
                logger.info("✅ Authentication Redis connection established")
            else:
                logger.warning("⚠️ No Redis URL configured - authentication caching disabled")
                self.redis_client = None
            
            # Create default admin user
            await self._create_default_admin_user()
            
            self._initialized = True
            logger.info("✅ Authentication service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize authentication service: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            raise
    
    async def close(self):
        """Close the authentication service"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("✅ Authentication Redis connection closed")
        
        if self.db_pool:
            await self.db_pool.close()
            logger.info("✅ Authentication database pool closed")
        
        self._initialized = False
    
    def _hash_password(self, password: str) -> Tuple[str, str]:
        """Hash a password using bcrypt; store a UUID salt for legacy schema only.

        NOTE: Bcrypt already manages salting internally; concatenating a separate
        salt causes 72-byte limit issues. We therefore hash the raw password
        directly and store a UUID in the salt column for compatibility without
        using it in hashing.
        """
        salt = str(uuid.uuid4())
        password_hash = pwd_context.hash(password)
        return password_hash, salt
    
    def _verify_password(self, password: str, salt: str, password_hash: str) -> bool:
        """Verify a password against its hash.

        The stored salt is ignored for bcrypt because bcrypt embeds its salt in
        the hash. We keep the column for schema compatibility.
        """
        return pwd_context.verify(password, password_hash)
    
    def _generate_jwt_token(self, user_data: Dict[str, Any]) -> Tuple[str, datetime]:
        """Generate JWT token for user"""
        now = datetime.utcnow()
        expiration = now + timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)
        
        # Convert to Unix timestamps for JWT standard compliance
        payload = {
            "user_id": user_data["user_id"],
            "username": user_data["username"],
            "role": user_data["role"],
            "exp": int(expiration.timestamp()),
            "iat": int(now.timestamp())
        }
        
        token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
        return token, expiration
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    async def _create_default_admin_user(self):
        """Create default admin user if it doesn't exist"""
        try:
            logger.info(f"🔧 ROOSEVELT'S ADMIN CREATION: Attempting to create default admin user")
            logger.info(f"🔧 Username from settings: {settings.ADMIN_USERNAME}")
            logger.info(f"🔧 Password length from settings: {len(settings.ADMIN_PASSWORD)} chars")
            logger.info(f"🔧 Email from settings: {settings.ADMIN_EMAIL}")
            
            async with self.db_pool.acquire() as conn:
                # Check if admin user already exists
                existing_admin = await conn.fetchrow(
                    "SELECT user_id FROM users WHERE username = $1",
                    settings.ADMIN_USERNAME
                )
                
                if existing_admin:
                    logger.info("✅ Default admin user already exists")
                    return
                
                logger.info(f"🔧 Creating new admin user with username: {settings.ADMIN_USERNAME}")
                
                # Create admin user
                user_id = str(uuid.uuid4())
                password_hash, salt = self._hash_password(settings.ADMIN_PASSWORD)
                
                await conn.execute("""
                    INSERT INTO users (
                        user_id, username, email, password_hash, salt, 
                        role, display_name, is_active, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, 
                    user_id, settings.ADMIN_USERNAME, settings.ADMIN_EMAIL,
                    password_hash, salt, "admin", "Administrator", True, datetime.utcnow()
                )
                
                logger.info(f"✅ Admin user created successfully: {settings.ADMIN_USERNAME}")
                
                # Create default folders for the admin user
                await self._create_default_folders_for_user(conn, user_id)
                
                logger.info(f"✅ Created default admin user: {settings.ADMIN_USERNAME}")
                
        except Exception as e:
            logger.error(f"❌ Failed to create default admin user: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
    
    async def _create_default_folders_for_user(self, conn, user_id: str):
        """Create default conversation folders for a user"""
        try:
            # Set user context for RLS policies (use false for session-level, not transaction-level)
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
            await conn.execute("SELECT set_config('app.current_user_role', 'admin', false)")
            
            # Verify the context was set
            check_user_id = await conn.fetchval("SELECT current_setting('app.current_user_id', true)")
            check_role = await conn.fetchval("SELECT current_setting('app.current_user_role', true)")
            logger.info(f"✅ RLS context verified: user_id={check_user_id}, role={check_role}")
            
            default_folders = [
                ('folder-general', 'General', 'General conversations and queries', '#2196F3', 1),
                ('folder-research', 'Research', 'Research-related conversations', '#4CAF50', 2),
                ('folder-analysis', 'Analysis', 'Document analysis and insights', '#FF9800', 3),
                ('folder-archived', 'Archived', 'Archived conversations', '#9E9E9E', 999)
            ]
            
            for folder_id, name, description, color, sort_order in default_folders:
                await conn.execute("""
                    INSERT INTO conversation_folders (folder_id, user_id, name, description, color, sort_order) 
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (folder_id) DO NOTHING
                """, f"{folder_id}-{user_id[:8]}", user_id, name, description, color, sort_order)
            
            logger.info(f"✅ Created default folders for user {user_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create default folders for user {user_id}: {e}")
            raise
    
    async def authenticate_user(self, login_request: LoginRequest) -> Optional[LoginResponse]:
        """Authenticate user with username and password"""
        try:
            async with self.db_pool.acquire() as conn:
                # Start explicit transaction
                async with conn.transaction():
                    logger.info(f"🔍 Transaction started")
                    
                    # Get user by username
                    user_row = await conn.fetchrow("""
                        SELECT user_id, username, email, password_hash, salt, role, 
                               display_name, is_active, failed_login_attempts, preferences
                        FROM users 
                        WHERE username = $1
                    """, login_request.username)
                
                    if not user_row:
                        logger.warning(f"Login attempt for non-existent user: {login_request.username}")
                        return None
                    
                    # Check if user is active
                    if not user_row["is_active"]:
                        logger.warning(f"Login attempt for inactive user: {login_request.username}")
                        return None
                    
                    # Check if account is locked due to failed attempts
                    if user_row["failed_login_attempts"] >= settings.MAX_FAILED_LOGINS:
                        logger.warning(f"Login attempt for locked account: {login_request.username}")
                        return None
                    
                    # Verify password
                    if not self._verify_password(
                        login_request.password, 
                        user_row["salt"], 
                        user_row["password_hash"]
                    ):
                        # Increment failed login attempts
                        await conn.execute("""
                            UPDATE users 
                            SET failed_login_attempts = failed_login_attempts + 1,
                                last_failed_login = $1
                            WHERE user_id = $2
                        """, datetime.utcnow(), user_row["user_id"])
                        
                        logger.warning(f"Failed login attempt for user: {login_request.username}")
                        return None
                    
                    # Reset failed login attempts and update last login
                    await conn.execute("""
                        UPDATE users 
                        SET failed_login_attempts = 0, last_login = $1
                        WHERE user_id = $2
                    """, datetime.utcnow(), user_row["user_id"])
                    
                    # Generate JWT token
                    user_data = {
                        "user_id": user_row["user_id"],
                        "username": user_row["username"],
                        "email": user_row["email"],
                        "role": user_row["role"],
                        "display_name": user_row["display_name"]
                    }
                    
                    token, expiration = self._generate_jwt_token(user_data)
                    
                    # Store session
                    session_id = str(uuid.uuid4())
                    token_hash = hashlib.sha256(token.encode()).hexdigest()
                    
                    # Set user context for RLS policies
                    await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_row["user_id"])
                    await conn.execute("SELECT set_config('app.current_user_role', $1, false)", user_row["role"])
                    logger.info(f"🔍 Set user context: {user_row['user_id']} with role {user_row['role']}")
                    
                    logger.info(f"🔍 Creating session for user {user_row['user_id']}")
                    logger.info(f"🔍 Token hash: {token_hash[:20]}...")
                    logger.info(f"🔍 Session expires: {expiration}")
                    
                    try:
                        await conn.execute("""
                            INSERT INTO user_sessions (
                                session_id, user_id, token_hash, expires_at
                            ) VALUES ($1, $2, $3, $4)
                        """, session_id, user_row["user_id"], token_hash, expiration)
                        logger.info(f"✅ Session INSERT completed successfully")
                    except Exception as e:
                        logger.error(f"❌ Session INSERT failed: {e}")
                        raise
                    
                    logger.info(f"✅ User authenticated successfully: {login_request.username}")
                    
                    logger.info(f"🔍 About to return LoginResponse")
                    response = LoginResponse(
                        access_token=token,
                        user=user_data,
                        expires_in=settings.JWT_EXPIRATION_MINUTES * 60
                    )
                    logger.info(f"🔍 LoginResponse created, transaction about to commit")
                    
                    # Let's verify the session was actually inserted within the transaction
                    session_check = await conn.fetchrow("""
                        SELECT session_id FROM user_sessions
                        WHERE token_hash = $1 AND user_id = $2
                    """, token_hash, user_row["user_id"])
                    if session_check:
                        logger.info(f"✅ Session verified within transaction: {session_check['session_id'][:8]}...")
                    else:
                        logger.error(f"❌ Session NOT found within transaction!")
                    
                    logger.info(f"🔍 Returning LoginResponse")
                    logger.info(f"🔍 Transaction will commit when exiting context manager")
                    
                # Transaction is now committed, let's add a longer delay to ensure it's fully propagated
                await asyncio.sleep(0.1)  # 100ms delay to ensure transaction commits across connection pool
                
                # IMMEDIATE CACHE: Cache the user session immediately to avoid database lookup timing issues
                try:
                    cache_key = f"auth:user:{token_hash}"
                    cached_user = AuthenticatedUserResponse(
                        user_id=user_row["user_id"],
                        username=user_row["username"],
                        email=user_row["email"],
                        role=user_row["role"],
                        display_name=user_row["display_name"],
                        preferences=user_row["preferences"]
                    )
                    
                    # Cache in Redis if available
                    if self.redis_client:
                        await self.redis_client.setex(
                            cache_key, 
                            self.cache_ttl, 
                            cached_user.model_dump_json()
                        )
                        logger.info(f"✅ IMMEDIATE CACHE (Redis): User session cached for token: {token_hash[:20]}...")
                    
                    # EMERGENCY FALLBACK: Also cache in memory for immediate access
                    expiry_time = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)
                    self._emergency_session_cache[token_hash] = (cached_user, expiry_time)
                    logger.info(f"✅ EMERGENCY CACHE: User session cached in memory for token: {token_hash[:20]}...")
                    
                except Exception as cache_e:
                    logger.warning(f"⚠️ Failed to immediately cache user session: {cache_e}")
                
                logger.info(f"🔍 Authentication completed, session should be available")
                return response
                
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return None
    
    async def _get_cached_user(self, token_hash: str) -> Optional[AuthenticatedUserResponse]:
        """Get user from cache"""
        
        # EMERGENCY CACHE CHECK FIRST: Check in-memory cache for immediate post-login access
        if token_hash in self._emergency_session_cache:
            cached_user, expiry_time = self._emergency_session_cache[token_hash]
            if datetime.utcnow() < expiry_time:
                logger.info(f"✅ EMERGENCY CACHE HIT: Retrieved user from emergency cache for token: {token_hash[:20]}...")
                return cached_user
            else:
                # Expired, remove from cache
                del self._emergency_session_cache[token_hash]
                logger.info(f"🧹 EMERGENCY CACHE CLEANUP: Removed expired session from emergency cache")
        
        # Regular Redis cache check
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(f"auth:user:{token_hash}")
            if cached_data:
                user_data = json.loads(cached_data)
                return AuthenticatedUserResponse(**user_data)
            return None
        except Exception as e:
            logger.debug(f"Cache read failed: {e}")
            return None
    
    async def _cache_user(self, token_hash: str, user: AuthenticatedUserResponse):
        """Cache user data"""
        if not self.redis_client:
            return
        
        try:
            user_data = user.dict()
            await self.redis_client.setex(
                f"auth:user:{token_hash}",
                self.cache_ttl,
                json.dumps(user_data)
            )
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")
    
    async def _invalidate_user_cache(self, token_hash: str):
        """Invalidate user cache"""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.delete(f"auth:user:{token_hash}")
        except Exception as e:
            logger.debug(f"Cache invalidation failed: {e}")

    async def refresh_token(self, token: str) -> Optional[LoginResponse]:
        """Refresh JWT token if it's still valid or recently expired"""
        try:
            if not self._initialized:
                logger.error("❌ Auth service not initialized")
                return None
            
            # Decode token without verification to check expiration
            try:
                import jwt
                payload = jwt.decode(
                    token, 
                    settings.JWT_SECRET_KEY, 
                    algorithms=[settings.JWT_ALGORITHM],
                    options={"verify_exp": False}  # Don't verify expiration yet
                )
            except jwt.InvalidTokenError as e:
                logger.warning(f"Invalid token for refresh: {e}")
                return None
            
            # Check if token is expired or will expire soon (within 5 minutes)
            exp_timestamp = payload.get('exp')
            if exp_timestamp:
                current_timestamp = datetime.utcnow().timestamp()
                time_until_expiry = exp_timestamp - current_timestamp
                
                # Allow refresh if token is still valid or expired within last 5 minutes
                if time_until_expiry < -300:  # Expired more than 5 minutes ago
                    logger.warning("Token expired too long ago for refresh")
                    return None
            
            # Get user from database
            async with self.db_pool.acquire() as conn:
                user_row = await conn.fetchrow("""
                    SELECT user_id, username, email, role, display_name, preferences, is_active
                    FROM users 
                    WHERE user_id = $1 AND is_active = true
                """, payload["user_id"])
                
                if not user_row:
                    logger.warning("User not found or inactive for token refresh")
                    return None
                
                # Generate new token
                user_data = {
                    "user_id": user_row["user_id"],
                    "username": user_row["username"],
                    "email": user_row["email"],
                    "role": user_row["role"],
                    "display_name": user_row["display_name"]
                }
                
                new_token, expiration = self._generate_jwt_token(user_data)
                
                # Update session in database
                new_token_hash = hashlib.sha256(new_token.encode()).hexdigest()
                old_token_hash = hashlib.sha256(token.encode()).hexdigest()
                
                # Set user context for RLS policies
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_row["user_id"])
                await conn.execute("SELECT set_config('app.current_user_role', $1, false)", user_row["role"])
                
                # Update existing session or create new one
                await conn.execute("""
                    UPDATE user_sessions 
                    SET token_hash = $1, expires_at = $2, last_accessed = $3
                    WHERE token_hash = $4
                """, new_token_hash, expiration, datetime.utcnow(), old_token_hash)
                
                # If no session was updated, create a new one
                if await conn.fetchval("SELECT COUNT(*) FROM user_sessions WHERE token_hash = $1", new_token_hash) == 0:
                    session_id = str(uuid.uuid4())
                    await conn.execute("""
                        INSERT INTO user_sessions (
                            session_id, user_id, token_hash, expires_at
                        ) VALUES ($1, $2, $3, $4)
                    """, session_id, user_row["user_id"], new_token_hash, expiration)
                
                # Invalidate old cache
                await self._invalidate_user_cache(old_token_hash)
                
                # Handle preferences properly
                preferences = user_row["preferences"]
                if isinstance(preferences, str):
                    try:
                        preferences = json.loads(preferences)
                    except (json.JSONDecodeError, TypeError):
                        preferences = {}
                elif preferences is None:
                    preferences = {}
                
                # Cache new session
                try:
                    cache_key = f"auth:user:{new_token_hash}"
                    cached_user = AuthenticatedUserResponse(
                        user_id=user_row["user_id"],
                        username=user_row["username"],
                        email=user_row["email"],
                        role=user_row["role"],
                        display_name=user_row["display_name"],
                        preferences=preferences
                    )
                    
                    if self.redis_client:
                        await self.redis_client.setex(
                            cache_key,
                            self.cache_ttl,
                            cached_user.model_dump_json()
                        )
                    
                    expiry_time = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)
                    self._emergency_session_cache[new_token_hash] = (cached_user, expiry_time)
                except Exception as cache_e:
                    logger.warning(f"Failed to cache refreshed session: {cache_e}")
                
                logger.info(f"✅ Token refreshed successfully for user: {user_row['username']}")
                
                return LoginResponse(
                    access_token=new_token,
                    user=user_data,
                    expires_in=settings.JWT_EXPIRATION_MINUTES * 60
                )
                
        except Exception as e:
            logger.error(f"❌ Token refresh failed: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return None
    
    async def get_current_user(self, token: str) -> Optional[AuthenticatedUserResponse]:
        """Get current user from JWT token with caching"""
        try:
            if not self._initialized:
                logger.error("❌ Auth service not initialized")
                return None
                
            logger.debug(f"🔍 Verifying JWT token: {token[:50]}...")
            payload = self.verify_jwt_token(token)
            if not payload:
                logger.warning("❌ JWT token verification failed")
                return None
            
            logger.debug(f"✅ JWT token verified for user: {payload.get('username', 'unknown')}")
            
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            
            # Try cache first
            cached_user = await self._get_cached_user(token_hash)
            if cached_user:
                logger.debug(f"✅ User cache hit for {cached_user.user_id}")
                return cached_user
            
            # Cache miss - query database (no RLS context needed for auth queries)
            async with self.db_pool.acquire() as conn:
                # Verify session exists and is valid
                logger.info(f"🔍 Checking session for token hash: {token_hash[:20]}...")
                session_row = await conn.fetchrow("""
                    SELECT s.user_id, s.expires_at 
                    FROM user_sessions s
                    WHERE s.token_hash = $1 AND s.expires_at > $2
                """, token_hash, datetime.utcnow())
                
                if not session_row:
                    logger.warning("⚠️ Database session not found or expired - checking JWT token validity")
                    
                    # Check if JWT token is still valid (not expired)
                    # exp is a Unix timestamp (number) from JWT payload
                    exp_timestamp = payload.get('exp')
                    if exp_timestamp and isinstance(exp_timestamp, (int, float)):
                        current_timestamp = datetime.utcnow().timestamp()
                        if current_timestamp < exp_timestamp:
                            logger.info("✅ JWT token is still valid - proceeding with JWT-only authentication")
                            
                            # Get user details directly from JWT payload
                            user_row = await conn.fetchrow("""
                                SELECT user_id, username, email, role, display_name, preferences, is_active
                                FROM users 
                                WHERE user_id = $1 AND is_active = true
                            """, payload["user_id"])
                            
                            if not user_row:
                                logger.warning("❌ User not found or inactive")
                                return None
                            
                            # Handle preferences properly
                            preferences = user_row["preferences"]
                            if isinstance(preferences, str):
                                try:
                                    preferences = json.loads(preferences)
                                except (json.JSONDecodeError, TypeError):
                                    preferences = {}
                            elif preferences is None:
                                preferences = {}
                            
                            user = AuthenticatedUserResponse(
                                user_id=user_row["user_id"],
                                username=user_row["username"],
                                email=user_row["email"],
                                role=user_row["role"],
                                display_name=user_row["display_name"],
                                preferences=preferences
                            )
                            
                            # Cache the user data
                            await self._cache_user(token_hash, user)
                            logger.info(f"✅ JWT-only authentication successful for user: {user.username}")
                            
                            return user
                        else:
                            logger.warning("❌ JWT token has expired")
                            return None
                    else:
                        logger.warning("❌ JWT token missing or invalid exp claim")
                        return None
                
                logger.info(f"✅ Session found for user: {session_row['user_id']}")
                
                # Get user details
                user_row = await conn.fetchrow("""
                    SELECT user_id, username, email, role, display_name, preferences, is_active
                    FROM users 
                    WHERE user_id = $1 AND is_active = true
                """, payload["user_id"])
                
                if not user_row:
                    return None
                
                # Handle preferences properly - asyncpg returns JSONB as dict already
                preferences = user_row["preferences"]
                if isinstance(preferences, str):
                    try:
                        preferences = json.loads(preferences)
                    except (json.JSONDecodeError, TypeError):
                        preferences = {}
                elif preferences is None:
                    preferences = {}
                
                user = AuthenticatedUserResponse(
                    user_id=user_row["user_id"],
                    username=user_row["username"],
                    email=user_row["email"],
                    role=user_row["role"],
                    display_name=user_row["display_name"],
                    preferences=preferences
                )
                
                # Cache the user data
                await self._cache_user(token_hash, user)
                logger.debug(f"✅ User cached for {user.user_id}")
                
                return user
                
        except Exception as e:
            logger.error(f"❌ Get current user failed: {e}")
            return None
    
    async def logout_user(self, token: str) -> bool:
        """Logout user by invalidating session and cache"""
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            
            # Invalidate cache first
            await self._invalidate_user_cache(token_hash)
            
            async with self.db_pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM user_sessions WHERE token_hash = $1",
                    token_hash
                )
                
                return result != "DELETE 0"
                
        except Exception as e:
            logger.error(f"❌ Logout failed: {e}")
            return False
    
    async def create_user(self, user_request: UserCreateRequest) -> Optional[UserResponse]:
        """Create a new user"""
        try:
            async with self.db_pool.acquire() as conn:
                # Check if username or email already exists
                existing_user = await conn.fetchrow("""
                    SELECT user_id FROM users 
                    WHERE username = $1 OR email = $2
                """, user_request.username, user_request.email)
                
                if existing_user:
                    logger.warning(f"User creation failed - username/email already exists: {user_request.username}")
                    return None
                
                # Hash password
                password_hash, salt = self._hash_password(user_request.password)
                
                # Create user
                user_id = str(uuid.uuid4())
                now = datetime.utcnow()
                
                await conn.execute("""
                    INSERT INTO users (
                        user_id, username, email, password_hash, salt, 
                        role, display_name, is_active, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, 
                    user_id, user_request.username, user_request.email,
                    password_hash, salt, user_request.role, 
                    user_request.display_name or user_request.username, 
                    True, now
                )
                
                logger.info(f"✅ User created successfully: {user_request.username}")
                
                # Post-create provisioning: folders and org files
                try:
                    from services.folder_service import FolderService
                    from services.org_files_service import ensure_user_org_files
                    from services.user_settings_kv_service import set_user_setting
                    folder_service = FolderService()
                    await folder_service.initialize()
                    folders = await folder_service.create_default_folders(user_id)
                    # Seed filesystem Org files
                    org_info = await ensure_user_org_files(user_id)
                    # Persist pointers in user_settings
                    # Save ORG_ROOT_FOLDER_ID if available
                    org_folder = next((f for f in folders if f.name == "Org"), None)
                    if org_folder:
                        await set_user_setting(user_id, "ORG_ROOT_FOLDER_ID", org_folder.folder_id)
                    await set_user_setting(user_id, "ORG_INBOX_PATH", org_info.get("inbox_path", ""))
                    await set_user_setting(user_id, "ORG_ARCHIVE_PATH", org_info.get("archive_path", ""))
                except Exception as pe:
                    logger.error(f"❌ Post-create provisioning failed for {user_id}: {pe}")

                return UserResponse(
                    user_id=user_id,
                    username=user_request.username,
                    email=user_request.email,
                    display_name=user_request.display_name or user_request.username,
                    avatar_url=None,
                    role=user_request.role,
                    is_active=True,
                    created_at=now
                )
                
        except Exception as e:
            logger.error(f"❌ User creation failed: {e}")
            return None
    
    async def get_users(self, skip: int = 0, limit: int = 100) -> UsersListResponse:
        """Get list of users"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get total count
                total = await conn.fetchval("SELECT COUNT(*) FROM users")
                
                # Get users
                rows = await conn.fetch("""
                    SELECT user_id, username, email, role, display_name, avatar_url,
                           is_active, created_at, last_login
                    FROM users 
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                """, limit, skip)
                
                users = [
                    UserResponse(
                        user_id=row["user_id"],
                        username=row["username"],
                        email=row["email"],
                        display_name=row["display_name"],
                        avatar_url=row.get("avatar_url"),
                        role=row["role"],
                        is_active=row["is_active"],
                        created_at=row["created_at"],
                        last_login=row["last_login"]
                    )
                    for row in rows
                ]
                
                return UsersListResponse(users=users, total=total)
                
        except Exception as e:
            logger.error(f"❌ Get users failed: {e}")
            return UsersListResponse(users=[], total=0)
    
    async def update_user(self, user_id: str, update_request: UserUpdateRequest) -> Optional[UserResponse]:
        """Update user details"""
        try:
            async with self.db_pool.acquire() as conn:
                # Build update query dynamically
                update_fields = []
                values = []
                param_count = 1
                
                if update_request.email is not None:
                    update_fields.append(f"email = ${param_count}")
                    values.append(update_request.email)
                    param_count += 1
                
                if update_request.display_name is not None:
                    update_fields.append(f"display_name = ${param_count}")
                    values.append(update_request.display_name)
                    param_count += 1
                
                if update_request.avatar_url is not None:
                    update_fields.append(f"avatar_url = ${param_count}")
                    values.append(update_request.avatar_url)
                    param_count += 1
                
                if update_request.role is not None:
                    update_fields.append(f"role = ${param_count}")
                    values.append(update_request.role)
                    param_count += 1
                
                if update_request.is_active is not None:
                    update_fields.append(f"is_active = ${param_count}")
                    values.append(update_request.is_active)
                    param_count += 1
                
                if not update_fields:
                    return None
                
                update_fields.append(f"updated_at = ${param_count}")
                values.append(datetime.utcnow())
                param_count += 1
                
                values.append(user_id)  # For WHERE clause
                
                query = f"""
                    UPDATE users 
                    SET {', '.join(update_fields)}
                    WHERE user_id = ${param_count}
                    RETURNING user_id, username, email, role, display_name, avatar_url,
                              is_active, created_at, last_login
                """
                
                row = await conn.fetchrow(query, *values)
                
                if not row:
                    return None
                
                updated_user = UserResponse(
                    user_id=row["user_id"],
                    username=row["username"],
                    email=row["email"],
                    display_name=row["display_name"],
                    avatar_url=row.get("avatar_url"),
                    role=row["role"],
                    is_active=row["is_active"],
                    created_at=row["created_at"],
                    last_login=row["last_login"]
                )
                
                # Invalidate any cached sessions for this user
                if self.redis_client:
                    try:
                        # Get all cached sessions for this user and invalidate them
                        pattern = f"auth:user:*"
                        keys = await self.redis_client.keys(pattern)
                        for key in keys:
                            cached_data = await self.redis_client.get(key)
                            if cached_data:
                                user_data = json.loads(cached_data)
                                if user_data.get("user_id") == user_id:
                                    await self.redis_client.delete(key)
                                    logger.debug(f"Invalidated cache for updated user {user_id}")
                    except Exception as e:
                        logger.debug(f"Cache invalidation failed during user update: {e}")
                
                return updated_user
                
        except Exception as e:
            logger.error(f"❌ User update failed: {e}")
            return None
    
    async def change_password(self, user_id: str, password_request: PasswordChangeRequest) -> bool:
        """Change user password"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get current user data
                user_row = await conn.fetchrow("""
                    SELECT password_hash, salt FROM users WHERE user_id = $1
                """, user_id)
                
                if not user_row:
                    return False
                
                # Verify current password
                if not self._verify_password(
                    password_request.current_password, 
                    user_row["salt"], 
                    user_row["password_hash"]
                ):
                    return False
                
                # Hash new password
                new_password_hash, new_salt = self._hash_password(password_request.new_password)
                
                # Update password
                await conn.execute("""
                    UPDATE users 
                    SET password_hash = $1, salt = $2, updated_at = $3
                    WHERE user_id = $4
                """, new_password_hash, new_salt, datetime.utcnow(), user_id)
                
                # Invalidate all existing sessions for this user
                await conn.execute(
                    "DELETE FROM user_sessions WHERE user_id = $1",
                    user_id
                )
                
                logger.info(f"✅ Password changed for user: {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Password change failed: {e}")
            return False
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM users WHERE user_id = $1",
                    user_id
                )
                
                if result != "DELETE 0":
                    # Invalidate all cached sessions for this user
                    if self.redis_client:
                        try:
                            pattern = f"auth:user:*"
                            keys = await self.redis_client.keys(pattern)
                            for key in keys:
                                cached_data = await self.redis_client.get(key)
                                if cached_data:
                                    user_data = json.loads(cached_data)
                                    if user_data.get("user_id") == user_id:
                                        await self.redis_client.delete(key)
                                        logger.debug(f"Invalidated cache for deleted user {user_id}")
                        except Exception as e:
                            logger.debug(f"Cache invalidation failed during user deletion: {e}")
                    
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"❌ User deletion failed: {e}")
            return False
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM user_sessions WHERE expires_at < $1",
                    datetime.utcnow()
                )
                
                logger.info(f"🧹 Cleaned up expired sessions: {result}")
                
        except Exception as e:
            logger.error(f"❌ Session cleanup failed: {e}")
    
    async def send_email_verification(self, user_id: str, base_url: str = None) -> Dict[str, Any]:
        """
        Generate and send email verification token
        
        Args:
            user_id: User ID to send verification for
            base_url: Base URL for verification link (optional, defaults to empty)
            
        Returns:
            Dict with success status and message
        """
        try:
            from services.email_service import email_service
            
            async with self.db_pool.acquire() as conn:
                # Get user info
                user = await conn.fetchrow("""
                    SELECT user_id, username, email, email_verified
                    FROM users
                    WHERE user_id = $1
                """, user_id)
                
                if not user:
                    return {"success": False, "message": "User not found"}
                
                if user["email_verified"]:
                    return {"success": False, "message": "Email already verified"}
                
                # Generate verification token
                verification_token = secrets.token_urlsafe(32)
                expires_at = datetime.utcnow() + timedelta(hours=settings.EMAIL_VERIFICATION_EXPIRY_HOURS)
                
                # Store token in database
                await conn.execute("""
                    UPDATE users
                    SET email_verification_token = $1,
                        email_verification_sent_at = $2,
                        email_verification_expires_at = $3
                    WHERE user_id = $4
                """, verification_token, datetime.utcnow(), expires_at, user_id)
                
                # Create verification URL
                if base_url:
                    verification_url = f"{base_url}/verify-email?token={verification_token}"
                else:
                    verification_url = f"/verify-email?token={verification_token}"
                
                # Send verification email
                email_sent = await email_service.send_verification_email(
                    to_email=user["email"],
                    username=user["username"],
                    verification_token=verification_token,
                    verification_url=verification_url
                )
                
                if email_sent:
                    logger.info(f"✅ Verification email sent to {user['email']}")
                    return {
                        "success": True,
                        "message": "Verification email sent",
                        "email": user["email"]
                    }
                else:
                    return {
                        "success": False,
                        "message": "Failed to send verification email"
                    }
                    
        except Exception as e:
            logger.error(f"❌ Failed to send email verification: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    async def verify_email_token(self, token: str) -> Dict[str, Any]:
        """
        Verify email token and update user status
        
        Args:
            token: Verification token from email
            
        Returns:
            Dict with success status and user info
        """
        try:
            async with self.db_pool.acquire() as conn:
                # Find user with this token
                user = await conn.fetchrow("""
                    SELECT user_id, username, email, email_verified, email_verification_expires_at
                    FROM users
                    WHERE email_verification_token = $1
                """, token)
                
                if not user:
                    return {"success": False, "message": "Invalid verification token"}
                
                if user["email_verified"]:
                    return {
                        "success": True,
                        "message": "Email already verified",
                        "user_id": user["user_id"],
                        "email": user["email"]
                    }
                
                # Check if token expired
                if user["email_verification_expires_at"]:
                    if datetime.utcnow() > user["email_verification_expires_at"]:
                        return {"success": False, "message": "Verification token has expired"}
                
                # Verify email
                await conn.execute("""
                    UPDATE users
                    SET email_verified = TRUE,
                        email_verification_token = NULL,
                        updated_at = $1
                    WHERE user_id = $2
                """, datetime.utcnow(), user["user_id"])
                
                logger.info(f"✅ Email verified for user {user['user_id']}")
                
                return {
                    "success": True,
                    "message": "Email verified successfully",
                    "user_id": user["user_id"],
                    "email": user["email"]
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to verify email token: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    async def resend_verification_email(self, user_id: str, base_url: str = None) -> Dict[str, Any]:
        """
        Resend email verification email
        
        Args:
            user_id: User ID to resend verification for
            base_url: Base URL for verification link (optional)
            
        Returns:
            Dict with success status and message
        """
        return await self.send_email_verification(user_id, base_url)
    
    async def get_email_verification_status(self, user_id: str) -> Dict[str, Any]:
        """
        Get email verification status for a user
        
        Args:
            user_id: User ID to check
            
        Returns:
            Dict with verification status and info
        """
        try:
            async with self.db_pool.acquire() as conn:
                user = await conn.fetchrow("""
                    SELECT email, email_verified, email_verification_sent_at, email_verification_expires_at
                    FROM users
                    WHERE user_id = $1
                """, user_id)
                
                if not user:
                    return {"success": False, "message": "User not found"}
                
                return {
                    "success": True,
                    "email": user["email"],
                    "email_verified": user["email_verified"],
                    "verification_sent_at": user["email_verification_sent_at"],
                    "verification_expires_at": user["email_verification_expires_at"]
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get email verification status: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}


# Global service instance
auth_service = AuthenticationService()


# Dependency function for FastAPI
async def get_current_user(token: str) -> Optional[AuthenticatedUserResponse]:
    """FastAPI dependency function to get current user from token"""
    if not auth_service._initialized:
        logger.warning("⚠️ Auth service not initialized, attempting to initialize...")
        await auth_service.initialize()
    return await auth_service.get_current_user(token)
