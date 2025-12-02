"""
Feature Flags for Gradual Service Rollout
Enables safe, controlled migration to microservices architecture
"""

import os
from enum import Enum
from typing import Dict


class FeatureFlag(str, Enum):
    """Available feature flags"""
    # LLM Orchestrator Migration
    USE_GRPC_ORCHESTRATOR = "use_grpc_orchestrator"
    GRPC_ORCHESTRATOR_PERCENTAGE = "grpc_orchestrator_percentage"
    
    # Tool Service
    USE_GRPC_TOOL_SERVICE = "use_grpc_tool_service"
    
    # Fallback behavior
    GRPC_FALLBACK_TO_LOCAL = "grpc_fallback_to_local"


class FeatureFlagService:
    """
    Manages feature flags for service migration
    
    Phase 1: All flags OFF (new service runs in parallel for testing)
    Phase 2: Enable tool callbacks (orchestrator can call backend)
    Phase 3: Enable orchestrator routing with percentage rollout
    Phase 4: 100% traffic to new service, remove legacy code
    """
    
    def __init__(self):
        self._flags: Dict[str, any] = {
            # Phase 4: Always enabled - gRPC orchestrator is primary path
            FeatureFlag.USE_GRPC_ORCHESTRATOR: True,
            FeatureFlag.GRPC_ORCHESTRATOR_PERCENTAGE: 100,  # 100% traffic to gRPC orchestrator
            
            # Phase 2: Enable tool callbacks
            FeatureFlag.USE_GRPC_TOOL_SERVICE: False,
            
            # Always allow fallback for safety (though should never be needed)
            FeatureFlag.GRPC_FALLBACK_TO_LOCAL: True,
        }
        
        # Override from environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load feature flags from environment variables"""
        for flag in FeatureFlag:
            env_key = flag.value.upper()
            env_value = os.getenv(env_key)
            
            if env_value is not None:
                # Handle boolean flags
                if flag in [
                    FeatureFlag.USE_GRPC_ORCHESTRATOR,
                    FeatureFlag.USE_GRPC_TOOL_SERVICE,
                    FeatureFlag.GRPC_FALLBACK_TO_LOCAL
                ]:
                    self._flags[flag] = env_value.lower() in ('true', '1', 'yes', 'on')
                
                # Handle percentage flags
                elif flag == FeatureFlag.GRPC_ORCHESTRATOR_PERCENTAGE:
                    try:
                        self._flags[flag] = max(0, min(100, int(env_value)))
                    except ValueError:
                        pass
    
    def is_enabled(self, flag: FeatureFlag) -> bool:
        """Check if a feature flag is enabled"""
        return self._flags.get(flag, False)
    
    def get_percentage(self, flag: FeatureFlag) -> int:
        """Get percentage value for gradual rollout flags"""
        return self._flags.get(flag, 0)
    
    def should_use_grpc_orchestrator(self, user_id: str = None) -> bool:
        """
        Determine if request should use gRPC orchestrator
        
        Always returns True - gRPC orchestrator is the primary and only path.
        Fallback backend orchestrator is deprecated.
        """
        # Always use gRPC orchestrator - it's the primary path
        return True


# Global feature flag service instance
feature_flags = FeatureFlagService()


# Convenience functions
def use_grpc_orchestrator(user_id: str = None) -> bool:
    """Should this request use the gRPC orchestrator service?"""
    return feature_flags.should_use_grpc_orchestrator(user_id)


def use_grpc_tool_service() -> bool:
    """Should tool calls use gRPC service?"""
    return feature_flags.is_enabled(FeatureFlag.USE_GRPC_TOOL_SERVICE)


def fallback_to_local_on_error() -> bool:
    """Should we fallback to local service on gRPC error?"""
    return feature_flags.is_enabled(FeatureFlag.GRPC_FALLBACK_TO_LOCAL)

