"""
Security Analysis Models
Pydantic models for security analysis requests and findings
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SecurityFinding(BaseModel):
    category: str
    severity: SeverityLevel
    title: str
    description: str
    url: Optional[str] = None
    evidence: Optional[str] = None
    remediation: str


class SecurityAnalysisRequest(BaseModel):
    target_url: str
    user_id: str = "system"
    scan_depth: str = "comprehensive"


class SecurityAnalysisResponse(BaseModel):
    success: bool
    target_url: str
    scan_timestamp: str
    findings: List[SecurityFinding]
    technology_stack: Dict[str, str]
    security_headers: Dict[str, Any]
    risk_score: float
    summary: str
    disclaimer: str
    error: Optional[str] = None
