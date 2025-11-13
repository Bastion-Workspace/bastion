"""
AWS Pricing Models - Roosevelt's "Cost Intelligence" Data Structures
Pydantic models for structured AWS pricing tool responses
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class AWSServiceType(str, Enum):
    """AWS service types for pricing calculations"""
    EC2 = "ec2"
    RDS = "rds"
    S3 = "s3"
    LAMBDA = "lambda"
    EBS = "ebs"
    CLOUDFRONT = "cloudfront"
    ROUTE53 = "route53"
    VPC = "vpc"
    ELB = "elb"
    CLOUDWATCH = "cloudwatch"
    SNS = "sns"
    SQS = "sqs"
    DYNAMODB = "dynamodb"
    ELASTICACHE = "elasticache"
    REDSHIFT = "redshift"
    KINESIS = "kinesis"
    API_GATEWAY = "apigateway"
    COGNITO = "cognito"
    SES = "ses"
    WORKSPACES = "workspaces"


class PricingDimension(BaseModel):
    """Individual pricing dimension (e.g., per hour, per GB, per request)"""
    unit: str = Field(description="Pricing unit (e.g., 'Hrs', 'GB-Mo', 'Requests')")
    price_per_unit: float = Field(description="Price per unit in USD")
    description: str = Field(description="Human-readable description of the pricing dimension")
    currency: str = Field(default="USD", description="Currency code")


class AWSServicePricing(BaseModel):
    """Pricing information for a specific AWS service configuration"""
    service_type: AWSServiceType = Field(description="AWS service type")
    service_name: str = Field(description="Human-readable service name")
    region: str = Field(description="AWS region code (e.g., 'us-east-1')")
    instance_type: Optional[str] = Field(None, description="Instance type if applicable (e.g., 't3.micro')")
    operating_system: Optional[str] = Field(None, description="Operating system if applicable")
    pricing_dimensions: List[PricingDimension] = Field(description="List of pricing dimensions for this service")
    on_demand_pricing: Optional[Dict[str, float]] = Field(None, description="On-demand pricing breakdown")
    reserved_pricing: Optional[Dict[str, float]] = Field(None, description="Reserved instance pricing if available")
    spot_pricing: Optional[Dict[str, float]] = Field(None, description="Spot pricing if available")
    free_tier_eligible: bool = Field(default=False, description="Whether service is eligible for AWS free tier")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="When pricing data was last updated")


class CostEstimate(BaseModel):
    """Cost estimate for a specific usage scenario"""
    service_type: AWSServiceType = Field(description="AWS service being estimated")
    configuration: Dict[str, Any] = Field(description="Service configuration parameters")
    usage_metrics: Dict[str, Union[int, float]] = Field(description="Usage metrics (hours, GB, requests, etc.)")
    estimated_monthly_cost: float = Field(description="Estimated monthly cost in USD")
    estimated_yearly_cost: float = Field(description="Estimated yearly cost in USD")
    cost_breakdown: Dict[str, float] = Field(description="Breakdown of costs by component")
    assumptions: List[str] = Field(description="Assumptions made in the calculation")
    region: str = Field(description="AWS region for the estimate")
    currency: str = Field(default="USD", description="Currency code")


class RegionalPricingComparison(BaseModel):
    """Comparison of pricing across different AWS regions"""
    service_type: AWSServiceType = Field(description="AWS service being compared")
    configuration: Dict[str, Any] = Field(description="Service configuration parameters")
    regional_costs: Dict[str, float] = Field(description="Monthly costs by region")
    cheapest_region: str = Field(description="Region with lowest cost")
    most_expensive_region: str = Field(description="Region with highest cost")
    cost_difference_percent: float = Field(description="Percentage difference between cheapest and most expensive")
    recommendations: List[str] = Field(description="Cost optimization recommendations")


class AWSPricingToolResult(BaseModel):
    """Structured result from AWS pricing tool operations"""
    success: bool = Field(description="Whether the operation was successful")
    operation_type: str = Field(description="Type of pricing operation performed")
    result_data: Optional[Union[AWSServicePricing, CostEstimate, RegionalPricingComparison, List[AWSServicePricing]]] = Field(
        None, description="The pricing data result"
    )
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    data_source: str = Field(description="Source of pricing data (e.g., 'AWS Price List API', 'AWS Pricing Calculator API')")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the operation was performed")
    cache_used: bool = Field(default=False, description="Whether cached data was used")
    api_calls_made: int = Field(default=0, description="Number of API calls made for this operation")


class WorkloadCostEstimate(BaseModel):
    """Cost estimate for a complete workload with multiple services"""
    workload_name: str = Field(description="Name/description of the workload")
    services: List[CostEstimate] = Field(description="Cost estimates for individual services")
    total_monthly_cost: float = Field(description="Total estimated monthly cost across all services")
    total_yearly_cost: float = Field(description="Total estimated yearly cost across all services")
    cost_breakdown_by_service: Dict[str, float] = Field(description="Monthly cost breakdown by service type")
    region: str = Field(description="Primary AWS region for the workload")
    optimization_suggestions: List[str] = Field(description="Cost optimization recommendations")
    confidence_level: str = Field(description="Confidence level of the estimate (High/Medium/Low)")
    assumptions: List[str] = Field(description="Key assumptions made in the calculation")
