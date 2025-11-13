"""
AWS Pricing Tools Module - Roosevelt's "Cost Intelligence Cavalry"
AWS pricing calculator integration for LangGraph agents using AWS Price List API and boto3
"""

import logging
import aiohttp
import asyncio
import boto3
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json
from decimal import Decimal

from models.aws_pricing_models import (
    AWSServiceType, AWSServicePricing, CostEstimate, RegionalPricingComparison,
    AWSPricingToolResult, WorkloadCostEstimate, PricingDimension
)

logger = logging.getLogger(__name__)


class AWSPricingTools:
    """AWS pricing tools for LangGraph agents using AWS APIs"""
    
    def __init__(self):
        self.price_list_base_url = "https://pricing.us-east-1.amazonaws.com"
        self.cache = {}  # Simple in-memory cache
        self.cache_duration = timedelta(hours=6)  # Pricing changes less frequently than weather
        self._boto3_client = None
        self._pricing_client = None
        
    async def _get_boto3_client(self):
        """Initialize boto3 pricing client with credentials if available"""
        try:
            from config import settings
            
            if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
                if not self._boto3_client:
                    self._boto3_client = boto3.Session(
                        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                        region_name=settings.AWS_DEFAULT_REGION
                    )
                    self._pricing_client = self._boto3_client.client('pricing', region_name='us-east-1')
                    logger.info("✅ AWS pricing client initialized with credentials")
                return self._pricing_client
            else:
                logger.info("ℹ️ AWS credentials not configured, using public Price List API only")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize AWS pricing client: {e}")
            return None
    
    def get_tools(self) -> Dict[str, Any]:
        """Get all AWS pricing tools"""
        return {
            "estimate_aws_costs": self.estimate_aws_costs,
            "get_aws_service_pricing": self.get_aws_service_pricing,
            "compare_aws_regions": self.compare_aws_regions,
            "estimate_aws_workload": self.estimate_aws_workload,
        }
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all AWS pricing tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "estimate_aws_costs",
                    "description": "Calculate estimated AWS costs for specific service configurations and usage patterns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_type": {
                                "type": "string",
                                "enum": [service.value for service in AWSServiceType],
                                "description": "AWS service type to estimate costs for"
                            },
                            "configuration": {
                                "type": "object",
                                "description": "Service configuration parameters (instance type, storage, etc.)",
                                "additionalProperties": True
                            },
                            "usage_metrics": {
                                "type": "object", 
                                "description": "Usage metrics (hours per month, GB stored, requests per month, etc.)",
                                "additionalProperties": True
                            },
                            "region": {
                                "type": "string",
                                "description": "AWS region code (e.g., 'us-east-1', 'eu-west-1')",
                                "default": "us-east-1"
                            }
                        },
                        "required": ["service_type", "configuration", "usage_metrics"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "get_aws_service_pricing",
                    "description": "Get current AWS pricing information for a specific service and configuration",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_type": {
                                "type": "string",
                                "enum": [service.value for service in AWSServiceType],
                                "description": "AWS service type to get pricing for"
                            },
                            "region": {
                                "type": "string", 
                                "description": "AWS region code (e.g., 'us-east-1', 'eu-west-1')",
                                "default": "us-east-1"
                            },
                            "filters": {
                                "type": "object",
                                "description": "Optional filters for specific configurations (instance type, OS, etc.)",
                                "additionalProperties": True
                            }
                        },
                        "required": ["service_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_aws_regions", 
                    "description": "Compare AWS service costs across different regions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_type": {
                                "type": "string",
                                "enum": [service.value for service in AWSServiceType],
                                "description": "AWS service type to compare across regions"
                            },
                            "configuration": {
                                "type": "object",
                                "description": "Service configuration to compare",
                                "additionalProperties": True
                            },
                            "regions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of AWS regions to compare (e.g., ['us-east-1', 'eu-west-1', 'ap-southeast-1'])",
                                "default": ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
                            },
                            "usage_metrics": {
                                "type": "object",
                                "description": "Monthly usage metrics for cost calculation",
                                "additionalProperties": True
                            }
                        },
                        "required": ["service_type", "configuration"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "estimate_aws_workload",
                    "description": "Estimate total costs for a complete AWS workload with multiple services",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "workload_name": {
                                "type": "string",
                                "description": "Name or description of the workload"
                            },
                            "services": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "service_type": {"type": "string", "enum": [s.value for s in AWSServiceType]},
                                        "configuration": {"type": "object", "additionalProperties": True},
                                        "usage_metrics": {"type": "object", "additionalProperties": True}
                                    },
                                    "required": ["service_type", "configuration", "usage_metrics"]
                                },
                                "description": "List of services in the workload with their configurations"
                            },
                            "region": {
                                "type": "string",
                                "description": "Primary AWS region for the workload",
                                "default": "us-east-1"
                            }
                        },
                        "required": ["workload_name", "services"]
                    }
                }
            }
        ]

    async def estimate_aws_costs(
        self, 
        service_type: str, 
        configuration: Dict[str, Any], 
        usage_metrics: Dict[str, Union[int, float]], 
        region: str = "us-east-1",
        user_id: str = None
    ) -> Dict[str, Any]:
        """Estimate AWS costs for a specific service configuration and usage"""
        try:
            logger.info(f"💰 Estimating AWS costs for {service_type} in {region}")
            
            # Validate service type
            try:
                service_enum = AWSServiceType(service_type)
            except ValueError:
                return self._create_error_result(f"Unsupported service type: {service_type}")
            
            # Check cache first
            cache_key = f"estimate_{service_type}_{region}_{hash(str(configuration))}_{hash(str(usage_metrics))}"
            if self._is_cached(cache_key):
                logger.info(f"🎯 Using cached cost estimate for {service_type}")
                cached_result = self.cache[cache_key]["data"]
                cached_result["cache_used"] = True
                return cached_result
            
            # Get pricing data for the service
            pricing_data = await self._get_service_pricing_data(service_enum, region, configuration)
            if not pricing_data["success"]:
                return pricing_data
            
            # Calculate costs based on usage
            cost_estimate = await self._calculate_service_costs(
                service_enum, configuration, usage_metrics, pricing_data["result_data"], region
            )
            
            # Cache the result
            self._cache_result(cache_key, cost_estimate)
            
            return cost_estimate
            
        except Exception as e:
            logger.error(f"❌ Cost estimation failed: {e}")
            return self._create_error_result(f"Cost estimation error: {str(e)}")

    async def get_aws_service_pricing(
        self,
        service_type: str,
        region: str = "us-east-1", 
        filters: Dict[str, Any] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Get current AWS pricing information for a specific service"""
        try:
            logger.info(f"💲 Getting AWS pricing for {service_type} in {region}")
            
            # Validate service type
            try:
                service_enum = AWSServiceType(service_type)
            except ValueError:
                return self._create_error_result(f"Unsupported service type: {service_type}")
            
            filters = filters or {}
            
            # Check cache first
            cache_key = f"pricing_{service_type}_{region}_{hash(str(filters))}"
            if self._is_cached(cache_key):
                logger.info(f"🎯 Using cached pricing data for {service_type}")
                cached_result = self.cache[cache_key]["data"]
                cached_result["cache_used"] = True
                return cached_result
            
            # Get pricing data
            pricing_result = await self._get_service_pricing_data(service_enum, region, filters)
            
            # Cache the result
            if pricing_result["success"]:
                self._cache_result(cache_key, pricing_result)
            
            return pricing_result
            
        except Exception as e:
            logger.error(f"❌ Service pricing request failed: {e}")
            return self._create_error_result(f"Service pricing error: {str(e)}")

    async def compare_aws_regions(
        self,
        service_type: str,
        configuration: Dict[str, Any],
        regions: List[str] = None,
        usage_metrics: Dict[str, Union[int, float]] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Compare AWS service costs across different regions"""
        try:
            logger.info(f"🌍 Comparing AWS costs for {service_type} across regions")
            
            # Validate service type
            try:
                service_enum = AWSServiceType(service_type)
            except ValueError:
                return self._create_error_result(f"Unsupported service type: {service_type}")
            
            regions = regions or ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
            usage_metrics = usage_metrics or {"hours_per_month": 730}  # Default full month
            
            # Check cache first
            cache_key = f"compare_{service_type}_{hash(str(configuration))}_{hash(str(regions))}"
            if self._is_cached(cache_key):
                logger.info(f"🎯 Using cached regional comparison for {service_type}")
                cached_result = self.cache[cache_key]["data"]
                cached_result["cache_used"] = True
                return cached_result
            
            # Get pricing for each region in parallel
            regional_tasks = []
            for region in regions:
                task = self._get_regional_cost_estimate(service_enum, configuration, usage_metrics, region)
                regional_tasks.append(task)
            
            regional_results = await asyncio.gather(*regional_tasks, return_exceptions=True)
            
            # Process results
            regional_costs = {}
            successful_regions = []
            
            for i, result in enumerate(regional_results):
                region = regions[i]
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ Failed to get pricing for {region}: {result}")
                    continue
                    
                if result and "estimated_monthly_cost" in result:
                    regional_costs[region] = result["estimated_monthly_cost"]
                    successful_regions.append(region)
            
            if not regional_costs:
                return self._create_error_result("Failed to get pricing data for any region")
            
            # Create comparison result
            comparison_result = self._create_regional_comparison(
                service_enum, configuration, regional_costs, successful_regions
            )
            
            # Cache the result
            self._cache_result(cache_key, comparison_result)
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"❌ Regional comparison failed: {e}")
            return self._create_error_result(f"Regional comparison error: {str(e)}")

    async def estimate_aws_workload(
        self,
        workload_name: str,
        services: List[Dict[str, Any]],
        region: str = "us-east-1",
        user_id: str = None
    ) -> Dict[str, Any]:
        """Estimate total costs for a complete AWS workload with multiple services"""
        try:
            logger.info(f"🏗️ Estimating AWS workload costs: {workload_name}")
            
            # Check cache first
            cache_key = f"workload_{hash(workload_name)}_{hash(str(services))}_{region}"
            if self._is_cached(cache_key):
                logger.info(f"🎯 Using cached workload estimate for {workload_name}")
                cached_result = self.cache[cache_key]["data"]
                cached_result["cache_used"] = True
                return cached_result
            
            # Estimate costs for each service in parallel
            service_tasks = []
            for service in services:
                task = self.estimate_aws_costs(
                    service["service_type"],
                    service["configuration"],
                    service["usage_metrics"],
                    region
                )
                service_tasks.append(task)
            
            service_results = await asyncio.gather(*service_tasks, return_exceptions=True)
            
            # Process service estimates
            successful_estimates = []
            total_monthly_cost = 0.0
            cost_breakdown = {}
            
            for i, result in enumerate(service_results):
                service = services[i]
                service_type = service["service_type"]
                
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ Failed to estimate costs for {service_type}: {result}")
                    continue
                
                if result and result.get("success") and "result_data" in result:
                    estimate_data = result["result_data"]
                    successful_estimates.append(estimate_data)
                    monthly_cost = estimate_data.get("estimated_monthly_cost", 0)
                    total_monthly_cost += monthly_cost
                    cost_breakdown[service_type] = monthly_cost
            
            if not successful_estimates:
                return self._create_error_result("Failed to estimate costs for any service in the workload")
            
            # Create workload estimate
            workload_result = self._create_workload_estimate(
                workload_name, successful_estimates, total_monthly_cost, cost_breakdown, region
            )
            
            # Cache the result
            self._cache_result(cache_key, workload_result)
            
            return workload_result
            
        except Exception as e:
            logger.error(f"❌ Workload estimation failed: {e}")
            return self._create_error_result(f"Workload estimation error: {str(e)}")

    async def _get_service_pricing_data(
        self, 
        service_type: AWSServiceType, 
        region: str, 
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get pricing data for a service using available AWS APIs"""
        try:
            # Try authenticated Pricing API first if credentials available
            pricing_client = await self._get_boto3_client()
            if pricing_client:
                return await self._get_pricing_via_boto3(pricing_client, service_type, region, filters)
            
            # Fall back to public Price List API
            return await self._get_pricing_via_public_api(service_type, region, filters)
            
        except Exception as e:
            logger.error(f"❌ Failed to get pricing data: {e}")
            return self._create_error_result(f"Pricing data retrieval error: {str(e)}")

    async def _get_pricing_via_boto3(
        self,
        pricing_client,
        service_type: AWSServiceType,
        region: str,
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get pricing using authenticated boto3 client"""
        try:
            logger.info(f"🔐 Using authenticated AWS Pricing API for {service_type}")
            
            # Map service type to AWS service code
            service_code = self._get_aws_service_code(service_type)
            
            # Build filters for the pricing API
            api_filters = self._build_pricing_filters(service_type, region, filters)
            
            # Get pricing data
            response = pricing_client.get_products(
                ServiceCode=service_code,
                Filters=api_filters,
                MaxResults=100
            )
            
            # Parse and structure the response
            pricing_data = self._parse_boto3_pricing_response(response, service_type, region)
            
            return AWSPricingToolResult(
                success=True,
                operation_type="get_service_pricing",
                result_data=pricing_data,
                data_source="AWS Pricing API (Authenticated)",
                api_calls_made=1
            ).dict()
            
        except Exception as e:
            logger.error(f"❌ Boto3 pricing request failed: {e}")
            return self._create_error_result(f"Authenticated pricing API error: {str(e)}")

    async def _get_pricing_via_public_api(
        self,
        service_type: AWSServiceType,
        region: str,
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get pricing using public Price List API"""
        try:
            logger.info(f"🌐 Using public AWS Price List API for {service_type}")
            
            # Get service pricing URL
            service_code = self._get_aws_service_code(service_type)
            pricing_url = f"{self.price_list_base_url}/offers/v1.0/aws/{service_code}/current/index.json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(pricing_url) as response:
                    if response.status != 200:
                        return self._create_error_result(f"AWS Price List API error: {response.status}")
                    
                    data = await response.json()
            
            # Parse and filter the pricing data
            pricing_data = self._parse_public_api_response(data, service_type, region, filters)
            
            return AWSPricingToolResult(
                success=True,
                operation_type="get_service_pricing",
                result_data=pricing_data,
                data_source="AWS Price List API (Public)",
                api_calls_made=1
            ).dict()
            
        except Exception as e:
            logger.error(f"❌ Public API pricing request failed: {e}")
            return self._create_error_result(f"Public pricing API error: {str(e)}")

    async def _calculate_service_costs(
        self,
        service_type: AWSServiceType,
        configuration: Dict[str, Any],
        usage_metrics: Dict[str, Union[int, float]],
        pricing_data: Union[AWSServicePricing, List[AWSServicePricing]],
        region: str
    ) -> Dict[str, Any]:
        """Calculate costs based on pricing data and usage metrics"""
        try:
            # Handle both single pricing and list of pricing options
            if isinstance(pricing_data, list):
                # Use first matching pricing option
                pricing_data = pricing_data[0] if pricing_data else None
            
            if not pricing_data:
                return self._create_error_result("No pricing data available for cost calculation")
            
            # Calculate monthly costs
            monthly_cost = 0.0
            cost_breakdown = {}
            assumptions = []
            
            # Process each pricing dimension
            for dimension in pricing_data.pricing_dimensions:
                usage_key = self._map_pricing_unit_to_usage(dimension.unit)
                if usage_key in usage_metrics:
                    usage_amount = float(usage_metrics[usage_key])
                    dimension_cost = usage_amount * dimension.price_per_unit
                    monthly_cost += dimension_cost
                    cost_breakdown[dimension.description] = dimension_cost
                    assumptions.append(f"Using {usage_amount} {dimension.unit} at ${dimension.price_per_unit} per {dimension.unit}")
            
            # Create cost estimate
            estimate = CostEstimate(
                service_type=service_type,
                configuration=configuration,
                usage_metrics=usage_metrics,
                estimated_monthly_cost=round(monthly_cost, 2),
                estimated_yearly_cost=round(monthly_cost * 12, 2),
                cost_breakdown=cost_breakdown,
                assumptions=assumptions,
                region=region
            )
            
            return AWSPricingToolResult(
                success=True,
                operation_type="estimate_costs",
                result_data=estimate,
                data_source="Calculated from AWS pricing data"
            ).dict()
            
        except Exception as e:
            logger.error(f"❌ Cost calculation failed: {e}")
            return self._create_error_result(f"Cost calculation error: {str(e)}")

    async def _get_regional_cost_estimate(
        self,
        service_type: AWSServiceType,
        configuration: Dict[str, Any],
        usage_metrics: Dict[str, Union[int, float]],
        region: str
    ) -> Optional[Dict[str, Any]]:
        """Get cost estimate for a specific region"""
        try:
            result = await self.estimate_aws_costs(
                service_type.value, configuration, usage_metrics, region
            )
            
            if result.get("success") and "result_data" in result:
                return result["result_data"]
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Regional estimate failed for {region}: {e}")
            return None

    def _create_regional_comparison(
        self,
        service_type: AWSServiceType,
        configuration: Dict[str, Any],
        regional_costs: Dict[str, float],
        successful_regions: List[str]
    ) -> Dict[str, Any]:
        """Create regional pricing comparison result"""
        try:
            if not regional_costs:
                return self._create_error_result("No regional pricing data available")
            
            # Find cheapest and most expensive regions
            cheapest_region = min(regional_costs, key=regional_costs.get)
            most_expensive_region = max(regional_costs, key=regional_costs.get)
            
            cheapest_cost = regional_costs[cheapest_region]
            most_expensive_cost = regional_costs[most_expensive_region]
            
            # Calculate percentage difference
            cost_difference = 0.0
            if cheapest_cost > 0:
                cost_difference = ((most_expensive_cost - cheapest_cost) / cheapest_cost) * 100
            
            # Generate recommendations
            recommendations = []
            if cost_difference > 10:
                recommendations.append(f"Consider using {cheapest_region} for {cost_difference:.1f}% cost savings")
            if len(successful_regions) < len(regional_costs):
                recommendations.append("Some regions failed to return pricing data - check service availability")
            
            comparison = RegionalPricingComparison(
                service_type=service_type,
                configuration=configuration,
                regional_costs=regional_costs,
                cheapest_region=cheapest_region,
                most_expensive_region=most_expensive_region,
                cost_difference_percent=round(cost_difference, 2),
                recommendations=recommendations
            )
            
            return AWSPricingToolResult(
                success=True,
                operation_type="compare_regions",
                result_data=comparison,
                data_source="AWS pricing data across regions"
            ).dict()
            
        except Exception as e:
            logger.error(f"❌ Regional comparison creation failed: {e}")
            return self._create_error_result(f"Regional comparison error: {str(e)}")

    def _create_workload_estimate(
        self,
        workload_name: str,
        service_estimates: List[Dict[str, Any]],
        total_monthly_cost: float,
        cost_breakdown: Dict[str, float],
        region: str
    ) -> Dict[str, Any]:
        """Create workload cost estimate result"""
        try:
            # Generate optimization suggestions
            optimization_suggestions = []
            if total_monthly_cost > 1000:
                optimization_suggestions.append("Consider Reserved Instances for consistent workloads to save up to 75%")
            if total_monthly_cost > 500:
                optimization_suggestions.append("Review Spot Instance opportunities for fault-tolerant workloads")
            if len(service_estimates) > 3:
                optimization_suggestions.append("Consider AWS Savings Plans for multi-service workloads")
            
            # Determine confidence level
            confidence_level = "High" if len(service_estimates) >= 3 else "Medium" if len(service_estimates) >= 2 else "Low"
            
            # Collect all assumptions
            all_assumptions = []
            for estimate in service_estimates:
                if "assumptions" in estimate:
                    all_assumptions.extend(estimate["assumptions"])
            
            workload = WorkloadCostEstimate(
                workload_name=workload_name,
                services=service_estimates,
                total_monthly_cost=round(total_monthly_cost, 2),
                total_yearly_cost=round(total_monthly_cost * 12, 2),
                cost_breakdown_by_service=cost_breakdown,
                region=region,
                optimization_suggestions=optimization_suggestions,
                confidence_level=confidence_level,
                assumptions=list(set(all_assumptions))  # Remove duplicates
            )
            
            return AWSPricingToolResult(
                success=True,
                operation_type="estimate_workload",
                result_data=workload,
                data_source="Calculated from AWS pricing data"
            ).dict()
            
        except Exception as e:
            logger.error(f"❌ Workload estimate creation failed: {e}")
            return self._create_error_result(f"Workload estimate error: {str(e)}")

    def _get_aws_service_code(self, service_type: AWSServiceType) -> str:
        """Map service type to AWS service code for API calls"""
        service_mapping = {
            AWSServiceType.EC2: "AmazonEC2",
            AWSServiceType.RDS: "AmazonRDS",
            AWSServiceType.S3: "AmazonS3",
            AWSServiceType.LAMBDA: "AWSLambda",
            AWSServiceType.EBS: "AmazonEC2",  # EBS is part of EC2 pricing
            AWSServiceType.CLOUDFRONT: "AmazonCloudFront",
            AWSServiceType.ROUTE53: "AmazonRoute53",
            AWSServiceType.VPC: "AmazonVPC",
            AWSServiceType.ELB: "AWSELB",
            AWSServiceType.CLOUDWATCH: "AmazonCloudWatch",
            AWSServiceType.SNS: "AmazonSNS",
            AWSServiceType.SQS: "AmazonSQS",
            AWSServiceType.DYNAMODB: "AmazonDynamoDB",
            AWSServiceType.ELASTICACHE: "AmazonElastiCache",
            AWSServiceType.REDSHIFT: "AmazonRedshift",
            AWSServiceType.KINESIS: "AmazonKinesis",
            AWSServiceType.API_GATEWAY: "AmazonApiGateway",
            AWSServiceType.COGNITO: "AmazonCognito",
            AWSServiceType.SES: "AmazonSES",
            AWSServiceType.WORKSPACES: "AmazonWorkSpaces"
        }
        return service_mapping.get(service_type, "AmazonEC2")

    def _build_pricing_filters(
        self, 
        service_type: AWSServiceType, 
        region: str, 
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build filters for AWS Pricing API calls"""
        api_filters = [
            {"Type": "TERM_MATCH", "Field": "location", "Value": self._get_region_name(region)},
            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"}
        ]
        
        # Add service-specific filters
        if service_type == AWSServiceType.EC2:
            if "instance_type" in filters:
                api_filters.append({"Type": "TERM_MATCH", "Field": "instanceType", "Value": filters["instance_type"]})
            if "operating_system" in filters:
                api_filters.append({"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": filters["operating_system"]})
        
        return api_filters

    def _get_region_name(self, region_code: str) -> str:
        """Convert region code to AWS pricing region name"""
        region_mapping = {
            "us-east-1": "US East (N. Virginia)",
            "us-west-1": "US West (N. California)", 
            "us-west-2": "US West (Oregon)",
            "eu-west-1": "Europe (Ireland)",
            "eu-central-1": "Europe (Frankfurt)",
            "ap-southeast-1": "Asia Pacific (Singapore)",
            "ap-southeast-2": "Asia Pacific (Sydney)",
            "ap-northeast-1": "Asia Pacific (Tokyo)"
        }
        return region_mapping.get(region_code, region_code)

    def _map_pricing_unit_to_usage(self, pricing_unit: str) -> str:
        """Map AWS pricing units to usage metric keys"""
        unit_mapping = {
            "Hrs": "hours_per_month",
            "Hour": "hours_per_month", 
            "GB-Mo": "gb_per_month",
            "GB": "gb_per_month",
            "Requests": "requests_per_month",
            "1M requests": "million_requests_per_month",
            "IOPS-Mo": "iops_per_month",
            "GB-Month": "gb_per_month"
        }
        return unit_mapping.get(pricing_unit, "usage_amount")

    def _parse_boto3_pricing_response(
        self, 
        response: Dict[str, Any], 
        service_type: AWSServiceType, 
        region: str
    ) -> AWSServicePricing:
        """Parse boto3 pricing API response into structured format"""
        try:
            price_list = response.get("PriceList", [])
            if not price_list:
                raise ValueError("No pricing data returned from AWS API")
            
            # Parse first pricing item (most relevant)
            pricing_item = json.loads(price_list[0])
            product = pricing_item["product"]
            terms = pricing_item["terms"]
            
            # Extract pricing dimensions
            pricing_dimensions = []
            on_demand_pricing = {}
            
            # Process On-Demand pricing
            if "OnDemand" in terms:
                for term_key, term_data in terms["OnDemand"].items():
                    for price_key, price_data in term_data["priceDimensions"].items():
                        dimension = PricingDimension(
                            unit=price_data["unit"],
                            price_per_unit=float(price_data["pricePerUnit"]["USD"]),
                            description=price_data["description"]
                        )
                        pricing_dimensions.append(dimension)
                        on_demand_pricing[price_data["unit"]] = float(price_data["pricePerUnit"]["USD"])
            
            # Create service pricing object
            service_pricing = AWSServicePricing(
                service_type=service_type,
                service_name=product.get("productFamily", service_type.value),
                region=region,
                instance_type=product.get("attributes", {}).get("instanceType"),
                operating_system=product.get("attributes", {}).get("operatingSystem"),
                pricing_dimensions=pricing_dimensions,
                on_demand_pricing=on_demand_pricing,
                free_tier_eligible=product.get("attributes", {}).get("freeTierEligible", "false").lower() == "true"
            )
            
            return service_pricing
            
        except Exception as e:
            logger.error(f"❌ Failed to parse boto3 pricing response: {e}")
            raise

    def _parse_public_api_response(
        self,
        data: Dict[str, Any],
        service_type: AWSServiceType,
        region: str,
        filters: Dict[str, Any]
    ) -> List[AWSServicePricing]:
        """Parse public API pricing response into structured format"""
        try:
            products = data.get("products", {})
            terms = data.get("terms", {})
            
            region_name = self._get_region_name(region)
            matching_services = []
            
            # Filter products by region and configuration
            for product_sku, product_data in products.items():
                attributes = product_data.get("attributes", {})
                
                # Check if this product matches our criteria
                if attributes.get("location") != region_name:
                    continue
                
                # Apply additional filters
                if filters:
                    match = True
                    for filter_key, filter_value in filters.items():
                        attr_key = self._map_filter_to_attribute(filter_key)
                        if attr_key and attributes.get(attr_key) != filter_value:
                            match = False
                            break
                    if not match:
                        continue
                
                # Extract pricing for this product
                pricing_dimensions = []
                on_demand_pricing = {}
                
                # Get On-Demand pricing if available
                on_demand_terms = terms.get("OnDemand", {}).get(product_sku, {})
                for term_key, term_data in on_demand_terms.items():
                    for price_key, price_data in term_data.get("priceDimensions", {}).items():
                        dimension = PricingDimension(
                            unit=price_data["unit"],
                            price_per_unit=float(price_data["pricePerUnit"]["USD"]),
                            description=price_data["description"]
                        )
                        pricing_dimensions.append(dimension)
                        on_demand_pricing[price_data["unit"]] = float(price_data["pricePerUnit"]["USD"])
                
                if pricing_dimensions:  # Only include if we found pricing
                    service_pricing = AWSServicePricing(
                        service_type=service_type,
                        service_name=attributes.get("servicename", service_type.value),
                        region=region,
                        instance_type=attributes.get("instanceType"),
                        operating_system=attributes.get("operatingSystem"),
                        pricing_dimensions=pricing_dimensions,
                        on_demand_pricing=on_demand_pricing,
                        free_tier_eligible=attributes.get("freeTierEligible", "false").lower() == "true"
                    )
                    matching_services.append(service_pricing)
            
            return matching_services[:10]  # Limit to top 10 matches
            
        except Exception as e:
            logger.error(f"❌ Failed to parse public API response: {e}")
            raise

    def _map_filter_to_attribute(self, filter_key: str) -> Optional[str]:
        """Map user filter keys to AWS attribute names"""
        mapping = {
            "instance_type": "instanceType",
            "operating_system": "operatingSystem",
            "storage_type": "storageMedia",
            "database_engine": "databaseEngine"
        }
        return mapping.get(filter_key)

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return AWSPricingToolResult(
            success=False,
            operation_type="error",
            error_message=error_message,
            data_source="Error"
        ).dict()

    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is in cache and still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key]["timestamp"]
        return datetime.utcnow() - cached_time < self.cache_duration
    
    def _cache_result(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Cache result with timestamp"""
        self.cache[cache_key] = {
            "data": data,
            "timestamp": datetime.utcnow()
        }
        
        # Simple cache cleanup - remove old entries
        if len(self.cache) > 50:  # Smaller cache for pricing data
            oldest_keys = sorted(self.cache.keys(), 
                               key=lambda k: self.cache[k]["timestamp"])[:10]
            for key in oldest_keys:
                del self.cache[key]


# Global instance for use by tool registry
_aws_pricing_tools_instance = None


async def _get_aws_pricing_tools():
    """Get global AWS pricing tools instance"""
    global _aws_pricing_tools_instance
    if _aws_pricing_tools_instance is None:
        _aws_pricing_tools_instance = AWSPricingTools()
    return _aws_pricing_tools_instance


# LangGraph tool functions
async def estimate_aws_costs(
    service_type: str,
    configuration: Dict[str, Any],
    usage_metrics: Dict[str, Union[int, float]],
    region: str = "us-east-1",
    user_id: str = None
) -> Dict[str, Any]:
    """LangGraph tool function: Estimate AWS costs for service configuration"""
    tools_instance = await _get_aws_pricing_tools()
    return await tools_instance.estimate_aws_costs(service_type, configuration, usage_metrics, region, user_id)


async def get_aws_service_pricing(
    service_type: str,
    region: str = "us-east-1",
    filters: Dict[str, Any] = None,
    user_id: str = None
) -> Dict[str, Any]:
    """LangGraph tool function: Get AWS service pricing information"""
    tools_instance = await _get_aws_pricing_tools()
    return await tools_instance.get_aws_service_pricing(service_type, region, filters, user_id)


async def compare_aws_regions(
    service_type: str,
    configuration: Dict[str, Any],
    regions: List[str] = None,
    usage_metrics: Dict[str, Union[int, float]] = None,
    user_id: str = None
) -> Dict[str, Any]:
    """LangGraph tool function: Compare AWS costs across regions"""
    tools_instance = await _get_aws_pricing_tools()
    return await tools_instance.compare_aws_regions(service_type, configuration, regions, usage_metrics, user_id)


async def estimate_aws_workload(
    workload_name: str,
    services: List[Dict[str, Any]],
    region: str = "us-east-1",
    user_id: str = None
) -> Dict[str, Any]:
    """LangGraph tool function: Estimate total AWS workload costs"""
    tools_instance = await _get_aws_pricing_tools()
    return await tools_instance.estimate_aws_workload(workload_name, services, region, user_id)
