"""gRPC handlers for Analysis operations (weather, visualization, file analysis, system modeling)."""

import json
import logging

import grpc
from protos import tool_service_pb2

logger = logging.getLogger(__name__)


class AnalysisHandlersMixin:
    """Mixin providing Analysis gRPC handlers.

    Mixed into ToolServiceImplementation; accesses self._get_search_service(),
    self._get_document_repo(), etc. via standard Python MRO.
    """

    # ===== Weather Operations =====
    
    async def GetWeatherData(
        self,
        request: tool_service_pb2.WeatherRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.WeatherResponse:
        """Get weather data (current, forecast, or historical)"""
        aborted = False  # Track if we've already aborted to avoid double-abort
        try:
            logger.info(f"GetWeatherData: location={request.location}, user_id={request.user_id}, date_str={request.date_str if request.HasField('date_str') else None}")
            
            # Normalize location: empty string or whitespace-only → None
            location = request.location.strip() if request.location and request.location.strip() else None
            
            # Determine units from request (default to imperial)
            units = "imperial"  # Default for status bar compatibility
            
            # Check if this is a historical request
            if request.HasField("date_str") and request.date_str:
                # Historical weather request
                # Import from tools-service (same pattern as RSS service)
                from tools_service.services.weather_tools import weather_history
                
                weather_result = await weather_history(
                    location=location,
                    date_str=request.date_str,
                    units=units,
                    user_id=request.user_id if request.user_id else None
                )
                
                if not weather_result.get("success"):
                    error_msg = weather_result.get("error", "Unknown error")
                    logger.warning(f"Historical weather fetch failed: {error_msg}")
                    aborted = True
                    await context.abort(grpc.StatusCode.INTERNAL, f"Historical weather data failed: {error_msg}")
                    return  # This line won't be reached, but included for clarity
                
                # Extract historical weather data
                location_name = weather_result.get("location", {}).get("name", location or "Unknown location")
                historical = weather_result.get("historical", {})
                period = weather_result.get("period", {})
                
                # Format historical data for response
                period_type = period.get("type", "")
                if period_type == "date_range":
                    avg_temp = historical.get("average_temperature", 0)
                    temp_unit = weather_result.get("units", {}).get("temperature", "°F")
                    months_retrieved = period.get("months_retrieved", 0)
                    months_in_range = period.get("months_in_range", 0)
                    current_conditions = f"Range average ({months_retrieved}/{months_in_range} months): {avg_temp:.1f}{temp_unit}"
                elif period_type == "monthly_average":
                    avg_temp = historical.get("average_temperature", 0)
                    temp_unit = weather_result.get("units", {}).get("temperature", "°F")
                    current_conditions = f"Monthly average: {avg_temp:.1f}{temp_unit}"
                else:
                    temp = historical.get("temperature", 0)
                    temp_unit = weather_result.get("units", {}).get("temperature", "°F")
                    conditions = historical.get("conditions", "")
                    current_conditions = f"{temp:.1f}{temp_unit}, {conditions}"
                
                # Build metadata with historical information
                metadata = {
                    "location_name": location_name,
                    "date_str": request.date_str,
                    "period_type": period.get("type", ""),
                    "temperature": str(historical.get("temperature", historical.get("average_temperature", 0))),
                    "conditions": historical.get("conditions", historical.get("most_common_conditions", "")),
                    "humidity": str(historical.get("humidity", historical.get("average_humidity", 0))),
                    "wind_speed": str(historical.get("wind_speed", historical.get("average_wind_speed", 0)))
                }
                
                # Add period-specific fields
                if period_type == "date_range":
                    metadata["average_temperature"] = str(historical.get("average_temperature", 0))
                    metadata["min_temperature"] = str(historical.get("min_temperature", 0))
                    metadata["max_temperature"] = str(historical.get("max_temperature", 0))
                    metadata["months_retrieved"] = str(period.get("months_retrieved", 0))
                    metadata["months_in_range"] = str(period.get("months_in_range", 0))
                    metadata["start_date"] = period.get("start_date", "")
                    metadata["end_date"] = period.get("end_date", "")
                elif period_type == "monthly_average":
                    metadata["average_temperature"] = str(historical.get("average_temperature", 0))
                    metadata["min_temperature"] = str(historical.get("min_temperature", 0))
                    metadata["max_temperature"] = str(historical.get("max_temperature", 0))
                    metadata["sample_days"] = str(historical.get("sample_days", 0))
                
                response = tool_service_pb2.WeatherResponse(
                    location=location_name,
                    current_conditions=current_conditions,
                    metadata=metadata
                )
                
                logger.info(f"✅ Historical weather data retrieved for {location_name} on {request.date_str}")
                return response
            
            # Check if forecast is requested
            data_types = list(request.data_types) if request.data_types else ["current"]
            is_forecast_request = "forecast" in data_types
            
            # Import from tools-service (same pattern as RSS service)
            from tools_service.services.weather_tools import weather_forecast, weather_conditions
            
            if is_forecast_request:
                # Forecast request
                
                # Default to 3 days if not specified
                days = 3
                weather_result = await weather_forecast(
                    location=location,
                    days=days,
                    units=units,
                    user_id=request.user_id if request.user_id else None
                )
                
                if not weather_result.get("success"):
                    error_msg = weather_result.get("error", "Unknown error")
                    logger.warning(f"Forecast fetch failed: {error_msg}")
                    aborted = True
                    await context.abort(grpc.StatusCode.INTERNAL, f"Weather forecast failed: {error_msg}")
                    return
                
                # Extract forecast data
                location_name = weather_result.get("location", {}).get("name", location or "Unknown location")
                forecast = weather_result.get("forecast", [])
                
                # Format forecast days for response
                forecast_strings = []
                for day in forecast[:days]:
                    high = day.get("temperature", {}).get("high", 0)
                    low = day.get("temperature", {}).get("low", 0)
                    conditions = day.get("conditions", "")
                    forecast_strings.append(f"{day.get('day_name', 'Day')}: {high}°F/{low}°F, {conditions}")
                
                # Build metadata with forecast information
                metadata = {
                    "location_name": location_name,
                    "forecast_days": str(days),
                    "forecast_data": json.dumps(forecast[:days]) if forecast else "[]"
                }
                
                # Format current conditions string (use first day of forecast)
                if forecast:
                    first_day = forecast[0]
                    high = first_day.get("temperature", {}).get("high", 0)
                    low = first_day.get("temperature", {}).get("low", 0)
                    conditions = first_day.get("conditions", "")
                    current_conditions = f"Forecast: {high}°F/{low}°F, {conditions}"
                else:
                    current_conditions = "Forecast unavailable"
                
                response = tool_service_pb2.WeatherResponse(
                    location=location_name,
                    current_conditions=current_conditions,
                    forecast=forecast_strings,
                    metadata=metadata
                )
                
                logger.info(f"✅ Weather forecast retrieved for {location_name}: {days} days")
                return response
            else:
                # Default to current conditions
                weather_result = await weather_conditions(
                    location=location,
                    units=units,
                    user_id=request.user_id if request.user_id else None
                )
                
                if not weather_result.get("success"):
                    error_msg = weather_result.get("error", "Unknown error")
                    logger.warning(f"Weather fetch failed: {error_msg}")
                    aborted = True
                    await context.abort(grpc.StatusCode.INTERNAL, f"Weather data failed: {error_msg}")
                    return
                
                # Extract weather data
                location_name = weather_result.get("location", {}).get("name", location or "Unknown location")
                current = weather_result.get("current", {})
                temperature = int(current.get("temperature", 0))
                conditions = current.get("conditions", "")
                moon_phase = weather_result.get("moon_phase", {})
                
                # Build metadata dict with all weather information
                metadata = {
                    "location_name": location_name,
                    "temperature": str(temperature),
                    "conditions": conditions,
                    "moon_phase_name": moon_phase.get("phase_name", ""),
                    "moon_phase_icon": moon_phase.get("phase_icon", ""),
                    "moon_phase_value": str(moon_phase.get("phase_value", 0)),
                    "humidity": str(current.get("humidity", 0)),
                    "wind_speed": str(current.get("wind_speed", 0)),
                    "feels_like": str(current.get("feels_like", 0))
                }
                
                # Format current conditions string
                current_conditions = f"{temperature}°F, {conditions}"
                
                # Build response
                response = tool_service_pb2.WeatherResponse(
                    location=location_name,
                    current_conditions=current_conditions,
                    metadata=metadata
                )
                
                logger.info(f"✅ Weather data retrieved for {location_name}: {temperature}°F, {conditions}")
                return response
            
        except grpc.RpcError:
            # Already aborted - don't abort again
            raise
        except Exception as e:
            logger.error(f"GetWeatherData error: {e}")
            # Only abort if we haven't already aborted
            if not aborted:
                await context.abort(grpc.StatusCode.INTERNAL, f"Weather data failed: {str(e)}")
    

    # ===== Visualization Operations =====
    
    async def CreateChart(
        self,
        request: tool_service_pb2.CreateChartRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.CreateChartResponse:
        """Create a chart or graph from structured data"""
        try:
            logger.info(f"CreateChart: chart_type={request.chart_type}, title={request.title}")

            from tools_service.services.analysis_ops import create_chart_operation

            result = await create_chart_operation(
                chart_type=request.chart_type,
                data_json=request.data_json,
                title=request.title,
                x_label=request.x_label,
                y_label=request.y_label,
                interactive=request.interactive,
                color_scheme=request.color_scheme if request.color_scheme else "plotly",
                width=request.width if request.width > 0 else 800,
                height=request.height if request.height > 0 else 600,
                include_static=request.include_static,
            )

            if not result.get("success"):
                err = result.get("error", "Unknown error creating chart")
                if err.startswith("Invalid JSON"):
                    logger.error("CreateChart: %s", err)
                return tool_service_pb2.CreateChartResponse(success=False, error=err)

            metadata_json = json.dumps(result.get("metadata", {}))
            response = tool_service_pb2.CreateChartResponse(
                success=True,
                chart_type=result.get("chart_type", request.chart_type),
                output_format=result.get("output_format", "html"),
                chart_data=result.get("chart_data", ""),
                metadata_json=metadata_json,
            )
            if result.get("static_svg"):
                response.static_svg = result["static_svg"]
            if result.get("static_format"):
                response.static_format = result["static_format"]
            return response

        except Exception as e:
            logger.error(f"CreateChart error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return tool_service_pb2.CreateChartResponse(
                success=False,
                error=f"Chart creation failed: {str(e)}"
            )
    
    # ===== File Analysis Operations =====
    
    async def AnalyzeTextContent(
        self,
        request: tool_service_pb2.TextAnalysisRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.TextAnalysisResponse:
        """Analyze text content and return metrics"""
        try:
            logger.info(f"AnalyzeTextContent: user={request.user_id}, include_advanced={request.include_advanced}")

            from tools_service.services.analysis_ops import analyze_text_operation

            out = analyze_text_operation(
                content=request.content,
                include_advanced=request.include_advanced,
            )
            metrics = out["metrics"]

            response = tool_service_pb2.TextAnalysisResponse(
                word_count=metrics.get("word_count", 0),
                line_count=metrics.get("line_count", 0),
                non_empty_line_count=metrics.get("non_empty_line_count", 0),
                character_count=metrics.get("character_count", 0),
                character_count_no_spaces=metrics.get("character_count_no_spaces", 0),
                paragraph_count=metrics.get("paragraph_count", 0),
                sentence_count=metrics.get("sentence_count", 0),
            )
            if request.include_advanced:
                response.avg_words_per_sentence = metrics.get("avg_words_per_sentence", 0.0)
                response.avg_words_per_paragraph = metrics.get("avg_words_per_paragraph", 0.0)
            response.metadata_json = out["metadata_json"]

            logger.debug(f"AnalyzeTextContent: Analyzed {metrics.get('word_count', 0)} words")
            return response

        except Exception as e:
            logger.error(f"AnalyzeTextContent error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return response with zero values on error
            return tool_service_pb2.TextAnalysisResponse(
                word_count=0,
                line_count=0,
                non_empty_line_count=0,
                character_count=0,
                character_count_no_spaces=0,
                paragraph_count=0,
                sentence_count=0,
                avg_words_per_sentence=0.0,
                avg_words_per_paragraph=0.0,
                metadata_json=json.dumps({"error": str(e)})
            )
    
    # ===== System Modeling Operations =====
    
    async def DesignSystemComponent(
        self,
        request: tool_service_pb2.DesignSystemComponentRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DesignSystemComponentResponse:
        """Design/add a system component to the topology"""
        try:
            logger.info(f"DesignSystemComponent: user={request.user_id}, component={request.component_id}")

            from tools_service.services.analysis_ops import design_system_component_operation

            result = design_system_component_operation(
                user_id=request.user_id,
                component_id=request.component_id,
                component_type=request.component_type,
                requires=list(request.requires),
                provides=list(request.provides),
                redundancy_group=request.redundancy_group if request.HasField("redundancy_group") else None,
                criticality=request.criticality,
                metadata=dict(request.metadata),
                dependency_logic=request.dependency_logic if request.dependency_logic else "AND",
                m_of_n_threshold=request.m_of_n_threshold,
                dependency_weights=dict(request.dependency_weights),
                integrity_threshold=request.integrity_threshold if request.integrity_threshold > 0 else 0.5,
            )

            response = tool_service_pb2.DesignSystemComponentResponse(
                success=result["success"],
                component_id=result["component_id"],
                message=result["message"],
                topology_json=result["topology_json"]
            )
            
            if not result["success"] and "error" in result:
                response.error = result["error"]
            
            return response
            
        except Exception as e:
            logger.error(f"DesignSystemComponent failed: {e}")
            return tool_service_pb2.DesignSystemComponentResponse(
                success=False,
                component_id=request.component_id,
                message=f"Failed to design component: {str(e)}",
                error=str(e),
                topology_json="{}"
            )
    
    async def SimulateSystemFailure(
        self,
        request: tool_service_pb2.SimulateSystemFailureRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.SimulateSystemFailureResponse:
        """Simulate system failure with cascade propagation"""
        try:
            logger.info(f"SimulateSystemFailure: user={request.user_id}, components={request.failed_component_ids}")

            from tools_service.services.analysis_ops import simulate_system_failure_operation

            result = simulate_system_failure_operation(
                user_id=request.user_id,
                failed_component_ids=list(request.failed_component_ids),
                failure_modes=list(request.failure_modes),
                simulation_type=request.simulation_type if request.HasField("simulation_type") else "cascade",
                monte_carlo_iterations=request.monte_carlo_iterations if request.HasField("monte_carlo_iterations") else None,
                failure_parameters=dict(request.failure_parameters),
            )
            
            if not result["success"]:
                return tool_service_pb2.SimulateSystemFailureResponse(
                    success=False,
                    simulation_id=result.get("simulation_id", ""),
                    error=result.get("error", "Unknown error"),
                    topology_json=result.get("topology_json", "{}")
                )
            
            # Build component states
            component_states = []
            for state in result["component_states"]:
                comp_state = tool_service_pb2.ComponentState(
                    component_id=state["component_id"],
                    state=state["state"],
                    failed_dependencies=state.get("failed_dependencies", []),
                    failure_probability=state.get("failure_probability", 0.0),
                    metadata=state.get("metadata", {})
                )
                component_states.append(comp_state)
            
            # Build failure paths
            failure_paths = []
            for path in result["failure_paths"]:
                failure_path = tool_service_pb2.FailurePath(
                    source_component_id=path["source_component_id"],
                    affected_component_ids=path["affected_component_ids"],
                    failure_type=path["failure_type"],
                    path_length=path["path_length"]
                )
                failure_paths.append(failure_path)
            
            # Build health metrics
            health = result["health_metrics"]
            health_metrics = tool_service_pb2.SystemHealthMetrics(
                total_components=health["total_components"],
                operational_components=health["operational_components"],
                degraded_components=health["degraded_components"],
                failed_components=health["failed_components"],
                system_health_score=health["system_health_score"],
                critical_vulnerabilities=health["critical_vulnerabilities"],
                redundancy_groups_at_risk=health["redundancy_groups_at_risk"]
            )
            
            return tool_service_pb2.SimulateSystemFailureResponse(
                success=True,
                simulation_id=result["simulation_id"],
                component_states=component_states,
                failure_paths=failure_paths,
                health_metrics=health_metrics,
                topology_json=result["topology_json"]
            )
            
        except Exception as e:
            logger.error(f"SimulateSystemFailure failed: {e}")
            from tools_service.services.analysis_ops import new_simulation_id

            return tool_service_pb2.SimulateSystemFailureResponse(
                success=False,
                simulation_id=new_simulation_id(),
                error=str(e),
                topology_json="{}"
            )
    
    async def GetSystemTopology(
        self,
        request: tool_service_pb2.GetSystemTopologyRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.GetSystemTopologyResponse:
        """Get system topology as JSON"""
        try:
            logger.info(f"GetSystemTopology: user={request.user_id}")

            from tools_service.services.analysis_ops import get_system_topology_operation

            result = get_system_topology_operation(
                user_id=request.user_id,
                system_name=request.system_name if request.HasField("system_name") else None,
            )
            
            response = tool_service_pb2.GetSystemTopologyResponse(
                success=result["success"],
                topology_json=result["topology_json"],
                component_count=result["component_count"],
                edge_count=result["edge_count"],
                redundancy_groups=result["redundancy_groups"]
            )
            
            if not result["success"] and "error" in result:
                response.error = result["error"]
            
            return response
            
        except Exception as e:
            logger.error(f"GetSystemTopology failed: {e}")
            return tool_service_pb2.GetSystemTopologyResponse(
                success=False,
                error=str(e),
                topology_json="{}",
                component_count=0,
                edge_count=0
            )
    
