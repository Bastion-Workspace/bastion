"""
Victoria plugin - VictoriaMetrics and VictoriaLogs integration for Agent Factory (Zone 4).

Provides typed tools for PromQL/MetricsQL queries, LogsQL search, log facets, and observability.
Connection config: victoriametrics_url, victorialogs_url, auth_token (optional).
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.plugins.base_plugin import BasePlugin, PluginToolSpec


# ── Shared output models (reused across tools) ─────────────────────────────

class MetricSample(BaseModel):
    """Single metric sample (instant query result)."""
    metric_name: str = Field(description="Metric name")
    labels: Dict[str, str] = Field(default_factory=dict, description="Label set")
    value: float = Field(description="Sample value")
    timestamp: str = Field(description="Sample timestamp ISO 8601")


class MetricSeries(BaseModel):
    """Time series (range query result)."""
    metric_name: str = Field(description="Metric name")
    labels: Dict[str, str] = Field(default_factory=dict, description="Label set")
    values: List[List[Any]] = Field(default_factory=list, description="[[timestamp, value], ...]")


class LogEntry(BaseModel):
    """Single log entry from VictoriaLogs."""
    timestamp: str = Field(description="Log time ISO 8601")
    message: str = Field(default="", description="Log message")
    stream: str = Field(default="", description="Stream identifier")
    fields: Dict[str, Any] = Field(default_factory=dict, description="Additional fields")


class HitsBucket(BaseModel):
    """Log volume bucket from /select/logsql/hits."""
    group_fields: Dict[str, str] = Field(default_factory=dict, description="Group-by field values")
    timestamps: List[str] = Field(default_factory=list, description="Bucket timestamps")
    values: List[int] = Field(default_factory=list, description="Hit counts per bucket")
    total: int = Field(default=0, description="Total hits in this group")


class FacetValue(BaseModel):
    """Single value in a facet with hit count."""
    value: str = Field(description="Field value")
    hits: int = Field(description="Number of matching logs")


class FacetField(BaseModel):
    """Top field values for one field (facets)."""
    field_name: str = Field(description="Field name")
    values: List[FacetValue] = Field(default_factory=list, description="Top values with hit counts")


# ── VictoriaMetrics: Input/Output models ───────────────────────────────────

class VictoriaQueryMetricsInputs(BaseModel):
    """Inputs for instant PromQL/MetricsQL query."""
    query: str = Field(description="PromQL/MetricsQL expression")
    time: str = Field(default="", description="Optional eval time (ISO 8601 or relative e.g. -5m)")


class VictoriaQueryMetricsOutputs(BaseModel):
    """Outputs for victoria_query_metrics."""
    results: List[MetricSample] = Field(description="Instant query results")
    count: int = Field(description="Number of results")
    result_type: str = Field(description="vector, scalar, or string")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class VictoriaQueryRangeInputs(BaseModel):
    """Inputs for range PromQL/MetricsQL query."""
    query: str = Field(description="PromQL/MetricsQL expression")
    start: str = Field(default="-1h", description="Start time (relative or ISO 8601)")
    end: str = Field(default="now", description="End time (relative or ISO 8601)")
    step: str = Field(default="1m", description="Query resolution step")


class VictoriaQueryRangeOutputs(BaseModel):
    """Outputs for victoria_query_range."""
    series: List[MetricSeries] = Field(description="Time series results")
    count: int = Field(description="Number of series")
    result_type: str = Field(description="matrix")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class VictoriaListSeriesInputs(BaseModel):
    """Inputs for listing matching time series."""
    match: str = Field(description="Series selector e.g. {job=\"api\"}")
    limit: int = Field(default=50, description="Max series to return")


class VictoriaListSeriesOutputs(BaseModel):
    """Outputs for victoria_list_series."""
    series: List[Dict[str, Any]] = Field(description="Label sets of matching series")
    count: int = Field(description="Number of series")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class VictoriaListLabelsInputs(BaseModel):
    """Inputs for listing label names or values."""
    label_name: str = Field(default="", description="If empty list all label names; else list values for this label")
    match: str = Field(default="", description="Optional series selector filter")


class VictoriaListLabelsOutputs(BaseModel):
    """Outputs for victoria_list_labels."""
    values: List[str] = Field(description="Label names or values")
    count: int = Field(description="Number of values")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class VictoriaTsdbStatusInputs(BaseModel):
    """No required inputs for victoria_tsdb_status."""

    pass


class VictoriaTsdbStatusOutputs(BaseModel):
    """Outputs for victoria_tsdb_status (no required inputs)."""
    total_series: int = Field(description="Total time series")
    total_label_value_pairs: int = Field(description="Total label value pairs")
    series_count_by_metric: List[Dict[str, Any]] = Field(default_factory=list, description="Top metrics by cardinality")
    series_count_by_label_value_pair: List[Dict[str, Any]] = Field(default_factory=list, description="Top label pairs")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


# ── VictoriaLogs: Input/Output models ─────────────────────────────────────

class VictoriaQueryLogsInputs(BaseModel):
    """Inputs for LogsQL query."""
    query: str = Field(description="LogsQL expression")
    limit: int = Field(default=100, description="Max log entries to return")
    start: str = Field(default="-1h", description="Start time (relative or ISO 8601)")
    end: str = Field(default="now", description="End time (relative or ISO 8601)")


class VictoriaQueryLogsOutputs(BaseModel):
    """Outputs for victoria_query_logs."""
    logs: List[LogEntry] = Field(description="Matching log entries")
    count: int = Field(description="Number of logs")
    query_used: str = Field(description="LogsQL query executed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class VictoriaLogHitsInputs(BaseModel):
    """Inputs for log volume histogram."""
    query: str = Field(description="LogsQL expression")
    start: str = Field(default="-3h", description="Start time")
    end: str = Field(default="now", description="End time")
    step: str = Field(default="15m", description="Bucket step")
    field: str = Field(default="", description="Optional group-by field e.g. level")


class VictoriaLogHitsOutputs(BaseModel):
    """Outputs for victoria_log_hits."""
    hits: List[HitsBucket] = Field(description="Hit buckets")
    total: int = Field(description="Total hits")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class VictoriaLogStatsInputs(BaseModel):
    """Inputs for log stats (aggregation)."""
    query: str = Field(description="LogsQL with | stats ... pipe")
    start: str = Field(default="-1h", description="Start time")
    end: str = Field(default="now", description="End time")
    step: str = Field(default="", description="If set use stats_query_range; else stats_query at end time")


class VictoriaLogStatsOutputs(BaseModel):
    """Outputs for victoria_log_stats."""
    results: List[Dict[str, Any]] = Field(description="Stat results")
    count: int = Field(description="Number of result series")
    result_type: str = Field(description="vector or matrix")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class VictoriaLogFacetsInputs(BaseModel):
    """Inputs for log facets (top field values)."""
    query: str = Field(description="LogsQL expression")
    limit: int = Field(default=10, description="Max fields to return")
    max_values_per_field: int = Field(default=10, description="Max values per field")


class VictoriaLogFacetsOutputs(BaseModel):
    """Outputs for victoria_log_facets."""
    facets: List[FacetField] = Field(description="Per-field top values")
    count: int = Field(description="Number of facets")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class VictoriaListStreamsInputs(BaseModel):
    """Inputs for listing log streams."""
    query: str = Field(default="*", description="LogsQL filter")
    start: str = Field(default="-1h", description="Start time")
    end: str = Field(default="now", description="End time")
    limit: int = Field(default=50, description="Max streams to return")


class VictoriaListStreamsOutputs(BaseModel):
    """Outputs for victoria_list_streams."""
    streams: List[Dict[str, Any]] = Field(description="Stream id and hit count")
    count: int = Field(description="Number of streams")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


# ── Plugin ─────────────────────────────────────────────────────────────────

class VictoriaPlugin(BasePlugin):
    """VictoriaMetrics and VictoriaLogs integration - metrics queries, log search, facets."""

    @property
    def plugin_name(self) -> str:
        return "victoria"

    @property
    def plugin_version(self) -> str:
        return "0.1.0"

    def get_connection_requirements(self) -> Dict[str, str]:
        return {
            "victoriametrics_url": "VictoriaMetrics base URL (e.g. http://victoriametrics:8428)",
            "victorialogs_url": "VictoriaLogs base URL (e.g. http://victorialogs:9428)",
            "auth_token": "Bearer token (optional, leave blank if no auth)",
        }

    def _headers(self) -> Dict[str, str]:
        config = getattr(self, "_config", None) or {}
        token = (config.get("auth_token") or "").strip()
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}

    def _vm_url(self) -> str:
        config = getattr(self, "_config", None) or {}
        base = (config.get("victoriametrics_url") or "").strip().rstrip("/")
        return base

    def _vl_url(self) -> str:
        config = getattr(self, "_config", None) or {}
        base = (config.get("victorialogs_url") or "").strip().rstrip("/")
        return base

    def get_tools(self) -> List[PluginToolSpec]:
        return [
            PluginToolSpec(
                name="victoria_query_metrics",
                category="plugin:victoria",
                description="Run instant PromQL/MetricsQL query against VictoriaMetrics",
                inputs_model=VictoriaQueryMetricsInputs,
                outputs_model=VictoriaQueryMetricsOutputs,
                tool_function=self._query_metrics,
            ),
            PluginToolSpec(
                name="victoria_query_range",
                category="plugin:victoria",
                description="Run range PromQL/MetricsQL query against VictoriaMetrics",
                inputs_model=VictoriaQueryRangeInputs,
                outputs_model=VictoriaQueryRangeOutputs,
                tool_function=self._query_range,
            ),
            PluginToolSpec(
                name="victoria_list_series",
                category="plugin:victoria",
                description="List time series matching a selector in VictoriaMetrics",
                inputs_model=VictoriaListSeriesInputs,
                outputs_model=VictoriaListSeriesOutputs,
                tool_function=self._list_series,
            ),
            PluginToolSpec(
                name="victoria_list_labels",
                category="plugin:victoria",
                description="List label names or values in VictoriaMetrics",
                inputs_model=VictoriaListLabelsInputs,
                outputs_model=VictoriaListLabelsOutputs,
                tool_function=self._list_labels,
            ),
            PluginToolSpec(
                name="victoria_tsdb_status",
                category="plugin:victoria",
                description="Get TSDB cardinality and status from VictoriaMetrics",
                inputs_model=VictoriaTsdbStatusInputs,
                outputs_model=VictoriaTsdbStatusOutputs,
                tool_function=self._tsdb_status,
            ),
            PluginToolSpec(
                name="victoria_query_logs",
                category="plugin:victoria",
                description="Search logs with LogsQL in VictoriaLogs",
                inputs_model=VictoriaQueryLogsInputs,
                outputs_model=VictoriaQueryLogsOutputs,
                tool_function=self._query_logs,
            ),
            PluginToolSpec(
                name="victoria_log_hits",
                category="plugin:victoria",
                description="Get log volume histogram from VictoriaLogs",
                inputs_model=VictoriaLogHitsInputs,
                outputs_model=VictoriaLogHitsOutputs,
                tool_function=self._log_hits,
            ),
            PluginToolSpec(
                name="victoria_log_stats",
                category="plugin:victoria",
                description="Run LogsQL stats aggregation in VictoriaLogs",
                inputs_model=VictoriaLogStatsInputs,
                outputs_model=VictoriaLogStatsOutputs,
                tool_function=self._log_stats,
            ),
            PluginToolSpec(
                name="victoria_log_facets",
                category="plugin:victoria",
                description="Get top field values (facets) for matching logs in VictoriaLogs",
                inputs_model=VictoriaLogFacetsInputs,
                outputs_model=VictoriaLogFacetsOutputs,
                tool_function=self._log_facets,
            ),
            PluginToolSpec(
                name="victoria_list_streams",
                category="plugin:victoria",
                description="List log streams in VictoriaLogs",
                inputs_model=VictoriaListStreamsInputs,
                outputs_model=VictoriaListStreamsOutputs,
                tool_function=self._list_streams,
            ),
        ]

    # ── VictoriaMetrics tools ─────────────────────────────────────────────

    async def _query_metrics(
        self,
        query: str,
        time: str = "",
    ) -> Dict[str, Any]:
        """Instant PromQL/MetricsQL query."""
        base = self._vm_url()
        if not base:
            return {
                "results": [],
                "count": 0,
                "result_type": "",
                "formatted": "Victoria plugin: set victoriametrics_url in connection config.",
            }
        try:
            import aiohttp
            url = f"{base}/api/v1/query"
            params = {"query": query}
            if time:
                params["time"] = time
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self._headers()) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        return {
                            "results": [],
                            "count": 0,
                            "result_type": "",
                            "formatted": f"VictoriaMetrics API error ({resp.status}): {text[:300]}",
                        }
            import json
            data = json.loads(text)
            status = data.get("status")
            if status != "success":
                return {
                    "results": [],
                    "count": 0,
                    "result_type": "",
                    "formatted": data.get("error", "Unknown error") or str(data)[:300],
                }
            inner = data.get("data", {})
            result_type = inner.get("resultType", "vector")
            raw = inner.get("result", [])
            results = []
            for r in raw:
                metric = r.get("metric", {})
                name = metric.get("__name__", "")
                labels = {k: v for k, v in metric.items() if k != "__name__"}
                val = r.get("value")
                if val is None:
                    continue
                ts, v = val[0], float(val[1]) if len(val) > 1 else 0.0
                from datetime import datetime
                ts_str = datetime.utcfromtimestamp(ts).isoformat() + "Z" if isinstance(ts, (int, float)) else str(ts)
                results.append(MetricSample(metric_name=name, labels=labels, value=v, timestamp=ts_str))
            lines = [f"{s.metric_name}{{{','.join(f'{k}=\"{v}\"' for k,v in s.labels.items())}}} = {s.value}" for s in results]
            formatted = "\n".join(lines) if lines else "No data points."
            return {
                "results": [s.model_dump() for s in results],
                "count": len(results),
                "result_type": result_type,
                "formatted": formatted,
            }
        except ImportError:
            return {"results": [], "count": 0, "result_type": "", "formatted": "Victoria plugin: aiohttp not installed."}
        except Exception as e:
            return {"results": [], "count": 0, "result_type": "", "formatted": f"Victoria query_metrics failed: {e}"}

    async def _query_range(
        self,
        query: str,
        start: str = "-1h",
        end: str = "now",
        step: str = "1m",
    ) -> Dict[str, Any]:
        """Range PromQL/MetricsQL query."""
        base = self._vm_url()
        if not base:
            return {
                "series": [],
                "count": 0,
                "result_type": "matrix",
                "formatted": "Victoria plugin: set victoriametrics_url in connection config.",
            }
        try:
            import aiohttp
            url = f"{base}/api/v1/query_range"
            params = {"query": query, "start": start, "end": end, "step": step}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self._headers()) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        return {
                            "series": [],
                            "count": 0,
                            "result_type": "matrix",
                            "formatted": f"VictoriaMetrics API error ({resp.status}): {text[:300]}",
                        }
            import json
            data = json.loads(text)
            if data.get("status") != "success":
                return {
                    "series": [],
                    "count": 0,
                    "result_type": "matrix",
                    "formatted": data.get("error", "Unknown error") or str(data)[:300],
                }
            raw = data.get("data", {}).get("result", [])
            series_list = []
            for r in raw:
                metric = r.get("metric", {})
                name = metric.get("__name__", "")
                labels = {k: v for k, v in metric.items() if k != "__name__"}
                vals = r.get("values", [])
                series_list.append(MetricSeries(metric_name=name, labels=labels, values=vals))
            lines = []
            for s in series_list:
                lbl = ",".join(f"{k}={v}" for k, v in s.labels.items())
                if not s.values:
                    lines.append(f"{s.metric_name}{{{lbl}}} (no points)")
                    continue
                vs = [float(v[1]) for v in s.values]
                lines.append(f"{s.metric_name}{{{lbl}}} min={min(vs):.2f} max={max(vs):.2f} avg={sum(vs)/len(vs):.2f} points={len(vs)}")
            formatted = "\n".join(lines) if lines else "No series."
            return {
                "series": [s.model_dump() for s in series_list],
                "count": len(series_list),
                "result_type": "matrix",
                "formatted": formatted,
            }
        except ImportError:
            return {"series": [], "count": 0, "result_type": "matrix", "formatted": "Victoria plugin: aiohttp not installed."}
        except Exception as e:
            return {"series": [], "count": 0, "result_type": "matrix", "formatted": f"Victoria query_range failed: {e}"}

    async def _list_series(
        self,
        match: str,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """List time series matching selector."""
        base = self._vm_url()
        if not base:
            return {"series": [], "count": 0, "formatted": "Victoria plugin: set victoriametrics_url in connection config."}
        try:
            import aiohttp
            url = f"{base}/api/v1/series"
            params = {"match[]": match}
            if limit:
                params["limit"] = limit
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self._headers()) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        return {"series": [], "count": 0, "formatted": f"VictoriaMetrics API error ({resp.status}): {text[:300]}"}
            import json
            data = json.loads(text)
            if data.get("status") != "success":
                return {"series": [], "count": 0, "formatted": data.get("error", "Unknown error") or str(data)[:300]}
            raw = data.get("data", [])[:limit]
            series = [dict(s) for s in raw]
            lines = [str(s) for s in series]
            formatted = "\n".join(lines) if lines else "No series matched."
            return {"series": series, "count": len(series), "formatted": formatted}
        except ImportError:
            return {"series": [], "count": 0, "formatted": "Victoria plugin: aiohttp not installed."}
        except Exception as e:
            return {"series": [], "count": 0, "formatted": f"Victoria list_series failed: {e}"}

    async def _list_labels(
        self,
        label_name: str = "",
        match: str = "",
    ) -> Dict[str, Any]:
        """List label names or values for a label."""
        base = self._vm_url()
        if not base:
            return {"values": [], "count": 0, "formatted": "Victoria plugin: set victoriametrics_url in connection config."}
        try:
            import aiohttp
            if label_name:
                url = f"{base}/api/v1/label/{label_name}/values"
                params = {}
                if match:
                    params["match[]"] = match
            else:
                url = f"{base}/api/v1/labels"
                params = {}
                if match:
                    params["match[]"] = match
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self._headers()) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        return {"values": [], "count": 0, "formatted": f"VictoriaMetrics API error ({resp.status}): {text[:300]}"}
            import json
            data = json.loads(text)
            if data.get("status") != "success":
                return {"values": [], "count": 0, "formatted": data.get("error", "Unknown error") or str(data)[:300]}
            values = data.get("data", [])
            formatted = ", ".join(str(v) for v in values) if values else "No values."
            return {"values": values, "count": len(values), "formatted": formatted}
        except ImportError:
            return {"values": [], "count": 0, "formatted": "Victoria plugin: aiohttp not installed."}
        except Exception as e:
            return {"values": [], "count": 0, "formatted": f"Victoria list_labels failed: {e}"}

    async def _tsdb_status(self) -> Dict[str, Any]:
        """TSDB cardinality and status."""
        base = self._vm_url()
        if not base:
            return {
                "total_series": 0,
                "total_label_value_pairs": 0,
                "series_count_by_metric": [],
                "series_count_by_label_value_pair": [],
                "formatted": "Victoria plugin: set victoriametrics_url in connection config.",
            }
        try:
            import aiohttp
            url = f"{base}/api/v1/status/tsdb"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self._headers()) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        return {
                            "total_series": 0,
                            "total_label_value_pairs": 0,
                            "series_count_by_metric": [],
                            "series_count_by_label_value_pair": [],
                            "formatted": f"VictoriaMetrics API error ({resp.status}): {text[:300]}",
                        }
            import json
            data = json.loads(text)
            if data.get("status") != "success":
                err = data.get("error", "Unknown error") or str(data)[:300]
                return {
                    "total_series": 0,
                    "total_label_value_pairs": 0,
                    "series_count_by_metric": [],
                    "series_count_by_label_value_pair": [],
                    "formatted": err,
                }
            inner = data.get("data", {})
            total_series = inner.get("numSeries", 0) or 0
            total_pairs = inner.get("numLabelValuePairs", 0) or 0
            by_metric = inner.get("seriesCountByMetricName", [])[:20]
            by_pair = inner.get("seriesCountByLabelValuePairs", [])[:20]
            by_metric_list = [{"name": x.get("name", ""), "count": x.get("value", 0)} for x in by_metric]
            by_pair_list = [{"name": x.get("name", ""), "count": x.get("value", 0)} for x in by_pair]
            lines = [f"Total series: {total_series}", f"Total label value pairs: {total_pairs}", "Top metrics by cardinality:", *[f"  {m['name']}: {m['count']}" for m in by_metric_list]]
            formatted = "\n".join(lines)
            return {
                "total_series": total_series,
                "total_label_value_pairs": total_pairs,
                "series_count_by_metric": by_metric_list,
                "series_count_by_label_value_pair": by_pair_list,
                "formatted": formatted,
            }
        except ImportError:
            return {
                "total_series": 0,
                "total_label_value_pairs": 0,
                "series_count_by_metric": [],
                "series_count_by_label_value_pair": [],
                "formatted": "Victoria plugin: aiohttp not installed.",
            }
        except Exception as e:
            return {
                "total_series": 0,
                "total_label_value_pairs": 0,
                "series_count_by_metric": [],
                "series_count_by_label_value_pair": [],
                "formatted": f"Victoria tsdb_status failed: {e}",
            }

    # ── VictoriaLogs tools ──────────────────────────────────────────────────

    async def _query_logs(
        self,
        query: str,
        limit: int = 100,
        start: str = "-1h",
        end: str = "now",
    ) -> Dict[str, Any]:
        """Search logs with LogsQL."""
        base = self._vl_url()
        if not base:
            return {
                "logs": [],
                "count": 0,
                "query_used": query,
                "formatted": "Victoria plugin: set victorialogs_url in connection config.",
            }
        try:
            import aiohttp
            url = f"{base}/select/logsql/query"
            params = {"query": query, "limit": limit, "start": start, "end": end}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self._headers()) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        return {
                            "logs": [],
                            "count": 0,
                            "query_used": query,
                            "formatted": f"VictoriaLogs API error ({resp.status}): {text[:300]}",
                        }
            import json
            logs = []
            for line in text.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    ts = obj.get("_time", "")
                    msg = obj.get("_msg", "")
                    stream = obj.get("_stream", "")
                    fields = {k: v for k, v in obj.items() if k not in ("_time", "_msg", "_stream")}
                    logs.append(LogEntry(timestamp=ts, message=msg, stream=stream, fields=fields))
                except json.JSONDecodeError:
                    continue
            lines = [f"{e.timestamp} {e.message}" for e in logs[:50]]
            formatted = "\n".join(lines) if lines else "No logs matched."
            if len(logs) > 50:
                formatted += f"\n... ({len(logs)} total)"
            return {
                "logs": [e.model_dump() for e in logs],
                "count": len(logs),
                "query_used": query,
                "formatted": formatted,
            }
        except ImportError:
            return {"logs": [], "count": 0, "query_used": query, "formatted": "Victoria plugin: aiohttp not installed."}
        except Exception as e:
            return {"logs": [], "count": 0, "query_used": query, "formatted": f"Victoria query_logs failed: {e}"}

    async def _log_hits(
        self,
        query: str,
        start: str = "-3h",
        end: str = "now",
        step: str = "15m",
        field: str = "",
    ) -> Dict[str, Any]:
        """Log volume histogram."""
        base = self._vl_url()
        if not base:
            return {"hits": [], "total": 0, "formatted": "Victoria plugin: set victorialogs_url in connection config."}
        try:
            import aiohttp
            url = f"{base}/select/logsql/hits"
            params = {"query": query, "start": start, "end": end, "step": step}
            if field:
                params["field"] = field
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self._headers()) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        return {"hits": [], "total": 0, "formatted": f"VictoriaLogs API error ({resp.status}): {text[:300]}"}
            import json
            data = json.loads(text)
            raw = data.get("hits", [])
            buckets = []
            for h in raw:
                buckets.append(HitsBucket(
                    group_fields=h.get("fields", {}),
                    timestamps=h.get("timestamps", []),
                    values=h.get("values", []),
                    total=h.get("total", 0),
                ))
            total = sum(b["total"] for b in raw) if raw else 0
            lines = []
            for b in buckets:
                lbl = f" {b.group_fields}" if b.group_fields else ""
                lines.append(f"{lbl}: {b.total} hits across {len(b.timestamps)} buckets")
            formatted = "\n".join(lines) if lines else "No hits."
            return {"hits": [b.model_dump() for b in buckets], "total": total, "formatted": formatted}
        except ImportError:
            return {"hits": [], "total": 0, "formatted": "Victoria plugin: aiohttp not installed."}
        except Exception as e:
            return {"hits": [], "total": 0, "formatted": f"Victoria log_hits failed: {e}"}

    async def _log_stats(
        self,
        query: str,
        start: str = "-1h",
        end: str = "now",
        step: str = "",
    ) -> Dict[str, Any]:
        """LogsQL stats aggregation."""
        base = self._vl_url()
        if not base:
            return {
                "results": [],
                "count": 0,
                "result_type": "",
                "formatted": "Victoria plugin: set victorialogs_url in connection config.",
            }
        try:
            import aiohttp
            if step:
                url = f"{base}/select/logsql/stats_query_range"
                params = {"query": query, "start": start, "end": end, "step": step}
            else:
                url = f"{base}/select/logsql/stats_query"
                params = {"query": query, "time": end}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self._headers()) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        return {
                            "results": [],
                            "count": 0,
                            "result_type": "",
                            "formatted": f"VictoriaLogs API error ({resp.status}): {text[:300]}",
                        }
            import json
            data = json.loads(text)
            if data.get("status") != "success":
                return {
                    "results": [],
                    "count": 0,
                    "result_type": "",
                    "formatted": data.get("error", "Unknown error") or str(data)[:300],
                }
            inner = data.get("data", {})
            result_type = inner.get("resultType", "vector")
            raw = inner.get("result", [])
            results = []
            for r in raw:
                metric = r.get("metric", {})
                val = r.get("value") or r.get("values", [])
                results.append({"metric": metric, "value": val})
            lines = [f"{r['metric']} -> {r['value']}" for r in results]
            formatted = "\n".join(lines) if lines else "No stats."
            return {"results": results, "count": len(results), "result_type": result_type, "formatted": formatted}
        except ImportError:
            return {"results": [], "count": 0, "result_type": "", "formatted": "Victoria plugin: aiohttp not installed."}
        except Exception as e:
            return {"results": [], "count": 0, "result_type": "", "formatted": f"Victoria log_stats failed: {e}"}

    async def _log_facets(
        self,
        query: str,
        limit: int = 10,
        max_values_per_field: int = 10,
    ) -> Dict[str, Any]:
        """Top field values (facets) for matching logs."""
        base = self._vl_url()
        if not base:
            return {"facets": [], "count": 0, "formatted": "Victoria plugin: set victorialogs_url in connection config."}
        try:
            import aiohttp
            url = f"{base}/select/logsql/facets"
            params = {"query": query, "limit": limit, "max_values_per_field": max_values_per_field}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self._headers()) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        return {"facets": [], "count": 0, "formatted": f"VictoriaLogs API error ({resp.status}): {text[:300]}"}
            import json
            data = json.loads(text)
            raw = data.get("facets", [])
            facets = []
            for f in raw:
                vals = [FacetValue(value=v.get("field_value", ""), hits=v.get("hits", 0)) for v in f.get("values", [])]
                facets.append(FacetField(field_name=f.get("field_name", ""), values=vals))
            lines = []
            for fa in facets:
                lines.append(f"{fa.field_name}:")
                for v in fa.values[:5]:
                    lines.append(f"  {v.value}: {v.hits}")
            formatted = "\n".join(lines) if lines else "No facets."
            return {"facets": [f.model_dump() for f in facets], "count": len(facets), "formatted": formatted}
        except ImportError:
            return {"facets": [], "count": 0, "formatted": "Victoria plugin: aiohttp not installed."}
        except Exception as e:
            return {"facets": [], "count": 0, "formatted": f"Victoria log_facets failed: {e}"}

    async def _list_streams(
        self,
        query: str = "*",
        start: str = "-1h",
        end: str = "now",
        limit: int = 50,
    ) -> Dict[str, Any]:
        """List log streams."""
        base = self._vl_url()
        if not base:
            return {"streams": [], "count": 0, "formatted": "Victoria plugin: set victorialogs_url in connection config."}
        try:
            import aiohttp
            url = f"{base}/select/logsql/streams"
            params = {"query": query, "start": start, "end": end}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self._headers()) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        return {"streams": [], "count": 0, "formatted": f"VictoriaLogs API error ({resp.status}): {text[:300]}"}
            import json
            data = json.loads(text)
            raw = data.get("values", [])[:limit]
            streams = [{"stream": v.get("value", ""), "hits": v.get("hits", 0)} for v in raw]
            lines = [f"{s['stream']}: {s['hits']} hits" for s in streams]
            formatted = "\n".join(lines) if lines else "No streams."
            return {"streams": streams, "count": len(streams), "formatted": formatted}
        except ImportError:
            return {"streams": [], "count": 0, "formatted": "Victoria plugin: aiohttp not installed."}
        except Exception as e:
            return {"streams": [], "count": 0, "formatted": f"Victoria list_streams failed: {e}"}
