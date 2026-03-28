"""
S3-compatible connector executor — list, read, write, delete via aioboto3.

Works with AWS S3, MinIO, Cloudflare R2, Backblaze B2, etc. when endpoint_url is set.
"""

import base64
import logging
from typing import Any, Dict, List, Optional

import aioboto3

from service.connector_executor import _substitute_params

logger = logging.getLogger(__name__)


def _decode_write_payload(params: Dict[str, Any]) -> bytes:
    if params.get("content_base64") is not None:
        raw = params["content_base64"]
        if isinstance(raw, str):
            return base64.b64decode(raw)
        return bytes(raw)
    text = params.get("content_text")
    if text is not None:
        return str(text).encode("utf-8")
    raise ValueError("write requires content_base64 or content_text in params")


async def execute_s3_operation(
    definition: Dict[str, Any],
    credentials: Dict[str, Any],
    endpoint_id: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    params = dict(params or {})
    bucket = (definition.get("bucket") or "").strip()
    if not bucket:
        return {
            "records": [],
            "count": 0,
            "formatted": "Missing bucket in definition",
            "error": "S3 connector requires bucket in definition",
        }

    access_key = credentials.get("access_key_id") or credentials.get("aws_access_key_id")
    secret_key = credentials.get("secret_access_key") or credentials.get("aws_secret_access_key")
    session_token = credentials.get("session_token") or credentials.get("aws_session_token")
    if not access_key or not secret_key:
        return {
            "records": [],
            "count": 0,
            "formatted": "Missing access_key_id or secret_access_key",
            "error": "S3 requires access_key_id and secret_access_key in credentials",
        }

    region = definition.get("region") or "us-east-1"
    endpoint_url = (definition.get("endpoint_url") or "").strip() or None
    default_prefix = (definition.get("prefix") or "").strip()

    endpoints = definition.get("endpoints") or {}
    if isinstance(endpoints, list):
        endpoints = {ep.get("id") or ep.get("name"): ep for ep in endpoints if ep.get("id") or ep.get("name")}
    endpoint_def = endpoints.get(endpoint_id) if isinstance(endpoints, dict) else None
    if not endpoint_def:
        return {
            "records": [],
            "count": 0,
            "formatted": "Endpoint not found",
            "error": f"Unknown endpoint: {endpoint_id}",
        }

    operation = (endpoint_def.get("operation") or "").lower()
    merged = {**endpoint_def.get("defaults") or {}, **params}

    session = aioboto3.Session()
    client_kwargs: Dict[str, Any] = {
        "region_name": region,
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
    }
    if session_token:
        client_kwargs["aws_session_token"] = session_token
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    try:
        async with session.client("s3", **client_kwargs) as client:
            if operation == "list":
                prefix_tmpl = endpoint_def.get("prefix")
                if prefix_tmpl:
                    prefix_val = _substitute_params(prefix_tmpl, merged)
                    if "{" in prefix_val and "}" in prefix_val:
                        prefix_val = merged.get("prefix") or default_prefix or ""
                else:
                    prefix_val = merged.get("prefix") or default_prefix or ""
                records = await _s3_list(client, bucket, prefix_val)
                formatted = f"Listed {len(records)} object(s) under prefix {prefix_val!r}"
                return {"records": records, "count": len(records), "formatted": formatted}
            if operation == "read":
                key_tmpl = endpoint_def.get("key") or "{key}"
                key = _substitute_params(key_tmpl, merged)
                if default_prefix and key and not key.startswith("/"):
                    key = f"{default_prefix.rstrip('/')}/{key.lstrip('/')}"
                rec = await _s3_read(client, bucket, key)
                formatted = f"Read {rec.get('size', 0)} byte(s) from s3://{bucket}/{key}"
                return {"records": [rec], "count": 1, "formatted": formatted}
            if operation == "write":
                key_tmpl = endpoint_def.get("key") or "{key}"
                key = _substitute_params(key_tmpl, merged)
                if default_prefix and key and not key.startswith("/"):
                    key = f"{default_prefix.rstrip('/')}/{key.lstrip('/')}"
                data = _decode_write_payload(params)
                content_type = params.get("content_type") or "application/octet-stream"
                await client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
                rec = {"key": key, "operation": "write", "success": True, "bytes_written": len(data)}
                return {"records": [rec], "count": 1, "formatted": f"Wrote {len(data)} byte(s) to s3://{bucket}/{key}"}
            if operation == "delete":
                key_tmpl = endpoint_def.get("key") or "{key}"
                key = _substitute_params(key_tmpl, merged)
                if default_prefix and key and not key.startswith("/"):
                    key = f"{default_prefix.rstrip('/')}/{key.lstrip('/')}"
                await client.delete_object(Bucket=bucket, Key=key)
                rec = {"key": key, "operation": "delete", "success": True}
                return {"records": [rec], "count": 1, "formatted": f"Deleted s3://{bucket}/{key}"}
            return {
                "records": [],
                "count": 0,
                "formatted": f"Unknown operation: {operation}",
                "error": f"Unsupported S3 operation: {operation}",
            }
    except Exception as e:
        logger.exception("S3 operation failed: %s", e)
        return {
            "records": [],
            "count": 0,
            "formatted": str(e),
            "error": str(e),
        }


async def _s3_list(client: Any, bucket: str, prefix: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    token: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix or ""}
        if token:
            kwargs["ContinuationToken"] = token
        resp = await client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents") or []:
            lm = obj.get("LastModified")
            records.append(
                {
                    "key": obj.get("Key"),
                    "size": int(obj.get("Size") or 0),
                    "last_modified": lm.isoformat() if lm is not None and hasattr(lm, "isoformat") else None,
                    "etag": (obj.get("ETag") or "").strip('"'),
                }
            )
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return records


async def _s3_read(client: Any, bucket: str, key: str) -> Dict[str, Any]:
    resp = await client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"]
    data = await body.read()
    b64 = base64.b64encode(data).decode("ascii")
    ct = resp.get("ContentType") or "application/octet-stream"
    return {
        "key": key,
        "content": b64,
        "content_base64": b64,
        "size": len(data),
        "content_type": ct,
    }
