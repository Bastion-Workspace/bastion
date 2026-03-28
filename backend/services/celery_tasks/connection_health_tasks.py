"""
Connection Health Tasks - Periodic sync of chat bot and OAuth connections.

sync_chat_bot_connections: Beat task that runs every 2 minutes, checks
  GetBotStatus for each active chat_bot connection, re-registers any that
  are stopped (e.g. after connections-service restart).

sync_oauth_connections: Beat task that runs every 30 minutes, proactively
  refreshes OAuth (email) tokens that expire within 10 minutes.
"""

import asyncio
import logging
from typing import Any, Dict

import grpc
from services.celery_app import celery_app
from services.celery_tasks.async_runner import run_async
from services.database_manager.database_helpers import fetch_all
from services.external_connections_service import external_connections_service

logger = logging.getLogger(__name__)


async def _async_sync_chat_bot_connections() -> Dict[str, Any]:
    """Query active chat_bot connections, check status on connections-service, re-register stopped bots."""
    from clients.connections_service_client import ConnectionsServiceClient

    # Use a fresh client in this event loop (Celery workers fork; global singleton may be bound to a closed loop).
    client = ConnectionsServiceClient()
    try:
        await client.initialize()
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            logger.warning(
                "Connections service unavailable (DNS/network). Skipping chat bot sync. "
                "Ensure connections-service is running and reachable if using chat bots."
            )
        elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
            logger.warning(
                "Connections service did not respond in time. Skipping chat bot sync. "
                "Ensure connections-service is running and not overloaded if using chat bots."
            )
        else:
            logger.exception("Connections service client error: %s", e)
        return {"checked": 0, "re_registered": 0, "error": "connections_service_unavailable"}
    except Exception as e:
        logger.exception("Failed to initialize connections service client: %s", e)
        return {"checked": 0, "re_registered": 0, "error": str(e)}

    try:
        rows = await fetch_all("SELECT user_id FROM users")
        user_ids = [r["user_id"] for r in rows] if rows else []
        checked = 0
        re_registered = 0
        for uid in user_ids:
            conns = await external_connections_service.get_user_connections(
                uid,
                connection_type="chat_bot",
                active_only=True,
                rls_context={"user_id": uid},
            )
            for conn in conns:
                conn_id = conn["id"]
                checked += 1
                try:
                    status_result = await client.get_bot_status(conn_id)
                except Exception as e:
                    logger.warning("GetBotStatus failed for connection %s: %s", conn_id, e)
                    status_result = {"status": "stopped"}
                status = (status_result.get("status") or "").strip().lower()
                if status != "running":
                    token = await external_connections_service.get_valid_access_token(
                        conn_id, rls_context={"user_id": uid}
                    )
                    if not token:
                        logger.warning("No valid token for connection %s, skip re-register", conn_id)
                        continue
                    meta = external_connections_service._parse_provider_metadata(
                        conn.get("provider_metadata")
                    )
                    config = {k: str(v) for k, v in meta.items() if v is not None}
                    result = await client.register_bot(
                        connection_id=conn_id,
                        user_id=uid,
                        provider=conn.get("provider", ""),
                        bot_token=token,
                        display_name=(conn.get("display_name") or conn.get("account_identifier") or ""),
                        config=config,
                    )
                    if result.get("success"):
                        re_registered += 1
                        logger.info("Re-registered chat bot connection %s", conn_id)
                    else:
                        logger.warning("Re-register failed for connection %s: %s", conn_id, result.get("error"))
        if checked or re_registered:
            logger.info("Chat bot health check: %s checked, %s re-registered", checked, re_registered)
        return {"checked": checked, "re_registered": re_registered}
    except Exception as e:
        logger.exception("sync_chat_bot_connections failed: %s", e)
        return {"checked": 0, "re_registered": 0, "error": str(e)}
    finally:
        await client.close()
        await asyncio.sleep(0.25)

@celery_app.task(bind=True, name="services.celery_tasks.connection_health_tasks.sync_chat_bot_connections")
def sync_chat_bot_connections(self) -> Dict[str, Any]:
    """Celery Beat: every 2 minutes, check chat bot status and re-register stopped bots."""
    try:
        return run_async(_async_sync_chat_bot_connections())
    except Exception as e:
        logger.exception("sync_chat_bot_connections failed: %s", e)
        return {"checked": 0, "re_registered": 0, "error": str(e)}


async def _async_sync_oauth_connections() -> Dict[str, Any]:
    """Proactively refresh OAuth (email) tokens that expire within 10 minutes."""
    from config import settings

    if not getattr(settings, "MICROSOFT_CLIENT_ID", ""):
        logger.warning(
            "MICROSOFT_CLIENT_ID not set; OAuth refresh will fail for Microsoft connections"
        )
    try:
        result = await external_connections_service.refresh_all_expired_oauth_tokens(
            buffer_seconds=600
        )
        if result.get("refreshed") or result.get("failed"):
            logger.info(
                "OAuth health check: %s refreshed, %s failed",
                result.get("refreshed", 0),
                result.get("failed", 0),
            )
        return result
    except Exception as e:
        logger.exception("sync_oauth_connections failed: %s", e)
        return {"refreshed": 0, "failed": 0, "error": str(e)}
    finally:
        await asyncio.sleep(0)


@celery_app.task(bind=True, name="services.celery_tasks.connection_health_tasks.sync_oauth_connections")
def sync_oauth_connections(self) -> Dict[str, Any]:
    """Celery Beat: every 30 minutes, refresh OAuth tokens expiring within 10 minutes."""
    try:
        return run_async(_async_sync_oauth_connections())
    except Exception as e:
        logger.exception("sync_oauth_connections failed: %s", e)
        return {"refreshed": 0, "failed": 0, "error": str(e)}
