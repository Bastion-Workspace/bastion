"""
HTTP webhook server for Microsoft Teams / Bot Framework inbound activities.
Runs alongside gRPC in the connections-service process.
"""

import json
import logging

from aiohttp import web

from providers.teams_provider import get_instance

logger = logging.getLogger(__name__)


async def handle_teams_activity(request: web.Request) -> web.Response:
    connection_id = request.match_info.get("connection_id") or ""
    provider = get_instance(connection_id)
    if not provider:
        return web.Response(status=404, text="unknown connection")
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.Response(status=400, text="invalid json")
    auth_header = request.headers.get("Authorization", "")
    try:
        await provider.handle_activity(body, auth_header)
    except ValueError as e:
        logger.warning("Teams webhook auth failed connection_id=%s: %s", connection_id, e)
        return web.Response(status=401, text="unauthorized")
    except Exception as e:
        logger.exception("Teams webhook handler error connection_id=%s: %s", connection_id, e)
        return web.Response(status=500, text="internal error")
    return web.Response(status=200)


async def start_teams_webhook_server(host: str, port: int) -> web.AppRunner:
    app = web.Application(client_max_size=10 * 1024 * 1024)
    app.router.add_post("/teams/webhook/{connection_id}", handle_teams_activity)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("Teams webhook listening on %s:%s", host, port)
    return runner
