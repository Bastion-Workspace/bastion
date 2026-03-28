"""
Slack plugin - list channels, read history, post message, search, user info, set topic (Zone 4).
Uses Slack Web API. Requires bot token in connection config.
"""
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from pydantic import BaseModel, Field
from orchestrator.plugins.base_plugin import BasePlugin, PluginToolSpec
from orchestrator.utils.tool_type_models import SlackChannel, SlackMessage, SlackSearchResult, SlackUser


class ListChannelsInputs(BaseModel):
    types: str = Field(default="public_channel", description="Channel types")
    limit: int = Field(default=100, description="Max channels")


class ListChannelsOutputs(BaseModel):
    channels: List[SlackChannel] = Field(description="List of channels")
    count: int = Field(description="Number of channels")
    formatted: str = Field(description="Human-readable summary")


class ReadChannelHistoryInputs(BaseModel):
    channel_id: str = Field(description="Slack channel ID")
    limit: int = Field(default=20, description="Max messages")


class ReadChannelHistoryParams(BaseModel):
    oldest: Optional[str] = Field(default=None, description="Unix timestamp")
    latest: Optional[str] = Field(default=None, description="Unix timestamp")


class ReadChannelHistoryOutputs(BaseModel):
    messages: List[SlackMessage] = Field(description="Messages")
    count: int = Field(description="Number of messages")
    has_more: bool = Field(description="Whether more exist")
    formatted: str = Field(description="Human-readable summary")


class PostMessageInputs(BaseModel):
    channel_id: str = Field(description="Slack channel ID")
    text: str = Field(description="Message text")


class PostMessageParams(BaseModel):
    thread_ts: Optional[str] = Field(default=None, description="Reply in thread")
    unfurl_links: bool = Field(default=True, description="Unfurl links")


class PostMessageOutputs(BaseModel):
    success: bool = Field(description="Whether send succeeded")
    message_ts: Optional[str] = Field(default=None, description="Message timestamp")
    channel: str = Field(default="", description="Channel ID")
    formatted: str = Field(description="Human-readable summary")


class SearchMessagesInputs(BaseModel):
    query: str = Field(description="Search query")


class SearchMessagesParams(BaseModel):
    sort: str = Field(default="score", description="Sort by score or timestamp")
    count: int = Field(default=20, description="Max results")


class SearchMessagesOutputs(BaseModel):
    messages: List[SlackSearchResult] = Field(description="Matching messages")
    total: int = Field(description="Total matches")
    formatted: str = Field(description="Human-readable summary")


class GetUserInfoInputs(BaseModel):
    user_id: str = Field(description="Slack user ID")


class GetUserInfoOutputs(BaseModel):
    user: Optional[SlackUser] = Field(default=None, description="User info")
    formatted: str = Field(description="Human-readable summary")


class SetChannelTopicInputs(BaseModel):
    channel_id: str = Field(description="Slack channel ID")
    topic: str = Field(description="New topic text")


class SetChannelTopicOutputs(BaseModel):
    success: bool = Field(description="Whether update succeeded")
    formatted: str = Field(description="Human-readable summary")


class SlackPlugin(BasePlugin):
    SLACK_API_BASE = "https://slack.com/api"

    @property
    def plugin_name(self) -> str:
        return "slack"

    @property
    def plugin_version(self) -> str:
        return "0.1.0"

    def get_connection_requirements(self) -> Dict[str, str]:
        return {"bot_token": "Slack Bot Token (xoxb-...)"}

    def get_tools(self) -> List[PluginToolSpec]:
        return [
            PluginToolSpec("slack_list_channels", "plugin:slack", "List Slack channels", ListChannelsInputs, ListChannelsOutputs, self._list_channels),
            PluginToolSpec("slack_read_channel_history", "plugin:slack", "Read channel history", ReadChannelHistoryInputs, ReadChannelHistoryOutputs, self._read_channel_history, ReadChannelHistoryParams),
            PluginToolSpec("slack_post_message", "plugin:slack", "Post message to channel", PostMessageInputs, PostMessageOutputs, self._post_message, PostMessageParams),
            PluginToolSpec("slack_search_messages", "plugin:slack", "Search messages", SearchMessagesInputs, SearchMessagesOutputs, self._search_messages, SearchMessagesParams),
            PluginToolSpec("slack_get_user_info", "plugin:slack", "Get user info", GetUserInfoInputs, GetUserInfoOutputs, self._get_user_info),
            PluginToolSpec("slack_set_channel_topic", "plugin:slack", "Set channel topic", SetChannelTopicInputs, SetChannelTopicOutputs, self._set_channel_topic),
        ]

    def _headers(self) -> Dict[str, str]:
        config = getattr(self, "_config", None) or {}
        token = config.get("bot_token", "")
        return {"Authorization": f"Bearer {token}" if token else "", "Content-Type": "application/json; charset=utf-8"}

    async def _api(self, method: str, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            import aiohttp
        except ImportError:
            return {"ok": False, "error": "aiohttp not installed"}
        url = f"{self.SLACK_API_BASE}/{endpoint}"
        kwargs = {"headers": self._headers()}
        if payload is not None:
            kwargs["json"] = payload
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, **kwargs) as resp:
                try:
                    return await resp.json()
                except Exception:
                    return {"ok": False, "error": await resp.text()}

    async def _list_channels(self, types: str = "public_channel", limit: int = 100) -> Dict[str, Any]:
        if not (getattr(self, "_config", None) or {}).get("bot_token"):
            return {"channels": [], "count": 0, "formatted": "Slack plugin: configure bot_token."}
        type_val = types.strip().lower() if types else "public_channel"
        if type_val not in ("public_channel", "private_channel", "mpim", "im"):
            type_val = "public_channel"
        data = await self._api("GET", f"conversations.list?types={type_val}&limit={min(limit, 1000)}")
        if not data.get("ok"):
            return {"channels": [], "count": 0, "formatted": f"Slack API error: {data.get('error', 'unknown')}"}
        out = [SlackChannel(channel_id=c.get("id", ""), name=c.get("name", ""), is_private=c.get("is_private", False), num_members=c.get("num_members", 0), topic=(c.get("topic") or {}).get("value", ""), purpose=(c.get("purpose") or {}).get("value", "")) for c in data.get("channels", [])[:limit]]
        return {"channels": [c.model_dump() for c in out], "count": len(out), "formatted": f"Found {len(out)} channel(s)." if out else "No channels."}

    async def _read_channel_history(self, channel_id: str, limit: int = 20, oldest: Optional[str] = None, latest: Optional[str] = None) -> Dict[str, Any]:
        if not (getattr(self, "_config", None) or {}).get("bot_token"):
            return {"messages": [], "count": 0, "has_more": False, "formatted": "Slack plugin: configure bot_token."}
        params = f"channel={channel_id}&limit={min(limit, 1000)}"
        if oldest:
            params += f"&oldest={oldest}"
        if latest:
            params += f"&latest={latest}"
        data = await self._api("GET", f"conversations.history?{params}")
        if not data.get("ok"):
            return {"messages": [], "count": 0, "has_more": False, "formatted": f"Slack API error: {data.get('error', 'unknown')}"}
        out = [SlackMessage(ts=m.get("ts", ""), user=m.get("user", ""), text=m.get("text", ""), thread_ts=m.get("thread_ts")) for m in data.get("messages", [])]
        return {"messages": [x.model_dump() for x in out], "count": len(out), "has_more": data.get("has_more", False), "formatted": f"Found {len(out)} message(s)." if out else "No messages."}

    async def _post_message(self, channel_id: str, text: str, thread_ts: Optional[str] = None, unfurl_links: bool = True) -> Dict[str, Any]:
        if not (getattr(self, "_config", None) or {}).get("bot_token"):
            return {"success": False, "message_ts": None, "channel": "", "formatted": "Slack plugin: configure bot_token."}
        payload = {"channel": channel_id, "text": text, "unfurl_links": unfurl_links}
        if thread_ts:
            payload["thread_ts"] = thread_ts
        data = await self._api("POST", "chat.postMessage", payload)
        if not data.get("ok"):
            return {"success": False, "message_ts": None, "channel": channel_id, "formatted": f"Slack API error: {data.get('error', 'unknown')}"}
        return {"success": True, "message_ts": data.get("ts"), "channel": channel_id, "formatted": f"Posted message to {channel_id}."}

    async def _search_messages(self, query: str, sort: str = "score", count: int = 20) -> Dict[str, Any]:
        if not (getattr(self, "_config", None) or {}).get("bot_token"):
            return {"messages": [], "total": 0, "formatted": "Slack plugin: configure bot_token."}
        sort_val = "timestamp" if sort == "timestamp" else "score"
        data = await self._api("GET", f"search.messages?query={quote(query, safe='')}&sort={sort_val}&count={min(count, 100)}")
        if not data.get("ok"):
            return {"messages": [], "total": 0, "formatted": f"Slack API error: {data.get('error', 'unknown')}"}
        matches = (data.get("messages") or {}).get("matches", [])
        out = [SlackSearchResult(ts=m.get("ts", ""), channel_id=m.get("channel", {}).get("id", ""), channel_name=m.get("channel", {}).get("name", ""), user=m.get("user", ""), text=m.get("text", ""), permalink=m.get("permalink")) for m in matches]
        total = (data.get("messages") or {}).get("total", 0) or len(out)
        return {"messages": [x.model_dump() for x in out], "total": total, "formatted": f"Found {len(out)} message(s)." if out else "No messages."}

    async def _get_user_info(self, user_id: str) -> Dict[str, Any]:
        if not (getattr(self, "_config", None) or {}).get("bot_token"):
            return {"user": None, "formatted": "Slack plugin: configure bot_token."}
        data = await self._api("GET", f"users.info?user={user_id}")
        if not data.get("ok"):
            return {"user": None, "formatted": f"Slack API error: {data.get('error', 'unknown')}"}
        u = data.get("user") or {}
        user = SlackUser(user_id=u.get("id", ""), name=u.get("name", ""), real_name=u.get("real_name", ""), display_name=(u.get("profile") or {}).get("display_name", ""), is_bot=u.get("is_bot", False))
        return {"user": user.model_dump(), "formatted": f"User: {user.name}"}

    async def _set_channel_topic(self, channel_id: str, topic: str) -> Dict[str, Any]:
        if not (getattr(self, "_config", None) or {}).get("bot_token"):
            return {"success": False, "formatted": "Slack plugin: configure bot_token."}
        data = await self._api("POST", "conversations.setTopic", {"channel": channel_id, "topic": topic})
        if not data.get("ok"):
            return {"success": False, "formatted": f"Slack API error: {data.get('error', 'unknown')}"}
        return {"success": True, "formatted": f"Topic set for {channel_id}."}
