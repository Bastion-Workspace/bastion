"""
Trello plugin - reference integration for Agent Factory (Zone 4).

Provides tools to list boards and create cards. Requires Trello API key and token
in connection config. Used to prove the plugin discovery and registration pattern.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from orchestrator.plugins.base_plugin import BasePlugin, PluginToolSpec


class ListTrelloBoardsInputs(BaseModel):
    """Inputs for listing Trello boards."""

    filter_archived: bool = Field(default=False, description="Exclude archived boards")


class TrelloBoardRef(BaseModel):
    """Reference to a Trello board."""

    id: str = Field(description="Board ID")
    name: str = Field(description="Board name")
    url: str = Field(description="Board URL")
    closed: bool = Field(default=False, description="Whether board is archived")


class ListTrelloBoardsOutputs(BaseModel):
    """Outputs for list Trello boards tool."""

    boards: List[TrelloBoardRef] = Field(description="List of boards")
    count: int = Field(description="Number of boards")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class CreateTrelloCardInputs(BaseModel):
    """Inputs for creating a Trello card."""

    board_id: str = Field(description="Board ID")
    list_id: str = Field(description="List ID on the board")
    name: str = Field(description="Card title")
    description: str = Field(default="", description="Card description")


class CreateTrelloCardOutputs(BaseModel):
    """Outputs for create Trello card tool."""

    card_id: str = Field(description="Created card ID")
    card_url: str = Field(description="Card URL")
    success: bool = Field(description="Whether creation succeeded")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class TrelloPlugin(BasePlugin):
    """Trello integration plugin - lists boards and creates cards."""

    @property
    def plugin_name(self) -> str:
        return "trello"

    @property
    def plugin_version(self) -> str:
        return "0.1.0"

    def get_connection_requirements(self) -> Dict[str, str]:
        return {
            "api_key": "Trello API Key",
            "token": "Trello Token",
        }

    def get_tools(self) -> List[PluginToolSpec]:
        return [
            PluginToolSpec(
                name="trello_list_boards",
                category="plugin:trello",
                description="List Trello boards for the connected account",
                inputs_model=ListTrelloBoardsInputs,
                outputs_model=ListTrelloBoardsOutputs,
                tool_function=self._list_boards,
            ),
            PluginToolSpec(
                name="trello_create_card",
                category="plugin:trello",
                description="Create a card on a Trello list",
                inputs_model=CreateTrelloCardInputs,
                outputs_model=CreateTrelloCardOutputs,
                tool_function=self._create_card,
            ),
        ]

    async def _list_boards(self, filter_archived: bool = False) -> Dict[str, Any]:
        """List Trello boards. Uses configured api_key and token when set."""
        config = getattr(self, "_config", None) or {}
        api_key = config.get("api_key")
        token = config.get("token")
        if not api_key or not token:
            return {
                "boards": [],
                "count": 0,
                "formatted": "Trello plugin: configure with API key and token to list boards.",
            }
        try:
            import aiohttp
            url = "https://api.trello.com/1/members/me/boards"
            params = {"key": api_key, "token": token, "fields": "name,url,closed"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return {
                            "boards": [],
                            "count": 0,
                            "formatted": f"Trello API error ({resp.status}): {text[:200]}",
                        }
                    data = await resp.json()
            boards = []
            for b in data:
                if filter_archived and b.get("closed"):
                    continue
                boards.append(
                    TrelloBoardRef(
                        id=b["id"],
                        name=b.get("name", ""),
                        url=b.get("url", ""),
                        closed=b.get("closed", False),
                    )
                )
            formatted = f"Found {len(boards)} board(s): " + ", ".join(b.name for b in boards) if boards else "No boards found."
            return {
                "boards": [b.model_dump() for b in boards],
                "count": len(boards),
                "formatted": formatted,
            }
        except ImportError:
            return {
                "boards": [],
                "count": 0,
                "formatted": "Trello plugin: aiohttp not installed; cannot call Trello API.",
            }
        except Exception as e:
            return {
                "boards": [],
                "count": 0,
                "formatted": f"Trello list boards failed: {e}",
            }

    async def _create_card(
        self,
        board_id: str,
        list_id: str,
        name: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """Create a Trello card. Uses configured api_key and token when set."""
        config = getattr(self, "_config", None) or {}
        api_key = config.get("api_key")
        token = config.get("token")
        if not api_key or not token:
            return {
                "card_id": "",
                "card_url": "",
                "success": False,
                "formatted": "Trello plugin: configure with API key and token to create cards.",
            }
        try:
            import aiohttp
            url = "https://api.trello.com/1/cards"
            params = {"key": api_key, "token": token}
            payload = {"idList": list_id, "name": name, "desc": description}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=params, json=payload) as resp:
                    if resp.status not in (200, 201):
                        text = await resp.text()
                        return {
                            "card_id": "",
                            "card_url": "",
                            "success": False,
                            "formatted": f"Trello API error ({resp.status}): {text[:200]}",
                        }
                    data = await resp.json()
            card_id = data.get("id", "")
            card_url = data.get("url", "")
            return {
                "card_id": card_id,
                "card_url": card_url,
                "success": True,
                "formatted": f"Created card: {name} ({card_url})",
            }
        except ImportError:
            return {
                "card_id": "",
                "card_url": "",
                "success": False,
                "formatted": "Trello plugin: aiohttp not installed; cannot call Trello API.",
            }
        except Exception as e:
            return {
                "card_id": "",
                "card_url": "",
                "success": False,
                "formatted": f"Trello create card failed: {e}",
            }
