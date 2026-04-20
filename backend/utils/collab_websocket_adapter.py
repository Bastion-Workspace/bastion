"""
Adapter between Starlette/FastAPI WebSocket and pycrdt-websocket's Websocket protocol.
"""

from __future__ import annotations

from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState


class FastAPIWebsocketAdapter:
    """Bridge FastAPI WebSocket to pycrdt-websocket's expected async bytes interface."""

    __slots__ = ("_websocket", "_path")

    def __init__(self, websocket: WebSocket, path: str):
        self._websocket = websocket
        self._path = path

    @property
    def path(self) -> str:
        return self._path

    async def send(self, message: bytes) -> None:
        if self._websocket.application_state != WebSocketState.CONNECTED:
            return
        await self._websocket.send_bytes(message)

    async def recv(self) -> bytes:
        message = await self._websocket.receive()
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(code=message.get("code", 1000))
        if message["type"] == "websocket.receive":
            data = message.get("bytes")
            if data is not None:
                return data
            text = message.get("text")
            if text is not None:
                return text.encode("utf-8")
        raise WebSocketDisconnect(code=1002)

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        try:
            return await self.recv()
        except WebSocketDisconnect:
            raise StopAsyncIteration
