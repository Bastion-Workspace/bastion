"""No-op WebSocket notifier for document-service (status goes via Redis)."""


class WebSocketNotifier:
    def __init__(self, ws_manager=None):
        self._ws = ws_manager

    async def notify_file_deleted(self, **kwargs):
        return None

    async def notify_folder_event(self, **kwargs):
        return None

    async def notify_folder_updated(self, **kwargs):
        return None

    async def notify_file_created(self, **kwargs):
        return None
