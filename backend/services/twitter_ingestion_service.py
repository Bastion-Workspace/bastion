import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx

from services.embedding_service_wrapper import get_embedding_service
from models.api_models import Chunk

logger = logging.getLogger(__name__)


@dataclass
class TwitterIngestionConfig:
    bearer_token: str
    user_id: str
    backfill_days: int = 60
    include_replies: bool = True
    include_retweets: bool = True
    include_quotes: bool = True


class TwitterIngestionService:
    """Service to fetch tweets and store embeddings into user-scoped vector collections."""

    def __init__(self):
        self._http = httpx.AsyncClient(timeout=30)
        self._embedding_manager = None

    async def initialize(self):
        if self._embedding_manager is None:
            self._embedding_manager = await get_embedding_service()

    async def close(self):
        try:
            await self._http.aclose()
        except Exception:
            pass

    async def fetch_following_tweets(self, cfg: TwitterIngestionConfig, since_id: Optional[str]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Fetch a page of home/Following timeline or per-followed users.
        Uses v2 search/recent as fallback; actual endpoint selection depends on token scope.
        Returns (tweets, next_since_id).
        """
        headers = {"Authorization": f"Bearer {cfg.bearer_token}"}

        # If we have user context, attempt home timeline (requires elevated scope); otherwise fallback to user tweets of followed accounts is non-trivial.
        # For now, use a conservative recent search on from:following handles is not feasible without following list.
        # Minimal viable: user's own tweets and mentions as seed; expandable later.
        query = f"from:{cfg.user_id} OR to:{cfg.user_id}"
        if not cfg.include_replies:
            query += " -is:reply"
        if not cfg.include_retweets:
            query += " -is:retweet"
        if not cfg.include_quotes:
            query += " -is:quote"

        params = {
            "query": query,
            "max_results": 100,
            "tweet.fields": "created_at,author_id,conversation_id,public_metrics,referenced_tweets,entities",
            "expansions": "referenced_tweets.id.author_id",
        }
        if since_id:
            params["since_id"] = since_id

        url = "https://api.twitter.com/2/tweets/search/recent"
        resp = await self._http.get(url, headers=headers, params=params)
        if resp.status_code == 401:
            raise ValueError("Twitter API unauthorized; check bearer token scope")
        resp.raise_for_status()
        data = resp.json()

        tweets = data.get("data", [])
        meta = data.get("meta", {})
        newest_id = meta.get("newest_id")
        return tweets, newest_id

    async def embed_tweets(self, tweets: List[Dict[str, Any]], user_id: str) -> int:
        if not tweets:
            return 0

        chunks: List[Chunk] = []
        for idx, t in enumerate(tweets):
            tweet_id = t.get("id")
            text = t.get("text", "").strip()
            if not text:
                continue

            metadata = {
                "source": "twitter",
                "tweet_id": tweet_id,
                "author_id": t.get("author_id"),
                "created_at": t.get("created_at"),
                "conversation_id": t.get("conversation_id"),
                "public_metrics": t.get("public_metrics"),
                "entities": t.get("entities"),
            }

            chunk = Chunk(
                chunk_id=f"tw_{tweet_id}",
                document_id=f"twitter_{tweet_id}",
                content=text,
                chunk_index=idx,
                quality_score=1.0,
                method="twitter",
                metadata=metadata,
            )
            chunks.append(chunk)

        if not chunks:
            return 0

        # Twitter data doesn't have category/tags metadata
        await self._embedding_manager.embed_and_store_chunks(
            chunks, 
            user_id=user_id,
            document_category=None,
            document_tags=None
        )
        return len(chunks)































