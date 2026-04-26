from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.db import get_pool

logger = logging.getLogger(__name__)


async def write_stats(stats: dict) -> None:
    try:
        pool = await get_pool()
    except RuntimeError:
        logger.debug("DB unavailable, skipping stats write for request %s", stats.get("request_id", ""))
        return

    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO api_requests
                (time, request_id, api_key, model, is_streaming,
                 prompt_tokens, completion_tokens, total_tokens,
                 finish_reason, latency_ms, error_message, api_path)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
            now,
            stats["request_id"],
            stats["api_key"],
            stats["model"],
            stats["is_streaming"],
            stats["prompt_tokens"],
            stats["completion_tokens"],
            stats["total_tokens"],
            stats["finish_reason"],
            stats["latency_ms"],
            stats.get("error_message"),
            stats.get("api_path", ""),
        )

        for tool_name in stats.get("tool_calls", []):
            await conn.execute(
                """
                INSERT INTO tool_calls (time, request_id, api_key, tool_name)
                VALUES ($1, $2, $3, $4)
                """,
                now,
                stats["request_id"],
                stats["api_key"],
                tool_name,
            )

    logger.debug(
        "Stats written: request_id=%s model=%s tokens=%d latency=%.0fms",
        stats["request_id"],
        stats["model"],
        stats["total_tokens"],
        stats["latency_ms"],
    )
