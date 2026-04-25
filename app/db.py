from __future__ import annotations

import asyncpg
import logging

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _pool


async def init_db(database_url: str) -> asyncpg.Pool:
    global _pool
    _pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)

    async with _pool.acquire() as conn:
        # Enable TimescaleDB extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

        # api_requests table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS api_requests (
                time        TIMESTAMPTZ  NOT NULL,
                request_id  TEXT         NOT NULL,
                api_key     TEXT         NOT NULL,
                model       TEXT         NOT NULL,
                is_streaming BOOLEAN     NOT NULL DEFAULT FALSE,
                prompt_tokens      INTEGER NOT NULL DEFAULT 0,
                completion_tokens  INTEGER NOT NULL DEFAULT 0,
                total_tokens       INTEGER NOT NULL DEFAULT 0,
                finish_reason TEXT      NOT NULL DEFAULT '',
                latency_ms   DOUBLE PRECISION NOT NULL DEFAULT 0,
                error_message TEXT,
                api_path     TEXT         NOT NULL DEFAULT ''
            )
        """)

        # Migration: add api_path column if missing (for existing deployments)
        await conn.execute("""
            ALTER TABLE api_requests ADD COLUMN IF NOT EXISTS api_path TEXT NOT NULL DEFAULT ''
        """)

        await conn.execute("""
            SELECT create_hypertable('api_requests', 'time',
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day')
        """)

        # tool_calls table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_calls (
                time        TIMESTAMPTZ  NOT NULL,
                request_id  TEXT         NOT NULL,
                api_key     TEXT         NOT NULL DEFAULT '',
                tool_name   TEXT         NOT NULL
            )
        """)

        await conn.execute("""
            SELECT create_hypertable('tool_calls', 'time',
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day')
        """)

        # Indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_requests_api_key
            ON api_requests (api_key, time DESC)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_requests_model
            ON api_requests (model, time DESC)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_requests_api_path
            ON api_requests (api_path, time DESC)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_calls_name
            ON tool_calls (tool_name, time DESC)
        """)

        # Migration: drop old continuous aggregates without api_path and recreate
        for view in ("api_requests_hourly", "api_requests_daily"):
            await conn.execute(f"DROP MATERIALIZED VIEW IF EXISTS {view}")

        # Continuous aggregates
        await conn.execute("""
            CREATE MATERIALIZED VIEW api_requests_hourly
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 hour', time) AS bucket,
                api_key,
                model,
                api_path,
                count(*) AS request_count,
                sum(prompt_tokens) AS prompt_tokens,
                sum(completion_tokens) AS completion_tokens,
                sum(total_tokens) AS total_tokens,
                avg(latency_ms) AS avg_latency_ms,
                count(*) FILTER (WHERE error_message IS NOT NULL) AS error_count
            FROM api_requests
            GROUP BY bucket, api_key, model, api_path
        """)

        await conn.execute("""
            CREATE MATERIALIZED VIEW api_requests_daily
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 day', time) AS bucket,
                api_key,
                model,
                api_path,
                count(*) AS request_count,
                sum(prompt_tokens) AS prompt_tokens,
                sum(completion_tokens) AS completion_tokens,
                sum(total_tokens) AS total_tokens,
                avg(latency_ms) AS avg_latency_ms,
                count(*) FILTER (WHERE error_message IS NOT NULL) AS error_count
            FROM api_requests
            GROUP BY bucket, api_key, model, api_path
        """)

        await conn.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS tool_calls_hourly
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 hour', time) AS bucket,
                api_key,
                tool_name,
                count(*) AS call_count
            FROM tool_calls
            GROUP BY bucket, api_key, tool_name
        """)

        # Refresh policies (every hour)
        await conn.execute("""
            SELECT add_continuous_aggregate_policy('api_requests_hourly',
                start_offset => INTERVAL '3 hours',
                end_offset => INTERVAL '1 hour',
                schedule_interval => INTERVAL '1 hour',
                if_not_exists => TRUE)
        """)
        await conn.execute("""
            SELECT add_continuous_aggregate_policy('api_requests_daily',
                start_offset => INTERVAL '3 days',
                end_offset => INTERVAL '1 day',
                schedule_interval => INTERVAL '1 hour',
                if_not_exists => TRUE)
        """)
        await conn.execute("""
            SELECT add_continuous_aggregate_policy('tool_calls_hourly',
                start_offset => INTERVAL '3 hours',
                end_offset => INTERVAL '1 hour',
                schedule_interval => INTERVAL '1 hour',
                if_not_exists => TRUE)
        """)

        # Retention: raw data 90 days
        await conn.execute("""
            SELECT add_retention_policy('api_requests', INTERVAL '90 days',
                if_not_exists => TRUE)
        """)
        await conn.execute("""
            SELECT add_retention_policy('tool_calls', INTERVAL '90 days',
                if_not_exists => TRUE)
        """)

    logger.info("Database initialized with tables, hypertables, and continuous aggregates")
    return _pool


async def close_db():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
