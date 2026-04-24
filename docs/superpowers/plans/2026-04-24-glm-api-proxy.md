# GLM API Monitor Proxy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a transparent reverse proxy for Zhipu GLM API that extracts usage statistics (tokens, tool calls, latency, errors) and stores them in TimescaleDB, with Grafana dashboard for visualization.

**Architecture:** FastAPI receives GLM API requests, forwards to upstream, parses responses to extract stats, writes to TimescaleDB. Streaming responses are forwarded chunk-by-chunk with the last chunk buffered for stats. Docker Compose runs proxy + TimescaleDB; Grafana is external.

**Tech Stack:** Python 3.12, FastAPI, httpx (async), asyncpg, TimescaleDB, Docker Compose

---

## File Structure

```
GLM-monitor/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry, catch-all proxy route
│   ├── config.py             # Environment config
│   ├── db.py                 # Database connection pool + init (tables, hypertables, aggregates)
│   ├── stats.py              # Stats extraction from response
│   └── writer.py             # Async writer: insert api_requests + tool_calls rows
├── grafana/
│   └── glm-monitor.json      # Grafana dashboard JSON (import manually)
└── tests/
    ├── __init__.py
    ├── conftest.py           # Shared fixtures (test client, mock upstream)
    ├── test_config.py
    ├── test_stats.py
    ├── test_proxy_nonstreaming.py
    ├── test_proxy_streaming.py
    └── test_writer.py
```

---

### Task 1: Project scaffolding + dependencies

**Files:**
- Create: `requirements.txt`
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create requirements.txt**

```
fastapi==0.115.12
uvicorn[standard]==0.34.2
httpx==0.28.1
asyncpg==0.30.0
pydantic-settings==2.9.1
pytest==8.3.5
pytest-asyncio==0.26.0
httpx  # already included above, used for TestClient too
```

- [ ] **Step 2: Create app/config.py**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    glm_upstream_url: str = "https://open.bigmodel.cn"
    database_url: str = "postgresql://glm:glm@timescaledb:5432/glm_monitor"
    proxy_port: int = 8000
    request_timeout: int = 120

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
```

- [ ] **Step 3: Create .env.example**

```
GLM_UPSTREAM_URL=https://open.bigmodel.cn
DATABASE_URL=postgresql://glm:glm@timescaledb:5432/glm_monitor
PROXY_PORT=8000
REQUEST_TIMEOUT=120
```

- [ ] **Step 4: Create Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 5: Create docker-compose.yml**

```yaml
services:
  proxy:
    build: .
    ports:
      - "${PROXY_PORT:-8000}:8000"
    environment:
      - GLM_UPSTREAM_URL=${GLM_UPSTREAM_URL:-https://open.bigmodel.cn}
      - DATABASE_URL=postgresql://glm:glm@timescaledb:5432/glm_monitor
      - REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-120}
    depends_on:
      timescaledb:
        condition: service_healthy
    restart: unless-stopped

  timescaledb:
    image: timescale/timescaledb:latest-pg16
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=glm
      - POSTGRES_PASSWORD=glm
      - POSTGRES_DB=glm_monitor
    volumes:
      - timescaledb_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U glm -d glm_monitor"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  timescaledb_data:
```

- [ ] **Step 6: Create app/__init__.py and tests/__init__.py**

Both files are empty.

- [ ] **Step 7: Create tests/conftest.py**

```python
import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_upstream_response():
    """Standard non-streaming GLM response."""
    return {
        "id": "test-req-123",
        "created": 1703487403,
        "model": "glm-4",
        "request_id": "test-req-123",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": "Hello!", "role": "assistant"},
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def mock_tool_call_response():
    """GLM response with tool calls."""
    return {
        "id": "test-req-456",
        "created": 1703487403,
        "model": "glm-4",
        "request_id": "test-req-456",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "index": 0,
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Beijing"}',
                            },
                        },
                        {
                            "id": "call_002",
                            "index": 1,
                            "type": "function",
                            "function": {
                                "name": "get_time",
                                "arguments": '{"timezone": "UTC"}',
                            },
                        },
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 15,
            "total_tokens": 65,
        },
    }


@pytest.fixture
def mock_db_writer():
    """Mock database writer for tests."""
    writer = AsyncMock()
    writer.write_request = AsyncMock()
    writer.write_tool_calls = AsyncMock()
    return writer
```

- [ ] **Step 8: Install dependencies and verify**

Run: `pip install -r requirements.txt`
Expected: all packages install successfully

- [ ] **Step 9: Commit**

```bash
git add requirements.txt Dockerfile docker-compose.yml .env.example app/__init__.py app/config.py tests/__init__.py tests/conftest.py
git commit -m "feat: project scaffolding with dependencies and config"
```

---

### Task 2: Database schema + connection

**Files:**
- Create: `app/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write test for database initialization**

Create `tests/test_db.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_init_db_creates_tables_and_hypertables():
    execute_calls = []

    mock_pool = AsyncMock()
    mock_conn = AsyncMock()

    async def mock_execute(sql, *args):
        execute_calls.append(sql)

    mock_conn.execute = mock_execute
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("app.db.asyncpg.create_pool", return_value=mock_pool) as mock_create_pool:
        from app.db import init_db

        await init_db("postgresql://glm:glm@localhost:5432/glm_monitor")
        mock_create_pool.assert_called_once()

    sql_text = "\n".join(execute_calls)
    assert "api_requests" in sql_text
    assert "tool_calls" in sql_text
    assert "create_hypertable" in sql_text
    assert "api_requests_hourly" in sql_text
    assert "tool_calls_hourly" in sql_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement app/db.py**

```python
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
                error_message TEXT
            )
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
            CREATE INDEX IF NOT EXISTS idx_tool_calls_name
            ON tool_calls (tool_name, time DESC)
        """)

        # Continuous aggregates
        await conn.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS api_requests_hourly
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 hour', time) AS bucket,
                api_key,
                model,
                count(*) AS request_count,
                sum(prompt_tokens) AS prompt_tokens,
                sum(completion_tokens) AS completion_tokens,
                sum(total_tokens) AS total_tokens,
                avg(latency_ms) AS avg_latency_ms,
                count(*) FILTER (WHERE error_message IS NOT NULL) AS error_count
            FROM api_requests
            GROUP BY bucket, api_key, model
        """)

        await conn.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS api_requests_daily
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 day', time) AS bucket,
                api_key,
                model,
                count(*) AS request_count,
                sum(prompt_tokens) AS prompt_tokens,
                sum(completion_tokens) AS completion_tokens,
                sum(total_tokens) AS total_tokens,
                avg(latency_ms) AS avg_latency_ms,
                count(*) FILTER (WHERE error_message IS NOT NULL) AS error_count
            FROM api_requests
            GROUP BY bucket, api_key, model
        """)

        await conn.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS tool_calls_hourly
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 hour', t.time) AS bucket,
                r.api_key,
                t.tool_name,
                count(*) AS call_count
            FROM tool_calls t
            JOIN api_requests r ON t.request_id = r.request_id
            GROUP BY bucket, r.api_key, t.tool_name
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_db.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/db.py tests/test_db.py
git commit -m "feat: database schema with TimescaleDB hypertables and continuous aggregates"
```

---

### Task 3: Stats extraction

**Files:**
- Create: `app/stats.py`
- Create: `tests/test_stats.py`

- [ ] **Step 1: Write tests for stats extraction**

Create `tests/test_stats.py`:

```python
from app.stats import mask_api_key, extract_stats, StreamingStatsCollector


def test_mask_api_key_normal():
    assert mask_api_key("abcdefghij1234567890") == "abcdef...34567890"


def test_mask_api_key_short():
    assert mask_api_key("abc") == "abc"


def test_mask_api_key_exact_12():
    assert mask_api_key("abcdefghijkl") == "abcdef...ghijkl"


def test_extract_stats_non_streaming():
    response = {
        "id": "req-123",
        "model": "glm-4",
        "request_id": "req-123",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "Hello!", "role": "assistant"},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    result = extract_stats(response)
    assert result["request_id"] == "req-123"
    assert result["model"] == "glm-4"
    assert result["prompt_tokens"] == 10
    assert result["completion_tokens"] == 20
    assert result["total_tokens"] == 30
    assert result["finish_reason"] == "stop"
    assert result["tool_calls"] == []


def test_extract_stats_with_tool_calls():
    response = {
        "id": "req-456",
        "model": "glm-4",
        "request_id": "req-456",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "get_weather"}},
                        {"function": {"name": "get_time"}},
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65},
    }

    result = extract_stats(response)
    assert result["finish_reason"] == "tool_calls"
    assert result["tool_calls"] == ["get_weather", "get_time"]


def test_extract_stats_no_usage():
    response = {
        "id": "req-789",
        "model": "glm-4",
        "request_id": "req-789",
        "choices": [],
    }

    result = extract_stats(response)
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert result["total_tokens"] == 0


def test_streaming_stats_collector():
    collector = StreamingStatsCollector()

    # Process chunks without usage
    collector.process_chunk('data: {"id":"req-s","model":"glm-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}\n\n')
    assert collector.request_id is None

    # Process final chunk with usage
    collector.process_chunk('data: {"id":"req-s","created":1703487403,"model":"glm-4","request_id":"req-s","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":5,"total_tokens":13}}\n\n')

    assert collector.request_id == "req-s"
    assert collector.model == "glm-4"
    assert collector.prompt_tokens == 8
    assert collector.finish_reason == "stop"

    # Process tool call in streaming
    collector2 = StreamingStatsCollector()
    collector2.process_chunk('data: {"id":"req-t","model":"glm-4","request_id":"req-t","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"function":{"name":"search","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n')
    assert collector2.tool_calls == ["search"]
    assert collector2.finish_reason == "tool_calls"


def test_streaming_stats_collector_ignores_done():
    collector = StreamingStatsCollector()
    collector.process_chunk("data: [DONE]\n\n")
    assert collector.request_id is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stats.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement app/stats.py**

```python
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def mask_api_key(key: str) -> str:
    if len(key) <= 12:
        return key
    return f"{key[:6]}...{key[-6:]}"


def extract_stats(response: dict) -> dict:
    usage = response.get("usage", {})
    choice = response.get("choices", [{}])[0] if response.get("choices") else {}
    message = choice.get("message", {})

    tool_calls = []
    for tc in message.get("tool_calls", []):
        func = tc.get("function", {})
        name = func.get("name")
        if name:
            tool_calls.append(name)

    return {
        "request_id": response.get("request_id") or response.get("id", ""),
        "model": response.get("model", ""),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "finish_reason": choice.get("finish_reason", ""),
        "tool_calls": tool_calls,
    }


@dataclass
class StreamingStatsCollector:
    request_id: str | None = None
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    tool_calls: list[str] = field(default_factory=list)

    def process_chunk(self, raw_line: str) -> None:
        line = raw_line.strip()
        if not line.startswith("data: "):
            return
        payload = line[len("data: "):]
        if payload == "[DONE]":
            return

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return

        if data.get("request_id"):
            self.request_id = data["request_id"]
        elif data.get("id"):
            self.request_id = data["id"]

        if data.get("model"):
            self.model = data["model"]

        usage = data.get("usage")
        if usage:
            self.prompt_tokens = usage.get("prompt_tokens", self.prompt_tokens)
            self.completion_tokens = usage.get("completion_tokens", self.completion_tokens)
            self.total_tokens = usage.get("total_tokens", self.total_tokens)

        choices = data.get("choices", [])
        if choices:
            choice = choices[0]
            fr = choice.get("finish_reason")
            if fr:
                self.finish_reason = fr

            delta = choice.get("delta", {})
            for tc in delta.get("tool_calls", []):
                func = tc.get("function", {})
                name = func.get("name")
                if name and name not in self.tool_calls:
                    self.tool_calls.append(name)

    def to_stats_dict(self) -> dict:
        return {
            "request_id": self.request_id or "",
            "model": self.model or "",
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "tool_calls": self.tool_calls,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stats.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/stats.py tests/test_stats.py
git commit -m "feat: stats extraction for non-streaming and streaming responses"
```

---

### Task 4: Async database writer

**Files:**
- Create: `app/writer.py`
- Create: `tests/test_writer.py`

- [ ] **Step 1: Write tests for the writer**

Create `tests/test_writer.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_write_request_and_tool_calls():
    mock_conn = AsyncMock()

    mock_pool = AsyncMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("app.writer.get_pool", return_value=mock_pool):
        from app.writer import write_stats

        await write_stats(
            stats={
                "request_id": "req-123",
                "api_key": "abcdef...xyz789",
                "model": "glm-4",
                "is_streaming": False,
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "finish_reason": "tool_calls",
                "latency_ms": 150.5,
                "tool_calls": ["get_weather", "get_time"],
            }
        )

    assert mock_conn.execute.await_count >= 1
    first_call_sql = mock_conn.execute.call_args_list[0][0][0]
    assert "api_requests" in first_call_sql

    assert mock_conn.execute.await_count >= 3
    tc_calls = [c for c in mock_conn.execute.call_args_list if "tool_calls" in c[0][0]]
    assert len(tc_calls) == 2


@pytest.mark.asyncio
async def test_write_request_no_tool_calls():
    mock_conn = AsyncMock()
    mock_pool = AsyncMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("app.writer.get_pool", return_value=mock_pool):
        from app.writer import write_stats

        await write_stats(
            stats={
                "request_id": "req-456",
                "api_key": "abcdef...xyz789",
                "model": "glm-4",
                "is_streaming": True,
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
                "finish_reason": "stop",
                "latency_ms": 2000.0,
                "tool_calls": [],
            }
        )

    assert mock_conn.execute.await_count == 1


@pytest.mark.asyncio
async def test_write_request_with_error():
    mock_conn = AsyncMock()
    mock_pool = AsyncMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("app.writer.get_pool", return_value=mock_pool):
        from app.writer import write_stats

        await write_stats(
            stats={
                "request_id": "",
                "api_key": "abcdef...xyz789",
                "model": "",
                "is_streaming": False,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "finish_reason": "error",
                "latency_ms": 5000.0,
                "error_message": "Connection timeout",
                "tool_calls": [],
            }
        )

    first_call_args = mock_conn.execute.call_args_list[0][0]
    assert "Connection timeout" in first_call_args
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_writer.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement app/writer.py**

```python
import logging
from datetime import datetime, timezone

from app.db import get_pool

logger = logging.getLogger(__name__)


async def write_stats(stats: dict) -> None:
    pool = await get_pool()
    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO api_requests
                (time, request_id, api_key, model, is_streaming,
                 prompt_tokens, completion_tokens, total_tokens,
                 finish_reason, latency_ms, error_message)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
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
        )

        for tool_name in stats.get("tool_calls", []):
            await conn.execute(
                """
                INSERT INTO tool_calls (time, request_id, tool_name)
                VALUES ($1, $2, $3)
                """,
                now,
                stats["request_id"],
                tool_name,
            )

    logger.debug(
        "Stats written: request_id=%s model=%s tokens=%d latency=%.0fms",
        stats["request_id"],
        stats["model"],
        stats["total_tokens"],
        stats["latency_ms"],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_writer.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/writer.py tests/test_writer.py
git commit -m "feat: async database writer for stats and tool calls"
```

---

### Task 5: FastAPI proxy — non-streaming

**Files:**
- Create: `app/main.py`
- Create: `tests/test_proxy_nonstreaming.py`

- [ ] **Step 1: Write tests for non-streaming proxy**

Create `tests/test_proxy_nonstreaming.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


def test_non_streaming_proxy_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "test-123",
        "created": 1703487403,
        "model": "glm-4",
        "request_id": "test-123",
        "choices": [
            {"finish_reason": "stop", "message": {"content": "Hi", "role": "assistant"}}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_response.headers = {"content-type": "application/json"}

    mock_httpx_client = AsyncMock()
    mock_httpx_client.post.return_value = mock_response
    mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
    mock_httpx_client.__aexit__ = AsyncMock(return_value=False)

    mock_writer = AsyncMock()

    with patch("app.main.httpx.AsyncClient", return_value=mock_httpx_client), \
         patch("app.main.write_stats", mock_writer), \
         patch("app.main.init_db", new_callable=AsyncMock):
        from app.main import app
        client = TestClient(app)

        response = client.post(
            "/api/paas/v4/chat/completions",
            json={"model": "glm-4", "messages": [{"role": "user", "content": "Hello"}]},
            headers={"Authorization": "Bearer abcdefghijklmnopqrstuvwxyz1234567890"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Hi"

    mock_writer.assert_called_once()
    call_stats = mock_writer.call_args[1]["stats"]
    assert call_stats["model"] == "glm-4"
    assert call_stats["prompt_tokens"] == 10
    assert call_stats["api_key"] == "abcdef...34567890"
    assert call_stats["finish_reason"] == "stop"
    assert call_stats["tool_calls"] == []


def test_non_streaming_proxy_with_tool_calls():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "test-456",
        "model": "glm-4",
        "request_id": "test-456",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "search", "arguments": '{"q": "test"}'}}
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    }
    mock_response.headers = {"content-type": "application/json"}

    mock_httpx_client = AsyncMock()
    mock_httpx_client.post.return_value = mock_response
    mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
    mock_httpx_client.__aexit__ = AsyncMock(return_value=False)

    mock_writer = AsyncMock()

    with patch("app.main.httpx.AsyncClient", return_value=mock_httpx_client), \
         patch("app.main.write_stats", mock_writer), \
         patch("app.main.init_db", new_callable=AsyncMock):
        from app.main import app
        client = TestClient(app)

        response = client.post(
            "/api/paas/v4/chat/completions",
            json={"model": "glm-4", "messages": [{"role": "user", "content": "Search"}]},
            headers={"Authorization": "Bearer abcdefghijklmnopqrstuvwxyz1234567890"},
        )

    assert response.status_code == 200
    call_stats = mock_writer.call_args[1]["stats"]
    assert call_stats["finish_reason"] == "tool_calls"
    assert call_stats["tool_calls"] == ["search"]


def test_proxy_upstream_error():
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.json.return_value = {"error": {"message": "Rate limited", "code": "429"}}
    mock_response.headers = {"content-type": "application/json"}

    mock_httpx_client = AsyncMock()
    mock_httpx_client.post.return_value = mock_response
    mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
    mock_httpx_client.__aexit__ = AsyncMock(return_value=False)

    mock_writer = AsyncMock()

    with patch("app.main.httpx.AsyncClient", return_value=mock_httpx_client), \
         patch("app.main.write_stats", mock_writer), \
         patch("app.main.init_db", new_callable=AsyncMock):
        from app.main import app
        client = TestClient(app)

        response = client.post(
            "/api/paas/v4/chat/completions",
            json={"model": "glm-4", "messages": [{"role": "user", "content": "Hello"}]},
            headers={"Authorization": "Bearer abcdefghijklmnopqrstuvwxyz1234567890"},
        )

    assert response.status_code == 429
    call_stats = mock_writer.call_args[1]["stats"]
    assert call_stats["finish_reason"] == "error"
    assert call_stats["error_message"] == "HTTP 429"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_proxy_nonstreaming.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement app/main.py (non-streaming part first, streaming added in Task 6)**

```python
import logging
import time

import httpx
from fastapi import FastAPI, Request, Response
from starlette.background import BackgroundTask

from app.config import settings
from app.db import init_db, close_db
from app.stats import mask_api_key, extract_stats
from app.writer import write_stats

logger = logging.getLogger(__name__)

app = FastAPI(title="GLM API Monitor Proxy")


@app.on_event("startup")
async def startup():
    await init_db(settings.database_url)
    logger.info("Proxy started, upstream=%s", settings.glm_upstream_url)


@app.on_event("shutdown")
async def shutdown():
    await close_db()


async def _proxy_non_streaming(
    client: httpx.AsyncClient, method: str, path: str, request: Request
) -> Response:
    start = time.monotonic()
    api_key_raw = ""
    model = ""
    request_body = {}

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key_raw = auth_header[7:]

    body = await request.body()
    try:
        import json
        request_body = json.loads(body)
        model = request_body.get("model", "")
    except Exception:
        pass

    is_streaming = request_body.get("stream", False)
    api_key_masked = mask_api_key(api_key_raw) if api_key_raw else ""

    upstream_url = f"{settings.glm_upstream_url}{path}"

    upstream_headers = dict(request.headers)
    upstream_headers.pop("host", None)
    upstream_headers.pop("content-length", None)

    try:
        upstream_resp = await client.request(
            method=method,
            url=upstream_url,
            content=body,
            headers=upstream_headers,
        )
    except httpx.TimeoutException as exc:
        latency = (time.monotonic() - start) * 1000
        await write_stats(stats={
            "request_id": "",
            "api_key": api_key_masked,
            "model": model,
            "is_streaming": is_streaming,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "finish_reason": "error",
            "latency_ms": latency,
            "error_message": f"Timeout: {exc}",
            "tool_calls": [],
        })
        return Response(content=str(exc), status_code=504)

    latency = (time.monotonic() - start) * 1000

    resp_body = upstream_resp.content
    resp_content = resp_body

    # Extract stats
    stats = extract_stats(upstream_resp.json() if upstream_resp.status_code == 200 else {})
    stats["api_key"] = api_key_masked
    stats["is_streaming"] = False
    stats["latency_ms"] = latency

    if upstream_resp.status_code >= 400:
        stats["finish_reason"] = "error"
        stats["error_message"] = f"HTTP {upstream_resp.status_code}"

    await write_stats(stats=stats)

    return Response(
        content=resp_content,
        status_code=upstream_resp.status_code,
        headers=dict(upstream_resp.headers),
    )


@app.api_route("/api/paas/v4/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_route(request: Request, path: str):
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        body = await request.body()
        import json
        try:
            request_body = json.loads(body)
        except Exception:
            request_body = {}

        is_streaming = request_body.get("stream", False)

        if is_streaming:
            return await _proxy_streaming(client, request, path)

        method = request.method
        return await _proxy_non_streaming(client, method, f"/api/paas/v4/{path}", request)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_proxy_nonstreaming.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_proxy_nonstreaming.py
git commit -m "feat: non-streaming proxy with stats extraction"
```

---

### Task 6: FastAPI proxy — streaming support

**Files:**
- Modify: `app/main.py`
- Create: `tests/test_proxy_streaming.py`

- [ ] **Step 1: Write tests for streaming proxy**

Create `tests/test_proxy_streaming.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


def test_streaming_proxy_forwards_chunks():
    """Verify streaming response is forwarded chunk by chunk."""
    from starlette.responses import StreamingResponse

    async def mock_stream():
        chunks = [
            'data: {"id":"s1","model":"glm-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}\n\n',
            'data: {"id":"s1","model":"glm-4","request_id":"s1","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}\n\n',
            "data: [DONE]\n\n",
        ]
        for chunk in chunks:
            yield chunk.encode("utf-8")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/event-stream"}
    mock_response.aiter_bytes.return_value = mock_stream()
    mock_response.aclose = AsyncMock()

    mock_httpx_client = AsyncMock()
    mock_httpx_client.send.return_value = mock_response
    mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
    mock_httpx_client.__aexit__ = AsyncMock(return_value=False)

    mock_writer = AsyncMock()

    with patch("app.main.httpx.AsyncClient", return_value=mock_httpx_client), \
         patch("app.main.write_stats", mock_writer), \
         patch("app.main.init_db", new_callable=AsyncMock):
        from app.main import app
        client = TestClient(app)

        response = client.post(
            "/api/paas/v4/chat/completions",
            json={"model": "glm-4", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
            headers={"Authorization": "Bearer abcdefghijklmnopqrstuvwxyz1234567890"},
        )

    assert response.status_code == 200
    assert "Hello" in response.text
    assert "world" in response.text

    mock_writer.assert_called_once()
    call_stats = mock_writer.call_args[1]["stats"]
    assert call_stats["is_streaming"] is True
    assert call_stats["prompt_tokens"] == 5
    assert call_stats["total_tokens"] == 8
    assert call_stats["finish_reason"] == "stop"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_proxy_streaming.py -v`
Expected: FAIL (`_proxy_streaming` not defined)

- [ ] **Step 3: Add streaming proxy to app/main.py**

Add the `_proxy_streaming` function and update imports in `app/main.py`:

```python
from app.stats import mask_api_key, extract_stats, StreamingStatsCollector
```

Add the `_proxy_streaming` function before the route handler:

```python
async def _proxy_streaming(
    client: httpx.AsyncClient, request: Request, path: str
) -> Response:
    start = time.monotonic()
    api_key_raw = ""
    model = ""

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key_raw = auth_header[7:]

    body = await request.body()
    try:
        import json
        request_body = json.loads(body)
        model = request_body.get("model", "")
    except Exception:
        pass

    api_key_masked = mask_api_key(api_key_raw) if api_key_raw else ""
    upstream_url = f"{settings.glm_upstream_url}/api/paas/v4/{path}"

    upstream_headers = dict(request.headers)
    upstream_headers.pop("host", None)
    upstream_headers.pop("content-length", None)

    req = client.build_request(
        method="POST",
        url=upstream_url,
        content=body,
        headers=upstream_headers,
    )
    upstream_resp = await client.send(req, stream=True)

    collector = StreamingStatsCollector()
    api_key_final = api_key_masked
    model_final = model

    async def stream_and_collect():
        try:
            async for chunk in upstream_resp.aiter_bytes():
                text = chunk.decode("utf-8", errors="replace")
                for line in text.split("\n"):
                    collector.process_chunk(line)
                yield chunk
        finally:
            latency = (time.monotonic() - start) * 1000
            stats = collector.to_stats_dict()
            stats["api_key"] = api_key_final
            stats["model"] = stats["model"] or model_final
            stats["is_streaming"] = True
            stats["latency_ms"] = latency

            if upstream_resp.status_code >= 400:
                stats["finish_reason"] = "error"
                stats["error_message"] = f"HTTP {upstream_resp.status_code}"

            await write_stats(stats=stats)
            await upstream_resp.aclose()

    return StreamingResponse(
        stream_and_collect(),
        status_code=upstream_resp.status_code,
        headers=dict(upstream_resp.headers),
        media_type="text/event-stream",
    )
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_proxy_streaming.py
git commit -m "feat: streaming proxy with SSE stats collection"
```

---

### Task 7: Grafana dashboard JSON

**Files:**
- Create: `grafana/glm-monitor.json`

- [ ] **Step 1: Create Grafana dashboard JSON**

Create `grafana/glm-monitor.json`:

```json
{
  "annotations": { "list": [] },
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 1,
  "id": null,
  "links": [],
  "panels": [
    {
      "title": "Token Usage Over Time",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
      "datasource": { "type": "postgres", "uid": "${DS_GLM}" },
      "targets": [
        {
          "rawSql": "SELECT bucket AS time, api_key, model, prompt_tokens, completion_tokens, total_tokens FROM api_requests_hourly WHERE bucket > $__timeFrom() AND bucket < $__timeTo() ORDER BY bucket",
          "format": "time_series"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "custom": { "drawStyle": "line", "fillOpacity": 10 },
          "unit": "short"
        },
        "overrides": [
          { "matcher": { "id": "byName", "options": "prompt_tokens" }, "properties": [{ "id": "color", "value": { "fixedColor": "blue", "mode": "fixed" } }] },
          { "matcher": { "id": "byName", "options": "completion_tokens" }, "properties": [{ "id": "color", "value": { "fixedColor": "green", "mode": "fixed" } }] }
        ]
      }
    },
    {
      "title": "Request Count & Latency",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 },
      "datasource": { "type": "postgres", "uid": "${DS_GLM}" },
      "targets": [
        {
          "rawSql": "SELECT bucket AS time, api_key, model, request_count, avg_latency_ms FROM api_requests_hourly WHERE bucket > $__timeFrom() AND bucket < $__timeTo() ORDER BY bucket",
          "format": "time_series"
        }
      ],
      "fieldConfig": {
        "defaults": { "custom": { "drawStyle": "line" } },
        "overrides": [
          { "matcher": { "id": "byName", "options": "avg_latency_ms" }, "properties": [{ "id": "unit", "value": "ms" }] }
        ]
      }
    },
    {
      "title": "Error Rate",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 8 },
      "datasource": { "type": "postgres", "uid": "${DS_GLM}" },
      "targets": [
        {
          "rawSql": "SELECT bucket AS time, api_key, model, error_count, request_count, CASE WHEN request_count > 0 THEN error_count::float / request_count * 100 ELSE 0 END AS error_rate_pct FROM api_requests_hourly WHERE bucket > $__timeFrom() AND bucket < $__timeTo() ORDER BY bucket",
          "format": "time_series"
        }
      ],
      "fieldConfig": {
        "defaults": {},
        "overrides": [
          { "matcher": { "id": "byName", "options": "error_rate_pct" }, "properties": [{ "id": "unit", "value": "percent" }] }
        ]
      }
    },
    {
      "title": "Tool Call Frequency",
      "type": "barchart",
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 8 },
      "datasource": { "type": "postgres", "uid": "${DS_GLM}" },
      "targets": [
        {
          "rawSql": "SELECT tool_name, sum(call_count) AS total_calls FROM tool_calls_hourly WHERE bucket > $__timeFrom() AND bucket < $__timeTo() GROUP BY tool_name ORDER BY total_calls DESC LIMIT 20",
          "format": "table"
        }
      ],
      "fieldConfig": {
        "defaults": { "links": [] },
        "overrides": []
      }
    },
    {
      "title": "Top Models by Token Usage",
      "type": "piechart",
      "gridPos": { "h": 8, "w": 6, "x": 0, "y": 16 },
      "datasource": { "type": "postgres", "uid": "${DS_GLM}" },
      "targets": [
        {
          "rawSql": "SELECT model, sum(total_tokens) AS tokens FROM api_requests_hourly WHERE bucket > $__timeFrom() AND bucket < $__timeTo() GROUP BY model ORDER BY tokens DESC",
          "format": "table"
        }
      ]
    },
    {
      "title": "Top API Keys by Requests",
      "type": "piechart",
      "gridPos": { "h": 8, "w": 6, "x": 6, "y": 16 },
      "datasource": { "type": "postgres", "uid": "${DS_GLM}" },
      "targets": [
        {
          "rawSql": "SELECT api_key, sum(request_count) AS total_requests FROM api_requests_hourly WHERE bucket > $__timeFrom() AND bucket < $__timeTo() GROUP BY api_key ORDER BY total_requests DESC",
          "format": "table"
        }
      ]
    },
    {
      "title": "Tool Calls by API Key",
      "type": "table",
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 16 },
      "datasource": { "type": "postgres", "uid": "${DS_GLM}" },
      "targets": [
        {
          "rawSql": "SELECT api_key, tool_name, sum(call_count) AS total_calls FROM tool_calls_hourly WHERE bucket > $__timeFrom() AND bucket < $__timeTo() GROUP BY api_key, tool_name ORDER BY total_calls DESC LIMIT 50",
          "format": "table"
        }
      ]
    },
    {
      "title": "Daily Token Usage Trend",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 24, "x": 0, "y": 24 },
      "datasource": { "type": "postgres", "uid": "${DS_GLM}" },
      "targets": [
        {
          "rawSql": "SELECT bucket AS time, api_key, model, prompt_tokens, completion_tokens FROM api_requests_daily WHERE bucket > $__timeFrom() AND bucket < $__timeTo() ORDER BY bucket",
          "format": "time_series"
        }
      ],
      "fieldConfig": {
        "defaults": { "custom": { "drawStyle": "bars", "fillOpacity": 80 } }
      }
    }
  ],
  "schemaVersion": 39,
  "tags": ["glm", "monitor"],
  "templating": {
    "list": [
      {
        "name": "DS_GLM",
        "type": "datasource",
        "query": "postgres",
        "current": { "selected": true, "text": "TimescaleDB", "value": "TimescaleDB" }
      }
    ]
  },
  "time": { "from": "now-7d", "to": "now" },
  "title": "GLM API Monitor",
  "uid": "glm-api-monitor"
}
```

- [ ] **Step 2: Commit**

```bash
git add grafana/glm-monitor.json
git commit -m "feat: Grafana dashboard JSON for GLM API monitoring"
```

---

### Task 8: Integration test + final wiring

**Files:**
- Modify: `app/main.py` (add .gitignore)
- Create: `.gitignore`

- [ ] **Step 1: Create .gitignore**

```
__pycache__/
*.pyc
.env
.pytest_cache/
*.egg-info/
dist/
build/
.venv/
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore"
```

- [ ] **Step 4: Verify Docker build**

Run: `docker compose build`
Expected: Build succeeds without errors
