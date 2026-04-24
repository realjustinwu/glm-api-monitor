import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_write_request_and_tool_calls():
    mock_conn = MagicMock()

    async def mock_execute(sql, *args):
        pass

    mock_conn.execute = mock_execute

    mock_pool = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire.return_value = ctx

    execute_spy = []
    original_execute = mock_conn.execute

    async def spy_execute(sql, *args):
        execute_spy.append((sql, args))
        await original_execute(sql, *args)

    mock_conn.execute = spy_execute

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

    assert len(execute_spy) == 3
    assert "api_requests" in execute_spy[0][0]
    tc_calls = [s for s in execute_spy if "tool_calls" in s[0]]
    assert len(tc_calls) == 2


@pytest.mark.asyncio
async def test_write_request_no_tool_calls():
    mock_conn = MagicMock()

    async def mock_execute(sql, *args):
        pass

    mock_conn.execute = mock_execute

    mock_pool = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire.return_value = ctx

    execute_spy = []
    original_execute = mock_conn.execute

    async def spy_execute(sql, *args):
        execute_spy.append((sql, args))
        await original_execute(sql, *args)

    mock_conn.execute = spy_execute

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

    assert len(execute_spy) == 1


@pytest.mark.asyncio
async def test_write_request_with_error():
    mock_conn = MagicMock()
    execute_spy = []

    async def spy_execute(sql, *args):
        execute_spy.append((sql, args))

    mock_conn.execute = spy_execute

    mock_pool = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire.return_value = ctx

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

    assert "Connection timeout" in execute_spy[0][1]
