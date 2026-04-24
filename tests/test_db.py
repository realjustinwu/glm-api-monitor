import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_init_db_creates_tables_and_hypertables():
    execute_calls = []

    mock_pool = MagicMock()
    mock_conn = MagicMock()

    async def mock_execute(sql, *args):
        execute_calls.append(sql)

    mock_conn.execute = mock_execute
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire.return_value = ctx

    with patch("app.db.asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool) as mock_create_pool:
        from app.db import init_db

        await init_db("postgresql://glm:glm@localhost:5432/glm_monitor")
        mock_create_pool.assert_called_once()

    sql_text = "\n".join(execute_calls)
    assert "api_requests" in sql_text
    assert "tool_calls" in sql_text
    assert "create_hypertable" in sql_text
    assert "api_requests_hourly" in sql_text
    assert "tool_calls_hourly" in sql_text
