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
    mock_response.content = b'{"id":"test-123"}'
    mock_response.headers = {"content-type": "application/json"}

    mock_httpx_client = AsyncMock()
    mock_httpx_client.request.return_value = mock_response
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
    mock_writer.assert_called_once()
    call_stats = mock_writer.call_args[1]["stats"]
    assert call_stats["model"] == "glm-4"
    assert call_stats["prompt_tokens"] == 10
    assert call_stats["api_key"] == "abcdef...567890"
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
    mock_response.content = b'{"id":"test-456"}'
    mock_response.headers = {"content-type": "application/json"}

    mock_httpx_client = AsyncMock()
    mock_httpx_client.request.return_value = mock_response
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
    mock_response.content = b'{"error":{}}'
    mock_response.headers = {"content-type": "application/json"}

    mock_httpx_client = AsyncMock()
    mock_httpx_client.request.return_value = mock_response
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
