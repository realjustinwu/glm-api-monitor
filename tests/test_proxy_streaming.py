import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


def test_streaming_proxy_forwards_chunks():
    """Verify streaming response is forwarded chunk by chunk."""

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
    mock_httpx_client.build_request.return_value = MagicMock()
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
