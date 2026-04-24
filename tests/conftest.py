import pytest
from unittest.mock import AsyncMock


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
