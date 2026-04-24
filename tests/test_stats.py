from app.stats import mask_api_key, extract_stats, StreamingStatsCollector


def test_mask_api_key_normal():
    assert mask_api_key("abcdefghij1234567890") == "abcdef...567890"


def test_mask_api_key_short():
    assert mask_api_key("abc") == "abc"


def test_mask_api_key_exact_12():
    assert mask_api_key("abcdefghijkl") == "abcdefghijkl"


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
    assert collector.request_id == "req-s"

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
