from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Anthropic API format stats extraction
# ---------------------------------------------------------------------------

def extract_anthropic_stats(response: dict) -> dict:
    usage = response.get("usage", {})
    tool_calls = []
    for block in response.get("content", []):
        if block.get("type") == "tool_use":
            name = block.get("name")
            if name:
                tool_calls.append(name)

    return {
        "request_id": response.get("id", ""),
        "model": response.get("model", ""),
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        "finish_reason": response.get("stop_reason", ""),
        "tool_calls": tool_calls,
    }


@dataclass
class AnthropicStreamingStatsCollector:
    request_id: str | None = None
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    tool_calls: list[str] = field(default_factory=list)
    _current_event: str = ""

    def process_chunk(self, raw_line: str) -> None:
        line = raw_line.strip()
        if line.startswith("event: "):
            self._current_event = line[len("event: "):]
            return
        if not line.startswith("data: "):
            return

        payload = line[len("data: "):]
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return

        event = self._current_event

        if event == "message_start":
            msg = data.get("message", data)
            self.request_id = msg.get("id")
            self.model = msg.get("model")
            usage = msg.get("usage", {})
            self.prompt_tokens = usage.get("input_tokens", 0)

        elif event == "content_block_start":
            block = data.get("content_block", data)
            if block.get("type") == "tool_use":
                name = block.get("name")
                if name and name not in self.tool_calls:
                    self.tool_calls.append(name)

        elif event == "message_delta":
            delta = data.get("delta", {})
            if delta.get("stop_reason"):
                self.finish_reason = delta["stop_reason"]
            usage = data.get("usage", {})
            self.completion_tokens = usage.get("output_tokens", self.completion_tokens)
            self.total_tokens = self.prompt_tokens + self.completion_tokens

        self._current_event = ""

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
