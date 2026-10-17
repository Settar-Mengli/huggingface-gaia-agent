"""Dual token-budget guards for smolagents (no network / no secrets)."""

from __future__ import annotations

import os
from typing import Any, Optional


class ContextBudgetExceeded(RuntimeError):
    """Raised when a model request would exceed configured token budgets."""


def estimate_messages_tokens(messages: Any) -> int:
    """Conservative input-token estimate from message text (chars / 4)."""
    total_chars = _messages_char_count(messages)
    return max(1, (total_chars + 3) // 4) if total_chars else 0


def _messages_char_count(messages: Any) -> int:
    if messages is None:
        return 0
    if isinstance(messages, str):
        return len(messages)
    if isinstance(messages, dict):
        return _content_char_count(messages.get("content")) + len(
            str(messages.get("role") or "")
        )
    if isinstance(messages, (list, tuple)):
        return sum(_messages_char_count(m) for m in messages)
    content = getattr(messages, "content", None)
    if content is not None:
        role = getattr(messages, "role", "")
        return _content_char_count(content) + len(str(role or ""))
    return len(str(messages))


def _content_char_count(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                total += len(str(block.get("text") or ""))
            else:
                total += len(str(block))
        return total
    return len(str(content))


class TokenBudgetTracker:
    """Tracks cumulative input tokens and enforces per-request + cumulative caps."""

    def __init__(
        self,
        max_cumulative_input_tokens: Optional[int] = None,
        max_request_input_tokens: Optional[int] = None,
    ):
        self.max_cumulative_input_tokens = (
            max_cumulative_input_tokens
            if max_cumulative_input_tokens is not None
            else int(os.getenv("GAIA_MAX_CUMULATIVE_INPUT_TOKENS", "100000"))
        )
        self.max_request_input_tokens = (
            max_request_input_tokens
            if max_request_input_tokens is not None
            else int(os.getenv("GAIA_MAX_REQUEST_INPUT_TOKENS", "24000"))
        )
        self.cumulative_input_tokens = 0

    def reset(self) -> None:
        self.cumulative_input_tokens = 0

    def estimate_request_tokens(self, messages: Any) -> int:
        return estimate_messages_tokens(messages)

    def check_can_generate(self, messages: Any) -> int:
        """
        Enforce both budgets for the forthcoming request.

        Returns the estimated request token count when allowed.
        """
        estimated = self.estimate_request_tokens(messages)
        if estimated > self.max_request_input_tokens:
            raise ContextBudgetExceeded(
                f"Context budget exceeded: per-request estimate {estimated} "
                f"> limit {self.max_request_input_tokens}."
            )
        projected = self.cumulative_input_tokens + estimated
        if projected > self.max_cumulative_input_tokens:
            raise ContextBudgetExceeded(
                f"Context budget exceeded: cumulative projection {projected} "
                f"> limit {self.max_cumulative_input_tokens}."
            )
        return estimated

    def record_usage(self, *, estimated: int, token_usage: Any = None) -> None:
        actual = None
        if token_usage is not None:
            actual = getattr(token_usage, "input_tokens", None)
            if actual is None and isinstance(token_usage, dict):
                actual = token_usage.get("input_tokens")
        if isinstance(actual, int) and actual >= 0:
            self.cumulative_input_tokens += actual
        else:
            self.cumulative_input_tokens += max(0, int(estimated))


class BudgetGuardedModel:
    """
    Model wrapper that blocks oversized generate/generate_stream calls.

    Compatible with smolagents CodeAgent: delegates parse_tool_calls and
    other attributes to the inner model.
    """

    def __init__(self, inner: Any, tracker: TokenBudgetTracker):
        self.inner = inner
        self.tracker = tracker

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def generate(self, messages, **kwargs):
        estimated = self.tracker.check_can_generate(messages)
        result = self.inner.generate(messages, **kwargs)
        self.tracker.record_usage(
            estimated=estimated, token_usage=getattr(result, "token_usage", None)
        )
        return result

    def generate_stream(self, messages, **kwargs):
        estimated = self.tracker.check_can_generate(messages)
        # Streaming is uncommon in this project; still gate before starting.
        stream = self.inner.generate_stream(messages, **kwargs)
        self.tracker.record_usage(estimated=estimated, token_usage=None)
        return stream

    def parse_tool_calls(self, message):
        return self.inner.parse_tool_calls(message)

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)
