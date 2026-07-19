"""A network-free provider for tests and harness smoke runs.

Two modes:
  * handler=callable(messages) -> ModelResponse   (drive the model from a test)
  * responses=[ModelResponse, ...]                (cycled in order)

Default behaviour returns an empty final answer, which lets the full
reset -> step -> grade -> record -> report loop run with zero dependencies.
"""
from __future__ import annotations

from typing import Any, Callable

from .base import Message, ModelResponse, Usage


class DryRunProvider:
    name = "dry_run"

    def __init__(
        self,
        handler: Callable[[list[Message]], ModelResponse] | None = None,
        responses: list[ModelResponse] | None = None,
    ) -> None:
        self._handler = handler
        self._responses = responses or []
        self._i = 0

    def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> ModelResponse:
        if self._handler is not None:
            resp = self._handler(messages)
        elif self._responses:
            resp = self._responses[self._i % len(self._responses)]
            self._i += 1
        else:
            resp = ModelResponse(text="")
        # Stamp a nominal token/latency cost so overhead metrics have signal.
        if resp.usage.total_tokens == 0:
            approx = sum(len(str(m.get("content", ""))) for m in messages) // 4
            resp.usage = Usage(prompt_tokens=approx, completion_tokens=8, latency_ms=1.0)
        return resp
