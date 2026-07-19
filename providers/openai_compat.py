"""OpenAI-compatible provider — covers ~all free/open backends.

Works with: vLLM, Ollama, llama.cpp server, LM Studio (local); Groq,
OpenRouter, Google AI Studio, Cerebras, Together (free-tier hosted). All
expose POST /v1/chat/completions.

Rate limiting, retry/backoff and token accounting live here so the rest of
the system stays backend-agnostic. Not exercised by the offline test suite
(needs a live endpoint); import-safe with no network.
"""
from __future__ import annotations

import json
import time
from typing import Any

from .base import Message, ModelResponse, ToolCall, Usage


class _TokenBucket:
    """Minimal requests-per-minute limiter for free-tier endpoints."""

    def __init__(self, rpm: int | None) -> None:
        self.rpm = rpm
        self._interval = 60.0 / rpm if rpm else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if not self._interval:
            return
        now = time.monotonic()
        gap = self._interval - (now - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


class OpenAICompatProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        rpm_limit: int | None = None,
        max_retries: int = 4,
        timeout: float = 120.0,
    ) -> None:
        self.name = f"openai_compat:{model}"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._bucket = _TokenBucket(rpm_limit)
        self._max_retries = max_retries
        self._timeout = timeout

    def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> ModelResponse:
        import httpx  # local import keeps module import cheap

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            self._bucket.wait()
            t0 = time.monotonic()
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    r = client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                if r.status_code == 429:  # rate limited -> backoff
                    raise httpx.HTTPStatusError("429", request=r.request, response=r)
                r.raise_for_status()
                latency_ms = (time.monotonic() - t0) * 1000
                return self._parse(r.json(), latency_ms)
            except Exception as e:  # noqa: BLE001 - retry any transient failure
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"{self.name}: request failed after retries") from last_err

    @staticmethod
    def _parse(data: dict, latency_ms: float) -> ModelResponse:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(ToolCall(name=fn.get("name", ""), arguments=args))
        u = data.get("usage", {})
        usage = Usage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            latency_ms=latency_ms,
        )
        return ModelResponse(
            text=msg.get("content") or "",
            tool_calls=tool_calls,
            usage=usage,
            raw=data,
        )
