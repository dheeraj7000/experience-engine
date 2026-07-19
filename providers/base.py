"""Model-provider abstraction.

Everything downstream depends only on this interface, never on a concrete
backend. Standardizing on the OpenAI chat-completions shape means local
(Ollama/vLLM) and free-tier hosted (Groq/OpenRouter/...) backends are
interchangeable behind `ModelProvider`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# A chat message is the OpenAI shape: {"role": "...", "content": "..."}.
Message = dict[str, Any]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.latency_ms + other.latency_ms,
        )


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelProvider(Protocol):
    """The single seam between the system and any LLM backend."""

    name: str

    def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> ModelResponse:
        ...
