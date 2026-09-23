"""Common adapter contract.

An adapter receives only what the model may see (context, question, options) and returns a
Result. status is one of:
  ok           a choice inside the option set was returned
  invalid      the model answered but the choice is not an option id / probabilities malformed
  unsupported  the model refused the input size (context or option count over its limit)
  api_error    HTTP / network / parse failure
Adapters never truncate context, never retry a wrong answer, never fall back to another model.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Result:
    status: str
    choice_id: str | None = None
    probabilities: dict | None = None   # option_id -> p, when the model provides them
    confidence: float | None = None
    usage: dict = field(default_factory=dict)  # provider-reported token usage
    latency_ms: float = 0.0
    http_status: int | None = None
    error: str | None = None
    model: str = ""


class Adapter:
    name = "base"

    def choose(self, context: str, question: str, options: list[dict]) -> Result:
        raise NotImplementedError


def load_adapter(name: str, **kw) -> Adapter:
    if name == "jev":
        from .jev import JevAdapter
        return JevAdapter(**kw)
    if name == "mock":
        from .mock import MockAdapter
        return MockAdapter(**kw)
    if name == "openai":
        from .openai_compat import OpenAICompatAdapter
        return OpenAICompatAdapter(**kw)
    raise ValueError(f"unknown adapter {name}")
