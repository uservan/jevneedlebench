"""OpenAI-compatible chat-completions adapter for ordinary LLMs (placeholder for later runs).

The model is asked for a JSON object {"choice": "<option_id>"}; no probabilities are available,
so calibration metrics are skipped for such rows. Env: OPENAI_BASE_URL, OPENAI_API_KEY.
"""
from __future__ import annotations

import json
import os
import re
import time

import httpx

from .base import Adapter, Result

SYSTEM = ("You answer multiple-choice questions using only the provided context. "
          "Reply with a JSON object of the form {\"choice\": \"<option id>\"} and nothing else.")


class OpenAICompatAdapter(Adapter):
    name = "openai"

    def __init__(self, model: str, base_url: str | None = None, timeout_s: float = 300.0, temperature: float = 0.0):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.client = httpx.Client(timeout=timeout_s)

    def choose(self, context, question, options) -> Result:
        opt_text = "\n".join(f"{o['id']}: {o['text']}" for o in options)
        user = f"Context:\n{context}\n\nQuestion: {question}\n\nOptions:\n{opt_text}"
        body = {"model": self.model, "temperature": self.temperature,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
        t0 = time.perf_counter()
        try:
            resp = self.client.post(f"{self.base_url}/chat/completions", json=body,
                                    headers={"Authorization": f"Bearer {self.key}"})
        except httpx.HTTPError as exc:
            return Result("api_error", error=f"{type(exc).__name__}: {exc}"[:300], model=self.model)
        latency = (time.perf_counter() - t0) * 1000
        try:
            data = resp.json()
        except ValueError:
            return Result("api_error", http_status=resp.status_code, error="non-JSON body", latency_ms=latency, model=self.model)
        if resp.status_code != 200:
            err = str(data)[:300]
            status = "unsupported" if re.search(r"context.length|maximum context|too long|max_tokens", err, re.I) else "api_error"
            return Result(status, http_status=resp.status_code, error=err, latency_ms=latency, model=self.model)
        usage = data.get("usage") or {}
        usage = {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}
        try:
            text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            return Result("api_error", http_status=200, error="no message content", latency_ms=latency, model=self.model)
        m = re.search(r"\{.*\}", text, re.S)
        choice = None
        if m:
            try:
                choice = json.loads(m.group(0)).get("choice")
            except ValueError:
                pass
        ids = {o["id"] for o in options}
        if choice not in ids:
            return Result("invalid", http_status=200, error=f"unparsable answer: {text[:100]!r}", usage=usage,
                          latency_ms=latency, model=data.get("model") or self.model)
        return Result("ok", choice_id=choice, usage=usage, latency_ms=latency, http_status=200,
                      model=data.get("model") or self.model)
