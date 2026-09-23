"""TypeSafe JEV via the native /v1/systemone endpoint.

Request: {"model", "state": {"context": ...}, "questions": {"decision": {"type": "choice",
          "instructions": question, "criteria": {option_id: option_text}}}}
Answer:  answers.decision.{choice, probabilities, confidence}; usage.{input_tokens, output_tokens}
Limits (docs.typesafe.ai/models): state + longest question <= 32k tokens. Over-limit requests
return HTTP 400 max_tokens_exceeded and are recorded as `unsupported`.
Key: env JEV_API_KEY (or .env in the project root). Never logged.
"""
from __future__ import annotations

import os
import time

import httpx

from .base import Adapter, Result

ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def _load_key() -> str:
    key = os.environ.get("JEV_API_KEY", "").strip()
    if not key:
        env = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        if os.path.exists(env):
            for line in open(env):
                if line.startswith("JEV_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError("JEV_API_KEY not set (env or .env)")
    return key


class JevAdapter(Adapter):
    name = "jev"

    def __init__(self, model: str = "jev-1.13.0", timeout_s: float = 120.0, max_429_retries: int = 3):
        self.model = model
        self.key = _load_key()
        self.timeout_s = timeout_s
        self.max_429_retries = max_429_retries
        self.client = httpx.Client(timeout=timeout_s)

    def choose(self, context, question, options) -> Result:
        criteria = {o["id"]: o["text"] for o in options}
        body = {"model": self.model, "state": {"context": context},
                "questions": {"decision": {"type": "choice", "instructions": question, "criteria": criteria}}}
        t0 = time.perf_counter()
        for attempt in range(self.max_429_retries + 1):
            try:
                resp = self.client.post(ENDPOINT, json=body, headers={"Authorization": f"Bearer {self.key}"})
            except httpx.HTTPError as exc:
                return Result("api_error", error=f"{type(exc).__name__}: {exc}"[:300], model=self.model,
                              latency_ms=(time.perf_counter() - t0) * 1000)
            if resp.status_code == 429 and attempt < self.max_429_retries:
                time.sleep(1.0 * 2 ** attempt)
                continue
            break
        latency = (time.perf_counter() - t0) * 1000
        try:
            data = resp.json()
        except ValueError:
            return Result("api_error", http_status=resp.status_code, error="non-JSON body", latency_ms=latency, model=self.model)
        if resp.status_code != 200:
            err = str(data)[:300]
            status = "unsupported" if "max_tokens_exceeded" in err else "api_error"
            return Result(status, http_status=resp.status_code, error=err, latency_ms=latency, model=self.model)
        ans = (data.get("answers") or {}).get("decision") or {}
        res = Result("ok", http_status=200, latency_ms=latency, usage=data.get("usage") or {},
                     model=data.get("model") or self.model, probabilities=ans.get("probabilities"),
                     confidence=ans.get("confidence"))
        choice = ans.get("choice")
        if ans.get("type") != "choice" or choice not in criteria:
            res.status, res.error = "invalid", f"choice {choice!r} not in options"
            return res
        probs = ans.get("probabilities")
        if not isinstance(probs, dict) or set(probs) != set(criteria) or abs(sum(probs.values()) - 1) > 0.02:
            res.status, res.error = "invalid", "malformed probabilities"
            return res
        res.choice_id = choice
        return res
