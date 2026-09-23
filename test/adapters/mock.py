"""Offline mock for pipeline checks. Not a model: `random` guesses uniformly, `first` always
picks option_0, `oracle` cheats by reading the record from the context (kv only) to prove the
scoring path; results from it must never be reported as a model's."""
from __future__ import annotations

import random
import re
import time

from .base import Adapter, Result


class MockAdapter(Adapter):
    name = "mock"

    def __init__(self, mode: str = "random", seed: int = 0):
        self.mode, self.rng = mode, random.Random(seed)

    def choose(self, context, question, options) -> Result:
        t0 = time.perf_counter()
        ids = [o["id"] for o in options]
        if self.mode == "first":
            pick = ids[0]
        elif self.mode == "oracle":
            m = re.search(r"user_(\d{5})", question)
            hit = re.search(rf"The access code of user_{m.group(1)} is (code_\d{{5}})\.", context) if m else None
            pick = next((o["id"] for o in options if hit and o["text"] == hit.group(1)), ids[0])
        else:
            pick = self.rng.choice(ids)
        probs = {i: (0.7 if i == pick else 0.3 / (len(ids) - 1)) for i in ids}
        return Result("ok", choice_id=pick, probabilities=probs, confidence=0.7, http_status=200,
                      usage={"input_tokens": len(context) // 4}, latency_ms=(time.perf_counter() - t0) * 1000,
                      model=f"mock-{self.mode}")
