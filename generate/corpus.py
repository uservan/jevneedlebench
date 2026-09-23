"""Paul Graham essays (from the original NIAH repo) as the natural-text haystack.

The source files are hard-wrapped and have lost paragraph breaks, so text is re-flowed into
sentences and regrouped into chunks of roughly 40-120 tokens; a chunk is the unit that needles
are inserted between."""
from __future__ import annotations

import random
import re
from functools import lru_cache

from .common import ROOT, ntok

CORPUS_DIR = ROOT / "corpus" / "paulgraham"
CHUNK_MIN, CHUNK_MAX = 40, 120


@lru_cache(maxsize=1)
def paragraphs() -> list[str]:
    """Deduplicated sentence-group chunks in a fixed order."""
    seen, out = set(), []
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        text = " ".join(text.split())
        text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)          # restore lost paragraph spacing
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunk, used = [], 0
        for s in sentences:
            n = ntok(s) + 1
            if used + n > CHUNK_MAX and used >= CHUNK_MIN:
                para = " ".join(chunk)
                if para not in seen and "user_" not in para and "code_" not in para:
                    seen.add(para); out.append(para)
                chunk, used = [], 0
            chunk.append(s); used += n
    return out


def sample_essay(rng: random.Random, budget: int) -> list[str]:
    """Random distinct chunks whose joined token count is <= budget (greedy)."""
    pool = paragraphs()[:]
    rng.shuffle(pool)
    out, used = [], 0
    for p in pool:
        n = ntok(p) + 1
        if used + n > budget:
            continue
        out.append(p); used += n
        if budget - used < CHUNK_MIN:
            break
    return out
