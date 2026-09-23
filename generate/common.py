"""Shared pieces: fixed tokenizer, id pools, seeding, item schema, jsonl io."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("o200k_base")
ROOT = Path(__file__).resolve().parents[1]

LENGTHS = [1024, 2048, 4096, 8192, 16384, 20480]   # o200k tokens; 20k kv ~= 30k JEV tokens
ESSAY_LENGTHS = LENGTHS + [28672]                    # essay tokenizes ~1.08x, so it can reach JEV's ceiling
POSITIONS = [0.1, 0.5, 0.9]
NUM_OPTIONS = [2, 4, 8, 16, 32, 64]
NUM_UPDATES = [1, 2, 4, 8, 16]
PROJECTS = [f"project_{c}" for c in "ABCDEFGHIJKL"]


def ntok(text: str) -> int:
    return len(ENC.encode(text))


def sub_rng(*parts) -> random.Random:
    """Deterministic child RNG from any hashable parts."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# ---------------------------------------------------------------- ids
def near_id(a: str, b: str) -> bool:
    """Edit distance <= 1 on equal-length digit strings: one substitution or one adjacent swap."""
    if a == b:
        return True
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    if len(diff) == 1:
        return True
    return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]


def draw_ids(rng: random.Random, n: int, *, avoid_near: str | None = None, taken: set | None = None) -> list[str]:
    """n unique 5-digit strings; none within edit distance 1 of avoid_near. Mutates `taken`."""
    out = []
    taken = taken if taken is not None else set()
    while len(out) < n:
        s = f"{rng.randrange(100000):05d}"
        if s in taken or (avoid_near and near_id(s, avoid_near)):
            continue
        taken.add(s)
        out.append(s)
    return out


def similar_ids(rng: random.Random, target: str, n: int, taken: set) -> list[str]:
    """n distinct ids at edit distance exactly 1 from target."""
    cands = set()
    for i in range(5):
        for d in "0123456789":
            if d != target[i]:
                cands.add(target[:i] + d + target[i + 1:])
    for i in range(4):
        if target[i] != target[i + 1]:
            cands.add(target[:i] + target[i + 1] + target[i] + target[i + 2:])
    cands = sorted(cands - taken)
    rng.shuffle(cands)
    return cands[:n]


# ---------------------------------------------------------------- items
@dataclass
class Option:
    id: str
    text: str


@dataclass
class Item:
    id: str
    base_id: str
    task: str
    condition: dict
    context: str
    question: str
    options: list
    answer_id: str
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def arrange_options(rng: random.Random, correct: str, distractors: list[str], correct_index: int) -> tuple[list[Option], str]:
    """Place correct at correct_index, distractors shuffled into the remaining slots."""
    d = list(distractors)
    rng.shuffle(d)
    texts = d[:correct_index] + [correct] + d[correct_index:]
    opts = [Option(f"option_{i}", t) for i, t in enumerate(texts)]
    return opts, f"option_{correct_index}"


def correct_positions(rng: random.Random, k: int, repeats: int) -> list[int]:
    """Balanced correct positions over repeats: a random permutation of range(k), cycled."""
    perm = list(range(k))
    rng.shuffle(perm)
    return [perm[r % k] for r in range(repeats)]


def finish_item(item: Item) -> Item:
    ctx = ntok(item.context)
    opt = sum(ntok(o.text) for o in item.options)
    item.metadata.update(actual_context_tokens=ctx, options_tokens=opt,
                         total_input_tokens=ctx + ntok(item.question) + opt)
    return item


def write_jsonl(items: list[Item], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.to_json() + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def fit_prefix(records: list[str], budget: int, extra_tokens: int = 0) -> int:
    """Largest n such that '\\n'.join(records[:n]) plus extra_tokens fits in budget."""
    lo, hi = 0, len(records)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ntok("\n".join(records[:mid])) + extra_tokens <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo
