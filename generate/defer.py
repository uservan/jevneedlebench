"""Scenario-3 items with an extra hand-off option: can Jev abstain on its own?

    python -m generate.defer --out data/h3_defer.jsonl

Takes the final-state items (h3_state), the snapshot-control items (h3_snapshot) and the verifier
items (h3_verify) and appends one option, inserted at a random position:
    "Cannot be determined reliably from the log; hand this question to a stronger model."
plus one sentence in the question telling the model to pick it when unsure. Everything else is
unchanged, so each item pairs with its original by id (suffix _defer). Never the correct answer."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from .common import Item, Option, finish_item, sub_rng, write_jsonl

DEFER = "Cannot be determined reliably from the log; hand this question to a stronger model."
HINT = " If you cannot determine the answer reliably from the log, choose the hand-off option."


def load(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/h3_defer.jsonl")
    args = ap.parse_args()
    src = [it for it in load("data/h3_state.jsonl.gz") if it["task"] == "log_final_state"]
    src += load("data/h3_snapshot.jsonl.gz") + load("data/h3_verify.jsonl.gz")
    items = []
    for it in src:
        texts = [o["text"] for o in it["options"]]
        correct_text = texts[[o["id"] for o in it["options"]].index(it["answer_id"])]
        pos = sub_rng("defer", it["id"]).randrange(len(texts) + 1)
        texts.insert(pos, DEFER)
        opts = [Option(f"option_{i}", t) for i, t in enumerate(texts)]
        items.append(finish_item(Item(
            id=f"{it['id']}_defer", base_id=it["base_id"], task=it["task"],
            condition={**it["condition"], "num_options": len(texts), "defer_option": True},
            context=it["context"], question=it["question"] + HINT, options=opts,
            answer_id=f"option_{texts.index(correct_text)}",
            metadata={**it["metadata"], "source_id": it["id"], "defer_id": f"option_{pos}",
                      "correct_index": texts.index(correct_text)})))
    sha = write_jsonl(items, Path(args.out))
    print(f"{len(items)} items -> {args.out} sha256 {sha[:12]}")


if __name__ == "__main__":
    main()
