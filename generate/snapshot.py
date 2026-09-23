"""Heatmap 3 control: the same final-state items with a compiled state line appended to the log.

    python -m generate.snapshot --src data/h3_state.jsonl.gz --out data/h3_snapshot.jsonl

Two variants per item (condition.snapshot):
  correct  the line states the true final set          -> does Jev use a compiled state?
  stale    the line states the set before the last update -> does Jev trust the line over the log?
Everything else (log, question, options, option order) is identical to the source item."""
from __future__ import annotations

import argparse
import gzip
import json
import re
from dataclasses import asdict
from pathlib import Path

from .common import Item, Option, finish_item, write_jsonl
from .state import set_text, simulate
from .validate import LOG_RE

SNAP = "Summary as of the last entry: user_{u} currently belongs to: {s}"


def parse_events(context: str):
    return [(int(m.group(1)), m.group(2), "join" if m.group(3) == "joined" else "leave", m.group(4))
            for m in LOG_RE.finditer(context)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/h3_state.jsonl.gz")
    ap.add_argument("--out", default="data/h3_snapshot.jsonl")
    args = ap.parse_args()
    opener = gzip.open if args.src.endswith(".gz") else open
    with opener(args.src, "rt", encoding="utf-8") as fh:
        src = [json.loads(l) for l in fh if l.strip()]
    items = []
    for it in src:
        if it["task"] != "log_final_state":
            continue
        target = it["metadata"]["target_user"].split("_")[1]
        events = parse_events(it["context"])
        t_events = [e for e in events if e[1] == target]
        correct = set_text(simulate(t_events).get(target, set()))
        stale = set_text(simulate(t_events[:-1]).get(target, set()))
        answer_text = next(o["text"] for o in it["options"] if o["id"] == it["answer_id"])
        assert correct == answer_text and stale != correct
        for tag, text in (("correct", correct), ("stale", stale)):
            context = it["context"] + "\n" + SNAP.format(u=target, s=text) + "."
            items.append(finish_item(Item(
                id=f"{it['id']}_snap_{tag}", base_id=it["base_id"], task="log_final_state",
                condition={**it["condition"], "snapshot": tag},
                context=context, question=it["question"],
                options=[Option(**o) for o in it["options"]], answer_id=it["answer_id"],
                metadata={**it["metadata"], "snapshot_text": text,
                          "stale_in_options": any(o["text"] == stale for o in it["options"])})))
    sha = write_jsonl(items, Path(args.out))
    print(f"{len(items)} items -> {args.out} sha256 {sha[:12]}")


if __name__ == "__main__":
    main()
