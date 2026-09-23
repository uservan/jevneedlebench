"""Heatmap 3 verifier test: can Jev tell whether a state summary is correct?

    python -m generate.verify --src data/h3_state.jsonl.gz --out data/h3_verify.jsonl

For every final-state item the log gets one summary line appended; half of the items get the
true final set, half get the state before the last update (a realistic Summary failure). Jev is
asked whether the summary is correct, as a 2-option choice (correct / incorrect, order shuffled).
Task: log_verify_summary; condition.snapshot = correct | stale; answer is "correct" iff snapshot
is correct."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from .common import Item, arrange_options, finish_item, sub_rng, write_jsonl
from .snapshot import SNAP, parse_events
from .state import RULES, set_text, simulate

OPT_OK = "Correct: the summary matches the user's projects after the last log entry."
OPT_BAD = "Incorrect: the summary does not match the user's projects after the last log entry."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/h3_state.jsonl.gz")
    ap.add_argument("--out", default="data/h3_verify.jsonl")
    args = ap.parse_args()
    opener = gzip.open if args.src.endswith(".gz") else open
    with opener(args.src, "rt", encoding="utf-8") as fh:
        src = [json.loads(l) for l in fh if l.strip()]
    items = []
    for it in src:
        if it["task"] != "log_final_state":
            continue
        target = it["metadata"]["target_user"].split("_")[1]
        t_events = [e for e in parse_events(it["context"]) if e[1] == target]
        correct = set_text(simulate(t_events).get(target, set()))
        stale = set_text(simulate(t_events[:-1]).get(target, set()))
        assert stale != correct
        question = (f"{RULES} The last line of the log is a summary of user_{target}'s projects. "
                    f"According to the log entries and the rules, is that summary correct?")
        for tag, text in (("correct", correct), ("stale", stale)):
            context = it["context"] + "\n" + SNAP.format(u=target, s=text) + "."
            rng = sub_rng("verify", it["id"], tag)
            right = OPT_OK if tag == "correct" else OPT_BAD
            wrong = OPT_BAD if tag == "correct" else OPT_OK
            ci = rng.randrange(2)
            opts, ans = arrange_options(rng, right, [wrong], ci)
            items.append(finish_item(Item(
                id=f"{it['id']}_verify_{tag}", base_id=it["base_id"], task="log_verify_summary",
                condition={**it["condition"], "num_options": 2, "snapshot": tag},
                context=context, question=question, options=opts, answer_id=ans,
                metadata={"seed": it["metadata"]["seed"], "repeat": it["metadata"]["repeat"], "correct_index": ci,
                          "target_user": it["metadata"]["target_user"], "final_set_size": it["metadata"]["final_set_size"],
                          "summary_text": text, "true_text": correct})))
    sha = write_jsonl(items, Path(args.out))
    print(f"{len(items)} items -> {args.out} sha256 {sha[:12]}")


if __name__ == "__main__":
    main()
