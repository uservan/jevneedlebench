"""Pair the hand-off-option run with the baseline run: does Jev abstain when it would have been wrong?

    python -m test.defer --baseline results/jev-1.13.0 --defer results/jev-1.13.0_defer

Per item: baseline outcome (right / wrong) x defer-run outcome (right / handed off / wrong)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_results(d: Path) -> dict:
    out = {}
    for p in d.glob("*.jsonl"):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/jev-1.13.0")
    ap.add_argument("--defer", default="results/jev-1.13.0_defer")
    ap.add_argument("--data", default="data/h3_defer.jsonl")
    args = ap.parse_args()
    base = load_results(Path(args.baseline)); new = load_results(Path(args.defer))
    meta = {}
    with open(args.data, encoding="utf-8") as fh:
        for l in fh:
            it = json.loads(l); meta[it["id"]] = it["metadata"]
    rows = []
    for iid, r in new.items():
        if r["status"] != "ok":
            continue
        m = meta[iid]; b = base.get(m["source_id"])
        if b is None:
            continue
        handed = r["choice_id"] == m["defer_id"]
        c = r["condition"]
        cond = {"log_final_state": "final state" if c.get("snapshot", "none") == "none" else f"snapshot {c['snapshot']}",
                "log_verify_summary": f"verify, summary {c.get('snapshot')}"}[r["task"]]
        rows.append({"cond": cond, "k": c["target_updates"], "L": c["target_context_tokens"],
                     "base_right": bool(b["correct"]), "base_p": b["p_chosen"],
                     "now": "handed off" if handed else ("right" if r["correct"] else "wrong"),
                     "p_defer": (r["probabilities"] or {}).get(m["defer_id"]), "p_chosen": r["p_chosen"]})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"{len(df)} paired items\n")
    for cond, g in df.groupby("cond", sort=False):
        print(f"== {cond}  (n={len(g)}, baseline acc {g.base_right.mean():.2f})")
        ct = pd.crosstab(g.base_right.map({True: "baseline right", False: "baseline wrong"}), g.now, normalize="index").round(2)
        cnt = pd.crosstab(g.base_right.map({True: "baseline right", False: "baseline wrong"}), g.now)
        print(pd.concat({"share": ct, "count": cnt}, axis=1).to_string())
        ans = g[g.now != "handed off"]
        print(f"   hand-off rate {(g.now == 'handed off').mean():.2f} | accuracy on answered {ans.now.eq('right').mean():.2f} (baseline {g.base_right.mean():.2f})"
              f" | baseline confidence of handed-off items {g[g.now == 'handed off'].base_p.mean():.2f} vs answered {ans.base_p.mean():.2f}\n")
    fs = df[df.cond == "final state"]
    print("final state: hand-off rate by updates x length")
    print(fs.assign(h=fs.now == "handed off").pivot_table(index="k", columns="L", values="h", aggfunc="mean").round(2).to_string())
    print("\nfinal state: accuracy on answered by updates x length")
    a = fs[fs.now != "handed off"]
    print(a.assign(r=a.now == "right").pivot_table(index="k", columns="L", values="r", aggfunc="mean").round(2).to_string())
    df.to_csv(Path(args.defer) / "defer_pairs.csv", index=False)


if __name__ == "__main__":
    main()
