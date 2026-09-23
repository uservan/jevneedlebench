"""Summaries, bootstrap CIs by base_id, and heatmaps.

    python -m test.analyze --results results/jev --out results/jev/report

Writes summary.csv (one row per cell), diagnosis.csv, and heatmap PNGs.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

CELL_KEYS = ["task", "haystack_type", "target_context_tokens", "num_options", "target_position", "target_updates"]


def load(results_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(results_dir.glob("*.jsonl")):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                c = r["condition"]
                rows.append({"dataset": p.stem, "id": r["id"], "base_id": r["base_id"], "task": r["task"],
                             "haystack_type": c["haystack_type"], "target_context_tokens": c["target_context_tokens"],
                             "num_options": c["num_options"], "target_position": c["target_position"],
                             "target_updates": c["target_updates"], "status": r["status"], "correct": bool(r["correct"]),
                             "p_correct": r.get("p_correct"), "p_chosen": r.get("p_chosen"),
                             "latency_ms": r["latency_ms"], "constructed_tokens": r["constructed_tokens"],
                             "api_input_tokens": (r.get("usage") or {}).get("input_tokens"),
                             "repeat": (r.get("metadata") or {}).get("repeat"),
                             "final_set_size": (r.get("metadata") or {}).get("final_set_size"),
                             "choice_id": r.get("choice_id")})
    df = pd.DataFrame(rows)
    df["target_position"] = df["target_position"].fillna(-1)
    return df


def bootstrap_ci(df: pd.DataFrame, n: int = 1000, seed: int = 0):
    """Accuracy CI by resampling base_ids (repeats of one base move together)."""
    rng = np.random.default_rng(seed)
    groups = [g["correct"].to_numpy() for _, g in df.groupby("base_id")]
    if not groups:
        return (math.nan, math.nan)
    accs = []
    for _ in range(n):
        idx = rng.integers(0, len(groups), len(groups))
        accs.append(np.concatenate([groups[i] for i in idx]).mean())
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def brier(df: pd.DataFrame) -> float:
    d = df[df["status"] == "ok"].dropna(subset=["p_correct"])
    if d.empty:
        return math.nan
    # multi-class Brier with one-hot target, assuming remaining mass is on wrong options
    pc = d["p_correct"].to_numpy()
    return float(np.mean((1 - pc) ** 2 + (1 - pc) ** 2 / np.maximum(d["num_options"].to_numpy() - 1, 1)))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(CELL_KEYS, dropna=False):
        cell = dict(zip(CELL_KEYS, key))
        ok = g[g["status"] == "ok"]
        scored = g[g["status"].isin(["ok", "invalid"])]   # unsupported / api_error / unattempted not counted as wrong
        k = int(cell["num_options"])
        acc = scored["correct"].mean() if len(scored) else math.nan
        lo, hi = bootstrap_ci(scored) if len(scored) else (math.nan, math.nan)
        wrong = ok[~ok["correct"]]
        # order sensitivity: bases whose repeats disagree on correctness
        flips = ok.groupby("base_id")["correct"].nunique()
        rows.append({**cell, "n": len(g), "n_scored": len(scored), "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
                     "chance": 1 / k, "acc_above_chance": (acc - 1 / k) / (1 - 1 / k) if not math.isnan(acc) else math.nan,
                     "mean_p_correct": ok["p_correct"].mean(), "wrong_confidence": wrong["p_chosen"].mean(),
                     "brier": brier(ok), "order_flip_rate": float((flips > 1).mean()) if len(flips) else math.nan,
                     "latency_p50_ms": g["latency_ms"].median(), "latency_p95_ms": g["latency_ms"].quantile(0.95),
                     "constructed_tokens": g["constructed_tokens"].mean(), "api_input_tokens": g["api_input_tokens"].mean(),
                     "rate_api_error": (g["status"] == "api_error").mean(), "rate_unsupported": (g["status"] == "unsupported").mean(),
                     "rate_invalid": (g["status"] == "invalid").mean()})
    return pd.DataFrame(rows).sort_values(CELL_KEYS).reset_index(drop=True)


def state_conditional(df: pd.DataFrame) -> pd.DataFrame:
    """P(final_state correct | event_lookup correct) per (length, updates), paired by base and repeat."""
    ev = df[df["task"] == "log_event_lookup"].set_index(["base_id", "target_context_tokens", "target_updates", "repeat"])["correct"]
    fs = df[df["task"] == "log_final_state"].set_index(["base_id", "target_context_tokens", "target_updates", "repeat"])["correct"]
    j = pd.concat({"event": ev, "final": fs}, axis=1).dropna()
    if j.empty:
        return pd.DataFrame()
    j = j.reset_index()
    out = j.groupby(["target_context_tokens", "target_updates"]).apply(
        lambda g: pd.Series({"n": len(g), "p_event": g["event"].mean(), "p_final": g["final"].mean(),
                             "p_final_given_event": g[g["event"]]["final"].mean() if g["event"].any() else math.nan,
                             "p_both_wrong": (~g["event"] & ~g["final"]).mean(),
                             "p_event_ok_final_wrong": (g["event"] & ~g["final"]).mean()}), include_groups=False)
    return out.reset_index()


def heatmap(summary: pd.DataFrame, row_key: str, col_key: str, title: str, path: Path, value="accuracy"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    piv = summary.pivot(index=row_key, columns=col_key, values=value)
    fig, ax = plt.subplots(figsize=(1.2 * len(piv.columns) + 2, 0.6 * len(piv.index) + 1.5))
    im = ax.imshow(piv.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax.set_xlabel(col_key); ax.set_ylabel(row_key); ax.set_title(title)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.iat[i, j]
            if not (isinstance(v, float) and math.isnan(v)):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label=value)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rdir = Path(args.results)
    out = Path(args.out or rdir / "report"); out.mkdir(parents=True, exist_ok=True)
    df = load(rdir)
    print(f"{len(df)} results; status counts: {df['status'].value_counts().to_dict()}")
    summary = summarize(df)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary[CELL_KEYS + ["n_scored", "accuracy", "ci_lo", "ci_hi", "mean_p_correct", "order_flip_rate"]].to_string(index=False))

    h1 = summary[(summary["task"] == "kv_lookup") & (summary["target_position"] >= 0)]
    for hay, g in h1.groupby("haystack_type"):
        heatmap(g, "target_position", "target_context_tokens", f"H1 retrieval ({hay}): accuracy", out / f"h1_retrieval_{hay}.png")
    h2 = summary[(summary["task"] == "kv_lookup") & (summary["target_position"] < 0)]
    if len(h2):
        heatmap(h2, "num_options", "target_context_tokens", "H2 selection: accuracy", out / "h2_selection.png")
        heatmap(h2, "num_options", "target_context_tokens", "H2 selection: accuracy above chance", out / "h2_selection_above_chance.png", value="acc_above_chance")
    for task in ("log_event_lookup", "log_final_state"):
        g = summary[summary["task"] == task]
        if len(g):
            heatmap(g, "target_updates", "target_context_tokens", f"H3 {task}: accuracy", out / f"h3_{task}.png")
    cond = state_conditional(df)
    if len(cond):
        cond.to_csv(out / "h3_conditional.csv", index=False)
        heatmap(cond, "target_updates", "target_context_tokens", "H3 P(final correct | event correct)", out / "h3_final_given_event.png", value="p_final_given_event")
        print(cond.to_string(index=False))
    print("report written to", out)


if __name__ == "__main__":
    main()
