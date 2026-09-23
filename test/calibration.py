"""Does the model know when it is wrong? Confidence analysis over per-item results.

    python -m test.calibration --results results/jev-1.13.0

For each condition: mean probability of the chosen option when right vs wrong, AUROC of that
probability as a predictor of correctness, ECE (10 bins), and selective accuracy when abstaining
below a threshold. Writes calibration.csv and calibration.png into <results>/report/."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def auroc(y, s) -> float:
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if y.all() or (~y).all():
        return math.nan
    order = np.argsort(s, kind="stable"); ranks = np.empty(len(s)); s_sorted = s[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2; i = j + 1
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(y, s, bins: int = 10) -> float:
    y = np.asarray(y, float); s = np.asarray(s, float)
    idx = np.minimum((s * bins).astype(int), bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - s[m].mean())
    return float(e)


def load(results_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(results_dir.glob("*.jsonl")):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); c = r["condition"]
                if r["status"] != "ok" or r.get("p_chosen") is None:
                    continue
                rows.append({"task": r["task"], "snapshot": c.get("snapshot", "none"), "k": c["target_updates"],
                             "L": c["target_context_tokens"], "correct": bool(r["correct"]), "p_chosen": float(r["p_chosen"])})
    return pd.DataFrame(rows)


def stats(name: str, d: pd.DataFrame) -> dict:
    right, wrong = d[d.correct], d[~d.correct]
    return {"condition": name, "n": len(d), "accuracy": d.correct.mean(),
            "p_when_right": right.p_chosen.mean(), "p_when_wrong": wrong.p_chosen.mean() if len(wrong) else math.nan,
            "auroc": auroc(d.correct, d.p_chosen), "ece": ece(d.correct, d.p_chosen)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    rdir = Path(args.results); out = rdir / "report"; out.mkdir(exist_ok=True)
    df = load(rdir)
    fs = df[(df.task == "log_final_state") & (df.snapshot == "none")]
    stale = df[(df.task == "log_final_state") & (df.snapshot == "stale")]
    ver = df[df.task == "log_verify_summary"]
    rows = [stats("final_state, own integration", fs)]
    rows += [stats(f"final_state, own integration, k={k}", fs[fs.k == k]) for k in sorted(fs.k.unique())]
    rows.append(stats("final_state, stale summary given", stale))
    rows += [stats(f"final_state, stale summary given, k={k}", stale[stale.k == k]) for k in sorted(stale.k.unique())]
    rows.append(stats("verify summary (all)", ver))
    rows.append(stats("verify summary, summary stale", ver[ver.snapshot == "stale"]))
    rows.append(stats("verify summary, summary correct", ver[ver.snapshot == "correct"]))
    table = pd.DataFrame(rows)
    table.to_csv(out / "calibration.csv", index=False)
    print(table.round(3).to_string(index=False))

    sel = []
    for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        keep = fs[fs.p_chosen >= t]
        sel.append({"threshold": t, "coverage": len(keep) / len(fs), "accuracy": keep.correct.mean()})
    sel = pd.DataFrame(sel); print("\nselective answering (final_state, own integration):"); print(sel.round(3).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    # reliability diagram
    ax = axes[0]
    for name, d, style in (("own integration", fs, "o-"), ("stale summary given", stale, "s--"), ("verify (stale half)", ver[ver.snapshot == "stale"], "^:")):
        bins = np.linspace(0, 1, 11); idx = np.minimum((d.p_chosen.to_numpy() * 10).astype(int), 9)
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 5:
                xs.append(d.p_chosen.to_numpy()[m].mean()); ys.append(d.correct.to_numpy()[m].mean())
        ax.plot(xs, ys, style, label=name)
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8); ax.set_xlabel("probability of chosen option"); ax.set_ylabel("actual accuracy")
    ax.set_title("Reliability"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    # AUROC by k
    ax = axes[1]
    ks = sorted(fs.k.unique())
    ax.plot(ks, [auroc(fs[fs.k == k].correct, fs[fs.k == k].p_chosen) for k in ks], "o-", label="own integration")
    ax.plot(ks, [auroc(stale[stale.k == k].correct, stale[stale.k == k].p_chosen) for k in ks], "s--", label="stale summary given")
    ax.axhline(0.5, color="gray", lw=0.8); ax.set_xscale("log", base=2); ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("target updates"); ax.set_ylabel("AUROC (confidence predicts correctness)"); ax.set_ylim(0.2, 1); ax.set_title("Does it know when it is wrong?"); ax.legend(fontsize=8)
    # selective accuracy
    ax = axes[2]
    ax.plot(sel.coverage, sel.accuracy, "o-")
    for _, r in sel.iterrows():
        ax.annotate(f"p≥{r.threshold:.2f}", (r.coverage, r.accuracy), fontsize=7, xytext=(3, -9), textcoords="offset points")
    ax.set_xlabel("coverage (fraction answered)"); ax.set_ylabel("accuracy on answered"); ax.set_title("Abstain below a threshold (own integration)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0.8, 1.01)
    fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=150)
    print("written", out / "calibration.csv", out / "calibration.png")


if __name__ == "__main__":
    main()
