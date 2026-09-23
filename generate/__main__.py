"""Generate and validate the three heatmap datasets.

    python -m generate --seed 42 --bases 10 --repeats 3 --out data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from . import retrieval, selection, state
from .common import ROOT, write_jsonl
from .validate import check_item, summarize

HEATMAPS = {"h1_retrieval": retrieval, "h2_selection": selection, "h3_state": state}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bases", type=int, default=10, help="scenarios per cell")
    ap.add_argument("--repeats", type=int, default=3, help="option shuffles per scenario")
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*", choices=list(HEATMAPS), default=list(HEATMAPS))
    args = ap.parse_args()
    out = ROOT / args.out
    corpus_hash = hashlib.sha256((ROOT / "corpus" / "paulgraham.sha256").read_bytes()).hexdigest()
    manifest = {"seed": args.seed, "bases": args.bases, "repeats": args.repeats,
                "tokenizer": "tiktoken/o200k_base", "corpus_sha256_list": corpus_hash,
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    failed = 0
    for name in args.only:
        t0 = time.time()
        items = HEATMAPS[name].generate(args.seed, args.bases, args.repeats)
        dicts = [asdict(i) for i in items]
        errs = [(it["id"], e) for it in dicts for e in check_item(it)]
        path = out / f"{name}.jsonl"
        sha = write_jsonl(items, path)
        order_free = hashlib.sha256("\n".join(sorted(i.to_json() for i in items)).encode()).hexdigest()
        manifest["files"][path.name] = {"n_items": len(items), "sha256": sha, "dataset_hash": order_free,
                                        "validation_errors": len(errs)}
        print(f"== {name}: {len(items)} items in {time.time() - t0:.1f}s, {len(errs)} validation errors")
        for iid, e in errs[:20]:
            print(f"   {iid}: {e}")
        failed += len(errs)
        print(summarize(dicts))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest written; total validation errors:", failed)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
