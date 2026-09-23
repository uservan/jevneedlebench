"""Run one adapter over one or more datasets, writing per-item results.

    python -m test.run --adapter jev --data data/h1_retrieval.jsonl --out results/jev
    python -m test.run --adapter mock --mode oracle --data data/*.jsonl --out results/mock

Results go to <out>/<dataset name>.jsonl, one line per item, appended and resumable (items
already present are skipped). Contexts are not copied into results. Stops on HTTP 401/403 or
three consecutive api_error results; remaining items stay unattempted.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .adapters.base import load_adapter

STOP_STATUSES = {401, 403}


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}


def record(item: dict, res) -> dict:
    correct = res.status == "ok" and res.choice_id == item["answer_id"]
    p_correct = res.probabilities.get(item["answer_id"]) if res.probabilities else None
    return {"id": item["id"], "base_id": item["base_id"], "task": item["task"], "condition": item["condition"],
            "num_options": len(item["options"]), "answer_id": item["answer_id"], "choice_id": res.choice_id,
            "status": res.status, "correct": correct, "p_correct": p_correct,
            "p_chosen": (res.probabilities or {}).get(res.choice_id) if res.choice_id else None,
            "confidence": res.confidence, "probabilities": res.probabilities,
            "usage": res.usage, "latency_ms": round(res.latency_ms, 1), "http_status": res.http_status,
            "error": res.error, "model": res.model, "constructed_tokens": item["metadata"]["total_input_tokens"],
            "metadata": {k: v for k, v in item["metadata"].items() if k in ("repeat", "correct_index", "final_set_size", "event_index")},
            "ts": time.time()}


def run_dataset(adapter, data_path: Path, out_dir: Path, workers: int, limit: int | None):
    out_path = out_dir / data_path.name.removesuffix(".gz")
    done = load_done(out_path)
    opener = gzip.open if data_path.suffix == ".gz" else open
    with opener(data_path, "rt", encoding="utf-8") as fh:
        items = [json.loads(l) for l in fh if l.strip()]
    todo = [it for it in items if it["id"] not in done][:limit]
    print(f"{data_path.name}: {len(items)} items, {len(done)} done, running {len(todo)}", flush=True)
    consecutive_errors, stopped, n_ok = 0, False, 0
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(workers) as pool:
        def work(it):
            return it, adapter.choose(it["context"], it["question"], it["options"])
        for i, (it, res) in enumerate(pool.map(work, todo), 1):
            if stopped:
                continue
            fh.write(json.dumps(record(it, res), ensure_ascii=False) + "\n"); fh.flush()
            consecutive_errors = consecutive_errors + 1 if res.status == "api_error" else 0
            n_ok += res.status == "ok"
            if res.http_status in STOP_STATUSES or consecutive_errors >= 3:
                print(f"STOP after {i}: http={res.http_status} error={res.error}", flush=True)
                stopped = True
                pool.shutdown(wait=False, cancel_futures=True)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} ok={n_ok} {time.time() - t0:.0f}s", flush=True)
    print(f"{data_path.name}: finished {len(todo)} in {time.time() - t0:.0f}s", flush=True)
    return not stopped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, choices=["jev", "mock", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", default="random", help="mock mode: random|first|oracle")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="max new items per dataset")
    args = ap.parse_args()
    kw = {}
    if args.adapter == "mock":
        kw["mode"] = args.mode
    elif args.model:
        kw["model"] = args.model
    adapter = load_adapter(args.adapter, **kw)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    for d in args.data:
        if not run_dataset(adapter, Path(d), out_dir, args.workers, args.limit):
            sys.exit(1)


if __name__ == "__main__":
    main()
