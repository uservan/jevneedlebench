"""Heatmap 2 (selection): context length x number of options, kv haystack.

All 64 key records (target + 63 candidate users) are always in the context, scattered at
random positions that are re-drawn on every repeat; the option subset is nested
(2 uses candidate 0, 4 uses candidates 0-2, ... 64 uses all 63)."""
from __future__ import annotations

from .common import (LENGTHS, NUM_OPTIONS, Item, arrange_options, correct_positions, draw_ids,
                     finish_item, fit_prefix, ntok, sub_rng)
from .retrieval import N_FILLER, kv_record, question_for

N_CANDIDATES = NUM_OPTIONS[-1] - 1


def scatter(rng, filler: list[str], keys: list[str]):
    """Merge key records into filler at random positions. Returns (records, index_of_keys[0])."""
    total = len(filler) + len(keys)
    slots = sorted(rng.sample(range(total), len(keys)))
    recs, ki, fi = [], 0, 0
    target_index = None
    for i in range(total):
        if ki < len(keys) and i == slots[ki]:
            if ki == 0:
                target_index = i
            recs.append(keys[ki]); ki += 1
        else:
            recs.append(filler[fi]); fi += 1
    return recs, target_index


def generate(seed: int, n_bases: int, repeats: int) -> list[Item]:
    items = []
    for b in range(n_bases):
        rng = sub_rng("h2", seed, b)
        target = draw_ids(rng, 1)[0]
        taken = {target}
        c_users = draw_ids(rng, N_CANDIDATES, avoid_near=target, taken=taken)
        f_users = draw_ids(rng, N_FILLER, avoid_near=target, taken=taken)
        codes = draw_ids(rng, 1 + N_CANDIDATES + N_FILLER)
        t_code, c_codes, f_codes = codes[0], codes[1:1 + N_CANDIDATES], codes[1 + N_CANDIDATES:]
        keys = [kv_record(target, t_code)] + [kv_record(u, c) for u, c in zip(c_users, c_codes)]
        filler = [kv_record(u, c) for u, c in zip(f_users, f_codes)]
        key_tokens = sum(ntok(k) for k in keys) + len(keys)
        question = question_for(target)
        base_id = f"h2_b{b:02d}"

        for L in LENGTHS:
            n = fit_prefix(filler, L, extra_tokens=key_tokens)
            cpos = {k: correct_positions(sub_rng("h2-pos", seed, b, L, k), k, repeats) for k in NUM_OPTIONS}
            for r in range(repeats):
                recs, ti = scatter(sub_rng("h2-scatter", seed, b, L, r), filler[:n], keys)
                context = "\n".join(recs)
                for k in NUM_OPTIONS:
                    opts, ans = arrange_options(sub_rng("h2-opt", seed, b, L, k, r), f"code_{t_code}",
                                                [f"code_{c}" for c in c_codes[:k - 1]], cpos[k][r])
                    items.append(finish_item(Item(
                        id=f"{base_id}_L{L}_k{k}_r{r}", base_id=base_id, task="kv_lookup",
                        condition={"haystack_type": "kv", "target_context_tokens": L, "num_options": k,
                                   "target_position": None, "distractor_level": "none", "target_updates": 0},
                        context=context, question=question, options=opts, answer_id=ans,
                        metadata={"seed": seed, "repeat": r, "correct_index": cpos[k][r],
                                  "target_user": f"user_{target}", "n_records": len(recs),
                                  "actual_position": round(ti / max(len(recs) - 1, 1), 4)})))
    return items
