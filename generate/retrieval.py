"""Heatmap 1 (retrieval): context length x target position, kv and essay haystacks, 4 options."""
from __future__ import annotations

from .common import (ESSAY_LENGTHS, LENGTHS, POSITIONS, Item, arrange_options, correct_positions, draw_ids,
                     finish_item, fit_prefix, ntok, sub_rng)
from .corpus import sample_essay

N_FILLER = 2100  # ~31k tokens of kv records, more than the 20k maximum


def kv_record(user: str, code: str) -> str:
    return f"The access code of user_{user} is code_{code}."


def question_for(user: str) -> str:
    return f"According to the context, what is the access code of user_{user}?"


def insert_keys(filler: list[str], key_recs: list[tuple[float, str]], target_rec: str, pos: float):
    """Insert distractor records at their fractions, then the target at fraction `pos`.
    Returns (records, target_index)."""
    recs = list(filler)
    for frac, rec in sorted(key_recs):
        recs.insert(round(frac * len(recs)), rec)
    ti = round(pos * len(recs))
    recs.insert(ti, target_rec)
    return recs, ti


def generate(seed: int, n_bases: int, repeats: int) -> list[Item]:
    items = []
    for b in range(n_bases):
        rng = sub_rng("h1", seed, b)
        target = draw_ids(rng, 1)[0]
        taken = {target}
        d_users = draw_ids(rng, 3, avoid_near=target, taken=taken)
        f_users = draw_ids(rng, N_FILLER, avoid_near=target, taken=taken)
        codes = draw_ids(rng, 4 + N_FILLER)
        t_code, d_codes, f_codes = codes[0], codes[1:4], codes[4:]
        target_rec = kv_record(target, t_code)
        d_recs = [(rng.random(), kv_record(u, c)) for u, c in zip(d_users, d_codes)]
        kv_filler = [kv_record(u, c) for u, c in zip(f_users, f_codes)]
        essay_filler = sample_essay(sub_rng("h1-essay", seed, b), ESSAY_LENGTHS[-1])
        key_tokens = ntok(target_rec) + sum(ntok(r) for _, r in d_recs) + 4
        question = question_for(target)
        base_id = f"h1_b{b:02d}"

        for hay, filler, sep, lengths in (("kv", kv_filler, "\n", LENGTHS), ("essay", essay_filler, "\n\n", ESSAY_LENGTHS)):
            for L in lengths:
                n = fit_prefix(filler, L, extra_tokens=key_tokens + (0 if hay == "kv" else 4))
                for pos in POSITIONS:
                    recs, ti = insert_keys(filler[:n], d_recs, target_rec, pos)
                    context = sep.join(recs)
                    cell = (hay, L, pos)
                    cpos = correct_positions(sub_rng("h1-pos", seed, b, *cell), 4, repeats)
                    for r in range(repeats):
                        opts, ans = arrange_options(sub_rng("h1-opt", seed, b, *cell, r),
                                                    f"code_{t_code}", [f"code_{c}" for c in d_codes], cpos[r])
                        items.append(finish_item(Item(
                            id=f"{base_id}_{hay}_L{L}_p{pos}_r{r}", base_id=base_id, task="kv_lookup",
                            condition={"haystack_type": hay, "target_context_tokens": L, "num_options": 4,
                                       "target_position": pos, "distractor_level": "none", "target_updates": 0},
                            context=context, question=question, options=opts, answer_id=ans,
                            metadata={"seed": seed, "repeat": r, "correct_index": cpos[r],
                                      "target_user": f"user_{target}", "n_records": len(recs),
                                      "actual_position": round(ti / max(len(recs) - 1, 1), 4)})))
    return items
