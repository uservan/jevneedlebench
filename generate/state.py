"""Heatmap 3 (state tracking): log length x number of target updates.

A day-ordered membership log. Every user starts with no projects; only legal operations are
generated (no duplicate join, no leave of a project the user is not in). Length is scaled by
adding other users' logs (a nested prefix of a fixed user list), so target events keep their
day numbers across lengths. Two questions share each context: event lookup and final state."""
from __future__ import annotations

import random

from .common import (LENGTHS, NUM_UPDATES, PROJECTS, Item, arrange_options, correct_positions,
                     draw_ids, finish_item, fit_prefix, ntok, sub_rng)

N_FILLER_USERS = 700   # ~2,400 events, above the 20k maximum
D_MAX = 300            # days are 1..D_MAX
MAX_TARGET_UPDATES = NUM_UPDATES[-1]

RULES = ("Rules: joining a project adds it to the user's current projects and keeps the others; "
         "leaving removes only that project; a user's projects do not change on days with no entry "
         "about that user. Every user starts with no projects.")


def log_line(day: int, user: str, op: str, proj: str) -> str:
    verb = "joined" if op == "join" else "left"
    return f"Day {day}: user_{user} {verb} {proj}."


def legal_sequence(rng: random.Random, n: int, max_size: int = 6) -> list[tuple[str, str]]:
    """n legal (op, project) steps from the empty set."""
    state, seq = set(), []
    for _ in range(n):
        if not state or (len(state) < max_size and rng.random() < 0.6):
            p = rng.choice([q for q in PROJECTS if q not in state]); state.add(p); seq.append(("join", p))
        else:
            p = rng.choice(sorted(state)); state.remove(p); seq.append(("leave", p))
    return seq


def simulate(events: list[tuple[int, str, str, str]]) -> dict[str, set]:
    """events: (day, user, op, proj) in log order. Illegal ops are no-ops (only used for distractors)."""
    st: dict[str, set] = {}
    for _, u, op, p in events:
        s = st.setdefault(u, set())
        s.add(p) if op == "join" else s.discard(p)
    return st


def set_text(s) -> str:
    return ", ".join(sorted(s)) if s else "(none)"


def final_state_distractors(target: str, t_events, ctx_events, rng: random.Random):
    """3 wrong sets in priority order: earlier target states, skip-one-update states,
    other users' final states, one-project perturbations. Returns (texts, sources)."""
    correct = frozenset(simulate(t_events)[target])
    cands: list[tuple[frozenset, str]] = []
    for i in range(len(t_events)):
        cands.append((frozenset(simulate(t_events[:i]).get(target, set())), f"previous_state_{i}"))
    for j in range(len(t_events)):
        cands.append((frozenset(simulate(t_events[:j] + t_events[j + 1:]).get(target, set())), f"skip_update_{j}"))
    others = simulate(ctx_events)
    for u in sorted(others):
        if u != target:
            cands.append((frozenset(others[u]), f"other_user_{u}"))
    pert = []
    for p in PROJECTS:
        pert.append((correct | {p}, f"perturb_add_{p}") if p not in correct else (correct - {p}, f"perturb_remove_{p}"))
    rng.shuffle(pert)
    cands += [(frozenset(s), src) for s, src in pert]
    out, seen = [], {correct}
    for s, src in cands:
        if s not in seen:
            seen.add(s); out.append((set_text(s), src))
        if len(out) == 3:
            break
    return [t for t, _ in out], [s for _, s in out]


def generate(seed: int, n_bases: int, repeats: int) -> list[Item]:
    items = []
    for b in range(n_bases):
        rng = sub_rng("h3", seed, b)
        target = draw_ids(rng, 1)[0]
        taken = {target}
        f_users = draw_ids(rng, N_FILLER_USERS, avoid_near=target, taken=taken)
        t_seq = legal_sequence(rng, MAX_TARGET_UPDATES)
        # filler users: each a legal sequence on distinct random days. Flattened user by user so
        # that any prefix of f_events is legal (a prefix of a legal per-user sequence is legal).
        f_events: list[tuple[int, str, str, str]] = []
        for u in f_users:
            n = rng.choice([1, 2, 3, 4, 5, 6])
            days = sorted(rng.sample(range(1, D_MAX + 1), n))
            f_events += [(d, u, op, p) for d, (op, p) in zip(days, legal_sequence(rng, n))]
        f_lines = [log_line(*e) for e in f_events]
        base_id = f"h3_b{b:02d}"

        for k in NUM_UPDATES:
            t_days = [round((i + 0.5) / k * D_MAX) for i in range(k)]
            t_events = [(d, target, op, p) for d, (op, p) in zip(t_days, t_seq[:k])]
            t_tokens = sum(ntok(log_line(*e)) + 1 for e in t_events)
            e_idx = k // 2
            e_day, _, e_op, e_proj = t_events[e_idx]
            for L in LENGTHS:
                m = fit_prefix(f_lines, L, extra_tokens=t_tokens)
                ctx_events = sorted(f_events[:m] + t_events, key=lambda e: (e[0], e[1] == target, e[1]))
                context = "\n".join(log_line(*e) for e in ctx_events)
                projects_in_log = sorted({e[3] for e in ctx_events})
                drng = sub_rng("h3-distract", seed, b, k, L)
                fs_texts, fs_sources = final_state_distractors(target, t_events, ctx_events, drng)
                el_texts = drng.sample([p for p in projects_in_log if p != e_proj], 3)
                final = simulate(t_events)[target]
                qs = {
                    "log_event_lookup": (f"According to the log, which project did user_{target} "
                                         f"{'join' if e_op == 'join' else 'leave'} on day {e_day}?",
                                         e_proj, el_texts, {"event_index": e_idx, "event_day": e_day}),
                    "log_final_state": (f"{RULES} After the last log entry, which projects does user_{target} belong to?",
                                        set_text(final), fs_texts,
                                        {"final_set_size": len(final), "distractor_sources": fs_sources}),
                }
                for task, (question, correct, distractors, extra) in qs.items():
                    cpos = correct_positions(sub_rng("h3-pos", seed, b, k, L, task), 4, repeats)
                    for r in range(repeats):
                        opts, ans = arrange_options(sub_rng("h3-opt", seed, b, k, L, task, r), correct, distractors, cpos[r])
                        items.append(finish_item(Item(
                            id=f"{base_id}_{task[4:]}_L{L}_k{k}_r{r}", base_id=base_id, task=task,
                            condition={"haystack_type": "kv", "target_context_tokens": L, "num_options": 4,
                                       "target_position": None, "distractor_level": "none", "target_updates": k},
                            context=context, question=question, options=opts, answer_id=ans,
                            metadata={"seed": seed, "repeat": r, "correct_index": cpos[r],
                                      "target_user": f"user_{target}", "n_records": len(ctx_events),
                                      "n_users": len({e[1] for e in ctx_events}), "target_days": t_days, **extra})))
    return items
