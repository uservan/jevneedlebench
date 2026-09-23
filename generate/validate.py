"""Programmatic checks on generated items: format, lengths, uniqueness, and answers recomputed
from the context alone."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .common import near_id
from .state import RULES, simulate

KV_RE = re.compile(r"The access code of user_(\d{5}) is code_(\d{5})\.")
LOG_RE = re.compile(r"Day (\d+): user_(\d{5}) (joined|left) (project_[A-L])\.")


def check_item(it: dict) -> list[str]:
    errs = []
    opts = it["options"]
    ids = [o["id"] for o in opts]
    texts = [o["text"] for o in opts]
    if len(set(ids)) != len(ids) or len(set(texts)) != len(texts):
        errs.append("duplicate option id/text")
    if it["answer_id"] not in ids:
        errs.append("answer_id not in options")
    if len(opts) != it["condition"]["num_options"]:
        errs.append("num_options mismatch")
    for banned in ("answer_id", "correct_index"):
        if banned in it["context"] or banned in it["question"]:
            errs.append(f"leak: {banned} in prompt text")
    L = it["condition"]["target_context_tokens"]
    actual = it["metadata"]["actual_context_tokens"]
    if not (0.90 * L <= actual <= L):
        errs.append(f"context tokens {actual} outside [0.90*{L}, {L}]")
    answer_text = texts[ids.index(it["answer_id"])] if it["answer_id"] in ids else None
    target = it["metadata"]["target_user"].split("_")[1]

    if it["task"] == "kv_lookup":
        recs = KV_RE.findall(it["context"])
        users = [u for u, _ in recs]
        if len(set(users)) != len(users):
            errs.append("duplicate user in context")
        codes = [c for _, c in recs]
        if len(set(codes)) != len(codes):
            errs.append("duplicate code in context")
        hits = [c for u, c in recs if u == target]
        if len(hits) != 1:
            errs.append(f"target record count {len(hits)}")
        elif f"code_{hits[0]}" != answer_text:
            errs.append("recomputed answer != answer_id text")
        if it["condition"]["distractor_level"] == "none" and any(u != target and near_id(u, target) for u in users):
            errs.append("near-duplicate id present under distractor_level=none")
        for t in texts:
            if t[5:] not in codes:
                errs.append(f"option {t} not in context")
        pos = it["condition"]["target_position"]
        # position can only be as precise as one record slot allows
        tol = max(0.02, 0.6 / it["metadata"]["n_records"])
        if pos is not None and abs(it["metadata"]["actual_position"] - pos) > tol:
            errs.append(f"actual_position {it['metadata']['actual_position']} vs {pos}")
    else:
        events = []
        for m in LOG_RE.finditer(it["context"]):
            events.append((int(m.group(1)), m.group(2), "join" if m.group(3) == "joined" else "leave", m.group(4)))
        if len(events) != it["context"].count("\n") + 1:
            errs.append("unparsed log lines")
        days = [e[0] for e in events]
        if days != sorted(days):
            errs.append("log not day-ordered")
        per_day = Counter((e[0], e[1]) for e in events)
        if max(per_day.values()) > 1:
            errs.append("a user has two events on one day")
        # legality replay
        st: dict[str, set] = defaultdict(set)
        for _, u, op, p in events:
            if (op == "join") == (p in st[u]):
                errs.append(f"illegal op for user_{u} on {p}")
                break
            st[u].add(p) if op == "join" else st[u].discard(p)
        t_events = [e for e in events if e[1] == target]
        if len(t_events) != it["condition"]["target_updates"]:
            errs.append(f"target updates {len(t_events)} != {it['condition']['target_updates']}")
        if it["task"] == "log_event_lookup":
            m = re.search(r"which project did user_\d+ (join|leave) on day (\d+)\?", it["question"])
            hit = [e for e in t_events if e[0] == int(m.group(2)) and e[2] == m.group(1)]
            if len(hit) != 1 or hit[0][3] != answer_text:
                errs.append("recomputed event != answer")
        else:
            if not it["question"].startswith(RULES):
                errs.append("rules missing from final-state question")
            final = simulate(events).get(target, set())
            want = ", ".join(sorted(final)) if final else "(none)"
            if want != answer_text:
                errs.append(f"recomputed final state {want!r} != {answer_text!r}")
    return errs


def summarize(items: list[dict]) -> str:
    lines = []
    by_cell = defaultdict(list)
    for it in items:
        c = it["condition"]
        by_cell[(it["task"], c["haystack_type"], c["target_context_tokens"], c["num_options"],
                 c["target_position"], c["target_updates"])].append(it)
    lines.append(f"{len(items)} items, {len({i['base_id'] for i in items})} bases, {len(by_cell)} cells")
    for key in sorted(by_cell, key=str):
        its = by_cell[key]
        pos = Counter(i["metadata"]["correct_index"] for i in its)
        toks = [i["metadata"]["actual_context_tokens"] for i in its]
        extra = ""
        if key[0] == "log_final_state":
            extra = " final_set_size=" + str(dict(sorted(Counter(i["metadata"]["final_set_size"] for i in its).items())))
        lines.append(f"  {key}: n={len(its)} ctx_tokens={min(toks)}-{max(toks)} correct_pos={dict(sorted(pos.items()))}{extra}")
    return "\n".join(lines)
