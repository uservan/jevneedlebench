# jevneedlebench

A controlled needle-in-a-haystack benchmark for Jev-class decision models (context + candidates → pick one).
It separates three ways a decision model can fail as the input grows: it cannot **find** the fact,
it cannot **select** among many candidates, or it cannot **integrate** several updates into a current state.

Every item is a context, a question and 2–64 options; the model returns one option id.
Answers are computed by a program. Design notes: [DESIGN.md](DESIGN.md).

## Data

Three heatmaps share the context-length axis (1k → 20k tokens, 28k for natural text; `o200k_base` counts).

| File | Filler | Needle | Other axis | Items |
| --- | --- | --- | --- | --- |
| `data/h1_retrieval.jsonl.gz` | other users' records, or Paul Graham essays | `The access code of user_28195 is code_31340.` | needle position 10 / 50 / 90 % | 1,170 |
| `data/h2_selection.jsonl.gz` | other users' records | 64 records scattered at random (target + 63 candidate users) | options 2 → 64 | 1,080 |
| `data/h3_state.jsonl.gz` | other users' day-ordered project logs | the target user's *k* lines: `Day 38: user_17824 joined project_H.` | updates *k* = 1 → 16 | 1,800 |
| `data/h3_snapshot.jsonl.gz` | as above + a correct or stale state summary line | same | snapshot correct / stale | 1,800 |
| `data/h3_verify.jsonl.gz` | as above; question: is the summary line correct? (2 options) | same | snapshot correct / stale | 1,800 |

`h3` asks two questions per log: **event lookup** (which project did the user join on day N; one line)
and **final state** (which projects does the user belong to at the end; *k* lines to merge).
Wrong final-state options are earlier states, one-update-skipped states, or other users' states.

Each cell = 10 scenarios × 3 option shuffles. A scenario is reused across all cells with nested contexts, so
cells differ only in the tested factor. Wrong options always occur in the context; ids are random 5-digit
numbers with no near-duplicates of the target. `data/manifest.json` holds hashes and generation parameters.

What each test looks like (contexts abbreviated):

```text
① retrieval (kv)          The access code of user_37874 is code_18405.
                          ...  the target record sits at 10% / 50% / 90% of 70–1,400 records ...
                          The access code of user_28195 is code_31340.        ← target
                          Q: what is the access code of user_28195?
                          code_20709 | code_31340 ✓ | code_59847 | code_18500

① retrieval (essay)       The world then was divided into two groups, grownups and kids. ...
                          The access code of user_28195 is code_31340.        ← inserted between paragraphs
                          Symbols differ from strings in that you can test equality ...

② selection               64 key records scattered in the haystack; 2–64 of their codes offered as options
                          Q: what is the access code of user_28391?
                          code_84078 ✓ | code_20405 | code_68467 | ... (up to 64)

③ state: event lookup     Day 38: user_17824 joined project_H.   ... Day 188: user_17824 joined project_A. ...
                          Q: which project did user_17824 join on day 188?
                          project_F | project_H | project_A ✓ | project_G

③ state: final state      same log; Q: after the last entry, which projects does user_17824 belong to?
                          project_E | project_A | project_A, project_E ✓ | (none)
                          (earlier state) (one update skipped)          (another user's state)

③ snapshot control        same log + "Summary as of the last entry: user_17824 currently belongs to: project_A, project_E."
                          (correct) or "...: project_A." (stale: last update missing); same final-state question

③ verifier                same log + summary line; Q: is that summary correct?
                          Correct: the summary matches ... | Incorrect: the summary does not match ...
```

Item format:

```json
{"id": "h3_b00_final_state_L1024_k4_r0", "base_id": "h3_b00", "task": "log_final_state",
 "condition": {"haystack_type": "kv", "target_context_tokens": 1024, "num_options": 4,
               "target_position": null, "distractor_level": "none", "target_updates": 4},
 "context": "Day 1: user_69288 joined project_K.\n...", "question": "Rules: ... which projects does user_17824 belong to?",
 "options": [{"id": "option_0", "text": "project_E"}, ...], "answer_id": "option_2",
 "metadata": {"seed": 42, "repeat": 0, "correct_index": 2, "actual_context_tokens": 1020, "final_set_size": 2, ...}}
```

Only `context`, `question` and `options` are sent to the model.

## Observations (Jev 1.13.0, 7,650 requests, all `ok`)

- **Retrieval and selection: 100% in all 75 cells**, correct-option probability 1.0 — up to 20k-token contexts
  (≈30k Jev tokens), 64 options, any needle position, both haystack types.
- **Event lookup: 100% in all 30 cells.** Finding one line in a 20k log is not a problem.
- **Final state degrades:** 100% with 1 update; 53–63% with 16 updates. It falls faster with the number of
  updates than with context length.

| updates \ context | 1k | 2k | 4k | 8k | 16k | 20k |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 1.00 | 1.00 | 0.93 | 0.80 | 0.60 | 0.93 |
| 4 | 1.00 | 1.00 | 1.00 | 0.90 | 0.80 | 0.73 |
| 8 | 1.00 | 0.73 | 0.77 | 0.73 | 0.77 | 0.70 |
| 16 | 0.80 | 0.63 | 0.60 | 0.53 | 0.63 | 0.57 |

- **Errors are stale states:** of 145 wrong final-state answers, 134 are an earlier state of the same user and
  137 are smaller than the correct set — later updates were missed. Users are almost never confused (3 cases).
- **Snapshot control** (`data/h3_snapshot.jsonl.gz`, `python -m generate.snapshot`): the same final-state items with one
  line appended to the log, `Summary as of the last entry: user_X currently belongs to: ...`.
  - *Correct* snapshot: **100% in all 30 cells** (p ≈ 0.97). Jev uses a compiled state.
  - *Stale* snapshot (the state before the last update): when the stale set is among the options Jev follows the
    snapshot 61–100% of the time, more so in long logs (68% at 1k → 93% at 20k), even though the log contradicts it.
- **Verifier test** (`data/h3_verify.jsonl.gz`, `python -m generate.verify`): same logs and summary line, but Jev is asked
  *is the summary correct?* (2 options). Overall 77% (chance 50%). Correct summaries are accepted 91%; stale ones are
  caught only 62%, and at 2 updates in 16k–20k logs 0–7% — it accepts the stale summary. Jev is not a usable checker
  for its own state input.
- **Does it know when it is wrong?** (`python -m test.calibration`, `report/calibration.png`). Every answer comes with
  probabilities, so we ask whether the chosen option's probability separates right from wrong.
  - Own integration errors (final state, no summary line): yes. p = 0.85 when right vs 0.64 when wrong, AUROC 0.83,
    ECE 0.03 — the probabilities are calibrated. Abstaining below p = 0.8 keeps 62% of items at 96% accuracy; below 0.9,
    48% at 97%. The signal fades when the task is far beyond it: AUROC 0.89 / 0.92 / 0.82 / 0.54 at 2 / 4 / 8 / 16 updates.
  - Misled by a stale summary: no. p = 0.82 when right vs 0.81 when wrong, AUROC 0.53, ECE 0.28; at 2 updates it is
    *more* confident when wrong (0.88 vs 0.77, AUROC 0.26). Judging whether a stale summary is correct: AUROC 0.53.
- **Takeaway:** Jev finds and selects reliably; it does not merge history, it trusts a summary over the raw log, and
  it cannot reliably check that summary either. Its confidence flags its own integration failures (usable as an
  abstain signal) but not a wrong input.
  In an agent loop, hand it a compiled state snapshot, not an appended event list — and the snapshot's correctness
  is the system's ceiling.

Per-cell numbers and heatmaps: `results/jev-1.13.0/report/`. Latency p50 ≈ 190 ms; cost ≈ $2 for the whole run.
Caveat: 10 scenarios per cell, so trends are reliable and individual cells are ±0.2.

## Usage

```bash
pip install tiktoken httpx pandas numpy matplotlib

python -m generate --seed 42 --bases 10 --repeats 3 --out data        # regenerate + validate (the shipped data/*.jsonl.gz is this exact run)
python -m test.run --adapter mock --mode oracle --data data/*.jsonl.gz --out results/mock   # pipeline check, no API
JEV_API_KEY=... python -m test.run --adapter jev --data data/*.jsonl.gz --out results/jev-1.13.0 --workers 4
python -m test.analyze --results results/jev-1.13.0                    # summary.csv + heatmaps
```

`test.run` resumes (done ids are skipped), never truncates a context, and records `unsupported`
(over the model's limit), `api_error` and `invalid` separately from wrong answers.
To add a model, implement `Adapter.choose(context, question, options) -> Result` in `test/adapters/`
(`openai_compat.py` is a template for chat models). All models see identical items and option order.

## Layout

```text
generate/   data generation and validation (common, corpus, retrieval, selection, state, validate)
data/       generated JSONL (gzipped) + manifest.json
test/       adapters/ (jev, mock, openai_compat), run.py, analyze.py, calibration.py
results/    <model>/<dataset>.jsonl per-item results, <model>/report/ summary + heatmaps
corpus/     Paul Graham essays from gkamradt/LLMTest_NeedleInAHaystack, with sha256
```
