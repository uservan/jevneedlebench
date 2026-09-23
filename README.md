# jevneedlebench

A controlled needle-in-a-haystack benchmark for Jev-class decision models (context + candidates → pick one).
It asks where such a model breaks as the input grows — finding a fact, choosing among many candidates, or
combining several updates into a current state — and whether the model knows when it is wrong.

Every item is a context, a question and 2–64 options; the model returns one option id, with probabilities when
it has them. Answers are computed by a program, never by a model. Results below are for **Jev 1.13.0**
(TypeSafe, `/v1/systemone`), 12,150 requests, about $6. Design notes: [DESIGN.md](DESIGN.md).

## Data

Three heatmaps share the context-length axis (1k → 20k tokens, 28k for natural text, `o200k_base` counts;
Jev's own count is ~1.45× for these records, so 20k ≈ 30k Jev tokens, close to its 32k limit).

| File | Filler | Needle | Other axis | Items |
| --- | --- | --- | --- | --- |
| `h1_retrieval` | other users' records, or Paul Graham essays | one record | needle position 10 / 50 / 90 % | 1,170 |
| `h2_selection` | other users' records | 64 records scattered at random (target + 63 candidate users) | options 2 → 64 | 1,080 |
| `h3_state` | other users' day-ordered project logs | the target user's *k* log lines | updates *k* = 1 → 16; two questions per log | 1,800 |
| `h3_snapshot` | `h3_state` logs + a correct or stale one-line state summary | same | summary correct / stale | 1,800 |
| `h3_verify` | same; question: is the summary line correct? (2 options) | same | summary correct / stale | 1,800 |
| `h3_defer` | final-state, snapshot and verify items + a hand-off option | same | — | 4,500 |

What the tests look like (contexts abbreviated):

```text
① retrieval               The access code of user_37874 is code_18405.
                          ...  the target record sits at 10% / 50% / 90% of 70–1,400 records (or essay paragraphs) ...
                          The access code of user_28195 is code_31340.        ← target
                          Q: what is the access code of user_28195?
                          code_20709 | code_31340 ✓ | code_59847 | code_18500

② selection               64 key records scattered in the haystack; 2–64 of their codes offered as options
                          Q: what is the access code of user_28391?
                          code_84078 ✓ | code_20405 | code_68467 | ... (up to 64)

③ state: event lookup     Day 38: user_17824 joined project_H.  ...  Day 188: user_17824 joined project_A.  ...
                          Q: which project did user_17824 join on day 188?
                          project_F | project_H | project_A ✓ | project_G

③ state: final state      same log; Q: after the last entry, which projects does user_17824 belong to?
                          project_E | project_A | project_A, project_E ✓ | (none)
                          (earlier state) (one update skipped)          (another user's state)

   snapshot               same log + "Summary as of the last entry: user_17824 currently belongs to: project_A, project_E."
                          (correct) or "...: project_A." (stale: last update missing); same final-state question
   verify                 same log + summary line; Q: is that summary correct?   Correct | Incorrect
   hand-off               any of the above + option "Cannot be determined reliably from the log; hand this question
                          to a stronger model." and a sentence telling the model to pick it when unsure
```

Each cell = 10 scenarios × 3 option shuffles (95% intervals about ±0.2 per cell; trends are reliable, single
cells are not). A scenario is reused across all cells with nested contexts, so cells differ only in the tested
factor. Wrong options always occur in the context; ids are random 5-digit numbers with no near-duplicates of the
target; the log contains only legal operations and every user starts with no projects. `data/manifest.json`
holds hashes and generation parameters; the shipped `data/*.jsonl.gz` is exactly `--seed 42 --bases 10 --repeats 3`.

Item format (only `context`, `question`, `options` are sent to the model):

```json
{"id": "h3_b00_final_state_L1024_k4_r0", "base_id": "h3_b00", "task": "log_final_state",
 "condition": {"haystack_type": "kv", "target_context_tokens": 1024, "num_options": 4,
               "target_position": null, "distractor_level": "none", "target_updates": 4},
 "context": "Day 1: user_69288 joined project_K.\n...", "question": "Rules: ... which projects does user_17824 belong to?",
 "options": [{"id": "option_0", "text": "project_E"}, ...], "answer_id": "option_2",
 "metadata": {"seed": 42, "repeat": 0, "correct_index": 2, "actual_context_tokens": 1020, "final_set_size": 2, ...}}
```

## Findings

### 1. Finding and selecting: no degradation up to the context limit

Retrieval (39 cells), selection (36 cells) and event lookup (30 cells) are all **100%**, with the correct option at
probability ≈ 1.0: 20k-token contexts, 64 options, needles in the middle, record haystacks and essay haystacks alike.

### 2. Combining updates: fails from two updates on

Final state — the same log, but *k* lines must be merged:

| updates \ context | 1k | 2k | 4k | 8k | 16k | 20k |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 1.00 | 1.00 | 0.93 | 0.80 | 0.60 | 0.93 |
| 4 | 1.00 | 1.00 | 1.00 | 0.90 | 0.80 | 0.73 |
| 8 | 1.00 | 0.73 | 0.77 | 0.73 | 0.77 | 0.70 |
| 16 | 0.80 | 0.63 | 0.60 | 0.53 | 0.63 | 0.57 |

Of 145 wrong answers, 134 are an earlier state of the same user and 137 are smaller than the correct set: later
updates are missed. Users are almost never confused (3 cases). The probability on the correct option falls
earlier than the accuracy (0.87 in cells that are still 100%) and reaches ≈ 0.5 at 16 updates.

### 3. A state summary: used, trusted, not checked

- Append one **correct** summary line to the log → **100% in all 30 cells** (p ≈ 0.97). Jev uses a compiled state.
- Append a **stale** line (state before the last update) → Jev follows it 61–100% of the time when it is among the
  options, more so in long logs (68% at 1k → 93% at 20k), although the log contradicts it.
- Ask *is the summary correct?* → 77% overall (chance 50%). Correct summaries accepted 91%; stale ones caught only
  62%, and at 2 updates in 16k–20k logs 0–7%.

### 4. Does it know when it is wrong? Only for its own failures, and only implicitly

`python -m test.calibration` (`report/calibration.png`), using the probability of the chosen option:

| condition | p when right | p when wrong | AUROC | ECE |
| --- | --- | --- | --- | --- |
| final state, own integration | 0.85 | 0.64 | **0.83** | 0.03 |
| &nbsp;&nbsp; 2 / 4 / 8 / 16 updates | | | 0.89 / 0.92 / 0.82 / **0.54** | |
| final state, stale summary given | 0.82 | 0.81 | **0.53** | 0.28 |
| &nbsp;&nbsp; 2 updates | 0.77 | 0.88 | **0.26** | 0.45 |
| verify a stale summary | 0.68 | 0.67 | 0.53 | 0.07 |

- For errors it makes itself the probabilities are calibrated: abstaining below p = 0.8 keeps 62% of the items at
  96% accuracy (p = 0.9: 48% at 97%). When wrong it is usually torn, and the correct option is second in 70% of cases.
- For errors inherited from a wrong summary there is no signal; at 2 updates it is *more* confident when wrong.
- The signal fades when the task is far beyond it (AUROC 0.54 at 16 updates).
- **Hand-off option** (`python -m test.defer`): given an explicit "hand this to a stronger model" option, Jev picks it
  1% of the time on final state (up to 10% at 16 updates), 0% on the snapshot and verifier tests — 3 of its 145
  wrong answers. The probability it puts on that option rises with difficulty (0.00 at 1 update → 0.13 at 16) and
  is inversely related to its own confidence (r = −0.56), but tracks how hard the item looks, not whether it is
  wrong (at 16 updates: 0.17 on items it gets right, 0.06 on items it gets wrong).

### Takeaway

Jev is a precise selector over a state someone else has compiled. It does not merge history, it trusts a summary
over the raw log, it cannot check that summary, and it will not decline a question. Its output distribution flags
its own integration errors — read `p_chosen` and route low-confidence steps elsewhere — but nothing flags a wrong
input. In an agent loop: hand it a state snapshot, not an event list; the snapshot's correctness is the ceiling.

Per-cell numbers and heatmaps: `results/jev-1.13.0/report/`; hand-off pairs: `results/jev-1.13.0_defer/`.
Latency p50 ≈ 190 ms, p95 ≈ 500 ms; no request failed except a transient 529 burst (retried).

## Usage

```bash
pip install tiktoken httpx pandas numpy matplotlib

python -m generate --seed 42 --bases 10 --repeats 3 --out data     # h1, h2, h3 + validation (exit 1 on any error)
python -m generate.snapshot && python -m generate.verify && python -m generate.defer   # derived h3 sets

python -m test.run --adapter mock --mode oracle --data data/*.jsonl.gz --out results/mock     # pipeline check, no API
JEV_API_KEY=... python -m test.run --adapter jev --data data/h1_retrieval.jsonl.gz data/h2_selection.jsonl.gz \
    data/h3_state.jsonl.gz data/h3_snapshot.jsonl.gz data/h3_verify.jsonl.gz --out results/jev-1.13.0 --workers 4
JEV_API_KEY=... python -m test.run --adapter jev --data data/h3_defer.jsonl.gz --out results/jev-1.13.0_defer

python -m test.analyze --results results/jev-1.13.0        # summary.csv + heatmaps
python -m test.calibration --results results/jev-1.13.0    # calibration.csv + calibration.png
python -m test.defer                                       # hand-off pairing
```

`test.run` resumes (done ids are skipped), never truncates a context, retries 429/503/529 with backoff, and records
`unsupported` (over the model's limit), `api_error` and `invalid` separately from wrong answers; it stops on
401/403 or three consecutive API errors. To add a model, implement `Adapter.choose(context, question, options)
-> Result` in `test/adapters/` (`openai_compat.py` is a template for chat models). All models see identical items
and option order.

## Layout

```text
generate/   common.py corpus.py retrieval.py selection.py state.py validate.py   (python -m generate)
            snapshot.py verify.py defer.py                                        (derived h3 sets)
data/       generated JSONL (gzipped) + manifest.json
test/       adapters/ (jev, mock, openai_compat), run.py, analyze.py, calibration.py, defer.py
results/    <run>/<dataset>.jsonl per-item results, <run>/report/ summary + figures
corpus/     Paul Graham essays from gkamradt/LLMTest_NeedleInAHaystack, with sha256
```
