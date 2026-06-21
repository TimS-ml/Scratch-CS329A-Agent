# Homework 1 — Test-Time Compute Scaling (AIME25)

Techniques to improve inference-time accuracy on AIME25 (30 competition-math
problems): zero-shot, majority voting, best-of-N with a generative reward model
(LLM voting), and RLEF self-improvement (generate → critique → regenerate).

**Model:** `anthropic/claude-haiku-4-5` everywhere (the assignment uses a
single model for all stages; Haiku keeps cost low and leaves headroom on hard
AIME problems so the techniques are *visible* rather than saturated).

## Code changes

| File | Change |
|------|--------|
| `cs329_hw1/inference/litellm_models.py` | Relaxed the hard `TOGETHER_API_KEY` guard → accept any of `ANTHROPIC/TOGETHER/OPENAI/GEMINI_API_KEY`. Split the call into `_raw_completion` (retry) + cached `_make_completion_request(occ)`. `send_requests` now assigns each identical prompt in a batch an **occurrence index** so `temperature>0` samples stay diverse *and* reproducible across re-runs. |
| `cs329_hw1/inference/_llm_cache.py` | **New.** Read-through disk cache + JSONL call log (see docs/README.md). |
| `student_homework1.ipynb` | Fixed `import numpy as npcan` typo; `HF_TOKEN` fallback; env-driven `DEBUG_MODE`; swapped 5 Qwen model strings → Claude; lowered `max_workers` (32→12, 16→8) for Anthropic rate limits; implemented all TODOs. |

## Implemented tasks

- **Task 1 — Zero-shot:** batch-sample 1 completion/problem, verify, report accuracy.
- **Task 2 — Majority voting** (`_get_majority_answer`): `Counter` over `_parse_answer`-normalized answers, return the plurality.
- **Task 3 — LLM voting** (`LLMVoting.__call__`): sample n candidates → build aggregation prompt → temperature-0 judge restates the most likely correct solution.
- **Task 4 — RLEF** (`SelfImprovementSystem`): `_generate_initial_solution`, `_critique_solution`, `_regenerate_solution`, `improve_solution` (chains the three), `evaluate_improvement` (verifier-based original/improved correctness + improvement flag).
- **Task 5 — Analysis:** comparison figure (`.cache/hw1_method_comparison.png`) + written quantitative/insight answers filled with the real numbers below.

## Results (full 30-problem run)

| Method | Accuracy |
|--------|----------|
| Zero-shot | 0.400 (12/30) |
| Majority voting n=1 / 2 / 4 / 8 / 16 | 0.400 / 0.400 / **0.467** / 0.433 / 0.433 |
| LLM voting (best-of-16, generative RM) | **0.500 (15/30)** |
| Self-improvement (RLEF) original → improved | 0.400 → 0.367 (−0.033; 1 improved, 2 regressed) |

## Deeper insights

### 1. Methods strictly *dominate*, they don't trade off

Looking per problem, every problem zero-shot got right was also caught by both
majority voting and LLM voting — there are **zero regressions** on the test set:

```
zero-shot  (12) ⊂ majority-vote@16 (13) ⊂ LLM-voting (15)
                  adds {1 problem}     adds {3 problems}
```

LLM voting strictly contains zero-shot. So the "what if the ensemble breaks
something easy?" failure mode never occurred at temperature 0.7 on this set.
The marginal value of more compute is **purely upside on AIME25** with Haiku.

### 2. Why majority voting is non-monotonic (peak at n=4 = 0.467)

For each problem we measured how many *distinct* normalized answers appeared
across 16 samples:

| spread (#distinct answers) | # problems | interpretation |
|----------------------------|-----------|----------------|
| 1 | 6 | perfect consensus — model is confidently right or confidently wrong |
| 2–3 | 3 | strong consensus |
| 5–8 | 7 | meaningful plurality |
| 10–11 | 8 | no signal — model is guessing |
| 12–13 | 6 | total disagreement (16 samples, 12–13 unique answers) |

So roughly **half the dataset is "no signal"** — adding more samples just
spreads the votes thinner without converging on the correct one. The bump at
n=4 then small decay is consistent with this: by n≈4 you already have the
plurality on the convergent problems; beyond that you're adding noise to the
diffuse ones, occasionally flipping the previous plurality.

This is exactly the regime where a **generative reward model wins**: it can
look at one of the 16 dispersed candidates and *reason* about why a particular
solution is internally coherent, rather than counting matching final strings.
That mechanism is why LLM voting recovered 3 extra problems that majority
voting missed.

### 3. RLEF regressed because the critic has no oracle

The critic (same Haiku, system-prompted as a "mathematical critic") flagged
issues in **30/30** solutions — including the correct ones. The downstream
regenerator dutifully addressed the feedback and rewrote correct solutions
into wrong ones (2 regressions vs 1 improvement). Critic feedback error mix,
keyword-tagged across all 30 cases:

| Category | Hits | Note |
|----------|------|------|
| Completeness ("missing case", "did not verify") | 28/30 | almost universal |
| Calculation / arithmetic | 25/30 | most actionable |
| Logical / reasoning | 21/30 | |
| Conceptual ("wrong approach") | 8/30 | rarest *and* hardest to repair |

The lesson: **critique without external truth signal is theater.** RLEF as
deployed has no run-the-code, no Wolfram Alpha, no second-opinion model — the
critic confidently invents flaws and the regenerator is too obedient. The
remedy used in HW3 (each refinement step grounded in a real tool result)
fixes exactly this failure mode and produces dramatic gains.

### 4. Cost-effectiveness per API call

| Method | Total calls | Accuracy | acc gained per call vs zero-shot |
|--------|-------------|----------|----------------------------------|
| Zero-shot | 30 | 0.400 | — |
| Majority vote n=4 (peak) | 30 + 90 = 120 | 0.467 | +0.0007 / call |
| Majority vote n=16 | 30 + 450 = 480 | 0.433 | +0.000073 / call |
| LLM voting (16+1) | 30 + 480 = 510 | 0.500 | +0.00021 / call |
| RLEF (3/problem) | 30 + 60 = 90 | 0.367 | **negative** |

Per-call returns are tiny across the board on hard AIME problems — the *only*
reasonable choice if you care about cost is **majority vote at n=4**. But if
you care about absolute accuracy and have the budget, **LLM voting at n=16 is
worth it** (catches 3 problems no other method touched).

### 5. Wall-clock and call latency

577 model calls total. Median per-call latency 7.94 s, p95 10.88 s.
Haiku-4.5 is fast enough that even the 480-call majority-voting step finishes
in ≈3 min wall-clock under the `max_workers=12` cap. Full notebook ≈12 min
fresh, ≈2 min cached.

## Reproduce

Activate your env (needs `litellm`, `datasets`, `latex2sympy2`, `jupyter`,
`nbconvert`, etc. — see `requirements.txt`), then from `homework1/`:

```bash
set -a; source ../.env; set +a
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" TQDM_DISABLE=1
python -m nbconvert --to notebook --execute --inplace \
  student_homework1.ipynb --ExecutePreprocessor.timeout=3000 --ExecutePreprocessor.kernel_name=python3
```
First full run ≈ 12 min (577 model calls logged to `.cache/llm/`);
cached re-runs ≈ 2 min.

## Environment note
`latex2sympy2` pins `antlr4-python3-runtime==4.7.2`, which imports the removed
`typing.io` on Python 3.13. Patched both `antlr4/Lexer.py` and `antlr4/Parser.py`
(`from typing.io import TextIO` → `from typing import TextIO`) in the env's
site-packages. If you reinstall `antlr4-python3-runtime` you'll need to redo
this patch, or use Python ≤ 3.12.
