# Homework 2 — Code Generation & Verification (HumanEval)

LLM code generation on the 164-problem HumanEval benchmark, with increasingly
sophisticated verification: zero-shot, pass@k sampling, LLM-as-a-judge, and
LLM-generated unit tests as weak verifiers.

**Models:** `anthropic/claude-haiku-4-5` for generation + judging (was the
small Llama-3.2-3B); `anthropic/claude-sonnet-4-5` for unit-test generation
(was Qwen-72B). **Sandbox:** Docker is not installed here, so generated code
runs in the repo's local subprocess sandbox `cs329_hw/run/sandbox.py`
(`run_python`), which isolates with POSIX rlimits, an isolated interpreter
(`-I -B -S`), a scrubbed env, and a wall-clock timeout. Verified working on
macOS.

## Code changes

| File | Change |
|------|--------|
| `cs329_hw/openai_inference/litellm_models.py` | Same as HW1: relaxed `TOGETHER_API_KEY` guard → any provider key; `_raw_completion` + cached `_make_completion_request(occ)`; occurrence-indexed batch caching. |
| `cs329_hw/openai_inference/_llm_cache.py` | **New** (copy of the shared cache). |
| `cs329_hw/methods/llm_unit_test.py` | Implemented `generate()` (prompt → sample → parse → dedup → cap) and `_build_prompt()` (JSON test-case schema prompt). |
| `homework.ipynb` | Import + use local `run_python` instead of `run_python_in_docker`; tolerant `load_dotenv` (no hard-fail on missing `environment.env`); env-driven `DEBUG_MODE`; `qwen_path`→haiku, `qwen_large`→sonnet; implemented all 7 code TODOs + 2 written analyses. |

## Implemented tasks

- **Q1a `calculate_accuracy`** — verify each prediction, return accuracy + per-problem stdout + failed indices (robust to `None` from the judge).
- **Q2a `k_shot_acc`** — pass@k = "any of k candidates passes all tests" (short-circuits on first pass to save sandbox calls).
- **Q2b `LLMJudge.judge` + `_build_messages`** — build a judge prompt listing candidates, sample, parse a one-line JSON `{choice, reason}`, validate index range.
- **Q2c `evaluate_judge`** — run the judge per problem, return the chosen snippet (or `None`).
- **Q3a `LLMUnitTestGenerator` + `evaluate_ground_truth_on_llm_unit_tests`** — generate synthetic tests, check ground-truth code against them (require >0 tests so empty lists aren't vacuous passes).
- **Q3b `k_shot_acc_synthetic`** — pass@k using the synthetic tests on the trusted subset.

## Results (full 164-problem run)

| Metric | Value |
|--------|-------|
| Q1a Zero-shot accuracy | **0.841** (138/164) |
| Q2a pass@k (k=1/3/6/9) | 0.841 / 0.915 / 0.927 / **0.939** |
| Q2c LLM-as-judge (best-of-3) | 0.793 |
| Q3a Ground truth passes LLM tests | 126/164 (**0.768**) |
| Q3b pass@3 on trusted subset (GT tests / LLM tests) | 0.92 / 0.94 |

## Deeper insights

### 1. The LLM judge is *worse than uniform random selection*

Of the 3 candidates per problem, here's how the ground-truth labels break
down (using sandbox verification):

| #candidates correct | # problems |
|---------------------|-----------|
| 3 of 3 | 123 |
| 2 of 3 | 14 |
| 1 of 3 | 13 |
| 0 of 3 | 14 |

The pass@3 ceiling — "if you always picked a correct candidate when one
exists" — is **150/164 = 0.915**. The expected accuracy of a uniform-random
pick over 3 candidates is:

```
random = 123 · 1.0  +  14 · (2/3)  +  13 · (1/3)  +  14 · 0
       = 136.67 / 164  =  0.833
```

The judge scored **0.793**. **The judge is meaningfully worse than uniform
random.** The judge's failure modes, decomposed:

| Judge behaviour | Count | What happened |
|-----------------|-------|----------------|
| Picked a correct candidate when ≥1 was correct | 130 | ✓ |
| Picked a wrong candidate when ≥1 was correct | 12 | judge lost an available win |
| Said `null` when ≥1 was correct | 8 | judge gave up unnecessarily |
| Picked any candidate when all 3 were wrong | 14 / 14 | judge **never** said null on impossible problems |
| Said `null` when all 3 were wrong | 0 / 14 | judge has no "I don't know" calibration |

The deep failure is **calibration**: the judge has no notion of "none of
these look right" even when prompted explicitly with that option. So it eats
the 14 impossible problems and *also* gives up on 8 winnable ones. Net result:
the judge sits **below random**.

**Why this happens (mechanism, not excuse):** picking an index forces a
discrete commitment with no room to articulate uncertainty. Compare with HW1's
*generative* reward model, which got to *restate* a solution in its own words
— that path strictly dominated zero-shot. **Generative reward > index-pick
judge** is consistent across both homeworks.

### 2. Zero-shot failures are mostly *formatting*, not reasoning

26 problems failed zero-shot (164 − 138). Manual classification of the
failures:

- **≈19 / 26 are parse / SyntaxError failures** — at temperature 0.7 the model
  occasionally leaks revision/reasoning prose into the output (e.g. literal
  lines like `Wait, let me reconsider...` between code lines), and the
  `extract_code` regex only strips ```` ``` ```` fences. The result is invalid
  Python that crashes before a single `assert` runs.
- **≈7 / 26 are genuine edge-case logic errors** — e.g. `is_nested` passed
  12/14 cases, `is_simple_power` passed 9/10. The model has the main idea but
  mishandles boundary conditions (empty input, single-element nesting,
  `power==1`).

This is the mechanism behind the smooth pass@k curve: most of the gain from
re-sampling is *recovering formatting failures*, not new reasoning. A
smarter `extract_code` (e.g. extracting the largest valid Python fragment via
`ast.parse`) would push k=1 closer to k=3.

### 3. Cost-effectiveness of pass@k

| Strategy | LLM calls | Sandbox runs (worst case) | Accuracy | Marginal acc / call |
|----------|-----------|---------------------------|----------|---------------------|
| Zero-shot (k=1) | 164 | 164 | 0.841 | — |
| pass@3 | 492 | up to 492 | 0.915 | **+0.00023** |
| pass@6 | 984 | up to 984 | 0.927 | +0.000087 (vs k=3) |
| pass@9 | 1476 | up to 1476 | 0.939 | +0.000049 (vs k=6) |
| LLM judge (best-of-3) | 492 + 164 = 656 | 1 / problem | 0.793 | **negative** |

Diminishing returns are very fast: 9× the calls for +0.098 accuracy.
**The k=1 → k=3 step is the only one with a healthy return-on-spend.**

### 4. LLM-generated unit tests inherit the same reasoning errors

The sonnet-generated test cases mark the ground-truth code wrong on 38/164
problems (76.8% trust rate). Two concrete instances (notebook analysis cell):

- `order_by_points([15,23,7,42,8])` → generator expected `[7,8,15,23,42]`
  (ignored the spec's "sort by sum of digits" rule, fell back to ordinary
  numeric sort).  
- `intersection([1,10],[3,9])` → generator computed length as `7` (off-by-one;
  should be `9-3=6`), concluded "7 is prime → YES" when the right answer is
  `NO`.

This is the same reasoning failure as code generation, applied to writing
test oracles. **A weak verifier is weak.** This is why the assignment
restricts Q3b to the "trusted subset" where ground truth passes the synthetic
tests — without that filter the metric is noise.

On the trusted subset, pass@3 with synthetic tests slightly *beats* pass@3
with ground-truth tests (0.94 vs 0.92) — but that's a selection effect, not a
real win: easier-to-solve problems also tend to have easier-to-write tests.

## Reproduce

From `homework2/`:

```bash
set -a; source ../.env; set +a
export TQDM_DISABLE=1                # HW_DEBUG=1 for the 20-problem subset
python -m nbconvert --to notebook --execute --inplace \
  homework.ipynb --ExecutePreprocessor.timeout=5400 --ExecutePreprocessor.kernel_name=python3
```

Full run ≈ 24 min (1804 model calls logged to `.cache/llm/`; sandbox runs
are serial and account for a large share of wall-clock).

## Note on safety
Running model-generated code locally is riskier than the intended Docker
sandbox. `sandbox.py` mitigates with rlimits + isolated interpreter +
scrubbed env + timeout, but it is not a true jail. Install Docker and switch
back to `run_python_in_docker` for stronger isolation if desired (one-line
import change in cell 1 of the notebook).
