# CS329A Homeworks — Claude/litellm-powered run

This repo contains three Stanford CS329A "Self-Improving AI Agents" homeworks.
They were originally written against **Together AI / Qwen** models. This run
re-wires the model calls through **litellm**, using Anthropic Claude as the
workhorse and OpenAI/Gemini where HW3 benefits from provider diversity.

## Global decisions

| Topic | Decision |
|-------|----------|
| Python | a single env (3.10+) with `litellm`, `datasets`, `jupyter`, `nbconvert`. Activate before running. |
| Default model | `anthropic/claude-haiku-4-5` (workhorse) |
| Hard-role comparison | Sonnet run preserved for HW3 comparison; current final run uses Haiku in the 70B slot |
| Cheapest judge | `openai/gpt-5-nano` (HW3 `evaluate_qa`; no `temperature` param) |
| Fusion diversity | Haiku + GPT-5-nano + Gemini 3.1 Flash-Lite (three providers) |
| Execution | notebooks run headless with `nbconvert --execute`, outputs saved inline |
| Caching | every LLM call (and HW3 tool call) logged to `<hw>/.cache/llm/calls.jsonl` + read-through cache (diversity-safe) |

## Keys / environment

Expected at run time, but never copied into the repo:
- Provider keys in the shell environment: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`
- repo-root `.env` (gitignored): `POLYGON_API_KEY`, `WOLFRAM_APP_ID`

Missing/unavailable in this environment: `GOOGLE_API_KEY`/`GOOGLE_CX_ID`
(→ DuckDuckGo fallback for web search in HW3); Yahoo Finance is network-blocked
here (→ Polygon used directly); Docker is not installed (→ HW2 uses the local
subprocess sandbox).

## Run recipe (any homework)

Activate the Python env that has the deps, make sure provider keys are already
in the shell environment, then from `homeworkN/`:

```bash
set -a; source ../.env; set +a          # POLYGON / WOLFRAM   (run from homeworkN/)
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" # HW1 dataset loader
export TQDM_DISABLE=1                    # clean notebook output
# export HW_DEBUG=1                      # optional: tiny subset for quick iteration

python -m nbconvert --to notebook --execute --inplace <notebook>.ipynb \
  --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=python3
```

## Status

| HW | Topic | Status | Detail |
|----|-------|--------|--------|
| 1 | Test-time compute scaling (AIME25) | ✅ done | [hw1.md](hw1.md) |
| 2 | HumanEval code-gen + weak verifiers | ✅ done | [hw2.md](hw2.md) |
| 3 | Agentic workflow + external APIs | ✅ done | [hw3.md](hw3.md) |

## Headline results

| HW | Best result |
|----|-------------|
| 1 | LLM-voting (best-of-16) **0.500** on AIME25 (zero-shot 0.400, majority-vote peak 0.467) |
| 2 | pass@9 **0.939** on HumanEval (zero-shot 0.841); the LLM-judge baseline is **worse than random** (0.793 vs 0.833) |
| 3 | Current cheap-stack `run_pipeline` **73.3%** (11/15) vs zero-shot 20.0%; preserved Sonnet comparison run reaches **93.3%** (14/15) |

Model calls (and HW3 tool calls) are cached under each `homeworkN/.cache/llm/`,
with HW3's preserved Sonnet comparison run under
`homework3/.cache/llm-sonnet-gpt41/`. The checked-in caches contain 3,050 call
records total.

## Three cross-HW observations

1. **Selection is harder than generation.** Both HW1 (LLM-voting picks
   restating among 16 samples) and HW2 (LLM-judge picking 1 of 3 candidates)
   show this. In HW1 a *generative* judge wins because it gets to "argue" for
   an answer in its own words; in HW2 the *index-picking* judge loses to
   uniform-random selection because it can't say "I don't know" and chronically
   over-commits. Conclusion: ask judges to *write*, not to *vote with an index*.

2. **Self-improvement only works when the feedback is grounded in something
   outside the model.** HW1's RLEF (critic = same Haiku, no oracle) *regressed*
   −0.033 — the critic confidently invents flaws and the regenerator dutifully
   "fixes" correct solutions. HW3's iterative refinement (each step grounded in
   a real tool result) gained +53.3 pp in the cheap stack and +66.6 pp in the
   Sonnet stack. Critique without an external truth signal is theater.

3. **Tools dominate, but orchestration model size still compounds.** In HW3,
   Haiku goes from 20.0% naked to 73.3% with tools; Sonnet goes from 26.7%
   naked to 93.3% with tools. Tool access is the dominant axis, but stronger
   orchestration matters inside multi-step loops where arithmetic, state
   tracking, and resisting bad snippets compound across steps.

4. **The highest-ROI engineering fixes are often unglamorous.** Diversity-safe
   cache keys made sampling reproducible without collapsing stochastic draws;
   robust code extraction would likely recover many HW2 formatting failures;
   and HW3's centralized `MODEL_MAP` made provider swapping measurable instead
   of anecdotal.

## Walkthroughs

- [HW1 code walkthrough](hw1-code-walkthrough.md): sampler, verifier, voting,
  and RLEF flow.
- [HW2 code walkthrough](hw2-code-walkthrough.md): generation, extraction,
  sandbox verification, LLM judge, and synthetic tests.
- [HW3 code walkthrough](hw3-code-walkthrough.md): routing, tool use,
  decomposition, fusion, iterative refinement, and deep research.

## Caching layout (per homework)

```
homeworkN/.cache/
  llm/
    calls.jsonl          # append log: every real model call (and HW3 tool call)
    responses/<sha>.json # read-through cache entries (also the persisted outputs)
  *.png                  # generated figures
```
Disable with `HW_CACHE=0`. Relocate with `HW_CACHE_DIR=/path`.

**Diversity-safe keying:** for sampling at `temperature > 0`, the wrapper
assigns each identical prompt in a batch an *occurrence index* and caches it
under its own key. So a re-run reproduces the same `N` *diverse* samples
rather than collapsing to a single response.
