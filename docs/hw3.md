# Homework 3 — Agentic Workflow with External APIs

An end-to-end agent that routes natural-language queries to external tools
(web search, stock data, math, weather), then layers self-improvement
(decomposition + multi-model fusion, iterative refinement) and a deep-research
agent. Evaluated on a 15-question dataset spanning math / google / weather /
stock.

## Provider & tool wiring (current)

| Concern | Original | This run |
|---------|----------|----------|
| LLM transport | OpenAI SDK → Together AI | **litellm** → Anthropic / OpenAI / Gemini (single chokepoint `utils.generate_together`) |
| 8B-class workhorse (`Meta-Llama-3.1-8B-Instruct-Turbo`) | Llama-3.1-8B | `anthropic/claude-haiku-4-5` |
| 70B-class hard roles (`Llama-3.3-70B-Instruct-Turbo-Free`) | Llama-3.3-70B | `anthropic/claude-haiku-4-5` (was sonnet; see "Sonnet vs Haiku" below) |
| LLM judge (`evaluate_qa`, `google/gemma-3n-E4B-it`) | gemma-3n | `openai/gpt-5-nano` (cheapest GPT-5; no `temperature` param) |
| Fusion 3rd model (`OpenAI/gpt-oss-20B`) | gpt-oss-20B | `gemini/gemini-3.1-flash-lite` |
| Web search | Google Custom Search | **DuckDuckGo** (`ddgs`) — no Google key available |
| Stock | Polygon | **Polygon** (real, key provided; free tier covers the ~2 yr window) |
| Math | Wolfram Alpha | **Wolfram** (real, key provided) |
| Weather | Open-Meteo + Nominatim | same (keyless) |

Model ids are translated centrally in `utils.MODEL_MAP`, so notebook model
strings did not need editing. `gpt-5-nano` requires dropping `temperature`
(only the API default is accepted) — handled transparently in
`generate_together`.

## Code changes

| File | Change |
|------|--------|
| `cs329a_hw3/utils.py` | Rewrote `generate_together` to call `litellm.completion`, preserving the `.content` return contract; added `MODEL_MAP`, default `max_tokens` (litellm requires it for Anthropic; Gemini "thinking" needs headroom), `response_format` only for OpenAI, retries, disk caching, and **GPT-5-family temperature drop**. Defensive: always returns a `Message` (with `content=""` on transient failure) so callers' `.content.strip()` never crashes. |
| `cs329a_hw3/_llm_cache.py` | **New** (shared cache). Does NOT cache `None` responses → transient API failures retry on next run. |
| `cs329a_hw3/api_manager.py` | Implemented `__init__`, `_parse_query_params`, `route_query`, `_extract_webpage_content`, `google_search` (Google CSE → DDG fallback), `get_stock_data` (Polygon), `get_weather` (Nominatim + Open-Meteo archive→forecast), `compute` (Wolfram). Added `_extract_json` robust parser and `_cached_tool` (caches *successful* tool results for reproducibility + analysis; errors are not cached so transient failures can retry). |
| `cs329a_hw3/multi_lm_agent.py` | Implemented all 11 methods (generate, single-API, decomposition prompt/parse/run, synthesis prompt, decompose-and-fuse, iterative-refinement prompt/loop, run_pipeline). |
| `cs329a_hw3/DeepResearchAgent.py` | Implemented `research` (plan searches → gather web evidence → cited 4-5 paragraph report + sources) and added the missing `generate` method the notebook calls. |
| `homework3.ipynb` | Key-loading cell no longer hard-fails on missing Google/Together keys; fixed the bar-graph double `/100` bug. |

## Results — current run (haiku-everywhere + gpt-5-nano + gemini-3.1-flash-lite)

| Pipeline | Dataset | Accuracy |
|----------|---------|----------|
| Part 0 — zero-shot (haiku) | full (15) | 20.0% |
| Part 1f — single API + LM call | debug (8) | 62.5% |
| Part 2c — decomposition + 3-model fusion | debug (8) | 87.5% |
| Part 2d — iterative refinement | debug (8) | **100.0%** |
| Part 3 — zero-shot (haiku, was sonnet) | full (15) | 20.0% |
| **Part 3 — `run_pipeline`** | full (15) | **73.3%** (11/15) ✅ > 70% target |

## Sonnet vs Haiku — controlled comparison

We ran HW3 twice. Identical code, identical tool results (cached), **only the
70B-slot model and judge/fusion sub-models differ**:

| Pipeline | A: sonnet + gpt-4.1-nano + gemini-2.5-flash | B: **haiku-everywhere + gpt-5-nano + gemini-3.1-flash-lite** | Δ |
|----------|---------------------------------------------|-------------------------------------------------------------|---|
| Part 0 zero-shot (haiku, unchanged) | 20.0% | 20.0% | 0 |
| Part 1f single-API+LM | 50.0% | 62.5% | **+12.5** |
| Part 2c decompose+fuse | 75.0% | 87.5% | **+12.5** |
| Part 2d iter-refine (debug-8, saturated) | 100.0% | 100.0% | 0 |
| Part 3 zero-shot (70B-slot) | 26.7% (sonnet) | 20.0% (haiku) | −6.7 |
| **Part 3 `run_pipeline` (full 15)** | **93.3%** (14/15) | **73.3%** (11/15) | **−20.0** |

Two important asymmetries here:

1. **Single-shot tasks improved or held steady** when switching to the cheaper
   stack (`single-API` 50→62.5, `decompose+fuse` 75→87.5). The `gpt-5-nano`
   judge appears slightly more lenient/accurate on long verbose answers than
   `gpt-4.1-nano` (it didn't fail-mark `"Boston was windier..."` style
   answers).

2. **Multi-step tasks collapsed** when haiku had to do the refinement
   reasoning (`run_pipeline` −20 pp). The 70B-class model was earning its
   keep specifically in the iterative loop, not in the zero-shot baseline
   (where it only gained +6.7 pp over haiku).

### Where does the −20 pp come from? All 4 failures inspected

| # | Cat | Question | Expected | Haiku-pipeline answer | Failure mode |
|---|-----|----------|----------|-----------------------|--------------|
| 1 | WEATHER | "morning 9AM temp in Miami first 3 days of Aug 2025 ever below 70°F?" | No | *"I cannot definitively answer..."* | **gave up on multi-day comparison** even with all 3 days' tool results in context |
| 2 | STOCK | "TSLA close on Aug 27, 2025 increased by 11.2% = ?" | 388.76 | $388.59 | **arithmetic error**: 349.60 × 1.112 = 388.76, haiku computed 388.59 (off by 17 cents) |
| 3 | GOOGLE | "How many SF restaurants have 3 Michelin stars in 2025?" | Three | *"the tool results do not contain..."* | DDG didn't return the count (same honest failure as the sonnet run) |
| 4 | GOOGLE | "Nationality of 2025 men's Wimbledon winner?" | Italian | "Alcaraz, who is Spanish" | **confabulation**: 2025 Wimbledon was won by Jannik Sinner (Italian); haiku invented Alcaraz on top of DDG snippets |

3 of the 4 failures are *model-reasoning* errors that sonnet handled in the
previous run (arithmetic, multi-step state tracking, resisting
confabulation). Only #3 is a genuine retrieval failure (DDG vs Google CSE).

**Conclusion of the controlled experiment:** in this agentic pipeline,
upgrading the orchestration LLM (haiku → sonnet) is worth **+20 pp full
accuracy** — much more than the +6.7 pp it buys in the bare zero-shot
baseline. The marginal value of model size compounds with the number of
reasoning steps in the loop.

## Other insights

### Router accuracy is perfect — even on haiku

Replaying every question through `APIManager.route_query` produces the
correct tool selection **15/15** across all four categories, with haiku as
router. The structured-output prompt + `_extract_json` fallback parser is
reliable enough that the cheap router with explicit JSON instructions matches
the Pydantic schema on every call. Notably this works even though `litellm`
doesn't pass dict-style `response_format` to Anthropic — we rely on
prompt-level JSON instructions + post-hoc extraction instead.

So **routing is not the bottleneck** — *reasoning on the routed results* is.

### Tools dominate model size

The same Haiku scores 20% naked and 73.3% with tools (+53.3 pp). Sonnet
scores 26.7% naked and 93.3% with tools (+66.6 pp). Both deltas are far
larger than the haiku↔sonnet gap (6.7 pp zero-shot, 20 pp wrapped). Tool
access is the dominant axis of value on this dataset.

### Why iterative refinement beats decomposition + fusion

On the debug-8 subset: decompose-and-fuse 87.5% vs iterative-refine 100%.
The mechanism:

- **decompose-and-fuse** breaks the question into independent sub-queries,
  runs them *in parallel*, and synthesizes. This is great when sub-questions
  are truly independent (high/low/close of TSLA on a date), but fails when
  one sub-question's answer is needed to formulate the next. *"If TSLA close
  × 1.112 = ?"* requires the close *first*, then arithmetic.

- **iterative refinement** runs sub-queries sequentially and lets each step
  condition on prior results — exactly the dependency structure those
  questions need.

### Cross-HW insight: critique needs an external truth signal

In HW1, RLEF (model critiques itself, model rewrites) *regressed* −0.033.
In HW3, iterative refinement (model issues a tool query, gets external
result, refines) *gained* +53.3 pp (haiku) / +66.6 pp (sonnet) over zero-shot.
The structural difference is whether feedback is grounded outside the model:

| | Feedback source | Result vs zero-shot |
|-|-----------------|---------------------|
| HW1 RLEF | another Haiku critic | −0.033 (regression) |
| HW3 iterative (haiku) | Wolfram / Polygon / Open-Meteo / DDG | +0.533 |
| HW3 iterative (sonnet) | Wolfram / Polygon / Open-Meteo / DDG | +0.667 |

**Self-improvement only works when "self" gets corrected by the world.**

### Provider mix proves genuine cross-provider use

Cache breakdown for the current run:

| Provider / tool | Calls |
|-----------------|-------|
| `anthropic/claude-haiku-4-5` | 192 (workhorse, generation, fusion, decomposition, refinement) |
| `openai/gpt-5-nano` | 62 (LLM judge in `evaluate_qa`, +1 fusion slot) |
| `gemini/gemini-3.1-flash-lite` | 8 (3rd fusion slot) |
| `gemini/gemini-2.5-flash-lite` | 8 (intermediate fusion rerun retained in cache) |
| `anthropic/claude-sonnet-4-5` | 6 (deep-research agent: planning + report synthesis) |
| `google_search` (DDG) | 26 |
| `compute` (Wolfram) | 6 |
| `get_weather` (Open-Meteo) | 5 |
| `get_stock_data` (Polygon) | 4 |

Note Sonnet still appears in the DeepResearchAgent (Part 4) where it does the
report-writing — that's hard-coded inside `DeepResearchAgent.research` and
isn't affected by `MODEL_MAP`. The 70B-slot agent paths now all run on haiku.

### Missing engineering insight: exact tools should own exact arithmetic

Two of the haiku-pipeline failures were not retrieval failures; they were
post-retrieval reasoning failures. The agent had enough weather/stock context,
but failed a multi-day comparison and a simple percentage calculation. The next
best improvement is not another retrieval provider; it is forcing exact numeric
post-processing through the math tool (or local deterministic code) after the
retrieval step. In other words, tool use should not stop at fetching evidence;
it should also own the brittle arithmetic and comparisons that small models are
most likely to blur.

### Where the prior run's cache lives

The previous (sonnet) run's cache is preserved at `.cache/llm-sonnet-gpt41/`
(352 call-log records). The current (haiku) run's cache is at `.cache/llm/`
(317 records: 276 LLM + 41 tool). Both can be diffed for further analysis.

## Reproduce

From `homework3/`:

```bash
set -a; source ../.env; set +a          # POLYGON_API_KEY, WOLFRAM_APP_ID
export TQDM_DISABLE=1
python -m nbconvert --to notebook --execute --inplace \
  homework3.ipynb --ExecutePreprocessor.timeout=3000 --ExecutePreprocessor.kernel_name=python3
```
≈ 9 min for a fresh run; cached re-runs ≈ 30 s.

## Extra dependencies installed
`geopy`, `google-api-python-client`, `polygon-api-client`, `textblob`,
`xmltodict`, `ddgs`, `nest_asyncio`, `google-genai`.
