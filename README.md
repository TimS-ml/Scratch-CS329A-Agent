# CS329A Self-Improving AI Agents Homeworks

Completed, reproducible runs for three CS329A homework assignments. The original
course code targeted Together/Qwen. This repo keeps the assignment structure but
routes model calls through `litellm`, uses Claude as the main model family, and
persists all model/tool outputs for auditability.

## Contents

| Path | Topic | Main result |
|------|-------|-------------|
| `homework1/` | AIME25 test-time compute | LLM-voting best-of-16 reaches 0.500 vs 0.400 zero-shot |
| `homework2/` | HumanEval code generation and weak verification | pass@9 reaches 0.939 vs 0.841 zero-shot |
| `homework3/` | Agentic workflows with tools | cheap-stack agent reaches 73.3%; preserved Sonnet run reaches 93.3% |
| `docs/` | Results, analysis, walkthroughs | Start with `docs/README.md` |

## What changed

- Replaced hard Together-only inference guards with provider-flexible `litellm`
  wrappers.
- Added read-through disk caches under `homeworkN/.cache/` so notebook runs are
  reproducible and inspectable.
- Completed all notebook TODOs for HW1, HW2, and HW3.
- Added local execution fallback for HW2 because Docker is unavailable in this
  environment.
- Implemented HW3 tool routing for DuckDuckGo, Polygon, Wolfram, and
  Open-Meteo/Nominatim.

## Key findings

1. Selection is harder than generation: HW1's generative reward model improves
   accuracy, while HW2's index-picking judge performs below random.
2. Self-improvement needs external grounding: HW1's self-critique regresses;
   HW3's tool-grounded refinement improves sharply.
3. Tools dominate retrieval/computation tasks, but stronger orchestration models
   still matter inside multi-step loops.
4. Reproducibility plumbing matters: diversity-safe cache keys preserve stochastic
   sampling without collapsing repeated prompts to the same answer.

## Documentation

- `docs/README.md`: consolidated run notes, headline results, and cross-HW
  observations.
- `docs/hw1.md`, `docs/hw2.md`, `docs/hw3.md`: per-homework implementation
  details and analysis.
- `docs/hw1-code-walkthrough.md`, `docs/hw2-code-walkthrough.md`,
  `docs/hw3-code-walkthrough.md`: code-path walkthroughs with Mermaid diagrams.

## Reproducing

Install the dependencies for the homework you want to run, activate your Python
environment, and make sure the needed provider keys are already in the shell
environment. Keep secrets out of git.

Common variables:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export HF_TOKEN=...
```

Repo-local `.env` is gitignored and may contain:

```bash
POLYGON_API_KEY=...
WOLFRAM_APP_ID=...
```

Run a notebook from its homework directory:

```bash
set -a; source ../.env; set +a
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export TQDM_DISABLE=1
python -m nbconvert --to notebook --execute --inplace <notebook>.ipynb \
  --ExecutePreprocessor.timeout=5400 --ExecutePreprocessor.kernel_name=python3
```

Use `HW_DEBUG=1` for smaller debug subsets where supported. Use `HW_CACHE=0` to
disable caches, or `HW_CACHE_DIR=/path/to/cache` to relocate them.

## Cache safety

The homework cache directories are intentionally versioned as reproducibility
artifacts. Before committing, they were scanned for provider key prefixes,
authorization headers, concrete loaded key prefixes, and absolute user paths.
No secrets were found.
