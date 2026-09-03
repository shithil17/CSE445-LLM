# Autonomous Local LLM ML Agent

A small, privacy-preserving ReAct agent that runs a 3B instruction model locally through Ollama and drives six ML tools. All work stays on-device with no external API calls. Built as a course project, it can autonomously compare a few algorithms across a couple of datasets.

## Setup

The project uses `uv` for Python and Ollama for the model. The steps below were run on Arch Linux (also works in Ubuntu/WSL).

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Pin Python and install deps (from pyproject.toml / uv.lock)
uv python pin 3.12
uv sync

# If you need a plain requirements file
uv export --format requirements-txt --no-hashes -o requirements.txt

# 3. Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
# (or keep it in a separate terminal)

# 4. Pull the model and check it's there
ollama pull llama3.2:3b
curl -s http://127.0.0.1:11434/api/tags
ollama list

# 5. Smoke test
uv run pytest tests/test_ml_tools.py tests/test_react_agent.py -v
uv run python code/react_agent.py
```

> Why `uv`? It gives a lockfile and reproducible installs without hand-managed virtualenvs.
> Why no LangChain? The loop is intentionally hand-written, using regex parsing of `Thought/Action/Action Input`, so the machinery stays visible.

## Usage

```bash
# Single query: ReAct loop, up to 6 turns, self-healing retries capped at 2
uv run python code/react_agent.py

# From Python
# from code.react_agent import run_agent_loop
# run_agent_loop("Analyze iris, compare Decision Tree vs Random Forest, recommend best.")

# Tools exposed (kept in sync between ml_tools.AVAILABLE_TOOLS and SYSTEM_PROMPT):
# - load_dataset_summary
# - train_sklearn_model
# - train_pytorch_mlp / train_pytorch_mlp_cv
# - tune_hyperparameters
# - select_features
# - train_regularized_mlp
```

Expected layout at the repo root:

- `code/ml_tools.py`, six tools, each returns a JSON string and never throws past its boundary
- `code/react_agent.py`, the ReAct controller (`query_local_llm`, `run_agent_loop`, regex parser, stop on `Observation:`)
- `code/benchmark_runner.py`, the task that evaluates 3 algorithms x 2 datasets

## Benchmark Reproduction

The benchmark tries three algorithms (decision_tree, random_forest, PyTorch MLP) on iris and breast_cancer with 5-fold CV and writes the markdown table to `report/benchmark_summary.md` plus details to `report/technical_report.md`.

By default the runner first tries a single-prompt approach for about 70 seconds (8 iterations) and, if that does not yield a valid six-row table, fans out to six smaller per-pair prompts. The per-pair path has a 400-second total budget, 75 seconds per pair.

**Two ways to run it (both valid):**

```bash
# Default: single-prompt attempt first, then split path (roughly 245s total)
uv run python code/benchmark_runner.py

# Faster: skip the single-prompt attempt, go straight to per-pair (about 227 seconds, which is what produced the reports here)
BENCHMARK_SKIP_MONOLITHIC=1 uv run python code/benchmark_runner.py
```

The faster route is what gave the 6/6 live-agent table currently in `report/benchmark_summary.md`. The September 3 run wrapped the six pairs in 227.3 seconds, with the model spending 223.0545 seconds across 24 calls (mean 9.2939s, median 8.9979s).

**What success looks like:**

```
# Benchmark Summary: 3 Algorithms x 2 Datasets (5-fold CV)
| iris | decision_tree | 0.9533 | 0.0340 | 0.9333 | agent |
| iris | random_forest | 0.9667 | 0.0211 | 0.9000 | agent |
| iris | pytorch_mlp | 0.8200 | 0.0452 | 0.7333 | agent |
| breast_cancer | decision_tree | 0.9209 | 0.0202 | 0.9386 | agent |
| breast_cancer | random_forest | 0.9543 | 0.0244 | 0.9561 | agent |
| breast_cancer | pytorch_mlp | 0.9508 | 0.0196 | 0.9386 | agent |
6/6 rows from live agent, 0/6 from deterministic fallback
agent_success_rate: 1.00 (6/6)
```

On stderr you will see `skip_monolithic enabled: skipping monolithic agent path`, then per-pair elapsed times and a final line like `split path total wall-clock 227.3s - 6/6 agent`.

After a run you can regenerate or inspect the reports:

```bash
cat report/technical_report.md   # timing details: two dozen latency values summing to 223.0545s
cat report/benchmark_summary.md  # the six-row table, all marked agent
```

**A note on jitter:** `train_test_split` is seeded at 42, but PyTorch initialization and Adam are not, so MLP scores move a little between runs (iris pytorch_mlp around 0.82/0.73 vs 0.84/0.80 in fallback). That is normal.

## Test & Verification

```bash
# Full suite
uv run pytest tests/ -v

# Is Ollama up and can the loop reach a Final Answer?
curl -s http://127.0.0.1:11434/api/tags
uv run python code/react_agent.py  # look for "Final Answer:" within the iteration limit

# Does the benchmark produce a proper six-row table?
uv run pytest tests/test_benchmark.py -v

# Logs and reports exist?
ls -1 execution_logs/   # you should see six split logs plus the raw/ stdout/ stderr files
cat report/benchmark_summary.md       # six data rows, all Source=agent
grep "Per-call latencies" report/technical_report.md
uv run pytest tests/ -v
```

## Project Structure

```
├── pyproject.toml / uv.lock
├── code/
│   ├── ml_tools.py
│   ├── react_agent.py
│   ├── benchmark_runner.py
│   └── requirements.txt
├── report/
│   ├── benchmark_summary.md
│   └── technical_report.md
├── execution_logs/          # multi-step ReAct traces
├── diagrams/
└── tests/
    ├── test_ml_tools.py
    ├── test_react_agent.py
    └── test_benchmark.py
```
