# DSPy: The Mostly Harmless Guide — Companion Code

### *A Practical, Project-Driven Guide to Programming (Not Prompting) Language Models*

**By [Ali Raza](https://aliirz.com)**

---

> *There are 42 ways to get an LLM to do what you want. Forty-one of them involve crying into your keyboard while tweaking prompts at 3 AM. The forty-second way is DSPy.*

This repository contains the tested, working code examples for every chapter of **DSPy: The Mostly Harmless Guide**.

## Quick Start

### Prerequisites

- Python 3.10–3.15 (DSPy 3.5 supports `>=3.10,<3.16`)
- [Poetry](https://python-poetry.org/docs/#installation)
- An API key from [Anthropic](https://console.anthropic.com/) or [OpenAI](https://platform.openai.com/)

### Setup

```bash
# Pick a chapter and install its dependencies
cd ch01_dont_panic
poetry install

# Copy the environment template and add your API key
cp .env.example .env
# Edit .env with your actual API key

# Run the tests
poetry run python -m pytest test_startup_roaster.py -v
```

### Run All Chapter Tests

```bash
python run_all_tests.py          # Full suite (all 8 chapters)
python run_all_tests.py -c 4     # Just Chapter 4
python run_all_tests.py --fast   # Offline tests only (live API tests reported as skipped)
python run_all_tests.py -v       # Full output for debugging
python run_all_tests.py --no-guard  # Skip the harness self-check
```

### Offline Runs

`SKIP_LIVE_TESTS=1` is the one switch that turns live tests off, and it is
authoritative everywhere: `--fast` sets it for every chapter, but exporting it
yourself has the same effect when you invoke a chapter's tests directly, e.g.
`SKIP_LIVE_TESTS=1 poetry run python -m pytest test_all_chapter_examples.py`.
An exported `ANTHROPIC_API_KEY` does **not** re-enable live tests.

`chapter_test_support.py` holds that policy (`live_tests_enabled()`,
`api_key_available()`, `env_flag()`), so script-style chapters and pytest
chapters decide the same way. Before any chapter runs, `run_all_tests.py`
executes `test_offline_policy.py`, which checks that:

* every chapter test file is registered in the runner (no orphaned tests);
* `SKIP_LIVE_TESTS=1` marks every live test skipped even with dummy provider
  keys exported, and leaves them enabled without it — through both entry
  points, `python test_all_chapter_examples.py` and `python -m pytest`;
* a key for one provider never unlocks another provider's live test
  (`ch03`'s embedding tests want OpenAI, the rest want Anthropic);
* the offline chapters stay green with dummy keys while
  `offline_guard/sitecustomize.py` turns any outbound socket into an error.

`--fast` also installs `offline_guard/` into the chapter processes, so a live
test that forgets its marker fails loudly instead of dialling out.

### Container Run

`Dockerfile.test` (plus `requirements-test.txt` and `.fleet-test.json`) builds
one image with the dependencies of all eight chapters and runs the whole
offline suite:

```bash
docker build -f Dockerfile.test -t harmless-code-tests .
docker run --rm harmless-code-tests
```

The image points `DSPY_REPO_PATH` at the installed `dspy` package, so ch03's
codebase-Q&A tests load a real source tree instead of skipping.

## Repository Structure

```
.
├── run_all_tests.py                    # Unified test runner
├── chapter_test_support.py             # Shared offline/live policy for every chapter
├── test_offline_policy.py              # Guard test: offline policy, registry, no network
├── offline_guard/sitecustomize.py      # Blocks outbound sockets in offline runs
├── Dockerfile.test                     # Container recipe for the whole suite
├── requirements-test.txt               # Dependency union for that image
├── .fleet-test.json                    # fleet-test profile: container-test
├── ch01_dont_panic/                    # Signatures, Modules, first program
│   ├── startup_roaster.py
│   └── test_startup_roaster.py
├── ch02_restaurant_pipeline/           # Composition, chains, Pydantic types
│   ├── lead_engine.py
│   └── test_all_chapter_examples.py
├── ch03_retrieval/                     # RAG, embeddings, evaluation
│   ├── codebase_qa.py
│   └── test_all_chapter_examples.py
├── ch04_babel_fish/                    # Optimizers (the whole zoo)
│   ├── ticket_classifier.py
│   └── test_all_chapter_examples.py
├── ch05_agents/                        # Agents, tools, ReAct
│   ├── research_agent.py
│   └── test_all_chapter_examples.py
├── ch06_production/                    # Caching, streaming, FastAPI
│   ├── content_moderator.py
│   └── test_all_chapter_examples.py
├── ch07_advanced/                      # Vision, reasoning, advanced optimizers
│   ├── multimodal_analyzer.py
│   └── test_all_chapter_examples.py
└── ch08_rlm/                           # Recursive Language Models, sub-LLMs
    ├── contract_engine.py
    └── test_contract_engine.py
```

## Chapters at a Glance

| Chapter | Title | Project | Key Concepts |
|---------|-------|---------|-------------|
| 1 | **Don't Panic** | Startup Idea Roaster | Signatures, Modules, first program |
| 2 | **Restaurant at the End of the Pipeline** | Lead Intelligence Engine | Composition, chains, Pydantic types |
| 3 | **Life, the Universe, and Retrieval** | Codebase Q&A System | RAG, embeddings, evaluation |
| 4 | **The Babel Fish** | Ticket Classifier | Optimizers (the whole zoo) |
| 5 | **So Long, and Thanks for All the Prompts** | Research Agent | Agents, tools, ReAct |
| 6 | **Mostly Harmless (in Production)** | Content Moderation Pipeline | Caching, streaming, FastAPI |
| 7 | **The Answer Is 42 (Tokens)** | Multimodal Product Analyzer | Vision, reasoning, advanced optimizers |
| 8 | **The Infinite Improbability Context** | Contract Intelligence Engine | Recursive Language Models, sandboxed tools, sub-LLM budgets |

## Technical Stack

| Component | Choice |
|-----------|--------|
| **DSPy** | 3.5.0 (pinned in every `pyproject.toml` and `poetry.lock`) |
| **Primary LLM** | Anthropic Claude |
| **Python** | 3.10–3.15 |
| **Package Manager** | Poetry |
| **Test Framework** | pytest |

Each chapter is self-contained with its own `pyproject.toml` and `poetry.lock`,
pinned to DSPy 3.5.0 (`poetry check --lock` passes in every chapter). `dspy.LM` defaults to `engine="auto"`, which resolves
models in its built-in registry (Anthropic, OpenAI, Gemini, Groq, OpenRouter, …)
to DSPy's native adapters — no extra installed. LiteLLM is the **compatibility fallback**: it handles unknown
model strings and provider options the native engine has no verdict for, and
it ships as the optional extra `dspy[litellm]`, which every chapter declares so
that fallback path works. Retrieval chapters add `dspy[numpy,openai]` for vector
search and hosted embeddings; the optimizer chapter adds `dspy[optuna]`.

Chapter 3 reads a DSPy source checkout. Clone it beside this repository (or set
`DSPY_REPO_PATH`) before running that chapter's retrieval tests:

```bash
git clone --branch 3.5.0 https://github.com/stanfordnlp/dspy ../dspy
```

## DSPy 3.5 Migration Notes

These examples track DSPy 3.5.0. The changes that affect the book's code:

| Area | DSPy 3.5 behavior | What the code does |
|------|------------------|--------------------|
| Default adapter | `XMLAdapter(use_native_function_calling=True)` | Adapter chapters name XML as the default and keep Chat/JSON as alternatives |
| `ChainOfThought` | exposes `reasoning` (typed `dspy.Reasoning`, may be `None`) | code stringifies with `str(result.reasoning or "")` |
| `dspy.Embeddings` | `save(path)` / `load(path, embedder)` instance methods, plus the `Embeddings.from_saved(path, embedder)` classmethod | examples build an instance then `load()`; `from_saved()` is still available and skips recomputing corpus embeddings |
| `dspy.ReAct` | auto-adds a `submit` tool; results expose `history.messages` and `termination_reason` | tests assert `submit`; examples read `result.history.messages` |
| `dspy.Image` | `from_path()` / `from_url()`; construction never touches disk or network | `from_file()` is gone and the `download=` / `verify=` *constructor* keywords now raise `TypeError` — but `from_url(url, verify=..., timeout=...)` still takes `verify`, and the code uses that form |
| Routing | `engine="auto"` uses native adapters for known providers; LiteLLM is the fallback | chapters keep `dspy[litellm]` declared for the fallback path |
| Imports | LiteLLM, numpy, openai, optuna are optional extras | chapters declare the extras they use |

## About the Book

This book is for developers who are tired of prompt engineering feeling like dark magic and want to treat LLM programming like *actual* programming — with modules, optimizers, metrics, and reproducible results.

**Get the book:** [Coming soon]

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*"The ships hung in the sky in much the same way that bricks don't. Your DSPy programs, however, will soar."*
