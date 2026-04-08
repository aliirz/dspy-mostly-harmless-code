# DSPy: The Mostly Harmless Guide — Companion Code

### *A Practical, Project-Driven Guide to Programming (Not Prompting) Language Models*

**By [Ali Raza](https://aliirz.com)**

---

> *There are 42 ways to get an LLM to do what you want. Forty-one of them involve crying into your keyboard while tweaking prompts at 3 AM. The forty-second way is DSPy.*

This repository contains the tested, working code examples for every chapter of **DSPy: The Mostly Harmless Guide**.

## Quick Start

### Prerequisites

- Python 3.11+
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
python run_all_tests.py          # Full suite (all 7 chapters)
python run_all_tests.py -c 4     # Just Chapter 4
python run_all_tests.py --fast   # Structural tests only (no API calls)
python run_all_tests.py -v       # Full output for debugging
```

## Repository Structure

```
.
├── run_all_tests.py                    # Unified test runner
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
└── ch07_advanced/                      # Vision, reasoning, advanced optimizers
    ├── multimodal_analyzer.py
    └── test_all_chapter_examples.py
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

## Technical Stack

| Component | Choice |
|-----------|--------|
| **DSPy** | 3.1.x |
| **Primary LLM** | Anthropic Claude |
| **Python** | 3.11+ |
| **Package Manager** | Poetry |
| **Test Framework** | pytest |

Each chapter is self-contained with its own `pyproject.toml` and `.env.example`.

## About the Book

This book is for developers who are tired of prompt engineering feeling like dark magic and want to treat LLM programming like *actual* programming — with modules, optimizers, metrics, and reproducible results.

**Get the book:** [Coming soon]

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*"The ships hung in the sky in much the same way that bricks don't. Your DSPy programs, however, will soar."*
