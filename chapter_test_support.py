"""Shared test policy for every chapter in this repository.

The book's examples run in two modes:

* **offline** — no provider call and no network traffic. Live tests are
  reported as *skipped*, never as passed and never as failures.
* **live** — real model calls, which cost money and need a provider key.

The switch is the ``SKIP_LIVE_TESTS`` environment variable. It is authoritative
for every chapter and for both ways of running the tests:

    python run_all_tests.py --fast              # harness sets SKIP_LIVE_TESTS=1
    SKIP_LIVE_TESTS=1 python -m pytest test_all_chapter_examples.py -v

A live test runs only when ``SKIP_LIVE_TESTS`` is unset/false *and* a provider
key is configured. Both conditions are implemented here so the chapters cannot
drift apart again.
"""

import os

#: Environment variables any chapter will accept as a provider credential.
API_KEY_VARS = ("LLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")

_FALSE_VALUES = {"", "0", "false", "no", "off"}


def env_flag(name: str) -> bool:
    """Return True unless *name* is unset or holds a false-y spelling."""
    return os.getenv(name, "").strip().lower() not in _FALSE_VALUES


def offline_mode() -> bool:
    """True when the run must not contact a model provider."""
    return env_flag("SKIP_LIVE_TESTS")


def api_key_available() -> bool:
    """True when some provider credential is present in the environment."""
    return any(os.getenv(name) for name in API_KEY_VARS)


def live_tests_enabled() -> bool:
    """True only when live model calls are both allowed and possible."""
    return api_key_available() and not offline_mode()


def live_skip_reason() -> str:
    """Explain why live tests are skipped ('' when they are enabled)."""
    if offline_mode():
        return "SKIP_LIVE_TESTS is set: offline run, no provider calls"
    if not api_key_available():
        return "no provider API key (LLM_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY)"
    return ""
