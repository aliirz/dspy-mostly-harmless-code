"""Shared test policy for every chapter in this repository.

The book's examples run in two modes:

* **offline** — no provider call and no network traffic. Live tests are
  reported as *skipped*, never as passed and never as failures.
* **live** — real model calls, which cost money and need a provider key.

Two things decide whether a live test runs:

1. the ``SKIP_LIVE_TESTS`` environment variable — authoritative, and the same
   for every chapter; and
2. a credential for the provider that the test actually talks to. Every
   chapter configures ``dspy.LM("anthropic/...")``, so its live tests need an
   Anthropic key; ``ch03``'s embedding tests talk to OpenAI and need
   ``OPENAI_API_KEY``. An OpenAI key alone must not enable an Anthropic test.

Both conditions live here so the chapters cannot drift apart, and both are
enforced for *both* ways of running a chapter's tests:

    python run_all_tests.py --fast              # harness sets SKIP_LIVE_TESTS=1
    SKIP_LIVE_TESTS=1 python -m pytest test_all_chapter_examples.py -v

Script-style chapters are also importable by pytest, so their live tests carry
real skip markers (see :func:`live_test`). A test that used to print
``SKIPPED`` and return now raises :class:`TestSkipped` when called directly,
which the chapter runner reports as a skip instead of counting it as a pass.
"""

import functools
import os

#: Credential variables per provider. ``LLM_API_KEY`` is the generic name the
#: book documents, and every chapter uses it for the Anthropic endpoint.
PROVIDER_KEY_VARS = {
    "anthropic": ("LLM_API_KEY", "ANTHROPIC_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
}

#: Provider assumed by a test that does not name one.
DEFAULT_PROVIDER = "anthropic"

_FALSE_VALUES = {"", "0", "false", "no", "off"}


def env_flag(name: str) -> bool:
    """Return True unless *name* is unset or holds a false-y spelling."""
    return os.getenv(name, "").strip().lower() not in _FALSE_VALUES


def offline_mode() -> bool:
    """True when the run must not contact a model provider."""
    return env_flag("SKIP_LIVE_TESTS")


def provider_key_vars(provider: str = DEFAULT_PROVIDER) -> tuple:
    """Credential variables that count for *provider*."""
    try:
        return PROVIDER_KEY_VARS[provider]
    except KeyError:
        raise ValueError(
            f"unknown provider {provider!r}; known: {sorted(PROVIDER_KEY_VARS)}"
        ) from None


def api_key_available(provider: str = DEFAULT_PROVIDER) -> bool:
    """True when a credential for *provider* is present in the environment."""
    return any(os.getenv(name) for name in provider_key_vars(provider))


def live_tests_enabled(provider: str = DEFAULT_PROVIDER) -> bool:
    """True only when provider calls are both allowed and possible."""
    return api_key_available(provider) and not offline_mode()


def live_skip_reason(provider: str = DEFAULT_PROVIDER) -> str:
    """Explain why live tests for *provider* are skipped ('' when enabled)."""
    if offline_mode():
        return "SKIP_LIVE_TESTS is set: offline run, no provider calls"
    if not api_key_available(provider):
        keys = " / ".join(provider_key_vars(provider))
        return f"no {provider} API key ({keys})"
    return ""


class TestSkipped(Exception):
    """Raised by a gated test that declines to run.

    pytest never sees this (the skip marker fires first); the chapter runners
    catch it and count a skip.
    """


def skip_test_when(predicate, reason) -> "callable":
    """Gate a test on ``predicate()`` in both entry points.

    Under pytest the test gets a ``skipif`` marker evaluated at collection
    time. Called directly (chapter script runner), it raises
    :class:`TestSkipped` instead of silently returning.

    ``reason`` may be a string or a zero-argument callable.
    """
    def reason_text() -> str:
        return reason() if callable(reason) else str(reason)

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if predicate():
                raise TestSkipped(reason_text())
            return fn(*args, **kwargs)

        # Record every gate so the offline-policy self-check can tell a
        # provider gate apart from a network/source-tree gate.
        wrapper.__chapter_gates__ = list(getattr(fn, "__chapter_gates__", ())) + [reason_text]

        try:  # pytest is optional: chapter scripts run without it installed
            import pytest
        except ImportError:  # pragma: no cover - script-only environments
            return wrapper
        return pytest.mark.skipif(bool(predicate()), reason=reason_text())(wrapper)

    return decorate


def skip_test_unless(predicate, reason) -> "callable":
    """:func:`skip_test_when` with the condition inverted."""
    return skip_test_when(lambda: not predicate(), reason)


def live_test(provider: str = DEFAULT_PROVIDER) -> "callable":
    """Mark a test as live: it needs *provider*'s key and a live-mode run.

    The decorated test also carries ``__live_providers__`` so the offline-policy
    self-check can verify provider scoping without importing a provider SDK.
    """
    gate = skip_test_when(
        lambda: not live_tests_enabled(provider),
        lambda: live_skip_reason(provider),
    )

    def decorate(fn):
        wrapped = gate(fn)
        wrapped.__live_providers__ = list(getattr(fn, "__live_providers__", ())) + [provider]
        return wrapped

    return decorate


def run_tests(tests, exit_on_failure: bool = True) -> dict:
    """Run chapter tests, counting real skips separately from passes.

    Returns ``{"passed": int, "failed": int, "skipped": int, "total": int}``.
    Exits with status 1 when something failed and *exit_on_failure* is true.
    """
    passed = failed = skipped = 0

    for test in tests:
        name = getattr(test, "__name__", str(test))
        try:
            result = test()
        except TestSkipped as exc:
            print(f"  SKIPPED: {name} ({exc})")
            skipped += 1
        except AssertionError as exc:
            print(f"  FAILED: {name}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001 - report, do not stop the suite
            print(f"  ERROR: {name}: {type(exc).__name__}: {exc}")
            failed += 1
        else:
            if result is False:
                print(f"  FAILED: {name}: returned False")
                failed += 1
            else:
                passed += 1
        print()

    total = len(tests)
    counts = {"passed": passed, "failed": failed, "skipped": skipped, "total": total}

    print("=" * 60)
    print(
        f"Results: {passed}/{total} passed, {failed} failed, {skipped} skipped"
    )
    print("=" * 60)

    if failed and exit_on_failure:
        raise SystemExit(1)
    return counts
