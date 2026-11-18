#!/usr/bin/env python3
"""
run_all_tests.py — The Whole Sort of General Testing Thing
DSPy: The Mostly Harmless Guide

Runs every test across all 8 chapters and prints a unified report.
Usage:
    python run_all_tests.py              # Run everything
    python run_all_tests.py --chapter 3  # Run only chapter 3
    python run_all_tests.py --fast       # Skip live API tests (structural only)
    python run_all_tests.py --no-guard   # Skip the offline-policy self-check

--fast sets SKIP_LIVE_TESTS=1 for each chapter. Every test file reads that flag
through chapter_test_support.py, so "skipped" is reported separately from
"passed" instead of a chapter hard-failing because an API key is missing — and
a real key in the environment cannot re-enable paid calls under --fast.

Before any chapter runs, the harness runs test_offline_policy.py, which fails
if a chapter ignores SKIP_LIVE_TESTS, if a chapter test file is missing from the
registry below, or if an offline run reaches the network.

Requirements:
    pip install dspy-ai python-dotenv fastapi uvicorn Pillow requests pytest

Exit code 0 = all chapters green, 1 = at least one failure.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Chapter registry ────────────────────────────────────────────────
# Each entry registers one chapter directory and *every* test file in it;
# test_offline_policy.py fails if a chapter grows a test file this list misses.
CHAPTERS = [
    {
        "num": 1,
        "dir": "ch01_dont_panic",
        "test_files": ["test_startup_roaster.py"],
        "runner": "python",  # custom main() runner
        "title": "Don't Panic — Startup Idea Roaster",
    },
    {
        "num": 2,
        "dir": "ch02_restaurant_pipeline",
        "test_files": ["test_all_chapter_examples.py", "test_lead_engine.py"],
        "runner": "python",
        "title": "Restaurant at the End of the Pipeline — Lead Intelligence Engine",
    },
    {
        "num": 3,
        "dir": "ch03_retrieval",
        "test_files": ["test_all_chapter_examples.py"],
        "runner": "python",
        "title": "Life, the Universe, and Retrieval — Codebase Q&A System",
    },
    {
        "num": 4,
        "dir": "ch04_babel_fish",
        "test_files": ["test_all_chapter_examples.py"],
        "runner": "python",
        "title": "The Babel Fish — Customer Support Ticket Classifier",
    },
    {
        "num": 5,
        "dir": "ch05_agents",
        "test_files": ["test_all_chapter_examples.py"],
        "runner": "python",
        "title": "So Long, and Thanks for All the Prompts — Agents",
    },
    {
        "num": 6,
        "dir": "ch06_production",
        "test_files": ["test_all_chapter_examples.py"],
        "runner": "pytest",
        "title": "Mostly Harmless (in Production) — Content Moderation",
    },
    {
        "num": 7,
        "dir": "ch07_advanced",
        "test_files": ["test_all_chapter_examples.py"],
        "runner": "pytest",
        "title": "The Answer Is 42 (Tokens) — Multimodal & Advanced",
    },
    {
        "num": 8,
        "dir": "ch08_rlm",
        "test_files": ["test_contract_engine.py"],
        "runner": "python",
        "title": "The Infinite Improbability Context — RLM Contract Engine",
    },
]


def find_code_root() -> Path:
    """Find the code/ directory relative to this script."""
    return Path(__file__).resolve().parent


def run_chapter(chapter: dict, code_root: Path, fast: bool = False) -> dict:
    """
    Run every test file registered for a chapter. Returns a result dict with:
        passed: bool, output: str, duration: float, summaries: list[str]
    A chapter passes only when all of its test files pass.
    """
    chapter_dir = code_root / chapter["dir"]

    if not chapter_dir.exists():
        return {
            "passed": False,
            "output": f"Directory not found: {chapter_dir}",
            "duration": 0.0,
            "summaries": [],
        }

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    if fast:
        # Every test file reads this through chapter_test_support.py: script
        # runners skip their own live list, pytest runners skip live-marked
        # tests. No -k expression is involved, so a real API key in the
        # environment cannot turn a --fast run into a paid run.
        env["SKIP_LIVE_TESTS"] = "1"

    outputs = []
    summaries = []
    passed = True
    start = time.time()

    for name in chapter["test_files"]:
        test_file = chapter_dir / name
        if not test_file.exists():
            outputs.append(f"Test file not found: {test_file}")
            summaries.append(f"{name}: MISSING")
            passed = False
            continue

        if chapter["runner"] == "pytest":
            cmd = [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"]
        else:
            cmd = [sys.executable, str(test_file)]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(chapter_dir),
                capture_output=True,
                text=True,
                timeout=300,  # 5 min per test file max
                env=env,
            )
            output = result.stdout
            if result.stderr:
                output += "\n--- stderr ---\n" + result.stderr
            passed = passed and result.returncode == 0
            outputs.append(output)
            summaries.append(f"{name}: {extract_summary_line(output)}")
        except subprocess.TimeoutExpired:
            passed = False
            text = f"TIMEOUT: {name} took longer than 5 minutes"
            outputs.append(text)
            summaries.append(f"{name}: TIMEOUT")
        except Exception as e:
            passed = False
            text = f"ERROR running {name}: {e}"
            outputs.append(text)
            summaries.append(f"{name}: ERROR")

    return {
        "passed": passed,
        "output": "\n".join(outputs),
        "duration": time.time() - start,
        "summaries": summaries,
    }


def extract_summary_line(output: str) -> str:
    """Pull out the results summary line from test output."""
    lines = output.strip().splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        # pytest style: "=== 20 passed in 45.32s ==="  or "=== 18 passed, 2 failed ==="
        if "passed" in stripped and ("=" in stripped or "failed" in stripped):
            return stripped
        # Custom runner style: "Results: 8/8 passed, 0 failed"
        if stripped.startswith("Results:"):
            return stripped
    # Fallback: last non-empty line
    for line in reversed(lines):
        if line.strip():
            return line.strip()[:100]
    return "(no output)"


def main():
    parser = argparse.ArgumentParser(
        description="Run all DSPy book tests — The Whole Sort of General Testing Thing"
    )
    parser.add_argument(
        "--chapter", "-c", type=int, help="Run only this chapter number (1-8)"
    )
    parser.add_argument(
        "--fast", "-f", action="store_true",
        help="Skip live API tests (structural tests only)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show full output from each chapter"
    )
    parser.add_argument(
        "--no-guard", action="store_true",
        help="Skip the offline-policy self-check (test_offline_policy.py)"
    )
    args = parser.parse_args()

    code_root = find_code_root()

    # Filter chapters if --chapter specified
    chapters_to_run = CHAPTERS
    if args.chapter:
        chapters_to_run = [c for c in CHAPTERS if c["num"] == args.chapter]
        if not chapters_to_run:
            print(f"No chapter {args.chapter} found. Valid: 1-8")
            sys.exit(1)

    # ── Banner ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  DSPy: The Mostly Harmless Guide — Test Suite")
    print("  The Whole Sort of General Testing Thing")
    if args.fast:
        print("  Mode: FAST (structural tests only, no API calls)")
    else:
        print("  Mode: FULL (structural + live API tests)")
    print("=" * 70)
    print()

    # ── Harness self-check ──────────────────────────────────────────
    if not args.no_guard:
        print(f"{'─' * 70}")
        print("  Harness self-check: offline policy, registry coverage, no network")
        print(f"{'─' * 70}")
        guard = subprocess.run(
            [sys.executable, "-m", "pytest", str(code_root / "test_offline_policy.py"),
             "-q", "--tb=short"],
            cwd=str(code_root),
            capture_output=True,
            text=True,
        )
        guard_output = guard.stdout + (guard.stderr or "")
        if args.verbose:
            print(guard_output)
        print(f"  {'PASS ✓' if guard.returncode == 0 else 'FAIL ✗'}  "
              f"{extract_summary_line(guard_output)}")
        print()
        if guard.returncode != 0:
            print(guard_output)
            print("  The offline policy is broken; fix it before trusting any")
            print("  chapter result below. Re-run with --no-guard to skip this check.")
            print()
            sys.exit(1)

    # ── Run each chapter ────────────────────────────────────────────
    results = {}
    total_start = time.time()

    for chapter in chapters_to_run:
        label = f"Ch{chapter['num']}: {chapter['title']}"
        print(f"{'─' * 70}")
        print(f"  Running {label}...")
        print(f"{'─' * 70}")

        result = run_chapter(chapter, code_root, fast=args.fast)
        results[chapter["num"]] = {**result, "label": label}

        if args.verbose:
            print(result["output"])
            print()

        # Show quick status
        status = "PASS ✓" if result["passed"] else "FAIL ✗"
        print(f"  {status}  ({result['duration']:.1f}s)")
        for summary in result["summaries"]:
            print(f"         {summary}")
        print()

    total_duration = time.time() - total_start

    # ── Final Report ────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  FINAL REPORT")
    print("=" * 70)
    print()

    all_passed = True
    for num in sorted(results.keys()):
        r = results[num]
        icon = "✓" if r["passed"] else "✗"
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{icon}] {status}  {r['label']}  ({r['duration']:.1f}s)")
        if not r["passed"]:
            all_passed = False
            # Show summary of failure
            for summary in r["summaries"]:
                print(f"         └─ {summary}")

    passed_count = sum(1 for r in results.values() if r["passed"])
    total_count = len(results)

    print()
    print(f"  {'─' * 50}")
    print(f"  Chapters: {passed_count}/{total_count} passed")
    print(f"  Total time: {total_duration:.1f}s")
    print(f"  {'─' * 50}")

    if all_passed:
        print()
        print("  Don't Panic — All tests passed! 🎉")
        print()
    else:
        print()
        print("  Some chapters failed. Run with --verbose to see full output,")
        print("  or --chapter N to re-run a specific chapter.")
        print()
        # Show failed chapter outputs
        print("=" * 70)
        print("  FAILURE DETAILS")
        print("=" * 70)
        for num in sorted(results.keys()):
            r = results[num]
            if not r["passed"]:
                print(f"\n{'─' * 70}")
                print(f"  {r['label']}")
                print(f"{'─' * 70}")
                # Show last 40 lines of output to keep it manageable
                lines = r["output"].strip().splitlines()
                if len(lines) > 40:
                    print("  ... (truncated, showing last 40 lines) ...")
                    for line in lines[-40:]:
                        print(f"  {line}")
                else:
                    for line in lines:
                        print(f"  {line}")
        print()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
