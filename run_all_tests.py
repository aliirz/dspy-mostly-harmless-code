#!/usr/bin/env python3
"""
run_all_tests.py — The Whole Sort of General Testing Thing
DSPy: The Mostly Harmless Guide

Runs every test across all 7 chapters and prints a unified report.
Usage:
    python run_all_tests.py              # Run everything
    python run_all_tests.py --chapter 3  # Run only chapter 3
    python run_all_tests.py --fast       # Skip live API tests (structural only)

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
# Each entry: (directory_name, test_command_args, description)
CHAPTERS = [
    {
        "num": 1,
        "dir": "ch01_dont_panic",
        "test_file": "test_startup_roaster.py",
        "runner": "python",  # custom main() runner
        "title": "Don't Panic — Startup Idea Roaster",
    },
    {
        "num": 2,
        "dir": "ch02_restaurant_pipeline",
        "test_file": "test_all_chapter_examples.py",
        "runner": "python",
        "title": "Restaurant at the End of the Pipeline — Lead Intelligence Engine",
    },
    {
        "num": 3,
        "dir": "ch03_retrieval",
        "test_file": "test_all_chapter_examples.py",
        "runner": "python",
        "title": "Life, the Universe, and Retrieval — Codebase Q&A System",
    },
    {
        "num": 4,
        "dir": "ch04_babel_fish",
        "test_file": "test_all_chapter_examples.py",
        "runner": "python",
        "title": "The Babel Fish — Customer Support Ticket Classifier",
    },
    {
        "num": 5,
        "dir": "ch05_agents",
        "test_file": "test_all_chapter_examples.py",
        "runner": "python",
        "title": "So Long, and Thanks for All the Prompts — Agents",
    },
    {
        "num": 6,
        "dir": "ch06_production",
        "test_file": "test_all_chapter_examples.py",
        "runner": "pytest",
        "title": "Mostly Harmless (in Production) — Content Moderation",
    },
    {
        "num": 7,
        "dir": "ch07_advanced",
        "test_file": "test_all_chapter_examples.py",
        "runner": "pytest",
        "title": "The Answer Is 42 (Tokens) — Multimodal & Advanced",
    },
]


def find_code_root() -> Path:
    """Find the code/ directory relative to this script."""
    return Path(__file__).resolve().parent


def run_chapter(chapter: dict, code_root: Path, fast: bool = False) -> dict:
    """
    Run a single chapter's tests. Returns a result dict with:
        passed: bool, output: str, duration: float
    """
    chapter_dir = code_root / chapter["dir"]

    if not chapter_dir.exists():
        return {
            "passed": False,
            "output": f"Directory not found: {chapter_dir}",
            "duration": 0.0,
        }

    test_file = chapter_dir / chapter["test_file"]
    if not test_file.exists():
        return {
            "passed": False,
            "output": f"Test file not found: {test_file}",
            "duration": 0.0,
        }

    # Build the command
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    if chapter["runner"] == "pytest":
        cmd = [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"]
        if fast:
            # Skip tests marked with live API markers
            cmd += ["-k", "not live and not api"]
    else:
        cmd = [sys.executable, str(test_file)]
        if fast:
            env["SKIP_LIVE_TESTS"] = "1"

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(chapter_dir),
            capture_output=True,
            text=True,
            timeout=300,  # 5 min per chapter max
            env=env,
        )
        duration = time.time() - start
        output = result.stdout
        if result.stderr:
            output += "\n--- stderr ---\n" + result.stderr

        return {
            "passed": result.returncode == 0,
            "output": output,
            "duration": duration,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "output": "TIMEOUT: Tests took longer than 5 minutes",
            "duration": 300.0,
        }
    except Exception as e:
        return {
            "passed": False,
            "output": f"ERROR running tests: {e}",
            "duration": time.time() - start,
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
        "--chapter", "-c", type=int, help="Run only this chapter number (1-7)"
    )
    parser.add_argument(
        "--fast", "-f", action="store_true",
        help="Skip live API tests (structural tests only)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show full output from each chapter"
    )
    args = parser.parse_args()

    code_root = find_code_root()

    # Filter chapters if --chapter specified
    chapters_to_run = CHAPTERS
    if args.chapter:
        chapters_to_run = [c for c in CHAPTERS if c["num"] == args.chapter]
        if not chapters_to_run:
            print(f"No chapter {args.chapter} found. Valid: 1-7")
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
        summary = extract_summary_line(result["output"])
        print(f"  {status}  ({result['duration']:.1f}s)  {summary}")
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
            summary = extract_summary_line(r["output"])
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
