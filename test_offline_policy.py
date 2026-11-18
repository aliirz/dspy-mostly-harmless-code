"""Guard rails for the test suite itself.

Three promises the book's promise depends on:

1. ``SKIP_LIVE_TESTS=1`` means no provider call — even when a real API key is
   exported in the environment, and even when a chapter's tests are run by
   hand with ``python -m pytest`` instead of through ``run_all_tests.py``.
2. Every test file in every chapter directory is registered in
   ``run_all_tests.py``; a chapter cannot grow a test file nobody runs.
3. An offline run reaches no network at all (``offline_guard/sitecustomize.py``
   turns any outbound socket into an error), so a regression fails here rather
   than on an invoice.

Run with::

    python -m pytest test_offline_policy.py -v
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
GUARD_DIR = REPO_ROOT / "offline_guard"

#: Credentials nobody can bill. Live tests must ignore them under
#: SKIP_LIVE_TESTS=1; if they do not, the run fails instead of calling out.
DUMMY_KEYS = {
    "LLM_API_KEY": "sk-offline-policy-dummy",
    "ANTHROPIC_API_KEY": "sk-ant-offline-policy-dummy",
    "OPENAI_API_KEY": "sk-openai-offline-policy-dummy",
}

PYTEST_CHAPTERS = {"ch06_production": "test_all_chapter_examples.py",
                   "ch07_advanced": "test_all_chapter_examples.py"}


def chapter_env(offline: bool) -> dict:
    """Environment for a child run: dummy keys in, real keys and network out."""
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env.update(DUMMY_KEYS)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(GUARD_DIR)
    env.pop("SKIP_LIVE_TESTS", None)
    if offline:
        env["SKIP_LIVE_TESTS"] = "1"
    return env


def run(cmd, cwd, offline=True):
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=300,
        env=chapter_env(offline),
    )


def chapter_table() -> list:
    sys.path.insert(0, str(REPO_ROOT))
    import run_all_tests
    return run_all_tests.CHAPTERS


# ---------------------------------------------------------------------------
# 1. Registry completeness
# ---------------------------------------------------------------------------

def test_registry_covers_every_test_file():
    """Every test_*.py in a chapter directory must be registered."""
    registered = {
        (chapter["dir"], name)
        for chapter in chapter_table()
        for name in chapter["test_files"]
    }
    on_disk = {
        (path.parent.name, path.name)
        for path in REPO_ROOT.glob("ch*/test_*.py")
    }
    assert on_disk - registered == set(), (
        "unregistered chapter test files: "
        f"{sorted(on_disk - registered)} — add them to run_all_tests.CHAPTERS"
    )
    assert registered - on_disk == set(), (
        f"registry lists files that do not exist: {sorted(registered - on_disk)}"
    )


def test_registry_covers_every_chapter_directory():
    registered = {chapter["dir"] for chapter in chapter_table()}
    on_disk = {path.name for path in REPO_ROOT.glob("ch*") if path.is_dir()}
    assert on_disk == registered


# ---------------------------------------------------------------------------
# 2. The flag, not a missing key, decides whether live tests run
# ---------------------------------------------------------------------------

MARKER_PROBE = """
import importlib.util, json, sys

spec = importlib.util.spec_from_file_location("chapter_tests", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

marks = {}
for name in dir(mod):
    obj = getattr(mod, name)
    if not name.startswith("test_") or not callable(obj):
        continue
    skipifs = [m for m in getattr(obj, "pytestmark", []) if getattr(m, "name", "") == "skipif"]
    if skipifs:
        marks[name] = [bool(m.args[0]) for m in skipifs if m.args]
print("PROBE:" + json.dumps(marks))
"""


def probe_skipif(chapter_dir: str, test_file: str, offline: bool) -> dict:
    chapter_path = REPO_ROOT / chapter_dir
    result = run(
        [sys.executable, "-c", MARKER_PROBE, str(chapter_path / test_file)],
        cwd=chapter_path,
        offline=offline,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(l for l in result.stdout.splitlines() if l.startswith("PROBE:"))
    return json.loads(line[len("PROBE:"):])


@pytest.mark.parametrize("chapter_dir,test_file", sorted(PYTEST_CHAPTERS.items()))
def test_offline_flag_skips_every_live_marker(chapter_dir, test_file):
    """With dummy keys present and SKIP_LIVE_TESTS=1, every live test skips."""
    marks = probe_skipif(chapter_dir, test_file, offline=True)
    assert marks, f"{chapter_dir} exposes no live-test markers"
    enabled = {name: conds for name, conds in marks.items() if not all(conds)}
    assert enabled == {}, (
        f"these live tests would still run under SKIP_LIVE_TESTS=1: {sorted(enabled)}"
    )


@pytest.mark.parametrize("chapter_dir,test_file", sorted(PYTEST_CHAPTERS.items()))
def test_live_tests_stay_enabled_without_the_flag(chapter_dir, test_file):
    """Without SKIP_LIVE_TESTS the same tests are *not* skipped.

    Together with the test above this pins the switch to the flag: the
    markers are not silently dead.
    """
    marks = probe_skipif(chapter_dir, test_file, offline=False)
    skipped = {name: conds for name, conds in marks.items() if all(conds)}
    assert skipped == {}, (
        f"these live tests are skipped even though live mode is allowed: {sorted(skipped)}"
    )


# ---------------------------------------------------------------------------
# 3. Offline runs make no calls and stay green
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chapter_dir,test_file", sorted(PYTEST_CHAPTERS.items()))
def test_pytest_chapter_is_green_offline_with_dummy_keys(chapter_dir, test_file):
    result = run(
        [sys.executable, "-m", "pytest", test_file, "-q", "-rs", "--tb=short"],
        cwd=REPO_ROOT / chapter_dir,
        offline=True,
    )
    output = result.stdout + result.stderr
    assert "blocked a network call" not in output, output
    assert result.returncode == 0, output
    assert "skipped" in output, f"{chapter_dir} reported no skipped live tests:\n{output}"


@pytest.mark.parametrize("chapter_dir", ["ch01_dont_panic", "ch02_restaurant_pipeline",
                                         "ch03_retrieval", "ch04_babel_fish",
                                         "ch05_agents", "ch08_rlm"])
def test_script_chapter_is_green_offline_with_dummy_keys(chapter_dir):
    """Script-style chapters skip their live test instead of calling out."""
    chapter = next(c for c in chapter_table() if c["dir"] == chapter_dir)
    for test_file in chapter["test_files"]:
        result = run([sys.executable, test_file], cwd=REPO_ROOT / chapter_dir, offline=True)
        output = result.stdout + result.stderr
        assert "blocked a network call" not in output, output
        assert result.returncode == 0, output
        assert "FAILED" not in output and "ERROR" not in output, output
        assert "SKIPPED" in output, f"{chapter_dir}/{test_file} skipped nothing:\n{output}"


def test_network_guard_actually_blocks():
    """Sanity check: the offline guard is not a no-op."""
    probe = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=1)\n"
        "except RuntimeError as exc:\n"
        "    print('BLOCKED', exc)\n"
        "else:\n"
        "    raise SystemExit('connect was allowed')\n"
    )
    result = run([sys.executable, "-c", probe], cwd=REPO_ROOT, offline=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BLOCKED" in result.stdout
