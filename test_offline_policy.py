"""Guard rails for the test suite itself.

Four promises the book's promise depends on:

1. ``SKIP_LIVE_TESTS=1`` means no provider call — even when a real API key is
   exported in the environment, and even when a chapter's tests are run by
   hand with ``python -m pytest`` or ``python test_*.py`` instead of through
   ``run_all_tests.py``. Script-style chapters enforce this with the same
   decorators via ``TestSkipped``, not with an ``if`` block only ``main()``
   honours.
2. A key for one provider is not a credential for another: an
   ``OPENAI_API_KEY`` never enables an Anthropic live test and vice versa.
3. Every test file in every chapter directory is registered in
   ``run_all_tests.py``; a chapter cannot grow a test file nobody runs.
4. An offline run reaches no network at all (``offline_guard/sitecustomize.py``
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

#: Key sets used to prove provider scoping without a single live call.
KEY_COMBOS = {
    "none": {},
    "anthropic": {"ANTHROPIC_API_KEY": DUMMY_KEYS["ANTHROPIC_API_KEY"]},
    "openai": {"OPENAI_API_KEY": DUMMY_KEYS["OPENAI_API_KEY"]},
    "all": dict(DUMMY_KEYS),
}

PROBE_COMBOS = [
    {"name": "offline_all", "offline": True, "keys": "all"},
    {"name": "live_none", "offline": False, "keys": "none"},
    {"name": "live_anthropic", "offline": False, "keys": "anthropic"},
    {"name": "live_openai", "offline": False, "keys": "openai"},
    {"name": "live_all", "offline": False, "keys": "all"},
]


def chapter_env(offline: bool, keys: str = "all") -> dict:
    """Environment for a child run: dummy keys in, real keys and network out."""
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env.update(KEY_COMBOS[keys])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(GUARD_DIR)
    env.pop("SKIP_LIVE_TESTS", None)
    if offline:
        env["SKIP_LIVE_TESTS"] = "1"
    return env


def run(cmd, cwd, offline=True, keys="all"):
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=600,
        env=chapter_env(offline, keys),
    )


def chapter_table() -> list:
    sys.path.insert(0, str(REPO_ROOT))
    import run_all_tests
    return run_all_tests.CHAPTERS


def registered_files() -> list:
    return [
        (chapter["dir"], name)
        for chapter in chapter_table()
        for name in chapter["test_files"]
    ]


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
        f"registry lists files that do not exist: {sorted(registered)}"
    )


def test_registry_covers_every_chapter_directory():
    registered = {chapter["dir"] for chapter in chapter_table()}
    on_disk = {path.name for path in REPO_ROOT.glob("ch*") if path.is_dir()}
    assert on_disk == registered


# ---------------------------------------------------------------------------
# 2. Provider scoping: keys, not the flag alone, decide *which* tests can run
# ---------------------------------------------------------------------------

#: Each combo is imported in its own module name inside one child process, so
#: the decorators re-evaluate against that combo's environment.
MARKER_PROBE = """
import importlib.util, json, sys

path = sys.argv[1]
combos = json.loads(sys.argv[2])
out = {}

def scan(name, keys):
    spec = importlib.util.spec_from_file_location("chapter_probe_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    found = {}
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if not attr.startswith("test_") or not callable(obj):
            continue
        skipifs = [
            m for m in getattr(obj, "pytestmark", [])
            if getattr(m, "name", "") == "skipif" and m.args
        ]
        if not skipifs:
            continue
        providers = sorted(getattr(obj, "__live_providers__", ()))
        found[attr] = {
            "would_skip": any(bool(m.args[0]) for m in skipifs),
            "providers": providers,
            # gates beyond the provider ones (network, Deno, source tree) are
            # environment facts, not credential policy
            "extra_gates": len(getattr(obj, "__chapter_gates__", ())) - len(providers),
        }
    return found

import os
real = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
for combo in combos:
    for key in list(os.environ):
        if key.endswith("_API_KEY") or key == "SKIP_LIVE_TESTS":
            del os.environ[key]
    os.environ.update(combo["keys"])
    if combo["offline"]:
        os.environ["SKIP_LIVE_TESTS"] = "1"
    out[combo["name"]] = scan(combo["name"], combo["keys"])
print("PROBE:" + json.dumps(out))
"""


_PROBE_CACHE: dict = {}


def probe_markers(chapter_dir: str, test_file: str) -> dict:
    """Import the file once per environment combo and report its gating."""
    key = (chapter_dir, test_file)
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    chapter_path = REPO_ROOT / chapter_dir
    combos = [_with_keys(c, KEY_COMBOS[c["keys"]]) for c in PROBE_COMBOS]
    result = run(
        [sys.executable, "-c", MARKER_PROBE, str(chapter_path / test_file), json.dumps(combos)],
        cwd=chapter_path,
        offline=True,  # the guard stays on for every probe: nothing may dial out
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(l for l in result.stdout.splitlines() if l.startswith("PROBE:"))
    _PROBE_CACHE[key] = json.loads(line[len("PROBE:"):])
    return _PROBE_CACHE[key]


def _with_keys(combo: dict, keys: dict) -> dict:
    return {"name": combo["name"], "offline": combo["offline"], "keys": keys}


@pytest.mark.parametrize("chapter_dir,test_file", registered_files())
def test_offline_flag_skips_every_live_marker(chapter_dir, test_file):
    """With dummy keys present and SKIP_LIVE_TESTS=1, every live test skips."""
    marks = probe_markers(chapter_dir, test_file)["offline_all"]
    live = {name: v for name, v in marks.items() if v["providers"]}
    assert live, f"{chapter_dir}/{test_file} exposes no live-test markers"
    enabled = {name for name, v in live.items() if not v["would_skip"]}
    assert enabled == set(), (
        f"{chapter_dir}/{test_file}: these live tests would still run under "
        f"SKIP_LIVE_TESTS=1: {sorted(enabled)}"
    )


@pytest.mark.parametrize("chapter_dir,test_file", registered_files())
def test_live_tests_stay_enabled_without_the_flag(chapter_dir, test_file):
    """Without SKIP_LIVE_TESTS, keys present, the same tests are *not* skipped.

    Together with the test above this pins the switch to the flag rather than
    to a missing key, so the markers are not silently dead.
    """
    marks = probe_markers(chapter_dir, test_file)["live_all"]
    live = {
        name: v for name, v in marks.items()
        if v["providers"] and not v["extra_gates"]
    }
    if not live:
        gated = sorted(name for name, v in marks.items() if v["providers"])
        assert gated, f"{chapter_dir}/{test_file} exposes no live-test markers"
        pytest.skip(
            f"{chapter_dir}/{test_file}: every live test also needs a network "
            f"connection or Deno ({', '.join(gated)}); this sandbox cannot "
            "satisfy those, so only the offline half of the gate is testable here"
        )
    skipped = {name for name, v in live.items() if v["would_skip"]}
    assert skipped == set(), (
        f"{chapter_dir}/{test_file}: skipped even though live mode is allowed "
        f"and every dummy key is present: {sorted(skipped)}"
    )


@pytest.mark.parametrize("chapter_dir,test_file", registered_files())
def test_credentials_are_scoped_to_their_provider(chapter_dir, test_file):
    """One provider's key never unlocks another provider's live test."""
    probe = probe_markers(chapter_dir, test_file)
    live = {
        name: v for name, v in probe["live_all"].items()
        if v["providers"] and not v["extra_gates"]
    }
    if not live:
        pytest.skip(f"{chapter_dir}/{test_file} has no provider-scoped live tests")

    for combo, available in (("live_none", set()),
                             ("live_anthropic", {"anthropic"}),
                             ("live_openai", {"openai"})):
        for name, base in live.items():
            needs = set(base["providers"])
            expected_skip = not needs.issubset(available)
            actual = probe[combo][name]["would_skip"]
            assert actual == expected_skip, (
                f"{chapter_dir}/{test_file}::{name} ({'/'.join(sorted(needs))}) "
                f"under keys={combo}: expected would_skip={expected_skip}, got {actual}"
            )


def test_provider_helpers_reject_unknown_providers():
    sys.path.insert(0, str(REPO_ROOT))
    import chapter_test_support as cts

    with pytest.raises(ValueError):
        cts.provider_key_vars("not-a-provider")


def test_llm_api_key_counts_as_anthropic_only():
    """``LLM_API_KEY`` builds a Claude LM in ch06/ch07, so it is Anthropic's."""
    sys.path.insert(0, str(REPO_ROOT))
    import chapter_test_support as cts

    assert "LLM_API_KEY" in cts.provider_key_vars("anthropic")
    assert "LLM_API_KEY" not in cts.provider_key_vars("openai")


# ---------------------------------------------------------------------------
# 3. Direct calls skip too — the second entry point cannot be bypassed
# ---------------------------------------------------------------------------

DIRECT_CALL_PROBE = """
import importlib.util, sys
sys.path.insert(0, sys.argv[2])
spec = importlib.util.spec_from_file_location("chapter_direct", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import chapter_test_support as cts

name = sys.argv[3]
try:
    getattr(mod, name)()
except cts.TestSkipped as exc:
    print("SKIPPED:" + str(exc))
else:
    raise SystemExit("live test ran even though SKIP_LIVE_TESTS=1")
"""


@pytest.mark.parametrize("chapter_dir,test_file", registered_files())
def test_direct_call_honours_the_offline_flag(chapter_dir, test_file):
    """Calling a live test by hand (not via main(), not via pytest) skips."""
    probe = probe_markers(chapter_dir, test_file)["offline_all"]
    live = sorted(name for name, v in probe.items() if v["providers"])
    if not live:
        pytest.skip(f"{chapter_dir}/{test_file} has no provider-scoped live tests")
    chapter_path = REPO_ROOT / chapter_dir
    result = run(
        [sys.executable, "-c", DIRECT_CALL_PROBE,
         str(chapter_path / test_file), str(REPO_ROOT), live[0]],
        cwd=chapter_path,
        offline=True,
    )
    output = result.stdout + result.stderr
    assert "blocked a network call" not in output, output
    assert result.returncode == 0, output
    assert "SKIPPED:" in output, output


# ---------------------------------------------------------------------------
# 4. Both entry points are green offline, with keys present
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chapter_dir,test_file", registered_files())
def test_pytest_entry_point_is_green_offline_with_dummy_keys(chapter_dir, test_file):
    result = run(
        [sys.executable, "-m", "pytest", test_file, "-q", "-rs", "-p", "no:cacheprovider",
         "--tb=short"],
        cwd=REPO_ROOT / chapter_dir,
        offline=True,
    )
    output = result.stdout + result.stderr
    assert "blocked a network call" not in output, output
    assert result.returncode == 0, output
    assert "skipped" in output, f"{chapter_dir}/{test_file} skipped nothing:\n{output}"


@pytest.mark.parametrize("chapter_dir,test_file", registered_files())
def test_script_entry_point_is_green_offline_with_dummy_keys(chapter_dir, test_file):
    result = run([sys.executable, test_file], cwd=REPO_ROOT / chapter_dir, offline=True)
    output = result.stdout + result.stderr
    assert "blocked a network call" not in output, output
    assert result.returncode == 0, output
    assert "FAILED" not in output and "ERROR" not in output, output
    assert "SKIPPED" in output, f"{chapter_dir}/{test_file} skipped nothing:\n{output}"


# ---------------------------------------------------------------------------
# 5. Importing a chapter's tests never dials out
# ---------------------------------------------------------------------------

#: Import the test module with the network primitives replaced by recorders.
#: The recorder raises (so a swallowed exception still leaves a trace) and the
#: probe reports every attempt it saw. ch05 used to probe GitHub at import time
#: and hide the failure behind `except Exception`, so `--fast` looked clean
#: while a direct pytest run still went out to api.github.com.
IMPORT_PROBE = """
import importlib.util, json, socket, sys

attempts = []

def record(what, target):
    attempts.append(what + " " + repr(target))
    raise RuntimeError("offline test run blocked a network call (" + what + " " + repr(target) + ")")

import requests
requests.get = lambda url, *a, **kw: record("requests.get", url)
requests.Session.request = lambda self, method, url, *a, **kw: record("requests.Session.request " + str(method), url)

real_connect = socket.socket.connect
def connect(self, address):
    if self.family == socket.AF_UNIX:
        return real_connect(self, address)
    return record("socket.connect", address)
socket.socket.connect = connect
socket.create_connection = lambda address, *a, **kw: record("socket.create_connection", address)
socket.getaddrinfo = lambda host, port, *a, **kw: record("socket.getaddrinfo", (host, port))

sys.path.insert(0, sys.argv[2])
spec = importlib.util.spec_from_file_location("chapter_import_probe", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("PROBE:" + json.dumps({
    "attempts": attempts,
    "has_network": getattr(mod, "HAS_NETWORK", None),
}))
"""


@pytest.mark.parametrize("chapter_dir,test_file", registered_files())
def test_import_has_no_network_side_effect(chapter_dir, test_file):
    """Importing a chapter's tests offline makes no request at all."""
    chapter_path = REPO_ROOT / chapter_dir
    result = run(
        [sys.executable, "-c", IMPORT_PROBE, str(chapter_path / test_file), str(REPO_ROOT)],
        cwd=chapter_path,
        offline=True,
        keys="none",  # credentials must not be the reason a probe stays quiet
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    line = next(l for l in result.stdout.splitlines() if l.startswith("PROBE:"))
    report = json.loads(line[len("PROBE:"):])
    assert report["attempts"] == [], (
        f"{chapter_dir}/{test_file} dials out just by being imported: {report['attempts']}"
    )
    assert report["has_network"] is not True, (
        f"{chapter_dir}/{test_file} claims network access in an offline run"
    )


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
