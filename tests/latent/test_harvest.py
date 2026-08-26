"""RED/GREEN tests for the sensorium-backed harvest runner (prereg §4/§5.1).

No GPU needed: every sample is a tiny pure-Python function, run as a
`sensorium run` subprocess under a small RLIMIT_AS / wall-clock timeout
(crucible.latent.config). Honesty rules apply here same as everywhere in
this repo: a timeout is reported as outcome="timeout", not swallowed as a
hang or invented as a return; truncation (ours or sensorium's own) is
counted in `truncated`, never silently dropped; two runs disagreeing on
outcome or captured state come back as `deterministic=False`, not averaged
away.
"""
import shutil
import sys
from pathlib import Path

from crucible.latent.harvest import HarvestResult, Snapshot, harvest


def test_sensorium_importable():
    """Step 0 pin: sensorium is installed editable into THIS venv."""
    import sensorium.store  # noqa: F401

    venv_bin = str(Path(sys.executable).parent)
    assert shutil.which("sensorium", path=venv_bin) is not None


def test_harvest_clean_return(tmp_path):
    r = harvest("def f(a, b):\n    c = a + b\n    return c\n", "(2, 3)", tmp_path)
    assert r.outcome == "return" and r.return_repr == "5" and r.deterministic
    assert any("c" in [n for n, _, _ in s.locals] for s in r.snapshots)   # locals really captured


def test_harvest_exception_names_the_type(tmp_path):
    r = harvest("def f(a):\n    return a[10]\n", "([1],)", tmp_path)
    assert r.outcome == "exception:IndexError" and r.return_repr is None


def test_harvest_timeout_is_marked_not_hung(tmp_path):
    r = harvest("def f():\n    while True:\n        pass\n", "()", tmp_path)
    assert r.outcome == "timeout" and r.truncated


def test_harvest_nondeterminism_is_detected(tmp_path):
    r = harvest("import random\ndef f():\n    return random.random()\n", "()", tmp_path)
    assert r.deterministic is False


def test_harvest_returns_frozen_dataclasses(tmp_path):
    """Interface shape: HarvestResult/Snapshot are frozen dataclasses."""
    r = harvest("def f():\n    return 1\n", "()", tmp_path)
    assert isinstance(r, HarvestResult)
    for s in r.snapshots:
        assert isinstance(s, Snapshot)


def test_harvest_locals_are_name_sorted(tmp_path):
    """Task 4 depends on each snapshot's locals tuple being name-sorted."""
    r = harvest(
        "def f(z, a, m):\n    total = z + a + m\n    return total\n",
        "(1, 2, 3)", tmp_path,
    )
    for s in r.snapshots:
        names = [n for n, _, _ in s.locals]
        assert names == sorted(names)


def test_harvest_leaves_no_traces_outside_workdir(tmp_path):
    """SENSORIUM_DIR must be scoped inside workdir -- never ~/.sensorium."""
    home_traces = Path.home() / ".sensorium" / "traces"
    before = set(home_traces.glob("*.db")) if home_traces.exists() else set()
    harvest("def f():\n    return 1\n", "()", tmp_path)
    after = set(home_traces.glob("*.db")) if home_traces.exists() else set()
    assert after == before
