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


def _n_assignments_src(n: int) -> str:
    """A no-arg function with exactly N single-name assignments followed by
    a `return`. Each assignment's LINE event fires (delta reported) at the
    FOLLOWING line -- sensorium's LINE events fire before their own line
    runs, so a line's effect is only visible on the next captured line, see
    tracer.py's module docstring -- which is why a trailing `return` after
    the Nth assignment is required: without it, the Nth assignment's delta
    would never be flushed into an event at all. Verified directly (not
    just asserted): N assignments + a trailing return produces EXACTLY N
    LINE events, for both N=3 and N=5 used below.
    """
    body = "".join(f"    v{i} = {i}\n" for i in range(n))
    return f"def f():\n{body}    return v{n - 1}\n"


def test_harvest_snapshot_cap_truncates_and_marks(tmp_path, monkeypatch):
    """More traced lines than MAX_SNAPSHOTS: dropped past the cap, and the
    drop is what sets truncated=True here -- nothing sensorium-side is
    truncated by this sample (every value is a plain int)."""
    monkeypatch.setattr("crucible.latent.harvest.MAX_SNAPSHOTS", 3)
    r = harvest(_n_assignments_src(5), "()", tmp_path)
    assert len(r.snapshots) == 3
    assert r.truncated is True


def test_harvest_snapshot_cap_boundary_is_not_truncated(tmp_path, monkeypatch):
    """Exactly MAX_SNAPSHOTS traced lines: nothing dropped, not truncated --
    pins the cap's `>` (not `>=`) comparison, so this exact count must NOT
    enter the truncation branch at all. (This test and the over-cap test
    above are companions: a `[:MAX_SNAPSHOTS] -> [:MAX_SNAPSHOTS - 1]`
    slice mutant is caught by the OVER-cap test, not this one -- at this
    exact boundary the truncation branch never runs, so the mutated slice
    line is never reached here; see task-1-report.md's mutation-testing
    note for the verified kill.)"""
    monkeypatch.setattr("crucible.latent.harvest.MAX_SNAPSHOTS", 3)
    r = harvest(_n_assignments_src(3), "()", tmp_path)
    assert len(r.snapshots) == 3
    assert r.truncated is False


def test_harvest_value_repr_is_capped_and_marks_truncated(tmp_path):
    """A 200-char string is inside sensorium's OWN 200-char str cap (so
    sensorium's own `trunc` flag on this value stays False) but its quoted
    repr (202 chars) is over harvest's own 64-char value_repr cap. Per THIS
    implementation (see harvest.py's `_build_snapshots` docstring -- the
    brief's mechanics note names only sensorium's marker, MAX_SNAPSHOTS,
    and the timeout), cutting a value_repr at 64 chars ALSO sets
    truncated=True, on the theory that a value this module itself had to
    cut is truncated the same as one sensorium already cut.
    """
    long_str = "x" * 200
    r = harvest(f"def f():\n    s = {long_str!r}\n    return len(s)\n", "()", tmp_path)
    value_reprs = [v for s in r.snapshots for (n, _t, v) in s.locals if n == "s"]
    assert value_reprs, "expected a snapshot capturing local 's'"
    assert all(len(v) <= 64 for v in value_reprs)
    assert r.truncated is True


def test_harvest_sensorium_marked_truncation_is_counted(tmp_path):
    """A local exceeding sensorium's own sample cap (CAPS['sample'] == 8,
    sensorium.record.capture) is marked `trunc: true` by sensorium's OWN
    capture_value at record time -- exercised for real (a live sensorium
    run), not simulated at the reader seam: cheap enough (a 20-item list)
    that a real trigger beats a hand-rolled Trace/meta stub which could
    silently drift from the real store contract. This run's
    meta["truncated_count"] is what `_meta_truncated` reads."""
    r = harvest("def f():\n    x = list(range(20))\n    return len(x)\n", "()", tmp_path)
    assert r.outcome == "return" and r.return_repr == "20"
    assert r.truncated is True
