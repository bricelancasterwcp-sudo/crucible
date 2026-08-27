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
import os
import shutil
import sys
from pathlib import Path

import pytest

from crucible.latent.harvest import HarvestError, HarvestResult, Snapshot, harvest


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
    # An exception no longer forces an empty snapshot sequence (final review
    # CRITICAL fix) -- sensorium's LINE event for the raising line fires
    # BEFORE that line executes, so even this single-statement body carries
    # one legitimate pre-raise snapshot of its parameter. The assertion here
    # is on LEGITIMACY, not on a specific count: whatever is present must be
    # real captured locals, never fabricated.
    for s in r.snapshots:
        assert isinstance(s, Snapshot)
        for name, type_name, value_repr in s.locals:
            assert isinstance(name, str) and isinstance(type_name, str) and isinstance(value_repr, str)
    names = {n for s in r.snapshots for (n, _t, _v) in s.locals}
    assert names <= {"a"}  # the only local ever in scope before the raise


def test_harvest_exception_keeps_pre_raise_snapshots(tmp_path):
    """THE critical-fix pin: an exception raised partway through a
    multi-line function must not discard the LINE-event snapshots recorded
    before the raise. Pre-fix, `label 0` (a non-"return" outcome) collapsed
    to an always-empty snapshot sequence -- a gate model conditioned on
    `s.snapshots` could read the label straight off that emptiness, which is
    exactly the leakage this fix closes. Verified empirically (not
    guessed): `b = a + 1` then `c = b + 1` then `return c[0]` (raises
    TypeError on line 4, `c` is an int) produces real pre-raise snapshots at
    lines 2-4, carrying `a`, `b`, and `c` -- confirmed against a live
    sensorium run before this test was written.
    """
    src = "def f(a):\n    b = a + 1\n    c = b + 1\n    return c[0]\n"
    r = harvest(src, "(1,)", tmp_path)
    assert r.outcome == "exception:TypeError" and r.return_repr is None
    assert len(r.snapshots) == 3
    assert [s.line for s in r.snapshots] == [2, 3, 4]
    names_by_line = {s.line: {n for n, _t, _v in s.locals} for s in r.snapshots}
    assert names_by_line[2] == {"a"}
    assert names_by_line[3] == {"a", "b"}
    assert names_by_line[4] == {"a", "b", "c"}


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


def test_harvest_relative_workdir_does_not_double_nest_the_store(tmp_path):
    """Round-2 live-fire regression, found via the corpus run's audit trail
    (not by any of the 13 tests already in this file -- every one of them
    already passed an absolute `tmp_path`).

    The `sensorium run` subprocess's cwd IS `workdir` (see `_execute_once`'s
    `cwd=str(workdir)`). With a RELATIVE `workdir`, a relative
    `SENSORIUM_DIR` inherited by that child resolves a SECOND time against
    the child's own (already-workdir) cwd when sensorium's
    `paths.trace_root()` creates it -- landing the real trace at
    `workdir/workdir/.sensorium/traces/run-a.db`, a path
    `_execute_once`'s own `trace_path.exists()` check (run back in THIS
    process, against THIS process's unrelated cwd) never looks at, raising
    `HarvestError` on every call. `harvest()` must resolve `workdir` to
    absolute before deriving anything from it -- verified live: a relative
    workdir reproduced exactly the double-nested path described above on
    the unfixed code.
    """
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rel = Path("rel_scratch")
        rel.mkdir()
        r = harvest("def f():\n    x = 1\n    return x\n", "()", rel)
    finally:
        os.chdir(old_cwd)
    assert r.outcome == "return" and r.return_repr == "1"
    assert r.snapshots


def test_harvest_shared_workdir_two_calls_do_not_replay_each_other(tmp_path):
    """THE production bug pin (round-3 CRITICAL, found by another
    implementer's real-harvest smoke and confirmed against production
    data): two `harvest()` calls sharing ONE workdir -- the real
    corpus-generation usage pattern, thousands of calls against one scratch
    dir -- must not silently replay each other's result.

    Pre-fix mechanism: `_execute_once` used FIXED run-id constants
    ("run-a"/"run-b") and every call shared one `SENSORIUM_DIR` derived
    straight from the caller's `workdir`. The second call's `sensorium run`
    refused to record (sensorium's own guard: a run id whose trace file
    already exists raises `TargetError`, exit code 2) -- but `_execute_once`
    never checked the subprocess's exit code, found the STALE trace file
    from call 1 already sitting at `trace_path`, and happily read it,
    returning call 1's outcome/return value for every call afterward.

    Confirmed against production data: every one of `runs/blite-corpus/
    samples.jsonl`'s 1000 rows shared the exact same outcome/return_repr/
    snapshot, traced back to the corpus's first successful harvest call.
    This test shares `tmp_path` (NOT a fresh one per call) deliberately --
    that is the shape that broke; every OTHER test in this file passes a
    fresh workdir per `harvest()` call and could never have caught this.
    """
    r1 = harvest("def f():\n    return 1\n", "()", tmp_path)
    r2 = harvest("def f():\n    return 2\n", "()", tmp_path)
    assert r1.outcome == "return" and r1.return_repr == "1"
    assert r2.outcome == "return" and r2.return_repr == "2"
    assert r1.return_repr != r2.return_repr


def test_harvest_nonzero_exit_raises_and_names_the_failure(tmp_path):
    """Fix #2, independent of fix #1: exercises the EXACT run-id-collision
    shape fix #1 (unique per-call subdirectories) now prevents `harvest()`
    from ever hitting -- but directly against `_execute_once`, so this pins
    fix #2 (the exit-code check) on its own, even if a future refactor ever
    reintroduces a shared run-id by some other path.

    First call succeeds and creates a REAL trace at
    `sensorium_dir/traces/run-a.db`. The second call, same `run_id`, same
    `sensorium_dir`, triggers sensorium's OWN real run-id-collision refusal
    (`boot.run_target`'s `TargetError`: "run id ... already has a trace
    at ..."; verified live -- EXIT=2, not caught by the pre-existing "no
    trace produced" check from round 1, since a trace file genuinely exists
    at that path, just a STALE one). Must raise `HarvestError` naming the
    failure, never silently re-read the stale trace from call 1 -- which is
    exactly what the pre-fix code did (0 is a clean target return, 1 is the
    TARGET's own uncaught exception -- `sensorium run` reports that via the
    same exit code, verified live: EXIT=1 for a genuine, correctly-recorded
    `IndexError` -- so the check is "not in {0, 1}", not "nonzero", or
    every legitimate exception-outcome sample would wrongly raise here too).
    """
    from crucible.latent.harvest import _execute_once, _write_runner_script

    fname = "f"
    script_path = _write_runner_script("def f():\n    return 1\n", fname, "()", tmp_path)
    focus = f"{script_path.stem}:{fname}"
    sensorium_dir = tmp_path / ".sensorium"

    first = _execute_once(script_path, focus, tmp_path, sensorium_dir, "run-a", fname)
    assert first[0] == "return" and first[1] == "1"   # sanity: the real first call worked

    with pytest.raises(HarvestError, match="run-a"):
        _execute_once(script_path, focus, tmp_path, sensorium_dir, "run-a", fname)
