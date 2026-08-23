"""Task 12 -- the arm driver: run one arm over a task set, resumably, from the S1 store.

Three load-bearing properties, each mutation-pinned:

* (a) *The per-task unit the agent repairs carries the MUTATED module, not the canonical
  one.* ``read_unit`` returns the unit with its correct ``module_src``; the bug lives in
  the mutant's ``mutated_src``. A driver that fed ``attempt_task`` the canonical source
  would ask the agent to repair code that is already correct -- no bug to see -- so the
  spy asserts the unit handed to every attempt has ``module_src == mutant.mutated_src``.
* (b) *Resumability.* A re-run must not re-attempt task_keys already in the partial
  ``task_records.jsonl``; the fake proposer's call count proves ``attempt_task`` was not
  invoked again.
* (c) *Seeded pilot sampling.* ``select_pilot_tasks`` draws phase-1 (``kind=="first"``)
  keys with ``random.Random(f"{seed}:pilot")`` -- never "first N", so the drawn set moves
  with the seed.

Run WRAPPED (R-T2-6): the stream build and every attempt touch the sandbox.
"""
import gzip
import json
import pathlib

import pytest

import crucible.run.driver as driver
from crucible.run.arm import ArmConfig
from crucible.run.driver import run_arm, select_pilot_tasks
from crucible.run.records import TASK_RECORDS_FILE, read_task_records
from crucible.run.types import Candidate
from crucible.stream import store
from crucible.stream.pipeline import BuildConfig, build_stream
from crucible.value.model import ConstantValue

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
FAKE_MODEL = "fake/model"


def _recs():
    out = []
    for n in ("mini_humaneval.jsonl.gz", "mini_mbpp.jsonl.gz"):
        with gzip.open(FIX / n, "rt") as fh:
            out += [json.loads(line) for line in fh]
    return out


@pytest.fixture(scope="module")
def stream(tmp_path_factory):
    """Build the 3-record fixture stream once (n_nov=0 -> 2 classes -> 4 tasks) and share it."""
    root = tmp_path_factory.mktemp("stream")
    cfg = BuildConfig(seed=0, C=2, n_nov=0, per_family=3, max_hidden=2, jobs=2)
    return build_stream(cfg, root, recs=_recs(), log=lambda *a: None)


class FakeProposer:
    """In-process proposer: returns scripted module sources, counting every generate call.

    Each call returns the mutated (buggy) source for its position -- guaranteed to import,
    so ``attempt_task`` gets a clean report. ``calls`` is the resumability probe.
    """

    def __init__(self, model, texts):
        self.model = model
        self._texts = texts
        self.calls = []

    def generate(self, prompt, *, n, seed, max_tokens=1024, temperature=0.7):
        idx = len(self.calls)
        self.calls.append({"n": n, "seed": seed})
        base = self._texts[idx % len(self._texts)] if self._texts else "def _f():\n    return 0\n"
        return [Candidate(base, None, 1.0) for _ in range(n)]


def _naive_arm():
    """A single-shot arm (one generate per task) served by the fake model -- fast + crisp counts."""
    return ArmConfig("drv", FAKE_MODEL, use_search=False)


def _first_two_keys(stream):
    man = store.read_manifest(stream)
    return [t.task_key for t in man.tasks[:2]]


def test_run_arm_writes_records_done_and_feeds_the_mutated_module(stream, tmp_path, monkeypatch):
    keys = _first_two_keys(stream)
    texts = [store.read_mutant(stream, k).mutated_src for k in keys]
    fake = FakeProposer(FAKE_MODEL, texts)

    # Spy: capture the (task_key, module_src) every attempt is handed, then run the real attempt.
    seen = []
    real = driver.attempt_task

    def spy(cfg, unit, taskspec, proposer, value):
        seen.append((taskspec.task_key, unit.module_src))
        return real(cfg, unit, taskspec, proposer, value)

    monkeypatch.setattr(driver, "attempt_task", spy)

    out = run_arm(_naive_arm(), stream, keys, fake, ConstantValue(), tmp_path / "out",
                  log=lambda *a: None)

    # Two records + the .DONE marker.
    recs = read_task_records(out)
    assert [r.task_key for r in recs] == keys
    done = out / ".DONE"
    assert done.exists()
    payload = json.loads(done.read_text())
    man = store.read_manifest(stream)
    assert payload == {"arm": "drv", "stream_hash": man.stream_hash, "seed": 0}

    # Every attempt saw the MUTATED module, not the canonical one (pins mutation 3).
    assert {k for k, _ in seen} == set(keys)
    by_task = {t.task_key: t for t in man.tasks}
    for key, module_src in seen:
        mutant = store.read_mutant(stream, key)
        assert module_src == mutant.mutated_src
        assert module_src != store.read_unit(stream, by_task[key].unit_id).module_src


def test_rerun_skips_already_recorded_tasks(stream, tmp_path):
    keys = _first_two_keys(stream)
    texts = [store.read_mutant(stream, k).mutated_src for k in keys]
    fake = FakeProposer(FAKE_MODEL, texts)
    arm, val, out = _naive_arm(), ConstantValue(), tmp_path / "out"

    run_arm(arm, stream, keys, fake, val, out, log=lambda *a: None)
    after_first = len(fake.calls)
    assert after_first == 2                                   # one generate per task
    assert len(read_task_records(out / "drv")) == 2

    run_arm(arm, stream, keys, fake, val, out, log=lambda *a: None)
    assert len(fake.calls) == after_first                     # resumed: attempt_task NOT re-called
    assert len(read_task_records(out / "drv")) == 2           # and no duplicate records


def test_select_pilot_tasks_is_seeded_and_phase_one_only(stream):
    man = store.read_manifest(stream)
    phase1 = {t.task_key for t in man.tasks if t.kind == "first"}
    phase2 = {t.task_key for t in man.tasks if t.kind != "first"}
    assert phase1 and phase2                                  # the fixture has both

    # Only phase-1 keys, for any seed.
    for seed in range(4):
        drawn = select_pilot_tasks(stream, 2, seed=seed)
        assert set(drawn) <= phase1
        assert not (set(drawn) & phase2)

    # Same seed -> same list (determinism).
    assert select_pilot_tasks(stream, 2, seed=0) == select_pilot_tasks(stream, 2, seed=0)

    # Different seed -> different draw: across seeds the single pick varies (pins first-N mutation:
    # phase1[:1] is seed-invariant -> the set collapses to size 1 and this assertion fails).
    picks = {tuple(select_pilot_tasks(stream, 1, seed=s)) for s in range(10)}
    assert len(picks) > 1
