"""Task 11 -- A_full: the hooks that turn the S2 driver into the memory arm.

A_full is not a different driver. It is the SAME driver with three hook calls
(:class:`crucible.run.full.ArmHooks`) and the memory/value/sleep organs behind them, so
everything these tests pin is a wiring claim, not an algorithm claim -- the algorithms
(retrieval, distillation, falsification, the value model, the calibrator, the sleep loop)
already have their own suites.

Five things here are load-bearing.

*``hooks=None`` must leave the S2 arms untouched, argument list included.*
``test_hooks_none_calls_attempt_task_with_the_s2_argument_list`` monkeypatches
``attempt_task`` with a spy whose signature is EXACTLY S2's -- no ``**kwargs``, no
``memory`` parameter -- so a driver that started passing ``memory=None`` unconditionally
fails on the ``TypeError``, not on a subtle record diff. Its companion assertion (the
records come back with ``retrieved_ids == ()`` and ``adapter_id is None``) pins that the
``dataclasses.replace`` stamp is not applied when there are no hooks.

*Lessons are written ONLY for verified episodes; episodes are written for every attempt.*
``test_after_task_writes_a_lesson_only_for_a_verified_attempt`` is the (a) mutation pin:
distilling an unverified episode mints a lesson from an attempt that never produced a
working fix, and ``distill`` itself would raise -- so the mutant that removes the
``verified`` gate does not merely fail this test, it crashes the arm.

*The retrieved ids really travel from ``before_task`` to the record.*
``test_after_task_returns_the_retrieved_ids_from_before_task`` and the end-to-end
``test_full_drive_stamps_the_retrieved_ids_of_a_second_exposure`` are the (b) mutation pin:
a hook that always returned ``()`` would leave "did this task get memory" unanswerable from
the records, which is the column E1 is measured on.

*The value model is trained ONLY on measured outcomes.* ``hidden_pass is None`` means the
attempt was never scored; folding it in as a ``False`` would teach the model that infra
failures are repair failures. Pinned on both sides (``..._updates_value_exactly_once...``
and ``..._does_not_update_value_when_hidden_pass_is_none``), for the calibrator too.

*Sleep fires BETWEEN tasks, and only the driver's own loop decides when.*
``test_full_drive_fires_a_sleep_between_tasks_and_stamps_the_adapter`` runs the real
driver over a real stream with fake GPU seams and asserts both halves: the trainer was
called, and the records written AFTER the accepted sleep carry the new adapter id (the
lens's adapter lineage comes from exactly that stamp).

Run WRAPPED (R-T2-6): the drive tests build a stream and execute real sandbox attempts.
"""
from __future__ import annotations

import dataclasses
import gzip
import json
import pathlib
import urllib.error
from dataclasses import replace

import pytest

import crucible.run.driver as driver
from crucible.memory.schema import content_id
from crucible.memory.schema import EpisodicRecord
from crucible.memory.store import MemoryIdentityMismatch, MemoryStore
from crucible.run.arm import ARMS, ArmConfig
from crucible.run.driver import run_arm, utc_now
from crucible.run.full import (
    build_full_hooks,
    FULL_FAMILY,
    RECALIBRATE_WINDOW,
    AdapterProposer,
    ArmHooks,
    DriverSliceRunner,
    FullHooks,
    MemHooks,
)
from crucible.run.lens import build_lens
from crucible.run.records import ExecRecord, TaskRecord, read_task_records, write_records
from crucible.run.types import Candidate
from crucible.search.loop import SearchResult
from crucible.proposer.identity import IdentityMismatch, ServedIdentity
from crucible.sleep import loop as sleep_loop
from crucible.sleep.loop import (FakeServerAdapter, FakeSliceRunner, SleepController,
                                 SleepRecord, VllmAdapterLoader)
from crucible.sleep.registry import AdapterRegistry
from crucible.sleep.train import FakeTrainer
from crucible.stream import store as stream_store
from crucible.stream.compose import TaskSpec
from crucible.stream.pipeline import BuildConfig, build_stream
from crucible.stream.units import Unit, sha256_text
from crucible.uncertainty.conformal import Calibrator, provenance_class
from crucible.value.online import OnlineValue

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
NOW = "2026-08-24T12:00:00Z"

# --- unit-level fixtures (no sandbox) ---------------------------------------------------
CORRECT = "def add(a, b):\n    return a + b\n"
BUGGY = "def add(a, b):\n    return a - b\n"
VIS = "from unit_x import add as candidate\ndef test_v0():\n    assert candidate(1, 2) == 3\n"
HID = "from unit_x import add as candidate\ndef test_h0():\n    assert candidate(0, 0) == 0\n"
U = Unit("X/0", "unit_x", "add", BUGGY, VIS, HID, sha256_text(BUGGY), 1, 1, ())
SPEC = TaskSpec("k1", "X/0", "ARITH", "X/0|ARITH", 1, "first", ((1, 0), (2, 0)), False, 1)


def _record(**kw) -> TaskRecord:
    base = dict(
        task_key=SPEC.task_key, arm="A_full", unit_id=SPEC.unit_id, family=SPEC.family,
        phase=SPEC.phase, kind=SPEC.kind, landed=True, status="verified_visible",
        confidence=0.6, visible_reward=1.0, executions_charged=2, hidden_pass=True,
        tampered=False, infra_error=None, tokens=None, wall_s=1.0, gpu_s=None,
    )
    base.update(kw)
    return TaskRecord(**base)


def _result(**kw) -> SearchResult:
    base = dict(
        best_patch=CORRECT, best_node_id="n7", visible_reward=1.0, executions_charged=2,
        landed=True, nodes=3, status="verified_visible", confidence=0.6,
        root_prompt="Fix the bug.", symptom_failed=("test_v0",),
    )
    base.update(kw)
    return SearchResult(**base)


class _SpyValue(OnlineValue):
    """``OnlineValue`` plus call logs -- the "exactly once per measured task" probe."""

    def __init__(self) -> None:
        super().__init__()
        self.begun: list[tuple[str, bool]] = []
        self.by_id: list[tuple[str, bool]] = []

    def begin_task(self, family: str, retrieval_hit: bool) -> None:
        self.begun.append((family, retrieval_hit))
        super().begin_task(family, retrieval_hit)

    def update_by_id(self, node_id: str, outcome: bool) -> bool:
        self.by_id.append((node_id, outcome))
        return super().update_by_id(node_id, outcome)


class _SpyCalibrator(Calibrator):
    """``Calibrator`` plus call logs for ``observe`` / ``recalibrate``."""

    def __init__(self) -> None:
        super().__init__()
        self.observed: list[tuple[float, str, bool]] = []
        self.windows: list[int] = []

    def observe(self, score: float, cls: str, outcome: bool) -> None:
        self.observed.append((score, cls, outcome))
        super().observe(score, cls, outcome)

    def recalibrate(self, window: int) -> None:
        self.windows.append(window)
        super().recalibrate(window)


class _SpyTrainer:
    """``FakeTrainer`` with a call log (the sleep-fired pin needs a COUNT)."""

    def __init__(self) -> None:
        self._inner = FakeTrainer()
        self.calls: list[int] = []

    def train(self, pairs, seed, out_dir):
        self.calls.append(len(pairs))
        return self._inner.train(pairs, seed, out_dir)


class _Rig:
    """A ``FullHooks`` plus every seam behind it, so a test can assert on any of them."""

    def __init__(self, tmp_path, *, threshold=999, units=None, counts=(1, 1, 1, 1), seed=0,
                 proposer=None, retrieval="full", sleep_enabled=True,
                 log=lambda *a: None):
        self.store = MemoryStore(tmp_path / "memory.sqlite3")
        self.value = _SpyValue()
        self.calibrator = _SpyCalibrator()
        self.trainer = _SpyTrainer()
        self.server = FakeServerAdapter()
        self.runner = FakeSliceRunner(counts)
        self.registry = AdapterRegistry(tmp_path / "adapters.jsonl")
        self.adapters_dir = tmp_path / "adapters"
        units = units if units is not None else {"X/0": U}
        self.controller = SleepController(
            self.store, self.trainer, self.server, self.runner, self.registry,
            unit_loader=lambda uid: units[uid], adapters_dir=self.adapters_dir,
            threshold=threshold, seed=seed,
        )
        self.sleep_records_path = tmp_path / "sleep_records.jsonl"
        self.logged: list[str] = []
        self.hooks = FullHooks(self.store, self.value, self.calibrator, self.controller,
                               self.registry, sleep_records_path=self.sleep_records_path,
                               proposer=proposer, retrieval=retrieval,
                               sleep_enabled=sleep_enabled, log=self.logged.append)


class _FakeClients:
    """A base fake client plus one minted per adapter id, wired through ``AdapterProposer``.

    Every client records its own ``generate`` calls, so "which model served which task" is
    read off the clients themselves rather than off a flag the code under test set.
    """

    def __init__(self, base_texts, adapter_texts=None, model="fake/model"):
        self.model = model
        self.base = FakeProposer(model, base_texts)
        self._adapter_texts = adapter_texts if adapter_texts is not None else base_texts
        self.built: dict[str, "FakeProposer"] = {}
        self.arm = AdapterProposer(self.base, self._mint, model)

    def _mint(self, adapter_id):
        client = FakeProposer(adapter_id, self._adapter_texts)
        self.built[adapter_id] = client
        return client


class _ScoredNode:
    """The shape ``OnlineValue.score`` reads: an id, a depth, and a candidate's two scores."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.depth = 1
        self.candidate = dataclasses.replace(Candidate("x", None, 0.5))


def _episode_for(task_key: str) -> EpisodicRecord:
    """One minimal episode, so a resume test can make the organ match the record count."""
    return EpisodicRecord(
        item_id=content_id("episode", {"task_key": task_key, "arm": "A_full"}),
        task_key=task_key, arm="A_full", unit_id="X/0", family="ARITH",
        class_id="X/0|ARITH", phase=1, kind="first", root_prompt="p", landed_module=CORRECT,
        visible_reward=1.0, executions_charged=1, hidden_pass=True, verified=True,
        memory_item_ids=(), created_at=NOW, confidence=0.6, status="active", version=1,
        source_locator="arm:A_full/task:" + task_key, valid_at=NOW, invalid_at=None,
        expired_at=None, last_verified_at=NOW, falsified_by=None,
        verification_method="hidden-suite",
    )


def _open_task(hooks, spec=SPEC, *, raw=0.6, unit=U):
    """Everything that happens before ``after_task``: retrieve, then calibrate the RAW score.

    The calibrate call is not test scaffolding -- it is what the search does in ``_finalize``,
    and ``after_task`` REFUSES to guess a raw score that never happened (review I5), because
    the only other number available is the calibrator's own output.
    """
    block = hooks.before_task(unit, spec)
    hooks.task_confidence().calibrate(raw)
    return block


@pytest.fixture
def rig(tmp_path):
    return _Rig(tmp_path)


# --- the arm config ---------------------------------------------------------------------

def test_a_full_is_a_nomem_s_serving_identity():
    """Arms differ by HOOKS, never by the served model: A_full and A_noMem must be the
    same proposer on the same serving surface, or the E1 comparison is confounded."""
    a_full, a_nomem = ARMS["A_full"], ARMS["A_noMem"]
    assert a_full == ArmConfig("A_full", a_nomem.model, True, chat=a_nomem.chat)
    assert (a_full.k, a_full.width, a_full.seed) == (a_nomem.k, a_nomem.width, a_nomem.seed)


def test_full_hooks_satisfies_the_arm_hooks_protocol(rig):
    assert isinstance(rig.hooks, ArmHooks)


# --- before_task ------------------------------------------------------------------------

def test_before_task_on_an_empty_organ_is_a_no_hit(rig):
    block = rig.hooks.before_task(U, SPEC)

    assert block is None                                  # None, never "" (None-vs-zero)
    assert rig.value.begun == [("ARITH", False)]          # the task context IS set, no-hit


def test_before_task_returns_the_retrieved_block_and_marks_a_hit(rig):
    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(), _result(), NOW)   # mints a lesson

    block = rig.hooks.before_task(U, replace(SPEC, task_key="k2", phase=2, kind="second"))

    assert block is not None and "Prior verified fix" in block
    assert rig.value.begun[-1] == ("ARITH", True)         # retrieval_hit reaches the value model


# --- after_task -------------------------------------------------------------------------

def test_after_task_writes_an_episode_for_every_attempt_verified_or_not(rig):
    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(hidden_pass=False, status="believed"),
                         _result(), NOW)

    episodes = rig.store.episodes()
    assert len(episodes) == 1
    assert episodes[0].verified is False
    assert episodes[0].item_id == content_id("episode", {"task_key": "k1", "arm": "A_full"})


def test_after_task_writes_a_lesson_only_for_a_verified_attempt(rig):
    """MUTATION (a): distilling an unverified episode. ``distill`` refuses one outright, so
    a mutant that drops the gate crashes the arm rather than merely mis-recording it."""
    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(hidden_pass=False), _result(), NOW)
    assert rig.store.semantic_family("ARITH") == []

    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(hidden_pass=True), _result(), NOW)
    lessons = rig.store.semantic_family("ARITH")
    assert len(lessons) == 1
    assert lessons[0].cited_episode_id == content_id("episode", {"task_key": "k1", "arm": "A_full"})


def test_a_tampered_pass_is_not_a_verified_episode(rig):
    # episode_verified = hidden_pass is True AND untampered -- the pre-reg definition.
    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(hidden_pass=True, tampered=True), _result(), NOW)

    assert rig.store.episodes()[0].verified is False
    assert rig.store.semantic_family("ARITH") == []


def test_after_task_writes_no_lesson_when_symptom_failed_is_empty(rig):
    """Review fix (M-2): a verified fix whose free symptom run produced no verdict
    (``symptom_failed == ()``) must not mint a lesson -- it would cite no tests and
    permanently sit in ``infra_broken_citation`` on every future sleep. The episode is
    still written (an attempt happened), only the lesson is skipped."""
    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(), _result(symptom_failed=()), NOW)

    assert len(rig.store.episodes()) == 1
    assert rig.store.episodes()[0].verified is True     # the episode itself is verified
    assert rig.store.semantic_family("ARITH") == []      # but no lesson was minted


def test_the_lesson_cites_the_symptom_tests_as_flipped_and_killing(rig):
    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(), _result(symptom_failed=("test_v0", "test_v1")), NOW)

    lesson = rig.store.semantic_family("ARITH")[0]
    assert lesson.flipped_tests == ("test_v0", "test_v1")
    assert lesson.killing_tests == ("test_v0", "test_v1")
    assert lesson.mutated_spans == (SPEC.span,)
    assert "return a + b" in lesson.landed_diff        # diffed against the MUTATED source


def test_the_lesson_diffs_both_sites_of_a_stacked_task(rig):
    spec2 = replace(SPEC, span2=((3, 0), (4, 0)))
    _open_task(rig.hooks, spec2)
    rig.hooks.after_task(U, spec2, _record(), _result(), NOW)

    assert rig.store.semantic_family("ARITH")[0].mutated_spans == (spec2.span, spec2.span2)


def test_the_episode_carries_the_root_prompt_and_the_landed_module(rig):
    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(), _result(root_prompt="ROOT", best_patch=CORRECT), NOW)

    ep = rig.store.episodes()[0]
    assert ep.root_prompt == "ROOT"                  # what sleep trains on: the prompt as SENT
    assert ep.landed_module == CORRECT
    assert ep.created_at == NOW and ep.valid_at == NOW and ep.last_verified_at == NOW


def test_an_unlanded_attempt_stores_no_landed_module(rig):
    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(landed=False, hidden_pass=False),
                         _result(landed=False, best_patch=""), NOW)

    assert rig.store.episodes()[0].landed_module is None


def test_after_task_updates_value_exactly_once_on_a_measured_outcome(rig):
    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(hidden_pass=False), _result(best_node_id="n9"), NOW)

    assert rig.value.by_id == [("n9", False)]
    assert rig.calibrator.observed == [(0.6, provenance_class(False, 1), False)]


def test_after_task_does_not_update_value_when_hidden_pass_is_none(rig):
    """``None`` is "never scored". Folding it in as a ``False`` would train the value model
    (and the calibrator) that an infra failure is a repair failure."""
    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(hidden_pass=None, infra_error="oom"), _result(), NOW)

    assert rig.value.by_id == []
    assert rig.calibrator.observed == []
    assert len(rig.store.episodes()) == 1            # ... but the episode is still written


def test_after_task_returns_the_retrieved_ids_from_before_task(rig):
    """MUTATION (b): a hook that always returns ``()`` leaves "did this task get memory"
    unanswerable from the records -- the column E1 is measured on."""
    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(), _result(), NOW)
    lesson_id = rig.store.semantic_family("ARITH")[0].item_id

    spec2 = replace(SPEC, task_key="k2", phase=2, kind="second")
    _open_task(rig.hooks, spec2)
    retrieved_ids, adapter_id = rig.hooks.after_task(U, spec2, _record(task_key="k2"),
                                                     _result(), NOW)

    assert lesson_id in retrieved_ids
    assert adapter_id is None                        # nothing accepted yet
    assert rig.store.episodes()[-1].memory_item_ids == retrieved_ids


def test_before_task_points_the_proposer_at_the_latest_accepted_adapter(tmp_path):
    # The stamp is the id the proposer was POINTED AT, and pointing it is what makes the next
    # generate request carry that model. Both halves are asserted here (review C1).
    clients = _FakeClients([CORRECT])
    r = _Rig(tmp_path, proposer=clients.arm)
    r.registry.record("ad-0123456789abcdef", "h" * 64, "d" * 64, True, NOW)
    r.registry.record("ad-rejectedrejected", "g" * 64, "d" * 64, False, NOW)

    _open_task(r.hooks)
    _ids, adapter_id = r.hooks.after_task(U, SPEC, _record(), _result(), NOW)

    assert adapter_id == "ad-0123456789abcdef"       # never the rejected candidate
    assert clients.arm.model == "ad-0123456789abcdef"   # ... and that is what it will REQUEST
    assert clients.arm.base_model == "fake/model"       # the arm's frozen base is unchanged


def test_with_no_accepted_adapter_the_proposer_stays_on_the_base_model(tmp_path):
    clients = _FakeClients([CORRECT])
    r = _Rig(tmp_path, proposer=clients.arm)

    _open_task(r.hooks)
    _ids, adapter_id = r.hooks.after_task(U, SPEC, _record(), _result(), NOW)

    assert adapter_id is None and clients.arm.model == "fake/model"
    assert clients.built == {}                       # no adapter client was ever constructed


def test_after_task_refuses_a_task_before_task_never_opened(rig):
    # Without before_task there is no retrieval context: memory_item_ids would silently be
    # empty and the value model's retrieval_hit feature would be a lie. Fail loudly instead.
    with pytest.raises(ValueError):
        rig.hooks.after_task(U, SPEC, _record(), _result(), NOW)


# --- between_tasks / sleep --------------------------------------------------------------

def test_between_tasks_below_the_threshold_does_nothing(tmp_path):
    r = _Rig(tmp_path, threshold=2)
    _open_task(r.hooks)
    r.hooks.after_task(U, SPEC, _record(), _result(), NOW)

    r.hooks.between_tasks(["k1"], NOW)

    assert r.trainer.calls == [] and r.hooks.sleep_records == []
    assert r.calibrator.windows == []


def test_between_tasks_sleeps_at_the_threshold_and_recalibrates_on_accept(tmp_path):
    r = _Rig(tmp_path, threshold=1, counts=(3, 3))
    _open_task(r.hooks)
    r.hooks.after_task(U, SPEC, _record(), _result(), NOW)

    r.hooks.between_tasks(["k1"], NOW)

    assert len(r.trainer.calls) == 1
    assert len(r.hooks.sleep_records) == 1 and r.hooks.sleep_records[0].accepted is True
    assert r.calibrator.windows == [RECALIBRATE_WINDOW]     # exchangeability broken by SFT
    assert r.server.calls                                    # the winner was hot-swapped


def test_a_rejected_sleep_does_not_recalibrate(tmp_path):
    r = _Rig(tmp_path, threshold=1, counts=(9, 5))           # drop of 4 > ACCEPT_MAX_DROP
    _open_task(r.hooks)
    r.hooks.after_task(U, SPEC, _record(), _result(), NOW)

    r.hooks.between_tasks(["k1"], NOW)

    assert r.hooks.sleep_records[0].accepted is False
    assert r.calibrator.windows == []
    assert r.server.calls == []


def test_sleep_records_are_appended_to_disk_and_round_trip(tmp_path):
    r = _Rig(tmp_path, threshold=1, counts=(3, 3))
    _open_task(r.hooks)
    r.hooks.after_task(U, SPEC, _record(), _result(), NOW)
    r.hooks.between_tasks(["k1"], NOW)

    lines = r.sleep_records_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert SleepRecord.from_dict(json.loads(lines[0])) == r.hooks.sleep_records[0]
    assert lines[0] == json.dumps(json.loads(lines[0]), sort_keys=True)


# --- the driver seam --------------------------------------------------------------------

def _recs():
    out = []
    for n in ("mini_humaneval.jsonl.gz", "mini_mbpp.jsonl.gz"):
        with gzip.open(FIX / n, "rt") as fh:
            out += [json.loads(line) for line in fh]
    return out


@pytest.fixture(scope="module")
def stream(tmp_path_factory):
    """The 3-record fixture stream (2 classes x 2 phases = 4 tasks), built once."""
    root = tmp_path_factory.mktemp("stream")
    cfg = BuildConfig(seed=0, C=2, n_nov=0, per_family=3, max_hidden=2, jobs=2)
    return build_stream(cfg, root, recs=_recs(), log=lambda *a: None)


class FakeProposer:
    """Returns scripted module sources in call order; counts every generate call."""

    def __init__(self, model, texts):
        self.model = model
        self._texts = list(texts)
        self.calls = []
        self.prompts = []

    def generate(self, prompt, *, n, seed, max_tokens=1024, temperature=0.7):
        idx = len(self.calls)
        self.calls.append({"n": n, "seed": seed, "temperature": temperature})
        self.prompts.append(prompt)
        text = self._texts[idx % len(self._texts)]
        return [Candidate(text, None, 1.0) for _ in range(n)]


def _class_pair(stream):
    """One class's (phase-1 key, phase-2 key) plus a second class's phase-1 key."""
    man = stream_store.read_manifest(stream)
    (first_a, second_a), (first_b, _second_b) = [man.classes[c] for c in sorted(man.classes)][:2]
    return first_a, first_b, second_a


def _naive_full(name="A_full"):
    """A single-shot arm under A_full's name: one generate per task, crisp counts."""
    return ArmConfig(name, "fake/model", use_search=False)


def test_hooks_none_calls_attempt_task_with_the_s2_argument_list(stream, tmp_path, monkeypatch):
    """The byte-identity guard. The spy's signature is EXACTLY S2's -- a driver that
    started passing ``memory=`` unconditionally dies on the TypeError."""
    keys = list(_class_pair(stream))[:2]
    seen = []

    def s2_only(cfg, unit, taskspec, proposer, value):      # NO memory parameter, no **kw
        seen.append(taskspec.task_key)
        rec = _record(task_key=taskspec.task_key, arm=cfg.name, unit_id=taskspec.unit_id,
                      family=taskspec.family, phase=taskspec.phase, kind=taskspec.kind)
        return rec, [ExecRecord(taskspec.task_key, cfg.name, "n0", 1.0, True, 0.1, None)], _result()

    monkeypatch.setattr(driver, "attempt_task", s2_only)
    out = run_arm(_naive_full("drv"), stream, keys, FakeProposer("fake/model", [CORRECT]),
                  OnlineValue(), tmp_path / "out", log=lambda *a: None, hooks=None)

    assert seen == keys
    recs = read_task_records(out)
    # The replace-stamp is NOT applied without hooks.
    assert all(r.retrieved_ids == () and r.adapter_id is None for r in recs)


def _drive_rig(stream, tmp_path, *, threshold=999, counts=(1, 1, 1, 1), proposer=None):
    units = {u: stream_store.read_unit(stream, u) for u in stream_store.read_manifest(stream).unit_ids}
    return _Rig(tmp_path, threshold=threshold, units=units, counts=counts, proposer=proposer)


def test_full_drive_writes_one_episode_per_task_and_lessons_only_for_verified(stream, tmp_path):
    """End-to-end through the REAL attempt_task: three attempts, two of them verified."""
    first_a, first_b, second_a = _class_pair(stream)
    keys = [first_a, first_b, second_a]
    by_key = {t.task_key: t for t in stream_store.read_manifest(stream).tasks}
    units = {k: stream_store.read_unit(stream, by_key[k].unit_id) for k in keys}
    wrong = f"def {units[first_b].entry_point}(*a, **k):\n    return None\n"
    texts = [units[first_a].module_src, wrong, units[second_a].module_src]

    r = _drive_rig(stream, tmp_path)
    proposer = FakeProposer("fake/model", texts)
    out = run_arm(_naive_full(), stream, keys, proposer, r.value, tmp_path / "out",
                  log=lambda *a: None, hooks=r.hooks)

    recs = read_task_records(out)
    assert len(proposer.calls) == 3                       # one generate per task: no extras
    assert len(r.store.episodes()) == 3                   # an episode for EVERY attempt
    verified = [ep for ep in r.store.episodes() if ep.verified]
    assert {ep.task_key for ep in verified} == {first_a, second_a}
    lessons = [item for fam in {t.family for t in by_key.values()}
               for item in r.store.semantic_family(fam)]
    assert {item.cited_episode_id for item in lessons} == {ep.item_id for ep in verified}
    # value trained once per MEASURED task, and never twice for one task.
    assert len(r.value.by_id) == sum(1 for rec in recs if rec.hidden_pass is not None)


def test_full_drive_stamps_the_retrieved_ids_of_a_second_exposure(stream, tmp_path):
    """MUTATION (b), end to end: the phase-2 task of a class whose phase-1 attempt was
    verified must carry the lesson's id on its record."""
    first_a, _first_b, second_a = _class_pair(stream)
    keys = [first_a, second_a]
    by_key = {t.task_key: t for t in stream_store.read_manifest(stream).tasks}
    units = {k: stream_store.read_unit(stream, by_key[k].unit_id) for k in keys}

    r = _drive_rig(stream, tmp_path)
    proposer = FakeProposer("fake/model", [units[first_a].module_src, units[second_a].module_src])
    out = run_arm(_naive_full(), stream, keys, proposer, r.value, tmp_path / "out",
                  log=lambda *a: None, hooks=r.hooks)

    recs = {rec.task_key: rec for rec in read_task_records(out)}
    assert recs[first_a].retrieved_ids == ()              # nothing to retrieve on first exposure
    assert recs[second_a].retrieved_ids != ()             # ... the lesson from the first
    # The block really reached the model, and on the second exposure only.
    assert "Prior experience with this code" in proposer.prompts[1]
    assert "Prior experience with this code" not in proposer.prompts[0]


def test_full_drive_fires_a_sleep_between_tasks_and_generates_from_the_adapter(stream, tmp_path):
    """THE C1 pin: after an accepted sleep the ARM generates from the adapter, and the record
    stamps the id it actually requested.

    MUTATION: drop the ``select`` in ``_select_adapter`` (or hand ``run_arm`` the unwrapped
    client) -> the second task's generate goes to the base client and the stamp is ``None``.
    """
    first_a, _first_b, second_a = _class_pair(stream)
    keys = [first_a, second_a]
    by_key = {t.task_key: t for t in stream_store.read_manifest(stream).tasks}
    units = {k: stream_store.read_unit(stream, by_key[k].unit_id) for k in keys}
    clients = _FakeClients([units[first_a].module_src], [units[second_a].module_src])

    r = _drive_rig(stream, tmp_path, threshold=1, counts=(2, 2, 2, 2), proposer=clients.arm)
    out = run_arm(_naive_full(), stream, keys, clients.arm, r.value, tmp_path / "out",
                  log=lambda *a: None, hooks=r.hooks)

    recs = {rec.task_key: rec for rec in read_task_records(out)}
    assert len(r.trainer.calls) == 2                       # a sleep after each verified task
    first_adapter = r.hooks.sleep_records[0].adapter_id
    assert recs[first_a].adapter_id is None                # attempted before any sleep: base
    # The stamp is the adapter that SERVED the attempt, not whatever is latest when the run
    # ends -- task 2 ran under sleep 1's adapter, and sleep 2 fired only after it was recorded.
    assert recs[second_a].adapter_id == first_adapter
    assert r.registry.latest_accepted() == r.hooks.sleep_records[1].adapter_id != first_adapter
    assert build_lens(read_task_records(out)).adapter_ids == (first_adapter,)

    # ... and the generate calls really went to the two different models.
    assert len(clients.base.calls) == 1                    # only the pre-sleep task
    assert list(clients.built) == [first_adapter]          # exactly one adapter client built
    assert len(clients.built[first_adapter].calls) == 1    # which served the post-sleep task
    assert clients.built[first_adapter].model == first_adapter


def test_the_clock_is_fixed_width_utc_and_lexicographically_ordered():
    stamp = utc_now()
    assert len(stamp) == 20 and stamp.endswith("Z") and stamp[10] == "T"
    assert stamp > "2026-01-01T00:00:00Z"                   # lexicographic ordering holds


# --- the slice runner (thin seam; LIVE verification is Task 12's smoke) -----------------

def test_slice_runner_loads_the_adapter_and_counts_hidden_passes(stream, tmp_path):
    first_a, first_b, _second_a = _class_pair(stream)
    keys = [first_a, first_b]
    by_key = {t.task_key: t for t in stream_store.read_manifest(stream).tasks}
    units = {k: stream_store.read_unit(stream, by_key[k].unit_id) for k in keys}
    wrong = f"def {units[first_b].entry_point}(*a, **k):\n    return None\n"

    server = FakeServerAdapter()
    built = []

    def proposer_for(model):
        built.append(model)
        return FakeProposer(model, [units[first_a].module_src, wrong])

    runner = DriverSliceRunner(_naive_full(), stream, proposer_for=proposer_for,
                              server=server, adapters_dir=tmp_path / "adapters")

    assert runner.solved(keys, "ad-cafe") == 1              # only the correct repair passes
    assert server.calls == [((tmp_path / "adapters" / "ad-cafe").resolve(), "ad-cafe")]
    assert built == ["ad-cafe"]                             # the proposer serves the ADAPTER


def test_slice_runner_on_the_base_model_loads_nothing(stream, tmp_path):
    first_a, _first_b, _second_a = _class_pair(stream)
    by_key = {t.task_key: t for t in stream_store.read_manifest(stream).tasks}
    unit = stream_store.read_unit(stream, by_key[first_a].unit_id)
    server = FakeServerAdapter()
    cfg = _naive_full()

    runner = DriverSliceRunner(cfg, stream, proposer_for=lambda m: FakeProposer(m, [unit.module_src]),
                               server=server, adapters_dir=tmp_path / "adapters")

    assert runner.solved([first_a], None) == 1
    assert server.calls == []                               # nothing to hot-swap for the base


def test_slice_runner_samples_greedily(stream, tmp_path):
    """K=1 GREEDY: one candidate, temperature 0 -- a regression gate must not re-roll dice."""
    first_a, _first_b, _second_a = _class_pair(stream)
    by_key = {t.task_key: t for t in stream_store.read_manifest(stream).tasks}
    unit = stream_store.read_unit(stream, by_key[first_a].unit_id)
    proposer = FakeProposer("fake/model", [unit.module_src])

    runner = DriverSliceRunner(_naive_full(), stream, proposer_for=lambda m: proposer,
                               server=FakeServerAdapter(), adapters_dir=tmp_path / "adapters")
    runner.solved([first_a], None)

    assert [c["n"] for c in proposer.calls] == [1]
    assert [c["temperature"] for c in proposer.calls] == [0.0]


# --- the calibrated abstention path, end to end -----------------------------------------
#
# The controller's ruling: abstention is wired through the SEARCH, not restamped onto a
# finished record. ``before_task`` builds the per-task hook from the calibrator + this task's
# provenance class, the driver threads it into ``attempt_task`` beside the memory block, and
# the loop applies the §6 gate BEFORE ``run_hidden`` runs. These tests use a REAL
# ``Calibrator``, trained through ``observe``, so what they pin is the actual gate and not a
# fake's opinion of it.
#
# Two mutants die here: dropping the ``confidence`` kwarg from the driver's A_full call
# (``..._rescues_a_raw_abstain`` flips to abstain and reports the raw 0.3), and a
# ``should_abstain`` that always returns False (``..._abstains_below_abstain_p`` flips to
# believed).

RAW = 0.3          # what the value model scores every node at in these tests
NOHIT_P1 = provenance_class(False, 1)


class _FixedValue(_SpyValue):
    """``OnlineValue`` whose ``score`` is pinned, so the RAW confidence is a known number.

    Still calls through to ``OnlineValue.score`` first: that is what caches the node's
    features, which is what makes the later ``update_by_id`` a real training step rather
    than a silent miss.
    """

    def __init__(self, raw: float = RAW) -> None:
        super().__init__()
        self._raw = raw

    def score(self, node) -> float:
        super().score(node)
        return self._raw


def _trained(calibrator: Calibrator, passes: int, fails: int) -> Calibrator:
    """Train ``NOHIT_P1`` at score ``RAW`` so it calibrates to ``passes/(passes+fails)``.

    All observations sit at the same score, so the isotonic fit there is exactly that ratio
    and MIN_OBS is satisfied by ``passes + fails >= 10``.
    """
    for i in range(passes + fails):
        calibrator.observe(RAW, NOHIT_P1, i < passes)
    return calibrator


def _search_full():
    """A SEARCH arm (A_full is one) on a tiny budget -- the abstain rule lives on this path.

    The single-shot control has no abstention rule to condition (see ``arm._naive_attempt``),
    so the calibrated gate can only be exercised through a search arm.
    """
    return ArmConfig("A_full", "fake/model", use_search=True, k=2, width=1)


def _one_task_drive(stream, tmp_path, calibrator_pair):
    """Drive ONE phase-1 task with a never-repairing proposer (reward 0) under A_full's hooks."""
    passes, fails = calibrator_pair
    first_a, _first_b, _second_a = _class_pair(stream)
    by_key = {t.task_key: t for t in stream_store.read_manifest(stream).tasks}
    task = by_key[first_a]
    assert (task.phase, task.kind) == (1, "first")        # so NOHIT_P1 is really its class
    unit = stream_store.read_unit(stream, task.unit_id)
    wrong = f"def {unit.entry_point}(*a, **k):\n    return None\n"

    r = _drive_rig(stream, tmp_path)
    r.value = _FixedValue()
    _trained(r.calibrator, passes, fails)
    r.hooks = FullHooks(r.store, r.value, r.calibrator, r.controller, r.registry,
                        sleep_records_path=r.sleep_records_path)
    out = run_arm(_search_full(), stream, [first_a], FakeProposer("fake/model", [wrong]),
                  r.value, tmp_path / "out", log=lambda *a: None, hooks=r.hooks)
    return r, read_task_records(out)[0]


def test_calibrated_confidence_rescues_a_raw_abstain(stream, tmp_path):
    """Raw 0.3 would abstain under the structural < 0.5 rule; calibrated 0.6 does not.

    MUTATION: drop ``confidence`` from the driver's A_full call -> status flips to abstain
    and the recorded confidence is the raw 0.3.
    """
    r, rec = _one_task_drive(stream, tmp_path, (6, 4))

    assert rec.visible_reward < 0.5                        # the reward half WOULD have fired
    assert rec.status == "believed"
    assert rec.confidence == pytest.approx(0.6)            # the CALIBRATED value it decided on
    # ... and the calibrator was trained on the RAW score, never on its own output.
    assert r.calibrator.observed[-1] == (RAW, NOHIT_P1, rec.hidden_pass)


def test_calibrated_confidence_abstains_below_abstain_p(stream, tmp_path):
    """MUTATION: ``should_abstain`` returning False always -> this flips to believed."""
    _r, rec = _one_task_drive(stream, tmp_path, (1, 9))

    assert rec.confidence == pytest.approx(0.1)            # below ABSTAIN_P = 0.2
    assert rec.status == "abstain"


def test_the_episode_carries_the_calibrated_confidence_not_a_recalibration(stream, tmp_path):
    # Calibrating an already-calibrated number would compose the isotonic map with itself.
    r, rec = _one_task_drive(stream, tmp_path, (6, 4))

    assert r.store.episodes()[0].confidence == pytest.approx(rec.confidence)


def test_task_confidence_is_bound_to_this_task_s_provenance_class(rig):
    rig.hooks.before_task(U, replace(SPEC, phase=2))
    assert rig.hooks.task_confidence().cls == provenance_class(False, 2)

    rig.hooks.before_task(U, SPEC)
    assert rig.hooks.task_confidence().cls == NOHIT_P1


def test_task_confidence_without_an_open_task_raises(rig):
    # A silently-absent hook would put A_full back on the raw < 0.5 rule with nothing in the
    # records to show it, so the failure is loud.
    with pytest.raises(ValueError):
        rig.hooks.task_confidence()


# --- review fixes: resume coherence, the miss counter, the raw-score invariant -----------

def test_after_task_refuses_to_train_the_calibrator_on_its_own_output(rig):
    """I5: no raw score means the confidence hook never reached the attempt.

    The old fallback (raw = record.confidence) would have fed the calibrator the number it
    had just produced, fitting the isotonic map against its own output a little more with
    every task. It raises instead -- and the raise also catches the wiring bug that produced
    the missing raw in the first place (the arm silently ran the uncalibrated abstain rule).
    """
    rig.hooks.before_task(U, SPEC)                   # ... but nothing ever calibrates

    with pytest.raises(ValueError) as e:
        rig.hooks.after_task(U, SPEC, _record(), _result(), NOW)
    assert "raw value score" in str(e.value)


def test_a_value_update_miss_is_counted_and_logged(rig):
    """I4: an outcome for a node the value model never scored trains nothing.

    Not fatal -- the run continues -- but never silent, or the model would simply learn less
    than the record count claims. The S3 smoke asserts this counter is 0.
    """
    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(hidden_pass=True), _result(best_node_id="never-scored"),
                         NOW)

    assert rig.hooks.value_update_misses == 1
    assert any("never-scored" in line for line in rig.logged)


def test_a_scored_node_is_not_a_miss(rig):
    # The counter must mean something: a node the model DID score trains and does not count.
    node = _ScoredNode("n7")
    rig.value.begin_task("ARITH", False)
    rig.value.score(node)

    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(hidden_pass=True), _result(best_node_id="n7"), NOW)

    assert rig.hooks.value_update_misses == 0 and rig.logged == []


def test_build_full_hooks_refuses_a_resume_whose_organ_is_missing_episodes(stream, tmp_path):
    """I3a: records without their episodes. The run would carry on with lessons and an SFT
    set missing exactly those tasks, and nothing downstream could tell that from failure."""
    arm_dir = tmp_path / "runs" / "A_full"
    arm_dir.mkdir(parents=True)
    write_records(arm_dir, [_record(task_key="k1"), _record(task_key="k2")], [])

    with pytest.raises(ValueError) as e:
        build_full_hooks(ARMS["A_full"], stream, tmp_path / "runs", base_url="http://x",
                         value=OnlineValue(), chat=True)
    assert "2 task record" in str(e.value) and "0 episode" in str(e.value)


def test_build_full_hooks_stamps_the_db_and_refuses_another_arms_organ(stream, tmp_path):
    """I3b: the organ is bound to (arm, stream_hash) -- arms never share memory."""
    db = tmp_path / "shared.sqlite3"
    build_full_hooks(ARMS["A_full"], stream, tmp_path / "runs", base_url="http://x",
                     value=OnlineValue(), chat=True, memory_db=db)
    assert MemoryStore(db).identity()["arm"] == "A_full"

    with pytest.raises(MemoryIdentityMismatch):
        build_full_hooks(dataclasses.replace(ARMS["A_full"], name="A_other"), stream,
                         tmp_path / "runs", base_url="http://x", value=OnlineValue(),
                         chat=True, memory_db=db)


def test_build_full_hooks_resumes_a_coherent_run(stream, tmp_path):
    # The guard must not fire on the legitimate resume: episodes >= records.
    arm_dir = tmp_path / "runs" / "A_full"
    arm_dir.mkdir(parents=True)
    write_records(arm_dir, [_record(task_key="k1")], [])
    store = MemoryStore(arm_dir / "memory.sqlite3")
    store.write_episode(_episode_for("k1"))
    store.close()

    hooks = build_full_hooks(ARMS["A_full"], stream, tmp_path / "runs", base_url="http://x",
                             value=OnlineValue(), chat=True)
    assert hooks.sleep_threshold == 16


def test_adapter_proposer_refuses_a_base_client_that_serves_something_else():
    """The relaxed guard in ``attempt_task`` trusts ``base_model``; this is where that
    declaration is EARNED. Minting is the only place a ``base_model`` attribute appears in
    this codebase, so a wrapper around the wrong checkpoint can never reach the guard.

    MUTATION: drop the constructor check -> this test is the only failure.
    """
    wrong_base = FakeProposer("Qwen/Qwen3.5-9B", [CORRECT])

    with pytest.raises(ValueError) as e:
        AdapterProposer(wrong_base, lambda model: None, "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    assert "base_model" in str(e.value)


def test_adapter_proposer_accepts_a_base_client_that_matches_its_declaration():
    base = FakeProposer("fake/model", [CORRECT])
    arm_proposer = AdapterProposer(base, lambda model: None, "fake/model")
    assert arm_proposer.model == "fake/model" and arm_proposer.adapter_id is None


# --- the adapter loader's already-loaded branch (Task 12 smoke, verified live 2026-08-24) --
#
# vLLM answers a second load_lora_adapter for a name it already serves with HTTP 400. ONE
# sleep cycle POSTs the same name twice by design -- DriverSliceRunner must load the candidate
# to measure it (nothing has accepted it yet), then the controller loads it again on accept --
# so a loader that treated every 400 as a failure would fail every sleep this instrument can
# win, after paying for the training run. The 400 is verified, not swallowed: absent from
# /v1/models, it still raises.

def _http_400():
    return urllib.error.HTTPError("http://x/v1/load_lora_adapter", 400, "Bad Request", {}, None)


def _serving(*ids):
    return lambda url, *a, **kw: ServedIdentity("vllm", ids[0], {"n_models": len(ids)}, ids)


def test_reloading_an_adapter_the_server_already_serves_is_not_a_failure(monkeypatch):
    monkeypatch.setattr(sleep_loop.urllib.request, "urlopen",
                        lambda *a, **kw: (_ for _ in ()).throw(_http_400()))
    monkeypatch.setattr(sleep_loop, "probe", _serving("base/m", "ad-cafe"))

    VllmAdapterLoader("http://x").load(pathlib.Path("/adapters/ad-cafe"), "ad-cafe")


def test_a_400_for_an_adapter_the_server_is_not_serving_still_raises(monkeypatch):
    """MUTATION: swallow every 400 without checking -> "the adapter never loaded" becomes a
    silent accept, and the run stamps adapter_id on records the BASE model generated."""
    monkeypatch.setattr(sleep_loop.urllib.request, "urlopen",
                        lambda *a, **kw: (_ for _ in ()).throw(_http_400()))
    monkeypatch.setattr(sleep_loop, "probe", _serving("base/m"))     # the adapter is NOT there

    with pytest.raises(urllib.error.HTTPError):
        VllmAdapterLoader("http://x").load(pathlib.Path("/adapters/ad-cafe"), "ad-cafe")


def test_an_unreachable_server_does_not_turn_a_400_into_a_success(monkeypatch):
    # No evidence the adapter is there => the original error is the honest thing to raise.
    def unreachable(url, *a, **kw):
        raise IdentityMismatch("no recognisable server")

    monkeypatch.setattr(sleep_loop.urllib.request, "urlopen",
                        lambda *a, **kw: (_ for _ in ()).throw(_http_400()))
    monkeypatch.setattr(sleep_loop, "probe", unreachable)

    with pytest.raises(urllib.error.HTTPError):
        VllmAdapterLoader("http://x").load(pathlib.Path("/adapters/ad-cafe"), "ad-cafe")


# --- exploratory ablation switches (docs/findings/ABLATIONS-A.md) -----------------------

def test_full_family_maps_each_ablation_to_a_full_minus_exactly_one_mechanism():
    """The CLI wires (retrieval_mode, sleep_enabled) straight from this map, so a flipped
    tuple here IS a mislabeled run: A_mem_nosleep quietly sleeping, or A_sleep_nomem quietly
    reading the store, with every record stamping the wrong arm name."""
    assert FULL_FAMILY == {"A_full": ("full", True),
                           "A_mem_nosleep": ("full", False),
                           "A_sleep_nomem": ("off", True),
                           "A_mem_exactonly": ("exact", False)}


def test_exact_mode_serves_repeats_but_silences_strangers(tmp_path):
    """retrieval="exact": after a verified attempt mints a lesson for SPEC's class, a
    second exposure of the SAME class still gets a block, but a task of the same family
    on a DIFFERENT unit gets None with item_ids=() (Phase-B §3 reading guide)."""
    rig = _Rig(tmp_path, retrieval="exact")
    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(), _result(), NOW)          # mints exact-class lesson
    same = rig.hooks.before_task(U, replace(SPEC, task_key="k2", phase=2, kind="second"))
    assert same is not None                                            # exact class: served
    rig.hooks.task_confidence().calibrate(0.6)
    rig.hooks.after_task(U, replace(SPEC, task_key="k2", phase=2, kind="second"), _record(), _result(), NOW)
    stranger = replace(SPEC, task_key="k3", unit_id="Y/9", class_id="Y/9|ARITH", kind="novel", phase=2)
    assert rig.hooks.before_task(U, stranger) is None                  # stranger: silence
    assert rig.value.begun[-1] == (SPEC.family, False)


def test_retrieval_disabled_offers_nothing_even_when_the_organ_holds_a_lesson(tmp_path):
    """A_sleep_nomem's read side: the store can hold a perfectly retrievable lesson and
    ``before_task`` must still hand the search NOTHING -- the prompt stays byte-for-byte
    the S2 prompt, ``retrieval_hit`` is False, and the record stamps ``item_ids=()``. The
    lesson is minted first (through the same hooks: the WRITE side is deliberately alive)
    and its presence is asserted, so this test fails if the organ was empty all along."""
    rig = _Rig(tmp_path, retrieval="off")
    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(), _result(), NOW)          # mints the lesson
    assert rig.store.semantic_for(SPEC.unit_id, SPEC.family)          # organ HAS content

    spec2 = replace(SPEC, task_key="k2", phase=2, kind="second")
    block = _open_task(rig.hooks, spec2)
    ids, _adapter = rig.hooks.after_task(U, spec2, _record(), _result(), NOW)

    assert block is None                     # nothing offered, not an empty string
    assert rig.value.begun[-1] == (SPEC.family, False)   # the hit feature is honestly False
    assert ids == ()                         # the record will stamp "no memory offered"


def test_sleep_disabled_returns_before_the_controller_is_asked(tmp_path):
    """A_mem_nosleep's sleep side, at threshold=1 -- the exact configuration
    ``test_between_tasks_sleeps_at_the_threshold_and_recalibrates_on_accept`` proves DOES
    fire when the switch is on. With the switch off the controller must not even be asked:
    no trainer call, no sleep record in memory or on disk, no recalibration."""
    rig = _Rig(tmp_path, threshold=1, counts=(3, 3), sleep_enabled=False)
    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(), _result(), NOW)          # verified episode

    rig.hooks.between_tasks(["k1"], NOW)

    assert rig.trainer.calls == [] and rig.hooks.sleep_records == []
    assert not rig.sleep_records_path.exists()
    assert rig.calibrator.windows == []


def test_ablation_switches_are_readable_and_default_on(tmp_path):
    """The CLI test asserts the wiring through these properties; they must reflect the
    constructed switches, and a bare FullHooks is A_full (both on)."""
    for sub in ("a", "b", "c"):
        (tmp_path / sub).mkdir()
    assert (_Rig(tmp_path / "a").hooks.retrieval_enabled,
            _Rig(tmp_path / "b").hooks.sleep_enabled) == (True, True)
    off = _Rig(tmp_path / "c", retrieval="off", sleep_enabled=False).hooks
    assert (off.retrieval_enabled, off.sleep_enabled) == (False, False)


def test_mem_hooks_write_episodes_and_verified_lessons_but_never_train(tmp_path):
    """MemHooks (Phase-B §3): episode per attempt, lesson on verified only, confidence
    hook is None (the S2 status rule), adapter stamp always None, between_tasks inert."""
    store = MemoryStore(tmp_path / "memory.sqlite3")
    hooks = MemHooks(store)
    assert hooks.task_confidence() is None
    hooks.before_task(U, SPEC)
    ids, adapter = hooks.after_task(U, SPEC, _record(), _result(), NOW)
    assert (ids, adapter) == ((), None)
    assert len(store.episodes()) == 1 and store.semantic_for(SPEC.unit_id, SPEC.family)
    spec2 = replace(SPEC, task_key="k2", phase=2, kind="second")
    assert hooks.before_task(U, spec2) is not None        # the store FILLED and now serves
    hooks.after_task(U, spec2, _record(hidden_pass=False, status="believed"), _result(), NOW)
    assert len(store.episodes()) == 2
    assert len(store.semantic_for(SPEC.unit_id, SPEC.family)) == 1   # no lesson from a failure
    hooks.between_tasks(["k1"], NOW)                       # must be a no-op, nothing raises


def test_mem_hooks_after_task_without_before_task_raises(tmp_path):
    hooks = MemHooks(MemoryStore(tmp_path / "memory.sqlite3"))
    with pytest.raises(ValueError, match="without a matching before_task"):
        hooks.after_task(U, SPEC, _record(), _result(), NOW)
    # A pending task_key that does not match the one after_task is called for is just as
    # unguarded a case as no pending at all -- a mutant that collapsed the guard to
    # ``pending is None`` (dropping the task_key compare) would sail through this call and
    # fabricate retrieved_ids for a task before_task never opened.
    hooks.before_task(U, SPEC)
    with pytest.raises(ValueError, match="without a matching before_task"):
        hooks.after_task(U, replace(SPEC, task_key="k-other"), _record(), _result(), NOW)
