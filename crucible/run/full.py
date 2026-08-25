"""A_full: the hooks that turn the S2 driver into the memory arm (spec S3 §8).

A_full is **not a second driver**. It is :func:`crucible.run.driver.run_arm` with three
calls -- ``before_task`` / ``after_task`` / ``between_tasks`` -- and the memory organ, the
v1 value model, the calibrator and the sleep loop sitting behind them. Everything those
organs do is already implemented and tested in its own module; what lives here is only the
wiring, and the wiring is where an experiment quietly dies, so each of its invariants is
written down.

**Four hard requirements, each of them a way this arm could fabricate its own result.**

*No extra sandbox executions.* This module never calls ``run``/``run_hidden`` on the task
path. Everything ``after_task`` writes down is read off the attempt that already happened
(the ``TaskRecord`` and the ``SearchResult`` the driver hands it) -- re-running anything to
"check" a result would be a second, differently-conditioned measurement of an attempt that
is already over, and it would be charged to nobody's budget. The only executions this
module can cause are sleep-internal: falsification (inside ``SleepController``) and the
regression slice (:class:`DriverSliceRunner`), both uncharged by design, both counted in
the ``SleepRecord``. — with ONE disclosed exception (Phase-C, spec §4.3): symptom-mode
`before_task` runs a single uncharged symptom probe per task, identical in inputs and
outcome to the free symptom run the search itself performs; it is counted in
`uncharged_symptom_runs` and never appears in `executions_charged`.

*No extra generate calls.* ``FullHooks`` holds no proposer and cannot talk to a model. The
one place in this file that generates is :class:`DriverSliceRunner`, and that is the sleep
loop's regression gate, not a task attempt.

*Verified-only distillation.* A lesson is minted only from an episode that is ``verified``
under the pre-reg definition (:func:`~crucible.memory.schema.episode_verified`: the hidden
suite passed AND the landing was untampered). Every attempt gets an EPISODE -- failures are
data -- but a lesson from a failed attempt would put a fix that does not work into a later
prompt and, transitively, into the SFT set. ``distill`` refuses one outright, so a wiring
bug here does not mis-record the arm, it stops it.

*Value updates only on measured outcomes.* ``hidden_pass is None`` means the attempt was
never scored (infra died); folding that in as a ``False`` would teach the value model and
the calibrator that an infra failure is a repair failure. Both are skipped on ``None`` --
the same None-vs-zero discipline the lens applies to its denominators.

**What travels between the two task-scoped hooks, and why it is guarded.** ``before_task``
retrieves and remembers the ``(task_key, item_ids, confidence, adapter_id)`` it produced;
``after_task`` needs those ids for the episode's ``memory_item_ids`` and for the record's
``retrieved_ids``, the attempt itself needs the confidence hook, and the record's
``adapter_id`` must be the one the proposer was actually pointed at.
Re-retrieving inside ``after_task`` would be cheap but WRONG: by then the store may hold a
lesson minted from this very task, so the second read could report items the prompt never
contained. The pending state is therefore keyed by ``task_key`` and an ``after_task`` that
does not match it raises -- an unguarded version would silently stamp an empty tuple and
make "did this task get memory" (the column E1 is measured on) a lie that no record shows.

**Timestamps.** Nothing here reads a clock. ``now`` arrives from
:func:`crucible.run.driver.utc_now`, the one place an arm run reads one, so an episode, the
lesson distilled from it, and a sleep fired immediately after all carry the same instant.

**Abstention is decided INSIDE the search, never restamped onto a record.** An attempt's
``status`` is a decision the arm makes about its own submission, so it has to happen while
the decision is still live -- before ``run_hidden`` answers. ``before_task`` therefore builds
a :class:`_TaskConfidence` bound to this task's provenance class, and the driver threads it
into ``attempt_task`` alongside the memory block; the search calibrates its best node's raw
value score through it and applies the §6 gate
(:func:`~crucible.uncertainty.conformal.Calibrator.should_abstain`, ``ABSTAIN_P = 0.2``) in
place of the structural ``< 0.5`` compare. Rewriting ``status`` here in ``after_task``
instead would restate history rather than change a decision, and is not done. The two
thresholds are deliberately different rules -- see ``crucible.search.loop``'s module
docstring.

**The arm must actually GENERATE from the accepted adapter (review C1).** vLLM routes a
runtime-loaded LoRA by MODEL NAME: a request whose ``model`` is the base checkpoint runs the
base weights no matter what has been loaded. A run that built one proposer for the base and
kept it forever would therefore train adapters, hot-swap them, stamp ``adapter_id`` on every
subsequent record -- and never once generate from one, making sleep inert on the measured
path while the records claimed otherwise. :class:`AdapterProposer` is the fix: ONE proposer
object for the whole run whose delegate is re-pointed, once per task in ``before_task``, at
whatever adapter the registry has accepted. The id it selects is snapshotted there and is
what ``after_task`` stamps -- the stamp and the serving decision read the same value at the
same instant, so the record cannot claim an adapter the attempt did not use.

**Known S3 limitation: only the ORGAN survives a resume.** The driver is resumable at task
granularity, and the memory db (episodes, lessons) and the adapter registry are both on disk,
so they resume with it. The value model's weights and the calibrator's observation history
are in-process only and restart from scratch -- a resumed A_full run re-learns both from the
tasks it has left. ``build_full_hooks`` refuses the incoherent case it CAN detect (records
without their episodes) and stamps the db with its ``(arm, stream_hash)`` so two runs cannot
share an organ, but it cannot resurrect an untrained scorer. Persisting them is S4 scope
(``OnlineValue.snapshot``/``restore`` and ``Calibrator.snapshot``/``restore`` already exist
for record-keeping; nothing writes them yet). The S3 exit smoke runs start-to-finish, so no
measurement in this slice depends on that gap -- said plainly here rather than left for
someone to discover from a suspiciously untrained model half-way through a resumed run.

*The calibrator is trained on the RAW score, never on its own output.* The record's
``confidence`` for A_full is the CALIBRATED p (the number the status decision used), so the
raw score it was mapped from is no longer on the record -- ``_TaskConfidence`` remembers it
(``calibrate`` is called exactly once per attempt, in the search's finalisation) and
``after_task`` observes THAT. Feeding a calibrated p back into ``observe`` would fit the
isotonic map against its own output, and the fit would drift toward the identity a little
more with every task; that is the kind of self-confirming instrument this whole spike exists
to avoid.
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from crucible.memory import symmatch
from crucible.memory.distill import distill
from crucible.memory.retrieve import RetrievedBlock, retrieve, retrieve_symptom
from crucible.memory.schema import EpisodicRecord, content_id, episode_verified
from crucible.memory.store import MemoryStore
from crucible.proposer.prompt import render_symptom
from crucible.run.arm import ArmConfig, attempt_task
from crucible.run.driver import mutated_unit
from crucible.run.records import TASK_RECORDS_FILE, TaskRecord, read_task_records
from crucible.sandbox.task_run import run
from crucible.search.loop import SearchResult, TaskConfidence
from crucible.sleep.loop import (SLEEP_THRESHOLD_DEFAULT, ServerAdapter, SleepController,
                                 SleepRecord, VllmAdapterLoader)
from crucible.sleep.registry import AdapterRegistry
from crucible.sleep.train import LoraTrainer
from crucible.stream import store as stream_store
from crucible.stream.compose import TaskSpec
from crucible.stream.units import Unit
from crucible.uncertainty.conformal import Calibrator, provenance_class
from crucible.value.model import ConstantValue
from crucible.value.online import OnlineValue

FULL_ARM = "A_full"
"""The gating memory arm -- both hook switches on (see ``FULL_FAMILY``)."""

FULL_FAMILY: dict[str, tuple[str, bool]] = {
    "A_full": ("full", True),
    "A_mem_nosleep": ("full", False),    # prereg A_mem−sleep: explicit memory, no LoRA
    "A_sleep_nomem": ("off", True),      # prereg A_sleep−mem: LoRA, no store in the prompt
    "A_mem_exactonly": ("exact", False), # prereg §4.3: exact-class recall only, no LoRA
    "A_symmem": ("symptom", False),      # Phase-C §4.2/§4.3: cross-unit symptom recall, no LoRA
}
"""The arms these hooks serve, mapped to ``(retrieval_mode, sleep_enabled)`` -- the CLI
gates on membership here (see ``cli.py``). ``retrieval_mode`` is ``"full"`` (class-exact
falling back to family-wide, ``retrieve``'s default policy), ``"exact"`` (class-exact only
-- a stranger unit gets silence rather than a family-wide lesson, see ``exact_only`` on
:func:`crucible.memory.retrieve.retrieve`), ``"symptom"`` (Phase-C prereg §4.2: class-exact
fast path, else a cross-unit symptom-similarity ranking gated by ``tau`` -- see
:func:`crucible.memory.retrieve.retrieve_symptom` and its ``before_task`` caller, both of
which run the disclosed uncharged probe described in this module's docstring), or ``"off"``
(the store is never consulted).
The two `_nosleep`/`_nomem` ablations and ``A_mem_exactonly`` are all EXPLORATORY
(non-gating, protocol ``docs/findings/ABLATIONS-A.md``): each is A_full minus exactly ONE
mechanism, with the value model and calibrator kept, so a difference from A_full's measured
rates reads as that mechanism's marginal contribution and nothing else. They are separate
ARM NAMES rather than run-time flags on A_full so every record stamps which configuration
produced it -- a gate record and an ablation record can never be confused by file mixing."""

MEM_ARMS: dict[str, str] = {"B_mem": "full", "B_symmem": "symptom"}
"""The Phase-B/Phase-C arms ``MemHooks`` serves: the store and nothing else -- no value
model, no calibrator, no sleep (prereg §3 for ``B_mem``; Phase-C prereg §4.2/§4.3 for
``B_symmem``). Maps each arm name to the ``retrieval`` mode ``build_mem_hooks`` /
``MemHooks.__init__`` is given -- ``"full"`` (class-exact falling back to family-wide) or
``"symptom"`` (class-exact fast path, else cross-unit symptom-similarity ranking gated by
``tau``, plus the disclosed uncharged probe -- see the module docstring). A dict, not a
frozenset, because membership alone can no longer say which policy an arm runs; kept as
its own mapping rather than folded into ``FULL_FAMILY`` so the CLI gate for "wire the full
organ" and "wire the store alone" stay two membership checks (``in MEM_ARMS`` still works
on a dict), never one dict lookup that has to be interpreted two different ways."""

MEMORY_DB_FILE = "memory.sqlite3"
ADAPTERS_DIR = "adapters"
ADAPTER_REGISTRY_FILE = "adapters.jsonl"
SLEEP_RECORDS_FILE = "sleep_records.jsonl"
SYMPTOM_PROBE_LOG_FILE = "symptom_probes.txt"
"""Everything an A_full/A_symmem/B_mem/B_symmem run writes beside its records, all under
``out_dir/<arm>/``: the organ, the trained adapters, the append-only adapter ledger, one
line per sleep cycle, and (symptom mode only -- see ``_persist_probe_count``) the current
``uncharged_symptom_runs`` count. One directory per arm run means a re-run against a fresh
``--out`` is a fresh organ, which is what "arms never share memory" (spec §2) means
operationally."""

RECALIBRATE_WINDOW = 50
"""Observations per provenance class the calibrator re-fits from after an ACCEPTED sleep.

Inferred, not pinned by the spec (§7 says only "``recalibrate(window)`` fires after every
accepted sleep"). 50 is chosen against the two numbers around it: the fit needs more than
``conformal.MIN_OBS`` (10) points per class to be worth more than raw passthrough, and the
window should be shorter than the sleep cadence's own scale (16 verified episodes, i.e.
substantially more attempts) so that "the last 50 observations in this class" is mostly
POST-sleep data by the time a second sleep lands. A knob, not a threshold: nothing in the
pre-registered analysis reads it."""

VERIFICATION_METHOD = "hidden-suite"
"""How an episode's ``verified`` flag was decided -- the driver-side hidden oracle. Recorded
on every episode, pass or fail: it names the METHOD, not the outcome."""


def build_episode(taskspec: TaskSpec, record: TaskRecord, result: SearchResult,
                  memory_item_ids: tuple[str, ...], now: str, confidence: float) -> EpisodicRecord:
    """The episode for one attempt -- written whether or not it worked.

    ``landed_module`` is the submitted module when the codec produced one, else ``None``
    ("nothing landed", not an empty module). ``last_verified_at`` is set only for a
    verified episode: the hidden suite just checked that claim, and for an unverified one
    there is no claim to have checked. ``confidence`` is the record's own confidence --
    for A_full the CALIBRATED P(hidden pass) the status decision was made on, carried
    verbatim rather than re-calibrated (composing the isotonic map with itself would make
    the episode's number mean nothing at all).
    """
    verified = episode_verified(record.hidden_pass, record.tampered)
    return EpisodicRecord(
        item_id=content_id("episode", {"task_key": taskspec.task_key, "arm": record.arm}),
        task_key=taskspec.task_key, arm=record.arm, unit_id=taskspec.unit_id,
        family=taskspec.family, class_id=taskspec.class_id, phase=taskspec.phase,
        kind=taskspec.kind, root_prompt=result.root_prompt,
        landed_module=result.best_patch if record.landed else None,
        visible_reward=record.visible_reward,
        executions_charged=record.executions_charged, hidden_pass=record.hidden_pass,
        verified=verified, memory_item_ids=memory_item_ids, created_at=now,
        confidence=confidence, status="active", version=1,
        source_locator=f"arm:{record.arm}/task:{taskspec.task_key}",
        valid_at=now, invalid_at=None, expired_at=None,
        last_verified_at=now if verified else None, falsified_by=None,
        verification_method=VERIFICATION_METHOD,
    )


@runtime_checkable
class ArmHooks(Protocol):
    """What :func:`crucible.run.driver.run_arm` may call between attempts. A_full implements
    it; every other arm passes ``None`` and the driver stays byte-identical to S2.

    ``before_task`` returns the memory block for this attempt's prompts (``None`` = no
    memory, never ``""``); ``task_confidence`` returns the calibration hook it just built for
    the same task, which the driver threads into ``attempt_task`` beside the block. They are
    two calls rather than one tuple so ``before_task``'s pinned return type stays what the
    driver actually passes as ``memory``. ``after_task`` returns ``(retrieved_ids, adapter_id)`` for the
    driver to stamp onto the record via ``dataclasses.replace`` -- the hooks never mutate a
    record themselves, because a frozen record that only the driver writes is what makes
    "the file traces attempt order" true. ``between_tasks`` is the sleep check.

    Both task-scoped hooks receive the MUTATED ``unit`` the agent actually repaired: the
    lesson's diff is taken against that source, so a hook handed the canonical unit would
    diff the fix against code that never had the bug.
    """

    def before_task(self, unit: Unit, taskspec: TaskSpec) -> str | None: ...

    def task_confidence(self) -> "TaskConfidence | None": ...

    def after_task(self, unit: Unit, taskspec: TaskSpec, record: TaskRecord,
                   result: SearchResult, now: str) -> tuple[tuple[str, ...], str | None]: ...

    def between_tasks(self, solved_task_keys: list[str], now: str) -> None: ...


class _TaskConfidence:
    """One task's calibrated confidence + abstention gate -- the search's ``TaskConfidence``.

    Bound to ONE provenance class (``(retrieval_hit, phase)``) at ``before_task`` time, so the
    search never has to know what a provenance class is. ``calibrate`` REMEMBERS the raw score
    it was handed (``self.raw``): the record keeps the calibrated value, so this is the only
    surviving copy of the calibrator's own input, and ``observe`` must be trained on that --
    see the module docstring.
    """

    def __init__(self, calibrator: Calibrator, cls: str) -> None:
        self._calibrator = calibrator
        self.cls = cls
        self.raw: float | None = None

    def calibrate(self, score: float) -> float:
        """Calibrated P(hidden pass) for ``score`` in this class; records ``score`` as raw."""
        self.raw = score
        return self._calibrator.confidence(score, self.cls)

    def should_abstain(self, p: float) -> bool:
        """The pre-reg §6 gate for an already-calibrated ``p`` (``ABSTAIN_P``, inclusive)."""
        return self._calibrator.should_abstain(p, self.cls)


@dataclasses.dataclass(frozen=True)
class _Pending:
    """What ``before_task`` retrieved, waiting for the ``after_task`` of the SAME task."""

    task_key: str
    item_ids: tuple[str, ...]
    confidence: _TaskConfidence
    """The task's calibration hook. Its ``cls`` encodes the retrieval hit, so the hit is not
    stored a second time -- one place to read it from, nothing to drift."""

    adapter_id: str | None
    """The adapter the proposer was pointed at for THIS attempt (``None`` = base model).
    Snapshotted at selection time and stamped verbatim by ``after_task``: the record's
    adapter lineage and the serving decision are the same value, not two readings of a
    registry that a sleep could have moved in between.

    Today the snapshot is DEFENCE IN DEPTH rather than a live difference -- the driver only
    sleeps between tasks, so a fresh ``latest_accepted()`` read in ``after_task`` would agree
    with it on every run this code can currently produce. It is the snapshot anyway because
    the agreement is a property of the driver's loop shape, not of this record: the day sleep
    moves seams (mid-task consolidation, a background trainer), the snapshot is still the
    honest answer to "what generated this attempt" and the re-read silently is not."""


def _persist_probe_count(probe_log_path: Path | None, count: int) -> None:
    """Overwrite ``probe_log_path`` with ``count`` -- the post-hoc artifact C5 needs because
    the in-process ``uncharged_symptom_runs`` counter dies with a detached run (spec §4.3).
    A no-op when ``probe_log_path`` is ``None``, which is how non-symptom modes never create
    the file at all: they simply never call this. Shared by both hook classes so the write
    (and its overwrite-not-append semantics -- the file always reads the CURRENT total) is
    defined exactly once."""
    if probe_log_path is None:
        return
    probe_log_path.parent.mkdir(parents=True, exist_ok=True)
    # temp+rename so a mid-write kill can never leave a truncated count that reads as a
    # smaller-but-plausible measurement (final-review fold-in; C5 compares equality to 450)
    tmp = probe_log_path.with_suffix(".tmp")
    tmp.write_text(str(count), encoding="utf-8")
    tmp.replace(probe_log_path)


class FullHooks:
    """A_full's :class:`ArmHooks`: retrieve -> attempt -> record + distill -> maybe sleep.

    ``value`` must be the v1 :class:`~crucible.value.online.OnlineValue` (not the bare
    ``Value`` protocol): this class calls ``begin_task`` and ``update_by_id``, which v0's
    ``ConstantValue`` does not have. That is deliberate -- an arm configured with the
    constant scorer would silently drop both the task-context features and the training
    signal, and a ``getattr``-guarded call would hide it. It must also be the SAME object
    the driver passes to ``attempt_task``, or the context ``begin_task`` sets is not the
    context the search's ``score`` calls read.

    ``proposer`` is the :class:`AdapterProposer` the driver generates through -- the same
    object, so pointing it at an accepted adapter here IS what the next attempt asks the
    server for. ``None`` (unit tests that drive the hooks directly) means no adapter is ever
    selected and every record stamps ``adapter_id=None``, which is the honest reading: with
    no proposer to re-point, nothing served an adapter.
    """

    def __init__(self, store: MemoryStore, value: OnlineValue, calibrator: Calibrator,
                 sleep_controller: SleepController, registry: AdapterRegistry, *,
                 sleep_records_path: Path, proposer: "AdapterProposer | None" = None,
                 recalibrate_window: int = RECALIBRATE_WINDOW,
                 retrieval: str = "full", sleep_enabled: bool = True,
                 probe_log_path: Path | None = None,
                 log=print) -> None:
        self._retrieval = retrieval
        self._sleep_enabled = sleep_enabled
        self._store = store
        self._value = value
        self._calibrator = calibrator
        self._sleep = sleep_controller
        self._proposer = proposer
        self._log = log
        self._registry = registry
        self._sleep_records_path = Path(sleep_records_path)
        self._recalibrate_window = recalibrate_window
        self._probe_log_path = probe_log_path
        self._pending: _Pending | None = None
        self.sleep_records: list[SleepRecord] = []
        self.value_update_misses: int = 0
        """Measured outcomes that trained NOTHING because the node was never scored.

        ``OnlineValue.update_by_id`` returns False for an id it has no cached features for --
        by design, so an upstream bug degrades to a skipped training step instead of a crashed
        run. Discarding that return would make the degradation invisible: the value model
        would simply learn less than the record count says it did. Every miss is counted here
        and logged; the S3 smoke asserts this is 0."""
        self.uncharged_symptom_runs: int = 0
        """How many uncharged symptom-mode probes ``before_task`` has run (Phase-C spec §4.3)
        -- see the module docstring's disclosed exception. Incrementing this is what makes
        the probe COUNTED rather than a silent extra execution; every increment is mirrored
        to ``probe_log_path`` (:func:`_persist_probe_count`) so a post-hoc read survives a
        detached run, since this in-process attribute does not. Stays 0 for every non-symptom
        retrieval mode -- the probe never runs, so there is nothing to count."""

    @property
    def proposer(self) -> "AdapterProposer | None":
        """The re-pointable arm proposer the CALLER must generate through (see C1)."""
        return self._proposer

    @property
    def sleep_threshold(self) -> int:
        """The configured sleep trigger, read off the controller that owns it."""
        return self._sleep.threshold

    @property
    def retrieval_mode(self) -> str:
        """Which retrieval policy ``before_task`` applies: ``"full"``, ``"exact"``,
        ``"symptom"``, or ``"off"`` -- see ``FULL_FAMILY``."""
        return self._retrieval

    @property
    def retrieval_enabled(self) -> bool:
        """Whether ``before_task`` consults the store at all (== ``retrieval_mode != "off"``).

        Kept alongside ``retrieval_mode`` so the Phase-A on/off assertions survive the mode
        migration unchanged: ``A_mem_exactonly`` reads ``True`` here, same as ``A_full``,
        because the store IS consulted -- the difference between "full" and "exact" is
        WHICH results are eligible once it is, not whether it is asked at all."""
        return self._retrieval != "off"

    @property
    def sleep_enabled(self) -> bool:
        """Whether ``between_tasks`` may fire sleep (False only for A_mem_nosleep)."""
        return self._sleep_enabled

    def before_task(self, unit: Unit, taskspec: TaskSpec) -> str | None:
        """Retrieve this task's memory block and set the value model's task context.

        The block is whatever :func:`crucible.memory.retrieve.retrieve` decides for this
        (unit_id, family) class -- ``None`` when the organ has nothing to say. The
        retrieval HIT (block present) is a task-level feature of the value model, so
        ``begin_task`` is called here, before any node is scored, exactly once per task.

        ``unit`` is part of the hook contract (pre-reg §9 writes retrieval as
        ``retrieve(unit, family, symptom)``); the "full"/"exact"/"off" branches read none
        of it -- retrieval keys on the CLASS -- ``(unit_id, family)`` -- not on the source
        text. Symptom mode is the branch that finally uses it: see below.

        *Symptom mode (Phase-C spec §4.3) is the ONE disclosed exception to this module's
        "no extra sandbox executions" invariant* -- see the module docstring. It runs a
        single UNCHARGED probe of the visible suite against ``unit``'s own (mutated, still
        broken) source, deterministic and identical in inputs and outcome to the free
        symptom run the search itself performs, so retrieval can be conditioned on the same
        symptom text the prompt's own ``## Symptom`` section will show. The probe is counted
        in ``uncharged_symptom_runs`` (persisted to ``probe_log_path``, if configured) and
        never touches ``executions_charged`` -- it happens here, entirely outside
        ``attempt_task``, before any budget meter for this task exists.
        """
        # The A_sleep_nomem ablation ("off") NEVER consults the store: the block is
        # structurally absent, not "retrieved and empty", so the prompt is byte-for-byte the
        # S2 prompt and item_ids stamp () -- the honest "no memory was offered" record. The
        # organ is still WRITTEN (after_task is unchanged); only the read side is severed.
        # A_mem_exactonly ("exact") consults the store but restricts it to the exact class --
        # see ``exact_only`` on ``crucible.memory.retrieve.retrieve``. A_symmem ("symptom")
        # runs the uncharged probe above and reads cross-unit via symptom similarity.
        if self._retrieval == "symptom":
            symptom = run(unit, unit.module_src, None)     # UNCHARGED driver-side probe:
            self.uncharged_symptom_runs += 1                # deterministic, byte-identical to
            _persist_probe_count(self._probe_log_path,       # the search's own free symptom
                                 self.uncharged_symptom_runs) # run; never executions_charged
            block = retrieve_symptom(
                self._store, unit.module_src, taskspec.unit_id, taskspec.family,
                render_symptom(symptom), tau=symmatch.TAU)
        elif self._retrieval != "off":
            block = retrieve(self._store, taskspec.unit_id, taskspec.family,
                             exact_only=(self._retrieval == "exact"))
        else:
            block = RetrievedBlock(None, ())
        hit = block.block is not None
        cls = provenance_class(hit, taskspec.phase)
        self._pending = _Pending(taskspec.task_key, block.item_ids,
                                 _TaskConfidence(self._calibrator, cls),
                                 self._select_adapter())
        self._value.begin_task(taskspec.family, hit)
        return block.block

    def _select_adapter(self) -> str | None:
        """Point the arm's proposer at the latest ACCEPTED adapter; return what it selected.

        This is the whole of "the arm runs its adapter": the returned id is both what the
        next generate request carries and what the record is stamped with (review C1). Read
        once, here, so a sleep firing later in the same task cannot make the two disagree.
        """
        if self._proposer is None:
            return None
        return self._proposer.select(self._registry.latest_accepted())

    def task_confidence(self) -> _TaskConfidence:
        """The calibration hook ``before_task`` just built -- the driver threads it into the
        attempt so the abstention decision is made on a CALIBRATED number, before the hidden
        oracle runs. Raises if no task is open: a silently absent hook would put A_full back
        on the raw ``< 0.5`` rule with nothing in the records to show it."""
        if self._pending is None:
            raise ValueError("task_confidence() with no task open -- call before_task first")
        return self._pending.confidence

    def after_task(self, unit: Unit, taskspec: TaskSpec, record: TaskRecord,
                   result: SearchResult, now: str) -> tuple[tuple[str, ...], str | None]:
        """Write the episode (always), the lesson (verified only), and learn from the outcome.

        By the time this runs the abstention decision is already made and recorded: the
        search applied ``task_confidence()`` before ``run_hidden``, so ``record.confidence``
        is the CALIBRATED p and ``record.status`` is the §6 gate's verdict. Nothing here
        re-decides either.

        Returns ``(retrieved_ids, adapter_id)`` for the driver to stamp. ``adapter_id`` is
        the id ``before_task`` actually pointed the proposer at -- not a fresh read of the
        registry -- so the record names the adapter that GENERATED this attempt and nothing
        else (review C1).

        The lesson's ``flipped_tests`` are ``result.symptom_failed`` -- the visible tests
        that were failing BEFORE the fix. A verified fix passes the whole visible suite, so
        those are exactly the tests that flipped fail->pass. ``killing_tests`` is the same
        tuple: the tests that killed the mutant ARE the ones its symptom run failed. Only
        the NAMES come from the symptom; the COUNT is carried independently by the stream's
        own validation (``taskspec.n_killing_visible``), so the two never have to agree by
        construction and a disagreement stays visible.
        """
        pending = self._pending
        if pending is None or pending.task_key != taskspec.task_key:
            raise ValueError(
                f"after_task for {taskspec.task_key!r} without a matching before_task "
                f"(pending={pending.task_key if pending else None!r}) -- the retrieval "
                f"context is what the record's retrieved_ids and the value model's "
                f"retrieval-hit feature are built from; guessing it would be a fabrication"
            )
        self._pending = None

        cls = pending.confidence.cls
        # The record's confidence IS the calibrated p, so the raw score it was mapped from
        # survives ONLY on the closure. There is no fallback to the record here (review I5):
        # `record.confidence` would be the calibrator's own OUTPUT, and training the isotonic
        # map on its output drifts it toward the identity a little more every task. A missing
        # raw means the hook never reached the attempt -- which also means the arm silently
        # ran the uncalibrated abstain rule -- so it fails loudly instead.
        raw = pending.confidence.raw
        if raw is None:
            raise ValueError(
                f"no raw value score for {taskspec.task_key!r}: the confidence hook never "
                f"reached the attempt, so this arm ran the uncalibrated abstain rule and "
                f"there is no honest input to train the calibrator on"
            )
        episode = self._episode(taskspec, record, result, pending, now, record.confidence)
        self._store.write_episode(episode)

        # The third clause mirrors distill()'s own guard (review fix): a verified fix whose
        # free symptom run produced no verdict (symptom_failed == ()) must not mint a lesson
        # that cites no tests -- distill() would refuse it anyway, but gating here keeps the
        # skip visible at the call site instead of relying on the callee's raise.
        if episode.verified and episode.landed_module is not None and result.symptom_failed:
            spans = (taskspec.span,) if taskspec.span2 is None else (taskspec.span, taskspec.span2)
            self._store.write_semantic(distill(
                episode, mutated_src=unit.module_src, spans=spans,
                flipped_tests=result.symptom_failed, killing_tests=result.symptom_failed,
                now=now,
            ))

        if record.hidden_pass is not None:           # measured, so it can be learned from
            if not self._value.update_by_id(result.best_node_id, record.hidden_pass):
                # Not fatal (an outcome for a node that was never scored trains nothing), but
                # never silent: counted and logged, and the smoke asserts the count is 0.
                self.value_update_misses += 1
                self._log(f"[value] no cached features for node {result.best_node_id} "
                          f"(task {taskspec.task_key}): outcome not trained on "
                          f"(misses={self.value_update_misses})")
            self._calibrator.observe(raw, cls, record.hidden_pass)

        return pending.item_ids, pending.adapter_id

    def between_tasks(self, solved_task_keys: list[str], now: str) -> None:
        """Let sleep fire if enough verified episodes have landed; recalibrate on an accept.

        ``recalibrate`` runs ONLY after an accepted sleep, because that is the only outcome
        that changes what the server is running: a rejected candidate leaves the serving
        model untouched, so the calibrator's observations are still exchangeable with the
        ones before it and re-fitting from a short window would just throw data away.

        The A_mem_nosleep ablation returns before the controller is even ASKED: its sleep
        count must be structurally zero, not "the threshold happened never to fire", so no
        run configuration (a low ``--sleep-threshold``, a long run) can sleep it by accident.
        """
        if not self._sleep_enabled:
            return
        record = self._sleep.maybe_sleep(list(solved_task_keys), now)
        if record is None:
            return
        self.sleep_records.append(record)
        self._append_sleep_record(record)
        if record.accepted:
            self._calibrator.recalibrate(self._recalibrate_window)

    def _episode(self, taskspec: TaskSpec, record: TaskRecord, result: SearchResult,
                 pending: _Pending, now: str, confidence: float) -> EpisodicRecord:
        """Delegates to build_episode -- see that function for details."""
        return build_episode(taskspec, record, result, pending.item_ids, now, confidence)

    def _append_sleep_record(self, record: SleepRecord) -> None:
        """One JSON object per line, keys sorted, UTF-8 -- the S1 store convention.

        Append-only, like the adapter registry: a sleep that happened must stay on disk even
        if the run later dies, or "why is this adapter serving" is unanswerable from the
        files alone.
        """
        self._sleep_records_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._sleep_records_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


class MemHooks:
    """B_mem/B_symmem's hooks: the store and NOTHING else (Phase-B prereg §3; Phase-C
    prereg §4.2/§4.3 for the symptom mode).

    The one treatment is the memory block in the prompt. Everything else that FullHooks
    carries is deliberately ABSENT: ``task_confidence()`` returns ``None`` so the search
    runs the S2 structural status rule the frozen B_search control ran (the driver threads
    the return unconditionally -- None IS the S2 configuration, see ``attempt_task``);
    there is no value model to train (the caller passes ``ConstantValue``, as B_search's
    run did) and no calibrator to observe. Episodes are written for every attempt and
    lessons distilled from verified ones -- the same two gates as FullHooks.after_task --
    because the store must FILL during the run for retrieval to have anything to say by
    phase 2.

    ``retrieval`` selects the read policy ``before_task`` applies -- ``"full"`` (B_mem's
    class-exact-falling-back-to-family-wide read, :func:`~crucible.memory.retrieve.retrieve`)
    or ``"symptom"`` (B_symmem's class-exact-then-cross-unit-symptom-similarity read,
    :func:`~crucible.memory.retrieve.retrieve_symptom`, plus the disclosed uncharged probe
    -- see the module docstring). Mirrors ``FullHooks``' dispatch (same shape, same
    ``uncharged_symptom_runs``/``probe_log_path`` discipline) so the two hook classes read
    as one policy, wired twice.
    """

    def __init__(self, store: MemoryStore, *, retrieval: str = "full",
                 probe_log_path: Path | None = None, log=print) -> None:
        self._store = store
        self._retrieval = retrieval
        self._probe_log_path = probe_log_path
        self._log = log
        self._pending: tuple[str, tuple[str, ...]] | None = None   # (task_key, item_ids)
        self.uncharged_symptom_runs: int = 0
        """See ``FullHooks.uncharged_symptom_runs`` -- same counter, same persistence via
        :func:`_persist_probe_count`, this class's own copy because the two hook classes
        never share instance state."""

    @property
    def retrieval(self) -> str:
        """Which retrieval policy ``before_task`` applies: ``"full"`` or ``"symptom"``."""
        return self._retrieval

    def before_task(self, unit: Unit, taskspec: TaskSpec) -> str | None:
        # Mirrors FullHooks.before_task's symptom branch -- see that method's docstring and
        # the module docstring's disclosed exception for the uncharged-probe discipline.
        if self._retrieval == "symptom":
            symptom = run(unit, unit.module_src, None)     # UNCHARGED driver-side probe
            self.uncharged_symptom_runs += 1
            _persist_probe_count(self._probe_log_path, self.uncharged_symptom_runs)
            block = retrieve_symptom(
                self._store, unit.module_src, taskspec.unit_id, taskspec.family,
                render_symptom(symptom), tau=symmatch.TAU)
        else:
            block = retrieve(self._store, taskspec.unit_id, taskspec.family)
        self._pending = (taskspec.task_key, block.item_ids)
        return block.block

    def task_confidence(self) -> None:
        return None                      # S2 status rule -- byte-identical to B_search

    def after_task(self, unit: Unit, taskspec: TaskSpec, record: TaskRecord,
                   result: SearchResult, now: str) -> tuple[tuple[str, ...], str | None]:
        pending = self._pending
        if pending is None or pending[0] != taskspec.task_key:
            raise ValueError(
                f"after_task for {taskspec.task_key!r} without a matching before_task "
                f"(pending={pending[0] if pending else None!r}) -- guessing retrieved_ids "
                f"would fabricate the record's memory column")
        self._pending = None
        episode = build_episode(taskspec, record, result, pending[1], now, record.confidence)
        self._store.write_episode(episode)
        if episode.verified and episode.landed_module is not None and result.symptom_failed:
            spans = (taskspec.span,) if taskspec.span2 is None else (taskspec.span, taskspec.span2)
            self._store.write_semantic(distill(
                episode, mutated_src=unit.module_src, spans=spans,
                flipped_tests=result.symptom_failed, killing_tests=result.symptom_failed,
                now=now,
            ))
        return pending[1], None          # adapter_id is always None: nothing ever trains

    def between_tasks(self, solved_task_keys: list[str], now: str) -> None:
        return None                      # no sleep, structurally


class AdapterProposer:
    """The arm's proposer, re-pointable at whichever adapter the run has ACCEPTED (review C1).

    vLLM serves a runtime-loaded LoRA under its own model name, so "the arm is running the
    adapter" means literally "the generate request carries ``model=<adapter_id>``". This
    wrapper owns that choice for the measured path: :meth:`select` swaps the delegate (and
    therefore ``self.model``, which IS what the next request will ask for), and clients are
    built once per adapter and cached -- a run with three accepted adapters builds three, not
    one per task.

    ``base_model`` is the arm's frozen base checkpoint and never changes.
    :func:`crucible.run.arm.attempt_task`'s served-identity guard accepts a proposer whose
    ``model`` is the arm's model OR one that declares this ``base_model`` -- so the guard
    still refuses a proposer serving some other checkpoint, while an adapter ON the arm's own
    base is recognised as the arm running its own adapter rather than as a mismatch.

    *That relaxation is only as honest as the declaration, so the declaration is CHECKED where
    it is minted.* The constructor refuses a base client whose own ``model`` is not
    ``base_model``: this class is the only thing in the codebase that grows a ``base_model``
    attribute, so a wrapper around the wrong checkpoint can never reach the guard and claim to
    be an adapter on the right one. The guard's own raise still stands for everything else
    (a proposer with no ``base_model`` at all serving a foreign model).

    ``select`` is called once per task from ``before_task``; nothing else may point this
    object anywhere, so "what served this task" has exactly one writer.

    *Resuming against a RESTARTED vLLM:* a fresh server has no adapters loaded, so ``select``
    on a previously-accepted id fails in ``VLLMProposer.__init__``'s ``assert_identity`` with
    :class:`~crucible.proposer.identity.IdentityMismatch`. That is loud by design -- silently
    serving the base under an adapter's name is exactly the label lie this class exists to
    prevent. The operator re-loads the adapter on the server (or starts a fresh ``--out``).
    """

    def __init__(self, base_proposer, proposer_for: Callable[[str], object],
                 base_model: str) -> None:
        served = getattr(base_proposer, "model", None)
        if served != base_model:
            raise ValueError(
                f"AdapterProposer base client serves {served!r} but declares "
                f"base_model={base_model!r} -- the declaration is what relaxes attempt_task's "
                f"served-identity guard, so it is checked here, where it is minted"
            )
        self._base = base_proposer
        self._proposer_for = proposer_for
        self.base_model = base_model
        self._cache: dict[str, object] = {}
        self._inner = base_proposer
        self.model = getattr(base_proposer, "model", base_model)
        self.adapter_id: str | None = None

    def select(self, adapter_id: str | None) -> str | None:
        """Point the next generate calls at ``adapter_id`` (``None`` = the base model).

        Returns the id actually selected, which the caller stamps on the record. Building the
        client asserts the server really advertises that adapter (``VLLMProposer.__init__`` ->
        ``assert_identity``), so an adapter the server never loaded fails HERE, loudly, rather
        than silently serving the base weights under an adapter's name in the records.
        """
        if adapter_id is None:
            self._inner = self._base
        else:
            if adapter_id not in self._cache:
                self._cache[adapter_id] = self._proposer_for(adapter_id)
            self._inner = self._cache[adapter_id]
        self.adapter_id = adapter_id
        self.model = getattr(self._inner, "model", self.base_model)
        return adapter_id

    def generate(self, prompt: str, *, n: int, seed: int, **kw):
        """Delegate to whichever client :meth:`select` last chose."""
        return self._inner.generate(prompt, n=n, seed=seed, **kw)


class _GreedyProposer:
    """A proposer wrapper that forces ``temperature=0.0`` -- the regression slice's K=1 GREEDY.

    The gate asks "does this adapter still solve what the run already solved". Sampling at
    the arm's usual temperature would answer a noisier question (the same adapter could pass
    the gate one cycle and fail it the next on identical weights), so the slice draws the
    model's single most likely completion instead. Wrapping rather than adding a parameter to
    ``_naive_attempt`` keeps the sampling change local to the one caller that wants it.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.model = inner.model                 # attempt_task asserts served identity on this

    def generate(self, prompt: str, *, n: int, seed: int, **kw):
        kw.pop("temperature", None)
        return self._inner.generate(prompt, n=n, seed=seed, temperature=0.0, **kw)


class DriverSliceRunner:
    """The real :class:`~crucible.sleep.loop.SliceRunner`: re-attempt solved tasks under an adapter.

    For each key: reload the unit + mutant through the driver's own ``mutated_unit`` (reused,
    not reimplemented, so the slice repairs byte-for-byte the module the arm did), then run a
    SINGLE-SHOT greedy attempt -- one prompt, one candidate, no refinement -- and count the
    ones whose HIDDEN suite passes. "Solved" therefore means the same thing on the slice as it
    does everywhere else in this experiment (the pre-reg success definition), not "the visible
    tests went green".

    *The slice loads the adapter itself.* ``SleepController`` measures the CANDIDATE before it
    has accepted anything, so nothing else can have loaded it -- ``crucible.sleep.loop``'s own
    docstring says a real runner must ("a real SliceRunner must load it to measure ``after``
    at all"). ``adapter_id=None`` means the base model and loads nothing. On the accept path
    the controller then loads the same adapter again; a re-POST of an already-loaded adapter
    name is vLLM behaviour this repo has not yet observed, and it is on the live smoke's
    checklist.

    *Never metered, never in the prompt path.* These executions go through ``attempt_task``'s
    single-shot branch, which calls the sandbox directly rather than through a
    ``BudgetMeter``, so nothing here can charge a task's K. No memory block is passed either:
    the gate isolates the ONE thing that changed between ``before`` and ``after`` (the
    adapter), and a retrieval block would put a second moving part in a two-point comparison.

    This seam is thin on purpose and its unit tests are correspondingly narrow (it is wired to
    fakes above the sandbox); the LIVE verification -- a real adapter, a real server, a real
    hot-swap -- is the S3 exit smoke.
    """

    def __init__(self, cfg: ArmConfig, stream_dir: Path, *,
                 proposer_for: Callable[[str], object], server: ServerAdapter,
                 adapters_dir: Path) -> None:
        self._cfg = cfg
        self._stream_dir = Path(stream_dir)
        self._proposer_for = proposer_for
        self._server = server
        self._adapters_dir = Path(adapters_dir).resolve()
        self._by_key: dict[str, TaskSpec] | None = None      # read lazily: construction is cheap

    def solved(self, task_keys: list[str], adapter_id: str | None) -> int:
        """How many of ``task_keys`` this model still solves, hidden-suite verified."""
        model = self._cfg.model if adapter_id is None else adapter_id
        if adapter_id is not None:
            self._server.load(self._adapters_dir / adapter_id, adapter_id)
        proposer = _GreedyProposer(self._proposer_for(model))
        cfg = dataclasses.replace(self._cfg, name=f"{self._cfg.name}:slice", model=model,
                                  use_search=False)
        solved = 0
        for task_key in task_keys:
            task = self._tasks()[task_key]                # KeyError: a key not in this stream
            # v0's ConstantValue, never the run's own scorer: the gate is a measurement, and
            # scoring its nodes into the live value model would let sleep-internal work leak
            # into the model the ARM's search consults.
            record, _execs, _result = attempt_task(cfg, mutated_unit(self._stream_dir, task),
                                                   task, proposer, ConstantValue())
            # The ONE success definition, shared with the driver's solved list and the
            # episode's own ``verified`` flag -- never a re-spelling of it here.
            solved += 1 if episode_verified(record.hidden_pass, record.tampered) else 0
        return solved

    def _tasks(self) -> dict[str, TaskSpec]:
        if self._by_key is None:
            self._by_key = {t.task_key: t
                            for t in stream_store.read_manifest(self._stream_dir).tasks}
        return self._by_key


def _assert_resume_coherent(store: MemoryStore, arm_dir: Path) -> None:
    """Refuse a resume whose records and organ disagree (review I3a).

    The driver resumes from ``task_records.jsonl`` and skips every task already in it; the
    organ resumes from the memory db. If the db holds FEWER episodes than there are prior
    records, some attempts have records but no memory of themselves -- the run would carry on
    with lessons and an SFT set missing exactly those tasks, and nothing downstream could tell
    that from a run where they had simply failed. Both numbers are named in the message
    because "which one is stale" is the operator's next question. More episodes than records
    is not an error: the db legitimately accumulates one episode per attempt while a partially
    written record file is always a prefix of them.
    """
    records_path = arm_dir / TASK_RECORDS_FILE
    if not records_path.exists():
        return
    n_records = len(read_task_records(arm_dir))
    n_episodes = len(store.episodes())
    if n_episodes < n_records:
        raise ValueError(
            f"resume is incoherent: {arm_dir} holds {n_records} task record(s) but the "
            f"memory db holds {n_episodes} episode(s) -- records and the organ must resume "
            f"together (a fresh --memory-db against an existing run, or the reverse)"
        )


def build_full_hooks(cfg: ArmConfig, stream_dir: Path, out_dir: Path, *, base_url: str,
                     value: OnlineValue, chat: bool, proposer=None,
                     memory_db: Path | None = None,
                     sleep_threshold: int = SLEEP_THRESHOLD_DEFAULT,
                     retrieval: str = "full", sleep_enabled: bool = True,
                     log=print) -> FullHooks:
    """Wire A_full's LIVE organs: memory db, calibrator, LoRA trainer, vLLM loader, slice.

    Everything lands under ``out_dir/<cfg.name>/`` (``memory_db`` overrides only the organ's
    path, per ``--memory-db``). Proposers are constructed PER MODEL NAME, because vLLM serves
    a runtime-loaded LoRA under its own name: both the arm (via :class:`AdapterProposer`) and
    the regression slice must ask for ``adapter_id`` rather than the base, or neither would
    ever exercise an adapter it just trained.

    ``proposer`` is the base-model client the caller already built; it is wrapped and handed
    back as ``hooks.proposer``, which is what the caller must give ``run_arm`` -- passing the
    unwrapped client instead would leave the arm permanently on the base weights. ``None``
    (tests that never generate) leaves the wrapper absent and every record stamps no adapter.

    Two resume guards run before anything is wired: the record/episode coherence check
    (:func:`_assert_resume_coherent`) and the db's own ``(arm, stream_hash)`` stamp
    (:meth:`~crucible.memory.store.MemoryStore.bind_identity`), which is what stops one arm's
    organ from being pointed at another's run. What CANNOT be guarded -- the value model and
    calibrator restarting untrained -- is documented in the module docstring.
    """
    from crucible.proposer.client import VLLMProposer          # local: the live serving path

    arm_dir = Path(out_dir) / cfg.name
    arm_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(memory_db) if memory_db is not None else arm_dir / MEMORY_DB_FILE
    db_path.parent.mkdir(parents=True, exist_ok=True)

    store = MemoryStore(db_path)
    store.bind_identity(cfg.name, stream_store.read_manifest(Path(stream_dir)).stream_hash)
    _assert_resume_coherent(store, arm_dir)

    registry = AdapterRegistry(arm_dir / ADAPTER_REGISTRY_FILE)
    adapters_dir = arm_dir / ADAPTERS_DIR
    server = VllmAdapterLoader(base_url)
    proposer_for = lambda model: VLLMProposer(base_url, model, chat=chat)   # noqa: E731
    slice_runner = DriverSliceRunner(
        cfg, stream_dir, server=server, adapters_dir=adapters_dir, proposer_for=proposer_for,
    )
    controller = SleepController(
        store, LoraTrainer(), server, slice_runner, registry,
        unit_loader=lambda unit_id: stream_store.read_unit(Path(stream_dir), unit_id),
        adapters_dir=adapters_dir, threshold=sleep_threshold, seed=cfg.seed,
    )
    arm_proposer = (None if proposer is None
                    else AdapterProposer(proposer, proposer_for, cfg.model))
    return FullHooks(store, value, Calibrator(), controller, registry,
                     sleep_records_path=arm_dir / SLEEP_RECORDS_FILE,
                     proposer=arm_proposer, retrieval=retrieval,
                     sleep_enabled=sleep_enabled,
                     probe_log_path=arm_dir / SYMPTOM_PROBE_LOG_FILE, log=log)


def build_mem_hooks(cfg: ArmConfig, stream_dir: Path, out_dir: Path, *,
                    memory_db: Path | None = None, retrieval: str = "full",
                    log=print) -> MemHooks:
    """Wire B_mem/B_symmem's organ ALONE: no registry, no controller, no proposer wrap.

    Mirrors ``build_full_hooks``'s store setup ONLY -- the arm_dir, the db path
    (``memory_db`` override else ``arm_dir / MEMORY_DB_FILE``), the store's identity stamp,
    and the resume coherence guard. Nothing else A_full wires (the adapter registry, the
    LoRA trainer, the vLLM adapter loader, the sleep controller) exists for B_mem: there is
    no sleep loop and no adapter to select, so building any of it would be dead weight that
    invites a future edit to wire it in and quietly turn B_mem into a second A_full.

    ``retrieval`` passes straight through to ``MemHooks`` (``"full"`` for B_mem, ``"symptom"``
    for B_symmem -- see ``MEM_ARMS``); the probe log always lands at
    ``arm_dir / SYMPTOM_PROBE_LOG_FILE``, same filename as ``build_full_hooks``, so a
    post-hoc reader does not need to know which builder produced a given arm's directory.
    """
    arm_dir = Path(out_dir) / cfg.name
    arm_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(memory_db) if memory_db is not None else arm_dir / MEMORY_DB_FILE
    db_path.parent.mkdir(parents=True, exist_ok=True)

    store = MemoryStore(db_path)
    store.bind_identity(cfg.name, stream_store.read_manifest(Path(stream_dir)).stream_hash)
    _assert_resume_coherent(store, arm_dir)

    return MemHooks(store, retrieval=retrieval,
                    probe_log_path=arm_dir / SYMPTOM_PROBE_LOG_FILE, log=log)
