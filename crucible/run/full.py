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
the ``SleepRecord``.

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
retrieves and remembers the ``(task_key, item_ids, hit)`` it produced; ``after_task`` needs
those ids for the episode's ``memory_item_ids`` and for the record's ``retrieved_ids``.
Re-retrieving inside ``after_task`` would be cheap but WRONG: by then the store may hold a
lesson minted from this very task, so the second read could report items the prompt never
contained. The pending state is therefore keyed by ``task_key`` and an ``after_task`` that
does not match it raises -- an unguarded version would silently stamp an empty tuple and
make "did this task get memory" (the column E1 is measured on) a lie that no record shows.

**Timestamps.** Nothing here reads a clock. ``now`` arrives from
:func:`crucible.run.driver.utc_now`, the one place an arm run reads one, so an episode, the
lesson distilled from it, and a sleep fired immediately after all carry the same instant.

**What is NOT wired here, deliberately.** The calibrator's ``should_abstain`` gate is
consulted by nothing in this module: an attempt's ``status`` is decided by the search that
made it, and rewriting it to ``"abstain"`` after the hidden oracle has already spoken would
restate history rather than change a decision. Abstention that actually suppresses a
submission has to live inside the search loop, where the decision is still live; recorded
here so the omission is a documented choice and not an oversight.
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from crucible.memory.distill import distill
from crucible.memory.retrieve import retrieve
from crucible.memory.schema import EpisodicRecord, content_id, episode_verified
from crucible.memory.store import MemoryStore
from crucible.run.arm import ArmConfig, attempt_task
from crucible.run.driver import _mutated_unit
from crucible.run.records import TaskRecord
from crucible.search.loop import SearchResult
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
"""The one arm these hooks are built for -- the CLI gates on this name (see ``cli.py``)."""

MEMORY_DB_FILE = "memory.sqlite3"
ADAPTERS_DIR = "adapters"
ADAPTER_REGISTRY_FILE = "adapters.jsonl"
SLEEP_RECORDS_FILE = "sleep_records.jsonl"
"""Everything an A_full run writes beside its records, all under ``out_dir/<arm>/``: the
organ, the trained adapters, the append-only adapter ledger, and one line per sleep cycle.
One directory per arm run means a re-run against a fresh ``--out`` is a fresh organ, which
is what "arms never share memory" (spec §2) means operationally."""

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


@runtime_checkable
class ArmHooks(Protocol):
    """What :func:`crucible.run.driver.run_arm` may call between attempts. A_full implements
    it; every other arm passes ``None`` and the driver stays byte-identical to S2.

    ``before_task`` returns the memory block for this attempt's prompts (``None`` = no
    memory, never ``""``). ``after_task`` returns ``(retrieved_ids, adapter_id)`` for the
    driver to stamp onto the record via ``dataclasses.replace`` -- the hooks never mutate a
    record themselves, because a frozen record that only the driver writes is what makes
    "the file traces attempt order" true. ``between_tasks`` is the sleep check.

    Both task-scoped hooks receive the MUTATED ``unit`` the agent actually repaired: the
    lesson's diff is taken against that source, so a hook handed the canonical unit would
    diff the fix against code that never had the bug.
    """

    def before_task(self, unit: Unit, taskspec: TaskSpec) -> str | None: ...

    def after_task(self, unit: Unit, taskspec: TaskSpec, record: TaskRecord,
                   result: SearchResult, now: str) -> tuple[tuple[str, ...], str | None]: ...

    def between_tasks(self, solved_task_keys: list[str], now: str) -> None: ...


@dataclasses.dataclass(frozen=True)
class _Pending:
    """What ``before_task`` retrieved, waiting for the ``after_task`` of the SAME task."""

    task_key: str
    item_ids: tuple[str, ...]
    hit: bool


class FullHooks:
    """A_full's :class:`ArmHooks`: retrieve -> attempt -> record + distill -> maybe sleep.

    ``value`` must be the v1 :class:`~crucible.value.online.OnlineValue` (not the bare
    ``Value`` protocol): this class calls ``begin_task`` and ``update_by_id``, which v0's
    ``ConstantValue`` does not have. That is deliberate -- an arm configured with the
    constant scorer would silently drop both the task-context features and the training
    signal, and a ``getattr``-guarded call would hide it. It must also be the SAME object
    the driver passes to ``attempt_task``, or the context ``begin_task`` sets is not the
    context the search's ``score`` calls read.
    """

    def __init__(self, store: MemoryStore, value: OnlineValue, calibrator: Calibrator,
                 sleep_controller: SleepController, registry: AdapterRegistry, *,
                 sleep_records_path: Path,
                 recalibrate_window: int = RECALIBRATE_WINDOW) -> None:
        self._store = store
        self._value = value
        self._calibrator = calibrator
        self._sleep = sleep_controller
        self._registry = registry
        self._sleep_records_path = Path(sleep_records_path)
        self._recalibrate_window = recalibrate_window
        self._pending: _Pending | None = None
        self.sleep_records: list[SleepRecord] = []

    @property
    def sleep_threshold(self) -> int:
        """The configured sleep trigger, read off the controller that owns it."""
        return self._sleep.threshold

    def before_task(self, unit: Unit, taskspec: TaskSpec) -> str | None:
        """Retrieve this task's memory block and set the value model's task context.

        The block is whatever :func:`crucible.memory.retrieve.retrieve` decides for this
        (unit_id, family) class -- ``None`` when the organ has nothing to say. The
        retrieval HIT (block present) is a task-level feature of the value model, so
        ``begin_task`` is called here, before any node is scored, exactly once per task.

        ``unit`` is part of the hook contract (pre-reg §9 writes retrieval as
        ``retrieve(unit, family, symptom)``) but this implementation reads none of it:
        retrieval keys on the CLASS -- ``(unit_id, family)`` -- not on the source text, and
        the symptom-conditioned form is not built. Taking the argument keeps the seam open
        for it; ignoring it keeps this implementation honest about what it actually uses.
        """
        block = retrieve(self._store, taskspec.unit_id, taskspec.family)
        hit = block.block is not None
        self._pending = _Pending(taskspec.task_key, block.item_ids, hit)
        self._value.begin_task(taskspec.family, hit)
        return block.block

    def after_task(self, unit: Unit, taskspec: TaskSpec, record: TaskRecord,
                   result: SearchResult, now: str) -> tuple[tuple[str, ...], str | None]:
        """Write the episode (always), the lesson (verified only), and learn from the outcome.

        Returns ``(retrieved_ids, adapter_id)`` for the driver to stamp. ``adapter_id`` is
        the registry's latest ACCEPTED adapter, read now rather than at ``before_task``:
        sleep only ever fires BETWEEN tasks, so what is accepted at the end of an attempt is
        what served it, and reading it here also covers a resumed run whose adapter was
        accepted by an earlier process.

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

        cls = provenance_class(pending.hit, taskspec.phase)
        confidence = self._calibrator.confidence(record.confidence, cls)
        episode = self._episode(taskspec, record, result, pending, now, confidence)
        self._store.write_episode(episode)

        if episode.verified and episode.landed_module is not None:
            spans = (taskspec.span,) if taskspec.span2 is None else (taskspec.span, taskspec.span2)
            self._store.write_semantic(distill(
                episode, mutated_src=unit.module_src, spans=spans,
                flipped_tests=result.symptom_failed, killing_tests=result.symptom_failed,
                now=now,
            ))

        if record.hidden_pass is not None:           # measured, so it can be learned from
            self._value.update_by_id(result.best_node_id, record.hidden_pass)
            self._calibrator.observe(record.confidence, cls, record.hidden_pass)

        return pending.item_ids, self._registry.latest_accepted()

    def between_tasks(self, solved_task_keys: list[str], now: str) -> None:
        """Let sleep fire if enough verified episodes have landed; recalibrate on an accept.

        ``recalibrate`` runs ONLY after an accepted sleep, because that is the only outcome
        that changes what the server is running: a rejected candidate leaves the serving
        model untouched, so the calibrator's observations are still exchangeable with the
        ones before it and re-fitting from a short window would just throw data away.
        """
        record = self._sleep.maybe_sleep(list(solved_task_keys), now)
        if record is None:
            return
        self.sleep_records.append(record)
        self._append_sleep_record(record)
        if record.accepted:
            self._calibrator.recalibrate(self._recalibrate_window)

    def _episode(self, taskspec: TaskSpec, record: TaskRecord, result: SearchResult,
                 pending: _Pending, now: str, confidence: float) -> EpisodicRecord:
        """The episode for one attempt -- written whether or not it worked.

        ``landed_module`` is the submitted module when the codec produced one, else ``None``
        ("nothing landed", not an empty module). ``last_verified_at`` is set only for a
        verified episode: the hidden suite just checked that claim, and for an unverified one
        there is no claim to have checked. ``confidence`` is the CALIBRATED P(hidden pass)
        for this attempt as the organ believed it just BEFORE this outcome was folded in --
        a real number the run can be audited against, not a mint-time constant.
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
            verified=verified, memory_item_ids=pending.item_ids, created_at=now,
            confidence=confidence, status="active", version=1,
            source_locator=f"arm:{record.arm}/task:{taskspec.task_key}",
            valid_at=now, invalid_at=None, expired_at=None,
            last_verified_at=now if verified else None, falsified_by=None,
            verification_method=VERIFICATION_METHOD,
        )

    def _append_sleep_record(self, record: SleepRecord) -> None:
        """One JSON object per line, keys sorted, UTF-8 -- the S1 store convention.

        Append-only, like the adapter registry: a sleep that happened must stay on disk even
        if the run later dies, or "why is this adapter serving" is unanswerable from the
        files alone.
        """
        self._sleep_records_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._sleep_records_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


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

    For each key: reload the unit + mutant through the driver's own ``_mutated_unit`` (reused,
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
            record, _execs, _result = attempt_task(cfg, _mutated_unit(self._stream_dir, task),
                                                   task, proposer, ConstantValue())
            solved += 1 if record.hidden_pass is True else 0
        return solved

    def _tasks(self) -> dict[str, TaskSpec]:
        if self._by_key is None:
            self._by_key = {t.task_key: t
                            for t in stream_store.read_manifest(self._stream_dir).tasks}
        return self._by_key


def build_full_hooks(cfg: ArmConfig, stream_dir: Path, out_dir: Path, *, base_url: str,
                     value: OnlineValue, chat: bool, memory_db: Path | None = None,
                     sleep_threshold: int = SLEEP_THRESHOLD_DEFAULT) -> FullHooks:
    """Wire A_full's LIVE organs: memory db, calibrator, LoRA trainer, vLLM loader, slice.

    Everything lands under ``out_dir/<cfg.name>/`` (``memory_db`` overrides only the organ's
    path, per ``--memory-db``). The proposer for the regression slice is constructed PER
    ADAPTER, because vLLM serves a runtime-loaded LoRA under its own model name: the slice
    must ask for ``adapter_id``, not for the base model, or it would measure the base twice
    and accept every candidate.
    """
    from crucible.proposer.client import VLLMProposer          # local: the live serving path

    arm_dir = Path(out_dir) / cfg.name
    arm_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(memory_db) if memory_db is not None else arm_dir / MEMORY_DB_FILE
    db_path.parent.mkdir(parents=True, exist_ok=True)

    store = MemoryStore(db_path)
    registry = AdapterRegistry(arm_dir / ADAPTER_REGISTRY_FILE)
    adapters_dir = arm_dir / ADAPTERS_DIR
    server = VllmAdapterLoader(base_url)
    slice_runner = DriverSliceRunner(
        cfg, stream_dir, server=server, adapters_dir=adapters_dir,
        proposer_for=lambda model: VLLMProposer(base_url, model, chat=chat),
    )
    controller = SleepController(
        store, LoraTrainer(), server, slice_runner, registry,
        unit_loader=lambda unit_id: stream_store.read_unit(Path(stream_dir), unit_id),
        adapters_dir=adapters_dir, threshold=sleep_threshold, seed=cfg.seed,
    )
    return FullHooks(store, value, Calibrator(), controller, registry,
                     sleep_records_path=arm_dir / SLEEP_RECORDS_FILE)
