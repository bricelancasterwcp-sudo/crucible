"""Run one arm over a task set, resumably, and select the ceiling pilot's tasks.

The driver (spec S4.7) is the loop *around* :func:`crucible.run.arm.attempt_task`: for each
task_key it loads the unit + mutant + taskspec from the S1 stream store, reconstructs the
per-task unit the agent must repair, attempts it, and accumulates the records. Task 14's
ceiling pilot calls :func:`run_arm` on 30 phase-1 tasks; nothing here spins a thread pool --
an arm serves one model, so the tasks run sequentially in the given order and the run is
reproducible from the seed alone.

Three things are load-bearing.

*The per-task unit carries the MUTATED module, not the canonical one.* ``read_unit`` returns
the unit with its correct ``module_src`` plus the visible/hidden test sources; the bug lives
in the mutant's ``mutated_src`` (``read_mutant``). :func:`attempt_task`/``search`` open with a
free run of ``unit.module_src`` to learn the symptom and put that same source in the repair
prompt -- so the module the agent sees must be the mutant's. ``_mutated_unit`` therefore
replaces just ``module_src`` with ``mutant.mutated_src`` and keeps everything else (the tests,
the module name, the entry point) from the canonical unit. Feeding the canonical source would
ask the agent to repair code that is already correct -- there would be no bug -- which
fabricates the experiment; the driver test's spy pins this.

*The run is resumable at task granularity.* A pilot of 30 sandboxed attempts can die
half-way (OOM, a killed shell), and re-attempting the tasks already scored would waste the
budget and, worse, could overwrite a recorded outcome with a second, differently-seeded one.
So the records are written after every attempt (a partial ``task_records.jsonl`` is always
consistent with what actually ran), and a re-run reads the task_keys already present and
skips them -- ``attempt_task`` is never called twice for the same key. The ``.DONE`` marker
(arm + stream_hash + seed) is written only once the whole requested set is processed.

*The pilot's tasks are drawn, not taken.* :func:`select_pilot_tasks` samples phase-1
(``kind == "first"``) keys with ``random.Random(f"{seed}:pilot")`` -- never the first N, which
would correlate the pilot with whatever order composition happened to shuffle the stream into.

Two further things, added by S3.

*``hooks`` is the ONE place an arm can be more than S2's driver, and ``hooks=None`` is
byte-identical to S2.* A_full is not a second driver: it is this loop with the memory/value/
sleep organs behind three calls (:class:`crucible.run.full.ArmHooks`). When ``hooks is None``
the memory keyword is not merely ``None``, it is ABSENT -- :func:`attempt_task` is called with
the exact S2 argument list -- so an S2 arm's records cannot drift by so much as a default, and
``dataclasses.replace`` is never applied to a record. The driver test's spy, whose signature
has no ``memory`` parameter at all, is what pins that.

*The clock is read HERE, once per task, and nowhere else in the organ.* Every module under
``crucible/memory/`` and ``crucible/sleep/`` takes ``now`` as an argument and reads no clock
(each says so in its own docstring), because a timestamp minted deep inside a store makes a
replay unreproducible and a record unauditable. :func:`utc_now` is that single read; the one
stamp is handed to BOTH hook calls of the task it belongs to, so an episode, the lesson
distilled from it and a sleep fired right after all carry the same instant. The format is
fixed-width UTC with a ``Z`` suffix (``2026-08-24T12:00:00Z``, no fractional seconds) because
``crucible.sleep.loop``'s staleness check compares these strings LEXICOGRAPHICALLY -- a mixed
offset or a variable-width fraction would silently break that ordering.
"""
from __future__ import annotations

import dataclasses
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from crucible.run.arm import ArmConfig, attempt_task
from crucible.run.records import (EXEC_RECORDS_FILE, TASK_RECORDS_FILE, ExecRecord,
                                  read_task_records, write_records)
from crucible.stream import store
from crucible.stream.compose import TaskSpec
from crucible.stream.units import Unit

if TYPE_CHECKING:                       # import-cycle-free: full.py imports this module
    from crucible.run.full import ArmHooks

DONE_FILE = ".DONE"

# Fixed-width UTC, Z-suffixed, no fractional seconds -- see the module docstring's clock note.
ISO_UTC = "%Y-%m-%dT%H:%M:%SZ"


def utc_now() -> str:
    """THE clock read of an arm run: ``2026-08-24T12:00:00Z``. See the module docstring.

    Fixed width and ``Z``-suffixed so ``a < b`` on two of these strings is a real ordering
    (``crucible.sleep.loop._is_stale`` depends on exactly that), and second-resolution
    because nothing downstream orders two events inside one second.
    """
    return datetime.now(timezone.utc).strftime(ISO_UTC)


def _mutated_unit(stream_dir: Path, task: TaskSpec) -> Unit:
    """The per-task unit the agent repairs: the canonical unit with the MUTANT's buggy module.

    ``read_unit`` gives the correct module plus the visible/hidden tests; ``read_mutant``
    carries the bug as ``mutated_src``. Only ``module_src`` is swapped -- the tests, module
    name and entry point stay -- so the agent sees the bug and is scored on the same suites.
    """
    unit = store.read_unit(stream_dir, task.unit_id)
    mutant = store.read_mutant(stream_dir, task.task_key)
    return dataclasses.replace(unit, module_src=mutant.mutated_src)


def _read_exec_records(out_path: Path) -> list[ExecRecord]:
    """Every prior ``ExecRecord`` under ``out_path`` (empty if none yet) -- the resume mirror.

    ``records`` ships a reader for task records but not exec records; the resume path needs
    both so a re-run appends to, rather than truncates, what earlier attempts wrote.
    """
    p = Path(out_path) / EXEC_RECORDS_FILE
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as fh:
        return [ExecRecord.from_dict(json.loads(line)) for line in fh if line.strip()]


def _prior_records(out_path: Path):
    """The (task_records, exec_records, done-keys) already on disk under ``out_path``."""
    task_recs = read_task_records(out_path) if (Path(out_path) / TASK_RECORDS_FILE).exists() else []
    return task_recs, _read_exec_records(out_path), {r.task_key for r in task_recs}


def _write_done(out_path: Path, cfg: ArmConfig, stream_hash: str) -> None:
    """Stamp the arm's output complete: arm + stream_hash + seed, byte-stable JSON."""
    payload = {"arm": cfg.name, "stream_hash": stream_hash, "seed": cfg.seed}
    (Path(out_path) / DONE_FILE).write_text(json.dumps(payload, sort_keys=True) + "\n",
                                            encoding="utf-8")


def run_arm(cfg: ArmConfig, stream_dir: Path, task_keys: list[str], proposer, value,
            out_dir: Path, *, log=print, hooks: "ArmHooks | None" = None) -> Path:
    """Attempt every ``task_key`` (in order) under arm ``cfg``; write its records + ``.DONE``.

    Loads each task's unit + mutant from ``stream_dir``, reconstructs the mutated unit, and
    calls :func:`attempt_task`. Records land under ``out_dir/<cfg.name>/`` after each attempt,
    so a crash leaves a consistent partial file. RESUMABLE: task_keys already present in that
    file are skipped -- ``attempt_task`` is not re-invoked for them. Returns the output dir.

    ``hooks`` (S3, A_full only) turns this loop into the memory arm without changing it:
    ``before_task`` supplies the retrieved block that becomes ``attempt_task``'s ``memory``,
    ``after_task`` writes the episode/lesson and returns the ``(retrieved_ids, adapter_id)``
    stamped onto the record, and ``between_tasks`` is where sleep may fire. ``None`` -- every
    other arm -- calls ``attempt_task`` with the S2 argument list and stamps nothing (see the
    module docstring).

    Three orderings inside the loop are load-bearing:

    * the record is written to disk BEFORE ``between_tasks``, so a sleep that crashes (or a
      GPU box that dies training an adapter) cannot cost the attempt that was already scored;
    * the stamp is applied BEFORE the write, so what is on disk is the final record and a
      resumed run never has to re-stamp;
    * ``solved_task_keys`` is recomputed from the accumulated records each time, so it counts
      the tasks this run has actually verified (``hidden_pass is True`` -- an unmeasured
      attempt is NOT solved) INCLUDING the ones a previous, crashed process recorded.
    """
    stream_dir = Path(stream_dir)
    manifest = store.read_manifest(stream_dir)
    by_key = {t.task_key: t for t in manifest.tasks}
    out_path = Path(out_dir) / cfg.name
    out_path.mkdir(parents=True, exist_ok=True)

    task_recs, exec_recs, done = _prior_records(out_path)
    for i, task_key in enumerate(task_keys, 1):
        if task_key in done:
            log(f"[{cfg.name}] {i}/{len(task_keys)} skip {task_key[:12]} (already recorded)")
            continue
        task = by_key.get(task_key)
        if task is None:
            raise KeyError(f"task_key {task_key!r} is not in stream {manifest.stream_hash[:12]}")
        unit = _mutated_unit(stream_dir, task)
        # hooks=None => the S2 call, argument for argument: the kwarg is ABSENT, not None.
        memory_kw = {} if hooks is None else {"memory": hooks.before_task(unit, task)}
        rec, execs, result = attempt_task(cfg, unit, task, proposer, value, **memory_kw)
        now = utc_now()                                   # THE clock read (see module docstring)
        if hooks is not None:
            retrieved_ids, adapter_id = hooks.after_task(unit, task, rec, result, now)
            rec = dataclasses.replace(rec, retrieved_ids=retrieved_ids, adapter_id=adapter_id)
        task_recs.append(rec)
        exec_recs.extend(execs)
        done.add(task_key)
        write_records(out_path, task_recs, exec_recs)     # after every attempt: crash-consistent
        log(f"[{cfg.name}] {i}/{len(task_keys)} {task_key[:12]} hidden_pass={rec.hidden_pass}")
        if hooks is not None:
            hooks.between_tasks([r.task_key for r in task_recs if r.hidden_pass is True], now)

    _write_done(out_path, cfg, manifest.stream_hash)
    return out_path


def select_pilot_tasks(stream_dir: Path, n: int, *, seed: int) -> list[str]:
    """``n`` phase-1 (``kind == "first"``) task_keys drawn with ``random.Random(f"{seed}:pilot")``.

    A seeded draw, never "first N": the sample moves with the seed, so the pilot is not
    pinned to composition's shuffle order. Raises rather than returning a short list when the
    stream has fewer than ``n`` phase-1 tasks.
    """
    manifest = store.read_manifest(Path(stream_dir))
    phase1 = [t.task_key for t in manifest.tasks if t.kind == "first"]
    if n > len(phase1):
        raise ValueError(f"stream has {len(phase1)} phase-1 tasks; cannot sample n={n}")
    return random.Random(f"{seed}:pilot").sample(phase1, n)
