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
"""
from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

from crucible.run.arm import ArmConfig, attempt_task
from crucible.run.records import (EXEC_RECORDS_FILE, TASK_RECORDS_FILE, ExecRecord,
                                  read_task_records, write_records)
from crucible.stream import store
from crucible.stream.compose import TaskSpec
from crucible.stream.units import Unit

DONE_FILE = ".DONE"


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
            out_dir: Path, *, log=print) -> Path:
    """Attempt every ``task_key`` (in order) under arm ``cfg``; write its records + ``.DONE``.

    Loads each task's unit + mutant from ``stream_dir``, reconstructs the mutated unit, and
    calls :func:`attempt_task`. Records land under ``out_dir/<cfg.name>/`` after each attempt,
    so a crash leaves a consistent partial file. RESUMABLE: task_keys already present in that
    file are skipped -- ``attempt_task`` is not re-invoked for them. Returns the output dir.
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
        rec, execs = attempt_task(cfg, _mutated_unit(stream_dir, task), task, proposer, value)
        task_recs.append(rec)
        exec_recs.extend(execs)
        done.add(task_key)
        write_records(out_path, task_recs, exec_recs)     # after every attempt: crash-consistent
        log(f"[{cfg.name}] {i}/{len(task_keys)} {task_key[:12]} hidden_pass={rec.hidden_pass}")

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
