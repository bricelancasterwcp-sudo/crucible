"""Arm configs + the per-task attempt -- where the experiment's PRIMARY endpoint is computed.

An *arm* is a configured way to attempt one repair task (spec S4.7). :func:`attempt_task`
runs the arm's search (or a single generate for the naive control), takes the final
submission, and computes THE OUTCOME: does that submission pass the HIDDEN suite. Task 12's
driver calls it over a whole task stream; Task 14 runs the ceiling pilot on top.

THE INVARIANT (non-negotiable, spec S4.7 / ruling R7): ``hidden_pass`` comes from
:func:`crucible.sandbox.task_run.run_hidden` and NOTHING else. The visible suite is the
surface the agent's search is scored on -- a candidate can overfit it -- so a submission
that passes every visible test can still be wrong. Reading ``hidden_pass`` off the visible
reward would fabricate the experiment's headline number, scoring an overfit patch as a
success. And when the hidden run itself throws an infra error it produced NO verdict, so
``hidden_pass`` is ``None`` ("not measured"), never ``False`` ("measured, failed") -- the
``None`` is load-bearing (see :mod:`crucible.run.records`).

Three further honest-measurement choices:

* *The hidden run is never charged and never seen by the agent.* It is a driver-side oracle
  run only here, after the search has already produced its final submission. It is not part
  of ``executions_charged`` and no candidate's prompt ever contains its result.

* *``tampered`` is always False here (ruling R-S2-PF1).* In S2's codec the agent emits module
  text only; the driver owns the test files, so a submission cannot reach the hidden oracle
  illegitimately. The field stays on the record for S3, where an agent might touch tests.

* *The naive arm is single-shot.* ``B_naive`` (spec S4.7) draws exactly ONE candidate
  (``n=1``) from one free-symptom prompt and never refines, so its budget is at most one
  charged visible execution. That is the no-search control the search arms are measured
  against; letting it refine or sample more would blur the comparison.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from crucible.proposer.prompt import build_prompt
from crucible.run.records import ExecRecord, TaskRecord
from crucible.sandbox.task_run import run, run_hidden
from crucible.search.loop import (BELIEVED, VERIFIED_VISIBLE, SearchResult, _value_score,
                                  search)
from crucible.search.node import Node
from crucible.stream.compose import TaskSpec
from crucible.stream.units import Unit


@dataclass(frozen=True)
class ArmConfig:
    """One arm: how a task is attempted, and the served-model id the run must be serving.

    ``use_search`` is True for the search arms (``A_noMem``/``B_search``) and False for the
    single-shot control (``B_naive``). ``k``/``width``/``seed`` are the search budget and the
    determinism seed; they are ignored on the single-shot path. Frozen so an arm's identity
    cannot drift mid-run.
    """

    name: str
    model: str
    use_search: bool
    k: int = 8
    width: int = 4
    seed: int = 0


ARMS: dict[str, ArmConfig] = {
    "A_noMem": ArmConfig("A_noMem", "Qwen/Qwen3.5-2B", True),
    "B_search": ArmConfig("B_search", "Qwen/Qwen3.5-9B", True),
    "B_naive": ArmConfig("B_naive", "Qwen/Qwen3.5-9B", False),
}


def _naive_attempt(cfg: ArmConfig, unit: Unit, proposer, value) -> SearchResult:
    """The single-shot control (``B_naive``): one free symptom, one candidate, no refinement.

    One free (uncharged) visible run learns the symptom for the prompt; ``generate`` is called
    exactly once with ``n=1``; the sole candidate's visible suite is run once -- the single
    charged execution (0 if that run threw infra, which is never charged, ruling R7). No REx,
    no tree, no refinement: the honest no-search baseline.
    """
    symptom = run(unit, unit.module_src, None)                 # free symptom, never charged
    cand = proposer.generate(build_prompt(unit, symptom), n=1, seed=cfg.seed)[0]
    node = Node.for_candidate(cand)
    report = run(unit, cand.text, None)                        # the one charged visible execution
    charged = 0 if report.infra_error is not None else 1
    if report.infra_error is None:
        node.apply_report(report)
    reward = node.visible_reward()
    status = VERIFIED_VISIBLE if reward >= 1.0 else BELIEVED
    return SearchResult(cand.text, node.node_id, reward, charged,
                        bool(cand.text.strip()), 1, status, _value_score(value, node))


def attempt_task(cfg: ArmConfig, unit: Unit, taskspec: TaskSpec, proposer, value,
                 ) -> tuple[TaskRecord, list[ExecRecord]]:
    """Attempt ``taskspec``'s repair under arm ``cfg``; return its record + exec records.

    Runs the search (or the single-shot control), takes the final submission, and computes
    THE OUTCOME via :func:`run_hidden` -- the driver-side hidden oracle, never charged, never
    seen by the agent. ``hidden_pass`` is the hidden report's ``all_passed`` when it produced
    a verdict, else ``None`` (an infra failure is "not measured", never a fail). The proposer
    must be serving the arm's declared model, or the attempt is not the one configured.
    """
    if getattr(proposer, "model", None) != cfg.model:
        raise ValueError(f"arm {cfg.name!r} expects model {cfg.model!r}, "
                         f"proposer serves {getattr(proposer, 'model', None)!r}")
    started = time.monotonic()
    if cfg.use_search:
        result = search(unit, proposer, value, seed=cfg.seed, k=cfg.k, width=cfg.width)
    else:
        result = _naive_attempt(cfg, unit, proposer, value)

    rh = run_hidden(unit, result.best_patch)                   # THE OUTCOME ORACLE (uncharged)
    hidden_pass = rh.all_passed if rh.infra_error is None else None   # None = not measured
    wall_s = time.monotonic() - started

    record = TaskRecord(
        task_key=taskspec.task_key, arm=cfg.name, unit_id=taskspec.unit_id,
        family=taskspec.family, phase=taskspec.phase, kind=taskspec.kind,
        landed=result.landed, status=result.status, confidence=result.confidence,
        visible_reward=result.visible_reward, executions_charged=result.executions_charged,
        hidden_pass=hidden_pass, tampered=False, infra_error=rh.infra_error,
        tokens=None, wall_s=wall_s, gpu_s=None,
    )
    # One per-attempt summary ExecRecord for the final submission's charged visible execution.
    # ``search`` does not expose per-node records, so this is a summary, not one row per node;
    # its ``infra_error`` is None (the hidden run's infra is the TaskRecord's, not this row's).
    execs = [ExecRecord(taskspec.task_key, cfg.name, result.best_node_id,
                        result.visible_reward, result.executions_charged > 0, wall_s, None)]
    return record, execs
