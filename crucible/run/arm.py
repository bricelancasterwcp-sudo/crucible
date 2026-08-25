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
from crucible.search.loop import (BELIEVED, VERIFIED_VISIBLE, SearchResult, TaskConfidence,
                                  _clamp01, _value_score, search)
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
    chat: bool = False
    k: int = 8
    width: int = 4
    seed: int = 0


# Amendment A2 (2026-08-23, pre-lock): the small-arm proposer is Qwen2.5-Coder-1.5B-Instruct,
# not Qwen3.5-2B. The 2B (a Qwen3-VL base) failed the §4.7 landing gate on the full-module
# codec by repetition-degeneration that no budget or sampler penalty could fix; the 1.5B is
# the pre-registered §2 alternative and clears the gate when chat-served. See
# docs/findings/S2-ceiling-pilot.md §7.
#
# ``chat`` is a per-arm property, NOT a CLI default: the serving surface is dictated by the
# model, so an *instruct* proposer (A_noMem's 1.5B) MUST be chat-served and a *base* proposer
# (the 9B B arms) MUST be raw-served. Binding it here means ``crucible arm run --arm A_noMem``
# cannot silently serve the instruct model raw -- which would reproduce the ~6% empty
# completions the §4.7 gate rejects and corrupt the very records the experiment scores. The
# 9B baseline keeps chat=False pending its own §4.7 probe (may flip if that model is instruct).
#
# ``A_full`` is A_noMem's ArmConfig in every field that reaches the server: the SAME model,
# the SAME serving surface, the SAME search budget and seed. That is deliberate and it is
# the reason ArmConfig grows no memory/sleep/value columns -- ``ArmConfig`` is the SERVING
# IDENTITY, and the two arms differ by the pre-registered column (memory + value v1 + sleep)
# alone, which lives in the HOOKS the driver is handed (:mod:`crucible.run.full`). Putting a
# `use_memory` flag here instead would make the arms differ in a second place that
# ``attempt_task`` could read, and every honest comparison depends on it not being able to.
ARMS: dict[str, ArmConfig] = {
    "A_noMem": ArmConfig("A_noMem", "Qwen/Qwen2.5-Coder-1.5B-Instruct", True, chat=True),
    "A_full": ArmConfig("A_full", "Qwen/Qwen2.5-Coder-1.5B-Instruct", True, chat=True),
    # §2 big-arm FALLBACK, activated 2026-08-24 (findings S2.5-stack2.md §6-§7; lock
    # record docs/LOCK-A.md): the 9B failed the §4.7 landing gate both raw and chat-served,
    # so the B arms run the 14B coder (AWQ-served under this SERVED name -- see the SERVE
    # table). Instruct model => chat-served, like the A arms (the A2 lesson).
    "B_search": ArmConfig("B_search", "Qwen/Qwen2.5-Coder-14B-Instruct", True, chat=True),
    "B_naive": ArmConfig("B_naive", "Qwen/Qwen2.5-Coder-14B-Instruct", False, chat=True),
}


def _naive_attempt(cfg: ArmConfig, unit: Unit, proposer, value, *,
                   memory: str | None = None,
                   confidence: TaskConfidence | None = None) -> SearchResult:
    """The single-shot control (``B_naive``): one free symptom, one candidate, no refinement.

    One free (uncharged) visible run learns the symptom for the prompt; ``generate`` is called
    exactly once with ``n=1``; the sole candidate's visible suite is run once -- the single
    charged execution (0 if that run threw infra, which is never charged, ruling R7). No REx,
    no tree, no refinement: the honest no-search baseline.

    ``memory`` is threaded for signature parity with :func:`search`, and the resulting
    ``SearchResult`` carries the same ``root_prompt``/``symptom_failed`` the search arms report,
    so the memory organ reads one shape of result whichever arm produced it.

    ``confidence`` (S3) is likewise threaded for parity and is applied to the REPORTED
    confidence only -- this path's STATUS rule is untouched, deliberately. The single-shot
    control has nothing to abstain FROM: it draws one candidate and submits it, so there is no
    withheld alternative that "abstain" could describe, and its status vocabulary is
    verified/believed by construction. Routing it through :func:`_status` instead would give
    ``B_naive`` an abstain rate it has never had, which is a change to a control arm, not a
    wiring detail.
    """
    symptom = run(unit, unit.module_src, None)                 # free symptom, never charged
    prompt = build_prompt(unit, symptom, memory=memory)
    cand = proposer.generate(prompt, n=1, seed=cfg.seed)[0]
    node = Node.for_candidate(cand)
    report = run(unit, cand.text, None)                        # the one charged visible execution
    charged = 0 if report.infra_error is not None else 1
    if report.infra_error is None:
        node.apply_report(report)
    reward = node.visible_reward()
    status = VERIFIED_VISIBLE if reward >= 1.0 else BELIEVED
    conf = _value_score(value, node)
    if confidence is not None:
        conf = _clamp01(float(confidence.calibrate(conf)))
    return SearchResult(cand.text, node.node_id, reward, charged,
                        bool(cand.text.strip()), 1, status, conf,
                        root_prompt=prompt,
                        symptom_failed=(tuple(symptom.failed) + tuple(symptom.timed_out)
                                        + tuple(symptom.errored)))


def attempt_task(cfg: ArmConfig, unit: Unit, taskspec: TaskSpec, proposer, value,
                 *, memory: str | None = None, confidence: TaskConfidence | None = None
                 ) -> tuple[TaskRecord, list[ExecRecord], SearchResult]:
    """Attempt ``taskspec``'s repair under arm ``cfg``; return record + exec records + result.

    Runs the search (or the single-shot control), takes the final submission, and computes
    THE OUTCOME via :func:`run_hidden` -- the driver-side hidden oracle, never charged, never
    seen by the agent. ``hidden_pass`` is the hidden report's ``all_passed`` when it produced
    a verdict, else ``None`` (an infra failure is "not measured", never a fail). The proposer
    must be serving the arm's declared model -- or an adapter trained on it, which a proposer
    declares by carrying ``base_model`` (see the guard's own comment) -- or the attempt is not
    the one configured.

    ``memory`` is the S3 retrieved-memory block, passed straight down to the search (or to the
    single-shot control). ``None`` -- the default, and what A_noMem and the B arms pass -- makes
    every prompt byte-for-byte its S2 self, so the arms differ only by the pre-registered column.

    ``confidence`` is the S3 per-task calibration hook, passed down the same way. It changes no
    prompt and no execution -- only the reported confidence and (on the search path) the
    abstention rule applied to it, BEFORE the hidden oracle runs. That ordering is the point:
    abstention is a decision the arm makes about its own submission, so it has to be decided
    while the decision is still live, not restamped onto a record after ``run_hidden`` has
    already answered.

    The third return value is the raw :class:`~crucible.search.loop.SearchResult`. The record
    is a REDUCTION of it (it drops the root prompt, the submitted module and the symptom's
    failing tests), and S3's memory organ needs exactly those three: an episode stores the
    prompt sleep will train on and the module it landed, and a lesson cites the tests that
    flipped. Recomputing them from the record is impossible and re-deriving them by re-running
    anything is a second measurement of an attempt that already happened -- so the result
    travels out with the record it summarises. S2 callers unpack the first two and are
    otherwise unaffected.
    """
    served = getattr(proposer, "model", None)
    # The guard accepts the arm's own checkpoint, or a proposer that DECLARES itself an
    # adapter on that checkpoint (``base_model``). vLLM routes a runtime-loaded LoRA by model
    # NAME, so an arm running its own accepted adapter legitimately asks for ``adapter_id``
    # rather than the base -- see ``crucible.run.full.AdapterProposer``. Anything serving some
    # OTHER base is still refused: the point of the guard is that an arm cannot silently run a
    # different checkpoint, not that its model string can never change.
    if served != cfg.model and getattr(proposer, "base_model", None) != cfg.model:
        raise ValueError(f"arm {cfg.name!r} expects model {cfg.model!r} (or an adapter on "
                         f"it), proposer serves {served!r}")
    started = time.monotonic()
    if cfg.use_search:
        result = search(unit, proposer, value, seed=cfg.seed, k=cfg.k, width=cfg.width,
                        memory=memory, confidence=confidence)
    else:
        result = _naive_attempt(cfg, unit, proposer, value, memory=memory,
                                confidence=confidence)

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
    return record, execs, result
