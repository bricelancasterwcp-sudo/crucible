"""The budgeted propose->execute->refine search loop -- the agent's reasoning (spec S4.6).

The loop *is* the agent: it proposes candidate repairs (full-module rewrites), executes
their VISIBLE test suites under a strict budget, refines the best failing attempts with
that suite's feedback, and returns the final submission. Four honest-measurement rules are
load-bearing -- getting any of them wrong fabricates the experiment, so each is pinned by a
mutation-checked test:

1. *The free symptom run is not charged.* The loop opens with one execution of the visible
   suite on the mutated module (``unit.module_src``) to learn the symptom. Spec S4.6 grants
   the agent this one free look, so it never touches the ``BudgetMeter``.

2. *``k`` charged executions is a hard ceiling.* ``BudgetMeter.check`` gates every spend;
   when the budget is out the loop stops. ``BudgetExhausted`` never escapes ``search``, and
   ``executions_charged`` never exceeds ``k``.

3. *Infra failures are not measurements.* A report whose ``infra_error`` is set produced no
   verdict: it is not charged, its arm is not REx-updated, and the node is retired. Only a
   non-infra report increments ``executions_charged`` and folds a reward into REx (ruling R7).

4. *A fully-passing node stops the search.* Reward ``1.0`` on the visible suite is a
   verified-visible submission; there is nothing left to refine.

REx (Thompson sampling) ranks the arms -- unexecuted candidates by a value-heuristic prior, and
executed candidates by their reward posterior ``Beta(alpha+reward, beta+1-reward)``. A candidate
is a deterministic function of its text, so a node is executed at most once (dedup by node id):
``select`` re-picking an already-executed node means *refine it* (expand its children), and a
close-to-passing node -- higher reward, higher posterior -- is thus re-picked and refined before
a hopeless one (ruling R-S2-T7-1). Each node is executed once and expanded at most once, then
dropped, so the loop stays bounded (see :func:`_budget_loop`).

``Value`` is declared here as a minimal protocol because Task 8 (``crucible/value/model.py``)
may not be committed yet; that task supplies the concrete ``ConstantValue``. Anything with a
``score(node) -> float`` satisfies the loop; ``None`` falls back to proposer self-certainty.

**Abstention has TWO rules, and which one applies depends on whether the arm is calibrated
(S3).** They are deliberately different numbers, not a constant that drifted:

* :data:`ABSTAIN_THRESHOLD` (``0.5``) is STRUCTURAL and unconditional. It gates the REWARD
  half -- "fewer than half the visible tests pass" -- for every arm, always.
* The CONFIDENCE half depends on what the confidence number MEANS. With no
  :class:`TaskConfidence` hook (``confidence=None``: A_noMem, both B arms) the reported
  confidence is a RAW value-model score -- a ranking signal, not a probability -- so the
  honest gate is the same structural ``< 0.5``: "the ranker does not believe this either".
  With a hook (A_full) the number is a CALIBRATED P(hidden pass) for this task's provenance
  class, and the gate becomes the pre-registered §6 rule
  :func:`crucible.uncertainty.conformal.Calibrator.should_abstain` at
  :data:`~crucible.uncertainty.conformal.ABSTAIN_P` (``0.2``, inclusive). Comparing a
  calibrated probability against ``0.5`` would abstain on most of a hard stream, and
  comparing a raw score against ``0.2`` would abstain almost never; the two numbers are not
  interchangeable and neither is a mis-copy of the other.

The hook composes rather than replaces: ``reward < ABSTAIN_THRESHOLD`` must ALSO hold, so a
calibrated arm can never abstain on a submission that is passing most of its visible suite.
The reported ``confidence`` is whatever the decision actually used -- calibrated when a hook
is present, raw when it is not -- because a record whose confidence field did not drive its
own status would be unauditable.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

from crucible.proposer.client import Proposer
from crucible.proposer.prompt import build_prompt
from crucible.run.types import Candidate
from crucible.sandbox.budget import BudgetExhausted, BudgetMeter
from crucible.sandbox.report import TestReport
from crucible.sandbox.task_run import run
from crucible.search.node import Node, Tree
from crucible.search.rex import RexScheduler
from crucible.stream.units import Unit

ABSTAIN_THRESHOLD = 0.5
VERIFIED_VISIBLE = "verified_visible"
BELIEVED = "believed"
ABSTAIN = "abstain"


@runtime_checkable
class Value(Protocol):
    """The heuristic the search ranks unexecuted candidates by (Task 8 ships a concrete one)."""

    def score(self, node: Node) -> float: ...


@runtime_checkable
class TaskConfidence(Protocol):
    """One task's calibrated confidence and its abstention gate (S3, A_full only).

    Bound to ONE task's provenance class before the search starts, so neither method takes a
    class argument: the search does not know what a provenance class is and must not have to.
    ``calibrate`` maps a raw value score to P(hidden pass); ``should_abstain`` answers the §6
    gate for an already-calibrated probability. ``None`` everywhere else -- see the module
    docstring's two-rules note. ``crucible.run.full`` supplies the implementation.
    """

    def calibrate(self, score: float) -> float: ...

    def should_abstain(self, p: float) -> bool: ...


@dataclass(frozen=True)
class SearchResult:
    """The final submission of one search, plus the honest counts that frame it.

    ``status`` is one of ``verified_visible`` (reward ``1.0``), ``abstain`` (best reward and
    value confidence both below :data:`ABSTAIN_THRESHOLD`), or ``believed``. ``confidence`` is
    the value score of the best node, clamped to ``[0, 1]``. ``to_dict``/``from_dict`` are exact
    inverses -- every field is a JSON-native scalar or a tuple of them.

    The two trailing fields are what S3's memory organ reads back out of a finished search:
    ``root_prompt`` is the ROOT seeding prompt exactly as sent (memory block included -- it is
    what the episode stores and what sleep trains on), and ``symptom_failed`` is the visible
    tests the FREE symptom run did not pass (failed + timed out + errored). A verified fix
    passes the whole visible suite, so those are precisely the tests it flipped. Both are
    trailing and defaulted, and ``from_dict`` fills them in when absent, so an S2-era record
    -- written before either existed -- still loads.

    An EMPTY ``symptom_failed`` is deliberately ambiguous between two situations: the symptom
    run produced no verdict at all (``infra_error`` set -- nothing was measured), and it
    produced a verdict in which nothing failed. This dataclass does not distinguish them
    because a consumer must not act on either: both mean *no test is known to have flipped*,
    so neither yields a citable claim. A consumer needing the distinction has the symptom
    report itself; a consumer that does not (Task 11 feeds this straight into a lesson's
    ``flipped_tests``) is already safe, since ``falsify`` classifies an empty
    ``flipped_tests`` as a broken citation and refuses to measure it.
    """

    best_patch: str
    best_node_id: str
    visible_reward: float
    executions_charged: int
    landed: bool
    nodes: int
    status: str
    confidence: float
    root_prompt: str = ""
    symptom_failed: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """JSON-ready form: ``symptom_failed`` becomes a list so a file round-trip is exact."""
        d = asdict(self)
        d["symptom_failed"] = list(self.symptom_failed)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SearchResult":
        """Inverse of :meth:`to_dict`, tolerant of S2-era dicts that predate the S3 fields."""
        d = dict(d)
        d["root_prompt"] = d.get("root_prompt", "")
        d["symptom_failed"] = tuple(d.get("symptom_failed", ()))
        return cls(**d)


def _clamp01(x: float) -> float:
    """Clamp to ``[0, 1]`` -- value scores and confidences live in that range."""
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _value_score(value, node: Node) -> float:
    """The value heuristic for ``node`` in ``[0, 1]``; falls back to proposer self-certainty."""
    if value is None:
        cert = node.candidate.self_certainty
        return _clamp01(cert if cert is not None else 0.0)
    return _clamp01(float(value.score(node)))


def _feedback(report: TestReport) -> str:
    """Which visible tests still fail on an executed node -- the refinement prompt's evidence."""
    lines: list[str] = []
    if report.failed:
        lines.append("still failing: " + ", ".join(report.failed))
    if report.timed_out:
        lines.append("timed out: " + ", ".join(report.timed_out))
    if report.errored:
        lines.append("errored: " + ", ".join(report.errored))
    if not lines:
        lines.append("no visible failures were reported")
    return "\n".join(lines)


def _drop(ctx: "_Ctx", node_id: str) -> None:
    """Remove a spent node from the live pool -- from both ``pending`` and REx, kept in lockstep.

    A node is dropped when it is spent: an infra-skipped node (no verdict) the moment it is run,
    or an executed node once it has ALSO been expanded (executed and refined: nothing more to do).
    An executed-but-unexpanded node is *not* dropped -- it stays so its updated Beta posterior
    keeps competing in ``select`` (a close-to-passing node is re-picked and refined preferentially).
    Thompson arms are independent, so removing one never perturbs the others' posteriors. REx ships
    ``add_arm``/``select``/``update`` and no remove, so the loop owns this arm-lifecycle step
    (``_arms`` is REx's own registry; ``pending`` mirrors it exactly).
    """
    ctx.pending.pop(node_id, None)
    ctx.rex._arms.pop(node_id, None)


@dataclass
class _Ctx:
    """The mutable search state threaded through the helpers (kept off ``search``'s signature)."""

    unit: Unit
    proposer: Proposer
    value: object
    symptom: TestReport
    seed: int
    width: int
    per_test_timeout_s: float
    wall_cap_s: float
    tree: Tree
    rex: RexScheduler
    memory: str | None = None
    confidence: "TaskConfidence | None" = None
    root_prompt: str = ""
    pending: dict[str, Node] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)
    depth: dict[str, int] = field(default_factory=dict)


def _add_candidate(ctx: _Ctx, cand: Candidate, parent_id: str, node_depth: int) -> None:
    """Add one candidate as a fresh node + REx arm, unless its rewrite was already seen."""
    node = Node.for_candidate(cand, parent_id=parent_id, depth=node_depth)
    if node.node_id in ctx.seen:
        return
    ctx.seen.add(node.node_id)
    ctx.depth[node.node_id] = node_depth
    ctx.tree.add(node)
    ctx.pending[node.node_id] = node
    ctx.rex.add_arm(node.node_id, _value_score(ctx.value, node))


def _seed_roots(ctx: _Ctx, root: Node) -> None:
    """Seed the tree with ``width`` root candidates drawn from the free-symptom prompt.

    The prompt is kept on the ctx verbatim (memory block and all) so :func:`_finalize` reports
    the text that was actually sent, not a reconstruction that could drift from it.
    """
    prompt = build_prompt(ctx.unit, ctx.symptom, memory=ctx.memory)
    ctx.root_prompt = prompt
    for cand in ctx.proposer.generate(prompt, n=ctx.width, seed=ctx.seed):
        _add_candidate(ctx, cand, root.node_id, 1)


def _expand(ctx: _Ctx, node: Node) -> None:
    """Refine a failing node: build a feedback prompt and add ``width`` children as arms."""
    depth = ctx.depth[node.node_id]
    prompt = build_prompt(ctx.unit, ctx.symptom, feedback=_feedback(node.report),
                          memory=ctx.memory)
    for cand in ctx.proposer.generate(prompt, n=ctx.width, seed=ctx.seed + depth):
        _add_candidate(ctx, cand, node.node_id, depth + 1)


def _execute(ctx: _Ctx, meter: BudgetMeter, node: Node) -> int:
    """Run one unexecuted node; return how many charged executions it added (0 or 1).

    Infra (no verdict): dropped, NOT charged, NOT REx-updated (rule 3). A measurement: charged,
    its report applied, and its reward folded into the arm's ``Beta(alpha+reward, beta+1-reward)``.
    The node is KEPT in the pool so that updated posterior competes in later ``select`` calls --
    the canonical REx-over-the-refinement-tree behaviour (ruling R-S2-T7-1): a close-to-passing
    node earns a higher posterior and is re-picked, and thus refined, before a hopeless one.
    """
    report = run(ctx.unit, node.candidate.text, None,
                 per_test_timeout_s=ctx.per_test_timeout_s, wall_cap_s=ctx.wall_cap_s)
    if report.infra_error is not None:
        _drop(ctx, node.node_id)
        return 0
    meter.charge(report)
    node.apply_report(report)
    ctx.rex.update(node.node_id, node.visible_reward())   # posterior now competes in future select
    return 1


def _budget_loop(ctx: _Ctx, k: int) -> int:
    """Spend at most ``k`` charged executions; return the number actually charged.

    ``select`` returns a live arm. An UNEXECUTED arm is executed and KEPT (its posterior stays in
    play). An already-EXECUTED arm is refined -- ``_expand`` adds its children as new arms, then the
    parent is dropped (executed and expanded: done). The four honest-measurement rules: the free
    symptom is charged elsewhere (never here); ``meter.check`` makes ``k`` a hard ceiling; infra is
    not counted/charged/updated (in ``_execute``); a full visible pass stops the search.
    """
    meter = BudgetMeter(k)
    charged = 0
    while ctx.pending:
        try:
            meter.check()                        # rule 2: k is a hard ceiling, never breached
        except BudgetExhausted:
            break
        node_id = ctx.rex.select()
        node = ctx.pending[node_id]
        if node.report is None:                  # unexecuted -> execute, keep its posterior live
            charged += _execute(ctx, meter, node)
            if node.report is not None and node.visible_reward() >= 1.0:
                break                            # rule 4: a full visible pass stops the search
        else:                                    # already executed -> refine it, then retire it
            _expand(ctx, node)
            _drop(ctx, node_id)
    return charged


def _status(reward: float, confidence: float,
            conf: "TaskConfidence | None" = None) -> str:
    """``verified_visible`` on a full pass; ``abstain`` when reward AND confidence are both low.

    The reward half is structural and unconditional (``< ABSTAIN_THRESHOLD``). The confidence
    half is the raw ``< ABSTAIN_THRESHOLD`` compare when there is no hook, and ``conf
    .should_abstain(confidence)`` -- the calibrated §6 gate -- when there is. See the module
    docstring for why those are two different numbers rather than one that drifted.
    """
    if reward >= 1.0:
        return VERIFIED_VISIBLE
    low_confidence = (confidence < ABSTAIN_THRESHOLD if conf is None
                      else conf.should_abstain(confidence))
    if reward < ABSTAIN_THRESHOLD and low_confidence:
        return ABSTAIN
    return BELIEVED


def _finalize(ctx: _Ctx, executions_charged: int) -> SearchResult:
    """Score the tree's best visible node into the frozen :class:`SearchResult`."""
    best = ctx.tree.best_visible()
    reward = best.visible_reward()
    # The raw ranker score, then the calibrated probability when this arm has a hook. The
    # REPORTED confidence is the one the status decision below actually used.
    confidence = _value_score(ctx.value, best)
    if ctx.confidence is not None:
        confidence = _clamp01(float(ctx.confidence.calibrate(confidence)))
    return SearchResult(
        best_patch=best.candidate.text,
        best_node_id=best.node_id,
        visible_reward=reward,
        executions_charged=executions_charged,
        landed=bool(best.candidate.text.strip()),
        nodes=len(ctx.seen),
        status=_status(reward, confidence, ctx.confidence),
        confidence=confidence,
        root_prompt=ctx.root_prompt,
        symptom_failed=(tuple(ctx.symptom.failed) + tuple(ctx.symptom.timed_out)
                        + tuple(ctx.symptom.errored)),
    )


def search(unit: Unit, proposer: Proposer, value, *, seed: int, k: int = 8, width: int = 4,
           memory: str | None = None, confidence: "TaskConfidence | None" = None,
           per_test_timeout_s: float = 5.0, wall_cap_s: float = 60.0) -> SearchResult:
    """Budgeted propose->execute->refine search; returns the final submission (spec S4.6).

    ``value`` may be any object with ``score(node) -> float`` (or ``None``). Determinism: the REx
    rng is seeded ``f"{seed}:search:{unit.unit_id}"`` and the proposer seed is passed through, so
    an identical ``(unit, proposer, value, seed, ...)`` reproduces the search exactly.

    ``memory`` is the S3 retrieved-memory block. It reaches EVERY prompt this search builds --
    the root seeding prompt and every refinement prompt alike -- because a block that fades
    after the first round would make "the agent had memory" true only of the first attempt.
    ``None`` (A_noMem) adds nothing to any prompt: the arm is byte-for-byte its S2 self.

    ``confidence`` is the S3 per-task :class:`TaskConfidence` hook (A_full). It touches
    NOTHING inside the loop -- not a prompt, not an execution, not the REx ordering, which
    still ranks on the RAW value score -- only the final confidence number and the status
    rule derived from it (see :func:`_status`). ``None`` -- the default, and what every other
    arm passes -- leaves both exactly as S2 computed them.
    """
    symptom = run(unit, unit.module_src, None,          # rule 1: the FREE symptom, never charged
                  per_test_timeout_s=per_test_timeout_s, wall_cap_s=wall_cap_s)
    rng = random.Random(f"{seed}:search:{unit.unit_id}")
    root = Node.for_candidate(Candidate(unit.module_src, None, None))
    ctx = _Ctx(unit, proposer, value, symptom, seed, width, per_test_timeout_s, wall_cap_s,
               Tree(root), RexScheduler(rng=rng), memory=memory, confidence=confidence)
    ctx.seen.add(root.node_id)
    ctx.depth[root.node_id] = 0
    _seed_roots(ctx, root)
    charged = _budget_loop(ctx, k)
    return _finalize(ctx, charged)
