"""A search ``Node`` (one candidate patch) and the refinement ``Tree`` over them.

Each candidate rewrite is a node; refining a node yields a child, so the search
(Task 7) walks a tree rooted at the first attempt. Two design choices are load-bearing.

*The reward denominator is the whole measured suite.* ``visible_reward`` is
``passed / (passed + failed + timed_out + errored)`` -- every test the visible suite
actually produced a verdict for. Timed-out and errored tests are *failures to pass*,
so they stay in the denominator; dropping them (or dividing by ``passed`` alone) would
score a half-passing patch as a full pass and let search stop early on a wrong answer.
A report that measured nothing (empty buckets, e.g. an infra error) has reward ``0.0``
rather than raising -- an unmeasured node is worth no more than an unexecuted one.

*``status`` is denormalised.* It duplicates what ``visible_reward``/``report`` already
imply, but the search loop checks it far more often than it recomputes a fraction, so a
cached coarse label (unexecuted / pass / fail / partial) earns its place. ``apply_report``
is the single write path that keeps ``report`` and ``status`` from drifting.

``best_visible`` ranks by reward, then by proposer ``self_certainty`` (an *unknown*
``None`` certainty ranks below any known value -- absent is not zero), then falls back to
insertion order so ties resolve deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass

from crucible.run.types import Candidate
from crucible.sandbox.report import TestReport
from crucible.stream.units import sha256_text

UNEXECUTED = "unexecuted"
VISIBLE_PASS = "visible_pass"
VISIBLE_FAIL = "visible_fail"
VISIBLE_PARTIAL = "visible_partial"


def _reward(report: TestReport) -> float:
    """Passing fraction over every test the suite returned a verdict for; 0.0 if none."""
    total = (len(report.passed) + len(report.failed)
             + len(report.timed_out) + len(report.errored))
    if total == 0:
        return 0.0
    return len(report.passed) / total


def classify_status(report: TestReport | None) -> str:
    """Coarse label for a report: unexecuted / visible_pass / visible_fail / visible_partial."""
    if report is None:
        return UNEXECUTED
    reward = _reward(report)
    if reward >= 1.0:
        return VISIBLE_PASS
    if reward <= 0.0:
        return VISIBLE_FAIL
    return VISIBLE_PARTIAL


@dataclass
class Node:
    """One candidate patch in the refinement tree, plus its visible-suite verdict.

    Mutable by design: ``report`` and ``status`` are ``None``/``"unexecuted"`` until the
    node is run, then set together via :meth:`apply_report`. ``node_id`` is the sha256 of
    ``candidate.text`` -- the cached-result key, so the same rewrite is the same node.
    """

    node_id: str
    candidate: Candidate
    parent_id: str | None
    report: TestReport | None = None
    status: str = UNEXECUTED

    @classmethod
    def for_candidate(cls, candidate: Candidate, parent_id: str | None = None) -> "Node":
        """Build an unexecuted node, keying ``node_id`` off ``candidate.text``."""
        return cls(sha256_text(candidate.text), candidate, parent_id)

    def visible_reward(self) -> float:
        """Fraction of visible tests passing; 0.0 if unexecuted."""
        if self.report is None:
            return 0.0
        return _reward(self.report)

    def apply_report(self, report: TestReport) -> None:
        """Record an execution result, keeping ``report`` and ``status`` consistent."""
        self.report = report
        self.status = classify_status(report)


class Tree:
    """A refinement tree of nodes, indexed by id and by parent for the search loop."""

    def __init__(self, root: Node) -> None:
        self._by_id: dict[str, Node] = {root.node_id: root}
        self._order: list[Node] = [root]
        self._children: dict[str, list[Node]] = {}

    def add(self, node: Node) -> None:
        """Insert ``node`` and file it under its parent (order preserved for stable ties)."""
        self._by_id[node.node_id] = node
        self._order.append(node)
        if node.parent_id is not None:
            self._children.setdefault(node.parent_id, []).append(node)

    def children(self, node_id: str) -> list[Node]:
        """Direct children of ``node_id`` in insertion order (a fresh, caller-owned list)."""
        return list(self._children.get(node_id, ()))

    def best_visible(self) -> Node:
        """Highest ``visible_reward``; ties broken by ``self_certainty``, then insertion order."""
        return max(self._order, key=self._rank)

    @staticmethod
    def _rank(node: Node) -> tuple[float, float]:
        # ``max`` returns the first element among equal keys, so equal (reward, certainty)
        # pairs resolve to the earliest-inserted node -- the stable tie-break.
        cert = node.candidate.self_certainty
        return (node.visible_reward(), cert if cert is not None else float("-inf"))
