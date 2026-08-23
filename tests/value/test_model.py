"""Tests for the v0 value function (pure objects, unwrapped).

The value function predicts P(hidden pass | node) *before* execution so the search
(Task 7) can order which candidates spend the scarce K executions. v0 is an untrained
constant scaffold -- ``score`` returns the same number for every node, and ``update`` is
a no-op that only records that it was called. Real training is S3; these tests pin the
interface and the two invariants S3 will build on:

  * ``score`` is node-independent (a constant), so a mutant that makes it read *any* node
    attribute is killed (:func:`test_score_is_constant_across_structurally_different_nodes`,
    :func:`test_score_ignores_the_node_argument_entirely`), and
  * ``update`` changes nothing a caller can observe *except* an incrementing call counter,
    proving the training hook is wired even though it learns nothing yet.

These construct ``Node``/``TestReport``/``Candidate`` directly -- no sandbox, no GPU -- so
they run under a plain ``pytest`` with nothing wrapped.
"""

from crucible.run.types import Candidate
from crucible.sandbox.report import TestReport
from crucible.search.node import Node
from crucible.value.model import ConstantValue, Value


def _cand(text: str, self_certainty: float | None = None) -> Candidate:
    return Candidate(text=text, mean_logprob=None, self_certainty=self_certainty)


def _unexecuted_node(text: str = "x = 1\n") -> Node:
    return Node.for_candidate(_cand(text))


def _passing_node(text: str = "def solve():\n    return 1\n") -> Node:
    node = Node.for_candidate(_cand(text, self_certainty=0.9))
    node.apply_report(TestReport(("a", "b"), (), (), (), 0.1, None))  # reward 1.0
    return node


# --- score: the untrained constant -----------------------------------------


def test_score_returns_the_default_constant():
    assert ConstantValue().score(_unexecuted_node()) == 0.5


def test_score_returns_the_configured_constant():
    assert ConstantValue(c=0.8).score(_unexecuted_node()) == 0.8


def test_score_is_constant_across_structurally_different_nodes():
    # THE mutation guard. ``low`` (unexecuted, reward 0.0, no certainty) and ``high``
    # (reward 1.0, certainty 0.9, different text -> different node_id) differ in every
    # attribute a value could key off. A mutant that returns ``node.visible_reward()``
    # (or any node-derived value) yields 0.0 vs 1.0 here and fails; the constant does not.
    value = ConstantValue(c=0.5)
    low = _unexecuted_node("low\n")
    high = _passing_node("high\n")
    assert value.score(low) == value.score(high) == 0.5


def test_score_ignores_the_node_argument_entirely():
    # v0 returns ``c`` for ANY node. Passing an object with none of a Node's attributes
    # still returns ``c``; a mutant that reads ``node.<anything>`` raises AttributeError.
    assert ConstantValue(c=0.3).score(object()) == 0.3


# --- update: a wired-but-untrained no-op ------------------------------------


def test_update_returns_none():
    assert ConstantValue().update(_unexecuted_node(), outcome=True) is None


def test_update_counter_starts_at_zero():
    assert ConstantValue().updates == 0


def test_update_increments_the_observable_counter():
    # S3 proves the training hook is wired by watching this counter move, regardless of
    # the outcome value passed in.
    value = ConstantValue()
    value.update(_unexecuted_node("a\n"), outcome=True)
    value.update(_passing_node("b\n"), outcome=False)
    assert value.updates == 2


def test_update_does_not_change_the_score():
    # v0 learns nothing: score is the same constant before and after any number of updates.
    value = ConstantValue(c=0.7)
    node = _unexecuted_node()
    before = value.score(node)
    value.update(node, outcome=True)
    value.update(node, outcome=False)
    assert value.score(node) == before == 0.7


# --- protocol conformance --------------------------------------------------


def test_constant_value_satisfies_the_value_protocol():
    # ``Value`` is runtime_checkable, so the concrete scaffold structurally satisfies the
    # interface Task 7's search and Task 10's arms consume.
    assert isinstance(ConstantValue(), Value)
