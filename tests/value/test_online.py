"""Tests for the v1 trained value function (pure objects, unwrapped).

``OnlineValue`` is a hand-rolled logistic regression over a fixed feature vector
(spec S3 Task 7 brief, section 6): ``[bias, mean_logprob, self_certainty, depth,
*family_onehot(7), retrieval_hit]``. Family and retrieval-hit are TASK-level context set
once per task via :meth:`OnlineValue.begin_task` -- the search loop only ever calls
``score``/``update``, never ``begin_task`` itself (that is the A_full driver's job), so
these tests set context directly to pin the contract the driver depends on.

Determinism is the load-bearing invariant (pre-reg: a value that varied between calls on
the same node would fabricate the experiment) -- ``score`` is pure arithmetic over cached
features, no randomness anywhere. The mutation guard lives in
:func:`test_update_moves_the_score_toward_the_outcome`: freezing ``update`` to a no-op
must fail exactly that test (see Step 5 in the task brief).

These construct ``Node``/``Candidate`` directly -- no sandbox, no GPU -- so they run under
a plain ``pytest`` with nothing wrapped.
"""

from crucible.run.types import Candidate
from crucible.search.node import Node
from crucible.value.model import Value
from crucible.value.online import FAMILIES, OnlineValue


def _cand(text: str, mean_logprob: float | None = None,
          self_certainty: float | None = None) -> Candidate:
    return Candidate(text=text, mean_logprob=mean_logprob, self_certainty=self_certainty)


def _node(text: str = "x = 1\n", depth: int = 0, mean_logprob: float | None = -0.5,
          self_certainty: float | None = 0.7) -> Node:
    return Node.for_candidate(_cand(text, mean_logprob, self_certainty), depth=depth)


# --- FAMILIES: pinned schema --------------------------------------------------


def test_families_is_the_sorted_seven_family_tuple():
    # Pins the exact tuple from the brief: sorted, CONST included for schema stability
    # even though rung-1 streams carry none of it.
    assert FAMILIES == ("ARITH", "BOOL", "CMP", "CONST", "FLOW", "SDL", "UNARY")
    assert list(FAMILIES) == sorted(FAMILIES)


# --- score: bounds + determinism ----------------------------------------------


def test_score_is_in_unit_interval():
    value = OnlineValue()
    s = value.score(_node())
    assert 0.0 <= s <= 1.0


def test_score_is_deterministic_across_repeated_calls():
    value = OnlineValue()
    node = _node()
    first = value.score(node)
    second = value.score(node)
    assert first == second


def test_score_handles_none_mean_logprob_and_self_certainty():
    # Both are None when the serving path returned no logprobs; must not raise, and
    # must fall back to 0.0 in the feature vector rather than propagating None into math.
    value = OnlineValue()
    node = _node(mean_logprob=None, self_certainty=None)
    s = value.score(node)
    assert 0.0 <= s <= 1.0


# --- update: moves the prediction toward the observed outcome -----------------


def test_update_moves_the_score_toward_the_outcome():
    # THE mutation guard (Step 5): freezing update's weight-write to a no-op must fail
    # this test, and only this test's core assertion.
    value = OnlineValue()
    node = _node()
    before = value.score(node)
    value.update(node, outcome=True)
    after = value.score(node)
    assert after > before


def test_update_moves_the_score_down_for_a_false_outcome():
    value = OnlineValue()
    node = _node()
    # Bias the weights up first so there is room to move down.
    for _ in range(5):
        value.update(node, outcome=True)
    before = value.score(node)
    value.update(node, outcome=False)
    after = value.score(node)
    assert after < before


# --- begin_task: task-level context is part of the feature vector -------------


def test_begin_task_changes_the_score_for_the_same_node():
    # w starts at 0, so score is 0.5 regardless of context until training moves weight
    # onto a component that differs between contexts (here: the family one-hot and the
    # retrieval-hit flag). Train under one context, then compare across two contexts.
    value = OnlineValue()
    node = _node()
    value.begin_task("ARITH", retrieval_hit=True)
    for _ in range(10):
        value.update(node, outcome=True)

    value.begin_task("ARITH", retrieval_hit=True)
    score_hit = value.score(node)
    value.begin_task("BOOL", retrieval_hit=False)
    score_miss = value.score(node)

    assert score_hit != score_miss


def test_begin_task_defaults_before_any_call_do_not_raise():
    # Before begin_task is ever called (e.g. a unit test scoring in isolation), family
    # context must default to something safe rather than raising.
    value = OnlineValue()
    s = value.score(_node())
    assert 0.0 <= s <= 1.0


# --- update_by_id: the driver's deferred-outcome path --------------------------


def test_update_by_id_on_unseen_id_returns_false():
    value = OnlineValue()
    assert value.update_by_id("never-scored", outcome=True) is False


def test_update_by_id_on_seen_id_returns_true_and_learns():
    value = OnlineValue()
    node = _node()
    before = value.score(node)  # populates self._seen[node.node_id]
    assert value.update_by_id(node.node_id, outcome=True) is True
    after = value.score(node)
    assert after > before


def test_update_by_id_uses_the_score_time_cached_context_not_current_context():
    # Score under one task context (caching those features), switch context, then
    # confirm update_by_id still trains against the CACHED features by checking the
    # weight on the cached family's one-hot moved rather than the new context's family.
    value = OnlineValue()
    node = _node()
    value.begin_task("ARITH", retrieval_hit=False)
    value.score(node)  # caches ARITH-context features for node.node_id

    value.begin_task("BOOL", retrieval_hit=False)  # context switches before the outcome lands
    assert value.update_by_id(node.node_id, outcome=True) is True

    arith_idx = 4 + FAMILIES.index("ARITH")
    bool_idx = 4 + FAMILIES.index("BOOL")
    assert value.w[arith_idx] != 0.0
    assert value.w[bool_idx] == 0.0


# --- snapshot/restore: record-keeping round-trip -------------------------------


def test_snapshot_restore_round_trips_subsequent_scores():
    value = OnlineValue()
    node = _node()
    value.begin_task("CMP", retrieval_hit=True)
    for _ in range(3):
        value.update(node, outcome=True)
    snap = value.snapshot()
    expected = value.score(node)

    restored = OnlineValue()
    restored.restore(snap)
    restored.begin_task("CMP", retrieval_hit=True)
    actual = restored.score(node)

    assert actual == expected


# --- protocol conformance -------------------------------------------------------


def test_online_value_satisfies_the_value_protocol():
    assert isinstance(OnlineValue(), Value)
