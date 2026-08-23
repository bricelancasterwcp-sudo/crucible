"""Tests for the search ``Node`` and refinement ``Tree`` (pure objects, unwrapped).

These construct ``Node``/``Tree``/``TestReport``/``Candidate`` directly -- no sandbox,
no GPU -- so they run under a plain ``pytest`` with nothing wrapped.
"""

from crucible.run.types import Candidate
from crucible.sandbox.report import TestReport
from crucible.search.node import Node, Tree, classify_status
from crucible.stream.units import sha256_text


def _cand(text: str, self_certainty: float | None = None) -> Candidate:
    return Candidate(text=text, mean_logprob=None, self_certainty=self_certainty)


# --- node_id derivation ----------------------------------------------------


def test_node_id_is_sha256_of_candidate_text():
    cand = _cand("def solve():\n    return 1\n")
    node = Node.for_candidate(cand, parent_id=None)
    assert node.node_id == sha256_text(cand.text)


def test_for_candidate_starts_unexecuted():
    node = Node.for_candidate(_cand("x = 1\n"))
    assert node.report is None
    assert node.status == "unexecuted"
    assert node.parent_id is None


# --- visible_reward: the denominator is the whole measured suite ------------


def test_reward_is_zero_when_unexecuted():
    assert Node.for_candidate(_cand("x = 1\n")).visible_reward() == 0.0


def test_reward_all_pass_is_one():
    node = Node.for_candidate(_cand("ok\n"))
    node.apply_report(TestReport(("a", "b"), (), (), (), 0.1, None))
    assert node.visible_reward() == 1.0


def test_reward_none_pass_is_zero():
    node = Node.for_candidate(_cand("bad\n"))
    node.apply_report(TestReport((), ("a", "b"), (), (), 0.1, None))
    assert node.visible_reward() == 0.0


def test_reward_mixed_report_is_the_passing_fraction():
    # 2 passed + 2 failed => 0.5. A mutant that divides by ``passed`` alone (2/2)
    # or otherwise drops the failing tests from the denominator yields 1.0 here.
    node = Node.for_candidate(_cand("half\n"))
    node.apply_report(TestReport(("a", "b"), ("c", "d"), (), (), 0.1, None))
    assert node.visible_reward() == 0.5


def test_reward_denominator_includes_timed_out_and_errored():
    # 1 passed out of {1 passed, 1 failed, 1 timed_out, 1 errored} => 0.25.
    # Pins timed_out and errored into the denominator, not just failed.
    node = Node.for_candidate(_cand("quarter\n"))
    node.apply_report(TestReport(("a",), ("b",), ("c",), ("d",), 0.1, None))
    assert node.visible_reward() == 0.25


def test_reward_is_zero_and_safe_when_nothing_was_measured():
    # Empty / infra-errored report has an empty denominator: 0.0, not ZeroDivisionError.
    node = Node.for_candidate(_cand("infra\n"))
    node.apply_report(TestReport((), (), (), (), 0.1, "server down"))
    assert node.visible_reward() == 0.0


# --- status classification -------------------------------------------------


def test_classify_unexecuted_when_no_report():
    assert classify_status(None) == "unexecuted"


def test_classify_visible_pass_when_all_pass():
    assert classify_status(TestReport(("a",), (), (), (), 0.1, None)) == "visible_pass"


def test_classify_visible_fail_when_none_pass():
    assert classify_status(TestReport((), ("a",), (), (), 0.1, None)) == "visible_fail"


def test_classify_visible_partial_on_mixed_report():
    mixed = TestReport(("a", "b"), ("c", "d"), (), (), 0.1, None)
    assert classify_status(mixed) == "visible_partial"


def test_apply_report_sets_report_and_status_together():
    node = Node.for_candidate(_cand("mixed\n"))
    mixed = TestReport(("a",), ("b",), (), (), 0.1, None)
    node.apply_report(mixed)
    assert node.report is mixed
    assert node.status == "visible_partial"


# --- Tree: parent/child index ----------------------------------------------


def _tree_with_root() -> tuple[Tree, Node]:
    root = Node.for_candidate(_cand("root\n"))
    return Tree(root), root


def test_children_returns_nodes_in_insertion_order():
    tree, root = _tree_with_root()
    a = Node.for_candidate(_cand("a\n"), parent_id=root.node_id)
    b = Node.for_candidate(_cand("b\n"), parent_id=root.node_id)
    tree.add(a)
    tree.add(b)
    assert tree.children(root.node_id) == [a, b]


def test_children_of_unknown_or_leaf_node_is_empty():
    tree, root = _tree_with_root()
    assert tree.children("no-such-id") == []
    assert tree.children(root.node_id) == []


def test_children_does_not_expose_internal_list():
    tree, root = _tree_with_root()
    child = Node.for_candidate(_cand("c\n"), parent_id=root.node_id)
    tree.add(child)
    got = tree.children(root.node_id)
    got.append("mutation")
    assert tree.children(root.node_id) == [child]  # unaffected by caller mutation


# --- Tree.best_visible: reward, then self_certainty, then stable ------------


def test_best_visible_picks_highest_reward():
    tree, root = _tree_with_root()  # root unexecuted -> reward 0.0
    lo = Node.for_candidate(_cand("lo\n"), parent_id=root.node_id)
    lo.apply_report(TestReport(("a",), ("b",), (), (), 0.1, None))  # 0.5
    hi = Node.for_candidate(_cand("hi\n"), parent_id=root.node_id)
    hi.apply_report(TestReport(("a", "b"), (), (), (), 0.1, None))  # 1.0
    tree.add(lo)
    tree.add(hi)
    assert tree.best_visible() is hi


def test_best_visible_breaks_reward_ties_by_self_certainty():
    root = Node.for_candidate(_cand("root\n"))
    tree = Tree(root)
    weak = Node.for_candidate(_cand("weak\n", self_certainty=0.2), parent_id=root.node_id)
    strong = Node.for_candidate(_cand("strong\n", self_certainty=0.9), parent_id=root.node_id)
    same = TestReport(("a",), ("b",), (), (), 0.1, None)  # both 0.5
    weak.apply_report(same)
    strong.apply_report(same)
    tree.add(weak)   # inserted first, but lower certainty
    tree.add(strong)
    assert tree.best_visible() is strong


def test_best_visible_prefers_known_certainty_over_unknown():
    root = Node.for_candidate(_cand("root\n"))
    tree = Tree(root)
    unknown = Node.for_candidate(_cand("unknown\n", self_certainty=None), parent_id=root.node_id)
    known = Node.for_candidate(_cand("known\n", self_certainty=0.01), parent_id=root.node_id)
    same = TestReport(("a",), ("b",), (), (), 0.1, None)  # both 0.5
    unknown.apply_report(same)
    known.apply_report(same)
    tree.add(unknown)
    tree.add(known)
    assert tree.best_visible() is known


def test_best_visible_is_stable_on_full_ties():
    root = Node.for_candidate(_cand("root\n"))
    tree = Tree(root)
    same = TestReport(("a",), ("b",), (), (), 0.1, None)  # 0.5
    first = Node.for_candidate(_cand("first\n", self_certainty=0.5), parent_id=root.node_id)
    second = Node.for_candidate(_cand("second\n", self_certainty=0.5), parent_id=root.node_id)
    first.apply_report(same)
    second.apply_report(same)
    tree.add(first)
    tree.add(second)
    assert tree.best_visible() is first  # equal reward and certainty -> earliest inserted
