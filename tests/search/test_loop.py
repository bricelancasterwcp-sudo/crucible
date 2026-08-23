"""The budgeted propose->execute->refine loop, pinned on its honest-measurement rules.

Every test uses a fake in-process ``Proposer`` (scripted candidates, no network) and a real
``Unit`` whose canonical repair (``a + b``) is known, so the visible-suite verdicts come from
the real sandbox. WRAP these tests (they execute the sandbox):

    systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 -- \
        .venv/bin/python -m pytest tests/search/test_loop.py

The four invariants each test pins:
  (a) a correct first candidate -> verified_visible, reward 1.0, exactly one CHARGED
      execution (the free symptom run is not charged);
  (b) a proposer that never lands -> the budget ``k`` is a hard ceiling, never exceeded;
  (c) a failing first attempt whose refinement child passes -> verified_visible with >=2
      charged executions (refinement + feedback works);
  (d) a candidate whose run returns an ``infra_error`` -> that execution is neither charged
      nor REx-updated, and a later good candidate still solves.
"""
import pytest

from crucible.run.types import Candidate
from crucible.sandbox.report import TestReport
from crucible.search import loop as loop_mod
from crucible.search.loop import SearchResult, search
from crucible.search.rex import RexScheduler
from crucible.stream.units import Unit, sha256_text

# The mutated (buggy) module: ``a - b`` fails both visible tests. Its correct repair is ``a + b``.
BUG = "def add(a, b):\n    return a - b\n"
FIX = "def add(a, b):\n    return a + b\n"
FIX2 = "def add(a, b):\n    return (a) + (b)\n"  # also correct, distinct text -> distinct node
VIS = (
    "from unit_x import add as candidate\n"
    "def test_v0():\n    assert candidate(1, 2) == 3\n"
    "def test_v1():\n    assert candidate(2, 3) == 5\n"
)
HID = (
    "from unit_x import add as candidate\n"
    "def test_h0():\n    assert candidate(0, 0) == 0\n"
)


def _unit() -> Unit:
    return Unit("X/0", "unit_x", "add", BUG, VIS, HID, sha256_text(BUG), 2, 1, ())


class ConstantValue:
    """A stand-in ``Value`` (Task 8 ships the concrete one): a fixed heuristic score."""

    def __init__(self, v: float) -> None:
        self._v = v

    def score(self, node) -> float:
        return self._v


class ScriptedProposer:
    """Returns a fixed list of candidate texts per successive ``generate`` call."""

    model = "fake"

    def __init__(self, scripts: list[list[str]], cert: float = 0.5) -> None:
        self._scripts = scripts
        self._cert = cert
        self.calls = 0

    def generate(self, prompt, *, n, seed, max_tokens=1024, temperature=0.7):
        texts = self._scripts[min(self.calls, len(self._scripts) - 1)]
        self.calls += 1
        return [Candidate(t, None, self._cert) for t in texts]


class FailingProposer:
    """Never proposes a passing patch; emits distinct wrong modules up to a bounded pool.

    Distinct texts (unique node ids) keep the frontier replenished so the budget -- not an
    empty tree -- is what stops the loop; the bound keeps a budget-guardless mutant from
    looping forever (it exceeds ``k`` and then drains, so the mutation check fails, not hangs).
    """

    model = "fake"

    def __init__(self, limit: int = 12) -> None:
        self._i = 0
        self._limit = limit

    def generate(self, prompt, *, n, seed, max_tokens=1024, temperature=0.7):
        out = []
        for _ in range(n):
            j = min(self._i, self._limit)
            self._i += 1
            out.append(Candidate(f"# variant {j}\n{BUG}", None, 0.4))
        return out


@pytest.mark.timeout(120)
def test_first_candidate_correct_is_verified_visible():
    proposer = ScriptedProposer([[FIX]])
    result = search(_unit(), proposer, ConstantValue(0.9), seed=1, k=8, width=1)
    assert result.status == "verified_visible"
    assert result.visible_reward == 1.0
    # The free symptom run is NOT charged: exactly one charged execution (the correct seed).
    assert result.executions_charged == 1
    assert 1 <= result.executions_charged <= 8
    assert result.landed
    assert result.best_patch == FIX


@pytest.mark.timeout(180)
def test_never_landing_never_exceeds_the_budget():
    k = 4
    result = search(_unit(), FailingProposer(), ConstantValue(0.2), seed=2, k=k, width=2)
    assert result.executions_charged <= k       # the budget is a hard ceiling
    assert result.executions_charged == k       # and it actually binds (frontier never empties)
    assert result.status in {"believed", "abstain"}
    assert result.visible_reward < 1.0


@pytest.mark.timeout(150)
def test_refinement_child_passes_after_first_fails():
    first_fail = "# attempt 1\n" + BUG        # distinct from the buggy root; fails both tests
    proposer = ScriptedProposer([[first_fail], [FIX]])
    result = search(_unit(), proposer, ConstantValue(0.9), seed=3, k=8, width=1)
    assert result.status == "verified_visible"
    assert result.visible_reward == 1.0
    # First attempt charged, then a refinement CHILD charged: proves refine + feedback.
    assert result.executions_charged >= 2
    assert result.executions_charged <= 8
    assert proposer.calls >= 2                 # a refinement round actually happened


@pytest.mark.timeout(150)
def test_infra_error_is_not_charged_or_rex_updated(monkeypatch):
    unit = _unit()
    real_run = loop_mod.run
    state = {"done": False, "infra_patch": None}
    calls: list[str] = []

    def fake_run(u, patch, subset, **kw):
        calls.append(patch)
        # The FIRST budget-loop execution (any candidate, never the free symptom) returns infra.
        if patch != u.module_src and not state["done"]:
            state["done"] = True
            state["infra_patch"] = patch
            return TestReport((), (), (), (), 0.0, "sandbox infra boom")
        return real_run(u, patch, subset, **kw)

    monkeypatch.setattr(loop_mod, "run", fake_run)

    updated: list[str] = []
    orig_update = RexScheduler.update

    def spy_update(self, arm_id, reward):
        updated.append(arm_id)
        return orig_update(self, arm_id, reward)

    monkeypatch.setattr(RexScheduler, "update", spy_update)

    proposer = ScriptedProposer([[FIX, FIX2]])   # two distinct correct seeds
    result = search(unit, proposer, ConstantValue(0.5), seed=4, k=8, width=2)

    assert state["done"]                                 # the infra path was actually exercised
    infra_id = sha256_text(state["infra_patch"])
    assert result.status == "verified_visible"
    assert result.visible_reward == 1.0
    # The infra execution is NOT counted; only the good (non-infra) run charges.
    assert result.executions_charged == 1
    # The infra node is NOT REx-updated; the good one is.
    assert infra_id not in updated
    good_patch = next(
        p for p in calls if p != unit.module_src and p != state["infra_patch"]
    )
    assert sha256_text(good_patch) in updated


# --- Fix round 1 (R-S2-T7-1): the reward posterior drives refinement order ---

# A_HALF passes test_v0 only (visible_reward 0.5); B_ZERO fails both (0.0). Both are distinct from
# the buggy root module, so neither dedups against it. Reports are canned (monkeypatched run) so a
# 200-seed sweep is instant AND fully deterministic (every REx draw is seeded).
A_HALF = "def add(a, b):\n    return 3 if (a, b) == (1, 2) else 0\n"
B_ZERO = "# b\n" + BUG


class OrderProbeProposer:
    """Seeds A and B; on every refinement returns A/B again (deduped), recording expansion order.

    A refinement prompt carries the parent's still-failing tests, so the feedback text names which
    node is being expanded: A -> ``still failing: test_v1``; B -> ``still failing: test_v0, test_v1``.
    """

    model = "fake"

    def __init__(self) -> None:
        self.expand_order: list[str] = []

    def generate(self, prompt, *, n, seed, max_tokens=1024, temperature=0.7):
        if "still failing:" in prompt:
            label = "B" if "still failing: test_v0, test_v1" in prompt else "A"
            if label not in self.expand_order:
                self.expand_order.append(label)
        return [Candidate(A_HALF, None, 0.0), Candidate(B_ZERO, None, 0.0)]


def _canned_run(u, patch, subset, **kw):
    if patch == A_HALF:
        return TestReport(("test_v0",), ("test_v1",), (), (), 0.0, None)   # visible_reward 0.5
    return TestReport((), ("test_v0", "test_v1"), (), (), 0.0, None)        # visible_reward 0.0


def _a_expanded_before_b(order: list[str]) -> bool:
    if "A" not in order:
        return False
    if "B" not in order:
        return True
    return order.index("A") < order.index("B")


@pytest.mark.timeout(120)
def test_posterior_drives_refinement_order(monkeypatch):
    # Canned reports -> no sandbox, instant + deterministic. value is constant, so REx priors are
    # identical across arms and any ordering signal comes purely from the reward posterior.
    monkeypatch.setattr(loop_mod, "run", _canned_run)
    unit = _unit()
    n_seeds = 200
    a_first = 0
    for s in range(n_seeds):
        probe = OrderProbeProposer()
        search(unit, probe, ConstantValue(0.0), seed=s, k=8, width=2)
        assert len(probe.expand_order) == 2          # both nodes get refined
        if _a_expanded_before_b(probe.expand_order):
            a_first += 1
    # A (reward 0.5) earns the higher posterior, so it is re-picked -- and thus refined -- before
    # B (0.0) in a strong majority. Measured deterministically: real 0.730 vs retire-on-execute
    # (inert posterior) 0.555, so this threshold passes the real loop and fails the inert mutant.
    assert a_first / n_seeds >= 0.70


def test_search_result_round_trip():
    r = SearchResult("def f():\n    pass\n", "abc123", 0.5, 3, True, 7, "believed", 0.42)
    assert SearchResult.from_dict(r.to_dict()) == r
