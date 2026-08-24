"""Task 10 -- arm configs + the per-task attempt, with THE honest-measurement invariant.

The load-bearing test is ``test_visible_pass_hidden_fail_is_not_scored_as_pass``: a
submission that passes the VISIBLE suite but fails the HIDDEN suite must be recorded with
``hidden_pass False`` (from ``run_hidden``), never ``True`` (from the visible reward).
Computing the outcome from the visible report fabricates the experiment's primary endpoint.

Run WRAPPED (R-T2-6): the attempt touches the sandbox through ``search`` + ``run_hidden``.
"""
from crucible.run.arm import ARMS, ArmConfig, attempt_task
from crucible.run.types import Candidate
from crucible.sandbox.report import TestReport
from crucible.stream.compose import TaskSpec
from crucible.value.model import ConstantValue
from crucible.stream.units import Unit, sha256_text

# --- fixtures: the discriminator unit from Task 1 (visible pass != hidden pass) -------------
CORRECT = "def add(a, b):\n    return a + b\n"
BUGGY = "def add(a, b):\n    return a - b\n"
# passes visible add(1, 2) == 3 (a truthy) but fails hidden add(0, 0) == 0 (a falsy -> 1)
DISCRIMINATOR = "def add(a, b):\n    return a + b if a else 1\n"
VIS = ("from unit_x import add as candidate\ndef test_v0():\n"
       "    assert candidate(1, 2) == 3\n")
HID = ("from unit_x import add as candidate\ndef test_h0():\n"
       "    assert candidate(0, 0) == 0\n")
U = Unit("X/0", "unit_x", "add", BUGGY, VIS, HID, sha256_text(BUGGY), 1, 1, ())
SPEC = TaskSpec("k1", "X/0", "ARITH", "X/0|ARITH", 2, "first", ((1, 0), (2, 0)), False, 1)


class FakeProposer:
    """In-process proposer that returns scripted module sources and records every call."""

    def __init__(self, model: str, texts: list[str]) -> None:
        self.model = model
        self._texts = texts
        self.calls: list[dict] = []

    def generate(self, prompt, *, n, seed, max_tokens=1024, temperature=0.7):
        self.calls.append({"n": n, "seed": seed})
        return [Candidate(self._texts[i % len(self._texts)], None, 1.0) for i in range(n)]


def test_a_nomem_correct_repair_gives_hidden_pass_and_verified_visible():
    fake = FakeProposer(ARMS["A_noMem"].model, [CORRECT])
    rec, execs = attempt_task(ARMS["A_noMem"], U, SPEC, fake, ConstantValue())
    assert rec.hidden_pass is True
    assert rec.status == "verified_visible"
    assert rec.visible_reward == 1.0
    assert rec.arm == "A_noMem" and rec.unit_id == "X/0" and rec.phase == 2
    assert rec.tampered is False
    assert execs and execs[0].visible_reward == rec.visible_reward


def test_b_naive_is_single_shot_one_generate_no_refinement():
    fake = FakeProposer(ARMS["B_naive"].model, [CORRECT])
    rec, execs = attempt_task(ARMS["B_naive"], U, SPEC, fake, ConstantValue())
    assert len(fake.calls) == 1          # exactly one generate call: no refinement
    assert fake.calls[0]["n"] == 1       # single candidate
    assert rec.executions_charged <= 1


def test_visible_pass_hidden_fail_is_not_scored_as_pass():
    # THE INVARIANT: hidden_pass comes from run_hidden, never the visible reward.
    fake = FakeProposer(ARMS["A_noMem"].model, [DISCRIMINATOR])
    rec, execs = attempt_task(ARMS["A_noMem"], U, SPEC, fake, ConstantValue())
    assert rec.visible_reward == 1.0     # the submission DID pass every visible test
    assert rec.hidden_pass is False      # ... but the hidden oracle says it is wrong


def test_hidden_infra_error_leaves_hidden_pass_none(monkeypatch):
    # An infra failure on the HIDDEN run is "not measured" -> None, never False.
    infra = TestReport((), (), (), (), 0.0, "sandbox exploded")
    monkeypatch.setattr("crucible.run.arm.run_hidden", lambda *a, **k: infra)
    fake = FakeProposer(ARMS["A_noMem"].model, [CORRECT])
    rec, execs = attempt_task(ARMS["A_noMem"], U, SPEC, fake, ConstantValue())
    assert rec.hidden_pass is None
    assert rec.infra_error == "sandbox exploded"


def test_arm_registry_matches_frozen_spec():
    # A2: A_noMem is the chat-served instruct 1.5B; the B arms are raw-served base 9B.
    assert ARMS["A_noMem"] == ArmConfig("A_noMem", "Qwen/Qwen2.5-Coder-1.5B-Instruct", True, chat=True)
    assert ARMS["B_search"] == ArmConfig("B_search", "Qwen/Qwen3.5-9B", True, chat=False)
    assert ARMS["B_naive"] == ArmConfig("B_naive", "Qwen/Qwen3.5-9B", False, chat=False)
    assert ARMS["B_naive"].use_search is False


def test_chat_serving_is_an_arm_property_not_a_cli_default():
    """The instruct A_noMem proposer MUST be chat-served; the base B proposers MUST be raw-served.
    Binding chat to the arm stops `arm run --arm A_noMem` (no --chat) from silently serving raw."""
    assert ARMS["A_noMem"].chat is True
    assert ARMS["B_search"].chat is False and ARMS["B_naive"].chat is False
