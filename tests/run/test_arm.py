"""Task 10 -- arm configs + the per-task attempt, with THE honest-measurement invariant.

The load-bearing test is ``test_visible_pass_hidden_fail_is_not_scored_as_pass``: a
submission that passes the VISIBLE suite but fails the HIDDEN suite must be recorded with
``hidden_pass False`` (from ``run_hidden``), never ``True`` (from the visible reward).
Computing the outcome from the visible report fabricates the experiment's primary endpoint.

Run WRAPPED (R-T2-6): the attempt touches the sandbox through ``search`` + ``run_hidden``.
"""
import pytest

from crucible.run import arm
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
        self.prompts: list[str] = []

    def generate(self, prompt, *, n, seed, max_tokens=1024, temperature=0.7):
        self.calls.append({"n": n, "seed": seed})
        self.prompts.append(prompt)
        return [Candidate(self._texts[i % len(self._texts)], None, 1.0) for i in range(n)]


def test_a_nomem_correct_repair_gives_hidden_pass_and_verified_visible():
    fake = FakeProposer(ARMS["A_noMem"].model, [CORRECT])
    rec, execs, _result = attempt_task(ARMS["A_noMem"], U, SPEC, fake, ConstantValue())
    assert rec.hidden_pass is True
    assert rec.status == "verified_visible"
    assert rec.visible_reward == 1.0
    assert rec.arm == "A_noMem" and rec.unit_id == "X/0" and rec.phase == 2
    assert rec.tampered is False
    assert execs and execs[0].visible_reward == rec.visible_reward


def test_b_naive_is_single_shot_one_generate_no_refinement():
    fake = FakeProposer(ARMS["B_naive"].model, [CORRECT])
    rec, execs, _result = attempt_task(ARMS["B_naive"], U, SPEC, fake, ConstantValue())
    assert len(fake.calls) == 1          # exactly one generate call: no refinement
    assert fake.calls[0]["n"] == 1       # single candidate
    assert rec.executions_charged <= 1


def test_visible_pass_hidden_fail_is_not_scored_as_pass():
    # THE INVARIANT: hidden_pass comes from run_hidden, never the visible reward.
    fake = FakeProposer(ARMS["A_noMem"].model, [DISCRIMINATOR])
    rec, execs, _result = attempt_task(ARMS["A_noMem"], U, SPEC, fake, ConstantValue())
    assert rec.visible_reward == 1.0     # the submission DID pass every visible test
    assert rec.hidden_pass is False      # ... but the hidden oracle says it is wrong


def test_hidden_infra_error_leaves_hidden_pass_none(monkeypatch):
    # An infra failure on the HIDDEN run is "not measured" -> None, never False.
    infra = TestReport((), (), (), (), 0.0, "sandbox exploded")
    monkeypatch.setattr("crucible.run.arm.run_hidden", lambda *a, **k: infra)
    fake = FakeProposer(ARMS["A_noMem"].model, [CORRECT])
    rec, execs, _result = attempt_task(ARMS["A_noMem"], U, SPEC, fake, ConstantValue())
    assert rec.hidden_pass is None
    assert rec.infra_error == "sandbox exploded"


def test_arm_registry_matches_frozen_spec():
    # A2 + the activated §2 fallback (2026-08-24): every arm is a chat-served instruct
    # model now -- the small arms the 1.5B, the B arms the 14B coder (AWQ-served under
    # this SERVED name; the 9B failed §4.7 both ways, findings S2.5-stack2.md §6-§7).
    assert ARMS["A_noMem"] == ArmConfig("A_noMem", "Qwen/Qwen2.5-Coder-1.5B-Instruct", True, chat=True)
    assert ARMS["B_search"] == ArmConfig("B_search", "Qwen/Qwen2.5-Coder-14B-Instruct", True, chat=True)
    assert ARMS["B_naive"] == ArmConfig("B_naive", "Qwen/Qwen2.5-Coder-14B-Instruct", False, chat=True)
    assert ARMS["B_naive"].use_search is False


def test_ablation_arms_are_a_full_s_serving_identity():
    """The EXPLORATORY ablations (docs/findings/ABLATIONS-A.md) must reach the server as
    A_full does in every field -- same checkpoint, chat surface, budget, seed -- or a rate
    difference from A_full would be confounded by serving instead of isolating the ablated
    mechanism. Like A_full itself, the ablation lives in the hooks, never in ArmConfig."""
    for name in ("A_mem_nosleep", "A_sleep_nomem"):
        assert ARMS[name] == ArmConfig(name, ARMS["A_full"].model, True, chat=True)
        assert (ARMS[name].k, ARMS[name].width, ARMS[name].seed) == (
            ARMS["A_full"].k, ARMS["A_full"].width, ARMS["A_full"].seed)


def test_phase_b_arms_carry_their_controls_serving_identity():
    """Phase-B (prereg §3): B_mem must reach the server exactly as the frozen B_search
    did, and A_mem_exactonly exactly as A_full/A_mem_nosleep — or a rate difference is
    confounded by serving instead of isolating the store / the retrieval policy."""
    assert ARMS["B_mem"] == ArmConfig("B_mem", ARMS["B_search"].model, True, chat=True)
    assert (ARMS["B_mem"].k, ARMS["B_mem"].width, ARMS["B_mem"].seed) == (
        ARMS["B_search"].k, ARMS["B_search"].width, ARMS["B_search"].seed)
    assert ARMS["A_mem_exactonly"] == ArmConfig(
        "A_mem_exactonly", ARMS["A_full"].model, True, chat=True)


def test_phase_c_arms_carry_their_controls_serving_identity():
    """Phase-C (prereg §3): B_symmem must reach the server exactly as the frozen B_search
    did, and A_symmem exactly as A_full -- or a rate difference is confounded by serving
    instead of isolating the cross-unit symptom-similarity retrieval mechanism."""
    assert ARMS["B_symmem"] == ArmConfig("B_symmem", ARMS["B_search"].model, True, chat=True)
    assert (ARMS["B_symmem"].k, ARMS["B_symmem"].width, ARMS["B_symmem"].seed) == (
        ARMS["B_search"].k, ARMS["B_search"].width, ARMS["B_search"].seed)
    assert ARMS["A_symmem"] == ArmConfig("A_symmem", ARMS["A_full"].model, True, chat=True)
    assert (ARMS["A_symmem"].k, ARMS["A_symmem"].width, ARMS["A_symmem"].seed) == (
        ARMS["A_full"].k, ARMS["A_full"].width, ARMS["A_full"].seed)


def test_chat_serving_is_an_arm_property_not_a_cli_default():
    """Every gating proposer is an instruct model and MUST be chat-served (the §2 fallback
    replaced the raw-served base 9B). Binding chat to the arm stops `arm run` without
    --chat from silently serving the wrong surface."""
    assert ARMS["A_noMem"].chat is True
    assert ARMS["B_search"].chat is True and ARMS["B_naive"].chat is True


# --- S3: the memory seam through attempt_task --------------------------------
#
# ``attempt_task`` is the ONLY place the driver's retrieved block gets handed to an arm. If
# either hand-off is ever dropped, A_full silently runs memory-free and the headline A_full
# vs A_noMem comparison becomes a null by construction -- a failure that no assertion about
# search internals can catch, because search would still be doing everything right with the
# argument it was (not) given. These two tests pin both hand-offs at that seam.

MEM_BLOCK = ("## Prior experience with this code\n"
             "- ARITH: a prior repair changed `a - b` to `a + b` and passed its hidden suite.")


def test_attempt_task_hands_the_memory_block_to_the_search(monkeypatch):
    real_search = arm.search
    captured: list[dict] = []

    def spy_search(unit, proposer, value, **kw):
        captured.append(kw)
        return real_search(unit, proposer, value, **kw)

    monkeypatch.setattr(arm, "search", spy_search)
    fake = FakeProposer(ARMS["A_noMem"].model, [CORRECT])
    attempt_task(ARMS["A_noMem"], U, SPEC, fake, ConstantValue(), memory=MEM_BLOCK)
    assert MEM_BLOCK in fake.prompts[0]            # it really reached the model, and ...
    assert captured[0]["memory"] is MEM_BLOCK      # ... as the very object, not a rebuild


def test_attempt_task_defaults_the_search_to_no_memory(monkeypatch):
    # A_noMem's own call: the kwarg is present and None, so the arm is byte-for-byte its S2
    # self. A mutant that drops the kwarg fails on the KeyError, not on a None-vs-absent nicety.
    real_search = arm.search
    captured: list[dict] = []

    def spy_search(unit, proposer, value, **kw):
        captured.append(kw)
        return real_search(unit, proposer, value, **kw)

    monkeypatch.setattr(arm, "search", spy_search)
    fake = FakeProposer(ARMS["A_noMem"].model, [CORRECT])
    attempt_task(ARMS["A_noMem"], U, SPEC, fake, ConstantValue())
    assert captured[0]["memory"] is None
    assert all("Prior experience" not in p for p in fake.prompts)


def test_attempt_task_hands_the_memory_block_to_the_naive_control(monkeypatch):
    # The single-shot path (use_search=False). The spy wraps the REAL ``_naive_attempt``, so
    # the prompt, the root_prompt and the symptom_failed asserted here are the genuine ones.
    real_naive = arm._naive_attempt
    captured: dict = {}

    def spy_naive(cfg, unit, proposer, value, **kw):
        captured.update(kw)
        result = real_naive(cfg, unit, proposer, value, **kw)
        captured["result"] = result
        return result

    monkeypatch.setattr(arm, "_naive_attempt", spy_naive)
    fake = FakeProposer(ARMS["B_naive"].model, [CORRECT])
    attempt_task(ARMS["B_naive"], U, SPEC, fake, ConstantValue(), memory=MEM_BLOCK)

    assert len(fake.prompts) == 1                  # still single-shot: one prompt, no refinement
    assert MEM_BLOCK in fake.prompts[0]
    assert captured["memory"] is MEM_BLOCK
    result = captured["result"]
    assert result.root_prompt == fake.prompts[0]   # the prompt as SENT, not a reconstruction
    assert result.symptom_failed == ("test_v0",)   # the buggy module fails the one visible test


# --- the served-identity guard, both sides -----------------------------------------------
#
# S3 relaxed the guard: a proposer may serve the arm's model OR declare ``base_model ==
# cfg.model`` (an adapter ON the arm's own base -- vLLM routes a runtime LoRA by model name,
# so an arm running its own accepted adapter legitimately asks for ``adapter_id``). The
# relaxation must not become a hole: a proposer serving some OTHER checkpoint is still
# refused, and the ``base_model`` declaration itself is checked where it is minted
# (``AdapterProposer.__init__``, pinned in tests/run/test_full.py).

def test_attempt_task_refuses_a_proposer_serving_a_foreign_checkpoint():
    # No ``base_model`` attribute at all: the plain S2 case, and the guard's own raise.
    fake = FakeProposer("Qwen/Qwen3.5-9B", [CORRECT])
    with pytest.raises(ValueError) as e:
        attempt_task(ARMS["A_full"], U, SPEC, fake, ConstantValue())
    assert "expects model" in str(e.value) and "Qwen/Qwen3.5-9B" in str(e.value)


def test_attempt_task_refuses_an_adapter_declared_on_a_DIFFERENT_base():
    # Declaring the wrong base is not a licence: the guard compares the declaration to THIS
    # arm's model, so an adapter on the 9B cannot serve the 1.5B arm.
    fake = FakeProposer("ad-0123456789abcdef", [CORRECT])
    fake.base_model = "Qwen/Qwen3.5-9B"
    with pytest.raises(ValueError):
        attempt_task(ARMS["A_full"], U, SPEC, fake, ConstantValue())


def test_attempt_task_accepts_an_adapter_on_the_arms_own_base():
    fake = FakeProposer("ad-0123456789abcdef", [CORRECT])
    fake.base_model = ARMS["A_full"].model
    rec, _execs, _result = attempt_task(ARMS["A_full"], U, SPEC, fake, ConstantValue())
    assert rec.hidden_pass is True
