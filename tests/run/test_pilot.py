"""Task 14 -- the ceiling pilot: run A_noMem over the phase-1 tasks and read off p0.

``p0`` is A_noMem's success rate on the phase-1 stream (spec §4.8.4). The pilot's whole job
is to answer one question before the arms are measured: is the stream *too easy* for the
frozen proposer -- ``p0 > 0.70``, STRICT -- in which case the ceiling is undiscriminating and
the stream must be hardened first. Three things are pinned here:

* ``p0`` is ``build_lens(...).succ_overall`` -- the honest measured-only success fraction --
  never ``landing_rate`` or any other rate. The 0.5 case makes ``landing_rate`` (1.0) differ
  from ``succ_overall`` (0.5), so reading p0 off the wrong field is caught.
* BOTH sides of the boundary run through the REAL pipeline: p0 0.5 -> ``too_easy False`` /
  "proceed"; p0 1.0 -> ``too_easy True`` + a hardening-ladder recommendation.
* The EXACT boundary is strict: ``_verdict(0.70)`` is "proceed", ``_verdict(>0.70)`` is too
  easy -- so flipping the threshold ``>`` to ``>=`` bites ``test_verdict_pins_the_strict_070_boundary``.

Run WRAPPED (R-T2-6): ``ceiling_pilot`` builds a stream and runs A_noMem through the sandbox
(``search`` + ``run_hidden``). The two ``_verdict``/``to_dict`` unit assertions touch no
sandbox but ride along in the wrapped file.
"""
import gzip
import json
import pathlib

import pytest

from crucible.run.pilot import (PROCEED, TOO_EASY_THRESHOLD, PilotVerdict, _verdict,
                                ceiling_pilot)
from crucible.run.types import Candidate
from crucible.stream import store
from crucible.stream.pipeline import BuildConfig, build_stream
from crucible.value.model import ConstantValue

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
PILOT_MODEL = "Qwen/Qwen3.5-2B"          # == ARMS["A_noMem"].model; the proposer must serve it


def _recs():
    out = []
    for n in ("mini_humaneval.jsonl.gz", "mini_mbpp.jsonl.gz"):
        with gzip.open(FIX / n, "rt") as fh:
            out += [json.loads(line) for line in fh]
    return out


@pytest.fixture(scope="module")
def stream(tmp_path_factory):
    """Build the 3-record fixture stream once (n_nov=0 -> 2 classes -> 4 tasks, 2 phase-1)."""
    root = tmp_path_factory.mktemp("stream")
    cfg = BuildConfig(seed=0, C=2, n_nov=0, per_family=3, max_hidden=2, jobs=2)
    return build_stream(cfg, root, recs=_recs(), log=lambda *a: None)


class ScriptedProposer:
    """In-process proposer serving A_noMem's model.

    Per task it returns either the CANONICAL module (a hidden-passing repair) or the buggy
    mutant itself (a hidden fail), keyed off the mutated source the repair prompt embeds --
    so the pilot's success fraction is exactly the fraction of tasks scripted to pass.
    """

    def __init__(self, model, repairs):
        self.model = model
        self._repairs = repairs           # list[(needle, output_src)]
        self.calls = []

    def generate(self, prompt, *, n, seed, max_tokens=1024, temperature=0.7):
        self.calls.append({"n": n, "seed": seed})
        for needle, output in self._repairs:
            if needle in prompt:
                return [Candidate(output, None, 1.0) for _ in range(n)]
        raise AssertionError("no scripted repair matched the prompt")


def _phase1(stream):
    man = store.read_manifest(stream)
    return [t for t in man.tasks if t.kind == "first"]


def _repairs(stream, pass_keys):
    """One ``(needle, output)`` per phase-1 task: the canonical module to pass the hidden
    suite, the mutant to fail it. ``needle`` is the mutated source the prompt embeds."""
    repairs = []
    for t in _phase1(stream):
        mutant = store.read_mutant(stream, t.task_key)
        canonical = store.read_unit(stream, t.unit_id).module_src
        output = canonical if t.task_key in pass_keys else mutant.mutated_src
        repairs.append((mutant.mutated_src.rstrip("\n"), output))
    return repairs


def test_half_success_is_not_too_easy(stream, tmp_path):
    # One of two phase-1 tasks scripted to pass -> p0 = 0.5. Reading p0 off landing_rate
    # (1.0 here) instead of succ_overall would fail this p0 assertion (pins that mutation).
    phase1 = _phase1(stream)
    assert len(phase1) == 2
    fake = ScriptedProposer(PILOT_MODEL, _repairs(stream, {phase1[0].task_key}))

    verdict = ceiling_pilot(stream, tmp_path / "half", fake, ConstantValue(),
                            n=len(phase1), seed=0, log=lambda *a: None)

    assert verdict.p0 == pytest.approx(0.5)
    assert verdict.n == 2
    assert verdict.too_easy is False
    assert verdict.recommendation == PROCEED


def test_high_success_is_too_easy_and_recommends_hardening(stream, tmp_path):
    # Both phase-1 tasks scripted to pass -> p0 = 1.0 > 0.70 -> too easy.
    phase1 = _phase1(stream)
    fake = ScriptedProposer(PILOT_MODEL, _repairs(stream, {t.task_key for t in phase1}))

    verdict = ceiling_pilot(stream, tmp_path / "high", fake, ConstantValue(),
                            n=len(phase1), seed=0, log=lambda *a: None)

    assert verdict.p0 == pytest.approx(1.0)
    assert verdict.too_easy is True
    assert verdict.recommendation != PROCEED
    assert "harden" in verdict.recommendation.lower()


def test_verdict_pins_the_strict_070_boundary():
    # spec §4.8.4: too_easy iff p0 STRICTLY > 0.70. Exactly 0.70 is "proceed", not too easy.
    at = _verdict(TOO_EASY_THRESHOLD, 10)
    assert at.too_easy is False and at.recommendation == PROCEED   # >= mutation flips this
    just_over = _verdict(TOO_EASY_THRESHOLD + 0.01, 10)
    assert just_over.too_easy is True and just_over.recommendation != PROCEED


def test_pilot_verdict_to_dict_is_json_native():
    v = PilotVerdict(p0=0.5, n=2, too_easy=False, recommendation=PROCEED)
    assert v.to_dict() == {"p0": 0.5, "n": 2, "too_easy": False, "recommendation": "proceed"}
    json.dumps(v.to_dict())          # must be JSON-serializable for the CLI
