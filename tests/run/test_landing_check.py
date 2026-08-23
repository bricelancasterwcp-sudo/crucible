"""Task 15 -- the codec-landing pre-check (spec §4.7): can a served model be trusted?

Before an arm draws real repair attempts from a served model, §4.7 asks one cheaper
question: does the model reliably answer in the full-module-rewrite codec at all? Draw
``n`` smoke tasks, prompt the model once each, and measure the *landing rate* -- the
fraction whose output parses as a Python module. A model that clears the 0.95 gate is
trusted; one that does not means the operator falls back to the §2 baseline proposer
(recorded by hand, not automated here).

Two properties are pinned, each with its own mutation:

* (a) *The rate is the LANDING rate of the actual outputs.* A fake proposer that lands
  exactly half its outputs must yield ``rate == 0.5`` -- computing the rate from anything
  other than the landing check (e.g. hard-coding 1.0) breaks
  ``test_landing_precheck_measures_the_fractional_rate``. This rides through the real
  pipeline (a free symptom ``run`` per drawn task), so it touches the sandbox.
* (b) *The 0.95 gate is INCLUSIVE (§4.7 says >= 0.95).* ``_verdict(0.95, ...)`` passes;
  flipping the gate to strict ``>`` makes ``test_verdict_pins_the_095_boundary`` fail.
  These two ``_verdict``/``to_dict`` assertions touch no sandbox but ride along in this
  wrapped file.

Run WRAPPED (R-T2-6): ``landing_precheck`` builds a per-task mutated unit and runs a free
symptom execution through the sandbox for every drawn task.
"""
import gzip
import json
import pathlib

import pytest

from crucible.run.landing_check import (LANDING_GATE, LandingResult, _verdict,
                                        landing_precheck)
from crucible.run.types import Candidate
from crucible.stream import store
from crucible.stream.pipeline import BuildConfig, build_stream

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
FAKE_MODEL = "fake/model"

# A parseable module lands; a bare ``)(`` is a syntax error and does not.
LANDS = "def f():\n    return 1\n"
GARBAGE = "def broken(:\n"


def _recs():
    out = []
    for n in ("mini_humaneval.jsonl.gz", "mini_mbpp.jsonl.gz"):
        with gzip.open(FIX / n, "rt") as fh:
            out += [json.loads(line) for line in fh]
    return out


@pytest.fixture(scope="module")
def stream(tmp_path_factory):
    """Build the 3-record fixture stream once (n_nov=0 -> 2 classes -> 4 tasks)."""
    root = tmp_path_factory.mktemp("stream")
    cfg = BuildConfig(seed=0, C=2, n_nov=0, per_family=3, max_hidden=2, jobs=2)
    return build_stream(cfg, root, recs=_recs(), log=lambda *a: None)


class FractionProposer:
    """In-process proposer: lands its first ``k`` generate calls, garbles the rest.

    Each call returns one already-codec-decoded module source (as the real proposer
    does): a parseable module for the first ``k`` calls, an unparseable stub after -- so
    the measured landing rate is exactly ``k / n``. ``calls`` counts every generate.
    """

    def __init__(self, model, k):
        self.model = model
        self._k = k
        self.calls = 0

    def generate(self, prompt, *, n, seed, max_tokens=1024, temperature=0.7):
        src = LANDS if self.calls < self._k else GARBAGE
        self.calls += 1
        return [Candidate(src, None, 1.0) for _ in range(n)]


def test_landing_precheck_measures_the_fractional_rate(stream, tmp_path):
    # 2 of 4 smoke outputs parse -> rate 0.5. Reading the rate off anything but the
    # landing check (e.g. a hard-coded 1.0) fails this assertion (pins that mutation).
    fake = FractionProposer(FAKE_MODEL, k=2)

    result = landing_precheck(stream, fake, n=4, seed=0)

    assert result.rate == pytest.approx(0.5)
    assert result.n == 4
    assert result.model == FAKE_MODEL
    assert result.passes is False                 # 0.5 < 0.95
    assert fake.calls == 4                         # one generate per drawn smoke task


def test_full_landing_passes_the_gate(stream, tmp_path):
    # Every smoke output parses -> rate 1.0 >= 0.95 -> trusted.
    fake = FractionProposer(FAKE_MODEL, k=4)

    result = landing_precheck(stream, fake, n=4, seed=0)

    assert result.rate == pytest.approx(1.0)
    assert result.passes is True


def test_verdict_pins_the_095_boundary():
    # §4.7: passes iff rate >= 0.95, INCLUSIVE. Exactly 0.95 passes; flipping the gate to
    # a strict ``>`` bites here (0.95 would then read as a fail).
    at = _verdict(LANDING_GATE, 20, FAKE_MODEL)
    assert at.rate == LANDING_GATE and at.passes is True
    assert _verdict(0.96, 25, FAKE_MODEL).passes is True
    assert _verdict(0.94, 20, FAKE_MODEL).passes is False
    assert _verdict(0.90, 10, FAKE_MODEL).passes is False


def test_landing_result_to_dict_is_json_native():
    r = LandingResult(model=FAKE_MODEL, n=30, rate=0.95, passes=True)
    assert r.to_dict() == {"model": FAKE_MODEL, "n": 30, "rate": 0.95, "passes": True}
    json.dumps(r.to_dict())        # must be JSON-serializable for the CLI/record
