import gzip
import json
import pathlib

from crucible.stream import build as build_mod
from crucible.stream.build import Dropped, build_unit, build_units
from crucible.stream.oracle import OracleError
from crucible.stream.units import Unit

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"


def _recs(name):
    with gzip.open(FIX / name, "rt") as fh:
        return [json.loads(l) for l in fh]


def test_build_unit_humaneval_is_deterministic_and_self_checked():
    rec = _recs("mini_humaneval.jsonl.gz")[0]
    u1 = build_unit(rec, seed=0, max_hidden=2)
    u2 = build_unit(rec, seed=0, max_hidden=2)
    assert isinstance(u1, Unit) and u1 == u2
    assert u1.module_name == "unit_humaneval_0" and u1.n_visible == 4 and u1.n_hidden == 2
    assert '"""' not in u1.module_src and "return a + b" in u1.module_src


def test_build_unit_mbpp_strips_prompt_docstring():
    rec = _recs("mini_mbpp.jsonl.gz")[0]
    u = build_unit(rec, seed=0)
    assert isinstance(u, Unit) and "Return first element" not in u.module_src and u.n_visible == 2


def test_hidden_inputs_are_a_seeded_sample_not_the_first_n():
    # plus_input is [[10, 20], [5, 5], [100, -1], [3, 4]]; the seeded sample for
    # seed 0 / HumanEval/0 / max_hidden 2 is indices [0, 3]. Taking "the first N"
    # instead would render [5, 5] and never [3, 4] -- so this pins the sampling
    # rule itself, which a count-only assertion cannot do.
    rec = _recs("mini_humaneval.jsonl.gz")[0]
    u = build_unit(rec, seed=0, max_hidden=2)
    assert isinstance(u, Unit) and u.n_hidden == 2
    assert "[10, 20]" in u.hidden_test_src and "[3, 4]" in u.hidden_test_src
    assert "[5, 5]" not in u.hidden_test_src


def test_nondeterministic_canonical_is_dropped_with_reason():
    # The oracle derives expected values FROM the canonical, so a merely-wrong canonical is self-consistent
    # and cannot be detected here. What the self-check catches is a canonical whose outputs differ between
    # the oracle process and the pytest process: pid-dependent output does exactly that, deterministically.
    rec = dict(_recs("mini_humaneval.jsonl.gz")[0]); rec["canonical_solution"] = "    import os\n    return os.getpid()\n"
    d = build_unit(rec, seed=0)
    assert isinstance(d, Dropped) and d.reason.startswith("canonical-fails-visible")


def test_oracle_failure_is_dropped_not_raised(monkeypatch):
    # An OracleError escaping build_unit would abort the whole ThreadPoolExecutor map
    # in build_units, losing every other record's work (ruling R-T8-1).
    def boom(*a, **k):
        raise OracleError("unit_humaneval_0", 1, False, "ModuleNotFoundError: no_such_module",
                          "driver did not exit cleanly")

    monkeypatch.setattr(build_mod, "compute_expected", boom)
    d = build_unit(_recs("mini_humaneval.jsonl.gz")[0], seed=0)
    assert isinstance(d, Dropped) and d.reason == "oracle-error:driver did not exit cleanly"


def test_render_failure_is_dropped_not_raised(monkeypatch):
    def boom(*a, **k):
        raise ValueError("render_tests: 4 inputs but 3 expectations")

    monkeypatch.setattr(build_mod, "render_tests", boom)
    d = build_unit(_recs("mini_humaneval.jsonl.gz")[0], seed=0)
    assert isinstance(d, Dropped) and d.reason.startswith("render-error:")


def test_build_units_partitions_units_and_dropped():
    recs = _recs("mini_humaneval.jsonl.gz") + _recs("mini_mbpp.jsonl.gz")
    bad = dict(recs[0]); bad["task_id"] = "HumanEval/999"; bad["canonical_solution"] = "    return a -\n"
    units, dropped = build_units(recs + [bad], seed=0, jobs=2)
    assert {u.unit_id for u in units} == {"HumanEval/0", "HumanEval/1", "Mbpp/2"}
    assert [d.unit_id for d in dropped] == ["HumanEval/999"]


def test_build_units_preserves_record_order_and_reports_progress():
    # Record order, not thread-completion order and not sorted order: downstream
    # tasks pair these lists back against the records they came from.
    recs = list(reversed(_recs("mini_humaneval.jsonl.gz") + _recs("mini_mbpp.jsonl.gz")))
    bad = []
    for tid in ("HumanEval/998", "HumanEval/997"):
        b = dict(recs[-1]); b["task_id"] = tid; b["canonical_solution"] = "    return a -\n"
        bad.append(b)
    ordered = [bad[0], *recs, bad[1]]
    seen = []
    units, dropped = build_units(ordered, seed=0, jobs=3, progress=seen.append)
    assert [u.unit_id for u in units] == ["Mbpp/2", "HumanEval/1", "HumanEval/0"]
    assert [d.unit_id for d in dropped] == ["HumanEval/998", "HumanEval/997"]
    assert [r.unit_id for r in seen] == ["HumanEval/998", "Mbpp/2", "HumanEval/1",
                                         "HumanEval/0", "HumanEval/997"]
