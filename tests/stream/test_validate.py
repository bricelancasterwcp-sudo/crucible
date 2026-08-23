"""Tests for mutant validation in the sandbox.

These run the real sandbox: every ``validate_mutant`` call here is one or two pytest
subprocesses. The monkeypatched cases count calls rather than fake reports wherever the
count is the point -- what must not happen (a second run on an empty hidden file, any run
at all for a non-compiling mutant) is only visible in the call count.
"""

from dataclasses import replace

from crucible.sandbox.report import TestReport
from crucible.stream import validate as validate_mod
from crucible.stream.mutants import enumerate_specs, make_mutant
from crucible.stream.units import Unit, sha256_text
from crucible.stream.validate import Validation, validate_many, validate_mutant

SRC = "def f(a, b):\n    return a + b\n"
VIS = "from unit_x import f as candidate\ndef test_v0():\n    assert candidate(1, 2) == 3\n"
HID = "from unit_x import f as candidate\ndef test_h0():\n    assert candidate(0, 0) == 0\n"
U = Unit("X/0", "unit_x", "f", SRC, VIS, HID, sha256_text(SRC), 1, 1, ())
# Task 8 emits units with no hidden tests at all; their hidden file is the empty string.
U_NO_HIDDEN = Unit("X/1", "unit_x", "f", SRC, VIS, "", sha256_text(SRC), 1, 0, ())


def _mut(op):
    spec = enumerate_specs(SRC, [op])[0]
    return make_mutant(U, spec)


def _count_runs(monkeypatch) -> list[str]:
    """Wrap the real ``run_tests`` in a call counter, in the namespace ``validate`` uses.

    Returns the list of test sources passed, so the *which* as well as the *how many* is
    inspectable.
    """
    real = validate_mod.run_tests
    calls: list[str] = []

    def counted(module_name, module_src, test_src, **kw):
        calls.append(test_src)
        return real(module_name, module_src, test_src, **kw)

    monkeypatch.setattr(validate_mod, "run_tests", counted)
    return calls


def test_killed_by_visible_is_valid():
    v = validate_mutant(U, _mut("ReplaceBinaryOperator_Add_Sub"))
    assert v.valid and v.reason == "killed-visible" and v.n_killing_visible == 1 and not v.kills_by_timeout


def test_hidden_only_kill_is_not_valid():
    # Passes visible (1,2)->3 because a is truthy; fails hidden (0,0)->0 because it is not.
    m = _mut("ReplaceBinaryOperator_Add_Sub")
    m2 = replace(m, mutated_src="def f(a, b):\n    return a + b if a else 1\n", key="k2")
    v = validate_mutant(U, m2)
    assert not v.valid and v.reason == "hidden-only"


def test_equivalent_is_not_valid():
    m = _mut("ReplaceBinaryOperator_Add_Sub")
    m3 = replace(m, mutated_src="def f(a, b):\n    return b + a\n", key="k3")
    assert validate_mutant(U, m3).reason == "equivalent"


def test_hang_is_valid_and_flagged():
    m = _mut("ReplaceBinaryOperator_Add_Sub")
    m4 = replace(m, mutated_src="def f(a, b):\n    while True: pass\n", key="k4")
    v = validate_mutant(U, m4, per_test_timeout_s=1.0)
    assert v.valid and v.kills_by_timeout


def test_no_hidden_suite_is_equivalent_and_never_runs_the_empty_file(monkeypatch):
    # A unit with n_hidden == 0 has an empty hidden file: running it would be an infra
    # error, not a measurement, so the surviving mutant must be labelled from the visible
    # run alone -- exactly one sandbox run.
    calls = _count_runs(monkeypatch)
    m = replace(_mut("ReplaceBinaryOperator_Add_Sub"), mutated_src="def f(a, b):\n    return b + a\n", key="k5")
    v = validate_mutant(U_NO_HIDDEN, m)
    assert not v.valid and v.reason == "equivalent"
    assert calls == [VIS]


def test_non_compiling_mutant_is_syntax_and_runs_nothing(monkeypatch):
    calls = _count_runs(monkeypatch)
    m = replace(_mut("ReplaceBinaryOperator_Add_Sub"), mutated_src="def f(a, b)\n    return a + b\n", key="k6")
    v = validate_mutant(U, m)
    assert not v.valid and v.reason == "syntax" and v.n_killing_visible == 0 and v.visible_failed == ()
    assert calls == []


def test_infra_error_is_not_valid_and_not_a_kill(monkeypatch):
    monkeypatch.setattr(validate_mod, "run_tests",
                        lambda *a, **k: TestReport((), (), (), (), 0.1, "test file does not collect"))
    v = validate_mutant(U, _mut("ReplaceBinaryOperator_Add_Sub"))
    assert not v.valid and v.reason == "infra" and v.n_killing_visible == 0 and not v.kills_by_timeout


def test_validate_many_preserves_order_and_round_trips():
    ms = [_mut("ReplaceBinaryOperator_Add_Sub"), _mut("ReplaceBinaryOperator_Add_Mul")]
    vs = validate_many(U, ms, jobs=2)
    assert [v.mutant_key for v in vs] == [m.key for m in ms]
    assert Validation.from_dict(vs[0].to_dict()) == vs[0]
    assert vs[0].visible_failed == ("test_v0",)
