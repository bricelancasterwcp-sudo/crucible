import pytest
from crucible.sandbox.budget import BudgetMeter, BudgetExhausted
from crucible.sandbox.report import TestReport

OK = TestReport(("t",), (), (), (), 0.1, None)
INFRA = TestReport((), (), (), (), 0.1, "server down")


def test_ninth_execution_is_refused():
    m = BudgetMeter(k=8)
    for _ in range(8):
        m.check(); m.charge(OK)
    assert m.remaining() == 0
    with pytest.raises(BudgetExhausted):
        m.check()


def test_infra_is_counted_but_not_charged():
    m = BudgetMeter(k=2)
    m.charge(INFRA); m.charge(INFRA); m.charge(INFRA)
    assert m.charged == 0 and m.infra == 3 and m.remaining() == 2
    m.check()  # still allowed


def test_to_dict_complete():
    assert set(BudgetMeter(3).to_dict()) == {"k", "charged", "infra"}
