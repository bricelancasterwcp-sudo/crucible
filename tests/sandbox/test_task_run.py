from crucible.sandbox.task_run import run, run_hidden
from crucible.stream.units import Unit, sha256_text

SRC = "def add(a, b):\n    return a + b\n"
VIS = "from unit_x import add as candidate\nimport math\nATOL = 1e-06\ndef _eq(a, b):\n    return a == b\ndef test_v0():\n    assert _eq(candidate(1, 2), 3)\n"
HID = "from unit_x import add as candidate\nimport math\nATOL = 1e-06\ndef _eq(a, b):\n    return a == b\ndef test_h0():\n    assert _eq(candidate(0, 0), 0)\n"
U = Unit("X/0", "unit_x", "add", SRC, VIS, HID, sha256_text(SRC), 1, 1, ())

def test_run_executes_visible_only():
    r = run(U, SRC, None)
    assert r.all_passed and r.infra_error is None

def test_run_reports_failure_for_a_bad_patch():
    r = run(U, "def add(a, b):\n    return a - b\n", None)
    assert not r.all_passed and "test_v0" in r.failed

def test_run_hidden_is_the_outcome_oracle():
    # The discriminating patch passes the VISIBLE suite (add(1, 2) == 3) but fails the
    # HIDDEN suite (add(0, 0) == 0 -> returns 1): only a correctly-wired run_hidden that
    # actually reaches the hidden tests can report this failure. (`a - b` would fail the
    # visible suite too -- add(0, 0) is symmetric under +/- so it can't fail hidden alone.)
    assert run_hidden(U, SRC).all_passed
    assert not run_hidden(U, "def add(a, b):\n    return a + b if a else 1\n").all_passed
