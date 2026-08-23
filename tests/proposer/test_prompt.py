from crucible.proposer.prompt import build_prompt
from crucible.sandbox.runner import TestReport
from crucible.stream.units import Unit, sha256_text

SRC = "def add(a, b):\n    return a - b\n"  # mutated (bug)
U = Unit("X/0","unit_x","add",SRC,"from unit_x import add as candidate\ndef test_v0():\n    assert candidate(1,2)==3\n","h",sha256_text(SRC),1,0,())
SYM = TestReport((), ("test_v0",), (), (), 0.1, None)

def test_prompt_contains_source_tests_symptom_and_codec_instruction():
    p = build_prompt(U, SYM)
    assert "def add(a, b):" in p and "test_v0" in p
    assert "complete" in p.lower() and "```python" in p
    assert "hidden" not in p.lower()      # never leak the notion of hidden tests as runnable

def test_feedback_is_included_on_refinement():
    p = build_prompt(U, SYM, feedback="attempt 1 still failed test_v0")
    assert "attempt 1 still failed test_v0" in p
