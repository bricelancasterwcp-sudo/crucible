import warnings

from crucible.proposer.codec import extract_module, landing_rate


def test_extracts_fenced_module_and_lands():
    t = "Here is the fix:\n```python\ndef add(a, b):\n    return a + b\n```\n"
    L = extract_module(t); assert L.ok and "return a + b" in L.module_src


def test_takes_last_fence_when_multiple():
    t = "```python\nWRONG\n```\nnow correct:\n```python\ndef f():\n    return 2\n```"
    assert extract_module(t).module_src.strip() == "def f():\n    return 2"


def test_unparseable_does_not_land():
    assert not extract_module("```python\ndef f(:\n```").ok


def test_no_fence_falls_back_to_whole_text_if_it_parses():
    assert extract_module("def f():\n    return 1\n").ok


def test_landing_rate():
    assert landing_rate(["```python\ndef f():\n    return 1\n```", "prose only, no code {"]) == 0.5


def test_syntax_warning_does_not_break_landing_under_werror():
    # R-T11-1: a module that emits `SyntaxWarning` at compile time (`is` with a str
    # literal) must still LAND even when the process runs under `-W error::SyntaxWarning`.
    # We simulate that hostile filter here so the assertion holds regardless of how the
    # suite is invoked; the codec's own suppression must neutralise it.
    t = "```python\ndef f():\n    return x is 'a'\n```"
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        L = extract_module(t)
    assert L.ok and L.reason is None
