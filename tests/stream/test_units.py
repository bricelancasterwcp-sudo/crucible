import ast
import asyncio
from crucible.stream.units import Unit, module_name_for, strip_docstrings, sha256_text

def test_module_name_for():
    assert module_name_for("HumanEval/0") == "unit_humaneval_0"
    assert module_name_for("Mbpp/2") == "unit_mbpp_2"

def test_strip_docstrings_removes_all_three_kinds_and_keeps_behaviour():
    src = ('"""mod doc"""\nclass C:\n    """cdoc"""\n    def m(self):\n        """mdoc"""\n        return 1\n\n'
           'def f(x):\n    """fdoc"""\n    return x + 1\n\n'
           'async def a():\n    """adoc"""\n    return 3\n')
    out = strip_docstrings(src)
    assert "doc" not in out
    ns = {}; exec(out, ns)
    assert ns["f"](1) == 2 and ns["C"]().m() == 1
    assert asyncio.run(ns["a"]()) == 3

def test_strip_docstrings_keeps_function_with_only_docstring_valid():
    assert ast.parse(strip_docstrings('def g():\n    """only"""\n'))

def test_strip_docstrings_removes_a_run_of_leading_strings_and_is_idempotent():
    # One pass promotes the second literal into a docstring; src_hash keys on this output,
    # so the function must reach its fixed point in a single call.
    out = strip_docstrings('def f():\n    """d"""\n    "second"\n    return 1\n')
    assert "second" not in out
    assert strip_docstrings(out) == out

def test_unit_round_trip():
    # Distinguishable values per field: the field ORDER is a frozen interface (T7/T8/T11-T14
    # construct Unit positionally), and the key-set assertion below is order-blind.
    src = "def f():\n    return 1\n"
    u = Unit("HumanEval/0", "unit_humaneval_0", "f", src, "vis", "hid",
             sha256_text(src), 2, 3, (("h0", "repr too long"),))
    r = Unit.from_dict(u.to_dict())
    assert r == u
    assert r.visible_test_src == "vis" and r.hidden_test_src == "hid"
    assert r.n_visible == 2 and r.n_hidden == 3
    assert set(u.to_dict()) == {"unit_id","module_name","entry_point","module_src","visible_test_src",
                                "hidden_test_src","src_hash","n_visible","n_hidden","dropped_inputs"}

def test_sha256_text_is_the_sha256_of_the_utf8_text():
    # Literal digest, not recomputed: pins sha256_text to the content hash it claims to be.
    assert sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
