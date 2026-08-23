import ast
from crucible.stream.units import Unit, module_name_for, strip_docstrings, sha256_text

def test_module_name_for():
    assert module_name_for("HumanEval/0") == "unit_humaneval_0"
    assert module_name_for("Mbpp/2") == "unit_mbpp_2"

def test_strip_docstrings_removes_all_three_kinds_and_keeps_behaviour():
    src = '"""mod doc"""\nclass C:\n    """cdoc"""\n    def m(self):\n        """mdoc"""\n        return 1\n\ndef f(x):\n    """fdoc"""\n    return x + 1\n'
    out = strip_docstrings(src)
    assert "doc" not in out
    ns = {}; exec(out, ns)
    assert ns["f"](1) == 2 and ns["C"]().m() == 1

def test_strip_docstrings_keeps_function_with_only_docstring_valid():
    assert ast.parse(strip_docstrings('def g():\n    """only"""\n'))

def test_unit_round_trip():
    u = Unit("HumanEval/0", "unit_humaneval_0", "f", "def f():\n    return 1\n", "t", "h",
             sha256_text("def f():\n    return 1\n"), 1, 1, (("h0", "repr too long"),))
    assert Unit.from_dict(u.to_dict()) == u
    assert set(u.to_dict()) == {"unit_id","module_name","entry_point","module_src","visible_test_src",
                                "hidden_test_src","src_hash","n_visible","n_hidden","dropped_inputs"}

def test_sha256_text_is_the_sha256_of_the_utf8_text():
    # Literal digest, not recomputed: pins sha256_text to the content hash it claims to be.
    assert sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
