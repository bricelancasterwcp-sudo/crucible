"""Tests for the StatementDeletion (SDL) mutation operator."""

from cosmic_ray.ast import ast_nodes, get_ast
from cosmic_ray.mutating import mutate_code

from crucible.stream.sdl import StatementDeletion

SRC = "import os\n\ndef f(x):\n    y = x + 1\n    z = y * 2\n    return z\n"


def _positions(src):
    op = StatementDeletion()
    return [p for n in ast_nodes(get_ast(src)) for p in op.mutation_positions(n)]


def test_positions_are_only_suite_statements():
    pos = _positions(SRC)
    # lines 4, 5, 6 -- the module-level import is not a candidate.
    assert len(pos) == 3
    assert pos[0][0][0] == 4
    assert pos[2][0][0] == 6


def test_mutate_replaces_with_pass_keeping_indent_and_compiles():
    out = mutate_code(SRC, StatementDeletion(), 0)
    assert out == "import os\n\ndef f(x):\n    pass\n    z = y * 2\n    return z\n"
    compile(out, "m", "exec")
    out2 = mutate_code(SRC, StatementDeletion(), 2)
    assert out2.endswith("    z = y * 2\n    pass\n")
    compile(out2, "m", "exec")


def test_skips_docstring_and_pass():
    src = 'def g():\n    """doc"""\n    pass\n'
    assert _positions(src) == []


def test_occurrence_out_of_range_returns_none():
    assert mutate_code(SRC, StatementDeletion(), 99) is None
