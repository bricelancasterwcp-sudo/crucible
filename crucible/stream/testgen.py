"""Render oracle results into a pytest file the sandbox runner can execute.

The file this module writes is read three times: by pytest's collector with the unit
stubbed out (the runner's probe), by pytest with the canonical solution in place (the
self-check), and by pytest once per mutant. Its shape is constrained by the first of
those.

*It must collect with a stub unit.* ``runner.run_tests`` first runs ``--collect-only``
against the file with the unit replaced by a PEP 562 ``__getattr__`` module; a file that
does not collect there is an ``infra_error``, never a kill. So: a plain import, plain
``def test_...()`` functions, and expected values inlined as literals. No parametrize over
unit-derived data, no fixtures, no star imports, no module-level calls into the unit --
each of those runs during collection, where the unit does not exist yet.

*Expected values are literals, not calls.* Every value came from the oracle and already
survived ``eval(repr(v)) == v`` inside the sandbox, so writing it in-line is exact. A test
never recomputes the expectation, which is what keeps a mutant from being judged against
itself. The controller never evaluates a ``value_repr`` at all -- it is copied verbatim
into the file and only the sandbox ever runs it.

*Arguments are literals too, and are checked here.* The oracle vets the value it returns
but not the input it was given, and both are rendered into the same file. An input whose
repr is not a literal is dropped with the ``no-roundtrip`` reason rather than emitted as a
test that would fail for every unit alike (ruling R-T7-2).

*Floats compare structurally, at any depth (ruling R-T7-4).* Each file defines its own
``_eq``: bools by ``==``, numbers by ``math.isclose(rel_tol=1e-7, abs_tol=ATOL)``, and
lists, tuples and dicts element-wise through the same rule. This mirrors EvalPlus's own
evaluator (recursive ``is_floats`` + ``allclose``, default ``atol`` 1e-6 when the problem
ships none). ``pytest.approx`` cannot do this job: it raises ``TypeError: pytest.approx()
does not support nested data structures``, so a nested float would have been compared
exactly and a last-bit difference would have counted as a kill -- silently, because the
canonical self-check reproduces its own bits.

Rendering is a pure function of its arguments: the same inputs and expectations always
produce byte-identical text, so a unit's test source is stable across runs and hashable.
"""
from __future__ import annotations

import ast

from .oracle import Expected

# EvalPlus's default absolute tolerance for problems that ship no atol of their own.
_DEFAULT_ATOL = 1e-6

_PRELUDE = "import math\nfrom {module} import {entry} as candidate\n\nATOL = {atol}\n\n\n"

# Rendered verbatim into every generated file. Kept as text, not built from the local
# ``_eq``, so what the tests compare with is exactly what is written down here.
_COMPARATOR = '''def _eq(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and (isinstance(a, float) or isinstance(b, float)):
        return math.isclose(a, b, rel_tol=1e-7, abs_tol=ATOL)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    return a == b


'''


def _round_trips(value: object) -> bool:
    """True when ``repr(value)`` reconstructs an equal value from literals alone.

    Applied to the *arguments* (ruling R-T7-2). They are rendered into the test file as a
    literal, so an input containing ``inf`` or ``nan`` -- ``Mbpp/404`` has 12 such rows --
    emits a bare name the file never binds. That raises ``NameError`` inside the test
    body, which pytest records as a *failure*: the unit gets blamed for something only the
    renderer did. Dropping the input with a reason is the honest answer.

    ``ast.literal_eval``, never bare ``eval`` (ruling R-T7-3): this runs in the controller
    process, on text a unit's ``__repr__`` produced inside the sandbox. It is stricter than
    the generated file needs -- a repr that calls a builtin (``range(0, 3)``) would in fact
    evaluate there -- and that conservatism is the point: an unrenderable input costs one
    named drop, an evaluated one costs the controller.
    """
    try:
        return ast.literal_eval(repr(value)) == value
    except Exception:
        return False


def _check_alignment(inputs: list[list], expected: list[Expected]) -> None:
    """Refuse expectations that do not line up one-to-one with the inputs (ruling R-T7-5).

    Rendering zips the two, so a short, long or misordered ``expected`` would silently
    attach one input's expected value to another input's arguments, or drop inputs without
    recording them in ``dropped``. An ``ok`` expectation with no ``value_repr`` would
    render ``== None`` -- an expectation nothing measured. All caller bugs, so they raise.
    """
    if len(inputs) != len(expected):
        raise ValueError(f"render_tests: {len(inputs)} inputs but {len(expected)} expectations")
    for i, e in enumerate(expected):
        if e.index != i:
            raise ValueError(f"render_tests: expected[{i}].index is {e.index}; "
                             f"expectations must be in input order")
        if e.ok and e.value_repr is None:
            raise ValueError(f"render_tests: expected[{i}] is ok but carries no value_repr")


def render_tests(module_name: str, entry_point: str, inputs: list[list], expected: list[Expected], *,
                 prefix: str, atol: float) -> tuple[str, list[tuple[str, str]]]:
    """Return (test file source, dropped).

    One ``def test_{prefix}{index}()`` per input the oracle measured *and* whose arguments
    render as literals. Inputs failing either are not rendered and not silently forgotten:
    they come back in ``dropped`` as ``(f"{prefix}{index}", reason)`` so a thinned test set
    is visible in the record rather than inferred from a count. Surviving tests keep their
    original index, so names never shift when a neighbour is dropped.
    """
    _check_alignment(inputs, expected)
    tolerance = float(atol) if atol > 0 else _DEFAULT_ATOL
    lines = [_PRELUDE.format(module=module_name, entry=entry_point, atol=repr(tolerance)), _COMPARATOR]
    dropped: list[tuple[str, str]] = []
    for e, args in zip(expected, inputs):
        name = f"{prefix}{e.index}"
        if not e.ok:
            dropped.append((name, e.reason or "unknown"))
            continue
        if not _round_trips(args):
            dropped.append((name, "no-roundtrip"))
            continue
        lines.append(f"def test_{name}():\n    assert _eq(candidate(*{args!r}), {e.value_repr})\n\n")
    return "".join(lines), dropped
