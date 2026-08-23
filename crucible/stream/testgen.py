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
itself.

*Arguments are literals too, and are checked here.* The oracle vets the value it returns
but not the input it was given, and both are rendered into the same file. An input whose
repr is not a literal the file can evaluate is dropped with the same ``no-roundtrip``
reason rather than emitted as a test that would fail for every unit alike.

*Floats compare with a tolerance.* EvalPlus ships ``atol`` for the problems that need it;
a float expectation with ``atol == 0`` still gets ``pytest.approx(..., abs=1e-9)`` rather
than ``==``, because a last-bit difference is not a behavioural kill.

Rendering is a pure function of its arguments: the same inputs and expectations always
produce byte-identical text, so a unit's test source is stable across runs and hashable.
"""
from __future__ import annotations

import ast

from .oracle import Expected

_HEADER = "import pytest\nfrom {module} import {entry} as candidate\n\n"
# Tolerance for a float expectation the problem itself did not qualify with an atol.
_DEFAULT_ATOL = 1e-9


def _is_floaty(value_repr: str) -> bool:
    """True when the expected value contains a float, so ``==`` would be too strict.

    ``ast.literal_eval``, not ``eval`` (ruling R-T7-3): this runs in the controller
    process, and the text is whatever a unit's ``__repr__`` produced inside the sandbox.
    Literal evaluation is also the more faithful question to ask -- it is exactly what the
    generated test file, which binds no names beyond ``pytest`` and ``candidate``, can do.
    Anything it cannot parse is simply not treated as floaty.
    """
    try:
        v = ast.literal_eval(value_repr)
    except Exception:
        return False
    return isinstance(v, float) or (isinstance(v, (list, tuple)) and any(isinstance(x, float) for x in v))


def _round_trips(value: object) -> bool:
    """True when ``repr(value)`` reconstructs an equal value from literals alone.

    Applied to the *arguments* (ruling R-T7-2). They are rendered into the test file as a
    literal, so an input containing ``inf`` or ``nan`` -- ``Mbpp/404`` has 12 such rows --
    emits a bare name the file never binds. That raises ``NameError`` inside the test
    body, which pytest records as a *failure*: the unit gets blamed for something only the
    renderer did. Dropping the input with a reason is the honest answer.
    """
    try:
        return ast.literal_eval(repr(value)) == value
    except Exception:
        return False


def render_tests(module_name: str, entry_point: str, inputs: list[list], expected: list[Expected], *,
                 prefix: str, atol: float) -> tuple[str, list[tuple[str, str]]]:
    """Return (test file source, dropped).

    One ``def test_{prefix}{index}()`` per input the oracle measured *and* whose arguments
    render as literals. Inputs failing either are not rendered and not silently forgotten:
    they come back in ``dropped`` as ``(f"{prefix}{index}", reason)`` so a thinned test set
    is visible in the record rather than inferred from a count. Surviving tests keep their
    original index, so names never shift when a neighbour is dropped.
    """
    lines = [_HEADER.format(module=module_name, entry=entry_point)]
    dropped: list[tuple[str, str]] = []
    for e, args in zip(expected, inputs):
        name = f"{prefix}{e.index}"
        if not e.ok:
            dropped.append((name, e.reason or "unknown"))
            continue
        if not _round_trips(args):
            dropped.append((name, "no-roundtrip"))
            continue
        call = f"candidate(*{args!r})"
        if atol > 0 or _is_floaty(e.value_repr or ""):
            cmp = f"assert {call} == pytest.approx({e.value_repr}, abs={atol or _DEFAULT_ATOL!r})"
        else:
            cmp = f"assert {call} == {e.value_repr}"
        lines.append(f"def test_{name}():\n    {cmp}\n\n")
    return "".join(lines), dropped
