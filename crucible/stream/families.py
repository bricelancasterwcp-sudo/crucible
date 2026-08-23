"""Operator → family map, frozen at lock (spec §4.2, amendment A2).

Every mutant in the stream belongs to exactly one *family*, and a class -- the unit
of exposure the whole experiment is built on -- is the pair (unit, family). So this
map is not a convenience taxonomy: it decides which mutants are interchangeable
for the purpose of "the agent has seen this kind of bug before". It is frozen at
the pre-registration lock and is not to be re-cut after results are in.

Two rules keep it honest.

*The map must be total over the engine's operator set.* cosmic-ray 8.7.0 ships 213
core operators; ``check_complete`` re-derives that set from the installed plugins
and returns every name that is neither mapped nor explicitly excluded. A test
asserts the result is empty, so upgrading cosmic-ray -- or losing a plugin entry
point -- fails loudly here instead of silently shrinking a family and biasing the
per-family tables downstream.

*Exclusion is explicit, never implicit.* ``VariableReplacer`` and ``VariableInserter``
(the spec's VAR family) are the only core operators that require per-variable
constructor arguments (``cause_variable``, ``effect_variable``), which our
argument-free mutation call cannot supply; amendment A2 drops them and leaves 8
families. They are named in ``EXCLUDED`` rather than quietly falling through the
map, which is what lets ``family_of`` return ``None`` for "excluded" and for
"unknown" alike without either case hiding a mapping bug.

``SDL_OPERATOR`` is ours, not cosmic-ray's: a ported MutPy statement-deletion
operator (Task 10). It appears in ``all_operator_names`` but never in the core
set, so ``check_complete`` does not look for it among the plugins.
"""
from __future__ import annotations

from cosmic_ray import plugins

FAMILIES: tuple[str, ...] = ("ARITH", "CMP", "BOOL", "UNARY", "CONST", "FLOW", "EXC", "SDL")
EXCLUDED: frozenset[str] = frozenset({"VariableReplacer", "VariableInserter"})
SDL_OPERATOR = "StatementDeletion"

_PREFIX = {"ReplaceBinaryOperator_": "ARITH", "ReplaceComparisonOperator_": "CMP", "ReplaceUnaryOperator_": "UNARY"}
_EXACT = {
    "AddNot": "BOOL", "ReplaceTrueWithFalse": "BOOL", "ReplaceFalseWithTrue": "BOOL",
    "ReplaceAndWithOr": "BOOL", "ReplaceOrWithAnd": "BOOL",
    "NumberReplacer": "CONST",
    "ReplaceBreakWithContinue": "FLOW", "ReplaceContinueWithBreak": "FLOW", "ZeroIterationForLoop": "FLOW",
    "ExceptionReplacer": "EXC", "RemoveDecorator": "EXC",
    SDL_OPERATOR: "SDL",
}


def family_of(op_name: str) -> str | None:
    """Family of ``op_name`` (given *without* the ``core/`` prefix), or ``None``.

    ``None`` means "carries no family": either the operator is excluded (A2) or the
    name is not one we map. Callers that need to tell those apart check ``EXCLUDED``.
    """
    if op_name in EXCLUDED:
        return None
    if op_name in _EXACT:
        return _EXACT[op_name]
    for pre, fam in _PREFIX.items():
        if op_name.startswith(pre):
            return fam
    return None


def core_operator_names() -> list[str]:
    """The installed cosmic-ray core operators, sorted, with the ``core/`` prefix stripped."""
    return sorted(n.removeprefix("core/") for n in plugins.operator_names())


def all_operator_names() -> list[str]:
    """The **full** engine operator set -- 213 cosmic-ray core names plus our
    ``StatementDeletion`` -- sorted, ``core/`` stripped.

    This is deliberately *not* the set the stream may use: it still contains the two
    A2-excluded names (``EXCLUDED``), which cannot even be instantiated argument-free.
    It exists so ``check_complete`` and the family tables are computed over everything
    the engine ships. **To enumerate usable operators, iterate ``operators_by_family()``
    (or filter this list by ``family_of(name) is not None``).**
    """
    return sorted(core_operator_names() + [SDL_OPERATOR])


def operators_by_family() -> dict[str, list[str]]:
    """Family → sorted operator names. Every family in ``FAMILIES`` is present as a key."""
    out: dict[str, list[str]] = {f: [] for f in FAMILIES}
    for n in all_operator_names():
        fam = family_of(n)
        if fam is not None:
            out[fam].append(n)
    return out


def check_complete() -> list[str]:
    """Core operator names that are neither mapped nor explicitly excluded. Must be ``[]``."""
    return [n for n in core_operator_names() if family_of(n) is None and n not in EXCLUDED]
