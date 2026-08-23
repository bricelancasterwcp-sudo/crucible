"""Enumerate mutation positions, apply one, and package it as a content-keyed ``Mutant``.

This is where a *unit* becomes *tasks*. For each operator we ask cosmic-ray for every
position it could mutate in the unit's stripped source, and each position becomes a
``MutantSpec``: a reproducible coordinate (operator name + occurrence index) plus the
family it belongs to and the span it touches.

Three things here are load-bearing.

*The occurrence index is cosmic-ray's, not ours.* ``cosmic_ray.mutating.mutate_code``
walks the tree and counts **one occurrence per position yielded by
``op.mutation_positions(node)``**, in the order ``ast_nodes`` visits nodes. So
``enumerate_specs`` numbers positions by walking the very same iteration, and
``apply_spec`` can hand the index straight back. Out-of-range occurrences return
``None`` rather than raising, which is how a spec that no longer matches its source
(a stale record, a re-stripped unit) shows up as "no mutant" instead of a crash.

*The key is the content, not the coordinate.* ``key = sha256(src_hash + "\\n" + diff)``.
Two different operators that happen to produce byte-identical mutated source collapse
onto one key, which is what we want: the task is the *bug*, not the recipe that made
it. That makes the diff text part of the identity, so ``_unified`` pins fixed headers
(``a/<module>.py`` -> ``b/<module>.py``) and passes **no** file dates -- difflib would
otherwise append a timestamp and every run would mint fresh keys for the same bug.

*``StatementDeletion`` is ours, not a plugin.* It is not registered as a cosmic-ray
entry point, so ``plugins.get_operator("StatementDeletion")`` fails with an obscure
``ValueError`` from the prefix-stripping code. ``operator_instance`` special-cases it
before ever reaching the plugin registry.
"""
from __future__ import annotations

import difflib
import random
from dataclasses import asdict, dataclass

from cosmic_ray import plugins
from cosmic_ray.ast import ast_nodes, get_ast
from cosmic_ray.mutating import mutate_code

from .families import SDL_OPERATOR, family_of
from .sdl import StatementDeletion
from .units import Unit, sha256_text

Span = tuple[tuple[int, int], tuple[int, int]]
"""``((start_line, start_col), (end_line, end_col))`` -- cosmic-ray's 1-based line, 0-based column."""


@dataclass(frozen=True)
class MutantSpec:
    """A mutation that *could* be applied: where it is, what makes it, which family it counts as.

    A spec is cheap and source-independent to carry around -- ``apply_spec`` is what
    actually runs the operator. ``span`` is carried so ``sample_specs`` can prefer
    distinct locations without re-parsing.
    """

    operator: str
    occurrence: int
    family: str
    span: Span


@dataclass(frozen=True)
class Mutant:
    """One applied mutation: the mutated source, its diff, and the key both hash to.

    Field order is frozen -- downstream tasks construct ``Mutant`` positionally, and
    ``key`` becomes ``TaskSpec.task_key`` unchanged.
    """

    unit_id: str
    key: str
    operator: str
    occurrence: int
    family: str
    span: Span
    mutated_src: str
    diff: str

    def to_dict(self) -> dict:
        """JSON-ready form: the nested span tuples become lists so a file round-trip is exact."""
        d = asdict(self)
        d["span"] = [list(d["span"][0]), list(d["span"][1])]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Mutant":
        """Inverse of :meth:`to_dict`; restores the tuple shape so equality holds."""
        d = dict(d)
        d["span"] = (tuple(d["span"][0]), tuple(d["span"][1]))
        return cls(**d)


def operator_instance(name: str):
    """Instantiate the operator called ``name`` (given *without* the ``core/`` prefix).

    ``StatementDeletion`` is ours (Task 10) and is not a registered plugin; everything
    else comes from cosmic-ray's registry, which returns the operator *class*.
    """
    if name == SDL_OPERATOR:
        return StatementDeletion()
    return plugins.get_operator(f"core/{name}")()


def enumerate_specs(src: str, operators: list[str]) -> list[MutantSpec]:
    """Every position each operator could mutate in ``src``, in operator then occurrence order.

    Operators with no family (``EXCLUDED`` or unknown -- see ``families.family_of``) are
    skipped silently: a mutant with no family has no class to belong to, so it cannot
    enter the stream. Operators that match nothing simply contribute no specs.
    """
    tree = get_ast(src)
    nodes = list(ast_nodes(tree))
    out: list[MutantSpec] = []
    for name in operators:
        fam = family_of(name)
        if fam is None:
            continue
        op = operator_instance(name)
        positions = [p for node in nodes for p in op.mutation_positions(node)]
        for i, span in enumerate(positions):
            out.append(MutantSpec(name, i, fam, (tuple(span[0]), tuple(span[1]))))
    return out


def apply_spec(src: str, spec: MutantSpec) -> str | None:
    """Apply ``spec`` to ``src``; ``None`` when that occurrence does not match this source."""
    return mutate_code(src, operator_instance(spec.operator), spec.occurrence)


def _unified(module_name: str, a: str, b: str) -> str:
    """Unified diff with fixed headers and **no** dates -- the key hashes this text."""
    return "".join(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True),
                                        fromfile=f"a/{module_name}.py", tofile=f"b/{module_name}.py", n=3))


def make_mutant(unit: Unit, spec: MutantSpec) -> Mutant | None:
    """Apply ``spec`` to ``unit`` and key the result by content, or ``None`` if it is not a task.

    Three ways a spec fails to become a mutant, all returning ``None`` rather than
    raising: the occurrence does not match this source, the "mutation" left the source
    byte-identical (an empty diff would key a task that is not a bug), or the result
    does not compile (a broken module fails every test for the wrong reason, which
    would score as a kill the agent never earned).
    """
    mutated = apply_spec(unit.module_src, spec)
    if mutated is None or mutated == unit.module_src:
        return None
    try:
        compile(mutated, unit.module_name, "exec")
    except SyntaxError:
        return None
    diff = _unified(unit.module_name, unit.module_src, mutated)
    return Mutant(unit.unit_id, sha256_text(unit.src_hash + "\n" + diff), spec.operator, spec.occurrence,
                  spec.family, spec.span, mutated, diff)


def sample_specs(specs: list[MutantSpec], *, per_family: int, rng: random.Random) -> list[MutantSpec]:
    """At most ``per_family`` specs per family, sampled with ``rng``, distinct spans preferred.

    The caller owns the randomness: ``rng`` is a seeded ``random.Random``, never module
    state, so the whole stream is reproducible from the seed in the run record. Families
    are visited in sorted order and the result is a deterministic function of
    ``(specs, per_family, rng state)``.

    "Distinct spans preferred" spreads the sample over the unit instead of stacking
    several operators on one token: a family's quota is filled first with specs whose
    spans have not been taken, and only then topped up with the leftovers.
    """
    by: dict[str, list[MutantSpec]] = {}
    for s in specs:
        by.setdefault(s.family, []).append(s)
    out: list[MutantSpec] = []
    for fam in sorted(by):
        pool = list(by[fam])
        rng.shuffle(pool)
        chosen: list[MutantSpec] = []
        seen_spans: set[Span] = set()
        for s in pool:                      # distinct spans first
            if s.span not in seen_spans and len(chosen) < per_family:
                chosen.append(s)
                seen_spans.add(s.span)
        for s in pool:                      # then fill
            if len(chosen) >= per_family:
                break
            if s not in chosen:
                chosen.append(s)
        out.extend(chosen)
    return out
