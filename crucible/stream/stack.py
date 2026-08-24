"""Rung-1 stacking: compose two same-family single-site mutants into one two-site mutant.

Spec: docs/superpowers/specs/2026-08-23-crucible-s2.5-stack2-design.md §3-§5.

The load-bearing invariant (spec §4): a stacked mutant's diff touches exactly the two
intended spans, or the pair is dropped -- never guessed. Applying component B's
original-source occurrence index to A-mutated source is unsafe: a mutation can CREATE a
new match for B's operator earlier in the AST visit order, silently landing B on the
wrong site. So the later-span component is taken as its already-applied ``mutated_src``,
and the earlier component is re-selected on that intermediate source by EXACT span
match -- a strictly later edit cannot move an earlier span. Not exactly one match ==
drop (``stack-apply``).

``spans_overlap`` treats span ends as INCLUSIVE (conservative): cosmic-ray end-position
semantics are not something this module should bet the stream on, so a boundary-touching
pair is rejected rather than risk an ill-defined composition.
"""
from __future__ import annotations

import warnings

from cosmic_ray.ast import ast_nodes, get_ast
from cosmic_ray.mutating import mutate_code

from .mutants import Component, Mutant, Span, _unified, operator_instance
from .units import Unit, sha256_text


def spans_overlap(a: Span, b: Span) -> bool:
    """Inclusive-end interval intersection under (line, col) tuple order. Nesting is a
    special case of intersection and is rejected by the same comparison."""
    return a[0] <= b[1] and b[0] <= a[1]


def _match_occurrence(src: str, operator: str, span: Span) -> int | None:
    """The occurrence index of ``operator`` at exactly ``span`` in ``src``, or ``None``
    unless exactly one position matches."""
    op = operator_instance(operator)
    positions = [p for node in ast_nodes(get_ast(src)) for p in op.mutation_positions(node)]
    hits = [i for i, s in enumerate(positions) if (tuple(s[0]), tuple(s[1])) == span]
    return hits[0] if len(hits) == 1 else None


def compose_pair(unit: Unit, ma: Mutant, mb: Mutant) -> Mutant | None:
    """The two-site Mutant stacking ``ma`` and ``mb``, or ``None`` if the pair is not a task.

    Argument order is irrelevant: the pair is internally ordered by span. Every ``None``
    is a ``stack-apply``-class drop except the guards shared with ``make_mutant``
    (compiles; differs from the original and from both singles).
    """
    if spans_overlap(ma.span, mb.span):
        return None
    early, late = (ma, mb) if ma.span < mb.span else (mb, ma)
    occ = _match_occurrence(late.mutated_src, early.operator, early.span)
    if occ is None:
        return None
    stacked = mutate_code(late.mutated_src, operator_instance(early.operator), occ)
    if stacked is None or stacked in (unit.module_src, early.mutated_src, late.mutated_src):
        return None
    try:
        # Single-threaded here, like make_mutant; SyntaxWarning must not become SyntaxError.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            compile(stacked, unit.module_name, "exec")
    except SyntaxError:
        return None
    diff = _unified(unit.module_name, unit.module_src, stacked)
    return Mutant(unit.unit_id, sha256_text(unit.src_hash + "\n" + diff), early.operator,
                  early.occurrence, early.family, early.span, stacked, diff,
                  components=(Component(early.operator, early.occurrence, early.span),
                              Component(late.operator, late.occurrence, late.span)))
