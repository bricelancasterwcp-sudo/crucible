"""StatementDeletion: MutPy's SDL idea (Apache-2.0, reimplemented) on cosmic-ray's Operator ABC.

MutPy's statement-deletion operator is reimplemented here -- no code was copied --
against ``cosmic_ray.operators.operator.Operator`` so that Task 11 can enumerate and
apply it with the same ``cosmic_ray.mutating.mutate_code(src, op, occurrence)``
machinery used for cosmic-ray's own 213 core operators.

A candidate is one ``simple_stmt`` that lives directly inside an indented ``suite``.
Deleting it means replacing it with ``pass`` (keeping the original indentation), which
keeps the enclosing block non-empty so the mutated source always compiles. Docstrings,
statements that are already ``pass``, and statements outside a suite (module-level
imports, defs) are never candidates.
"""

from __future__ import annotations

import parso
from cosmic_ray.operators.operator import Operator


def _is_docstring(stmt) -> bool:
    """True if ``stmt`` is a bare string expression (a doc comment)."""
    first = stmt.children[0]
    return getattr(first, "type", "") == "string"


def _is_pass(stmt) -> bool:
    """True if ``stmt`` is already a ``pass`` statement -- deleting it is a no-op."""
    first = stmt.children[0]
    return getattr(first, "type", "") == "keyword" and first.value == "pass"


def _deletable(node) -> bool:
    """True if ``node`` is a suite-level statement this operator may delete."""
    return (
        node.type == "simple_stmt"
        and node.parent is not None
        and node.parent.type == "suite"
        and not _is_docstring(node)
        and not _is_pass(node)
    )


class StatementDeletion(Operator):
    """Delete one statement from an indented block by replacing it with ``pass``."""

    def mutation_positions(self, node):
        """Yield ``((start_line, start_col), (end_line, end_col))`` if ``node`` is deletable."""
        if _deletable(node):
            yield (node.start_pos, node.end_pos)

    def mutate(self, node, index):
        """Return a ``pass`` statement carrying ``node``'s leading indentation."""
        prefix = node.get_first_leaf().prefix
        new = parso.parse("pass\n").children[0]
        new.get_first_leaf().prefix = prefix
        new.parent = node.parent
        return new

    @classmethod
    def examples(cls):
        """No cosmic-ray ``Example``\\s: this operator is exercised by ``tests/stream/test_sdl.py``."""
        return ()
