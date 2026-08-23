"""The value function -- P(hidden pass | node), used to ORDER candidates before execution.

The scarce resource in the search (Task 7) is the ``K`` charged executions: only a handful
of candidate rewrites can afford to have their visible suite run. The value function ranks
the *unexecuted* candidates so those executions go to the ones most likely to pass the
HIDDEN suite -- the thing the visible suite is only a proxy for. It scores a node from its
features alone, before any execution, so it is cheap to consult for every arm.

This module is the canonical home of that interface. Task 7's ``crucible/search/loop.py``
declares a local one-method ``Value`` protocol only because this task landed after it; the
two are structurally compatible (the loop calls ``value.score(node)`` and nothing else), so
``ConstantValue`` drops straight into ``search`` without touching the loop.

v0 is deliberately untrained: :class:`ConstantValue` returns the SAME score for every node.
Ordering by a constant is a no-op tie-break, which is exactly the honest baseline for
slice-2 -- the search's execution budget and REx posterior do the real work, and S3 will
swap in a trained scorer against this measured floor rather than a guessed one. Only
:meth:`ConstantValue.score` is wired into the loop today, where it orders candidates under the
REx prior; :meth:`ConstantValue.update` is the training-hook *interface* plus an observable
counter, but the search loop does NOT call it yet -- S3 will add the ``value.update(...)`` call
when it wires value training on.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

DEFAULT_CONSTANT = 0.5


@runtime_checkable
class Value(Protocol):
    """What the search ranks unexecuted candidates by; S3 supplies a trained implementation.

    ``score`` maps a node to P(hidden pass) in ``[0, 1]`` and MUST be deterministic -- the
    search is reproducible from its seed, so a value that varied between calls on the same
    node would fabricate the experiment. ``update`` is the training hook: the loop hands back
    each node's realised ``outcome`` (did its submission pass the hidden suite) for online
    learning. ``runtime_checkable`` so a concrete scorer can be asserted to conform.
    """

    def score(self, node) -> float:
        """P(hidden pass | node's features) in ``[0, 1]``. Deterministic; no execution."""
        ...

    def update(self, node, outcome: bool) -> None:
        """Fold one realised ``outcome`` (hidden pass?) for ``node`` into the model."""
        ...


class ConstantValue:
    """v0 scaffold: scores every node with the same constant; learns nothing (training is S3).

    ``score`` returns ``c`` for ANY node -- it never inspects its argument, so ranking by it
    is a stable no-op tie-break, the honest slice-2 baseline. ``update`` is a no-op that only
    increments :attr:`updates`, an observable call counter that lets S3 verify the search
    loop wires the training hook in before any real learning exists to test. Structurally
    satisfies :class:`Value`.
    """

    def __init__(self, c: float = DEFAULT_CONSTANT) -> None:
        self.c = c
        self.updates = 0

    def score(self, node) -> float:
        """Return the constant ``c`` regardless of ``node`` -- v0 is untrained."""
        return self.c

    def update(self, node, outcome: bool) -> None:
        """No-op training step: learn nothing, but count the call so S3 can prove it fires."""
        self.updates += 1
