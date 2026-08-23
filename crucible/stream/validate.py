"""Decide whether a mutant is a *task*: does the visible suite kill it?

A mutant only earns a place in the stream if the tests the agent can see already fail on
it. That single rule is what makes the search well-posed -- the agent is asked to fix a
bug it can observe -- and it is why ``valid`` is exactly ``reason == "killed-visible"``
and nothing else. Three things follow from it, and each is load-bearing.

*The hidden suite is a label, not a gate.* It runs **only** when the visible suite did
not kill, and only to tell a mutant the visible tests happen to miss (``hidden-only``)
from one no test can see at all (``equivalent``). Neither is valid; the distinction is
recorded because the two say different things about the unit's test set, and Task 13
counts them separately.

*An empty hidden file is not a measurement.* Task 8 emits units with ``n_hidden == 0``,
whose ``hidden_test_src`` is the empty string; ``run_tests`` on it returns an
``infra_error`` (pytest collects nothing), which would relabel a plain ``equivalent``
mutant as ``infra``. The ``if unit.n_hidden:`` guard is what keeps that from happening.

*A non-compiling mutant is never run.* ``make_mutant`` already refuses those, but
``validate_mutant`` is called with mutants from records too, and a broken module fails
every test for the wrong reason -- it would score as a kill nobody earned. So the source
is compiled here first and reported as ``syntax``, with no sandbox run at all.

``infra`` is never valid and is counted by the caller, never charged (ruling R7).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from ..sandbox.runner import run_tests
from .mutants import Mutant
from .units import Unit

@dataclass(frozen=True)
class Validation:
    """The verdict on one mutant. Field order is frozen -- callers construct positionally.

    ``reason`` is one of ``killed-visible``, ``hidden-only``, ``equivalent``, ``infra``,
    ``syntax`` -- a closed vocabulary, because Task 13's exclusion counts key on it.

    ``n_killing_visible`` and ``visible_failed`` describe the *visible* kill only: they
    stay ``0`` / ``()`` for every non-valid reason, so a count read off a record can
    never come from the hidden suite.
    """

    mutant_key: str
    valid: bool
    reason: str
    kills_by_timeout: bool
    n_killing_visible: int
    visible_failed: tuple[str, ...]

    def to_dict(self) -> dict:
        """JSON-ready form: the tuple becomes a list so a file round-trip is exact."""
        d = asdict(self)
        d["visible_failed"] = list(self.visible_failed)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Validation":
        """Inverse of :meth:`to_dict`; restores the tuple shape so equality holds."""
        d = dict(d)
        d["visible_failed"] = tuple(d["visible_failed"])
        return cls(**d)


def validate_mutant(unit: Unit, mutant: Mutant, *, per_test_timeout_s: float = 5.0,
                    wall_cap_s: float = 60.0) -> Validation:
    """Run ``mutant`` against ``unit``'s suites and label it. Valid ⇔ killed by the visible suite."""
    try:
        # Unguarded on purpose. ``make_mutant`` wraps its identical ``compile`` in
        # ``warnings.catch_warnings()`` so that ``-W error`` cannot promote a SyntaxWarning
        # (``x is 1``) into a SyntaxError and drop a valid mutant -- but that context
        # manager mutates process-global state and this function runs inside
        # ``validate_many``'s thread pool, so the same fix is not available here. The
        # interpreter running the stream must therefore not turn SyntaxWarning into an
        # error; nothing in the repo does today.
        compile(mutant.mutated_src, unit.module_name, "exec")
    except SyntaxError:
        return Validation(mutant.key, False, "syntax", False, 0, ())
    rv = run_tests(unit.module_name, mutant.mutated_src, unit.visible_test_src,
                   per_test_timeout_s=per_test_timeout_s, wall_cap_s=wall_cap_s)
    if rv.infra_error is not None:
        return Validation(mutant.key, False, "infra", False, 0, ())
    if rv.killed:
        # A hang counts as a kill *and* is flagged: the mutant is observable, but a task
        # whose failure mode is "wait for the timeout" is worth knowing about downstream.
        killing = rv.failed + rv.timed_out + rv.errored
        return Validation(mutant.key, True, "killed-visible", bool(rv.timed_out), len(killing), killing)
    if unit.n_hidden:
        rh = run_tests(unit.module_name, mutant.mutated_src, unit.hidden_test_src,
                       per_test_timeout_s=per_test_timeout_s, wall_cap_s=wall_cap_s)
        if rh.infra_error is not None:
            return Validation(mutant.key, False, "infra", False, 0, ())
        if rh.killed:
            return Validation(mutant.key, False, "hidden-only", bool(rh.timed_out), 0, ())
    return Validation(mutant.key, False, "equivalent", False, 0, ())


def validate_many(unit: Unit, mutants: list[Mutant], *, jobs: int = 8) -> list[Validation]:
    """Validate every mutant of one unit concurrently, **in input order**.

    Only ``validate_mutant`` runs in the pool: ``execute()`` is thread-safe, but
    ``make_mutant`` is not (its ``warnings.catch_warnings()`` is process-global), so
    mutants are built by the caller before they get here. ``pool.map`` preserves order,
    which is what keeps the result zippable with ``mutants``.
    """
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(lambda m: validate_mutant(unit, m), mutants))
