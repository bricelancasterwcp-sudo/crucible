"""The two sandbox entry points S2's search and driver run a ``Unit`` through.

Both are thin wrappers over :func:`crucible.sandbox.runner.run_tests`; the value they add
is *which test file* each one is allowed to see:

* :func:`run` -- the VISIBLE suite only. This is what the agent's search is scored on, so
  it must never reach the hidden tests. It optionally takes a ``subset`` of visible test
  names (bare function names) so search can probe a candidate against a slice.
* :func:`run_hidden` -- the HIDDEN suite: the driver-side OUTCOME oracle for a final
  submission. The agent never calls this; keeping it in a separate function (rather than a
  flag on ``run``) makes an accidental hidden-suite read a visible code change, not a
  parameter typo.

Neither function chooses filenames or validates ``module_name`` -- ``run_tests`` owns the
sandbox filesystem and its own name check; these wrappers only route the source.
"""
from __future__ import annotations

from ..stream.units import Unit
from .runner import run_tests
from .report import TestReport


def run(unit: Unit, patch: str, subset: list[str] | None, *,
        per_test_timeout_s: float = 5.0, wall_cap_s: float = 60.0) -> TestReport:
    """Run ``patch`` (a full-module rewrite) against ``unit``'s VISIBLE suite.

    ``subset`` selects visible tests by bare function name; ``None`` runs them all. The
    hidden suite is never touched -- this is the surface the agent's search is scored on.
    """
    return run_tests(unit.module_name, patch, unit.visible_test_src, subset=subset,
                     per_test_timeout_s=per_test_timeout_s, wall_cap_s=wall_cap_s)


def run_hidden(unit: Unit, patch: str, *,
               per_test_timeout_s: float = 5.0, wall_cap_s: float = 60.0) -> TestReport:
    """Run ``patch`` against ``unit``'s HIDDEN suite: the driver-only outcome oracle.

    The agent never calls this. It scores a final submission against tests the search
    could not see, so a candidate that overfit the visible suite is caught here.
    """
    return run_tests(unit.module_name, patch, unit.hidden_test_src, subset=None,
                     per_test_timeout_s=per_test_timeout_s, wall_cap_s=wall_cap_s)
