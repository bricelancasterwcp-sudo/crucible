"""Falsification: re-execute a lesson's cited tests before it keeps its place (pillar 3).

A ``SemanticItem`` is a claim -- "this repair, at these spans, flips these tests from
failing to passing." Claims rot: the module the lesson cites can be re-run at any later
point, and this is the ONE place that re-running happens. ``refalsify`` re-executes
exactly the tests the item itself named when it was minted (``item.flipped_tests``)
against the source the item's cited episode actually landed
(``episode.landed_module``) -- never the item's own ``landed_diff`` text, which is a
rendering artifact, not an executable module.

*Four outcomes, two of which mutate the store.* All flipped tests still pass -> the
claim is re-verified: ``store.mark_verified(item.item_id, now)``, return ``True``. Any
flipped test now fails, times out, or errors -> the claim is false:
``store.mark_falsified(item.item_id, <description>)``, return ``False``. Anything that
prevented a real measurement at all is NOT a measurement (``crucible/sandbox/report.py``'s
R7 instrument-honesty rule, extended here to the store: no verdict means no write) --
the item is returned exactly as it stood before the call, and the function still returns
``True``. Two different things can cause that, and ``FalsifyTally`` counts them in two
different buckets (see below): ``run`` itself reports ``TestReport.infra_error``
(``infra``), or the CITATION ITSELF IS BROKEN (``infra_broken_citation`` -- see the next
note). A caller that only checks ``refalsify``'s boolean return cannot tell "still
verified" apart from either "could not check" reason; that is deliberate (an infra
hiccup must never look like a fresh falsification to a caller scanning for failures), but
it does mean ``falsify_batch``'s tally is the only place all four outcomes are visible.
Pinned by ``tests/memory/test_falsify.py``.

*A broken citation is its own outcome, not folded into sandbox infra (controller ruling,
finding 2).* ``item.cited_episode_id`` can point at an episode that is missing, that has
no ``landed_module``, that is NOT verified, or that has since been falsified itself
(``episode.falsified_by is not None`` -- ``EpisodicRecord`` carries the same field
``SemanticItem`` does); an item can also carry an empty ``flipped_tests`` tuple. All five
are "this citation cannot ground a measurement" and are classified
``infra_broken_citation``, distinct from a genuine sandbox ``infra_error`` -- the two have
different remedies (a broken citation means the memory organ's own bookkeeping drifted;
a sandbox infra error means the execution environment hiccuped) and a caller triaging a
``FalsifyTally`` needs to tell them apart. ``not episode.verified`` and
``episode.falsified_by is not None`` are checked explicitly here rather than assumed away:
the store's write path is INSERT-OR-REPLACE by identity (``store.py``'s module
docstring), so nothing in this module's own field of view guarantees the row
``item.cited_episode_id`` currently resolves to is still the same verified attempt the
item was originally minted from -- a later, failed re-attempt under the same
``(task_key, arm)`` identity could have overwritten it with an unverified row that still
happens to carry a ``landed_module`` string. This module does not get to assume the
no-silent-rewrite invariant holds elsewhere; it checks the two fields that would prove it
didn't. An empty ``flipped_tests`` gets the same classification for a narrower reason: the
sandbox runner (``crucible/sandbox/runner.py``) treats an empty ``subset`` list as "run
everything" rather than "run nothing", so an item with no cited tests at all must never
reach ``run`` -- it would silently measure the WRONG thing (every visible test) instead of
correctly reporting "there is nothing to check."

*Where the episode comes from.* This module takes no ``episode`` argument -- only
``item`` and the ``store`` it already lives in. ``item.cited_episode_id`` is looked up
against ``store.episodes()`` (a full scan, matching ``retrieve.py``'s ``_pick_exemplar``
idiom; ``MemoryStore`` exposes no by-id episode lookup and this leaf module does not add
one). This keeps the interface honest about what falsification actually re-checks: not
"does the given module still pass" (a caller could hand it anything) but "does the module
the store currently believes this VERIFIED, unfalsified episode landed still pass" -- the
store is the single source of truth for both sides of the claim.

*Never metered.* This module holds no import of the sandbox's per-task spend-cap meter
(the class is ``BudgetMeter``, defined alongside the sandbox runner) and counts nothing
toward any task's charged-execution cap. Falsification's re-executions are a maintenance
concern of the memory organ, not a search-time cost the arms' spend caps were ever meant
to track. Pinned by ``test_falsify_module_never_imports_a_spend_meter``'s crude source-scan:
the lowercase substring naming that meter's module must not appear anywhere in this
file's text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..sandbox.report import TestReport
from ..sandbox.task_run import run
from ..stream.units import Unit
from .schema import EpisodicRecord, SemanticItem
from .store import MemoryStore

_Outcome = Literal["passed", "falsified", "infra", "infra_broken_citation"]


@dataclass(frozen=True)
class FalsifyTally:
    """Counts from one ``falsify_batch`` call.

    ``checked == passed + falsified + infra + infra_broken_citation`` always.
    ``infra_broken_citation`` is a TRAILING field with a default (controller ruling,
    finding 2's plan deviation): the four originally-pinned fields keep their order, and
    this one was added after the fact without displacing them, so any existing positional
    construction of the first four fields still holds.
    """

    checked: int
    passed: int
    falsified: int
    infra: int
    infra_broken_citation: int = 0


def _find_episode(store: MemoryStore, episode_id: str) -> EpisodicRecord | None:
    """Full scan for ``episode_id`` -- see the module docstring's "where the episode comes from"."""
    for ep in store.episodes():
        if ep.item_id == episode_id:
            return ep
    return None


def _broken_citation(episode: EpisodicRecord | None, item: SemanticItem) -> bool:
    """True iff ``item``'s citation cannot ground a real measurement -- see the module
    docstring's "a broken citation is its own outcome" note for why each of these, on its
    own, is sufficient."""
    return (episode is None or episode.landed_module is None or not episode.verified
            or episode.falsified_by is not None or not item.flipped_tests)


def _falsified_by(episode: EpisodicRecord, item: SemanticItem, report: TestReport) -> str:
    """Description string for ``mark_falsified``: which of the cited tests stopped passing."""
    bad = sorted(set(report.failed) | set(report.timed_out) | set(report.errored))
    return (f"re-exec against episode {episode.item_id}: "
            f"cited tests {list(item.flipped_tests)} now failing={bad}")


def _run_and_classify(store: MemoryStore, item: SemanticItem, unit: Unit, *, now: str) -> _Outcome:
    """Re-execute ``item``'s cited tests once, apply the resulting store mutation (if any),
    and report which of the four outcomes it was. The one place ``refalsify`` and
    ``falsify_batch`` both funnel through, so a claim is only ever measured once per call."""
    episode = _find_episode(store, item.cited_episode_id)
    if _broken_citation(episode, item):
        return "infra_broken_citation"  # citation itself unusable -- see module docstring
    assert episode is not None and episode.landed_module is not None  # narrowed by the guard above

    report = run(unit, episode.landed_module, list(item.flipped_tests))
    if report.infra_error is not None:
        return "infra"

    if report.all_passed:
        store.mark_verified(item.item_id, now)
        return "passed"

    store.mark_falsified(item.item_id, _falsified_by(episode, item, report))
    return "falsified"


def refalsify(store: MemoryStore, item: SemanticItem, unit: Unit, *, now: str) -> bool:
    """Re-run ``item.flipped_tests`` against its cited episode's landed module.

    Returns ``True`` when the claim still holds (all flipped tests pass), when a sandbox
    ``infra_error`` prevented a measurement, AND when the citation itself is broken (see
    module docstring); returns ``False`` only on a genuine, measured falsification. The
    store is mutated on the passed outcome (``mark_verified``) and the falsified outcome
    (``mark_falsified``) respectively; either infra outcome mutates nothing.
    """
    return _run_and_classify(store, item, unit, now=now) != "falsified"


def falsify_batch(store: MemoryStore, items_with_units: list[tuple[SemanticItem, Unit]],
                   *, now: str) -> FalsifyTally:
    """Re-run ``refalsify`` over every ``(item, unit)`` pair and tally the four outcomes.

    Each pair is measured exactly once (via the same ``_run_and_classify`` ``refalsify``
    itself calls), so this is not "call ``refalsify`` in a loop and re-derive the outcome
    from its ambiguous ``bool``" -- the tally sees the real four-way split ``refalsify``'s
    return value collapses.
    """
    checked = passed = falsified = infra = infra_broken_citation = 0
    for item, unit in items_with_units:
        outcome = _run_and_classify(store, item, unit, now=now)
        checked += 1
        if outcome == "passed":
            passed += 1
        elif outcome == "falsified":
            falsified += 1
        elif outcome == "infra":
            infra += 1
        else:
            infra_broken_citation += 1
    return FalsifyTally(checked=checked, passed=passed, falsified=falsified, infra=infra,
                         infra_broken_citation=infra_broken_citation)
