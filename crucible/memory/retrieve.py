"""Retrieval: what A_full's prompt block actually contains (design spec §3, task 4 brief).

*Class-exact, then family-wide -- never merged, and the trigger is POST-FILTER.*
``retrieve`` calls ``store.semantic_for`` first; it falls back to ``store.semantic_family``
(the "novel task" path, spec §3: "a NOVEL task can only ever get family-level lessons")
unless the exact-class pool has at least one LIVE (non-falsified) lesson. A class whose
only lessons are all falsified is therefore treated the same as a class with no lessons at
all -- both fall back to family-wide -- rather than surfacing zero lessons when a
perfectly good family-wide fallback exists. The two pools are never combined -- when the
class path IS taken (>=1 live exact-class lesson), the family-wide pool is never
consulted.

*The episodic exemplar is decoupled from the lesson path.* It does NOT follow whichever
pool the lessons came from. It is included whenever a verified, unfalsified episode
exists for this exact (unit_id, family) class -- regardless of whether the lessons above
came from the class-exact or the family-wide pool -- and absent only when no such episode
exists (the genuinely-novel-unit case: no class content at all). A class with only
falsified lessons can therefore still produce a class exemplar alongside family-wide
lessons: falsification of a *lesson* says nothing about whether the *episode* it was
templated from is still a fact about this unit.

*Honest storage means retrieval does its own filtering.* ``store.py``'s module docstring
is explicit that ``semantic_for``/``semantic_family``/``episodes`` return falsified rows
unfiltered -- filtering is retrieval's job. Both the semantic-lesson pool and the
episodic-exemplar candidate pool are filtered to ``falsified_by is None`` here.
Filtering the exemplar's episode pool the same way is an inferred choice (the brief's
"filter OUT falsified items" sentence sits directly before the ranking rule for lessons,
so it is unambiguous there; extending it to episodes is not literally pinned) --
presenting a since-falsified episode as "a prior working version" would be actively
dishonest, so the same discipline applies. Flagging for later tasks in case this needs
to be relaxed.

*Ranking (semantic lessons): last_verified_at DESC (None last), then confidence DESC,
then item_id ASC for determinism.* Implemented as three stable sorts applied from least
to most significant (Python's ``list.sort``/``sorted`` are documented-stable even with
``reverse=True`` -- equal elements never swap relative order -- so a later, more
significant sort's ties fall through to whatever an earlier, less significant sort already
decided). ``last_verified_at or ""`` maps ``None`` to the empty string, which sorts
lexicographically before every real ISO-8601 timestamp, so a single ``reverse=True`` pass
on that key gives "real timestamps DESC, None last" in one step -- no separate
None-handling branch needed.

*The episodic exemplar has its own, simpler selection rule: most recent, not the general
rank chain.* The brief defines it as "the landed module of the most recent verified
episode for the same (unit, family) class" -- recency of the episode's own ``created_at``,
not the lesson-ranking formula above (episodes don't participate in that ranking; they are
a single yes/no pick). Ties broken by ``item_id`` ASC for the same determinism reason.
``store.episodes()`` has no ``unit_id``/``family`` filter, so this module filters the full
(verified-only) episode list itself.

*The hard char budget's drop order is exemplar first, then the second lesson -- never
mid-item.* ``_budget_candidates`` yields, in order: [lesson(s) + exemplar], then
[lesson(s) alone] (only if an exemplar was actually present -- otherwise this candidate is
identical to the first and is never separately yielded), then [first lesson alone] (only
if there was a second lesson to drop). ``retrieve`` takes the first candidate that fits
and stops; if none fit -- including the single-lesson-alone case -- it returns
``RetrievedBlock(None, ())``. An all-empty candidate is never yielded, so an organ with
nothing to say never falls through to a candidate that "fits" trivially; the None-vs-zero
discipline is enforced structurally, not by a special-cased length check.

*Deterministic.* No clock is read, no randomness anywhere in this module; every sort ends
in an ``item_id`` tie-break. Two calls against the same store state return equal
``RetrievedBlock``s.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .distill import render_lesson
from .schema import EpisodicRecord, SemanticItem
from .store import MemoryStore

CONTEXT_BUDGET_CHARS = 4800

_BLOCK_HEADER = "## Prior experience with this code"
_EXEMPLAR_HEADER = "### A prior working version of this module"

# (rendered text, item_id) -- the unit both the ranking chain and the budget drop-order
# machinery below carry around, so a part's provenance travels with its text.
_Part = tuple[str, str]


@dataclass(frozen=True)
class RetrievedBlock:
    """The A_full prompt block, or nothing. ``block=None`` <=> nothing retrieved -- never
    ``""``; an empty organ (or a first lesson too big to fit at all) is ``None``, not an
    empty string standing in for it (None-vs-zero, per the module docstring)."""

    block: str | None
    item_ids: tuple[str, ...]


def retrieve(store: MemoryStore, unit_id: str, family: str, *, exact_only: bool = False) -> RetrievedBlock:
    """See the module docstring for the full policy. Pure function of the store's current
    contents -- no writes, no clock, so two calls against the same state are equal.

    ``exact_only`` (Phase-B prereg §3, arm A_mem_exactonly) changes exactly one decision:
    when the exact-class pool has no live lesson, the eligible-lesson pool is EMPTY instead
    of the family-wide pool -- a stranger unit gets silence, never a family-wide lesson. The
    exemplar path (``_pick_exemplar``, already class-exact) is deliberately untouched, so a
    class with a live exact lesson returns a byte-identical block under both flags.
    """
    exact = store.semantic_for(unit_id, family)
    live_exact = [item for item in exact if item.falsified_by is None]
    if live_exact:
        eligible = live_exact
    elif exact_only:
        eligible = []   # A_mem_exactonly (Phase-B §3): strangers get silence, never family-wide lessons
    else:
        family_pool = store.semantic_family(family)
        eligible = [item for item in family_pool if item.falsified_by is None]
    lessons = _rank_lessons(eligible)[:2]

    # Decoupled from the lesson path above -- see module docstring.
    exemplar = _pick_exemplar(store, unit_id, family)

    lesson_parts: list[_Part] = [(render_lesson(item), item.item_id) for item in lessons]
    exemplar_part: _Part | None = None
    if exemplar is not None:
        exemplar_part = (_render_exemplar(exemplar.landed_module), exemplar.item_id)

    for parts in _budget_candidates(lesson_parts, exemplar_part):
        rendered = _assemble(parts)
        if len(rendered) <= CONTEXT_BUDGET_CHARS:
            return RetrievedBlock(rendered, tuple(item_id for _, item_id in parts))
    return RetrievedBlock(None, ())


def _rank_lessons(items: list[SemanticItem]) -> list[SemanticItem]:
    """last_verified_at DESC (None last), then confidence DESC, then item_id ASC. See the
    module docstring's "Ranking" note for why three stable sorts, least-significant first,
    is the correct way to compose this rather than one composite key."""
    ranked = sorted(items, key=lambda item: item.item_id)
    ranked.sort(key=lambda item: item.confidence, reverse=True)
    ranked.sort(key=lambda item: item.last_verified_at or "", reverse=True)
    return ranked


def _pick_exemplar(store: MemoryStore, unit_id: str, family: str) -> EpisodicRecord | None:
    """Most recent (by ``created_at``, ties by ``item_id``) verified, unfalsified episode
    for exactly this (unit_id, family) class, or ``None`` if there is none. ``store.
    episodes()`` carries no unit/family filter, so it is applied here."""
    candidates = [
        ep for ep in store.episodes(verified_only=True)
        if ep.unit_id == unit_id and ep.family == family
        and ep.falsified_by is None and ep.landed_module is not None
    ]
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda ep: ep.item_id)
    ranked.sort(key=lambda ep: ep.created_at, reverse=True)
    return ranked[0]


def _render_exemplar(landed_module: str) -> str:
    return f"{_EXEMPLAR_HEADER}\n```python\n{landed_module}\n```"


def _budget_candidates(lesson_parts: list[_Part], exemplar_part: _Part | None) -> Iterator[list[_Part]]:
    """Yield part-lists in strict drop order: everything, then exemplar dropped, then the
    second lesson also dropped. Never yields an empty list -- see the module docstring's
    None-vs-zero note."""
    full = list(lesson_parts) + ([exemplar_part] if exemplar_part is not None else [])
    if full:
        yield full
    if exemplar_part is not None and lesson_parts:
        yield list(lesson_parts)
    if len(lesson_parts) >= 2:
        yield lesson_parts[:1]


def _assemble(parts: list[_Part]) -> str:
    return "\n\n".join([_BLOCK_HEADER] + [text for text, _ in parts])
