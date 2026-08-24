"""Tests for retrieval: what A_full's prompt gets from the memory organ (design spec §3,
task 4 brief). Exercises the policy in full -- class-exact-then-family, the falsified
filter, the ranking chain, the hard char budget's exact drop order, and determinism.

Fixtures follow ``tests/memory/test_store.py``'s local ``_episode``/``_semantic`` helper
pattern, extended with the knobs this file's scenarios need (``last_verified_at``,
``confidence``, ``falsified_by``, ``landed_diff``/``landed_module`` length, ``created_at``).

Per Task 3's note: minted lessons all default to ``confidence=1.0`` in real use, so every
ranking test here sets DISTINCT ``last_verified_at`` and DISTINCT ``confidence`` values
explicitly rather than relying on any mint default -- otherwise a ranking bug could hide
behind a tie nothing here actually forces.

Four things are load-bearing, not incidental.

*Exact-class beats family-wide, structurally.* ``retrieve`` never merges the two pools --
if the exact-class query returns anything at all (even a since-falsified item), the
family-wide pool is not consulted, full stop. ``test_exact_class_beats_family_wide``
seeds both pools and checks the family-only item's id never appears.

*Falsified items are invisible, not just deprioritised.* The store returns falsified rows
(honest storage, Task 2); the retriever must filter them out itself.
``test_falsified_items_never_appear`` seeds a falsified item that would rank #1 by
recency if the filter were dropped, so removing the filter changes the observable
``item_ids`` -- the brief's named mutation target (see the task report for the
pyc-purge evidence).

*The budget's drop order is exemplar-first, then the second lesson, never mid-item.*
``test_budget_drops_exemplar_first`` sizes lessons/exemplar so that dropping the
exemplar (and only the exemplar) makes the block fit -- an inverted implementation that
drops the second lesson first would instead keep the exemplar and produce a different,
observably wrong ``item_ids`` tuple (the brief's other named mutation target).
``test_budget_drops_second_lesson_after_exemplar_still_over`` goes one step further, and
``test_budget_oversized_first_lesson_yields_none_block`` pins the terminal case.

*The episodic exemplar is class-specific, never carried across the family-fallback path.*
``test_exemplar_omitted_on_family_fallback_path`` seeds a verified episode that exactly
matches the query's (unit_id, family) -- an exemplar it WOULD pick class-exact -- but
forces the family-fallback path by leaving the exact-class semantic table empty, and
asserts the exemplar is entirely absent anyway.
"""

from pathlib import Path

from crucible.memory.distill import render_lesson
from crucible.memory.retrieve import CONTEXT_BUDGET_CHARS, RetrievedBlock, retrieve
from crucible.memory.schema import EpisodicRecord, SemanticItem, content_id
from crucible.memory.store import MemoryStore

UNIT = "X/0"
FAMILY = "ARITH"
OTHER_UNIT = "X/1"
OTHER_FAMILY = "OFFBY1"


def _episode(task_key: str, arm: str = "A_full", *, unit_id: str = UNIT, family: str = FAMILY,
             verified: bool = True, landed_module: str | None = "def f():\n    return 1\n",
             created_at: str = "2026-08-24T10:00:00Z", falsified_by: str | None = None) -> EpisodicRecord:
    item_id = content_id("episode", {"task_key": task_key, "arm": arm})
    return EpisodicRecord(
        item_id=item_id, task_key=task_key, arm=arm, unit_id=unit_id, family=family,
        class_id=f"{unit_id}|{family}", phase=1, kind="first",
        root_prompt="Fix the bug.", landed_module=landed_module, visible_reward=1.0,
        executions_charged=2, hidden_pass=True if verified else False, verified=verified,
        memory_item_ids=(), created_at=created_at, confidence=0.8,
        status="active", version=1, source_locator=f"run:t/task:{task_key}",
        valid_at=created_at, invalid_at=None, expired_at=None,
        last_verified_at=None, falsified_by=falsified_by, verification_method="hidden-suite",
    )


def _semantic(cited_episode_id: str, *, unit_id: str = UNIT, family: str = FAMILY,
              landed_diff: str = "-    return 0\n+    return 1\n",
              flipped_tests: tuple = ("test_v0",),
              last_verified_at: str | None = None, confidence: float = 0.75,
              falsified_by: str | None = None) -> SemanticItem:
    item_id = content_id("semantic", {"cited_episode_id": cited_episode_id})
    return SemanticItem(
        item_id=item_id, unit_id=unit_id, family=family, class_id=f"{unit_id}|{family}",
        cited_episode_id=cited_episode_id, mutated_spans=(((2, 5), (2, 9)),),
        landed_diff=landed_diff, flipped_tests=flipped_tests, killing_tests=flipped_tests,
        created_at="2026-08-24T10:06:00Z", confidence=confidence,
        status="active", version=1, source_locator=f"run:t/episode:{cited_episode_id}",
        valid_at="2026-08-24T10:06:00Z", invalid_at=None, expired_at=None,
        last_verified_at=last_verified_at, falsified_by=falsified_by,
        verification_method="mechanical-template",
    )


def test_empty_store_returns_none_block_and_empty_ids(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    result = retrieve(store, UNIT, FAMILY)
    assert result == RetrievedBlock(None, ())
    store.close()


def test_exact_class_beats_family_wide(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    exact = _semantic("ep-exact", unit_id=UNIT, family=FAMILY, last_verified_at="2026-08-24T09:00:00Z")
    family_only = _semantic("ep-family", unit_id=OTHER_UNIT, family=FAMILY, last_verified_at="2026-08-24T09:00:00Z",
                             landed_diff="-    return 99\n+    return -99\n")
    store.write_semantic(exact)
    store.write_semantic(family_only)
    result = retrieve(store, UNIT, FAMILY)
    assert result.item_ids == (exact.item_id,)
    assert family_only.landed_diff not in result.block
    store.close()


def test_family_fallback_used_when_exact_class_is_empty(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    family_only = _semantic("ep-family", unit_id=OTHER_UNIT, family=FAMILY)
    store.write_semantic(family_only)
    result = retrieve(store, UNIT, FAMILY)
    assert result.item_ids == (family_only.item_id,)
    assert result.block is not None
    store.close()


def test_falsified_items_never_appear(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    # Falsified item ranks #1 by recency if the filter is dropped -- a live discriminator.
    falsified = _semantic("ep-falsified", last_verified_at="2026-08-24T12:00:00Z",
                           confidence=0.9, falsified_by="re-run:tk-9 flipped back to failing",
                           landed_diff="-    return 42\n+    return -42\n")
    live = _semantic("ep-live", last_verified_at="2026-08-20T09:00:00Z", confidence=0.5)
    store.write_semantic(falsified)
    store.write_semantic(live)
    result = retrieve(store, UNIT, FAMILY)
    assert result.item_ids == (live.item_id,)
    assert falsified.landed_diff not in (result.block or "")
    store.close()


def test_ranking_by_last_verified_at_desc_none_last(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    # Distinct last_verified_at AND distinct confidence (Task 3's note) so recency, not
    # confidence, is what's actually discriminating here. item_none has the HIGHEST
    # confidence but no verification timestamp -- it must still sort last.
    item_none = _semantic("ep-none", last_verified_at=None, confidence=0.99)
    item_recent = _semantic("ep-recent", last_verified_at="2026-08-24T09:00:00Z", confidence=0.4)
    item_older = _semantic("ep-older", last_verified_at="2026-08-20T09:00:00Z", confidence=0.6)
    for item in (item_none, item_recent, item_older):
        store.write_semantic(item)
    result = retrieve(store, UNIT, FAMILY)
    # Top-2 only: the two with real timestamps, most recent first.
    assert result.item_ids == (item_recent.item_id, item_older.item_id)
    store.close()


def test_ranking_falls_back_to_confidence_when_last_verified_at_ties(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    same_ts = "2026-08-24T09:00:00Z"
    high_conf = _semantic("ep-high", last_verified_at=same_ts, confidence=0.9)
    low_conf = _semantic("ep-low", last_verified_at=same_ts, confidence=0.2)
    store.write_semantic(low_conf)  # insertion order deliberately reversed
    store.write_semantic(high_conf)
    result = retrieve(store, UNIT, FAMILY)
    assert result.item_ids == (high_conf.item_id, low_conf.item_id)
    store.close()


def test_ranking_falls_back_to_item_id_when_last_verified_at_and_confidence_tie(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    same_ts = "2026-08-24T09:00:00Z"
    item_a = _semantic("ep-tiebreak-a", last_verified_at=same_ts, confidence=0.5)
    item_b = _semantic("ep-tiebreak-b", last_verified_at=same_ts, confidence=0.5)
    store.write_semantic(item_a)
    store.write_semantic(item_b)
    expected = tuple(i.item_id for i in sorted([item_a, item_b], key=lambda i: i.item_id))
    result = retrieve(store, UNIT, FAMILY)
    assert result.item_ids == expected
    store.close()


def test_exemplar_is_the_most_recent_verified_episode_of_the_same_class(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    lesson = _semantic("ep-lesson")
    store.write_semantic(lesson)
    older_episode = _episode("tk-old", created_at="2026-08-20T08:00:00Z",
                              landed_module="def f():\n    return 1  # old\n")
    newer_episode = _episode("tk-new", created_at="2026-08-24T08:00:00Z",
                              landed_module="def f():\n    return 1  # new\n")
    store.write_episode(older_episode)
    store.write_episode(newer_episode)
    result = retrieve(store, UNIT, FAMILY)
    assert newer_episode.item_id in result.item_ids
    assert older_episode.item_id not in result.item_ids
    assert "### A prior working version of this module" in result.block
    assert "```python" in result.block
    assert newer_episode.landed_module in result.block
    store.close()


def test_exemplar_excludes_falsified_and_unverified_episodes(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    lesson = _semantic("ep-lesson")
    store.write_semantic(lesson)
    unverified = _episode("tk-unverified", created_at="2026-08-24T09:00:00Z", verified=False)
    falsified = _episode("tk-falsified", created_at="2026-08-24T09:30:00Z",
                          falsified_by="re-run:tk-99 failed")
    good = _episode("tk-good", created_at="2026-08-20T08:00:00Z")
    store.write_episode(unverified)
    store.write_episode(falsified)
    store.write_episode(good)
    result = retrieve(store, UNIT, FAMILY)
    assert good.item_id in result.item_ids
    assert unverified.item_id not in result.item_ids
    assert falsified.item_id not in result.item_ids
    store.close()


def test_exemplar_omitted_on_family_fallback_path(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    # A verified episode that WOULD be picked class-exact...
    episode = _episode("tk-1")
    store.write_episode(episode)
    # ...but the exact-class semantic table is empty, so retrieve must fall back to
    # family-wide, and the exemplar (class-specific) must be omitted entirely, not just
    # substituted for a family-wide episode.
    family_only = _semantic("ep-family", unit_id=OTHER_UNIT, family=FAMILY)
    store.write_semantic(family_only)
    result = retrieve(store, UNIT, FAMILY)
    assert episode.item_id not in result.item_ids
    assert "### A prior working version of this module" not in result.block
    store.close()


def test_block_header_line(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    item = _semantic("ep-1")
    store.write_semantic(item)
    result = retrieve(store, UNIT, FAMILY)
    assert result.block.splitlines()[0] == "## Prior experience with this code"
    assert render_lesson(item) in result.block
    store.close()


def _padded_semantic(cited_episode_id: str, target_len: int, *, fill: str = "x", **kwargs) -> SemanticItem:
    """Build a semantic item whose ``render_lesson`` output is exactly ``target_len`` chars.

    Measures the template's fixed overhead empirically (build with an empty diff, render,
    measure) rather than hand-computing it, so the padding is exact regardless of any
    future wording change to LESSON_TEMPLATE. ``fill`` lets two padded items in the same
    test have distinct rendered text (different fill character) so a "this item's text is
    not in the block" assertion is a real check, not a vacuous one against identical text.
    """
    probe = _semantic(cited_episode_id, landed_diff="", **kwargs)
    overhead = len(render_lesson(probe))
    pad = fill * max(0, target_len - overhead)
    return _semantic(cited_episode_id, landed_diff=pad, **kwargs)


def test_budget_drops_exemplar_first(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    lesson1 = _padded_semantic("ep-1", 2100, last_verified_at="2026-08-24T09:00:00Z", confidence=0.9)
    lesson2 = _padded_semantic("ep-2", 2100, last_verified_at="2026-08-23T09:00:00Z", confidence=0.9)
    store.write_semantic(lesson1)
    store.write_semantic(lesson2)
    episode = _episode("tk-1", landed_module="z" * 1500)
    store.write_episode(episode)

    # Sanity: the full set (2 lessons + exemplar) must exceed budget, but 2 lessons alone
    # must fit, for this test to actually exercise the drop-exemplar-first branch.
    full_len = len(render_lesson(lesson1)) + len(render_lesson(lesson2)) + len(episode.landed_module)
    assert full_len > CONTEXT_BUDGET_CHARS

    result = retrieve(store, UNIT, FAMILY)
    assert result.item_ids == (lesson1.item_id, lesson2.item_id)
    assert len(result.block) <= CONTEXT_BUDGET_CHARS
    assert "### A prior working version of this module" not in result.block
    store.close()


def test_budget_drops_second_lesson_after_exemplar_still_over(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    lesson1 = _padded_semantic("ep-1", 2600, fill="x", last_verified_at="2026-08-24T09:00:00Z", confidence=0.9)
    lesson2 = _padded_semantic("ep-2", 2600, fill="y", last_verified_at="2026-08-23T09:00:00Z", confidence=0.9)
    store.write_semantic(lesson1)
    store.write_semantic(lesson2)
    episode = _episode("tk-1", landed_module="z" * 1500)
    store.write_episode(episode)

    # Even without the exemplar, 2 full-size lessons alone must still exceed budget.
    assert len(render_lesson(lesson1)) + len(render_lesson(lesson2)) > CONTEXT_BUDGET_CHARS

    result = retrieve(store, UNIT, FAMILY)
    assert result.item_ids == (lesson1.item_id,)
    assert len(result.block) <= CONTEXT_BUDGET_CHARS
    assert "### A prior working version of this module" not in result.block
    assert render_lesson(lesson2) not in result.block
    store.close()


def test_budget_oversized_first_lesson_yields_none_block(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    huge = _padded_semantic("ep-huge", CONTEXT_BUDGET_CHARS + 200)
    store.write_semantic(huge)
    assert len(render_lesson(huge)) > CONTEXT_BUDGET_CHARS

    result = retrieve(store, UNIT, FAMILY)
    assert result == RetrievedBlock(None, ())
    store.close()


def test_determinism_two_calls_are_equal(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    lesson1 = _semantic("ep-1", last_verified_at="2026-08-24T09:00:00Z", confidence=0.9)
    lesson2 = _semantic("ep-2", last_verified_at="2026-08-23T09:00:00Z", confidence=0.4)
    store.write_semantic(lesson1)
    store.write_semantic(lesson2)
    store.write_episode(_episode("tk-1"))
    first = retrieve(store, UNIT, FAMILY)
    second = retrieve(store, UNIT, FAMILY)
    assert first == second
    store.close()
