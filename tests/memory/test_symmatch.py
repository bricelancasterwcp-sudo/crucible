"""Tests for the deterministic lexical scorer (Phase-C prereg §4.2, task-2 brief). Each
test below names the mutant it exists to kill.

Fixtures follow ``tests/memory/test_retrieve.py``'s local ``_semantic``-style helper
pattern: ``_semantic_item`` builds a minimal real ``SemanticItem`` (never a fake stand-in
class) so ``lesson_text`` runs against the actual dataclass shape it will see in
production. ``lesson_text_stub`` wraps that with the ``diff``/``symptom``/``family``
keywords the brief's own pseudo-call uses, so the scorer test reads the same as the brief.
"""
from __future__ import annotations

from crucible.memory.schema import SemanticItem, content_id
from crucible.memory.symmatch import (
    lesson_text,
    query_text,
    rank,
    score,
    symptom_section,
    tokenize,
)


def _semantic_item(cited_episode_id: str = "ep-stub", *, family: str = "ARITH",
                    landed_diff: str = "-    return 0\n+    return 1\n") -> SemanticItem:
    item_id = content_id("semantic", {"cited_episode_id": cited_episode_id})
    return SemanticItem(
        item_id=item_id, unit_id="X/0", family=family, class_id=f"X/0|{family}",
        cited_episode_id=cited_episode_id, mutated_spans=(((2, 5), (2, 9)),),
        landed_diff=landed_diff, flipped_tests=("test_v0",), killing_tests=("test_v0",),
        created_at="2026-08-24T10:06:00Z", confidence=0.75,
        status="active", version=1, source_locator=f"run:t/episode:{cited_episode_id}",
        valid_at="2026-08-24T10:06:00Z", invalid_at=None, expired_at=None,
        last_verified_at=None, falsified_by=None,
        verification_method="mechanical-template",
    )


def lesson_text_stub(*, diff: str, symptom: str, family: str) -> str:
    """Minimal real ``SemanticItem`` -> ``lesson_text``, per the brief's fixture note."""
    item = _semantic_item(family=family, landed_diff=diff)
    return lesson_text(item, symptom)


def test_tokenize_drops_unit_local_test_names_but_keeps_code_tokens():
    toks = tokenize("failed: test_v0, test_h1\nreturn a + b")
    assert "test_v0" not in toks and "test_h1" not in toks and "return" in toks
    # MUTANT KILLED: removing the test\w* filter


def test_score_is_binary_cosine_and_zero_on_empty():
    q, l = tokenize("alpha beta gamma"), tokenize("beta gamma delta")
    assert abs(score(q, l) - 2 / 3) < 1e-9  # 2 shared / sqrt(3*3)
    assert score(frozenset(), l) == 0.0
    # MUTANT KILLED: swapping intersection for union, or dropping the sqrt


def test_family_token_boosts_same_family_pairs():
    q = tokenize(query_text("return x", "boom", "ARITH"))
    same = tokenize(lesson_text_stub(diff="return y", symptom="boom", family="ARITH"))
    other = tokenize(lesson_text_stub(diff="return y", symptom="boom", family="SDL"))
    assert score(q, same) > score(q, other)


def test_rank_breaks_ties_by_item_id():
    item_x = _semantic_item(cited_episode_id="ep-x", family="ARITH")
    item_y = _semantic_item(cited_episode_id="ep-y", family="ARITH")
    ordered_ids = sorted([item_x.item_id, item_y.item_id])
    by_id = {item_x.item_id: item_x, item_y.item_id: item_y}
    # Feed candidates in DESCENDING item_id order: identical token sets mean an equal
    # score for both, so a stable sort with no tie-break would preserve THIS (wrong)
    # order instead of resolving to ascending item_id.
    ordered_candidates = [by_id[item_id] for item_id in reversed(ordered_ids)]
    tokens = frozenset({"alpha", "beta"})
    q = frozenset({"alpha"})
    ranked = rank(q, [(item, tokens) for item in ordered_candidates])
    assert [item.item_id for _, item in ranked] == ordered_ids
    # MUTANT KILLED: dropping the tie-break


def test_symptom_section_extracts_between_headers():
    rp = "## Module under repair\nX\n\n## Symptom\nfailed: t\nboom\n\n## Instruction\nY"
    assert symptom_section(rp) == "failed: t\nboom"
    assert symptom_section("no symptom here") == ""
