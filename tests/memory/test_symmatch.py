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


def test_tokenize_drops_single_char_tokens_but_keeps_two_char_tokens():
    toks = tokenize("a + b == ab")
    assert "a" not in toks and "b" not in toks and "ab" in toks
    # MUTANT KILLED: deleting the `len(token) >= _MIN_TOKEN_LEN` filter (round 1, finding 1)


def test_score_is_binary_cosine_and_zero_on_empty():
    q, l = tokenize("alpha beta gamma"), tokenize("beta gamma delta")
    assert abs(score(q, l) - 2 / 3) < 1e-9  # 2 shared / sqrt(3*3)
    assert score(frozenset(), l) == 0.0
    # MUTANT KILLED: swapping intersection for union, or dropping the sqrt


def test_score_denominator_is_sqrt_of_product_not_max_of_sizes():
    # Deliberately UNEQUAL set sizes: every other fixture in this file uses equal-length
    # token sets, where sqrt(a*a) == max(a,a) and a max()-substituted denominator can't
    # be told apart from the real sqrt(|Q|*|L|) one. |Q|=2, |L|=8, sharing 2 tokens:
    # sqrt(2*8) = 4 -> 2/4 = 0.5, but max(2,8) = 8 would give 2/8 = 0.25 -- the two
    # formulas diverge only when the sizes differ.
    q = frozenset({"alpha", "beta"})
    l = frozenset({"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"})
    assert abs(score(q, l) - 0.5) < 1e-9
    # MUTANT KILLED: denominator sqrt(|Q|*|L|) -> max(|Q|,|L|) (round 2, finding 1)


def test_family_token_boosts_same_family_pairs():
    q = tokenize(query_text("return x", "boom", "ARITH"))
    same = tokenize(lesson_text_stub(diff="return y", symptom="boom", family="ARITH"))
    other = tokenize(lesson_text_stub(diff="return y", symptom="boom", family="SDL"))
    assert score(q, same) > score(q, other)


def test_rank_orders_higher_score_first():
    item_hi = _semantic_item(cited_episode_id="ep-hi", family="ARITH")
    item_lo = _semantic_item(cited_episode_id="ep-lo", family="ARITH")
    q = frozenset({"alpha", "beta"})
    hi_tokens = frozenset({"alpha", "beta"})  # 2 shared / sqrt(2*2) = 1.0
    lo_tokens = frozenset({"alpha", "gamma"})  # 1 shared / sqrt(2*2) = 0.5
    # Candidates fed in the WRONG (low-then-high) order, so a correct primary sort has
    # to actually reorder them -- an accidental pass-through of input order can't fake it.
    ranked = rank(q, [(item_lo, lo_tokens), (item_hi, hi_tokens)])
    assert ranked[0][1].item_id == item_hi.item_id
    assert ranked[0][0] > ranked[1][0]
    # MUTANT KILLED: flipping -score to +score in rank's sort key (round 1, finding 2)


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


def test_symptom_section_runs_to_end_of_string_when_no_trailing_header():
    # No section follows "## Symptom" here, so the extraction must fall through to the
    # "no next header" branch and take the rest of the string. A preceding section is
    # included (rather than starting the prompt at "## Symptom") because this
    # implementation matches the header as "\n## Symptom\n" -- it needs a real newline
    # in front of "##", the same shape a real root_prompt has (see
    # test_symptom_section_extracts_between_headers above), not a bare start-of-string.
    rp = "## Module under repair\nX\n\n## Symptom\nboom"
    assert symptom_section(rp) == "boom"
    # MUTANT KILLED: symptom_section's end-of-string branch, `next_header is None ->
    # content_end = len(root_prompt)` (round 1, finding 3)


def test_tau_matches_the_lock_record():
    """LOCK-C (prereg-lock-c): tau = P95 of the unrelated-pair score distribution over the
    four Phase-A/B databases = 0.8051 (calibrate_tau 2026-08-25: n_unrelated=292660,
    n_related=921, median_related=0.875, sanity PASS). A drifted constant here is a
    different experiment than the one locked."""
    from crucible.memory.symmatch import TAU
    assert TAU == 0.8051
