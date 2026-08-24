"""Tests for the mechanical distiller: verified episode -> SemanticItem + the one fixed
lesson template (R-S3-2).

Fixtures follow ``tests/memory/test_store.py``'s ``_episode``/``_semantic`` helper
pattern (a local positional-helper factory) rather than ``test_schema.py``'s literal
style, since this file exercises ``distill``/``render_lesson``, not the dataclass field
order those tests already pin.

Three things are load-bearing here, not incidental.

*The guard is the whole point of R-S3-2.* ``distill`` must refuse a non-verified
episode and an episode with no ``landed_module`` -- these are the two failure modes a
caller could otherwise use to mint a lesson from an attempt that never actually fixed
anything. ``test_distill_refuses_a_non_verified_episode`` is the brief's named mutation
target (see the task report for the pyc-purge evidence of dropping this guard).

*The landed diff is against the MUTATED source, not the original.* ``mutated_src`` (the
bug the agent saw) plays the ``a`` side of the diff; ``episode.landed_module`` (the fix
that passed re-execution) plays the ``b`` side -- mirroring ``stream/mutants.py``'s
``_unified`` idiom (fixed ``a/<module>.py -> b/<module>.py`` headers, no dates).

*``render_lesson`` is byte-stable.* It is a pure function of the item -- no clock, no
randomness -- so two calls on the same item must produce identical bytes.
"""

from crucible.memory.distill import LESSON_TEMPLATE, distill, render_lesson
from crucible.memory.schema import EpisodicRecord, SemanticItem, content_id


def _episode(task_key: str = "tk-1", arm: str = "A_full", *, unit_id: str = "HumanEval/12",
             family: str = "ARITH", verified: bool = True,
             landed_module: str | None = "def f(a, b):\n    return a + b\n") -> EpisodicRecord:
    item_id = content_id("episode", {"task_key": task_key, "arm": arm})
    return EpisodicRecord(
        item_id=item_id, task_key=task_key, arm=arm, unit_id=unit_id, family=family,
        class_id=f"{unit_id}|{family}", phase=1, kind="first",
        root_prompt="Fix the bug in unit_humaneval_12.f so tests pass.",
        landed_module=landed_module, visible_reward=1.0, executions_charged=3,
        hidden_pass=True if verified else False, verified=verified,
        memory_item_ids=(), created_at="2026-08-24T10:00:00Z", confidence=0.8,
        status="active", version=1, source_locator=f"run:t/task:{task_key}",
        valid_at="2026-08-24T10:00:00Z", invalid_at=None, expired_at=None,
        last_verified_at=None, falsified_by=None, verification_method="hidden-suite",
    )


MUTATED_SRC = "def f(a, b):\n    return a - b\n"
LANDED_MODULE = "def f(a, b):\n    return a + b\n"
SPANS = (((2, 5), (2, 9)),)
FLIPPED_TESTS = ("test_v0", "test_v1")
KILLING_TESTS = ("test_v0",)
NOW = "2026-08-24T10:06:00Z"


def _distilled() -> SemanticItem:
    episode = _episode(landed_module=LANDED_MODULE)
    return distill(episode, mutated_src=MUTATED_SRC, spans=SPANS,
                    flipped_tests=FLIPPED_TESTS, killing_tests=KILLING_TESTS, now=NOW)


def test_distill_cites_the_episode_and_carries_its_class():
    episode = _episode(landed_module=LANDED_MODULE)
    item = distill(episode, mutated_src=MUTATED_SRC, spans=SPANS,
                    flipped_tests=FLIPPED_TESTS, killing_tests=KILLING_TESTS, now=NOW)
    assert isinstance(item, SemanticItem)
    assert item.cited_episode_id == episode.item_id
    assert item.item_id == content_id("semantic", {"cited_episode_id": episode.item_id})
    assert item.unit_id == episode.unit_id
    assert item.family == episode.family
    assert item.class_id == episode.class_id
    assert item.mutated_spans == SPANS
    assert item.flipped_tests == FLIPPED_TESTS
    assert item.killing_tests == KILLING_TESTS
    assert item.verification_method == "mechanical-template"
    assert item.falsified_by is None
    assert item.last_verified_at is None


def test_distill_landed_diff_is_computed_against_the_mutated_source():
    item = _distilled()
    # a-side is the bug (mutated_src), b-side is the fix (landed_module) -- a known
    # one-line change, so the diff must contain exactly this - and + pair.
    assert "-    return a - b" in item.landed_diff
    assert "+    return a + b" in item.landed_diff
    assert "a/unit_humaneval_12.py" in item.landed_diff
    assert "b/unit_humaneval_12.py" in item.landed_diff
    # No dates: mirrors mutants._unified's fixed-header, no-timestamp discipline.
    assert "+++" in item.landed_diff and "\t" not in item.landed_diff


def test_distill_refuses_a_non_verified_episode():
    episode = _episode(verified=False, landed_module=LANDED_MODULE)
    try:
        distill(episode, mutated_src=MUTATED_SRC, spans=SPANS,
                flipped_tests=FLIPPED_TESTS, killing_tests=KILLING_TESTS, now=NOW)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_distill_refuses_an_episode_with_no_landed_module():
    episode = _episode(verified=True, landed_module=None)
    try:
        distill(episode, mutated_src=MUTATED_SRC, spans=SPANS,
                flipped_tests=FLIPPED_TESTS, killing_tests=KILLING_TESTS, now=NOW)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_render_lesson_contains_family_diff_fence_and_flipped_tests():
    item = _distilled()
    rendered = render_lesson(item)
    assert "family ARITH" in rendered
    assert "```diff" in rendered
    assert item.landed_diff in rendered
    assert "test_v0, test_v1" in rendered


def test_render_lesson_is_byte_stable_for_identical_items():
    item = _distilled()
    first = render_lesson(item)
    second = render_lesson(item)
    assert first == second


def test_render_lesson_matches_the_pinned_template_exactly():
    item = _distilled()
    expected = LESSON_TEMPLATE.format(
        family="ARITH",
        spans="[[[2, 5], [2, 9]]]",
        landed_diff=item.landed_diff,
        flipped_tests="test_v0, test_v1",
    )
    assert render_lesson(item) == expected


def test_lesson_template_is_the_module_constant_spec_locked_text():
    # Pinned literally (not recomputed) so an accidental edit to the wording is caught
    # here even if render_lesson's own tests happen to still pass.
    assert LESSON_TEMPLATE == (
        "### Prior verified fix in this code (family {family})\n"
        "The altered region was at spans {spans}. The repair that passed re-execution:\n"
        "```diff\n"
        "{landed_diff}\n"
        "```\n"
        "Visible tests that flipped from failing to passing: {flipped_tests}."
    )
