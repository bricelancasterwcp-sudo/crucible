"""Tests for the tau calibration script (Phase-C prereg §4.4/§7, task-6 brief).

This script's output becomes part of the experiment's LOCK-C record, so the two things
this file exists to prove are: (1) the self-citation exclusion actually removes pairs
from the count (not just "sanity happens to still say PASS" -- a self-citation pair that
leaks through would inflate ``n_related`` specifically, so the count itself is the proof),
and (2) the p95/p50 percentile index is the one actually wired in (a PASS/FAIL sanity
verdict alone can't tell a correct p95 lookup from an accidental p50 one -- both indices
land below the single related score in this fixture -- so the test recomputes the exact
expected percentile independently, using the same ``symmatch`` functions the script
itself calls, and asserts byte-for-byte equality against ``result["tau_p95_unrelated"]``).

Fixtures follow ``tests/memory/test_retrieve.py``'s local ``_episode``/``_semantic``
helper pattern. Two units x two families, one DB, four (episode, lesson) pairs total:

- (E1, lesson_self) -- SELF-CITATION (``lesson_self.cited_episode_id == E1.item_id``):
  EXCLUDED. If this exclusion were dropped, the pair would classify as RELATED (same
  class as E1, since lesson_self was minted "from" E1), so a missing-exclusion bug would
  observably change ``n_related`` from 1 to 2 -- the assertion on the exact count is what
  proves the exclusion fired, not an assumption about it.
- (E1, lesson_orphan) -- RELATED (``class_id`` match). ``lesson_orphan`` cites an episode
  id that resolves to nothing in this store, exercising the brief's None-safe clause
  ("the LESSON side uses its OWN cited episode's symptom... None-safe -> \"\"") for real.
- (E2, lesson_self) -- UNRELATED (unit AND family both differ).
- (E2, lesson_orphan) -- UNRELATED (unit AND family both differ).

``unit_src_for`` is a stub dict lookup (the brief's injected-loader seam), never a real
stream read -- the CLI wiring to ``crucible.stream.store.read_unit`` is exercised only by
reading the source, not by this test.
"""
from __future__ import annotations

import statistics

import pytest

from crucible.memory.schema import EpisodicRecord, SemanticItem, content_id
from crucible.memory.symmatch import lesson_text, query_text, score, symptom_section, tokenize
from scripts.calibrate_tau import calibrate

UNIT_A, FAMILY_A = "X/0", "ARITH"
UNIT_B, FAMILY_B = "X/1", "OFFBY1"

UNIT_SRC_A = "def solve_a():\n    return fluxcap_overflow_marker\n"
UNIT_SRC_B = "def solve_b():\n    return zeta_unrelated_glitch\n"

E1_ROOT_PROMPT = "## Module under repair\ndef solve(): pass\n\n## Symptom\nfluxcap_overflow_marker\n"
E2_ROOT_PROMPT = "## Module under repair\ndef other(): pass\n\n## Symptom\nzeta_unrelated_glitch\n"

# lesson_self's landed_diff deliberately also contains "zeta_unrelated_glitch" (a stray
# comment token) so its unrelated-pair score differs from lesson_orphan's -- two distinct
# unrelated scores are required for p50 and p95 to actually differ, which is what makes
# the p95-vs-p50 mutant killable at all (see module docstring).
LESSON_SELF_DIFF = "-    return 0\n+    return fluxcap_overflow_marker\n# zeta_unrelated_glitch\n"
LESSON_ORPHAN_DIFF = "-    return 0\n+    return fluxcap_overflow_marker\n"

UNIT_SRC_MAP = {UNIT_A: UNIT_SRC_A, UNIT_B: UNIT_SRC_B}


def _unit_src_for(unit_id: str) -> str:
    return UNIT_SRC_MAP[unit_id]


def _episode(task_key: str, *, unit_id: str, family: str, root_prompt: str) -> EpisodicRecord:
    item_id = content_id("episode", {"task_key": task_key, "arm": "A_full"})
    return EpisodicRecord(
        item_id=item_id, task_key=task_key, arm="A_full", unit_id=unit_id, family=family,
        class_id=f"{unit_id}|{family}", phase=1, kind="first",
        root_prompt=root_prompt, landed_module="def f():\n    return 1\n", visible_reward=1.0,
        executions_charged=2, hidden_pass=True, verified=True,
        memory_item_ids=(), created_at="2026-08-24T10:00:00Z", confidence=0.8,
        status="active", version=1, source_locator=f"run:t/task:{task_key}",
        valid_at="2026-08-24T10:00:00Z", invalid_at=None, expired_at=None,
        last_verified_at=None, falsified_by=None, verification_method="hidden-suite",
    )


def _semantic(cited_episode_id: str, *, unit_id: str, family: str, landed_diff: str) -> SemanticItem:
    item_id = content_id("semantic", {"cited_episode_id": cited_episode_id})
    return SemanticItem(
        item_id=item_id, unit_id=unit_id, family=family, class_id=f"{unit_id}|{family}",
        cited_episode_id=cited_episode_id, mutated_spans=(((2, 5), (2, 9)),),
        landed_diff=landed_diff, flipped_tests=("test_v0",), killing_tests=("test_v0",),
        created_at="2026-08-24T10:06:00Z", confidence=0.75,
        status="active", version=1, source_locator=f"run:t/episode:{cited_episode_id}",
        valid_at="2026-08-24T10:06:00Z", invalid_at=None, expired_at=None,
        last_verified_at=None, falsified_by=None, verification_method="mechanical-template",
    )


def _seed_store(tmp_path):
    from crucible.memory.store import MemoryStore

    store = MemoryStore(tmp_path / "mem.sqlite3")
    e1 = _episode("tk-1", unit_id=UNIT_A, family=FAMILY_A, root_prompt=E1_ROOT_PROMPT)
    e2 = _episode("tk-2", unit_id=UNIT_B, family=FAMILY_B, root_prompt=E2_ROOT_PROMPT)
    store.write_episode(e1)
    store.write_episode(e2)

    lesson_self = _semantic(e1.item_id, unit_id=UNIT_A, family=FAMILY_A, landed_diff=LESSON_SELF_DIFF)
    lesson_orphan = _semantic("ep-external-orphan-001", unit_id=UNIT_A, family=FAMILY_A,
                              landed_diff=LESSON_ORPHAN_DIFF)
    store.write_semantic(lesson_self)
    store.write_semantic(lesson_orphan)
    store.close()
    return e1, e2, lesson_self, lesson_orphan


def _expected_scores():
    """The exact scores the four pairs must produce, computed with the SAME symmatch
    functions the script calls -- independent of ``calibrate``'s own glue code, so this
    is a real cross-check, not a restatement of the implementation."""
    q1 = tokenize(query_text(UNIT_SRC_A, symptom_section(E1_ROOT_PROMPT), FAMILY_A))
    q2 = tokenize(query_text(UNIT_SRC_B, symptom_section(E2_ROOT_PROMPT), FAMILY_B))
    # lesson_self's OWN cited episode is E1 -> its symptom is E1's, not "".
    l_self = tokenize(lesson_text(_semantic("e1-placeholder", unit_id=UNIT_A, family=FAMILY_A,
                                            landed_diff=LESSON_SELF_DIFF),
                                  symptom_section(E1_ROOT_PROMPT)))
    # lesson_orphan's OWN cited episode does not resolve -> "" (None-safe).
    l_orphan = tokenize(lesson_text(_semantic("orphan-placeholder", unit_id=UNIT_A, family=FAMILY_A,
                                              landed_diff=LESSON_ORPHAN_DIFF), ""))
    related = [score(q1, l_orphan)]              # (E1, lesson_orphan)
    unrelated = [score(q2, l_self), score(q2, l_orphan)]  # (E2, lesson_self), (E2, lesson_orphan)
    return related, unrelated


def test_calibrate_excludes_self_citation_and_passes_sanity_with_exact_percentile(tmp_path):
    e1, e2, lesson_self, lesson_orphan = _seed_store(tmp_path)
    db_path = str(tmp_path / "mem.sqlite3")

    result = calibrate([db_path], _unit_src_for)

    related, unrelated = _expected_scores()
    expected_tau_p95 = statistics.quantiles(sorted(unrelated), n=100)[94]
    expected_tau_p50 = statistics.quantiles(sorted(unrelated), n=100)[49]
    expected_median = statistics.median(related)

    # Exact counts: 4 total pairs minus 1 self-citation = 3 classifiable pairs, split
    # 1 related / 2 unrelated. A missing self-citation exclusion would make
    # (E1, lesson_self) count as RELATED too (same class as E1) -- n_related would be 2,
    # not 1 -- so this assertion is the proof the exclusion actually fired.
    assert result["n_related"] == 1
    assert result["n_unrelated"] == 2

    # p95, not p50 (or any other index): exact equality against an independently
    # computed value pins the index, since sanity alone would PASS under either.
    assert result["tau_p95_unrelated"] == pytest.approx(expected_tau_p95)
    assert expected_tau_p95 != expected_tau_p50  # the fixture actually distinguishes them
    assert result["tau_p95_unrelated"] != pytest.approx(expected_tau_p50)

    assert result["median_related"] == pytest.approx(expected_median)
    assert result["sanity"] == "PASS"
    assert "reason" not in result

    # n_related == 1 < 2: quantile-based related_summary is guarded to nulls, never a
    # fabricated number from a single-point "percentile".
    assert result["related_summary"] == {"p50": None, "p90": None, "p99": None}
    assert result["unrelated_summary"]["p50"] == pytest.approx(
        statistics.quantiles(sorted(unrelated), n=100)[49]
    )


def test_calibrate_on_a_db_with_no_episodes_or_lessons_never_crashes(tmp_path):
    from crucible.memory.store import MemoryStore

    MemoryStore(tmp_path / "empty.sqlite3").close()
    result = calibrate([str(tmp_path / "empty.sqlite3")], _unit_src_for)

    assert result["n_related"] == 0
    assert result["n_unrelated"] == 0
    assert result["tau_p95_unrelated"] is None
    assert result["median_related"] is None
    assert result["unrelated_summary"] == {"p50": None, "p90": None, "p99": None}
    assert result["related_summary"] == {"p50": None, "p90": None, "p99": None}
    assert result["sanity"] == "FAIL"
    assert result["reason"]  # non-empty: names why, never a silent FAIL


def test_calibrate_iterates_dbs_in_argv_order_and_lessons_falsified_out(tmp_path):
    """Two dbs, second listed first in argv: pairs from db B alone can't produce the
    single related score this test checks for -- proving both dbs were actually visited
    and in the order given, not just the first one found. Also seeds a falsified lesson
    (semantic_all() still returns it -- honest storage) and asserts it contributes
    nothing, per the brief's ``falsified_by is None`` filter on lessons."""
    from crucible.memory.store import MemoryStore

    (tmp_path / "db_a").mkdir()
    e1, e2, lesson_self, lesson_orphan = _seed_store(tmp_path / "db_a")
    store_a = MemoryStore(tmp_path / "db_a" / "mem.sqlite3")
    falsified = _semantic("ep-falsified-orphan", unit_id=UNIT_A, family=FAMILY_A,
                          landed_diff=LESSON_ORPHAN_DIFF)
    store_a.write_semantic(falsified)
    store_a.mark_falsified(falsified.item_id, "re-run flipped back to failing")
    store_a.close()

    (tmp_path / "db_b").mkdir()
    MemoryStore(tmp_path / "db_b" / "mem.sqlite3").close()  # empty db, listed FIRST

    result = calibrate(
        [str(tmp_path / "db_b" / "mem.sqlite3"), str(tmp_path / "db_a" / "mem.sqlite3")],
        _unit_src_for,
    )

    assert result["n_related"] == 1
    assert result["n_unrelated"] == 2
    assert result["sanity"] == "PASS"
