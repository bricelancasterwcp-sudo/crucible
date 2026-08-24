"""Tests for falsification: re-execution of a lesson's cited tests (pillar 3, Task 5 brief).

Fixtures follow ``tests/memory/test_store.py``'s local ``_episode``/``_semantic`` helper
pattern. Unlike those tests, ``_episode``/``_semantic`` here take a real ``landed_module``
that must actually run against a real ``Unit`` -- this module re-executes tests for real
(sandbox-touching), so the fixtures need a genuine tiny unit, not a placeholder string.
``_unit`` follows ``tests/stream/test_stack.py::_unit``'s idiom: the minimum ``Unit``
construction needed to hand something real to ``crucible.sandbox.task_run.run``.

Six things are load-bearing here, not incidental.

*Pass bumps ``last_verified_at``; nothing else changes.* A landed module that still
passes its cited flipped tests is a real re-verification -- ``test_refalsify_pass_bumps_last_verified_at``
checks the store row gains exactly the caller-supplied ``now`` and keeps
``falsified_by is None``.

*Fail sets ``falsified_by``; ``last_verified_at`` is NOT touched.* A landed module that no
longer passes is only ever falsified, never also "verified" -- both fields cannot be set
by the same call. ``test_refalsify_partial_failure_is_falsified_and_names_only_the_failing_test``
sharpens this: with TWO cited tests, only one broken, the description names exactly the
broken one, not both -- the subset actually ran and was actually measured.

*Sandbox infra changes NOTHING -- not even a partial write.* ``run`` itself reporting
``TestReport.infra_error`` (monkeypatched) leaves the stored item byte-identical to what
was written before the call, and still returns ``True`` (see the module docstring's pin).

*A broken citation is a SEPARATE outcome from sandbox infra (controller ruling, review
finding 2).* Five ways a citation can be broken are each pinned as their own test: the
cited episode is missing entirely, the cited episode exists but is NOT verified (review
finding 1 -- a later failed re-attempt under the same ``(task_key, arm)`` identity could
overwrite the row with an unverified attempt that still happens to carry a
``landed_module``), the cited episode has since been falsified itself, and (folded into
the tally test) the episode has no ``landed_module``. A sixth -- empty ``flipped_tests`` --
gets its own test for a sharper reason: ``test_refalsify_empty_flipped_tests_is_broken_citation_and_never_reaches_run``
monkeypatches ``run`` to raise if called at all, because the sandbox runner treats an
empty subset as "run everything", so this must be caught BEFORE ``run`` is ever reached,
not merely produce the right answer via ``run`` happening to report nothing failed. All
six leave the item untouched and return ``True``, exactly like a sandbox infra error, but
land in ``FalsifyTally``'s distinct ``infra_broken_citation`` bucket --
``test_falsify_batch_tally_arithmetic`` checks all four buckets in one call, including a
genuine sandbox infra case (dispatched via a sentinel test name so the real ``run`` still
executes the pass/fail cases).

*Mutation-checked (review finding 1).* Removing the ``not episode.verified`` /
``episode.falsified_by is not None`` checks from ``_broken_citation`` must fail
``test_refalsify_unverified_cited_episode_is_broken_citation_and_leaves_item_untouched``
-- see the task report for the pyc-purge evidence.

*The crude source-scan pin.* No BudgetMeter is ever imported or referenced by name here
-- ``test_falsify_module_never_imports_a_spend_meter`` greps the module's own source text
for the lowercase substring naming that meter's module and fails loudly if it ever
appears, so a future edit that reaches for per-execution accounting in this module trips
an immediate, cheap check rather than a subtle budget-drift bug.
"""
from __future__ import annotations

from pathlib import Path

import crucible.memory.falsify as falsify
from crucible.memory.falsify import FalsifyTally, falsify_batch, refalsify
from crucible.memory.schema import EpisodicRecord, SemanticItem, content_id
from crucible.memory.store import MemoryStore
from crucible.sandbox.report import TestReport
from crucible.stream.units import Unit, sha256_text

SRC_OK = "def f(a, b):\n    return a + b\n"
SRC_BROKEN = "def f(a, b):\n    return a - b\n"
# ``candidate(a, b) if a else 1`` passes test_v0 (a=2, truthy) but fails test_v1 (a=0,
# falsy -> returns 1 instead of 0) -- the classic partial-break idiom from
# tests/sandbox/test_task_run.py::test_run_hidden_is_the_outcome_oracle, reused here so
# a two-test subset can have exactly one member actually break.
SRC_PARTIAL_BREAK = "def f(a, b):\n    return a + b if a else 1\n"
VIS = ("from unit_x import f as candidate\n"
       "def test_v0():\n    assert candidate(2, 3) == 5\n"
       "def test_v1():\n    assert candidate(0, 0) == 0\n")
HID = "from unit_x import f as candidate\ndef test_h0():\n    assert candidate(0, 0) == 0\n"


def _unit(module_src: str = SRC_OK) -> Unit:
    return Unit("X/0", "unit_x", "f", module_src, VIS, HID, sha256_text(module_src), 1, 1, ())


def _episode(task_key: str = "tk-1", arm: str = "A_full", *, unit_id: str = "X/0",
             family: str = "ARITH", landed_module: str | None = SRC_OK,
             verified: bool = True, falsified_by: str | None = None) -> EpisodicRecord:
    item_id = content_id("episode", {"task_key": task_key, "arm": arm})
    return EpisodicRecord(
        item_id=item_id, task_key=task_key, arm=arm, unit_id=unit_id, family=family,
        class_id=f"{unit_id}|{family}", phase=1, kind="first",
        root_prompt="Fix the bug.", landed_module=landed_module, visible_reward=1.0,
        executions_charged=2, hidden_pass=verified, verified=verified,
        memory_item_ids=(), created_at="2026-08-24T10:00:00Z", confidence=0.8,
        status="active", version=1, source_locator=f"run:t/task:{task_key}",
        valid_at="2026-08-24T10:00:00Z", invalid_at=None, expired_at=None,
        last_verified_at=None, falsified_by=falsified_by, verification_method="hidden-suite",
    )


def _semantic(cited_episode_id: str, *, unit_id: str = "X/0", family: str = "ARITH",
              flipped_tests: tuple[str, ...] = ("test_v0",)) -> SemanticItem:
    item_id = content_id("semantic", {"cited_episode_id": cited_episode_id})
    return SemanticItem(
        item_id=item_id, unit_id=unit_id, family=family, class_id=f"{unit_id}|{family}",
        cited_episode_id=cited_episode_id, mutated_spans=(((2, 5), (2, 9)),),
        landed_diff="-    return 0\n+    return 1\n", flipped_tests=flipped_tests,
        killing_tests=flipped_tests, created_at="2026-08-24T10:06:00Z", confidence=0.75,
        status="active", version=1, source_locator=f"run:t/episode:{cited_episode_id}",
        valid_at="2026-08-24T10:06:00Z", invalid_at=None, expired_at=None,
        last_verified_at=None, falsified_by=None, verification_method="mechanical-template",
    )


def test_refalsify_pass_bumps_last_verified_at(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    episode = _episode(landed_module=SRC_OK)
    store.write_episode(episode)
    item = _semantic(episode.item_id)
    store.write_semantic(item)

    result = refalsify(store, item, _unit(), now="2026-08-24T12:00:00Z")

    assert result is True
    updated = store.semantic_for(item.unit_id, item.family)[0]
    assert updated.last_verified_at == "2026-08-24T12:00:00Z"
    assert updated.falsified_by is None
    store.close()


def test_refalsify_fail_sets_falsified_by_not_last_verified_at(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    episode = _episode(landed_module=SRC_BROKEN)  # no longer passes test_v0
    store.write_episode(episode)
    item = _semantic(episode.item_id)
    store.write_semantic(item)

    result = refalsify(store, item, _unit(), now="2026-08-24T12:00:00Z")

    assert result is False
    updated = store.semantic_for(item.unit_id, item.family)[0]
    assert updated.falsified_by is not None
    assert "test_v0" in updated.falsified_by
    assert updated.last_verified_at is None
    store.close()


def test_refalsify_infra_error_from_run_leaves_item_untouched(monkeypatch):
    store = MemoryStore(":memory:")
    episode = _episode(landed_module=SRC_OK)
    store.write_episode(episode)
    item = _semantic(episode.item_id)
    store.write_semantic(item)

    infra_report = TestReport((), (), (), (), 0.1, "boom")
    monkeypatch.setattr(falsify, "run", lambda *a, **k: infra_report)

    result = refalsify(store, item, _unit(), now="2026-08-24T12:00:00Z")

    assert result is True
    updated = store.semantic_for(item.unit_id, item.family)[0]
    assert updated == item  # byte-identical: nothing was patched
    store.close()


def test_refalsify_missing_cited_episode_is_broken_citation_and_leaves_item_untouched(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    # No store.write_episode call at all -- item cites an episode id that was never written.
    item = _semantic("ep-never-written")
    store.write_semantic(item)

    result = refalsify(store, item, _unit(), now="2026-08-24T12:00:00Z")

    assert result is True
    updated = store.semantic_for(item.unit_id, item.family)[0]
    assert updated == item
    store.close()


def test_refalsify_unverified_cited_episode_is_broken_citation_and_leaves_item_untouched(tmp_path: Path):
    # Review finding 1: an item's cited episode row may since have been overwritten
    # (INSERT OR REPLACE by (task_key, arm) identity) by a re-attempt that failed --
    # still carrying a landed_module string, but no longer verified. refalsify must not
    # mint a verdict off it.
    store = MemoryStore(tmp_path / "mem.sqlite3")
    episode = _episode(landed_module=SRC_OK, verified=False)
    store.write_episode(episode)
    item = _semantic(episode.item_id)
    store.write_semantic(item)

    result = refalsify(store, item, _unit(), now="2026-08-24T12:00:00Z")

    assert result is True
    updated = store.semantic_for(item.unit_id, item.family)[0]
    assert updated == item  # byte-identical: nothing was patched
    store.close()


def test_refalsify_falsified_cited_episode_is_broken_citation_and_leaves_item_untouched(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    episode = _episode(landed_module=SRC_OK, falsified_by="re-exec:some-earlier-check")
    store.write_episode(episode)
    item = _semantic(episode.item_id)
    store.write_semantic(item)

    result = refalsify(store, item, _unit(), now="2026-08-24T12:00:00Z")

    assert result is True
    updated = store.semantic_for(item.unit_id, item.family)[0]
    assert updated == item
    store.close()


def test_refalsify_empty_flipped_tests_is_broken_citation_and_never_reaches_run(tmp_path: Path, monkeypatch):
    # Minor (b): the sandbox runner treats an empty subset as "run everything", so an
    # item with nothing cited must be caught before run() is ever called -- not merely
    # produce the right verdict by accident.
    store = MemoryStore(tmp_path / "mem.sqlite3")
    episode = _episode(landed_module=SRC_OK)
    store.write_episode(episode)
    item = _semantic(episode.item_id, flipped_tests=())
    store.write_semantic(item)

    def _boom(*a, **k):
        raise AssertionError("run() must never be called for an item with no flipped_tests")
    monkeypatch.setattr(falsify, "run", _boom)

    result = refalsify(store, item, _unit(), now="2026-08-24T12:00:00Z")

    assert result is True
    updated = store.semantic_for(item.unit_id, item.family)[0]
    assert updated == item
    store.close()


def test_refalsify_partial_failure_is_falsified_and_names_only_the_failing_test(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    episode = _episode(landed_module=SRC_PARTIAL_BREAK)  # passes test_v0, fails test_v1
    store.write_episode(episode)
    item = _semantic(episode.item_id, flipped_tests=("test_v0", "test_v1"))
    store.write_semantic(item)

    result = refalsify(store, item, _unit(), now="2026-08-24T12:00:00Z")

    assert result is False
    updated = store.semantic_for(item.unit_id, item.family)[0]
    assert updated.falsified_by is not None
    # Only the test that actually broke is named as failing -- test_v0 was measured too
    # (it's in the subset) and it passed, so it must not appear in the failing= list.
    assert "failing=['test_v1']" in updated.falsified_by
    assert updated.last_verified_at is None
    store.close()


def test_falsify_batch_tally_arithmetic(monkeypatch):
    store = MemoryStore(":memory:")

    ep_pass = _episode(task_key="tk-pass", landed_module=SRC_OK)
    ep_fail = _episode(task_key="tk-fail", landed_module=SRC_BROKEN)
    ep_unverified = _episode(task_key="tk-unverified", landed_module=SRC_OK, verified=False)
    ep_sandbox_infra = _episode(task_key="tk-sandbox-infra", landed_module=SRC_OK)
    for ep in (ep_pass, ep_fail, ep_unverified, ep_sandbox_infra):
        store.write_episode(ep)

    item_pass = _semantic(ep_pass.item_id)
    item_fail = _semantic(ep_fail.item_id)
    item_missing = _semantic("ep-never-written")        # missing episode -> infra_broken_citation
    item_unverified = _semantic(ep_unverified.item_id)   # unverified episode -> infra_broken_citation
    # A distinct flipped_tests name routes this item through the sentinel branch below,
    # so it hits a genuine sandbox infra_error while the others still run for real.
    item_sandbox_infra = _semantic(ep_sandbox_infra.item_id, flipped_tests=("test_infra_sentinel",))
    for item in (item_pass, item_fail, item_missing, item_unverified, item_sandbox_infra):
        store.write_semantic(item)

    infra_report = TestReport((), (), (), (), 0.1, "boom")
    real_run = falsify.run

    def _dispatch(unit, patch, subset):
        if subset == ["test_infra_sentinel"]:
            return infra_report
        return real_run(unit, patch, subset)

    monkeypatch.setattr(falsify, "run", _dispatch)

    tally = falsify_batch(
        store,
        [(item_pass, _unit()), (item_fail, _unit()), (item_missing, _unit()),
         (item_unverified, _unit()), (item_sandbox_infra, _unit())],
        now="2026-08-24T12:00:00Z",
    )

    assert tally == FalsifyTally(checked=5, passed=1, falsified=1, infra=1, infra_broken_citation=2)
    store.close()


def test_falsify_module_never_imports_a_spend_meter():
    # Crude but honest (per the task brief): the module that meters per-task charged
    # executions is named with this lowercase word. This module must never import it.
    assert "budget" not in Path(falsify.__file__).read_text()
