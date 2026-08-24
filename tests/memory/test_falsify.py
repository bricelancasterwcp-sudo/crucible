"""Tests for falsification: re-execution of a lesson's cited tests (pillar 3, Task 5 brief).

Fixtures follow ``tests/memory/test_store.py``'s local ``_episode``/``_semantic`` helper
pattern. Unlike those tests, ``_episode``/``_semantic`` here take a real ``landed_module``
that must actually run against a real ``Unit`` -- this module re-executes tests for real
(sandbox-touching), so the fixtures need a genuine tiny unit, not a placeholder string.
``_unit`` follows ``tests/stream/test_stack.py::_unit``'s idiom: the minimum ``Unit``
construction needed to hand something real to ``crucible.sandbox.task_run.run``.

Four things are load-bearing here, not incidental.

*Pass bumps ``last_verified_at``; nothing else changes.* A landed module that still
passes its cited flipped tests is a real re-verification -- ``test_refalsify_pass_bumps_last_verified_at``
checks the store row gains exactly the caller-supplied ``now`` and keeps
``falsified_by is None``.

*Fail sets ``falsified_by``; ``last_verified_at`` is NOT touched.* A landed module that no
longer passes is only ever falsified, never also "verified" -- both fields cannot be set
by the same call.

*Infra changes NOTHING -- not even a partial write.* Two infra sources are pinned
separately: (a) the cited episode is missing from the store entirely (identity contract:
an item may cite an episode a caller never wrote), and (b) ``run`` itself reports
``infra_error`` (monkeypatched). Both leave the stored item byte-identical to what was
written before the call, and both still return ``True`` (see the module docstring's pin).
This is the brief's named mutation target -- see the task report for the pyc-purge
evidence of flipping the infra branch to falsify.

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
VIS = ("from unit_x import f as candidate\n"
       "def test_v0():\n    assert candidate(2, 3) == 5\n")
HID = "from unit_x import f as candidate\ndef test_h0():\n    assert candidate(0, 0) == 0\n"


def _unit(module_src: str = SRC_OK) -> Unit:
    return Unit("X/0", "unit_x", "f", module_src, VIS, HID, sha256_text(module_src), 1, 1, ())


def _episode(task_key: str = "tk-1", arm: str = "A_full", *, unit_id: str = "X/0",
             family: str = "ARITH", landed_module: str | None = SRC_OK) -> EpisodicRecord:
    item_id = content_id("episode", {"task_key": task_key, "arm": arm})
    return EpisodicRecord(
        item_id=item_id, task_key=task_key, arm=arm, unit_id=unit_id, family=family,
        class_id=f"{unit_id}|{family}", phase=1, kind="first",
        root_prompt="Fix the bug.", landed_module=landed_module, visible_reward=1.0,
        executions_charged=2, hidden_pass=True, verified=True,
        memory_item_ids=(), created_at="2026-08-24T10:00:00Z", confidence=0.8,
        status="active", version=1, source_locator=f"run:t/task:{task_key}",
        valid_at="2026-08-24T10:00:00Z", invalid_at=None, expired_at=None,
        last_verified_at=None, falsified_by=None, verification_method="hidden-suite",
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


def test_refalsify_missing_cited_episode_is_infra_and_leaves_item_untouched(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    # No store.write_episode call at all -- item cites an episode id that was never written.
    item = _semantic("ep-never-written")
    store.write_semantic(item)

    result = refalsify(store, item, _unit(), now="2026-08-24T12:00:00Z")

    assert result is True
    updated = store.semantic_for(item.unit_id, item.family)[0]
    assert updated == item
    store.close()


def test_falsify_batch_tally_arithmetic(monkeypatch):
    store = MemoryStore(":memory:")

    ep_pass = _episode(task_key="tk-pass", landed_module=SRC_OK)
    ep_fail = _episode(task_key="tk-fail", landed_module=SRC_BROKEN)
    store.write_episode(ep_pass)
    store.write_episode(ep_fail)

    item_pass = _semantic(ep_pass.item_id)
    item_fail = _semantic(ep_fail.item_id)
    item_infra = _semantic("ep-never-written")  # missing episode -> infra
    for item in (item_pass, item_fail, item_infra):
        store.write_semantic(item)

    tally = falsify_batch(
        store,
        [(item_pass, _unit()), (item_fail, _unit()), (item_infra, _unit())],
        now="2026-08-24T12:00:00Z",
    )

    assert tally == FalsifyTally(checked=3, passed=1, falsified=1, infra=1)
    store.close()


def test_falsify_module_never_imports_a_spend_meter():
    # Crude but honest (per the task brief): the module that meters per-task charged
    # executions is named with this lowercase word. This module must never import it.
    assert "budget" not in Path(falsify.__file__).read_text()
