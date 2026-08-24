"""Tests for the memory organ's SQLite store: typed round-trips, honest retrieval, the
sleep-trigger counters, and the R-S3-1 procedural non-write pin.

Every fixture is built through the two small helpers below (``_episode``/``_semantic``)
rather than nine-to-thirteen inline keyword calls per test -- unlike ``test_schema.py``,
which pins the dataclasses' own field *order* and so writes every fixture out literally,
this file is exercising the *store*, not the dataclass shape, and the field order is
already pinned there. The same positional-helper pattern is used by
``tests/stream/test_store.py``'s ``_unit``/``_mutant``/``_manifest``. Both helpers mint
``item_id`` through ``content_id``, using the "episode"/"semantic" kind strings Task 1's
note pinned as this task's to keep consistent.

Three things here are load-bearing, not incidental.

*Honest storage.* ``mark_falsified`` followed by ``semantic_for`` must still return the
falsified item -- filtering falsified items out is the retriever's job (a later task),
not the store's. ``test_mark_falsified_item_is_still_returned_by_semantic_for`` is the
one test that would fail if the store started silently hiding falsified rows.

*``verified_only`` really filters.* The mutation check (see the task report) breaks this
by dropping the ``WHERE`` clause in ``episodes()``; only
``test_episodes_verified_only_filters_to_verified`` catches it.

*Re-opening an existing db file doesn't lose or duplicate data.* ``MemoryStore.__init__``
uses ``CREATE TABLE IF NOT EXISTS``, so a second open of the same path must see the first
open's rows, not an empty table and not a schema error.
"""

from pathlib import Path

from crucible.memory.schema import EpisodicRecord, SemanticItem, content_id
from crucible.memory.store import MemoryStore


def _episode(task_key: str, arm: str, unit_id: str = "X/0", family: str = "ARITH", *,
             verified: bool = True, hidden_pass: bool | None = True,
             landed_module: str | None = "def f():\n    return 1\n") -> EpisodicRecord:
    item_id = content_id("episode", {"task_key": task_key, "arm": arm})
    return EpisodicRecord(
        item_id=item_id, task_key=task_key, arm=arm, unit_id=unit_id, family=family,
        class_id=f"{unit_id}|{family}", phase=1, kind="first",
        root_prompt="Fix the bug.", landed_module=landed_module, visible_reward=1.0,
        executions_charged=2, hidden_pass=hidden_pass, verified=verified,
        memory_item_ids=(), created_at="2026-08-24T10:00:00Z", confidence=0.8,
        status="active", version=1, source_locator=f"run:t/task:{task_key}",
        valid_at="2026-08-24T10:00:00Z", invalid_at=None, expired_at=None,
        last_verified_at=None, falsified_by=None, verification_method="hidden-suite",
    )


def _semantic(cited_episode_id: str, unit_id: str = "X/0", family: str = "ARITH") -> SemanticItem:
    item_id = content_id("semantic", {"cited_episode_id": cited_episode_id})
    return SemanticItem(
        item_id=item_id, unit_id=unit_id, family=family, class_id=f"{unit_id}|{family}",
        cited_episode_id=cited_episode_id, mutated_spans=(((2, 5), (2, 9)),),
        landed_diff="-    return 0\n+    return 1\n", flipped_tests=("test_v0",),
        killing_tests=("test_v0",), created_at="2026-08-24T10:06:00Z", confidence=0.75,
        status="active", version=1, source_locator=f"run:t/episode:{cited_episode_id}",
        valid_at="2026-08-24T10:06:00Z", invalid_at=None, expired_at=None,
        last_verified_at=None, falsified_by=None, verification_method="mechanical-template",
    )


def test_episode_round_trips_through_the_store(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    rec = _episode("tk-1", "A_full")
    store.write_episode(rec)
    assert store.episodes() == [rec]
    store.close()


def test_semantic_round_trips_through_the_store(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    item = _semantic("ep-1")
    store.write_semantic(item)
    assert store.semantic_for(item.unit_id, item.family) == [item]
    store.close()


def test_write_episode_is_insert_or_replace_by_item_id(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    rec = _episode("tk-1", "A_full", hidden_pass=False, verified=False)
    store.write_episode(rec)
    rewritten = _episode("tk-1", "A_full", hidden_pass=True, verified=True)
    assert rewritten.item_id == rec.item_id  # same identity fields -> same id
    store.write_episode(rewritten)
    rows = store.episodes()
    assert len(rows) == 1
    assert rows[0] == rewritten
    store.close()


def test_episodes_returns_insertion_order(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    a = _episode("tk-a", "A_full")
    b = _episode("tk-b", "A_full")
    c = _episode("tk-c", "A_full")
    for rec in (a, b, c):
        store.write_episode(rec)
    assert store.episodes() == [a, b, c]
    store.close()


def test_episodes_verified_only_filters_to_verified(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    verified = _episode("tk-1", "A_full", verified=True, hidden_pass=True)
    unverified = _episode("tk-2", "A_full", verified=False, hidden_pass=False)
    store.write_episode(verified)
    store.write_episode(unverified)
    assert store.episodes() == [verified, unverified]
    assert store.episodes(verified_only=True) == [verified]
    store.close()


def test_semantic_for_is_exact_class_match(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    exact = _semantic("ep-1", unit_id="X/0", family="ARITH")
    other_unit = _semantic("ep-2", unit_id="X/1", family="ARITH")
    other_family = _semantic("ep-3", unit_id="X/0", family="OFFBY1")
    for item in (exact, other_unit, other_family):
        store.write_semantic(item)
    assert store.semantic_for("X/0", "ARITH") == [exact]
    store.close()


def test_semantic_family_is_a_superset_of_semantic_for(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    exact = _semantic("ep-1", unit_id="X/0", family="ARITH")
    other_unit_same_family = _semantic("ep-2", unit_id="X/1", family="ARITH")
    other_family = _semantic("ep-3", unit_id="X/0", family="OFFBY1")
    for item in (exact, other_unit_same_family, other_family):
        store.write_semantic(item)
    family_results = store.semantic_family("ARITH")
    assert set(family_results) == {exact, other_unit_same_family}
    assert set(store.semantic_for("X/0", "ARITH")) <= set(family_results)
    store.close()


def test_mark_falsified_item_is_still_returned_by_semantic_for(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    item = _semantic("ep-1")
    store.write_semantic(item)
    store.mark_falsified(item.item_id, "re-run:tk-99 flipped back to failing")
    results = store.semantic_for(item.unit_id, item.family)
    assert len(results) == 1
    assert results[0].falsified_by == "re-run:tk-99 flipped back to failing"
    assert results[0].item_id == item.item_id
    # Every other field is untouched -- honest storage patches only what it was told to.
    assert results[0] == item.__class__.from_dict({**item.to_dict(), "falsified_by": "re-run:tk-99 flipped back to failing"})
    store.close()


def test_mark_falsified_on_an_episode_is_still_returned_by_episodes(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    rec = _episode("tk-1", "A_full")
    store.write_episode(rec)
    store.mark_falsified(rec.item_id, "hidden suite re-run failed")
    rows = store.episodes()
    assert len(rows) == 1
    assert rows[0].falsified_by == "hidden suite re-run failed"
    store.close()


def test_mark_verified_sets_last_verified_at(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    item = _semantic("ep-1")
    store.write_semantic(item)
    store.mark_verified(item.item_id, "2026-08-25T09:00:00Z")
    results = store.semantic_for(item.unit_id, item.family)
    assert results[0].last_verified_at == "2026-08-25T09:00:00Z"
    store.close()


def test_mark_falsified_unknown_item_id_raises(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    try:
        store.mark_falsified("no-such-id", "reason")
        assert False, "expected KeyError"
    except KeyError:
        pass
    store.close()


def test_reopening_an_existing_db_sees_the_prior_writes(tmp_path: Path):
    db_path = tmp_path / "mem.sqlite3"
    store1 = MemoryStore(db_path)
    rec = _episode("tk-1", "A_full")
    store1.write_episode(rec)
    store1.close()

    store2 = MemoryStore(db_path)  # idempotent re-open: no error, tables not clobbered
    assert store2.episodes() == [rec]
    store2.close()


def test_verified_count_and_count_verified_since(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    store.write_episode(_episode("tk-1", "A_full", verified=True))
    store.write_episode(_episode("tk-2", "A_full", verified=True))
    store.write_episode(_episode("tk-3", "A_full", verified=False, hidden_pass=False))
    assert store.verified_count() == 2

    marker = store.verified_count()
    store.write_episode(_episode("tk-4", "A_full", verified=True))
    assert store.count_verified_since(marker) == 1
    assert store.count_verified_since(0) == 3
    # A marker at or beyond the current count never goes negative.
    assert store.count_verified_since(1000) == 0
    store.close()


def test_procedural_table_exists(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    row = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='procedural'"
    ).fetchone()
    assert row is not None
    store.close()


def test_write_procedural_method_does_not_exist(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    assert not hasattr(store, "write_procedural")
    store.close()
