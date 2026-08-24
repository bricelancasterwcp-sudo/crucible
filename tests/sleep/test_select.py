"""Tests for sleep-cycle SFT selection: which verified episodes train the next adapter,
and the content hash that identifies that exact set (Task 9 brief).

Fixtures follow ``tests/memory/test_falsify.py``'s/``tests/memory/test_store.py``'s local
``_episode`` helper pattern -- ``content_id`` mints a real ``item_id`` from
``(task_key, arm)`` so every fixture episode carries the same kind of identity a real one
would; the helper exposes ``created_at``/``verified``/``landed_module``/``root_prompt``
per test.

Two things here are load-bearing, not incidental.

*Deterministic order is (created_at, item_id), not store (rowid/insertion) order.*
``test_sft_pairs_orders_by_created_at_not_insertion_order`` writes episodes to the store
in one sequence and asserts ``sft_pairs`` returns them in ``created_at`` order, which is
the REVERSE of write order in that fixture -- a function that forgot the explicit sort
and simply trusted ``store.episodes()``'s own row order would pass every other test here
but fail this one specifically.

*Verified-only is load-bearing, not just correct-by-accident.*
``test_sft_pairs_drops_unverified_episodes`` writes one verified and one unverified
episode and asserts the returned pair COUNT is exactly one, not merely that the verified
pair's content is present -- a mutant that dropped the ``verified_only=True`` filter
(``store.episodes()`` instead of ``store.episodes(verified_only=True)``) would still
produce a list containing the right verified pair, but with an extra one alongside it;
only the count assertion is sensitive to that specific mutation. See the task report for
the literal mutation-check evidence.
"""
from __future__ import annotations

import pytest

from crucible.memory.schema import EpisodicRecord, content_id
from crucible.memory.store import MemoryStore
from crucible.sleep.select import episode_set_hash, sft_pairs


def _episode(task_key: str, *, arm: str = "A_full", created_at: str = "2026-08-24T10:00:00Z",
             verified: bool = True, landed_module: str | None = "def f():\n    return 1\n",
             root_prompt: str = "Fix the bug.") -> EpisodicRecord:
    item_id = content_id("episode", {"task_key": task_key, "arm": arm})
    return EpisodicRecord(
        item_id=item_id, task_key=task_key, arm=arm, unit_id="X/0", family="ARITH",
        class_id="X/0|ARITH", phase=1, kind="first",
        root_prompt=root_prompt, landed_module=landed_module, visible_reward=1.0,
        executions_charged=2, hidden_pass=verified, verified=verified,
        memory_item_ids=(), created_at=created_at, confidence=0.8,
        status="active", version=1, source_locator=f"run:t/task:{task_key}",
        valid_at=created_at, invalid_at=None, expired_at=None,
        last_verified_at=None, falsified_by=None, verification_method="hidden-suite",
    )


def test_sft_pairs_returns_root_prompt_landed_module_tuples(tmp_path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    ep = _episode("tk-1", root_prompt="Fix the off-by-one.", landed_module="def f(): return 2\n")
    store.write_episode(ep)

    pairs = sft_pairs(store)

    assert pairs == [("Fix the off-by-one.", "def f(): return 2\n")]
    store.close()


def test_sft_pairs_drops_unverified_episodes(tmp_path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    verified_ep = _episode("tk-verified", verified=True, landed_module="def f(): return 1\n")
    unverified_ep = _episode("tk-unverified", verified=False, landed_module="def f(): return 2\n")
    store.write_episode(verified_ep)
    store.write_episode(unverified_ep)

    pairs = sft_pairs(store)

    # Count, not just content: a filter-dropping mutant would still contain the verified
    # pair but with an extra unverified one alongside it.
    assert len(pairs) == 1
    assert pairs == [(verified_ep.root_prompt, verified_ep.landed_module)]
    store.close()


def test_sft_pairs_orders_by_created_at_not_insertion_order(tmp_path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    later = _episode("tk-later", created_at="2026-08-24T12:00:00Z", root_prompt="later", landed_module="m-later")
    earlier = _episode("tk-earlier", created_at="2026-08-24T09:00:00Z", root_prompt="earlier", landed_module="m-earlier")
    # Write the LATER episode first -- insertion (rowid) order is the reverse of created_at order.
    store.write_episode(later)
    store.write_episode(earlier)

    pairs = sft_pairs(store)

    assert pairs == [("earlier", "m-earlier"), ("later", "m-later")]
    store.close()


def test_sft_pairs_cumulative_across_all_verified_episodes(tmp_path):
    # R-S3-3: every verified episode ever, not just ones since a marker -- three
    # episodes "from different sleep cycles" all come back in one call.
    store = MemoryStore(tmp_path / "mem.sqlite3")
    for i, ts in enumerate(("2026-08-24T09:00:00Z", "2026-08-24T10:00:00Z", "2026-08-24T11:00:00Z")):
        store.write_episode(_episode(f"tk-{i}", created_at=ts, root_prompt=f"p{i}", landed_module=f"m{i}"))

    pairs = sft_pairs(store)

    assert pairs == [("p0", "m0"), ("p1", "m1"), ("p2", "m2")]
    store.close()


def test_sft_pairs_raises_on_verified_episode_with_landed_module_none(tmp_path):
    store = MemoryStore(tmp_path / "mem.sqlite3")
    # Defensive: cannot happen via episode_verified in practice, but the store itself
    # will accept a hand-constructed row that violates the invariant.
    broken = _episode("tk-broken", verified=True, landed_module=None)
    store.write_episode(broken)

    with pytest.raises(ValueError, match="landed_module"):
        sft_pairs(store)
    store.close()


def test_episode_set_hash_is_stable_for_equal_content():
    pairs_a = [("prompt one", "module one"), ("prompt two", "module two")]
    pairs_b = [("prompt one", "module one"), ("prompt two", "module two")]

    assert episode_set_hash(pairs_a) == episode_set_hash(pairs_b)


def test_episode_set_hash_changes_when_a_pair_changes():
    base = [("prompt one", "module one"), ("prompt two", "module two")]
    changed = [("prompt one", "module one"), ("prompt two", "module TWO CHANGED")]

    assert episode_set_hash(base) != episode_set_hash(changed)


def test_episode_set_hash_changes_when_a_pair_is_added():
    base = [("prompt one", "module one")]
    extended = [("prompt one", "module one"), ("prompt two", "module two")]

    assert episode_set_hash(base) != episode_set_hash(extended)


def test_episode_set_hash_is_sensitive_to_order():
    forward = [("a", "1"), ("b", "2")]
    backward = [("b", "2"), ("a", "1")]

    assert episode_set_hash(forward) != episode_set_hash(backward)


def test_episode_set_hash_of_empty_pairs_is_stable_and_sha256_shaped():
    assert episode_set_hash([]) == episode_set_hash([])
    assert len(episode_set_hash([])) == 64  # sha256 hex digest length
