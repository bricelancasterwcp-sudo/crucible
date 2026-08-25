"""Tests for the sleep loop: trigger, refalsification, regression gate, honest record (Task 10).

Every test here runs on fakes -- ``FakeTrainer`` (Task 9), ``FakeServerAdapter`` and
``FakeSliceRunner`` (defined alongside the loop itself) -- so the whole suite is GPU-free
and server-free. Exactly ONE test (``test_refalsify_tally_counts_a_real_re_execution_once``)
pays for a real sandbox run, because the tally landing in the record is the one thing no
fake can honestly stand in for.

Fixtures follow ``tests/memory/test_falsify.py``'s ``_episode``/``_semantic``/``_unit``
helpers (same module source, same visible/hidden suites) so a lesson re-executed here is
re-executed against exactly the shape ``crucible.memory.falsify`` was pinned against.

Five things here are load-bearing, not incidental.

*The counter-reset pins are behavioural, not introspective.* Nothing asserts on
``SleepController``'s private counter directly. "The counter reset" is pinned as "a second
``maybe_sleep`` with no new verified episodes returns ``None``"; "the counter did NOT
reset" is pinned as "a second ``maybe_sleep`` with no new verified episodes fires again
and TRAINS again" (``_SpyTrainer.calls`` length 2). A mutant that resets the counter on
the reject path passes every other test in this file and fails
``test_reject_does_not_reset_the_counter_so_the_next_check_trains_again`` alone.

*The accept rule is pinned on BOTH sides of its boundary.* ``ACCEPT_MAX_DROP = 1`` means a
drop of exactly 1 ACCEPTS and a drop of 2 REJECTS; both are separate tests
(``test_accept_boundary_drop_of_exactly_one_accepts`` and
``test_reject_when_the_slice_drops_two``), so an off-by-one mutant (``<`` for ``<=``) is
killed by the first while the second keeps the rule from being weakened in the other
direction.

*``server.load`` is pinned by ABSENCE on the reject path.* ``FakeServerAdapter.calls``
being empty after a rejected sleep is the only thing standing between a regression gate
and a gate that measures a drop, writes ``accepted=False``, and hot-swaps the loser
anyway.

*The slice is a function of the solved-task SET, the seed, and the sleep index -- never of
the caller's list order.* ``test_slice_is_the_same_for_a_shuffled_solved_list`` pins that
specifically: a controller that sampled straight from the caller's list would pass the
plain determinism test (same list twice) and fail this one.

*The refalsify batch's two clauses are tested apart.* An item cited by an SFT episode and
an item that is merely stale (retrieval-eligible, not re-checked since the previous sleep)
reach the batch by different routes, so each has its own test, plus one for the dedupe
where both routes name the same item, and one for the item that neither route reaches.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, fields
from pathlib import Path

import pytest

from crucible.memory.schema import EpisodicRecord, SemanticItem, content_id
from crucible.memory.store import MemoryStore
from crucible.sleep import loop
from crucible.sleep.loop import (
    ACCEPT_MAX_DROP,
    BASE_DIGEST,
    SLEEP_THRESHOLD_DEFAULT,
    SLICE_SIZE,
    FakeServerAdapter,
    FakeSliceRunner,
    ServerAdapter,
    SleepController,
    SleepRecord,
    SliceRunner,
    VllmAdapterLoader,
)
from crucible.sleep.registry import AdapterRegistry, adapter_id_for
from crucible.sleep.select import episode_set_hash, sft_pairs
from crucible.sleep.train import BASE_MODEL, FakeTrainer
from crucible.stream.units import Unit, sha256_text

SRC_OK = "def f(a, b):\n    return a + b\n"
SRC_BROKEN = "def f(a, b):\n    return a - b\n"
VIS = ("from unit_x import f as candidate\n"
       "def test_v0():\n    assert candidate(2, 3) == 5\n"
       "def test_v1():\n    assert candidate(0, 0) == 0\n")
HID = "from unit_x import f as candidate\ndef test_h0():\n    assert candidate(0, 0) == 0\n"

SOLVED_20 = tuple(f"solved-{i:02d}" for i in range(20))


def _unit(module_src: str = SRC_OK) -> Unit:
    return Unit("X/0", "unit_x", "f", module_src, VIS, HID, sha256_text(module_src), 1, 1, ())


def _episode(task_key: str, *, arm: str = "A_full", unit_id: str = "X/0", family: str = "ARITH",
             verified: bool = True, landed_module: str | None = SRC_OK,
             created_at: str = "2026-08-24T10:00:00Z",
             root_prompt: str = "Fix the bug.") -> EpisodicRecord:
    item_id = content_id("episode", {"task_key": task_key, "arm": arm})
    return EpisodicRecord(
        item_id=item_id, task_key=task_key, arm=arm, unit_id=unit_id, family=family,
        class_id=f"{unit_id}|{family}", phase=1, kind="first",
        root_prompt=root_prompt, landed_module=landed_module, visible_reward=1.0,
        executions_charged=2, hidden_pass=verified, verified=verified,
        memory_item_ids=(), created_at=created_at, confidence=0.8,
        status="active", version=1, source_locator=f"run:t/task:{task_key}",
        valid_at=created_at, invalid_at=None, expired_at=None,
        last_verified_at=None, falsified_by=None, verification_method="hidden-suite",
    )


def _semantic(cited_episode_id: str, *, unit_id: str = "X/0", family: str = "ARITH",
              flipped_tests: tuple[str, ...] = ("test_v0",),
              falsified_by: str | None = None,
              last_verified_at: str | None = None) -> SemanticItem:
    item_id = content_id("semantic", {"cited_episode_id": cited_episode_id})
    return SemanticItem(
        item_id=item_id, unit_id=unit_id, family=family, class_id=f"{unit_id}|{family}",
        cited_episode_id=cited_episode_id, mutated_spans=(((2, 5), (2, 9)),),
        landed_diff="-    return a - b\n+    return a + b\n", flipped_tests=flipped_tests,
        killing_tests=flipped_tests, created_at="2026-08-24T10:06:00Z", confidence=0.75,
        status="active", version=1, source_locator=f"run:t/episode:{cited_episode_id}",
        valid_at="2026-08-24T10:06:00Z", invalid_at=None, expired_at=None,
        last_verified_at=last_verified_at, falsified_by=falsified_by,
        verification_method="mechanical-template",
    )


class _SpyTrainer:
    """``FakeTrainer`` plus a call log.

    The "trains again after a reject" pin needs a call COUNT, and ``FakeTrainer``'s
    on-disk ``adapter_config.json`` cannot distinguish one call from two identical ones
    (same pairs, same seed, same dir -- the second write is byte-identical to the first).
    """

    def __init__(self) -> None:
        self._inner = FakeTrainer()
        self.calls: list[tuple[list[tuple[str, str]], int, Path]] = []

    def train(self, pairs: list[tuple[str, str]], seed: int, out_dir: Path) -> Path:
        self.calls.append((list(pairs), seed, Path(out_dir)))
        return self._inner.train(pairs, seed, out_dir)


class _SpyUnitLoader:
    """``unit_id -> Unit`` with a call log; raises ``KeyError`` on an unknown id (loud, by design)."""

    def __init__(self, units: dict[str, Unit] | None = None) -> None:
        self._units = units if units is not None else {"X/0": _unit()}
        self.calls: list[str] = []

    def __call__(self, unit_id: str) -> Unit:
        self.calls.append(unit_id)
        return self._units[unit_id]


@dataclass
class _Rig:
    """One wired-up controller plus every fake, so a test can assert on any seam."""

    controller: SleepController
    store: MemoryStore
    trainer: _SpyTrainer
    server: FakeServerAdapter
    runner: FakeSliceRunner
    registry: AdapterRegistry
    adapters_dir: Path
    unit_loader: _SpyUnitLoader


def _rig(tmp_path: Path, *, verified: int = 3, counts: tuple[int, ...] = (10, 10),
         threshold: int = 3, seed: int = 7, units: dict[str, Unit] | None = None,
         prior_rows: int = 0) -> _Rig:
    """A store holding ``verified`` verified episodes, wired to fakes with scripted ``counts``.

    ``prior_rows`` writes that many registry rows BEFORE the controller is constructed --
    the resume case, where sleeps already happened in an earlier process.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(tmp_path / "mem.sqlite3")
    for i in range(verified):
        store.write_episode(_episode(f"tk-{i:02d}", created_at=f"2026-08-24T10:{i:02d}:00Z",
                                     root_prompt=f"prompt {i}", landed_module=f"# module {i}\n"))
    trainer = _SpyTrainer()
    server = FakeServerAdapter()
    runner = FakeSliceRunner(counts)
    registry = AdapterRegistry(tmp_path / "adapters.jsonl")
    for i in range(prior_rows):
        registry.record(f"ad-prior{i:012d}", "0" * 64, BASE_DIGEST, False,
                        f"2026-08-23T0{i}:00:00Z")
    adapters_dir = tmp_path / "adapters"
    unit_loader = _SpyUnitLoader(units)
    controller = SleepController(store, trainer, server, runner, registry,
                                 unit_loader=unit_loader, adapters_dir=adapters_dir,
                                 threshold=threshold, seed=seed)
    return _Rig(controller, store, trainer, server, runner, registry, adapters_dir, unit_loader)


# --------------------------------------------------------------------------- constants

def test_pinned_constants():
    assert SLEEP_THRESHOLD_DEFAULT == 16
    assert SLICE_SIZE == 12
    assert ACCEPT_MAX_DROP == 1
    assert BASE_DIGEST == sha256_text(BASE_MODEL)


# ----------------------------------------------------------------------------- trigger

def test_below_threshold_returns_none_and_touches_nothing(tmp_path: Path):
    # 15 verified episodes against the DEFAULT threshold of 16 -- one short.
    rig = _rig(tmp_path, verified=15, threshold=SLEEP_THRESHOLD_DEFAULT)

    assert rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z") is None

    assert rig.trainer.calls == []
    assert rig.server.calls == []
    assert rig.runner.calls == []
    assert rig.registry.latest_accepted() is None
    assert rig.unit_loader.calls == []
    rig.store.close()


def test_at_the_default_threshold_the_pipeline_fires(tmp_path: Path):
    rig = _rig(tmp_path, verified=16, threshold=SLEEP_THRESHOLD_DEFAULT)

    record = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")

    assert record is not None
    assert len(rig.trainer.calls) == 1
    rig.store.close()


def test_full_pipeline_fills_every_record_field(tmp_path: Path):
    rig = _rig(tmp_path, verified=16, threshold=SLEEP_THRESHOLD_DEFAULT, counts=(10, 10))
    expected_hash = episode_set_hash(sft_pairs(rig.store))

    record = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")

    assert record.sleep_index == 0
    assert record.adapter_id == adapter_id_for(expected_hash)
    assert record.episode_set_hash == expected_hash
    assert record.episodes_selected == 16
    assert len(record.slice_task_keys) == SLICE_SIZE
    assert set(record.slice_task_keys) <= set(SOLVED_20)
    assert record.slice_before == 10
    assert record.slice_after == 10
    assert record.accepted is True
    assert record.refalsify == {"checked": 0, "passed": 0, "falsified": 0, "infra": 0,
                                "infra_broken_citation": 0}
    # S4 (2026-08-24): gpu_s is the wall-clock seconds of the Trainer.train call --
    # the pre-reg Section 3 declared asymmetry (A_full's sleep GPU time) must be REPORTED,
    # so an unmeasured None here would break a pre-registered commitment at write-up.
    assert isinstance(record.gpu_s, float) and record.gpu_s >= 0.0
    assert record.created_at == "2026-08-24T12:00:00Z"
    rig.store.close()


def test_trainer_gets_the_cumulative_pairs_the_seed_and_a_content_addressed_dir(tmp_path: Path):
    rig = _rig(tmp_path, verified=3, threshold=3, seed=11)
    expected_pairs = sft_pairs(rig.store)
    expected_id = adapter_id_for(episode_set_hash(expected_pairs))

    rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")

    (pairs, seed, out_dir), = rig.trainer.calls
    assert pairs == expected_pairs
    assert seed == 11
    assert out_dir == (rig.adapters_dir / expected_id).resolve()
    assert (out_dir / "adapter_config.json").exists()
    rig.store.close()


def test_slice_runner_is_asked_before_with_the_live_adapter_and_after_with_the_candidate(tmp_path: Path):
    rig = _rig(tmp_path, verified=3, threshold=3, counts=(9, 9))

    record = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")

    before_call, after_call = rig.runner.calls
    assert before_call == (record.slice_task_keys, None)  # no accepted adapter yet
    assert after_call == (record.slice_task_keys, record.adapter_id)
    rig.store.close()


def test_the_second_sleep_measures_before_against_the_live_adapter(tmp_path: Path):
    # `before` is "what is serving now" -- once an adapter has been accepted, the base model
    # is no longer the comparison. A controller that always asked with None would pass every
    # first-sleep test in this file and silently grade every later candidate against base.
    rig = _rig(tmp_path, verified=3, threshold=3, counts=(10, 10, 10, 10))
    first = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")
    assert first.accepted is True
    for i in range(3, 6):
        rig.store.write_episode(_episode(f"tk-{i:02d}", created_at=f"2026-08-24T10:{i:02d}:00Z",
                                         root_prompt=f"prompt {i}", landed_module=f"# module {i}\n"))

    second = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T13:00:00Z")

    assert second.adapter_id != first.adapter_id  # a new episode set mints a new id
    before_call, after_call = rig.runner.calls[2:]
    assert before_call == (second.slice_task_keys, first.adapter_id)
    assert after_call == (second.slice_task_keys, second.adapter_id)
    rig.store.close()


# ------------------------------------------------------------------------ accept path

def test_accept_calls_server_load_exactly_once_with_the_adapter_dir_and_id(tmp_path: Path):
    rig = _rig(tmp_path, verified=3, threshold=3, counts=(10, 10))

    record = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")

    assert record.accepted is True
    assert rig.server.calls == [((rig.adapters_dir / record.adapter_id).resolve(), record.adapter_id)]
    rig.store.close()


def test_accept_records_an_accepted_registry_row(tmp_path: Path):
    rig = _rig(tmp_path, verified=3, threshold=3, counts=(10, 10))

    record = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")

    assert rig.registry.latest_accepted() == record.adapter_id
    row, = [json.loads(line) for line in (tmp_path / "adapters.jsonl").read_text().splitlines()]
    assert row == {"adapter_id": record.adapter_id, "episode_set_hash": record.episode_set_hash,
                   "base_digest": BASE_DIGEST, "accepted": True,
                   "created_at": "2026-08-24T12:00:00Z"}
    rig.store.close()


def test_accept_resets_the_counter_so_the_next_check_returns_none(tmp_path: Path):
    rig = _rig(tmp_path, verified=3, threshold=3, counts=(10, 10))
    assert rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z") is not None

    # No new verified episodes landed since -- the accepted sleep consumed them all.
    second = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T13:00:00Z")

    assert second is None
    assert len(rig.trainer.calls) == 1
    rig.store.close()


def test_accept_boundary_drop_of_exactly_one_accepts(tmp_path: Path):
    # ACCEPT_MAX_DROP = 1: before - after == 1 is INSIDE the gate. A `<` mutant rejects here.
    rig = _rig(tmp_path, verified=3, threshold=3, counts=(12, 11))

    record = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")

    assert record.accepted is True
    assert len(rig.server.calls) == 1
    rig.store.close()


def test_an_adapter_that_solves_more_accepts(tmp_path: Path):
    rig = _rig(tmp_path, verified=3, threshold=3, counts=(9, 12))

    record = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")

    assert record.accepted is True
    assert record.slice_before == 9 and record.slice_after == 12
    rig.store.close()


# ------------------------------------------------------------------------ reject path

def test_reject_when_the_slice_drops_two(tmp_path: Path):
    rig = _rig(tmp_path, verified=3, threshold=3, counts=(12, 10))

    record = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")

    assert record.accepted is False
    assert record.slice_before == 12 and record.slice_after == 10
    rig.store.close()


def test_reject_never_calls_the_server(tmp_path: Path):
    rig = _rig(tmp_path, verified=3, threshold=3, counts=(12, 10))

    rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")

    assert rig.server.calls == []
    rig.store.close()


def test_reject_records_a_rejected_row_and_leaves_latest_accepted_alone(tmp_path: Path):
    rig = _rig(tmp_path, verified=3, threshold=3, counts=(12, 10))

    record = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")

    assert rig.registry.latest_accepted() is None
    row, = [json.loads(line) for line in (tmp_path / "adapters.jsonl").read_text().splitlines()]
    assert row["accepted"] is False
    assert row["adapter_id"] == record.adapter_id
    rig.store.close()


def test_reject_does_not_reset_the_counter_so_the_next_check_trains_again(tmp_path: Path):
    # THE pin for mutation (a): reset the counter on the reject path and this test alone fails.
    rig = _rig(tmp_path, verified=3, threshold=3, counts=(12, 10, 12, 10))
    first = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")
    assert first.accepted is False

    # Not one new verified episode since -- and it must still fire, because nothing was accepted.
    second = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T13:00:00Z")

    assert second is not None
    assert second.sleep_index == 1
    assert len(rig.trainer.calls) == 2
    rig.store.close()


def test_accepted_sleep_snapshots_the_total_rather_than_advancing_by_threshold(tmp_path: Path):
    # Review finding 2. An accepted sleep consumes EVERY verified episode that existed when
    # it fired (5), not `threshold` of them -- so one new episode afterwards leaves 6 - 5 = 1
    # < 3 and the next check must stay quiet. A `+= threshold` mutant snapshots 3 instead,
    # leaving a phantom surplus of 2 and firing this second sleep early.
    rig = _rig(tmp_path, verified=5, threshold=3, counts=(10, 10, 10, 10))
    first = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")
    assert first.accepted is True and first.episodes_selected == 5

    rig.store.write_episode(_episode("tk-05", created_at="2026-08-24T10:05:00Z",
                                     root_prompt="prompt 5", landed_module="# module 5\n"))
    second = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T13:00:00Z")

    assert second is None
    assert len(rig.trainer.calls) == 1
    rig.store.close()


# ------------------------------------------------------------------------------- resume

def test_sleep_index_is_seeded_from_the_registrys_committed_rows(tmp_path: Path):
    # Review finding 1: a resumed controller must not restart the index at 0 -- that would
    # stamp duplicate sleep_index values AND re-issue index 0's slice draw on a different run.
    resumed = _rig(tmp_path / "resumed", verified=3, threshold=3, prior_rows=2)
    fresh = _rig(tmp_path / "fresh", verified=3, threshold=3)

    resumed_record = resumed.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")
    fresh_record = fresh.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")

    assert resumed_record.sleep_index == 2
    assert fresh_record.sleep_index == 0
    # Same seed, same solved set: only the index differs, and it must move the draw.
    assert resumed_record.slice_task_keys != fresh_record.slice_task_keys
    resumed.store.close()
    fresh.store.close()


def test_sleep_index_keeps_counting_from_the_seeded_start(tmp_path: Path):
    rig = _rig(tmp_path, verified=3, threshold=3, counts=(12, 10, 12, 10), prior_rows=2)

    first = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")
    second = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T13:00:00Z")

    assert (first.sleep_index, second.sleep_index) == (2, 3)
    rig.store.close()


# ----------------------------------------------------------------------------- slicing

def test_slice_is_deterministic_for_the_same_seed_and_index(tmp_path: Path):
    rig_a = _rig(tmp_path / "a", seed=7)
    rig_b = _rig(tmp_path / "b", seed=7)

    keys_a = rig_a.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z").slice_task_keys
    keys_b = rig_b.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z").slice_task_keys

    assert keys_a == keys_b
    rig_a.store.close()
    rig_b.store.close()


def test_slice_is_the_same_for_a_shuffled_solved_list(tmp_path: Path):
    # The slice is a function of the solved SET, not of the caller's list order.
    rig_a = _rig(tmp_path / "a", seed=7)
    rig_b = _rig(tmp_path / "b", seed=7)
    shuffled = list(reversed(SOLVED_20))

    keys_a = rig_a.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z").slice_task_keys
    keys_b = rig_b.controller.maybe_sleep(shuffled, now="2026-08-24T12:00:00Z").slice_task_keys

    assert keys_a == keys_b
    rig_a.store.close()
    rig_b.store.close()


def test_slice_differs_by_seed(tmp_path: Path):
    rig_a = _rig(tmp_path / "a", seed=7)
    rig_b = _rig(tmp_path / "b", seed=8)

    keys_a = rig_a.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z").slice_task_keys
    keys_b = rig_b.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z").slice_task_keys

    assert keys_a != keys_b
    rig_a.store.close()
    rig_b.store.close()


def test_slice_differs_by_sleep_index(tmp_path: Path):
    rig = _rig(tmp_path, verified=3, threshold=3, counts=(12, 10, 12, 10))

    first = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")
    second = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T13:00:00Z")

    assert first.slice_task_keys != second.slice_task_keys
    rig.store.close()


def test_slice_is_capped_at_min_of_slice_size_and_solved_count(tmp_path: Path):
    rig = _rig(tmp_path, verified=3, threshold=3)
    five = ["s-0", "s-1", "s-2", "s-3", "s-4"]

    record = rig.controller.maybe_sleep(five, now="2026-08-24T12:00:00Z")

    assert len(record.slice_task_keys) == 5
    assert sorted(record.slice_task_keys) == five
    rig.store.close()


def test_no_solved_tasks_yields_an_empty_slice_and_a_visibly_vacuous_gate(tmp_path: Path):
    rig = _rig(tmp_path, verified=3, threshold=3, counts=(0, 0))

    record = rig.controller.maybe_sleep([], now="2026-08-24T12:00:00Z")

    assert record.slice_task_keys == ()
    assert record.slice_before == 0 and record.slice_after == 0
    assert record.accepted is True  # nothing measured => nothing regressed; the record shows it
    rig.store.close()


# ------------------------------------------------------------------------ refalsification

def test_refalsify_tally_counts_a_real_re_execution_once(tmp_path: Path):
    # The one test that pays for a real sandbox run. The item is reachable by BOTH batch
    # clauses (its cited episode enters SFT, and it has never been re-checked) -- checked
    # == 1 pins the dedupe as well as the tally.
    rig = _rig(tmp_path, verified=3, threshold=3)
    # Overwrite the rig's placeholder tk-00 (same (task_key, arm) identity) with one whose
    # landed module is real, runnable source -- the claim below is re-executed against it.
    cited = _episode("tk-00", landed_module=SRC_OK)
    rig.store.write_episode(cited)
    item = _semantic(cited.item_id)
    rig.store.write_semantic(item)

    record = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")

    assert record.refalsify["checked"] == 1
    assert record.refalsify["passed"] == 1
    assert record.refalsify["falsified"] == 0
    assert rig.unit_loader.calls == ["X/0"]
    reloaded, = rig.store.semantic_for("X/0", "ARITH")
    assert reloaded.last_verified_at == "2026-08-24T12:00:00Z"
    rig.store.close()


def test_refalsify_includes_a_stale_retrieval_eligible_item(tmp_path: Path):
    # Clause two on its own: this item's cited episode is NOT verified, so it never enters
    # SFT -- it is in the batch only because it is live and has never been re-checked.
    rig = _rig(tmp_path, verified=3, threshold=3)
    unverified = _episode("tk-unverified", verified=False)
    rig.store.write_episode(unverified)
    rig.store.write_semantic(_semantic(unverified.item_id))

    record = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")

    assert record.refalsify["checked"] == 1
    assert record.refalsify["infra_broken_citation"] == 1  # unverified citation: no verdict
    assert rig.unit_loader.calls == ["X/0"]
    rig.store.close()


def test_refalsify_skips_a_falsified_item_no_sft_episode_cites(tmp_path: Path):
    rig = _rig(tmp_path, verified=3, threshold=3)
    unverified = _episode("tk-unverified", verified=False)
    rig.store.write_episode(unverified)
    rig.store.write_semantic(_semantic(unverified.item_id, falsified_by="re-exec: earlier check"))

    record = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")

    assert record.refalsify["checked"] == 0
    assert rig.unit_loader.calls == []
    rig.store.close()


def test_refalsify_skips_an_item_verified_since_the_previous_sleep(tmp_path: Path):
    # The staleness watermark is the PREVIOUS sleep's `now`, accepted or not. The item is
    # re-checked at the first sleep; by the second it carries a last_verified_at AFTER that
    # first sleep, so clause two no longer reaches it (and no SFT episode cites it).
    rig = _rig(tmp_path, verified=3, threshold=3, counts=(12, 10, 12, 10))
    unverified = _episode("tk-unverified", verified=False)
    rig.store.write_episode(unverified)
    rig.store.write_semantic(_semantic(unverified.item_id, last_verified_at="2026-08-24T12:30:00Z"))

    first = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")
    second = rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T13:00:00Z")

    assert first.refalsify["checked"] == 1
    assert second.refalsify["checked"] == 0
    rig.store.close()


def test_refalsify_runs_before_training(tmp_path: Path):
    # Pipeline order: a claim is re-checked BEFORE the episodes it belongs to train an
    # adapter -- pinned by the store mutation being visible at the moment train is called.
    rig = _rig(tmp_path, verified=3, threshold=3)
    cited = _episode("tk-00", landed_module=SRC_OK)
    rig.store.write_episode(cited)
    rig.store.write_semantic(_semantic(cited.item_id))
    seen: list[str | None] = []

    inner = rig.trainer.train

    def _recording_train(pairs, seed, out_dir):
        item, = rig.store.semantic_for("X/0", "ARITH")
        seen.append(item.last_verified_at)
        return inner(pairs, seed, out_dir)

    rig.trainer.train = _recording_train  # type: ignore[method-assign]
    rig.controller.maybe_sleep(list(SOLVED_20), now="2026-08-24T12:00:00Z")

    assert seen == ["2026-08-24T12:00:00Z"]
    rig.store.close()


# ------------------------------------------------------------------------- SleepRecord

def _record() -> SleepRecord:
    return SleepRecord(
        sleep_index=2, adapter_id="ad-0123456789abcdef", episode_set_hash="f" * 64,
        episodes_selected=17, slice_task_keys=("a", "b"), slice_before=9, slice_after=8,
        accepted=True,
        refalsify={"checked": 3, "passed": 2, "falsified": 1, "infra": 0, "infra_broken_citation": 0},
        gpu_s=41.5, created_at="2026-08-24T12:00:00Z",
    )


def test_sleep_record_round_trips_through_json():
    rec = _record()

    assert SleepRecord.from_dict(json.loads(json.dumps(rec.to_dict()))) == rec


def test_sleep_record_round_trips_with_gpu_s_none():
    rec = SleepRecord(**{**_record().to_dict(), "slice_task_keys": ("a", "b"), "gpu_s": None})

    assert SleepRecord.from_dict(json.loads(json.dumps(rec.to_dict()))) == rec


def test_sleep_record_to_dict_is_field_complete():
    assert set(_record().to_dict()) == {f.name for f in fields(SleepRecord)}


def test_sleep_record_field_order_is_frozen():
    assert [f.name for f in fields(SleepRecord)] == [
        "sleep_index", "adapter_id", "episode_set_hash", "episodes_selected",
        "slice_task_keys", "slice_before", "slice_after", "accepted", "refalsify",
        "gpu_s", "created_at",
    ]


# ----------------------------------------------------------------------------- the seams

def test_fakes_and_the_live_loader_satisfy_their_protocols():
    assert isinstance(FakeServerAdapter(), ServerAdapter)
    assert isinstance(VllmAdapterLoader("http://127.0.0.1:8010"), ServerAdapter)
    assert isinstance(FakeSliceRunner([1]), SliceRunner)


def test_fake_slice_runner_raises_when_its_script_runs_out():
    runner = FakeSliceRunner([1])
    runner.solved(["a"], None)

    with pytest.raises(AssertionError, match="scripted"):
        runner.solved(["a"], "ad-x")


def test_vllm_adapter_loader_posts_the_lora_name_and_path(monkeypatch):
    captured = {}

    class _Resp:
        status = 200

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = json.loads(req.data)
        captured["content_type"] = req.get_header("Content-type")
        return _Resp()

    monkeypatch.setattr(loop.urllib.request, "urlopen", _fake_urlopen)
    VllmAdapterLoader("http://127.0.0.1:8010/").load(Path("/tmp/adapters/ad-abc"), "ad-abc")

    assert captured["url"] == "http://127.0.0.1:8010/v1/load_lora_adapter"
    assert captured["method"] == "POST"
    assert captured["body"] == {"lora_name": "ad-abc", "lora_path": "/tmp/adapters/ad-abc"}
    assert captured["content_type"] == "application/json"


# -------------------------------------------------------------------------- source pins

def test_loop_module_never_imports_a_spend_meter():
    # Same crude source-scan pin as crucible/memory/falsify.py: consolidation is a
    # maintenance concern of the memory organ, never charged to a task's execution cap.
    assert "budget" not in Path(loop.__file__).read_text()


def test_loop_module_reads_no_wall_clock():
    # `now` is caller-supplied everywhere in this codebase; a wall-clock read here would
    # be the first step toward a record that cannot be reproduced. The S4 gpu_s amendment
    # (2026-08-24) legitimately measures a DURATION, so `time` is admitted under a tighter
    # pin: the ONLY name this module may take from the time module is `monotonic`, through
    # ANY import form at ANY nesting level (review finding: the first version of this pin
    # missed `from time import time as x` and `import time as t` aliases, and the pin it
    # replaced never saw function-level imports at all). `datetime` stays banned outright.
    source = Path(loop.__file__).read_text()
    tree = ast.parse(source)
    module_aliases = set()   # names that refer to the time MODULE (time, or `import time as t`)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root != "datetime", f"datetime import in {loop.__file__}"
                if root == "time":
                    module_aliases.add(alias.asname or root)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root != "datetime", f"datetime import in {loop.__file__}"
            if root == "time":
                taken = {a.name for a in node.names}
                assert taken <= {"monotonic"}, \
                    f"non-monotonic from-time import in {loop.__file__}: {taken}"
    attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name) and node.value.id in module_aliases
    }
    assert attrs <= {"monotonic"}, f"non-monotonic time use in {loop.__file__}: {attrs}"
    assert module_aliases or attrs, "expected an import of time for the gpu_s measurement"
    assert "monotonic" in attrs or not module_aliases, \
        "time imported but monotonic never used -- gpu_s measurement missing?"
