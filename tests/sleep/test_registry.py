"""Tests for the adapter registry (append-only JSONL ledger) and the trainer seam it
records against (Task 9 brief). The trainer-seam tests (``FakeTrainer`` + the
import-discipline ``ast`` check for ``crucible.sleep.train``) live in this file rather
than a separate ``test_train.py`` -- the task's own file list names only
``test_select.py``/``test_registry.py`` for this task, and a trainer's output is exactly
what the registry records against, so the two are tested together.

Three things here are load-bearing, not incidental.

*``latest_accepted`` really skips rejected rows -- not just "the common case looks
right."* ``test_latest_accepted_skips_a_later_rejected_row`` writes an ACCEPTED row
followed by a REJECTED row and asserts the accepted id still comes back -- a mutant that
returned "the last row's id regardless of ``accepted``" would pass every other registry
test here (they only ever have one relevant terminal row) but fail this one specifically.
See the task report for the literal mutation-check evidence.

*``crucible.sleep.train`` must import with zero heavy dependencies installed.* This
environment does not have torch/peft/trl/transformers installed at all (a deliberate
choice -- see ``pyproject.toml``'s ``serve`` extra), so
``test_train_module_imports_without_torch_peft_trl_transformers`` is not a simulation: the
module is actually being imported, for real, in an environment that would ImportError
immediately if any of those names were imported at module level.
``test_train_module_has_no_heavy_import_at_module_level`` additionally parses the
module's own source with ``ast`` and asserts none of those four names appear in a
module-level ``import``/``from ... import`` statement -- a static pin that survives even
if this environment's dependency set ever changes to include them (at which point the
plain-import test alone would stop being able to catch a regression).

*No BudgetMeter anywhere under ``crucible/sleep/``.* Same crude source-scan pin
``tests/memory/test_falsify.py`` uses: falsification and sleep-training are maintenance
concerns of the memory/adapter lifecycle, never a search-time spend a task's execution
budget was meant to track. ``test_sleep_package_never_imports_a_spend_meter`` greps every
``.py`` file directly under ``crucible/sleep/`` for the lowercase substring naming that
meter's module.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import crucible.sleep.train as train_module
from crucible.sleep.registry import AdapterRegistry, adapter_id_for
from crucible.sleep.train import FakeTrainer, Trainer

_HEAVY_MODULE_NAMES = {"torch", "peft", "trl", "transformers"}


# --- Trainer seam -----------------------------------------------------------

def test_train_module_imports_without_torch_peft_trl_transformers():
    # If this test file collected at all, the import at the top of this module already
    # succeeded in an environment with none of those four packages installed -- this
    # assertion documents *why* that's meaningful, not a separate act.
    assert train_module.Trainer is Trainer


def test_train_module_has_no_heavy_import_at_module_level():
    source = Path(train_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:  # module level ONLY -- deliberately not ast.walk, which would
                             # also catch the imports inside LoraTrainer.train (the point).
        if isinstance(node, ast.Import):
            names = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {(node.module or "").split(".")[0]}
        else:
            continue
        bad = names & _HEAVY_MODULE_NAMES
        assert not bad, f"module-level import of {bad} in crucible.sleep.train"


def test_fake_trainer_writes_adapter_config_and_returns_out_dir(tmp_path):
    trainer: Trainer = FakeTrainer()
    pairs = [("prompt one", "module one"), ("prompt two", "module two")]

    result = trainer.train(pairs, seed=7, out_dir=tmp_path / "adapter")

    assert result == tmp_path / "adapter"
    written = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert written == {"pairs": 2, "seed": 7}


def test_fake_trainer_pairs_count_reflects_actual_list_length(tmp_path):
    trainer = FakeTrainer()

    result = trainer.train([], seed=0, out_dir=tmp_path / "empty")

    written = json.loads((result / "adapter_config.json").read_text())
    assert written == {"pairs": 0, "seed": 0}


def test_sleep_package_never_imports_a_spend_meter():
    sleep_dir = Path(train_module.__file__).resolve().parent
    for py_file in sleep_dir.glob("*.py"):
        assert "budget" not in py_file.read_text(encoding="utf-8"), py_file


# --- AdapterRegistry ---------------------------------------------------------

def test_adapter_id_for_is_ad_prefixed_16_hex_of_the_hash():
    digest = "a" * 64
    assert adapter_id_for(digest) == "ad-" + "a" * 16


def test_record_then_latest_accepted_round_trips_a_single_accepted_row(tmp_path):
    registry = AdapterRegistry(tmp_path / "registry.jsonl")

    registry.record(adapter_id="ad-0000000000000001", episode_set_hash="hash-1",
                     base_digest="base-digest-1", accepted=True, created_at="2026-08-24T10:00:00Z")

    assert registry.latest_accepted() == "ad-0000000000000001"


def test_latest_accepted_is_none_for_an_empty_registry(tmp_path):
    registry = AdapterRegistry(tmp_path / "registry.jsonl")
    assert registry.latest_accepted() is None


def test_latest_accepted_is_none_when_every_row_is_rejected(tmp_path):
    registry = AdapterRegistry(tmp_path / "registry.jsonl")
    registry.record("ad-a", "hash-a", "base-1", accepted=False, created_at="2026-08-24T10:00:00Z")
    registry.record("ad-b", "hash-b", "base-1", accepted=False, created_at="2026-08-24T11:00:00Z")

    assert registry.latest_accepted() is None


def test_latest_accepted_skips_a_later_rejected_row(tmp_path):
    # The load-bearing test: a later REJECTED row must not shadow an earlier ACCEPTED one.
    registry = AdapterRegistry(tmp_path / "registry.jsonl")
    registry.record("ad-accepted", "hash-accepted", "base-1", accepted=True, created_at="2026-08-24T10:00:00Z")
    registry.record("ad-rejected", "hash-rejected", "base-1", accepted=False, created_at="2026-08-24T11:00:00Z")

    assert registry.latest_accepted() == "ad-accepted"


def test_latest_accepted_returns_the_most_recent_of_two_accepted_rows(tmp_path):
    registry = AdapterRegistry(tmp_path / "registry.jsonl")
    registry.record("ad-first", "hash-first", "base-1", accepted=True, created_at="2026-08-24T10:00:00Z")
    registry.record("ad-second", "hash-second", "base-1", accepted=True, created_at="2026-08-24T11:00:00Z")

    assert registry.latest_accepted() == "ad-second"


def test_registry_persists_across_reopen(tmp_path):
    path = tmp_path / "registry.jsonl"
    AdapterRegistry(path).record("ad-x", "hash-x", "base-1", accepted=True, created_at="2026-08-24T10:00:00Z")

    reopened = AdapterRegistry(path)

    assert reopened.latest_accepted() == "ad-x"


def test_registry_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "registry.jsonl"
    registry = AdapterRegistry(path)
    registry.record("ad-x", "hash-x", "base-1", accepted=True, created_at="2026-08-24T10:00:00Z")

    assert path.exists()
    assert registry.latest_accepted() == "ad-x"
