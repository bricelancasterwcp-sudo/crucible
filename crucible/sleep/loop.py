"""The sleep loop: when to consolidate, what to re-check first, and whether to keep the result.

This is the organ's consolidation engine. Everything it needs already exists behind a seam
-- selection (``select.py``), training (``train.py``), the ledger (``registry.py``),
re-execution (``crucible.memory.falsify``) -- so what lives here is only the policy that
strings them together: trigger, refalsify, train, gate, accept-or-reject, record.

*The trigger counts verified episodes since the last ACCEPTED sleep, not since the last
sleep.* ``self._last_accepted_count`` is a snapshot of ``store.verified_count()`` taken at
acceptance and at no other moment. A rejected candidate leaves it untouched, so the very
next check fires again on the same episodes -- which is the point: those episodes are still
unconsolidated, and a gate that swallowed them on a rejection would quietly discard every
lesson learned since the last accepted adapter until another ``threshold`` arrived. The
arithmetic is ``store.count_verified_since(self._last_accepted_count) >= threshold``, which
is exactly the pinned ``verified_count() - _last_accepted_count >= threshold`` (Task 2's
helper clamps the difference at 0, and a negative difference can never reach a positive
threshold, so the clamp is unreachable here -- confirming Task 2's open note rather than
re-deriving the subtraction inline). ``_last_accepted_count`` starts at 0, so a controller
built over a store that ALREADY holds ``threshold`` verified episodes sleeps on its first
check: from this controller's point of view those episodes have never trained anything.

*Resume is a supported mode, and only PART of this controller's state survives it
(review finding 1).* ``crucible.run.driver`` is resumable at task granularity, so a
``SleepController`` can legitimately be constructed part-way through a run. ``_sleep_index``
is therefore seeded at construction from ``registry.count()`` -- every sleep appends exactly
one row, so the committed row count IS the number of sleeps that already happened. That is
the load-bearing half: without it a resumed run would re-issue index 0's slice draw (grading
the new candidate on the same sample the pre-crash run used) and stamp duplicate
``sleep_index`` values on records that are not the same sleep. The other two pieces of state
deliberately do NOT survive, and both fail conservatively:

- ``_last_accepted_count`` restarts at 0, so the first post-resume check counts the whole
  verified history again and fires one sleep a continuous run would not have fired. The cost
  is one extra consolidation cycle, gated exactly like every other -- it can waste a training
  run, never accept a worse adapter.
- ``_last_refalsified_at`` restarts at ``None``, so that first post-resume sleep re-checks
  every live item instead of only the stale ones. More re-execution, never less.

Persisting either would mean a second on-disk state file whose staleness is itself a failure
mode; the honest trade is to restart them and say so here. Note for the design doc's "a
replay sleeps at identical points" (§1): that holds for a REPLAY FROM ZERO -- same stream,
same seed, same task order gives the same trigger points and the same slice draws. A RESUMED
run is not a replay and shifts by the extra sleep above.

*What the refalsify batch covers, and why it is two clauses.* (1) Every semantic item cited
by an episode entering SFT -- the claims the training set itself rests on, re-checked before
they are baked into an adapter. (2) Every retrieval-eligible (non-falsified) item that has
not been re-checked since the previous sleep -- claims that are still being SHOWN to the
proposer but have gone stale. The union is deduped by ``item_id``, so an item both clauses
name is measured once. Clause (1) is applied without a live filter, on purpose: an item
already marked falsified whose cited episode still enters SFT is re-measured anyway and its
outcome shows in the tally. Re-verification cannot resurrect it into retrieval (that filter
keys off ``falsified_by``, which is never cleared), so the only thing this buys is an honest
count -- and an honest count of what was actually re-run is what the tally is for.

*Staleness is measured against the previous SLEEP, accepted or not.*
``self._last_refalsified_at`` is set on every sleep, because the re-execution happened at
that moment regardless of what the gate decided afterwards. An item is stale when its
``last_verified_at`` is ``None`` or predates that watermark; before the first sleep the
watermark is ``None`` and every live item is stale, which is the honest starting position
(nothing has been re-checked yet).

Staleness compares timestamps LEXICOGRAPHICALLY, which is only valid because every ``now``
in this codebase is a fixed-width, ``Z``-suffixed UTC ISO-8601 string (``2026-08-24T12:00:00Z``);
mixed UTC offsets or variable-width fractional seconds would silently break the ordering.

*Enumerating the items to re-check goes through the episodes' families.* ``MemoryStore``
exposes no "every semantic item" query -- only ``semantic_for(unit_id, family)`` and
``semantic_family(family)`` -- so the family set is derived from the store's episodes and
each family is swept once. Every semantic item is minted from an episode of the same family
(``crucible.memory.distill``), so the only item this can miss is one whose episode row is
gone entirely, in which case its citation is broken and refalsification could report nothing
but ``infra_broken_citation`` anyway.

*The gate is a regression check, not an improvement check.* ``accepted = before - after <=
ACCEPT_MAX_DROP``: the candidate has to avoid LOSING more than one already-solved task on the
slice; it does not have to win any. A drop of exactly one accepts (the boundary is inclusive,
matching §4.7's inclusive gates); a drop of two rejects. ``before`` is measured against
whatever adapter is currently live (``registry.latest_accepted()``, ``None`` when the base
model is serving), ``after`` against the candidate -- so the comparison is always
"what is serving now" vs "what would serve next", never base-vs-candidate once an adapter
has been accepted.

*The slice is a function of the solved-task SET, the seed, and the sleep index -- never of
the caller's list order.* ``sorted(set(solved_task_keys))`` before sampling, seeded
``random.Random(f"{seed}:slice:{sleep_index}")`` (a string seed, hashed by ``Random`` itself,
so it does not depend on ``PYTHONHASHSEED``). Two runs with the same seed re-measure the same
tasks; two sleeps in one run do not, so a rejected candidate's successor is not graded on the
identical sample. Same discipline as ``select.py``'s explicit sort: a caller's accumulation
order must not leak into a measurement.

*A server that refuses the adapter fails the sleep, loudly.* ``server.load`` raising
propagates straight out of ``maybe_sleep``: no registry row, no counter reset, no record. The
alternative -- catching it and writing ``accepted=False`` -- would file an ops failure under
"the candidate lost the regression gate", which is a different claim and a false one. The cost
is that a wedged server stops the run instead of quietly degrading it; for a spike whose entire
output is a measurement, that is the right way round.

*``gpu_s`` is written as ``None`` here, always.* Training happens behind the ``Trainer``
seam -- possibly on another box -- and this controller reads no clock at all (``now`` is
caller-supplied, like every timestamp in this codebase). The honest options were "measure it
somewhere that can" or "invent a number here"; the field is carried in the record so the ops
path (the smoke) can fill it in from the side that owns the GPU, and ``None`` means NOT
MEASURED, never zero seconds (the None-vs-zero discipline used throughout this codebase).

*``BASE_DIGEST`` is a digest of the base model's IDENTITY, not of its weights.* The registry
column answers "which frozen base was this adapter attached to". This module never sees the
weights, so it commits to the one stable thing it does know: ``sha256_text(BASE_MODEL)``,
which changes if and only if the pinned base in ``train.py`` changes. Stated here so nobody
later reads that column as a checkpoint hash.

*Never metered.* Consolidation's re-executions and regression slice are a maintenance concern
of the memory organ, not search-time cost charged to any task's execution cap -- the same rule
``crucible.memory.falsify`` follows, pinned the same crude way (a source scan for the meter's
module name, which must not appear anywhere in this file).
"""
from __future__ import annotations

import json
import random
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..memory.falsify import FalsifyTally, falsify_batch
from ..memory.schema import SemanticItem
from ..memory.store import MemoryStore
from ..stream.units import Unit, sha256_text
from .registry import AdapterRegistry, adapter_id_for
from .select import episode_set_hash, sft_pairs
from .train import BASE_MODEL, Trainer

# Verified episodes that must accumulate since the last ACCEPTED sleep before the next one
# fires (spec §5). Named so the CLI's --sleep-threshold has one default to point at.
SLEEP_THRESHOLD_DEFAULT = 16
# How many already-solved tasks the regression gate re-runs. min(SLICE_SIZE, |solved|).
SLICE_SIZE = 12
# The gate's tolerance: losing this many previously-solved slice tasks still accepts.
# Inclusive -- a drop of exactly ACCEPT_MAX_DROP is inside the gate.
ACCEPT_MAX_DROP = 1
# See the module docstring: the base model's identity digest, NOT a hash of its weights.
BASE_DIGEST = sha256_text(BASE_MODEL)

# A server that is wedged mid-load should fail the sleep, not hang the run forever.
_LOAD_TIMEOUT_S = 60.0


@runtime_checkable
class ServerAdapter(Protocol):
    """Hot-swap seam: make ``adapter_id`` servable from ``adapter_dir``."""

    def load(self, adapter_dir: Path, adapter_id: str) -> None: ...


class VllmAdapterLoader:
    """The live seam: ``POST {base_url}/v1/load_lora_adapter`` (vLLM's runtime-LoRA endpoint).

    Mirrors ``scripts/lora_attach_smoke.py``'s already-proven call (S1 findings §7: HTTP 200,
    the adapter then listed by ``/v1/models``) and uses stdlib ``urllib`` like every other
    HTTP caller in this codebase (``crucible.proposer.client``/``identity``) -- no ``requests``
    dependency. The server needs ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`` or the endpoint
    404s; that is an ops precondition, checked live rather than here.

    ``lora_path`` is sent exactly as given -- the server resolves it against ITS OWN working
    directory, so the path must be absolute. ``SleepController`` resolves its adapter root at
    construction for precisely this reason. No unit test beyond the payload shape and the
    protocol conformance check: its real verification is the live hot-swap smoke.
    """

    def __init__(self, base_url: str, *, timeout_s: float = _LOAD_TIMEOUT_S) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    def load(self, adapter_dir: Path, adapter_id: str) -> None:
        body = json.dumps({"lora_name": adapter_id, "lora_path": str(adapter_dir)}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/v1/load_lora_adapter", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        # urlopen raises HTTPError on any non-2xx, so this guard only catches an odd 2xx
        # (a 202/204 that means "accepted, maybe later") -- which must not read as loaded.
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            if resp.status != 200:
                detail = resp.read().decode("utf-8", "replace")[:200]
                raise RuntimeError(f"load_lora_adapter returned HTTP {resp.status} for "
                                   f"{adapter_id}: {detail}")


class FakeServerAdapter:
    """Unit-test double: records ``(adapter_dir, adapter_id)`` per call, loads nothing.

    The reject path is pinned by this list staying EMPTY, so recording every call -- rather
    than a bare "was it called" flag -- is what makes "exactly once, with these arguments"
    assertable.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def load(self, adapter_dir: Path, adapter_id: str) -> None:
        self.calls.append((Path(adapter_dir), adapter_id))


@runtime_checkable
class SliceRunner(Protocol):
    """Regression seam: how many of ``task_keys`` are solved under ``adapter_id``.

    ``adapter_id=None`` means the base model with no adapter loaded. The real implementation
    re-runs the tasks at K=1 greedy through the existing driver pieces (wired in a later task);
    it is not defined here because this module must stay importable without a server.
    """

    def solved(self, task_keys: list[str], adapter_id: str | None) -> int: ...


class FakeSliceRunner:
    """Unit-test double: returns scripted counts in call order and records every call.

    Running out of script raises rather than returning 0 -- a silent 0 would look like "the
    adapter solved nothing", which is a measurement, and an exhausted fake has not measured
    anything (the instrument-honesty rule from ``crucible/sandbox/report.py``, applied to a
    test double).
    """

    def __init__(self, counts: Sequence[int]) -> None:
        self._counts = list(counts)
        self.calls: list[tuple[tuple[str, ...], str | None]] = []

    def solved(self, task_keys: list[str], adapter_id: str | None) -> int:
        self.calls.append((tuple(task_keys), adapter_id))
        if not self._counts:
            raise AssertionError("FakeSliceRunner ran out of scripted counts")
        return self._counts.pop(0)


@dataclass(frozen=True)
class SleepRecord:
    """One sleep cycle, win or lose -- the honest record of what happened.

    Field order is frozen (later tasks read and construct this). ``refalsify`` is the
    ``FalsifyTally`` as a plain dict so the record stays JSON-native end to end;
    ``slice_task_keys`` records WHICH tasks the gate measured, not just how many, so a
    surprising accept/reject can be re-run by hand from the record alone. ``gpu_s`` is
    ``None`` from this controller -- see the module docstring.
    """

    sleep_index: int
    adapter_id: str
    episode_set_hash: str
    episodes_selected: int
    slice_task_keys: tuple[str, ...]
    slice_before: int
    slice_after: int
    accepted: bool
    refalsify: dict[str, int]
    gpu_s: float | None
    created_at: str

    def to_dict(self) -> dict:
        """JSON-ready form: ``slice_task_keys`` becomes a list so a file round-trip is exact."""
        d = asdict(self)
        d["slice_task_keys"] = list(self.slice_task_keys)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SleepRecord":
        """Inverse of :meth:`to_dict`; restores the tuple shape so equality holds."""
        d = dict(d)
        d["slice_task_keys"] = tuple(d["slice_task_keys"])
        return cls(**d)


class SleepController:
    """Trigger -> refalsify -> train -> regression gate -> accept/reject -> record.

    ``unit_loader`` resolves a ``unit_id`` to its ``Unit`` for refalsification (construction
    time, so ``maybe_sleep`` stays a two-argument call); a ``unit_id`` it cannot resolve
    raises, loudly, rather than being skipped -- an item whose unit the stream no longer holds
    is a data-integrity bug, the same class of thing ``select.sft_pairs`` refuses to paper
    over. ``adapters_dir`` is where trained adapters land (one content-addressed
    subdirectory per ``adapter_id``); it is resolved to an absolute path at construction
    because the server that later loads it does not share this process's working directory.
    """

    def __init__(self, store: MemoryStore, trainer: Trainer, server: ServerAdapter,
                 slice_runner: SliceRunner, registry: AdapterRegistry, *,
                 unit_loader: Callable[[str], Unit], adapters_dir: Path,
                 threshold: int = SLEEP_THRESHOLD_DEFAULT, seed: int) -> None:
        self._store = store
        self._trainer = trainer
        self._server = server
        self._slice_runner = slice_runner
        self._registry = registry
        self._unit_loader = unit_loader
        self._adapters_dir = Path(adapters_dir).resolve()
        self._threshold = threshold
        self._seed = seed
        self._last_accepted_count = 0
        self._last_refalsified_at: str | None = None
        # Seeded from disk, not from zero -- see the module docstring's resume note.
        self._sleep_index = registry.count()

    @property
    def threshold(self) -> int:
        """The verified-episode trigger this controller was configured with (read-only).

        Exposed so the caller that WIRED the controller (``crucible.run.full``) can report
        what a run is configured with without keeping a second copy of the number that could
        drift from the one the trigger actually uses.
        """
        return self._threshold

    def maybe_sleep(self, solved_task_keys: list[str], now: str) -> SleepRecord | None:
        """Sleep if enough verified episodes have accumulated; return the record, else ``None``.

        The pipeline order is load-bearing: claims are re-checked BEFORE the episodes they
        belong to are selected for training, so a lesson that stopped being true is already
        marked falsified when the adapter is built.
        """
        if self._store.count_verified_since(self._last_accepted_count) < self._threshold:
            return None

        sleep_index = self._sleep_index
        tally = self._refalsify(now)

        pairs = sft_pairs(self._store)
        set_hash = episode_set_hash(pairs)
        adapter_id = adapter_id_for(set_hash)
        adapter_dir = self._trainer.train(pairs, self._seed, self._adapters_dir / adapter_id)

        slice_keys = self._slice(solved_task_keys, sleep_index)
        before = self._slice_runner.solved(list(slice_keys), self._registry.latest_accepted())
        after = self._slice_runner.solved(list(slice_keys), adapter_id)
        accepted = (before - after) <= ACCEPT_MAX_DROP

        if accepted:
            # Resolved at the call site so the absolute-path invariant VllmAdapterLoader
            # documents is true by construction, whatever the Trainer seam handed back.
            self._server.load(Path(adapter_dir).resolve(), adapter_id)
            self._registry.record(adapter_id, set_hash, BASE_DIGEST, True, now)
            # A SNAPSHOT of the total, never an advance-by-threshold: a sleep consumes every
            # verified episode that existed when it fired, not `threshold` of them (review
            # finding 2). Advancing by threshold would leave a phantom surplus behind and
            # fire the next sleep early, on episodes already consolidated.
            self._last_accepted_count = self._store.verified_count()
        else:
            # No server call and no counter reset: a rejected adapter is never SELECTED as
            # the serving adapter (a real SliceRunner must load it to measure `after` at
            # all), and the episodes it failed to consolidate stay unconsolidated.
            self._registry.record(adapter_id, set_hash, BASE_DIGEST, False, now)

        self._sleep_index += 1
        self._last_refalsified_at = now
        return SleepRecord(
            sleep_index=sleep_index, adapter_id=adapter_id, episode_set_hash=set_hash,
            episodes_selected=len(pairs), slice_task_keys=slice_keys,
            slice_before=before, slice_after=after, accepted=accepted,
            refalsify=asdict(tally), gpu_s=None, created_at=now,
        )

    def _refalsify(self, now: str) -> FalsifyTally:
        """Re-execute every targeted claim once and return the four-way tally."""
        targets = self._refalsify_targets()
        return falsify_batch(self._store, [(item, self._unit_loader(item.unit_id))
                                           for item in targets], now=now)

    def _refalsify_targets(self) -> list[SemanticItem]:
        """The deduped union of the two clauses, in ``item_id`` order -- see the module docstring."""
        sft_episode_ids = {ep.item_id for ep in self._store.episodes(verified_only=True)}
        families = sorted({ep.family for ep in self._store.episodes()})
        targets: dict[str, SemanticItem] = {}
        for family in families:
            for item in self._store.semantic_family(family):
                if item.item_id in targets:
                    continue
                if item.cited_episode_id in sft_episode_ids or self._is_stale(item):
                    targets[item.item_id] = item
        return [targets[item_id] for item_id in sorted(targets)]

    def _is_stale(self, item: SemanticItem) -> bool:
        """Retrieval-eligible and not re-checked since the previous sleep.

        The comparison below is lexicographic on ISO-8601 strings, valid only for the
        fixed-width, ``Z``-suffixed UTC form every timestamp in this codebase uses.
        """
        if item.falsified_by is not None:
            return False  # not retrieval-eligible: retrieval filters these out entirely
        if self._last_refalsified_at is None:
            return True  # first sleep of this controller: nothing has been re-checked yet
        return item.last_verified_at is None or item.last_verified_at < self._last_refalsified_at

    def _slice(self, solved_task_keys: list[str], sleep_index: int) -> tuple[str, ...]:
        """A seeded sample of the solved-task SET -- order-independent by construction."""
        keys = sorted(set(solved_task_keys))
        rng = random.Random(f"{self._seed}:slice:{sleep_index}")
        return tuple(rng.sample(keys, min(SLICE_SIZE, len(keys))))
